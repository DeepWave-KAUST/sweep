#include <torch/extension.h>
#include <algorithm>

#include "acoustic_vrz2d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "driver_traits.cuh"
#include "../../common/adapt_inputs.h"   // *InputCore, InputArena, to_torch

namespace acoustic_vrz2d {

namespace {

// The checkpoint replay state backward_ckpt steps, set 0 of
// p.forward_wavefields (Python-zeroed per backward call): the Driver::CKPT_NVAR
// snapshot slots u_prev, u_now, psix, psiz, zetax, zetaz plus u_next -- the
// forward slot list without the psi double-buffer shadows -- in the struct's
// bind order u_prev, u_now, u_next, psix, psiz, zetax, zetaz.
constexpr int REPLAY_STATE_NVAR = Driver::CKPT_NVAR + 1;   // 7

// p.checkpoint_replay (cuda_layout.checkpoint_replay_shapes): allocated by the
// propagator next to the checkpoint snapshots, never re-zeroed.
enum ReplaySlot : int {
    CHUNK_FORWARD = 0   // the replayed segment's pressure, (max_segment, B, 1, nz, nx)
};

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::generic_backward<Driver>(in);
}

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_core<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::generic_backward_bs<Driver>(in);
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_bs_core<Driver>(in);
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(p.models.size() == 2, "AcousticVRZ backward_ckpt expects models [vp, z].");
    SWEEP_CHECK(!p.checkpoints.empty(), "AcousticVRZ backward_ckpt expects checkpoints.");
    SWEEP_CHECK(p.checkpoint_interval > 0, "AcousticVRZ backward_ckpt expects positive checkpoint_interval.");

    auto vp = p.models[0];
    auto z = p.models[1];
    auto inv_z = derived::reciprocal(p, z, "acoustic_vrz2d::backward_ckpt");

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;
    int M = p.M;
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};

    // _c.py binds cp.adjoint_wavefields on every backward
    // (_ensure_wavefield_buffers: base_nvar 3 + pml_nvar 6 = 9 slots), the same
    // list Driver::bind_or_alloc_adjoint takes on the full/bs path.
    AcousticWavefieldTensor adjoint;
    SWEEP_CHECK(!p.adjoint_wavefields.empty(),
                "acoustic_vrz2d/backward_ckpt requires the propagator-bound "
                "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar = 9 tensors)");
    adjoint.bind(p.adjoint_wavefields, 2, true);
    Driver::zero_wavefield_state(adjoint);

    // The replay state (REPLAY_STATE_NVAR above), bound through the struct's
    // full PML bind: with no psi shadows in the set the replay writes psi in
    // place (swap() rotates u only) and the propagator zeroed the set per call.
    // _c.py hands it over on every checkpoint-mode backward
    // (_forward_state_buffers over cp.forward_state_shapes, derived from
    // slot_table.ACOUSTIC_VRZ2D's forward slots without the psi shadows), so
    // there is no unbound caller to allocate for.
    AcousticWavefieldTensor forward;
    SWEEP_CHECK(static_cast<int>(p.forward_wavefields.size()) == REPLAY_STATE_NVAR,
                "acoustic_vrz2d/backward_ckpt requires the propagator-bound "
                "forward_wavefields (cuda_layout.slots, the forward slots without the "
                "psi double-buffer shadows): one replay state set of ",
                REPLAY_STATE_NVAR, " tensors, got ", p.forward_wavefields.size());
    forward.bind(wavefield_set(p.forward_wavefields, 0, REPLAY_STATE_NVAR,
                               "acoustic_vrz2d ckpt replay state"), 2, true);

    // {grad_vp, grad_z}: p.grads_out as the propagator binds it for the acoustic
    // family ({grad_wavelet, grad_vp, grad_z}; slot 0 unused, VRZ computes no
    // grad_wavelet), zeroed by Python once per backward -- the binding the
    // full/bs skeleton takes (Driver::bind_backward_outputs).  _c.py builds it
    // for every backward, so it is never empty.
    SWEEP_CHECK(p.grads_out.size() == p.models.size() + 1,
                "acoustic_vrz2d/backward_ckpt requires the propagator-bound grads_out "
                "(cuda_layout.grads_out_has_wavelet + one slot per model = "
                "models.size()+1 tensors, slot 0 = grad_wavelet, unused for VRZ), got ",
                p.grads_out.size());
    auto grad_vp = pool_required(p.grads_out, 1, vp, "grads_out");
    auto grad_z  = pool_required(p.grads_out, 2, z, "grads_out");
    // Model-shaped per-call scratch from the propagator's pool (Driver::
    // WorkspaceSlot, the same seven slots the full/bs skeleton takes), zero at
    // entry; Driver::workspace_slots requires the binding.
    const auto& ws = Driver::workspace_slots(p);
    auto C0  = pool_required(ws, Driver::COEF_C0, vp, "adjoint_workspace");   // vp²       (time-invariant adjoint coeffs)
    auto Cx  = pool_required(ws, Driver::COEF_CX, vp, "adjoint_workspace");   // ∂ₓb·κ
    auto Cz  = pool_required(ws, Driver::COEF_CZ, vp, "adjoint_workspace");   // ∂_z b·κ
    auto c_x = pool_required(ws, Driver::C_X, vp, "adjoint_workspace");       // split gradient scratch (order>=6 path)
    auto c_z = pool_required(ws, Driver::C_Z, vp, "adjoint_workspace");
    auto e_x = pool_required(ws, Driver::E_X, vp, "adjoint_workspace");
    auto e_z = pool_required(ws, Driver::E_Z, vp, "adjoint_workspace");
    const Buf& checkpoint_steps_cpu = p.checkpoint_steps;   // a host copy, made by the adapter (undefined -> numel 0)
    const bool recursive_checkpoint = checkpoint_steps_cpu.numel() > 0;
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        Driver::CKPT_NVAR,
        true,
        recursive_checkpoint,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        recursive_checkpoint ? "backward_recursive" : "backward_chunk",
        "acoustic_vrz2d"
    );

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    // Time-invariant adjoint transpose coefficients, computed once for all segments.
    BUILD_VRZ_ADJOINT_COEFFS(
        order,
        launch_config.grid,
        launch_config.block,
        vp.data_ptr<float>(),
        z.data_ptr<float>(),
        inv_z.data_ptr<float>(),
        C0.data_ptr<float>(),
        Cx.data_ptr<float>(),
        Cz.data_ptr<float>(),
        grad_ctx,
        ctx
    );

    int chunk_size = p.checkpoint_interval;
    int nt = static_cast<int>(p.nt);
    int num_chunks = (nt + chunk_size - 1) / chunk_size;
    int num_segments = num_chunks;
    int max_segment_length = chunk_size;
    int num_saved_checkpoints = 0;
    const int* checkpoint_steps = nullptr;

    if (recursive_checkpoint) {
        num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
        num_segments = num_saved_checkpoints + 1;
        checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();
        SWEEP_CHECK(
            static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
            "AcousticVRZ checkpoint buffer is smaller than required chunk count."
        );
        max_segment_length = 0;
        for (int segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
            int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
            int end = (segment_idx == num_saved_checkpoints) ? nt : checkpoint_steps[segment_idx];
            max_segment_length = std::max(max_segment_length, end - start);
        }
    } else {
        SWEEP_CHECK(
            static_cast<int>(p.checkpoints[0].size(0)) >= num_chunks,
            "AcousticVRZ checkpoint buffer is smaller than required chunk count."
        );
    }

    // The replayed segment's pressure history from p.checkpoint_replay
    // (ReplaySlot above), taken once at the full max_segment rows and reused
    // by every segment: each row the reverse pass reads was written by the
    // replay earlier in the same segment, so it is never zeroed.  The
    // propagator allocates it next to the checkpoint snapshots for both
    // checkpoint modes (cuda_layout.checkpoint_replay_shapes is unconditional
    // for this equation), so the binding is required.
    auto chunk_forward = pool_required(p.checkpoint_replay, CHUNK_FORWARD,
                                       {max_segment_length, N, C, nz, nx},
                                       "checkpoint_replay (acoustic_vrz2d/backward_ckpt, "
                                       "cuda_layout.checkpoint_replay_shapes)");

    for (int segment_id = num_segments - 1; segment_id >= 0; --segment_id) {
        int start;
        int end;
        int checkpoint_idx;
        if (recursive_checkpoint) {
            start = (segment_id == 0) ? 0 : checkpoint_steps[segment_id - 1];
            end = (segment_id == num_saved_checkpoints) ? nt : checkpoint_steps[segment_id];
            checkpoint_idx = segment_id - 1;
        } else {
            start = segment_id * chunk_size;
            end = std::min(nt, start + chunk_size);
            checkpoint_idx = segment_id;
        }

        if (checkpoint_idx < 0)
            checkpoint_runtime.zero_state(forward.state_tensors());
        else
            checkpoint_runtime.load(checkpoint_idx, forward.checkpoint_tensors(), forward.next_tensors());

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            ACOUSTIC_VRZ2D(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
                false,
                nullptr,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                for_view.u_next,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                ctx
            );

            forward.swap();
            copy_tensor_device_to_device_async(chunk_forward.select(0, it - start), forward.u_now_t);
        }

        for (int it = end - 1; it >= start; --it) {
            auto adj_view = adjoint.view();

            ACOUSTIC_VRZ2D_ADJOINT_FUSED(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                C0.data_ptr<float>(),
                Cx.data_ptr<float>(),
                Cz.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source_signed<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                -1.0f,   // NEGATED residual: sign bit flipped in-kernel, no negated copy
                ctx
            );

            adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

            CALCULATE_GRAD_VRZ2D_AUTO(
                order,
                launch_config.grid,
                launch_config.block,
                chunk_forward.select(0, it - start).data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                c_x.data_ptr<float>(),
                c_z.data_ptr<float>(),
                e_x.data_ptr<float>(),
                e_z.data_ptr<float>(),
                grad_vp.data_ptr<float>(),
                grad_z.data_ptr<float>(),
                grad_ctx,
                lap_ctx,
                ctx
            );
        }
    }

    out.grads = {grad_vp, grad_z};
    return out;
}

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    return backward_ckpt_core(in);
}

BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in)
{
    return std::make_shared<eqdrv::GenericBackwardBsRunner<Driver>>(in);
}

BackwardRunnerPtr backward_bs_runner(const BackwardInput& in)
{
    return std::make_shared<TorchBackwardRunner<eqdrv::GenericBackwardBsRunner<Driver>>>(in);
}


BackwardOutput backward_ckpt(const BackwardInput& in_torch)
{
    InputArena arena;
    const BackwardInputCore in = adapt_input(in_torch, arena);
    return to_torch(backward_ckpt_core(in), in_torch);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in_torch)
{
    InputArena arena;
    const BackwardInputCore in = adapt_input(in_torch, arena);
    return to_torch(backward_recursive_ckpt_core(in), in_torch);
}

} // namespace acoustic_vrz2d
