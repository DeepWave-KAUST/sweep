#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
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

namespace acoustic_vrz2d {

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::generic_backward<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::generic_backward_bs<Driver>(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ backward_ckpt expects models [vp, z].");
    TORCH_CHECK(!p.checkpoints.empty(), "AcousticVRZ backward_ckpt expects checkpoints.");
    TORCH_CHECK(p.checkpoint_interval > 0, "AcousticVRZ backward_ckpt expects positive checkpoint_interval.");

    auto vp = p.models[0];
    auto z = p.models[1];
    auto inv_z = derived::reciprocal(p, z, "acoustic_vrz2d::backward_ckpt");
    auto neg_adjoint_source = -p.adjoint_source;

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

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    Driver::zero_wavefield_state(adjoint);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, true);
    else
        forward.allocate(vp, 2, true);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto C0 = torch::zeros_like(vp);    // vp²       (time-invariant adjoint coeffs)
    auto Cx = torch::zeros_like(vp);    // (∂ₓb·κ)
    auto Cz = torch::zeros_like(vp);    // (∂_z b·κ)
    auto c_x = torch::zeros_like(vp);   // split gradient scratch (order>=6 path)
    auto c_z = torch::zeros_like(vp);
    auto e_x = torch::zeros_like(vp);
    auto e_z = torch::zeros_like(vp);
    auto checkpoint_steps_cpu = p.checkpoint_steps.defined()
        ? p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous()
        : torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32));
    const bool recursive_checkpoint = checkpoint_steps_cpu.numel() > 0;
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        recursive_checkpoint,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        recursive_checkpoint ? "backward_recursive" : "backward_chunk",
        "acoustic_vrz2d"
    );

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
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
        TORCH_CHECK(
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
        TORCH_CHECK(
            static_cast<int>(p.checkpoints[0].size(0)) >= num_chunks,
            "AcousticVRZ checkpoint buffer is smaller than required chunk count."
        );
    }

    auto chunk_forward = torch::zeros({max_segment_length, N, C, nz, nx}, vp.options());

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
            copy_tensor_device_to_device_async(chunk_forward[it - start], forward.u_now_t);
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

            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                neg_adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

            CALCULATE_GRAD_VRZ2D_AUTO(
                order,
                launch_config.grid,
                launch_config.block,
                chunk_forward[it - start].data_ptr<float>(),
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

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_ckpt(in);
}

BackwardRunnerPtr backward_bs_runner(const BackwardInput& in)
{
    return std::make_shared<eqdrv::GenericBackwardBsRunner<Driver>>(in);
}

} // namespace acoustic_vrz2d
