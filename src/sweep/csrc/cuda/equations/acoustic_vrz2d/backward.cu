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
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace acoustic_vrz2d {

namespace {

void zero_wavefield_state_vrz(AcousticWavefieldTensor& wf)
{
    wf.u_prev_t.zero_();
    wf.u_now_t.zero_();
    wf.u_next_t.zero_();
    wf.psix_t.zero_();
    wf.psiz_t.zero_();
    wf.zetax_t.zero_();
    wf.zetaz_t.zero_();
}


} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    TORCH_CHECK(
        in.u_forward.defined() && in.u_forward.numel() > 0,
        "AcousticVRZ backward expects saved full forward wavefields."
    );
    TORCH_CHECK(in.models.size() == 2, "AcousticVRZ backward expects models [vp, z].");
    TORCH_CHECK(
        in.u_forward.dim() == 6 && in.u_forward.size(1) == 5,
        "AcousticVRZ backward expects forward wavefields with shape (nt, 5, B, 1, nz, nx)."
    );

    BackwardOutput out;

    auto vp = in.models[0];
    auto z = in.models[1];
    auto inv_z = torch::reciprocal(z);
    auto neg_adjoint_source = -in.adjoint_source;

    float dx = in.spacing[0];
    float dz = in.spacing[1];
    float dt = in.dt;

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;
    int M = in.M;
    int adjoint_nsrc = in.adjoint_sources_loc.size(1);
    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, in.nt, in.M, in.abcn, in.free_surface,
                      in.lap_coes.data_ptr<float>(), in.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!in.adjoint_wavefields.empty())
        adjoint.bind(in.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    zero_wavefield_state_vrz(adjoint);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto C0 = torch::zeros_like(vp);    // vp²       (time-invariant adjoint coeffs)
    auto Cx = torch::zeros_like(vp);    // (∂ₓb·κ)
    auto Cz = torch::zeros_like(vp);    // (∂_z b·κ)
    auto c_x = torch::zeros_like(vp);   // split gradient scratch (order>=6 path)
    auto c_z = torch::zeros_like(vp);
    auto e_x = torch::zeros_like(vp);
    auto e_z = torch::zeros_like(vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(in.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, in.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, in.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, in.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, in.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    // Time-invariant adjoint transpose coefficients (vp², ∂ₓb·κ, ∂_z b·κ),
    // computed once so the fused adjoint kernel only multiplies by λ per step.
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

    for (int it = in.nt - 1; it >= 0; --it) {
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
            in.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        CALCULATE_GRAD_VRZ2D_AUTO(
            order,
            launch_config.grid,
            launch_config.block,
            in.u_forward.select(0, it).select(0, 0).data_ptr<float>(),
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

    out.grads = {grad_vp, grad_z};
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ backward_bs expects models [vp, z].");
    TORCH_CHECK(p.u_last_two.defined() && p.u_last_two.numel() > 0,
                "AcousticVRZ backward_bs expects saved last_two wavefields.");

    auto vp = p.models[0];
    auto z = p.models[1];
    auto inv_z = torch::reciprocal(z);

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
    zero_wavefield_state_vrz(adjoint);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, false);
    else
        forward.allocate(vp, 2, false);
    forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
    forward.u_next_t.zero_();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto C0 = torch::zeros_like(vp);    // vp²       (time-invariant adjoint coeffs)
    auto Cx = torch::zeros_like(vp);    // (∂ₓb·κ)
    auto Cz = torch::zeros_like(vp);    // (∂_z b·κ)
    auto c_x = torch::zeros_like(vp);   // split gradient scratch (order>=6 path)
    auto c_z = torch::zeros_like(vp);
    auto e_x = torch::zeros_like(vp);
    auto e_z = torch::zeros_like(vp);
    auto neg_adjoint_source = -p.adjoint_source;

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    int save_width = p.M + 1;
    int boundary_offset = -p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    // tangent_pad = M to match Python boundary_tangent_pad (= so//2 for VRZ);
    // keeps the FP32 staging in step with the persistent int8 buffers.
    const int boundary_tangent_pad = p.M;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory,
                                boundary_tangent_pad);
    } else {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, {}, p.use_pinned_memory,
                                boundary_tangent_pad);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        2,
        true,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

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

    // Zero the boundary/PML band of the initial reconstruction state so stale
    // PML values carried in u_last_two don't leak inward during reverse
    // propagation.  acoustic2d backward_bs does this; its absence was the main
    // cause of the VRZ bs accuracy gap (acoustic bs is bit-exact, VRZ was not).
    {
        auto for_init = forward.view();
        set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(
            for_init.u_prev, ctx.abcn + ctx.M, nx, nz, ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));
        set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(
            for_init.u_now, ctx.abcn + ctx.M, nx, nz, ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));
    }

    for (int it = p.nt - 1; it >= 1; --it) {
        auto adj_view = adjoint.view();
        auto for_view = forward.view();

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

        ACOUSTIC_VRZ2D_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
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

        boundary_runtime.restore_backward_2d(
            it,
            for_view.u_next,
            launch_config.grid,
            launch_config.block,
            bs,
            save_width,
            boundary_offset,
            ctx
        );

        forward.swap();

        CALCULATE_GRAD_VRZ2D_AUTO(
            order,
            launch_config.grid,
            launch_config.block,
            forward.u_now_t.data_ptr<float>(),
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

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    out.grads = {grad_vp, grad_z};
    return out;
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
    auto inv_z = torch::reciprocal(z);
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
    zero_wavefield_state_vrz(adjoint);

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

} // namespace acoustic_vrz2d
