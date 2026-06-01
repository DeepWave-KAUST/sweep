#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "acoustic_vrz3d.h"
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

namespace acoustic_vrz3d {

namespace {

void zero_wavefield_state_vrz3d(AcousticWavefieldTensor& wf)
{
    wf.u_prev_t.zero_();
    wf.u_now_t.zero_();
    wf.u_next_t.zero_();
    wf.psix_t.zero_();
    wf.psiy_t.zero_();
    wf.psiz_t.zero_();
    wf.zetax_t.zero_();
    wf.zetay_t.zero_();
    wf.zetaz_t.zero_();
}

BackwardOutput backward_full_impl(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    TORCH_CHECK(
        in.u_forward.defined() && in.u_forward.numel() > 0,
        "AcousticVRZ3D backward expects saved full forward wavefields."
    );
    TORCH_CHECK(in.models.size() == 2, "AcousticVRZ3D backward expects models [vp, z].");
    TORCH_CHECK(
        in.u_forward.dim() == 6 && in.u_forward.size(1) == 7,
        "AcousticVRZ3D backward expects forward wavefields with shape (nt, 7, B, nz, ny, nx)."
    );

    BackwardOutput out;

    auto vp = in.models[0];
    auto z = in.models[1];
    auto inv_z = torch::reciprocal(z);
    auto neg_adjoint_source = -in.adjoint_source;

    float dx = in.spacing[0];
    float dy = in.spacing[1];
    float dz = in.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;
    int adjoint_nsrc = in.adjoint_sources_loc.size(1);
    const int order = (in.M <= 4) ? static_cast<int>(2 * in.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, in.dt, in.nt, in.M, in.abcn, in.free_surface,
                      in.lap_coes.data_ptr<float>(), in.grad_coes.data_ptr<float>(),
                      dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    if (!in.adjoint_wavefields.empty())
        adjoint.bind(in.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);
    zero_wavefield_state_vrz3d(adjoint);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto aq0 = torch::zeros_like(vp);   // adjoint transpose buffers (b·κ·λ, (∂b·κ)λ)
    auto aqx = torch::zeros_like(vp);
    auto aqy = torch::zeros_like(vp);
    auto aqz = torch::zeros_like(vp);
    auto c_x = torch::zeros_like(vp);   // gradient assembly buffers (λ·vp·∂p, λ·vp²z·∂p)
    auto c_y = torch::zeros_like(vp);
    auto c_z = torch::zeros_like(vp);
    auto e_x = torch::zeros_like(vp);
    auto e_y = torch::zeros_like(vp);
    auto e_z = torch::zeros_like(vp);
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(in.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, in.M, in.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx * ny, in.M, in.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, in.M, in.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, in.M, in.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, in.M, in.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = in.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();
        BUILD_VRZ_ADJOINT_FIELDS_3D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view.u_now,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            aq0.data_ptr<float>(),
            aqx.data_ptr<float>(),
            aqy.data_ptr<float>(),
            aqz.data_ptr<float>(),
            grad_ctx,
            ctx
        );

        ACOUSTIC_VRZ3D_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            aq0.data_ptr<float>(),
            aqx.data_ptr<float>(),
            aqy.data_ptr<float>(),
            aqz.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            neg_adjoint_source.data_ptr<float>(),
            in.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        BUILD_VRZ_GRAD_FIELDS_3D(
            order,
            launch_config.grid,
            launch_config.block,
            in.u_forward.select(0, it).select(0, 0).data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            c_x.data_ptr<float>(),
            c_y.data_ptr<float>(),
            c_z.data_ptr<float>(),
            e_x.data_ptr<float>(),
            e_y.data_ptr<float>(),
            e_z.data_ptr<float>(),
            grad_ctx,
            ctx
        );

        CALCULATE_GRAD_VRZ3D(
            order,
            launch_config.grid,
            launch_config.block,
            in.u_forward.select(0, it).select(0, 0).data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            c_x.data_ptr<float>(),
            c_y.data_ptr<float>(),
            c_z.data_ptr<float>(),
            e_x.data_ptr<float>(),
            e_y.data_ptr<float>(),
            e_z.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_z.data_ptr<float>(),
            grad_ctx,
            lap_ctx,
            ctx
        );
    }

    const auto normalize_grad = [](const torch::Tensor& model_grad, const torch::Tensor& model) {
        if (!model_grad.defined()) return model_grad;
        if (
            model_grad.dim() == static_cast<int>(model.dim()) - 1 &&
            model_grad.size(0) == model.size(0) &&
            model.dim() >= 2 &&
            model.size(1) == 1
        ) {
            return model_grad.unsqueeze(1);
        }

        return model_grad;
    };

    grad_vp = normalize_grad(grad_vp, vp);
    grad_z = normalize_grad(grad_z, z);

    out.grads = {grad_vp, grad_z};
    return out;
}

BackwardOutput backward_bs_impl(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ3D backward_bs expects models [vp, z].");
    TORCH_CHECK(p.u_last_two.defined() && p.u_last_two.numel() > 0,
                "AcousticVRZ3D backward_bs expects last two wavefields.");

    auto vp = p.models[0];
    auto z = p.models[1];
    auto inv_z = torch::reciprocal(z);
    auto neg_adjoint_source = -p.adjoint_source;

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);
    zero_wavefield_state_vrz3d(adjoint);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 3, true);
    else
        forward.allocate(vp, 3, true);

    forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
    forward.u_next_t.zero_();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto aq0 = torch::zeros_like(vp);   // adjoint transpose buffers (b·κ·λ, (∂b·κ)λ)
    auto aqx = torch::zeros_like(vp);
    auto aqy = torch::zeros_like(vp);
    auto aqz = torch::zeros_like(vp);
    auto c_x = torch::zeros_like(vp);   // gradient assembly buffers (λ·vp·∂p, λ·vp²z·∂p)
    auto c_y = torch::zeros_like(vp);
    auto c_z = torch::zeros_like(vp);
    auto e_x = torch::zeros_like(vp);
    auto e_y = torch::zeros_like(vp);
    auto e_z = torch::zeros_like(vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int save_width = p.M + 1;
    int boundary_offset = -p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
            {}, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, {}, p.use_pinned_memory
        );
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        3,
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

    for (int it = p.nt - 1; it >= 1; --it) {
        auto adj_view = adjoint.view();

        BUILD_VRZ_ADJOINT_FIELDS_3D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view.u_now,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            aq0.data_ptr<float>(),
            aqx.data_ptr<float>(),
            aqy.data_ptr<float>(),
            aqz.data_ptr<float>(),
            grad_ctx,
            ctx
        );

        ACOUSTIC_VRZ3D_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            aq0.data_ptr<float>(),
            aqx.data_ptr<float>(),
            aqy.data_ptr<float>(),
            aqz.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            neg_adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        auto for_view = forward.view();

        ACOUSTIC_VRZ3D_NOPML(
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

        add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        boundary_runtime.restore_backward_3d(
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

        BUILD_VRZ_GRAD_FIELDS_3D(
            order,
            launch_config.grid,
            launch_config.block,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            c_x.data_ptr<float>(),
            c_y.data_ptr<float>(),
            c_z.data_ptr<float>(),
            e_x.data_ptr<float>(),
            e_y.data_ptr<float>(),
            e_z.data_ptr<float>(),
            grad_ctx,
            ctx
        );

        CALCULATE_GRAD_VRZ3D(
            order,
            launch_config.grid,
            launch_config.block,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            c_x.data_ptr<float>(),
            c_y.data_ptr<float>(),
            c_z.data_ptr<float>(),
            e_x.data_ptr<float>(),
            e_y.data_ptr<float>(),
            e_z.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_z.data_ptr<float>(),
            grad_ctx,
            lap_ctx,
            ctx
        );

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    const auto normalize_grad = [](const torch::Tensor& model_grad, const torch::Tensor& model) {
        if (!model_grad.defined()) return model_grad;
        if (
            model_grad.dim() == static_cast<int>(model.dim()) - 1 &&
            model_grad.size(0) == model.size(0) &&
            model.dim() >= 2 &&
            model.size(1) == 1
        ) {
            return model_grad.unsqueeze(1);
        }

        return model_grad;
    };

    grad_vp = normalize_grad(grad_vp, vp);
    grad_z = normalize_grad(grad_z, z);
    out.grads = {grad_vp, grad_z};
    return out;
}

BackwardOutput backward_ckpt_impl(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ3D backward_ckpt expects models [vp, z].");
    TORCH_CHECK(!p.checkpoints.empty(), "AcousticVRZ3D backward_ckpt expects checkpoints.");
    TORCH_CHECK(p.checkpoint_interval > 0, "AcousticVRZ3D backward_ckpt expects positive checkpoint_interval.");

    auto vp = p.models[0];
    auto z = p.models[1];
    auto inv_z = torch::reciprocal(z);
    auto neg_adjoint_source = -p.adjoint_source;

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);
    zero_wavefield_state_vrz3d(adjoint);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 3, true);
    else
        forward.allocate(vp, 3, true);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto aq0 = torch::zeros_like(vp);   // adjoint transpose buffers (b·κ·λ, (∂b·κ)λ)
    auto aqx = torch::zeros_like(vp);
    auto aqy = torch::zeros_like(vp);
    auto aqz = torch::zeros_like(vp);
    auto c_x = torch::zeros_like(vp);   // gradient assembly buffers (λ·vp·∂p, λ·vp²z·∂p)
    auto c_y = torch::zeros_like(vp);
    auto c_z = torch::zeros_like(vp);
    auto e_x = torch::zeros_like(vp);
    auto e_y = torch::zeros_like(vp);
    auto e_z = torch::zeros_like(vp);
    auto checkpoint_steps_cpu = p.checkpoint_steps.defined()
        ? p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous()
        : torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32));
    const bool recursive_checkpoint = checkpoint_steps_cpu.numel() > 0;
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        8,
        true,
        recursive_checkpoint,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        recursive_checkpoint ? "backward_recursive" : "backward_chunk",
        "acoustic_vrz3d"
    );

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

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
            "AcousticVRZ3D checkpoint buffer is smaller than required chunk count."
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
            "AcousticVRZ3D checkpoint buffer is smaller than required chunk count."
        );
    }

    auto chunk_forward = torch::zeros({max_segment_length, N, C, nz, ny, nx}, vp.options());

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
            ACOUSTIC_VRZ3D(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_y,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
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

            BUILD_VRZ_ADJOINT_FIELDS_3D(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view.u_now,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                aq0.data_ptr<float>(),
                aqx.data_ptr<float>(),
                aqy.data_ptr<float>(),
                aqz.data_ptr<float>(),
                grad_ctx,
                ctx
            );

            ACOUSTIC_VRZ3D_ADJOINT(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                aq0.data_ptr<float>(),
                aqx.data_ptr<float>(),
                aqy.data_ptr<float>(),
                aqz.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_y,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                neg_adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap();

            BUILD_VRZ_GRAD_FIELDS_3D(
                order,
                launch_config.grid,
                launch_config.block,
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                c_x.data_ptr<float>(),
                c_y.data_ptr<float>(),
                c_z.data_ptr<float>(),
                e_x.data_ptr<float>(),
                e_y.data_ptr<float>(),
                e_z.data_ptr<float>(),
                grad_ctx,
                ctx
            );

            CALCULATE_GRAD_VRZ3D(
                order,
                launch_config.grid,
                launch_config.block,
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                c_x.data_ptr<float>(),
                c_y.data_ptr<float>(),
                c_z.data_ptr<float>(),
                e_x.data_ptr<float>(),
                e_y.data_ptr<float>(),
                e_z.data_ptr<float>(),
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                grad_vp.data_ptr<float>(),
                grad_z.data_ptr<float>(),
                grad_ctx,
                lap_ctx,
                ctx
            );
        }
    }

    const auto normalize_grad = [](const torch::Tensor& model_grad, const torch::Tensor& model) {
        if (!model_grad.defined()) return model_grad;
        if (
            model_grad.dim() == static_cast<int>(model.dim()) - 1 &&
            model_grad.size(0) == model.size(0) &&
            model.dim() >= 2 &&
            model.size(1) == 1
        ) {
            return model_grad.unsqueeze(1);
        }

        return model_grad;
    };

    grad_vp = normalize_grad(grad_vp, vp);
    grad_z = normalize_grad(grad_z, z);
    out.grads = {grad_vp, grad_z};
    return out;
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_full_impl(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_bs_impl(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_ckpt_impl(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_ckpt_impl(in);
}


} // namespace acoustic_vrz3d
