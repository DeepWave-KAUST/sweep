#include <algorithm>
#include <torch/extension.h>

#include "acoustic_vrz3d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
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
    auto kappa_lambda = torch::zeros_like(vp);

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
        ACOUSTIC_VRZ3D_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
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

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            neg_adjoint_source.data_ptr<float>(),
            in.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        build_kappa_lambda_vrz3d<<<launch_config.grid, launch_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            kappa_lambda.data_ptr<float>(),
            ctx
        );

        CALCULATE_GRAD_VRZ3D(
            order,
            launch_config.grid,
            launch_config.block,
            in.u_forward.select(0, it).select(0, 0).data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            kappa_lambda.data_ptr<float>(),
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
    auto kappa_lambda = torch::zeros_like(vp);

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

        ACOUSTIC_VRZ3D_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
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

        build_kappa_lambda_vrz3d<<<launch_config.grid, launch_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            kappa_lambda.data_ptr<float>(),
            ctx
        );

        CALCULATE_GRAD_VRZ3D(
            order,
            launch_config.grid,
            launch_config.block,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            kappa_lambda.data_ptr<float>(),
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

BackwardOutput backward_ckpt_impl(const BackwardInput& in)
{
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
    auto kappa_lambda = torch::zeros_like(vp);

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
    TORCH_CHECK(
        static_cast<int>(p.checkpoints[0].size(0)) >= num_chunks,
        "AcousticVRZ3D checkpoint buffer is smaller than required chunk count."
    );

    auto chunk_forward = torch::zeros({chunk_size, N, C, nz, ny, nx}, vp.options());

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(nt, start + chunk_size);

        forward.u_prev_t.copy_(p.checkpoints[0].select(0, chunk_id));
        forward.u_now_t.copy_(p.checkpoints[1].select(0, chunk_id));
        forward.psix_t.copy_(p.checkpoints[2].select(0, chunk_id));
        forward.psiy_t.copy_(p.checkpoints[3].select(0, chunk_id));
        forward.psiz_t.copy_(p.checkpoints[4].select(0, chunk_id));
        forward.zetax_t.copy_(p.checkpoints[5].select(0, chunk_id));
        forward.zetay_t.copy_(p.checkpoints[6].select(0, chunk_id));
        forward.zetaz_t.copy_(p.checkpoints[7].select(0, chunk_id));
        forward.u_next_t.zero_();

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
            chunk_forward[it - start].copy_(forward.u_now_t);
        }

        for (int it = end - 1; it >= start; --it) {
            auto adj_view = adjoint.view();

            ACOUSTIC_VRZ3D_ADJOINT(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
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

            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                neg_adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap();

            build_kappa_lambda_vrz3d<<<launch_config.grid, launch_config.block>>>(
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                kappa_lambda.data_ptr<float>(),
                ctx
            );

            CALCULATE_GRAD_VRZ3D(
                order,
                launch_config.grid,
                launch_config.block,
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                kappa_lambda.data_ptr<float>(),
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
    return backward_full_impl(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return backward_bs_impl(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return backward_ckpt_impl(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return backward_ckpt_impl(in);
}


} // namespace acoustic_vrz3d
