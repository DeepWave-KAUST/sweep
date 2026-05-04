#include <torch/extension.h>
#include <algorithm>

#include "acoustic_vrz2d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
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

BackwardOutput backward_not_implemented(const char* fn_name)
{
    TORCH_CHECK(
        false,
        "AcousticVRZ CUDA ",
        fn_name,
        " is not implemented yet for boundary-saving/checkpoint modes. "
        "Use the full-wavefield backward path first."
    );
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
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
        adjoint.allocate(vp, 2, true);
    zero_wavefield_state_vrz(adjoint);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto kappa_lambda = torch::zeros_like(vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(in.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, in.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, in.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, in.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, in.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = in.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();
        ACOUSTIC_VRZ2D_ADJOINT(
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

        adjoint.swap();

        build_kappa_lambda_vrz2d<<<launch_config.grid, launch_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            kappa_lambda.data_ptr<float>(),
            ctx
        );

        CALCULATE_GRAD_VRZ2D(
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

    out.grads = {grad_vp, grad_z};
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
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
        adjoint.allocate(vp, 2, true);
    zero_wavefield_state_vrz(adjoint);

    AcousticWavefieldTensor forward;
    forward.allocate(vp, 2, false);
    forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
    forward.u_next_t.zero_();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto kappa_lambda = torch::zeros_like(vp);
    auto neg_adjoint_source = -p.adjoint_source;

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    int save_width = p.M + 1;
    int boundary_offset = -p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, {}, p.use_pinned_memory);
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

    int interval = p.transfer_interval;
    int buf_idx = 0;

    AsyncCopyContext async_copy(staged_boundary);
    if (staged_boundary) {
        int it0 = p.nt - 1;
        int buf_idx0 = (it0 - 1) % interval;
        int chunk_start = it0 - buf_idx0 - 1;
        int chunk_len = buf_idx0 + 1;
        boundary_saver.load_cpu_to_gpu(chunk_start, chunk_len, async_copy.copy_stream);
        async_copy.record_copy_ready();
    }

    for (int it = p.nt - 1; it >= 1; --it) {
        auto adj_view = adjoint.view();
        auto for_view = forward.view();

        ACOUSTIC_VRZ2D_ADJOINT(
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

        adjoint.swap();

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

        buf_idx = (it - 1) % interval;
        if (staged_boundary && buf_idx == interval - 1)
            async_copy.wait_for_copy();

        float* top_ptr = staged_boundary ? boundary_saver.top_gpu.data_ptr<float>() + buf_idx * boundary_saver.top_stride
                                         : bs.top;
        float* bottom_ptr = staged_boundary ? boundary_saver.bottom_gpu.data_ptr<float>() + buf_idx * boundary_saver.bottom_stride
                                            : bs.bottom;
        float* left_ptr = staged_boundary ? boundary_saver.left_gpu.data_ptr<float>() + buf_idx * boundary_saver.left_stride
                                          : bs.left;
        float* right_ptr = staged_boundary ? boundary_saver.right_gpu.data_ptr<float>() + buf_idx * boundary_saver.right_stride
                                           : bs.right;

        boundary_kernel2d<<<launch_config.grid, launch_config.block>>>(
            for_view.u_next,
            top_ptr,
            bottom_ptr,
            left_ptr,
            right_ptr,
            staged_boundary ? 0 : it - 1,
            save_width,
            boundary_offset,
            ctx,
            BOUNDARY_RESTORE
        );

        forward.swap();

        if (staged_boundary && buf_idx == 0 && it > 1) {
            int next_chunk = (it - 1) / interval - 1;
            if (next_chunk >= 0) {
                int next_start = next_chunk * interval;
                int remain = static_cast<int>(p.nt) - next_start;
                int next_len = std::min(interval, remain);
                boundary_saver.load_cpu_to_gpu(next_start, next_len, async_copy.copy_stream);
                async_copy.record_copy_ready();
            }
        }

        build_kappa_lambda_vrz2d<<<launch_config.grid, launch_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            kappa_lambda.data_ptr<float>(),
            ctx
        );

        CALCULATE_GRAD_VRZ2D(
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

    out.grads = {grad_vp, grad_z};
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    (void)in;
    return backward_not_implemented("backward_ckpt");
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    (void)in;
    return backward_not_implemented("backward_recursive_ckpt");
}

} // namespace acoustic_vrz2d
