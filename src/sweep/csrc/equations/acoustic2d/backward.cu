#include <torch/extension.h>
#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace acoustic2d {

namespace {

void init_rtm_output_2d(RTMOutput& out, const torch::Tensor& vp)
{
    out.image = torch::zeros_like(vp);
    out.source_illumination = torch::zeros_like(vp);
    out.receiver_illumination = torch::zeros_like(vp);
}

void accumulate_imaging_2d(
    dim3 wave_grid,
    dim3 wave_block,
    const float* forward_ptr,
    const float* adjoint_ptr,
    const torch::Tensor& vp,
    torch::Tensor* grad,
    RTMOutput* rtm_out,
    int nx,
    int nz
)
{
    if (grad != nullptr) {
        calculate_grad<<<wave_grid, wave_block>>>(
            forward_ptr,
            adjoint_ptr,
            vp.data_ptr<float>(),
            grad->data_ptr<float>(),
            nx, nz
        );
    }

    if (rtm_out != nullptr) {
        accumulate_rtm_image_2d<<<wave_grid, wave_block>>>(
            forward_ptr,
            adjoint_ptr,
            rtm_out->image.data_ptr<float>(),
            rtm_out->source_illumination.data_ptr<float>(),
            rtm_out->receiver_illumination.data_ptr<float>(),
            nx, nz
        );
    }
}

void accumulate_source_gradient_2d(
    dim3 source_grid,
    dim3 source_block,
    const float* adjoint_ptr,
    const BackwardInput& p,
    torch::Tensor* grad_wavelet,
    int it,
    const SolverContext& ctx,
    int nsrc
)
{
    if (grad_wavelet == nullptr) {
        return;
    }

    accumulate_source_grad_2d<<<source_grid, source_block>>>(
        adjoint_ptr,
        grad_wavelet->data_ptr<float>(),
        p.forward_sources_loc.data_ptr<int>(),
        it,
        nsrc,
        ctx
    );
}

void run_full_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    torch::Tensor* grad_wavelet,
    RTMOutput* rtm_out
)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    int M = p.M;
    float dt = p.dt;

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);

    float* u_thist = nullptr;

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto forward_source_config = fdtd::Geom::make(forward_nsrc, B);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = p.nt - 1; it >= 0; --it) {

        auto adj_view = adjoint.view();

        ACOUSTIC2D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            u_thist,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        accumulate_source_gradient_2d(
            forward_source_config.grid,
            forward_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            it,
            ctx,
            forward_nsrc
        );

        accumulate_imaging_2d(
            launch_config.grid,
            launch_config.block,
            p.u_forward[it].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            rtm_out,
            nx,
            nz
        );
    }
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    BackwardOutput out;
    auto grad = torch::zeros_like(in.models[0]);
    auto grad_wavelet = torch::zeros_like(in.forward_source);
    RTMOutput illumination;
    init_rtm_output_2d(illumination, in.models[0]);
    run_full_imaging(in, &grad, &grad_wavelet, &illumination);
    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    return out;
}

RTMOutput rtm(const BackwardInput& in)
{
    TORCH_CHECK(
        in.u_forward.defined() && in.u_forward.numel() > 0,
        "Acoustic2D RTM currently requires full forward wavefields."
    );
    TORCH_CHECK(
        !in.u_last_two.defined() || in.u_last_two.numel() == 0,
        "Acoustic2D RTM does not yet support boundary-saving mode."
    );
    TORCH_CHECK(
        in.checkpoints.empty(),
        "Acoustic2D RTM does not yet support checkpoint mode."
    );

    RTMOutput out;
    init_rtm_output_2d(out, in.models[0]);
    run_full_imaging(in, nullptr, nullptr, &out);
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{

    const auto& p = in;
    BackwardOutput out;

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    auto vp = p.models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);
    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, false);
    else
        forward.allocate(vp, 2, false);
    forward.u_prev_t.copy_(p.u_last_two.select(1,1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1,0).squeeze(0));

    auto grad = torch::zeros_like(vp);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_2d(illumination, vp);

    // For checking wavefields
    // torch::Tensor u_allt = torch::zeros({nt, B, 1, nz, nx}, vp.options());

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    // Boundary wavefields (for saving all wavefields)
    int save_width = p.abcn > 0 ? M + 1 : M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, false, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, true, 1, {}, p.boundary_gpu, {}, false, p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto for_view = forward.view();
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_prev, ctx.abcn+ctx.M, nx, nz, p.free_surface);
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_now, ctx.abcn+ctx.M, nx, nz, p.free_surface);
    
    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
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
        buf_idx = (it - 1) % interval;

        auto adj_view = adjoint.view();
        auto for_view = forward.view();

        // u_allt[it].copy_(forward.u_now_t);

        // adjoint modeling
        ACOUSTIC2D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );
        
        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        accumulate_source_gradient_2d(
            fwd_source_config.grid,
            fwd_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            &grad_wavelet,
            it,
            ctx,
            forward_nsrc
        );
        
        ACOUSTIC2D_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            vp.data_ptr<float>(),
            lap_ctx,
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
            staged_boundary ? 0 : it-1,
            save_width,
            0,
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
        
        calculate_grad_utt<<<launch_config.grid, launch_config.block>>>(
            forward.u_next_t.data_ptr<float>(),
            forward.u_now_t.data_ptr<float>(),
            forward.u_prev_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            nx, nz, dt
        );
        accumulate_rtm_image_2d<<<launch_config.grid, launch_config.block>>>(
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            illumination.image.data_ptr<float>(),
            illumination.source_illumination.data_ptr<float>(),
            illumination.receiver_illumination.data_ptr<float>(),
            nx, nz
        );

    }

    if (p.nt > 0) {
        auto adj_view = adjoint.view();

        ACOUSTIC2D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            0,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        accumulate_source_gradient_2d(
            fwd_source_config.grid,
            fwd_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            &grad_wavelet,
            0,
            ctx,
            forward_nsrc
        );
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    return out;

}

namespace {

void zero_wavefield_state_2d(AcousticWavefieldTensor& wf)
{
    wf.u_prev_t.zero_();
    wf.u_now_t.zero_();
    wf.u_next_t.zero_();
    wf.psix_t.zero_();
    wf.psiz_t.zero_();
    wf.zetax_t.zero_();
    wf.zetaz_t.zero_();
}

void copy_wavefield_state_2d(AcousticWavefieldTensor& dst, const AcousticWavefieldTensor& src)
{
    dst.u_prev_t.copy_(src.u_prev_t);
    dst.u_now_t.copy_(src.u_now_t);
    dst.u_next_t.copy_(src.u_next_t);
    dst.psix_t.copy_(src.psix_t);
    dst.psiz_t.copy_(src.psiz_t);
    dst.zetax_t.copy_(src.zetax_t);
    dst.zetaz_t.copy_(src.zetaz_t);
}

void load_checkpoint_state_2d(
    AcousticWavefieldTensor& dst,
    const std::vector<torch::Tensor>& checkpoints,
    int checkpoint_idx
)
{
    dst.u_prev_t.copy_(checkpoints[0].select(0, checkpoint_idx));
    dst.u_now_t.copy_(checkpoints[1].select(0, checkpoint_idx));
    dst.psix_t.copy_(checkpoints[2].select(0, checkpoint_idx));
    dst.psiz_t.copy_(checkpoints[3].select(0, checkpoint_idx));
    dst.zetax_t.copy_(checkpoints[4].select(0, checkpoint_idx));
    dst.zetaz_t.copy_(checkpoints[5].select(0, checkpoint_idx));
    dst.u_next_t.zero_();
}

void advance_forward_interval_2d(
    AcousticWavefieldTensor& forward,
    int start,
    int end,
    int order,
    dim3 wave_grid,
    dim3 wave_block,
    dim3 source_grid,
    dim3 source_block,
    const BackwardInput& p,
    const torch::Tensor& vp,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc
)
{
    for (int it = start; it < end; ++it) {
        auto view = forward.view();

        ACOUSTIC2D(
            order,
            wave_grid,
            wave_block,
            view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<source_grid, source_block>>>(
            view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        forward.swap();
    }
}

void process_recursive_interval_2d(
    int start,
    int end,
    const AcousticWavefieldTensor& start_state,
    AcousticWavefieldTensor& adjoint,
    const BackwardInput& p,
    const torch::Tensor& vp,
    torch::Tensor* grad,
    torch::Tensor* grad_wavelet,
    RTMOutput* rtm_out,
    int order,
    dim3 wave_grid,
    dim3 wave_block,
    dim3 forward_source_grid,
    dim3 forward_source_block,
    dim3 adj_source_grid,
    dim3 adj_source_block,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    int adjoint_nsrc,
    int nx,
    int nz
)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        AcousticWavefieldTensor forward_step;
        forward_step.allocate(vp, 2, true);
        copy_wavefield_state_2d(forward_step, start_state);

        auto u_this = torch::zeros_like(vp);
        auto fwd_view = forward_step.view();

        ACOUSTIC2D(
            order,
            wave_grid,
            wave_block,
            fwd_view,
            true,
            u_this.data_ptr<float>(),
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<forward_source_grid, forward_source_block>>>(
            fwd_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            start,
            forward_nsrc,
            ctx
        );

        forward_step.swap();

        auto adj_view = adjoint.view();

        ACOUSTIC2D(
            order,
            wave_grid,
            wave_block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        accumulate_source_gradient_2d(
            forward_source_grid,
            forward_source_block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            start,
            ctx,
            forward_nsrc
        );

        accumulate_imaging_2d(
            wave_grid,
            wave_block,
            u_this.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            rtm_out,
            nx,
            nz
        );
        return;
    }

    int mid = start + (end - start) / 2;

    AcousticWavefieldTensor mid_state;
    mid_state.allocate(vp, 2, true);
    copy_wavefield_state_2d(mid_state, start_state);
    advance_forward_interval_2d(
        mid_state,
        start,
        mid,
        order,
        wave_grid,
        wave_block,
        forward_source_grid,
        forward_source_block,
        p,
        vp,
        lap_ctx,
        grad_ctx,
        grad_ctx_x,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc
    );

    process_recursive_interval_2d(
        mid,
        end,
        mid_state,
        adjoint,
        p,
        vp,
        grad,
        grad_wavelet,
        rtm_out,
        order,
        wave_grid,
        wave_block,
        forward_source_grid,
        forward_source_block,
        adj_source_grid,
        adj_source_block,
        lap_ctx,
        grad_ctx,
        grad_ctx_x,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        nx,
        nz
    );

    process_recursive_interval_2d(
        start,
        mid,
        start_state,
        adjoint,
        p,
        vp,
        grad,
        grad_wavelet,
        rtm_out,
        order,
        wave_grid,
        wave_block,
        forward_source_grid,
        forward_source_block,
        adj_source_grid,
        adj_source_block,
        lap_ctx,
        grad_ctx,
        grad_ctx_x,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        nx,
        nz
    );
}

} // namespace

BackwardOutput backward_ckpt(const BackwardInput& in)
{

    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(
        p.checkpoints.size() == 6,
        "Acoustic 2D checkpointing expects 6 checkpoint tensors"
    );

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    auto vp = p.models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, true);
    else
        forward.allocate(vp, 2, true);

    auto grad = torch::zeros_like(vp);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_2d(illumination, vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    auto chunk_forward = torch::zeros({chunk_size, B, nz, nx}, vp.options());

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        forward.u_prev_t.copy_(p.checkpoints[0].select(0, chunk_id));
        forward.u_now_t.copy_(p.checkpoints[1].select(0, chunk_id));
        forward.psix_t.copy_(p.checkpoints[2].select(0, chunk_id));
        forward.psiz_t.copy_(p.checkpoints[3].select(0, chunk_id));
        forward.zetax_t.copy_(p.checkpoints[4].select(0, chunk_id));
        forward.zetaz_t.copy_(p.checkpoints[5].select(0, chunk_id));

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* u_this = chunk_forward[it - start].data_ptr<float>();

            ACOUSTIC2D(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
                true,
                u_this,
                vp.data_ptr<float>(),
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
        }

        for (int it = end - 1; it >= start; --it) {
            auto adj_view = adjoint.view();

            ACOUSTIC2D(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
                false,
                nullptr,
                vp.data_ptr<float>(),
                lap_ctx,
                grad_ctx,
                grad_ctx_x,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap();

            accumulate_source_gradient_2d(
                fwd_source_config.grid,
                fwd_source_config.block,
                adjoint.u_now_t.data_ptr<float>(),
                p,
                &grad_wavelet,
                it,
                ctx,
                forward_nsrc
            );

            accumulate_imaging_2d(
                launch_config.grid,
                launch_config.block,
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp,
                &grad,
                &illumination,
                nx,
                nz
            );
        }
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    return out;
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(
        p.checkpoints.size() == 6,
        "Acoustic 2D recursive checkpointing expects 6 checkpoint tensors"
    );

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    float dt = p.dt;

    auto vp = p.models[0];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);
    zero_wavefield_state_2d(adjoint);

    auto grad = torch::zeros_like(vp);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_2d(illumination, vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    TORCH_CHECK(
        p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
        "checkpoint_count does not match checkpoint_steps"
    );
    TORCH_CHECK(
        static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
        "checkpoint buffer is smaller than checkpoint_steps"
    );

    const int* checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();

    AcousticWavefieldTensor start_state;
    start_state.allocate(vp, 2, true);

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            zero_wavefield_state_2d(start_state);
        else
            load_checkpoint_state_2d(start_state, p.checkpoints, segment_idx - 1);

        process_recursive_interval_2d(
            start,
            end,
            start_state,
            adjoint,
            p,
            vp,
            &grad,
            &grad_wavelet,
            &illumination,
            order,
            launch_config.grid,
            launch_config.block,
            fwd_source_config.grid,
            fwd_source_config.block,
            adj_source_config.grid,
            adj_source_config.block,
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx,
            forward_nsrc,
            adjoint_nsrc,
            nx,
            nz
        );
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    return out;
}

}
