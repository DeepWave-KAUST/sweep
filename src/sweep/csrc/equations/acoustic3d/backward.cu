#include <torch/extension.h>
#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../common/boundarysaver.cuh"
#include "../../launch/config.h"

namespace acoustic3d {

namespace {

BackwardOutput backward_full_imaging_impl(const BackwardInput& p);
BackwardOutput backward_bs_imaging_impl(const BackwardInput& p);
BackwardOutput backward_ckpt_imaging_impl(const BackwardInput& p);
BackwardOutput backward_recursive_imaging_impl(const BackwardInput& p);
RTMOutput rtm_full_impl(const BackwardInput& p);
RTMOutput rtm_bs_impl(const BackwardInput& p);
RTMOutput rtm_ckpt_impl(const BackwardInput& p);
RTMOutput rtm_recursive_ckpt_impl(const BackwardInput& p);

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    return backward_full_imaging_impl(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return backward_bs_imaging_impl(in);
}

namespace {

void init_rtm_output_3d(RTMOutput& out, const torch::Tensor& vp)
{
    out.image = torch::zeros_like(vp);
    out.source_illumination = torch::zeros_like(vp);
    out.receiver_illumination = torch::zeros_like(vp);
}

void accumulate_rtm_3d(
    dim3 wave_grid,
    dim3 wave_block,
    const float* forward_ptr,
    const float* adjoint_ptr,
    RTMOutput& out,
    int B,
    int nx,
    int ny,
    int nz
)
{
    accumulate_rtm_image_3d<<<wave_grid, wave_block>>>(
        forward_ptr,
        adjoint_ptr,
        out.image.data_ptr<float>(),
        out.source_illumination.data_ptr<float>(),
        out.receiver_illumination.data_ptr<float>(),
        B, nx, ny, nz
    );
}

void accumulate_imaging_3d(
    dim3 wave_grid,
    dim3 wave_block,
    const float* forward_ptr,
    const float* adjoint_ptr,
    const torch::Tensor& vp,
    torch::Tensor* grad,
    RTMOutput* rtm_out,
    int B,
    int nx,
    int ny,
    int nz
)
{
    if (grad != nullptr) {
        calculate_grad_3d<<<wave_grid, wave_block>>>(
            forward_ptr,
            adjoint_ptr,
            vp.data_ptr<float>(),
            grad->data_ptr<float>(),
            B, nx, ny, nz
        );
        return;
    }

    TORCH_CHECK(rtm_out != nullptr, "Imaging accumulation requires grad or RTM output.");
    accumulate_rtm_3d(wave_grid, wave_block, forward_ptr, adjoint_ptr, *rtm_out, B, nx, ny, nz);
}

void accumulate_imaging_utt_3d(
    dim3 wave_grid,
    dim3 wave_block,
    const float* u_next_ptr,
    const float* u_now_ptr,
    const float* u_prev_ptr,
    const float* adjoint_ptr,
    const torch::Tensor& vp,
    float dt,
    torch::Tensor* grad,
    RTMOutput* rtm_out,
    int B,
    int nx,
    int ny,
    int nz
)
{
    if (grad != nullptr) {
        calculate_grad_utt_3d<<<wave_grid, wave_block>>>(
            u_next_ptr,
            u_now_ptr,
            u_prev_ptr,
            adjoint_ptr,
            vp.data_ptr<float>(),
            grad->data_ptr<float>(),
            B, nx, ny, nz, dt
        );
        return;
    }

    TORCH_CHECK(rtm_out != nullptr, "Imaging accumulation requires grad or RTM output.");
    accumulate_rtm_3d(wave_grid, wave_block, u_now_ptr, adjoint_ptr, *rtm_out, B, nx, ny, nz);
}

void zero_wavefield_state_3d(AcousticWavefieldTensor& wf)
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

void copy_wavefield_state_3d(AcousticWavefieldTensor& dst, const AcousticWavefieldTensor& src)
{
    dst.u_prev_t.copy_(src.u_prev_t);
    dst.u_now_t.copy_(src.u_now_t);
    dst.u_next_t.copy_(src.u_next_t);
    dst.psix_t.copy_(src.psix_t);
    dst.psiy_t.copy_(src.psiy_t);
    dst.psiz_t.copy_(src.psiz_t);
    dst.zetax_t.copy_(src.zetax_t);
    dst.zetay_t.copy_(src.zetay_t);
    dst.zetaz_t.copy_(src.zetaz_t);
}

void load_checkpoint_state_3d(
    AcousticWavefieldTensor& dst,
    const std::vector<torch::Tensor>& checkpoints,
    int checkpoint_idx
)
{
    dst.u_prev_t.copy_(checkpoints[0].select(0, checkpoint_idx));
    dst.u_now_t.copy_(checkpoints[1].select(0, checkpoint_idx));
    dst.psix_t.copy_(checkpoints[2].select(0, checkpoint_idx));
    dst.psiy_t.copy_(checkpoints[3].select(0, checkpoint_idx));
    dst.psiz_t.copy_(checkpoints[4].select(0, checkpoint_idx));
    dst.zetax_t.copy_(checkpoints[5].select(0, checkpoint_idx));
    dst.zetay_t.copy_(checkpoints[6].select(0, checkpoint_idx));
    dst.zetaz_t.copy_(checkpoints[7].select(0, checkpoint_idx));
    dst.u_next_t.zero_();
}

void advance_forward_interval_3d(
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
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc
)
{
    for (int it = start; it < end; ++it) {
        auto view = forward.view();

        ACOUSTIC3D(
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<source_grid, source_block>>>(
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

void process_recursive_interval_3d(
    int start,
    int end,
    const AcousticWavefieldTensor& start_state,
    AcousticWavefieldTensor& adjoint,
    const BackwardInput& p,
    const torch::Tensor& vp,
    torch::Tensor* grad,
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
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    int adjoint_nsrc,
    int B,
    int nx,
    int ny,
    int nz
)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        AcousticWavefieldTensor forward_step;
        forward_step.allocate(vp, 3, true);
        copy_wavefield_state_3d(forward_step, start_state);

        auto u_this = torch::zeros_like(vp);
        auto fwd_view = forward_step.view();

        ACOUSTIC3D(
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<forward_source_grid, forward_source_block>>>(
            fwd_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            start,
            forward_nsrc,
            ctx
        );

        forward_step.swap();

        auto adj_view = adjoint.view();

        ACOUSTIC3D(
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        if (grad != nullptr) {
            calculate_grad_3d<<<wave_grid, wave_block>>>(
                u_this.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                grad->data_ptr<float>(),
                B, nx, ny, nz
            );
        } else {
            TORCH_CHECK(rtm_out != nullptr, "Recursive RTM accumulation requested without RTM output.");
            accumulate_rtm_3d(
                wave_grid,
                wave_block,
                u_this.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                *rtm_out,
                B,
                nx,
                ny,
                nz
            );
        }
        return;
    }

    int mid = start + (end - start) / 2;

    AcousticWavefieldTensor mid_state;
    mid_state.allocate(vp, 3, true);
    copy_wavefield_state_3d(mid_state, start_state);
    advance_forward_interval_3d(
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc
    );

    process_recursive_interval_3d(
        mid,
        end,
        mid_state,
        adjoint,
        p,
        vp,
        grad,
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        B,
        nx,
        ny,
        nz
    );

    process_recursive_interval_3d(
        start,
        mid,
        start_state,
        adjoint,
        p,
        vp,
        grad,
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        B,
        nx,
        ny,
        nz
    );
}

void run_full_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    RTMOutput* rtm_out
)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);

    float* u_thist = nullptr;

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        ACOUSTIC3D(
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        accumulate_imaging_3d(
            launch_config.grid,
            launch_config.block,
            p.u_forward[it].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            rtm_out,
            B,
            nx,
            ny,
            nz
        );
    }
}

BackwardOutput backward_full_imaging_impl(const BackwardInput& p)
{
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    run_full_imaging(p, &grad, nullptr);
    out.grads = {grad};
    return out;
}

RTMOutput rtm_full_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_full_imaging(p, nullptr, &out);
    return out;
}

void run_bs_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    RTMOutput* rtm_out
)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, nullptr, nullptr, dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 3, false);
    else
        forward.allocate(vp, 3, false);
    forward.u_prev_t.copy_(p.u_last_two.select(1,1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1,0).squeeze(0));

    auto f_this = torch::zeros_like(vp);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
            {}, false, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, {}, false, p.use_pinned_memory
        );
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    int interval = p.transfer_interval;
    int buf_idx = 0;
    int gpu_idx = 0;

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

        ACOUSTIC3D(
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        ACOUSTIC3D_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            f_this.data_ptr<float>(),
            vp.data_ptr<float>(),
            lap_ctx,
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

        float* top_ptr = nullptr;
        float* bottom_ptr = nullptr;
        float* front_ptr = nullptr;
        float* back_ptr = nullptr;
        float* left_ptr = nullptr;
        float* right_ptr = nullptr;

        if (staged_boundary && buf_idx == interval - 1)
            async_copy.wait_for_copy();

        if (staged_boundary) {
            gpu_idx = buf_idx;
            top_ptr = boundary_saver.top_gpu.data_ptr<float>() + gpu_idx * boundary_saver.top_stride;
            bottom_ptr = boundary_saver.bottom_gpu.data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride;
            front_ptr = boundary_saver.front_gpu.data_ptr<float>() + gpu_idx * boundary_saver.front_stride;
            back_ptr = boundary_saver.back_gpu.data_ptr<float>() + gpu_idx * boundary_saver.back_stride;
            left_ptr = boundary_saver.left_gpu.data_ptr<float>() + gpu_idx * boundary_saver.left_stride;
            right_ptr = boundary_saver.right_gpu.data_ptr<float>() + gpu_idx * boundary_saver.right_stride;
        } else {
            top_ptr = bs.top;
            bottom_ptr = bs.bottom;
            front_ptr = bs.front;
            back_ptr = bs.back;
            left_ptr = bs.left;
            right_ptr = bs.right;
        }

        boundary_kernel3d<<<launch_config.grid, launch_config.block>>>(
            for_view.u_next,
            top_ptr,
            bottom_ptr,
            front_ptr,
            back_ptr,
            left_ptr,
            right_ptr,
            staged_boundary ? 0 : it - 1,
            save_width,
            0,
            ctx,
            BOUNDARY_RESTORE
        );

        forward.swap();

        if (staged_boundary && buf_idx == 0 && it > 1) {
            int chunk_id = (it - 1) / interval;
            int next_chunk = chunk_id - 1;
            if (next_chunk >= 0) {
                int next_start = next_chunk * interval;
                int remain = static_cast<int>(p.nt) - next_start;
                int next_len = std::min(interval, remain);
                boundary_saver.load_cpu_to_gpu(next_start, next_len, async_copy.copy_stream);
                async_copy.record_copy_ready();
            }
        }

        accumulate_imaging_utt_3d(
            launch_config.grid,
            launch_config.block,
            forward.u_next_t.data_ptr<float>(),
            forward.u_now_t.data_ptr<float>(),
            forward.u_prev_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            p.dt,
            grad,
            rtm_out,
            B,
            nx,
            ny,
            nz
        );
    }
}

BackwardOutput backward_bs_imaging_impl(const BackwardInput& p)
{
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    run_bs_imaging(p, &grad, nullptr);
    out.grads = {grad};
    return out;
}

RTMOutput rtm_bs_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_bs_imaging(p, nullptr, &out);
    return out;
}

void run_ckpt_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    RTMOutput* rtm_out
)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 3, true);
    else
        forward.allocate(vp, 3, true);

    auto chunk_forward = torch::zeros({p.checkpoint_interval, B, nz, ny, nx}, vp.options());

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        forward.u_prev_t.copy_(p.checkpoints[0].select(0, chunk_id));
        forward.u_now_t.copy_(p.checkpoints[1].select(0, chunk_id));
        forward.psix_t.copy_(p.checkpoints[2].select(0, chunk_id));
        forward.psiy_t.copy_(p.checkpoints[3].select(0, chunk_id));
        forward.psiz_t.copy_(p.checkpoints[4].select(0, chunk_id));
        forward.zetax_t.copy_(p.checkpoints[5].select(0, chunk_id));
        forward.zetay_t.copy_(p.checkpoints[6].select(0, chunk_id));
        forward.zetaz_t.copy_(p.checkpoints[7].select(0, chunk_id));

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* u_this = chunk_forward[it - start].data_ptr<float>();

            ACOUSTIC3D(
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
        }

        for (int it = end - 1; it >= start; --it) {
            auto adj_view = adjoint.view();

            ACOUSTIC3D(
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
                grad_ctx_y,
                grad_ctx_z,
                cpml,
                ctx
            );

            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap();

            accumulate_imaging_3d(
                launch_config.grid,
                launch_config.block,
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp,
                grad,
                rtm_out,
                B,
                nx,
                ny,
                nz
            );
        }
    }
}

BackwardOutput backward_ckpt_imaging_impl(const BackwardInput& p)
{
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    run_ckpt_imaging(p, &grad, nullptr);
    out.grads = {grad};
    return out;
}

RTMOutput rtm_ckpt_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_ckpt_imaging(p, nullptr, &out);
    return out;
}

void run_recursive_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    RTMOutput* rtm_out
)
{
    auto vp = p.models[0];

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);
    zero_wavefield_state_3d(adjoint);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

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
    start_state.allocate(vp, 3, true);

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            zero_wavefield_state_3d(start_state);
        else
            load_checkpoint_state_3d(start_state, p.checkpoints, segment_idx - 1);

        process_recursive_interval_3d(
            start,
            end,
            start_state,
            adjoint,
            p,
            vp,
            grad,
            rtm_out,
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx,
            forward_nsrc,
            adjoint_nsrc,
            B,
            nx,
            ny,
            nz
        );
    }

}

BackwardOutput backward_recursive_imaging_impl(const BackwardInput& p)
{
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    run_recursive_imaging(p, &grad, nullptr);
    out.grads = {grad};
    return out;
}

RTMOutput rtm_recursive_ckpt_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_recursive_imaging(p, nullptr, &out);
    return out;
}

} // namespace

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return backward_ckpt_imaging_impl(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return backward_recursive_imaging_impl(in);
}

RTMOutput rtm(const BackwardInput& in)
{
    const auto& p = in;
    if (!p.checkpoints.empty()) {
        if (p.checkpoint_steps.defined() && p.checkpoint_steps.numel() > 0) {
            return rtm_recursive_ckpt_impl(p);
        }
        return rtm_ckpt_impl(p);
    }
    if (p.u_last_two.defined() && p.u_last_two.numel() > 0) {
        return rtm_bs_impl(p);
    }
    TORCH_CHECK(
        p.u_forward.defined() && p.u_forward.numel() > 0,
        "Acoustic3D RTM requires full wavefields, boundary buffers, or checkpoints."
    );
    return rtm_full_impl(p);
}

}
