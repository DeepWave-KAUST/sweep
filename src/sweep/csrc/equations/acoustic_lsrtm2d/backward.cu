#include <algorithm>

#include <torch/extension.h>

#include "acoustic_lsrtm2d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace acoustic_lsrtm2d {

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

        ACOUSTIC_LSRTM2D_SINGLE(
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
    torch::Tensor& grad_mp,
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

        auto bg_utt = torch::zeros_like(vp);
        auto fwd_view = forward_step.view();

        ACOUSTIC_LSRTM2D_SINGLE(
            order,
            wave_grid,
            wave_block,
            fwd_view,
            true,
            bg_utt.data_ptr<float>(),
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
        ACOUSTIC_LSRTM2D_SINGLE(
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

        calculate_grad_lsrtm_mp<<<wave_grid, wave_block>>>(
            bg_utt.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx,
            nz,
            ctx.dt
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
        grad_mp,
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
        grad_mp,
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

void run_full_imaging(const BackwardInput& p, torch::Tensor& grad_mp)
{
    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int B = N * C;
    int M = p.M;

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 7), 2, true);
    else
        adjoint.allocate(vp, 2, true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;
    SolverContext ctx{2, nx, 0, nz, B, p.dt, p.nt, M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        ACOUSTIC_LSRTM2D_SINGLE(
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

        calculate_grad_lsrtm_mp<<<launch_config.grid, launch_config.block>>>(
            p.u_forward[it].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx,
            nz,
            ctx.dt
        );
    }
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    BackwardOutput out;
    TORCH_CHECK(in.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");

    auto grad_wavelet = torch::zeros_like(in.forward_source);
    auto grad_vp = torch::zeros_like(in.models[0]);
    auto grad_mp = torch::zeros_like(in.models[1]);

    run_full_imaging(in, grad_mp);

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");

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

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 7), 2, true);
    else
        adjoint.allocate(vp, 2, true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(std::vector<torch::Tensor>(p.forward_wavefields.begin(), p.forward_wavefields.begin() + 7), 2, true);
    else
        forward.allocate(vp, 2, true);
    forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));

    auto grad_wavelet = torch::zeros_like(p.forward_source);
    auto grad_vp = torch::zeros_like(p.models[0]);
    auto grad_mp = torch::zeros_like(p.models[1]);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    int save_width = p.abcn > 0 ? M + 1 : M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
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

    auto for_view = forward.view();
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_prev, ctx.abcn + ctx.M, nx, nz, p.free_surface);
    set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.u_now, ctx.abcn + ctx.M, nx, nz, p.free_surface);

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

    for (int it = p.nt - 1; it >= 1; --it) {
        auto adj_view = adjoint.view();
        auto for_view_iter = forward.view();

        ACOUSTIC_LSRTM2D_SINGLE(
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

        ACOUSTIC_LSRTM2D_SINGLE_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view_iter,
            vp.data_ptr<float>(),
            lap_ctx,
            ctx
        );

        add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view_iter.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        boundary_runtime.restore_backward_2d(
            it,
            for_view_iter.u_next,
            launch_config.grid,
            launch_config.block,
            bs,
            save_width,
            0,
            ctx
        );

        forward.swap();

        calculate_grad_lsrtm_mp_utt<<<launch_config.grid, launch_config.block>>>(
            forward.u_next_t.data_ptr<float>(),
            forward.u_now_t.data_ptr<float>(),
            forward.u_prev_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad_mp.data_ptr<float>(),
            nx, nz, dt
        );
    }

    if (p.nt > 0) {
        auto adj_view = adjoint.view();
        ACOUSTIC_LSRTM2D_SINGLE(
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
    }

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");
    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 6, "Acoustic LSRTM 2D checkpointing expects 6 checkpoint tensors");

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

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 7), 2, true);
    else
        adjoint.allocate(vp, 2, true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(std::vector<torch::Tensor>(p.forward_wavefields.begin(), p.forward_wavefields.begin() + 7), 2, true);
    else
        forward.allocate(vp, 2, true);

    auto grad_wavelet = torch::zeros_like(p.forward_source);
    auto grad_vp = torch::zeros_like(p.models[0]);
    auto grad_mp = torch::zeros_like(p.models[1]);

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

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    auto chunk_forward = torch::zeros({chunk_size, B, nz, nx}, vp.options());

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        load_checkpoint_state_2d(forward, p.checkpoints, chunk_id);

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* bg_utt = chunk_forward[it - start].data_ptr<float>();

            ACOUSTIC_LSRTM2D_SINGLE(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
                true,
                bg_utt,
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

            ACOUSTIC_LSRTM2D_SINGLE(
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

            calculate_grad_lsrtm_mp<<<launch_config.grid, launch_config.block>>>(
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                grad_mp.data_ptr<float>(),
                nx,
                nz,
                ctx.dt
            );
        }
    }

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D backward expects two models.");
    TORCH_CHECK(p.checkpoints.size() == 6, "Acoustic LSRTM 2D recursive checkpointing expects 6 checkpoint tensors");

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

    const int order = (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(std::vector<torch::Tensor>(p.adjoint_wavefields.begin(), p.adjoint_wavefields.begin() + 7), 2, true);
    else
        adjoint.allocate(vp, 2, true);
    zero_wavefield_state_2d(adjoint);

    auto grad_wavelet = torch::zeros_like(p.forward_source);
    auto grad_vp = torch::zeros_like(p.models[0]);
    auto grad_mp = torch::zeros_like(p.models[1]);

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
            grad_mp,
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

    out.grads = {grad_wavelet, grad_vp, grad_mp};
    return out;
}

} // namespace acoustic_lsrtm2d
