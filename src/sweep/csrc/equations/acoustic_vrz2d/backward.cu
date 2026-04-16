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

void copy_wavefield_state_vrz(AcousticWavefieldTensor& dst, const AcousticWavefieldTensor& src)
{
    dst.u_prev_t.copy_(src.u_prev_t);
    dst.u_now_t.copy_(src.u_now_t);
    dst.u_next_t.copy_(src.u_next_t);
    dst.psix_t.copy_(src.psix_t);
    dst.psiz_t.copy_(src.psiz_t);
    dst.zetax_t.copy_(src.zetax_t);
    dst.zetaz_t.copy_(src.zetaz_t);
}

void load_checkpoint_state_vrz(
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

void save_forward_state_vrz(torch::Tensor& history, int idx, const AcousticWavefieldTensor& forward)
{
    history.select(0, idx).select(0, 0).copy_(forward.u_now_t);
    history.select(0, idx).select(0, 1).copy_(forward.psix_t);
    history.select(0, idx).select(0, 2).copy_(forward.psiz_t);
    history.select(0, idx).select(0, 3).copy_(forward.zetax_t);
    history.select(0, idx).select(0, 4).copy_(forward.zetaz_t);
}

void accumulate_grad_from_history_vrz(
    int order,
    dim3 wave_grid,
    dim3 wave_block,
    const torch::Tensor& history,
    int history_idx,
    const torch::Tensor& adjoint_now,
    const torch::Tensor& vp,
    const torch::Tensor& z,
    torch::Tensor& grad_vp,
    torch::Tensor& grad_z,
    const AcousticCPMLPointer& cpml,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const LaplaceParam& lap_ctx,
    const SolverContext& ctx
)
{
    CALCULATE_GRAD_VRZ2D(
        order,
        wave_grid,
        wave_block,
        history.select(0, history_idx).select(0, 0).data_ptr<float>(),
        history.select(0, history_idx).select(0, 1).data_ptr<float>(),
        history.select(0, history_idx).select(0, 2).data_ptr<float>(),
        history.select(0, history_idx).select(0, 3).data_ptr<float>(),
        history.select(0, history_idx).select(0, 4).data_ptr<float>(),
        adjoint_now.data_ptr<float>(),
        vp.data_ptr<float>(),
        z.data_ptr<float>(),
        grad_vp.data_ptr<float>(),
        grad_z.data_ptr<float>(),
        cpml,
        grad_ctx,
        grad_ctx_x,
        grad_ctx_z,
        lap_ctx,
        ctx
    );
}

BackwardOutput backward_from_history(const BackwardInput& p, const torch::Tensor& u_forward)
{
    BackwardOutput out;

    auto vp = p.models[0];
    auto z = p.models[1];

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

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        ACOUSTIC_VRZ2D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
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

        accumulate_grad_from_history_vrz(
            order,
            launch_config.grid,
            launch_config.block,
            u_forward,
            it,
            adjoint.u_now_t,
            vp,
            z,
            grad_vp,
            grad_z,
            cpml,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            lap_ctx,
            ctx
        );
    }

    out.grads = {grad_vp, grad_z};
    return out;
}

void process_segment_vrz(
    int start,
    int end,
    AcousticWavefieldTensor& forward,
    AcousticWavefieldTensor& adjoint,
    const BackwardInput& p,
    const torch::Tensor& vp,
    const torch::Tensor& z,
    torch::Tensor& grad_vp,
    torch::Tensor& grad_z,
    int order,
    dim3 wave_grid,
    dim3 wave_block,
    dim3 fwd_source_grid,
    dim3 fwd_source_block,
    dim3 adj_source_grid,
    dim3 adj_source_block,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    int adjoint_nsrc
)
{
    if (start >= end)
        return;

    auto history = torch::zeros(
        {end - start, 5, vp.size(0) * vp.size(1), 1, vp.size(2), vp.size(3)},
        vp.options()
    );

    for (int it = start; it < end; ++it) {
        save_forward_state_vrz(history, it - start, forward);

        auto for_view = forward.view();
        ACOUSTIC_VRZ2D(
            order,
            wave_grid,
            wave_block,
            for_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        add_source<<<fwd_source_grid, fwd_source_block>>>(
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
        ACOUSTIC_VRZ2D(
            order,
            wave_grid,
            wave_block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
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
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap();

        accumulate_grad_from_history_vrz(
            order,
            wave_grid,
            wave_block,
            history,
            it - start,
            adjoint.u_now_t,
            vp,
            z,
            grad_vp,
            grad_z,
            cpml,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            lap_ctx,
            ctx
        );
    }
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
    return backward_from_history(in, in.u_forward);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ backward expects models [vp, z].");
    TORCH_CHECK(p.u_last_two.defined() && p.u_last_two.numel() > 0, "AcousticVRZ backward_bs requires u_last_two.");
    TORCH_CHECK(
        p.forward_source.defined() && p.forward_sources_loc.defined(),
        "AcousticVRZ backward_bs requires forward_source and forward_sources_loc."
    );

    auto vp = p.models[0];
    auto z = p.models[1];

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
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, false);
    else
        forward.allocate(vp, 2, false);
    forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
    forward.u_next_t.zero_();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    int save_width = 2 * M + 1;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {},
                                false, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, {}, false, p.use_pinned_memory);
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

    int interval = p.transfer_interval;
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
        int buf_idx = (it - 1) % interval;
        auto adj_view = adjoint.view();
        auto recon_view = forward.view();

        ACOUSTIC_VRZ2D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
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

        ACOUSTIC_VRZ2D_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            recon_view,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            ctx
        );

        add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
            recon_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        if (staged_boundary && buf_idx == interval - 1)
            async_copy.wait_for_copy();

        float* top_ptr = staged_boundary ? boundary_saver.top_gpu.data_ptr<float>() + buf_idx * boundary_saver.top_stride : bs.top;
        float* bottom_ptr = staged_boundary ? boundary_saver.bottom_gpu.data_ptr<float>() + buf_idx * boundary_saver.bottom_stride : bs.bottom;
        float* left_ptr = staged_boundary ? boundary_saver.left_gpu.data_ptr<float>() + buf_idx * boundary_saver.left_stride : bs.left;
        float* right_ptr = staged_boundary ? boundary_saver.right_gpu.data_ptr<float>() + buf_idx * boundary_saver.right_stride : bs.right;

        boundary_kernel2d<<<launch_config.grid, launch_config.block>>>(
            recon_view.u_next,
            top_ptr,
            bottom_ptr,
            left_ptr,
            right_ptr,
            staged_boundary ? 0 : it - 1,
            save_width,
            -2 * p.M,
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

        CALCULATE_GRAD_VRZ2D_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            forward.u_prev_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
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
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ backward expects models [vp, z].");
    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 6, "AcousticVRZ checkpointing expects 6 checkpoint tensors");
    TORCH_CHECK(
        p.forward_source.defined() && p.forward_sources_loc.defined(),
        "AcousticVRZ backward_ckpt requires forward_source and forward_sources_loc."
    );

    auto vp = p.models[0];
    auto z = p.models[1];

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
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, true);
    else
        forward.allocate(vp, 2, true);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);

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
    TORCH_CHECK(
        static_cast<int>(p.checkpoints[0].size(0)) >= num_chunks,
        "checkpoint buffer is smaller than required checkpoint chunks"
    );

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        load_checkpoint_state_vrz(forward, p.checkpoints, chunk_id);

        process_segment_vrz(
            start,
            end,
            forward,
            adjoint,
            p,
            vp,
            z,
            grad_vp,
            grad_z,
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
            adjoint_nsrc
        );
    }

    out.grads = {grad_vp, grad_z};
    return out;
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ backward expects models [vp, z].");
    TORCH_CHECK(p.checkpoints.size() == 6, "AcousticVRZ recursive checkpointing expects 6 checkpoint tensors");
    TORCH_CHECK(p.checkpoint_steps.defined(), "AcousticVRZ recursive checkpointing expects checkpoint_steps");
    TORCH_CHECK(
        p.forward_source.defined() && p.forward_sources_loc.defined(),
        "AcousticVRZ backward_recursive_ckpt requires forward_source and forward_sources_loc."
    );

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");

    auto vp = p.models[0];
    auto z = p.models[1];

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

    AcousticWavefieldTensor start_state;
    start_state.allocate(vp, 2, true);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);

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

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            zero_wavefield_state_vrz(start_state);
        else
            load_checkpoint_state_vrz(start_state, p.checkpoints, segment_idx - 1);

        AcousticWavefieldTensor forward_segment;
        forward_segment.allocate(vp, 2, true);
        copy_wavefield_state_vrz(forward_segment, start_state);

        process_segment_vrz(
            start,
            end,
            forward_segment,
            adjoint,
            p,
            vp,
            z,
            grad_vp,
            grad_z,
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
            adjoint_nsrc
        );
    }

    out.grads = {grad_vp, grad_z};
    return out;
}

} // namespace acoustic_vrz2d
