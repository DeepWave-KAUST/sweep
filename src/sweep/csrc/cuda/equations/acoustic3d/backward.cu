#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
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
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_full_imaging_impl(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_bs_imaging_impl(in);
}

namespace {

// Scratch for the 3D exact discrete-adjoint step: 9 per-cell fields packed
// into one zeros tensor (halo/air cells stay 0; see acoustic_adjoint_prepare_3d).

// One exact-adjoint 3D time step: PREPARE (local scratch) then APPLY
// (transposed stencils on scratch).  Replaces compute_v2_lambda_3d +
// acoustic_adjoint_kernel_3d (which copied the forward in the PML band).
static inline void run_acoustic3d_adjoint_step(
    int order, dim3 grid, dim3 block,
    AcousticWavefieldPointer adj_view,
    const float* vp_ptr,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_y, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx,
    const float* grad_forward_img = nullptr, float* grad_out = nullptr)
{
    // FUSED single-kernel exact 3D adjoint: recompute the per-cell g_* inline at
    // each stencil tap and write next-step psi/zeta into the wavefield's
    // double-buffer out-tensors (psi*n/zeta*n).  Read-old/write-new => race-free
    // even though aux is read at neighbours; no 9-field scratch round-trip =>
    // ~forward bandwidth.  Caller must swap_aux() each step.
    TORCH_CHECK(adj_view.zetaxn != nullptr && adj_view.psixn != nullptr,
        "fused 3D adjoint needs the adjoint wavefield bound with psi+zeta "
        "double-buffer (15 tensors); set cuda_layout.adjoint_extra_nvar=3.");
    (void)grad_ctx;
    ACOUSTIC3D_ADJOINT_FUSED(order, grid, block,
        adj_view, vp_ptr, lap_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z, cpml, ctx,
        adj_view.psixn, adj_view.psiyn, adj_view.psizn,
        adj_view.zetaxn, adj_view.zetayn, adj_view.zetazn,
        grad_forward_img, grad_out);
}

void init_rtm_output_3d(RTMOutput& out, const torch::Tensor& vp,
                        bool want_adcig = false, int nlag = 1)
{
    out.image = torch::zeros_like(vp);
    out.source_illumination = torch::zeros_like(vp);
    out.receiver_illumination = torch::zeros_like(vp);
    if (want_adcig) {
        // vp is (N, C, nz, ny, nx); ADCIG cube (nlag, N, C, nz, ny, nx),
        // runtime-padded, cropped + summed over batch Python-side.
        out.adcig = torch::zeros({(long)nlag, vp.size(0), vp.size(1),
                                  vp.size(2), vp.size(3), vp.size(4)}, vp.options());
    }
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
    int nz,
    float dt
)
{
    if (grad != nullptr) {
        calculate_grad_3d<<<wave_grid, wave_block>>>(
            forward_ptr,
            adjoint_ptr,
            vp.data_ptr<float>(),
            grad->data_ptr<float>(),
            B, nx, ny, nz, dt
        );
    }

    if (rtm_out != nullptr) {
        accumulate_rtm_3d(wave_grid, wave_block, forward_ptr, adjoint_ptr, *rtm_out, B, nx, ny, nz);
    }
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
    }

    if (rtm_out != nullptr) {
        accumulate_rtm_3d(wave_grid, wave_block, u_now_ptr, adjoint_ptr, *rtm_out, B, nx, ny, nz);
    }
    // Space-lag ADCIG on the reconstructed raw pressure ``u_now_ptr`` (this
    // BS-only helper is the one imaging site that has the raw pressure; the
    // full/ckpt helpers correlate vp^2*Lap(u), so ADCIG is not launched there).
    if (rtm_out != nullptr && rtm_out->adcig.defined() && rtm_out->adcig.numel() > 0) {
        int nlag = rtm_out->adcig.size(0);
        int Bloc = rtm_out->adcig.size(1) * rtm_out->adcig.size(2);  // N*C
        int max_lag = (nlag - 1) / 2;
        accumulate_adcig_3d<<<wave_grid, wave_block>>>(
            u_now_ptr, adjoint_ptr, rtm_out->adcig.data_ptr<float>(),
            nlag, max_lag, Bloc, nx, ny, nz
        );
    }
}

void accumulate_source_gradient_3d(
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

    accumulate_source_grad_3d<<<source_grid, source_block>>>(
        adjoint_ptr,
        grad_wavelet->data_ptr<float>(),
        p.forward_sources_loc.data_ptr<int>(),
        it,
        nsrc,
        ctx
    );
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

int recursive_checkpoint_scratch_depth(int interval_length)
{
    int depth = 0;
    while (interval_length > 1) {
        interval_length = (interval_length + 1) / 2;
        ++depth;
    }
    return depth;
}

void process_recursive_interval_3d(
    int start,
    int end,
    AcousticWavefieldTensor& start_state,
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
    const GradParam& grad_ctx_y,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    int adjoint_nsrc,
    CheckpointRuntime& checkpoint_runtime,
    std::vector<AcousticWavefieldTensor>& scratch_states,
    int scratch_depth,
    torch::Tensor& u_this_scratch,
    int B,
    int nx,
    int ny,
    int nz
)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        zero_tensor_device_async(u_this_scratch);
        float* u_this = u_this_scratch.data_ptr<float>();
        auto fwd_view = start_state.view();

        ACOUSTIC3D(
            order,
            wave_grid,
            wave_block,
            fwd_view,
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

        add_source_3d<<<forward_source_grid, forward_source_block>>>(
            fwd_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            start,
            forward_nsrc,
            ctx
        );

        start_state.swap();

        auto adj_view = adjoint.view();

        run_acoustic3d_adjoint_step(
            order, wave_grid, wave_block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
            cpml, ctx);

        add_source_3d<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

        accumulate_source_gradient_3d(
            forward_source_grid,
            forward_source_block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            start,
            ctx,
            forward_nsrc
        );

        accumulate_imaging_3d(
            wave_grid,
            wave_block,
            u_this,
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            rtm_out,
            B,
            nx,
            ny,
            nz,
            ctx.dt
        );
        return;
    }

    int mid = start + (end - start) / 2;

    TORCH_CHECK(
        scratch_depth < static_cast<int>(scratch_states.size()),
        "Recursive checkpoint scratch depth exhausted."
    );
    AcousticWavefieldTensor& mid_state = scratch_states[scratch_depth];
    checkpoint_runtime.copy_state(mid_state.state_tensors(), start_state.state_tensors());
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        checkpoint_runtime,
        scratch_states,
        scratch_depth + 1,
        u_this_scratch,
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
        grad_ctx_y,
        grad_ctx_z,
        cpml,
        ctx,
        forward_nsrc,
        adjoint_nsrc,
        checkpoint_runtime,
        scratch_states,
        scratch_depth + 1,
        u_this_scratch,
        B,
        nx,
        ny,
        nz
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

    // Buffer for pre-computed v^2 * lambda_now (proper-adjoint kernel input).
    // Python-side (PropTorch) manages the buffer via adjoint_workspace[0] so
    // it can be reused across backward calls.

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto forward_source_config = fdtd::Geom::make(forward_nsrc, B);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    float* grad_ptr = (grad != nullptr) ? grad->data_ptr<float>() : nullptr;

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        // Fold the vp-gradient imaging of step it+1 into the adjoint kernel: at
        // kernel entry u_now == post-source adjoint[it+1], the exact field
        // calculate_grad_3d would correlate.  Skip on the first reverse step
        // (it+1 == nt has no forward wavefield); step 0 is imaged after the loop.
        const float* img_fwd = (grad_ptr != nullptr && it + 1 < p.nt)
                             ? p.u_forward[it + 1].data_ptr<float>() : nullptr;

        run_acoustic3d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
            cpml, ctx,
            img_fwd, img_fwd ? grad_ptr : nullptr);

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

        accumulate_source_gradient_3d(
            forward_source_config.grid,
            forward_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            it,
            ctx,
            forward_nsrc
        );

        // Illumination still needs the per-step pass over the post-source
        // adjoint[it]; the vp gradient is folded above, so pass grad=nullptr here.
        if (rtm_out != nullptr) {
            accumulate_imaging_3d(
                launch_config.grid,
                launch_config.block,
                p.u_forward[it].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp,
                nullptr,
                rtm_out,
                B,
                nx,
                ny,
                nz,
                ctx.dt
            );
        }
    }

    // Trailing: image step 0, the one reverse step never folded into an adjoint
    // kernel.  adjoint.u_now_t now holds the post-source adjoint[0].
    if (grad_ptr != nullptr) {
        accumulate_imaging_3d(
            launch_config.grid,
            launch_config.block,
            p.u_forward[0].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            nullptr,
            B,
            nx,
            ny,
            nz,
            ctx.dt
        );
    }
}

BackwardOutput backward_full_imaging_impl(const BackwardInput& p)
{
    c10::cuda::CUDAGuard device_guard(p.models[0].device());
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_3d(illumination, p.models[0]);
    // Skip the per-timestep illumination pass for a plain FWI gradient that does
    // not request it; the vp gradient (calculate_grad) is unaffected.
    run_full_imaging(p, &grad, &grad_wavelet,
                     p.compute_illumination ? &illumination : nullptr);
    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    return out;
}

RTMOutput rtm_full_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_full_imaging(p, nullptr, nullptr, &out);
    return out;
}

void run_bs_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    torch::Tensor* grad_wavelet,
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
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }

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

    // Buffer for v^2 * lambda_now; Python-managed via adjoint_workspace[0].

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
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
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
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
        auto for_view = forward.view();

        run_acoustic3d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
            cpml, ctx);

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

        accumulate_source_gradient_3d(
            fwd_source_config.grid,
            fwd_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            it,
            ctx,
            forward_nsrc
        );

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

        boundary_runtime.restore_backward_3d(
            it,
            for_view.u_next,
            launch_config.grid,
            launch_config.block,
            bs,
            save_width,
            0,
            ctx
        );

        accumulate_imaging_utt_3d(
            launch_config.grid,
            launch_config.block,
            forward.u_prev_t.data_ptr<float>(),
            for_view.u_next,
            forward.u_now_t.data_ptr<float>(),
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

        add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        forward.swap();

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    if (p.nt > 0) {
        auto adj_view = adjoint.view();

        run_acoustic3d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
            cpml, ctx);

        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            0,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

        accumulate_source_gradient_3d(
            fwd_source_config.grid,
            fwd_source_config.block,
            adjoint.u_now_t.data_ptr<float>(),
            p,
            grad_wavelet,
            0,
            ctx,
            forward_nsrc
        );
    }
}

BackwardOutput backward_bs_imaging_impl(const BackwardInput& p)
{
    c10::cuda::CUDAGuard device_guard(p.models[0].device());
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_3d(illumination, p.models[0],
                       p.compute_adcig, 2 * p.adcig_max_lag + 1);
    // Gate illumination on compute_illumination (mirror the FULL path). When
    // off, accumulate_imaging_utt_3d skips the per-step RTM kernel because
    // rtm_out==nullptr; the buffer stays zero. Previously this BS path passed
    // &illumination unconditionally, so the RTM pass ran every step and the
    // flag was ignored (no time saved, illumination computed then discarded).
    // ADCIG (space-lag) rides the same imaging pass, so pass the RTMOutput when
    // either is requested.
    run_bs_imaging(p, &grad, &grad_wavelet,
                   (p.compute_illumination || p.compute_adcig) ? &illumination : nullptr);
    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

RTMOutput rtm_bs_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_bs_imaging(p, nullptr, nullptr, &out);
    return out;
}

void run_ckpt_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    torch::Tensor* grad_wavelet,
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
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }

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

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        8,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "acoustic3d"
    );

    auto chunk_forward = torch::zeros({p.checkpoint_interval, B, nz, ny, nx}, vp.options());

    // Buffer for v^2 * lambda_now; Python-managed via adjoint_workspace[0].

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

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors());

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

            run_acoustic3d_adjoint_step(
                order, launch_config.grid, launch_config.block,
                adj_view, vp.data_ptr<float>(),
                lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_y, grad_ctx_z,
                cpml, ctx);

            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

            accumulate_source_gradient_3d(
                fwd_source_config.grid,
                fwd_source_config.block,
                adjoint.u_now_t.data_ptr<float>(),
                p,
                grad_wavelet,
                it,
                ctx,
                forward_nsrc
            );

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
                nz,
                ctx.dt
            );
        }
    }
}

BackwardOutput backward_ckpt_imaging_impl(const BackwardInput& p)
{
    c10::cuda::CUDAGuard device_guard(p.models[0].device());
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_3d(illumination, p.models[0]);
    // Gate illumination on compute_illumination (see BS path above).
    run_ckpt_imaging(p, &grad, &grad_wavelet,
                     p.compute_illumination ? &illumination : nullptr);
    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    return out;
}

RTMOutput rtm_ckpt_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_ckpt_imaging(p, nullptr, nullptr, &out);
    return out;
}

void run_recursive_imaging(
    const BackwardInput& p,
    torch::Tensor* grad,
    torch::Tensor* grad_wavelet,
    RTMOutput* rtm_out
)
{
    auto vp = p.models[0];

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        8,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "acoustic3d"
    );

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
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

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

    int max_segment_length = 0;
    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];
        max_segment_length = std::max(max_segment_length, end - start);
    }

    std::vector<AcousticWavefieldTensor> scratch_states(recursive_checkpoint_scratch_depth(max_segment_length));
    for (auto& scratch_state : scratch_states)
        scratch_state.allocate(vp, 3, true);
    auto u_this_scratch = torch::empty_like(vp);
    // Scratch for the exact discrete-adjoint step (prepare/apply).

    AcousticWavefieldTensor start_state;
    start_state.allocate(vp, 3, true);

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(), start_state.next_tensors());

        process_recursive_interval_3d(
            start,
            end,
            start_state,
            adjoint,
            p,
            vp,
            grad,
            grad_wavelet,
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
            checkpoint_runtime,
            scratch_states,
            0,
            u_this_scratch,
            B,
            nx,
            ny,
            nz
        );
    }

}

BackwardOutput backward_recursive_imaging_impl(const BackwardInput& p)
{
    c10::cuda::CUDAGuard device_guard(p.models[0].device());
    BackwardOutput out;
    auto grad = torch::zeros_like(p.models[0]);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_3d(illumination, p.models[0]);
    // Gate illumination on compute_illumination (see BS path above).
    run_recursive_imaging(p, &grad, &grad_wavelet,
                          p.compute_illumination ? &illumination : nullptr);
    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    return out;
}

RTMOutput rtm_recursive_ckpt_impl(const BackwardInput& p)
{
    RTMOutput out;
    init_rtm_output_3d(out, p.models[0]);
    run_recursive_imaging(p, nullptr, nullptr, &out);
    return out;
}

} // namespace

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    return backward_ckpt_imaging_impl(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
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
