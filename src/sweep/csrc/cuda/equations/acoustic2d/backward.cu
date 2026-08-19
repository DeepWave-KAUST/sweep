#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace acoustic2d {

namespace {

// Scratch buffers for the exact discrete-adjoint step (prepare/apply).  Six
// per-cell fields (g_lx, g_lz, g_gx, g_gz, g_qx, g_qz) packed into one tensor.
// Allocated with zeros so halo/air cells (never touched by PREPARE) carry the
// correct adjoint value 0; see acoustic2nd_adjoint_prepare.

// One exact-adjoint time step: PREPARE (local scratch) then APPLY (transposed
// stencils on scratch).  Replaces the old compute_v2_lambda + acoustic2nd_adjoint
// pair, which only transposed the interior and copied the forward in the PML
// band.  Works for interior + PML uniformly (PML coeffs are 0 in the interior).
static inline void run_acoustic2d_adjoint_step(
    int order, dim3 grid, dim3 block,
    AcousticWavefieldPointer adj_view,
    const float* vp_ptr,
    LaplaceParam lap_ctx, GradParam grad_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx,
    const float* grad_forward_img = nullptr, float* grad_out = nullptr)
{
    // FUSED single-kernel exact adjoint: recompute the per-cell g_* inline at
    // each stencil tap and write next-step psi/zeta into the wavefield's
    // double-buffer out-tensors (psixn/psizn/zetaxn/zetazn).  Read-old/write-new
    // => race-free even though aux is read at neighbours; no 6-field scratch
    // round-trip => ~forward bandwidth.  Caller must swap_aux() each step.
    TORCH_CHECK(adj_view.zetaxn != nullptr && adj_view.psixn != nullptr,
        "fused adjoint needs the adjoint wavefield bound with psi+zeta "
        "double-buffer (11 tensors in 2D); set cuda_layout.adjoint_extra_nvar=2.");
    (void)grad_ctx;
    ACOUSTIC2D_ADJOINT_FUSED(order, grid, block,
        adj_view, vp_ptr, lap_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx,
        adj_view.psixn, adj_view.psizn, adj_view.zetaxn, adj_view.zetazn,
        grad_forward_img, grad_out);
}

void init_rtm_output_2d(RTMOutput& out, const torch::Tensor& vp,
                        bool want_adcig = false, int nlag = 1)
{
    out.image = torch::zeros_like(vp);
    out.source_illumination = torch::zeros_like(vp);
    out.receiver_illumination = torch::zeros_like(vp);
    if (want_adcig) {
        // vp is (N, C, nz, nx); ADCIG cube is (nlag, N, C, nz, nx) on the
        // runtime-padded grid, cropped + summed over batch Python-side.
        out.adcig = torch::zeros({(long)nlag, vp.size(0), vp.size(1),
                                  vp.size(2), vp.size(3)}, vp.options());
    }
}

// Validate the stepped-backward segment fields (bw_it_begin/bw_it_end).
// ``need_recon`` is true for boundary-saving mode, where the 3-tensor
// reconstruction wavefield list must be Python-owned to survive segments.
void check_stepped_backward_2d(const BackwardInput& p, bool need_recon)
{
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    TORCH_CHECK(0 <= it_lo && it_lo < it_hi && it_hi <= static_cast<int>(p.nt),
                "stepped backward: require 0 <= bw_it_end < bw_it_begin <= nt, got [",
                it_lo, ", ", it_hi, ") with nt=", p.nt);
    TORCH_CHECK((p.cut_face_mask & ~0xF) == 0,
                "2D cut_face_mask uses bits 0..3 (x_lo, x_hi, z_lo, z_hi) only, got ",
                p.cut_face_mask);
    // Domain decomposition is boundary-saving only: the full-storage backward
    // keeps the whole forward history resident (no reconstruction halo to
    // exchange), so DD's memory win is moot there.  Use backward_bs for DD.
    TORCH_CHECK(need_recon || p.cut_face_mask == 0,
                "domain-decomposed backward (cut_face_mask) is boundary-saving "
                "only; the full-storage path does not support DD (use backward_bs)");
    if (need_recon && p.cut_face_mask != 0) {
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "domain-decomposed backward_bs (cut_face_mask) supports "
                    "gpu-direct boundary storage only "
                    "(boundary_on_cpu/boundary_on_disk unsupported in v1)");
    }
    if (!p.bw_stepped())
        return;
    TORCH_CHECK(p.adjoint_wavefields.size() == 11,
                "stepped backward requires the 11-tensor adjoint wavefield list "
                "(u triple + psi/zeta double-buffer) bound from Python");
    TORCH_CHECK(p.grads_out.size() == p.models.size() + 1,
                "stepped backward requires Python-bound grads_out "
                "(slot 0 = grad_wavelet, then one per model)");
    TORCH_CHECK(p.illum_out.size() == 2,
                "stepped backward requires Python-bound illum_out "
                "{source_illumination, receiver_illumination}");
    if (need_recon) {
        TORCH_CHECK(p.forward_wavefields.size() == 3,
                    "stepped backward_bs requires the 3-tensor reconstruction "
                    "wavefield list bound from Python");
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "stepped backward_bs supports gpu-direct boundary storage only "
                    "(boundary_on_cpu/boundary_on_disk unsupported in v1)");
    }
}

// Bind grad/illum accumulators from Python when provided (stepped), else
// fall back to internal zero allocation (legacy monolithic behaviour).
// Bound tensors are accumulated "+=" and NOT zeroed here — Python zeroes
// them once before the first segment.
void bind_backward_outputs_2d(
    const BackwardInput& p,
    torch::Tensor& grad_wavelet,
    torch::Tensor& grad,
    RTMOutput& illumination)
{
    if (!p.grads_out.empty()) {
        TORCH_CHECK(p.grads_out.size() == p.models.size() + 1,
                    "grads_out must hold models.size()+1 tensors "
                    "(slot 0 = grad_wavelet)");
        grad_wavelet = p.grads_out[0];
        grad = p.grads_out[1];
    } else {
        grad_wavelet = torch::zeros_like(p.forward_source);
        grad = torch::zeros_like(p.models[0]);
    }
    if (!p.illum_out.empty()) {
        TORCH_CHECK(p.illum_out.size() == 2,
                    "illum_out must be {source_illumination, receiver_illumination}");
        // image is never returned from backward (only RTM consumes it, and
        // RTM rejects stepping) — keep it internal per call.
        illumination.image = torch::zeros_like(p.models[0]);
        illumination.source_illumination = p.illum_out[0];
        illumination.receiver_illumination = p.illum_out[1];
        // NO ADCIG here -- see the 3-D twin. This is the segmented (stepped /
        // DD) path; source/receiver illumination survive across segments only
        // because Python owns and accumulates into those buffers, and ADCIG has
        // no such carrier. A per-segment cube would be filled for one step and
        // dropped, silently losing the user's ADCIG.
        TORCH_CHECK(!p.compute_adcig,
                    "compute_adcig is not supported on the segmented "
                    "(stepped / domain-decomposed) 2-D backward: the ADCIG cube "
                    "has no cross-segment accumulator. Run ADCIG on the "
                    "single-segment backward instead.");
    } else {
        init_rtm_output_2d(illumination, p.models[0],
                           p.compute_adcig, 2 * p.adcig_max_lag + 1);
    }
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
    int nz,
    float dt
)
{
    if (grad != nullptr) {
        calculate_grad<<<wave_grid, wave_block>>>(
            forward_ptr,
            adjoint_ptr,
            vp.data_ptr<float>(),
            grad->data_ptr<float>(),
            nx, nz, dt
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
    // NOTE: ADCIG is NOT accumulated here.  This helper's ``forward_ptr`` is the
    // full/checkpoint forward store, which for acoustic is vp^2*Lap(u) (kept for
    // the vp gradient), NOT the raw pressure the space-lag imaging condition
    // needs.  ADCIG is launched only from the boundary-saving reverse loop where
    // ``forward.u_now_t`` is the reconstructed raw pressure (see backward_bs).
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

    // Scratch for the exact discrete-adjoint step (prepare/apply).

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto forward_source_config = fdtd::Geom::make(forward_nsrc, B);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    ctx.cut_mask = 0;  // full-storage / RTM path is DD-free (DD is backward_bs only)
    acoustic_init_aux_slabs(ctx, adjoint);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    float* grad_ptr = (grad != nullptr) ? grad->data_ptr<float>() : nullptr;
    // Stepped/DD segments image the vp gradient with a per-step calculate_grad
    // pass (it composes with segmentation); the non-DD monolithic sweep keeps
    // dev's fused vp-grad imaging (lagged into the adjoint kernel + a trailing
    // step-0 pass), which is bit-identical to calculate_grad.  bw_begin()/
    // bw_it_end default to the full [0, nt).
    const bool fused_img = !p.bw_stepped() && p.cut_face_mask == 0;

    for (int it = p.bw_begin() - 1; it >= p.bw_it_end; --it) {

        auto adj_view = adjoint.view();

        // Fold the vp-gradient imaging of step it+1 into the adjoint kernel: at
        // kernel entry u_now == post-source adjoint[it+1], the exact field
        // calculate_grad would correlate.  Skip on the first reverse step
        // (it+1 == nt has no forward wavefield); step 0 is imaged after the loop.
        const float* img_fwd = (fused_img && grad_ptr != nullptr && it + 1 < p.nt)
                             ? p.u_forward[it + 1].data_ptr<float>() : nullptr;

        // Exact discrete adjoint of the CPML forward step (interior + PML),
        // optionally accumulating the lagged vp-gradient imaging.
        run_acoustic2d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx,
            img_fwd, img_fwd ? grad_ptr : nullptr);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

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

        // RTM illumination needs the per-step pass over the post-source
        // adjoint[it].  Fused monolithic path: the vp gradient is folded into
        // the adjoint kernel above, so pass grad=nullptr here (no double-count).
        // Stepped/DD path: image the vp gradient here per step (calculate_grad,
        // bit-identical to the fused term) so it composes with segmentation.
        torch::Tensor* step_grad = fused_img ? nullptr : grad;
        if (step_grad != nullptr || rtm_out != nullptr) {
            accumulate_imaging_2d(
                launch_config.grid,
                launch_config.block,
                p.u_forward[it].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp,
                step_grad,
                rtm_out,
                nx,
                nz,
                dt
            );
        }
    }

    // Trailing: image step 0, the one reverse step never folded into an adjoint
    // kernel (fused monolithic path only; the stepped path imaged it in-loop).
    if (fused_img && grad_ptr != nullptr) {
        accumulate_imaging_2d(
            launch_config.grid,
            launch_config.block,
            p.u_forward[0].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            nullptr,
            nx,
            nz,
            dt
        );
    }
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    check_stepped_backward_2d(in, /*need_recon=*/false);
    BackwardOutput out;
    torch::Tensor grad, grad_wavelet;
    RTMOutput illumination;
    // Bind Python-owned grad/illum accumulators when stepping, else allocate
    // zeros (monolithic).  Also allocates `illumination` (subsumes the old
    // init_rtm_output_2d, ADCIG cube included) and the grad/grad_wavelet
    // tensors run_full_imaging accumulates into.
    bind_backward_outputs_2d(in, grad_wavelet, grad, illumination);
    // Illumination (RTM image + source/receiver illumination) is a per-timestep
    // extra grid pass; skip it for a plain FWI gradient that does not request it.
    // The vp gradient (calculate_grad) is unaffected.  ADCIG (space-lag) rides
    // the same imaging pass, so pass the RTMOutput when either is requested.
    run_full_imaging(in, &grad, &grad_wavelet,
                     (in.compute_illumination || in.compute_adcig) ? &illumination : nullptr);
    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

RTMOutput rtm(const BackwardInput& in)
{
    TORCH_CHECK(!in.bw_stepped(), "stepped RTM not supported in v1");
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
    init_rtm_output_2d(out, in.models[0],
                       in.compute_adcig, 2 * in.adcig_max_lag + 1);
    run_full_imaging(in, nullptr, nullptr, &out);
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

    const auto& p = in;
    BackwardOutput out;

    check_stepped_backward_2d(p, /*need_recon=*/true);
    // Segment bounds: process [it_lo, it_hi) in descending order.  Defaults
    // (bw_it_begin = -1 => nt, bw_it_end = 0) reproduce the monolithic call.
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    const bool first_segment = (it_hi == static_cast<int>(p.nt));

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
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    // DD: skip cut faces in the strip restore, the seed rim-zeroing, the
    // NOPML exclusion band and the fused-adjoint pure_interior test.
    ctx.cut_mask = p.cut_face_mask;

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
    acoustic_init_aux_slabs(ctx, adjoint);
    // Seed the reverse reconstruction from the saved last two snapshots —
    // FIRST segment only; re-running this mid-stream would clobber the
    // carried reconstruction state.
    if (first_segment) {
        forward.u_prev_t.copy_(p.u_last_two.select(1,1).squeeze(0));
        forward.u_now_t.copy_(p.u_last_two.select(1,0).squeeze(0));
    }

    torch::Tensor grad, grad_wavelet;
    RTMOutput illumination;
    bind_backward_outputs_2d(p, grad_wavelet, grad, illumination);

    // Scratch for the exact discrete-adjoint step (prepare/apply).

    // For checking wavefields
    // torch::Tensor u_allt = torch::zeros({nt, B, 1, nz, nx}, vp.options());

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    // Boundary wavefields (for saving all wavefields)
    int save_width = p.abcn > 0 ? M + 1 : M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 1, ctx, vp, save_width, 2, true, true, 1, {}, p.boundary_gpu, {}, p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    // FIRST segment only: re-running the rim-zeroing mid-stream would clobber
    // the carried reconstruction state (DD/stepped runs re-enter here).
    if (first_segment) {
        auto for_view = forward.view();
        set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(
            for_view.u_prev, ctx.abcn+ctx.M, nx, nz,
            ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2), ctx.cut_mask);
        set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(
            for_view.u_now, ctx.abcn+ctx.M, nx, nz,
            ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2), ctx.cut_mask);
    }


    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
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

    // Main loop covers [max(it_lo,1), it_hi); the it==0 adjoint-only tail
    // below runs only in the segment whose bw_it_end == 0.
    for (int it = it_hi - 1; it >= std::max(it_lo, 1); --it) {
        auto adj_view = adjoint.view();
        auto for_view = forward.view();

        // u_allt[it].copy_(forward.u_now_t);

        // adjoint modeling (exact discrete adjoint of the CPML forward step)
        run_acoustic2d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

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

        boundary_runtime.restore_backward_2d(
            it,
            for_view.u_next,
            launch_config.grid,
            launch_config.block,
            bs,
            save_width,
            0,
            ctx
        );
        
        calculate_grad_utt<<<launch_config.grid, launch_config.block>>>(
            forward.u_prev_t.data_ptr<float>(),
            for_view.u_next,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            nx, nz, dt
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

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);

        // Gate illumination on compute_illumination (mirror FULL path). When off,
        // skip the per-step RTM pass entirely; the FWI vp-gradient is produced by
        // calculate_grad_utt above and is unaffected. Previously this BS path ran
        // it every step regardless, so the flag was ignored (no time saved).
        if (in.compute_illumination) {
            accumulate_rtm_image_2d<<<launch_config.grid, launch_config.block>>>(
                forward.u_now_t.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                illumination.image.data_ptr<float>(),
                illumination.source_illumination.data_ptr<float>(),
                illumination.receiver_illumination.data_ptr<float>(),
                nx, nz
            );
        }
        // Space-lag ADCIG rides the same co-resident (u_s(t), u_r(t)) pair.
        if (illumination.adcig.defined() && illumination.adcig.numel() > 0) {
            int nlag = illumination.adcig.size(0);
            int Bloc = illumination.adcig.size(1) * illumination.adcig.size(2);
            int max_lag = (nlag - 1) / 2;
            accumulate_adcig_2d<<<launch_config.grid, launch_config.block>>>(
                forward.u_now_t.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                illumination.adcig.data_ptr<float>(),
                nlag, max_lag, Bloc, nx, nz
            );
        }

    }

    if (it_lo == 0 && p.nt > 0) {
        auto adj_view = adjoint.view();

        run_acoustic2d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            0,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

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
    out.adcig = illumination.adcig;
    return out;

}

namespace {

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

int recursive_checkpoint_scratch_depth(int interval_length)
{
    int depth = 0;
    while (interval_length > 1) {
        interval_length = (interval_length + 1) / 2;
        ++depth;
    }
    return depth;
}

void process_recursive_interval_2d(
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
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    int adjoint_nsrc,
    CheckpointRuntime& checkpoint_runtime,
    std::vector<AcousticWavefieldTensor>& scratch_states,
    int scratch_depth,
    torch::Tensor& u_this_scratch,
    int nx,
    int nz
)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        zero_tensor_device_async(u_this_scratch);
        float* u_this = u_this_scratch.data_ptr<float>();
        auto fwd_view = start_state.view();

        ACOUSTIC2D(
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

        start_state.swap();

        auto adj_view = adjoint.view();

        run_acoustic2d_adjoint_step(
            order, wave_grid, wave_block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx);

        add_source<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

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
            u_this,
            adjoint.u_now_t.data_ptr<float>(),
            vp,
            grad,
            rtm_out,
            nx,
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
        checkpoint_runtime,
        scratch_states,
        scratch_depth + 1,
        u_this_scratch,
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
        checkpoint_runtime,
        scratch_states,
        scratch_depth + 1,
        u_this_scratch,
        nx,
        nz
    );
}

} // namespace

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    TORCH_CHECK(!in.bw_stepped(),
                "checkpoint backward does not support bw_it_begin/bw_it_end in v1");

    const auto& p = in;
    BackwardOutput out;

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "acoustic2d"
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
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 2, true);
    else
        // Aux shapes must follow the Python-allocated checkpoint slots
        // (possibly per-axis slabs); a plain allocate() would build
        // full-domain aux and break the snapshot copies.
        forward.allocate_from_snapshots(vp, p.checkpoints, 2);
    // Slab geometry follows the FORWARD-state aux layout (the recompute runs
    // the forward kernel); the adjoint aux stays full-domain and its legacy
    // kernels never consult the slab geometry.
    acoustic_init_aux_slabs(ctx, forward);

    auto grad = torch::zeros_like(vp);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_2d(illumination, vp,
                       in.compute_adcig, 2 * in.adcig_max_lag + 1);

    // Scratch for the exact discrete-adjoint step (prepare/apply).

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

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors());

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

            run_acoustic2d_adjoint_step(
                order, launch_config.grid, launch_config.block,
                adj_view, vp.data_ptr<float>(),
                lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
                cpml, ctx);

            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

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
                // Gate illumination on compute_illumination (mirror FULL path);
                // accumulate_imaging_2d skips the RTM kernel when rtm_out==nullptr.
                (in.compute_illumination || in.compute_adcig) ? &illumination : nullptr,
                nx,
                nz,
                dt
            );
        }
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    TORCH_CHECK(!in.bw_stepped(),
                "checkpoint backward does not support bw_it_begin/bw_it_end in v1");
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(
        p.checkpoints.size() == 6,
        "Acoustic 2D recursive checkpointing expects 6 checkpoint tensors"
    );

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "acoustic2d"
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
    if (p.has_topo) { ctx.topo_rows = p.topo_rows.data_ptr<int>(); ctx.has_topo = true; }
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 2, true);
    else
        adjoint.allocate(vp, 2, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    auto grad = torch::zeros_like(vp);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output_2d(illumination, vp,
                       in.compute_adcig, 2 * in.adcig_max_lag + 1);

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

    int max_segment_length = 0;
    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];
        max_segment_length = std::max(max_segment_length, end - start);
    }

    AcousticWavefieldTensor start_state;
    start_state.allocate_from_snapshots(vp, p.checkpoints, 2);
    // Slab geometry follows the forward-state layout carried by the psi
    // slots of the Python-allocated checkpoint buffers (adjoint aux stays
    // full-domain; its legacy kernels never consult the slab geometry).
    acoustic_init_aux_slabs(ctx, start_state);

    std::vector<AcousticWavefieldTensor> scratch_states(recursive_checkpoint_scratch_depth(max_segment_length));
    for (auto& scratch_state : scratch_states)
        scratch_state.allocate_like(vp, start_state);
    auto u_this_scratch = torch::empty_like(vp);
    // Scratch for the exact discrete-adjoint step (prepare/apply).

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(), start_state.next_tensors());

        process_recursive_interval_2d(
            start,
            end,
            start_state,
            adjoint,
            p,
            vp,
            &grad,
            &grad_wavelet,
            // Gate illumination on compute_illumination (see BS path above).
            (in.compute_illumination || in.compute_adcig) ? &illumination : nullptr,
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
            checkpoint_runtime,
            scratch_states,
            0,
            u_this_scratch,
            nx,
            nz
        );
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

}
