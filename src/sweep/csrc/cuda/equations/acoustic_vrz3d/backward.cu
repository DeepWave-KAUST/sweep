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

// Per-backward reusable scratch for the DD (per-step) VRZ backward.  A domain-
// decomposed backward drives ONE single-step backward_bs call per time step
// (ModelParallel._run_adjoint), and the time-invariant adjoint coefficients
// (inv_z, C0/Cx/Cy/Cz = vp², ∂b·κ) + the split-gradient scratch (c_*/e_*) were
// being reallocated + recomputed on EVERY step — ~30 s/iter of pure waste at
// production scale (nt≈9000).  These depend only on the model, which is fixed
// within a backward, so they are computed ONCE (first segment) and reused.
// Leaked singleton: never destroyed, so no torch-tensor teardown races with CUDA
// context shutdown at process exit.
struct VrzBwdScratch {
    torch::Tensor inv_z, C0, Cx, Cy, Cz, c_x, c_y, c_z, e_x, e_y, e_z;
    void* vp_ptr = nullptr;
    void* z_ptr = nullptr;
};
static VrzBwdScratch& vrz_bwd_scratch() {
    static VrzBwdScratch* s = new VrzBwdScratch();
    return *s;
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
    ctx.cut_mask = in.cut_face_mask;   // DD cut-aware: skip cut faces in bs reconstruction

    AcousticWavefieldTensor adjoint;
    if (!in.adjoint_wavefields.empty())
        adjoint.bind(in.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true, /*double_buffer_psi=*/true);
    zero_wavefield_state_vrz3d(adjoint);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto C0 = torch::zeros_like(vp);    // vp²       (time-invariant adjoint coeffs)
    auto Cx = torch::zeros_like(vp);    // ∂ₓb·κ
    auto Cy = torch::zeros_like(vp);    // ∂_yb·κ
    auto Cz = torch::zeros_like(vp);    // ∂_z b·κ
    auto c_x = torch::zeros_like(vp);   // split gradient scratch (order>=6 path)
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

    // Time-invariant adjoint transpose coefficients, computed once.
    BUILD_VRZ_ADJOINT_COEFFS_3D(
        order,
        launch_config.grid,
        launch_config.block,
        vp.data_ptr<float>(),
        z.data_ptr<float>(),
        inv_z.data_ptr<float>(),
        C0.data_ptr<float>(),
        Cx.data_ptr<float>(),
        Cy.data_ptr<float>(),
        Cz.data_ptr<float>(),
        grad_ctx,
        ctx
    );

    for (int it = in.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        ACOUSTIC_VRZ3D_ADJOINT_FUSED(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            C0.data_ptr<float>(),
            Cx.data_ptr<float>(),
            Cy.data_ptr<float>(),
            Cz.data_ptr<float>(),
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

        adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

        CALCULATE_GRAD_VRZ3D_AUTO(
            order,
            launch_config.grid,
            launch_config.block,
            in.u_forward.select(0, it).select(0, 0).data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            c_x.data_ptr<float>(),
            c_y.data_ptr<float>(),
            c_z.data_ptr<float>(),
            e_x.data_ptr<float>(),
            e_y.data_ptr<float>(),
            e_z.data_ptr<float>(),
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
    ctx.cut_mask = p.cut_face_mask;   // DD cut-aware: skip cut faces in boundary reconstruction

    // Stepped backward: process [bw_it_end, bw_begin()) in descending order so a
    // DD driver can halo-exchange the adjoint + reconstruction fields between
    // single reverse steps.  Defaults (bw_it_begin=-1 => nt, bw_it_end=0)
    // reproduce the monolithic call.  Phased (overlap) backward is not ported;
    // ModelParallel drives VRZ with the serial step-then-exchange loop.
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    const bool first_segment = (it_hi == static_cast<int>(p.nt));
    TORCH_CHECK(0 <= it_lo && it_lo < it_hi && it_hi <= static_cast<int>(p.nt),
                "AcousticVRZ3D stepped backward: require 0 <= bw_it_end < "
                "bw_it_begin <= nt, got [", it_lo, ", ", it_hi, ") with nt=", p.nt);
    // Phased DD backward (Fix A). The variable-density gradient is a spatial
    // divergence of the coupling field c/e = lambda*vp*grad(p), so unlike acoustic's
    // pointwise u_tt*lambda it needs the NEIGHBOUR's c/e at a cut seam. Split the
    // per-step backward into three driver-visible phases so ModelParallel can
    // halo-exchange lambda,p (after phase 1) AND c/e (after phase 2) before the
    // divergence (phase 3):
    //   phase 1 = adjoint advance + forward reconstruction (+restore, swaps)
    //   phase 2 = build c/e from the POST-exchange lambda,p (split kernel)
    //   phase 3 = divergence of the POST-exchange c/e -> gradient accumulate
    // step_phase 0 stays the monolithic single-GPU path (AUTO/fused; unchanged).
    TORCH_CHECK(p.step_phase >= 0 && p.step_phase <= 4,
                "AcousticVRZ3D backward step_phase must be 0, 1, 2, 3 or 4");
    const bool phased     = (p.step_phase != 0);
    const bool do_advance = (p.step_phase == 0 || p.step_phase == 1);
    const bool do_build   = (p.step_phase == 0 || p.step_phase == 2);
    const bool do_grad    = (p.step_phase == 0 || p.step_phase == 3);
    const bool do_coeff   = (p.step_phase == 0 || p.step_phase == 4);   // 4 = build adjoint coeffs only
    if (phased) {
        TORCH_CHECK(it_hi == it_lo + 1,
                    "AcousticVRZ3D phased backward requires a single-step segment "
                    "(bw_it_begin == bw_it_end + 1)");
        TORCH_CHECK(p.adjoint_workspace.size() == 10,
                    "AcousticVRZ3D phased/DD backward requires 10 adjoint_workspace "
                    "tensors ([0-5]=c_x..e_z coupling, [6-9]=C0,Cx,Cy,Cz adjoint coeffs); "
                    "bind them from Python (see ModelParallel._capture)");
    }
    if (p.bw_stepped()) {
        TORCH_CHECK(!p.adjoint_wavefields.empty() && !p.forward_wavefields.empty(),
                    "AcousticVRZ3D stepped backward requires Python-bound adjoint "
                    "and forward (reconstruction) wavefields");
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "AcousticVRZ3D stepped backward supports gpu-direct boundary "
                    "storage only");
    }

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true, /*double_buffer_psi=*/true);
    // FIRST segment only: Python zeroes the bound adjoint once before segment 1;
    // continuation segments must carry the propagated adjoint state.
    if (first_segment && do_advance)
        zero_wavefield_state_vrz3d(adjoint);

    AcousticWavefieldTensor forward;
    // Recon steps with the NOPML kernel — u triple buffer only; psi/zeta
    // would be dead weight (use_pml=false, 3 tensors, like vrz2d/acoustic).
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 3, false);
    else
        forward.allocate(vp, 3, false);

    // Seed the reverse reconstruction from the saved last two snapshots — FIRST
    // segment only; continuation segments carry the reconstruction state.
    if (first_segment && do_advance) {
        forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
        forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
        forward.u_next_t.zero_();
    }

    // Stepped/DD accumulate the gradient into Python-bound buffers across
    // segments (calculate_grad does +=; Python zeroes them once before segment
    // 1).  A monolithic call with no grads_out uses fresh per-call buffers.
    torch::Tensor grad_vp, grad_z;
    if (!p.grads_out.empty()) {
        TORCH_CHECK(p.grads_out.size() == p.models.size() + 1,
                    "AcousticVRZ3D grads_out must hold models.size()+1 tensors "
                    "(slot 0 = grad_wavelet, then one per model)");
        grad_vp = p.grads_out[1];
        grad_z = p.grads_out[2];
    } else {
        grad_vp = torch::zeros_like(vp);
        grad_z = torch::zeros_like(z);
    }
    // Adjoint coeffs (C0/Cx/Cy/Cz) + split-grad scratch (c_*/e_*): allocate ONCE
    // and, for a DD per-step backward, recompute/zero only on the FIRST segment
    // (they depend only on the fixed-within-a-backward model).  See VrzBwdScratch.
    // Bit-exact vs the old per-step alloc+zero: BUILD_VRZ_ADJOINT_COEFFS overwrites
    // C0..Cz, and build_vrz_grad_fields overwrites every INTERIOR c_*/e_* cell each
    // step while their halo stays at the once-zeroed 0 (the divergence reads a 0
    // halo either way).  ``grads_out`` (grad_vp/grad_z) stays Python-bound.
    auto& _sc = vrz_bwd_scratch();
    const bool _sc_realloc = !_sc.C0.defined() || _sc.C0.sizes() != vp.sizes()
                          || _sc.C0.device() != vp.device();
    const bool _sc_recompute = first_segment || _sc_realloc
                            || _sc.vp_ptr != vp.data_ptr() || _sc.z_ptr != z.data_ptr();
    if (_sc_realloc) {
        _sc.C0 = torch::empty_like(vp); _sc.Cx = torch::empty_like(vp);
        _sc.Cy = torch::empty_like(vp); _sc.Cz = torch::empty_like(vp);
        _sc.c_x = torch::zeros_like(vp); _sc.c_y = torch::zeros_like(vp);
        _sc.c_z = torch::zeros_like(vp); _sc.e_x = torch::zeros_like(vp);
        _sc.e_y = torch::zeros_like(vp); _sc.e_z = torch::zeros_like(vp);
    } else if (_sc_recompute) {
        _sc.c_x.zero_(); _sc.c_y.zero_(); _sc.c_z.zero_();
        _sc.e_x.zero_(); _sc.e_y.zero_(); _sc.e_z.zero_();
    }
    _sc.vp_ptr = vp.data_ptr(); _sc.z_ptr = z.data_ptr();
    // C0/Cx/Cy/Cz adjoint coeffs: DD (phased) uses the Python-bound adjoint_workspace
    // [6-9] so the driver halo-exchanges them ONCE (model-only, constant within a
    // backward) before the reverse loop; single-GPU reuses the persistent scratch.
    torch::Tensor C0, Cx, Cy, Cz;
    if (phased) {
        C0 = p.adjoint_workspace[6]; Cx = p.adjoint_workspace[7];
        Cy = p.adjoint_workspace[8]; Cz = p.adjoint_workspace[9];
    } else {
        C0 = _sc.C0; Cx = _sc.Cx; Cy = _sc.Cy; Cz = _sc.Cz;
    }
    // c/e coupling buffers: DD (phased) uses the Python-bound adjoint_workspace so
    // the driver can halo-exchange them between build (phase 2) and divergence
    // (phase 3); single-GPU (monolithic) reuses the persistent VrzBwdScratch.
    // (torch::Tensor copies share storage, so .data_ptr() hits the right buffer.)
    torch::Tensor c_x, c_y, c_z, e_x, e_y, e_z;
    if (phased) {
        c_x = p.adjoint_workspace[0]; c_y = p.adjoint_workspace[1]; c_z = p.adjoint_workspace[2];
        e_x = p.adjoint_workspace[3]; e_y = p.adjoint_workspace[4]; e_z = p.adjoint_workspace[5];
    } else {
        c_x = _sc.c_x; c_y = _sc.c_y; c_z = _sc.c_z;
        e_x = _sc.e_x; e_y = _sc.e_y; e_z = _sc.e_z;
    }

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    int save_width = p.M + 1;
    int boundary_offset = -p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    // Must match Python boundary_tangent_pad (= so//2 = M for VRZ) so the FP32
    // staging matches the persistent int8 buffers' per-step stride; see the
    // forward.cu note.  Restore reads top_t.stride(0) cells from staging.
    const int boundary_tangent_pad = p.M;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
            {}, p.use_pinned_memory, boundary_tangent_pad
        );
    } else {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, {}, p.use_pinned_memory, boundary_tangent_pad
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
    if (do_advance)
        boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    // Time-invariant adjoint transpose coefficients — computed ONCE per backward
    // (first DD segment / model change) into the reused scratch, not on every
    // single-step segment.
    // Adjoint coeffs: monolithic builds on the first segment; phased builds ONLY in the
    // pre-loop coeff phase (step_phase 4 -> do_coeff), after which the driver exchanges
    // their cut halo once and phases 1/2/3 reuse them (do_coeff false there).
    if (do_coeff && _sc_recompute) {
        BUILD_VRZ_ADJOINT_COEFFS_3D(
            order,
            launch_config.grid,
            launch_config.block,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            C0.data_ptr<float>(),
            Cx.data_ptr<float>(),
            Cy.data_ptr<float>(),
            Cz.data_ptr<float>(),
            grad_ctx,
            ctx
        );
    }

    // SWEEP_VRZ_GRAD_SPLIT=1 forces the O(M) split gradient (materialise c_d/e_d,
    // then a single-level divergence) even for order<=4, where AUTO otherwise
    // picks the fused O(M^2) nested-stencil kernel.  The fused/split crossover is
    // GPU-dependent (fused wins on RTX 6000 Ada; V100 prefers split — measured
    // ~12s/iter faster at production scale), so it stays a per-run toggle.
    const int _gradSplit = [](){ const char* e = std::getenv("SWEEP_VRZ_GRAD_SPLIT");
                                 return e ? std::atoi(e) : 0; }();

    // Main loop covers [max(bw_it_end, 1), bw_begin()) in descending order; step 0
    // contributes no gradient (matches the single-GPU VRZ backward, which also
    // stops at it == 1), so the last segment (bw_it_end == 0) simply ends there.
    for (int it = it_hi - 1; it >= std::max(it_lo, 1); --it) {
        if (do_advance) {
            auto adj_view = adjoint.view();

            ACOUSTIC_VRZ3D_ADJOINT_FUSED(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                C0.data_ptr<float>(),
                Cx.data_ptr<float>(),
                Cy.data_ptr<float>(),
                Cz.data_ptr<float>(),
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

            adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

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
        }

        // Gradient. step_phase 0 = monolithic single-GPU (AUTO/fused; unchanged).
        // Phased DD: build c/e (phase 2) and take their divergence (phase 3) as
        // SEPARATE launches so the driver halo-exchanges c/e in between -- the
        // divergence then reads the neighbour's c/e at the cut seam (not zero).
        // The split CALCULATE_GRAD_VRZ3D reads c_x..e_z with a raw (guard-free)
        // stencil, so a filled halo makes the seam correct; the fused AUTO kernel
        // must NOT be used here (its accessor zeroes cut-side taps).
        if (!phased) {
            if (_gradSplit && (order == 2 || order == 4)) {
                BUILD_VRZ_GRAD_FIELDS_3D(
                    order, launch_config.grid, launch_config.block,
                    forward.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
                    vp.data_ptr<float>(), z.data_ptr<float>(),
                    c_x.data_ptr<float>(), c_y.data_ptr<float>(), c_z.data_ptr<float>(),
                    e_x.data_ptr<float>(), e_y.data_ptr<float>(), e_z.data_ptr<float>(),
                    grad_ctx, ctx);
                CALCULATE_GRAD_VRZ3D(
                    order, launch_config.grid, launch_config.block,
                    forward.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
                    c_x.data_ptr<float>(), c_y.data_ptr<float>(), c_z.data_ptr<float>(),
                    e_x.data_ptr<float>(), e_y.data_ptr<float>(), e_z.data_ptr<float>(),
                    vp.data_ptr<float>(), z.data_ptr<float>(), inv_z.data_ptr<float>(),
                    grad_vp.data_ptr<float>(), grad_z.data_ptr<float>(),
                    grad_ctx, lap_ctx, ctx);
            } else {
                CALCULATE_GRAD_VRZ3D_AUTO(
                    order,
                    launch_config.grid,
                    launch_config.block,
                    forward.u_now_t.data_ptr<float>(),
                    adjoint.u_now_t.data_ptr<float>(),
                    vp.data_ptr<float>(),
                    z.data_ptr<float>(),
                    inv_z.data_ptr<float>(),
                    c_x.data_ptr<float>(),
                    c_y.data_ptr<float>(),
                    c_z.data_ptr<float>(),
                    e_x.data_ptr<float>(),
                    e_y.data_ptr<float>(),
                    e_z.data_ptr<float>(),
                    grad_vp.data_ptr<float>(),
                    grad_z.data_ptr<float>(),
                    grad_ctx,
                    lap_ctx,
                    ctx
                );
            }
        } else {
            if (do_build) {
                BUILD_VRZ_GRAD_FIELDS_3D(
                    order, launch_config.grid, launch_config.block,
                    forward.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
                    vp.data_ptr<float>(), z.data_ptr<float>(),
                    c_x.data_ptr<float>(), c_y.data_ptr<float>(), c_z.data_ptr<float>(),
                    e_x.data_ptr<float>(), e_y.data_ptr<float>(), e_z.data_ptr<float>(),
                    grad_ctx, ctx);
            }
            if (do_grad) {
                CALCULATE_GRAD_VRZ3D(
                    order, launch_config.grid, launch_config.block,
                    forward.u_now_t.data_ptr<float>(), adjoint.u_now_t.data_ptr<float>(),
                    c_x.data_ptr<float>(), c_y.data_ptr<float>(), c_z.data_ptr<float>(),
                    e_x.data_ptr<float>(), e_y.data_ptr<float>(), e_z.data_ptr<float>(),
                    vp.data_ptr<float>(), z.data_ptr<float>(), inv_z.data_ptr<float>(),
                    grad_vp.data_ptr<float>(), grad_z.data_ptr<float>(),
                    grad_ctx, lap_ctx, ctx);
            }
        }

        if (do_advance)
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
    ctx.cut_mask = p.cut_face_mask;   // DD cut-aware: skip cut faces in boundary reconstruction

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true, /*double_buffer_psi=*/true);
    zero_wavefield_state_vrz3d(adjoint);

    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 3, true);
    else
        forward.allocate(vp, 3, true);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_z = torch::zeros_like(z);
    auto C0 = torch::zeros_like(vp);    // vp²       (time-invariant adjoint coeffs)
    auto Cx = torch::zeros_like(vp);    // ∂ₓb·κ
    auto Cy = torch::zeros_like(vp);    // ∂_yb·κ
    auto Cz = torch::zeros_like(vp);    // ∂_z b·κ
    auto c_x = torch::zeros_like(vp);   // split gradient scratch (order>=6 path)
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

    // Time-invariant adjoint transpose coefficients, computed once for all segments.
    BUILD_VRZ_ADJOINT_COEFFS_3D(
        order,
        launch_config.grid,
        launch_config.block,
        vp.data_ptr<float>(),
        z.data_ptr<float>(),
        inv_z.data_ptr<float>(),
        C0.data_ptr<float>(),
        Cx.data_ptr<float>(),
        Cy.data_ptr<float>(),
        Cz.data_ptr<float>(),
        grad_ctx,
        ctx
    );

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

            ACOUSTIC_VRZ3D_ADJOINT_FUSED(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                C0.data_ptr<float>(),
                Cx.data_ptr<float>(),
                Cy.data_ptr<float>(),
                Cz.data_ptr<float>(),
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

            adjoint.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi

            CALCULATE_GRAD_VRZ3D_AUTO(
                order,
                launch_config.grid,
                launch_config.block,
                chunk_forward[it - start].data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                z.data_ptr<float>(),
                inv_z.data_ptr<float>(),
                c_x.data_ptr<float>(),
                c_y.data_ptr<float>(),
                c_z.data_ptr<float>(),
                e_x.data_ptr<float>(),
                e_y.data_ptr<float>(),
                e_z.data_ptr<float>(),
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
