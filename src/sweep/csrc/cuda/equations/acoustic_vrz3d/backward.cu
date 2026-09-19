#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "acoustic_vrz3d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundary/session.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
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

// The checkpoint snapshot set (u_prev, u_now, psix, psiy, psiz, zetax, zetay,
// zetaz) and the replay state backward_ckpt_impl steps, set 0 of
// p.forward_wavefields (Python-zeroed per backward call): those slots plus
// u_next -- the forward slot list without the psi double-buffer shadows -- in
// the struct's bind order u_prev, u_now, u_next, psix, psiz, zetax, zetaz,
// psiy, zetay.
constexpr int CKPT_NVAR = 8;
constexpr int REPLAY_STATE_NVAR = CKPT_NVAR + 1;   // 9

// p.checkpoint_replay (cuda_layout.checkpoint_replay_shapes): allocated by the
// propagator next to the checkpoint snapshots, never re-zeroed.
enum ReplaySlot : int {
    CHUNK_FORWARD = 0   // the replayed segment's pressure, (max_segment, B, 1, nz, ny, nx)
};

// Per-backward reusable scratch for the DD (per-step) VRZ backward.  A domain-
// decomposed backward drives ONE single-step backward_bs call per time step
// (ModelParallel._run_adjoint), and the time-invariant adjoint coefficients
// (inv_z, C0/Cx/Cy/Cz = vp², ∂b·κ) + the split-gradient scratch (c_*/e_*) were
// being reallocated + recomputed on EVERY step — ~30 s/iter of pure waste at
// production scale (nt≈9000).  These depend only on the model, which is fixed
// within a backward, so they are computed ONCE (first segment) and reused.
// Leaked singleton: never destroyed, so no torch-tensor teardown races with CUDA
// context shutdown at process exit.
// Layout of p.adjoint_workspace, declared on the Python side as
// AcousticVRZ3D.cuda_layout.backward_workspace_nvar and shared with the DD
// runner's binding (one padded grid per shot each): the six c/e coupling
// scratch grids of the split gradient, then the four time-invariant adjoint
// coefficients C0/Cx/Cy/Cz, recomputed on the first segment of every backward
// (under DD the driver halo-exchanges them once, which is why they are
// Python-bound there). The pool is zero at backward entry -- the propagator
// zeroes it before every gradient-bearing forward, the DD runner before
// segment 1 -- which is what keeps the c/e halo cells at zero: each step
// overwrites only their interior, and the divergence reads a zero halo.
enum WorkspaceSlot : int {
    C_X = 0, C_Y, C_Z, E_X, E_Y, E_Z,
    COEF_C0, COEF_CX, COEF_CY, COEF_CZ,
    N_SLOTS
};

// Exactly N_SLOTS.  The propagator allocates the pool for every
// gradient-bearing forward (_ensure_adjoint_workspace_buffers over
// AcousticVRZ3D.cuda_layout.backward_workspace_nvar = 10) and the DD runner
// rebinds the same ten grids, so an empty or differently sized pool means the
// Python declaration drifted.
static const std::vector<torch::Tensor>& workspace_slots(const BackwardInput& p)
{
    TORCH_CHECK(p.adjoint_workspace.size() == N_SLOTS,
                "acoustic_vrz3d/backward requires the propagator-bound adjoint_workspace "
                "(cuda_layout.backward_workspace_nvar): ",
                static_cast<int>(N_SLOTS), " tensors ([0-5]=c_x..e_z coupling, "
                "[6-9]=C0,Cx,Cy,Cz adjoint coeffs), got ", p.adjoint_workspace.size());
    return p.adjoint_workspace;
}

// p.grads_out as the propagator binds it for the acoustic family:
// {grad_wavelet, grad_vp, grad_z}. Slot 0 is the family's wavelet slot, which
// this equation never produces; slots 1 and 2 are accumulated here (Python
// zeroes them once per backward, or once per stepped/DD segment sequence).
// _c.py builds it for every backward (_gradient_buffers from
// cuda_layout.grads_out_has_wavelet = true plus one slot per model) and the DD
// runner rebinds the same list, so it is never empty.
static const std::vector<torch::Tensor>& grad_slots(const BackwardInput& p)
{
    TORCH_CHECK(p.grads_out.size() == p.models.size() + 1,
                "acoustic_vrz3d/backward requires the propagator-bound grads_out "
                "(cuda_layout.grads_out_has_wavelet + one slot per model = "
                "models.size()+1 tensors, slot 0 = grad_wavelet), got ", p.grads_out.size());
    return p.grads_out;
}

// The adjoint state: cuda_layout base_nvar 3 + pml_nvar 9 = 12 slots, bound by
// _c.py on every backward (_ensure_wavefield_buffers allocates them whenever
// the forward required a gradient) and rebound by the stepped / DD drivers.
static void bind_adjoint_state(AcousticWavefieldTensor& wf, const BackwardInput& p,
                               const char* mode)
{
    TORCH_CHECK(!p.adjoint_wavefields.empty(),
                "acoustic_vrz3d/", mode, " requires the propagator-bound "
                "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar = 12 tensors)");
    wf.bind(p.adjoint_wavefields, 3, true);
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
    auto inv_z = derived::reciprocal(in, z, "acoustic_vrz3d::backward_full_impl");

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
    ctx.set_cut_mask(in.cut_face_mask);   // DD cut-aware: skip cut faces in bs reconstruction

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, in, "backward");
    zero_wavefield_state_vrz3d(adjoint);

    const auto& gs = grad_slots(in);
    const auto& ws = workspace_slots(in);
    auto grad_vp = pool_required(gs, 1, vp, "grads_out");
    auto grad_z = pool_required(gs, 2, z, "grads_out");
    auto C0 = pool_required(ws, COEF_C0, vp, "adjoint_workspace");   // vp²       (time-invariant adjoint coeffs)
    auto Cx = pool_required(ws, COEF_CX, vp, "adjoint_workspace");   // ∂ₓb·κ
    auto Cy = pool_required(ws, COEF_CY, vp, "adjoint_workspace");   // ∂_yb·κ
    auto Cz = pool_required(ws, COEF_CZ, vp, "adjoint_workspace");   // ∂_z b·κ
    auto c_x = pool_required(ws, C_X, vp, "adjoint_workspace");      // split gradient scratch (order>=6 path)
    auto c_y = pool_required(ws, C_Y, vp, "adjoint_workspace");
    auto c_z = pool_required(ws, C_Z, vp, "adjoint_workspace");
    auto e_x = pool_required(ws, E_X, vp, "adjoint_workspace");
    auto e_y = pool_required(ws, E_Y, vp, "adjoint_workspace");
    auto e_z = pool_required(ws, E_Z, vp, "adjoint_workspace");
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

        add_source_3d_signed<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            in.adjoint_source.data_ptr<float>(),
            in.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            -1.0f,   // NEGATED residual: sign bit flipped in-kernel, no negated copy
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
    auto inv_z = derived::reciprocal(p, z, "acoustic_vrz3d::backward_bs_impl");

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
    ctx.set_cut_mask(p.cut_face_mask);   // DD cut-aware: skip cut faces in boundary reconstruction

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
        TORCH_CHECK(!p.boundary_on_disk,
                    "AcousticVRZ3D stepped backward supports gpu-direct or cpu "
                    "boundary storage only (boundary_on_disk unsupported in v1)");
        // The persistent runtime keeps a pointer to the saver, and this file's
        // EffectiveBoundarySaver is a per-CALL local (the skeletons' is a runner
        // member that outlives the segments). bind() re-points it before any use
        // and synchronize() never touches it, so cpu staging is safe -- but the
        // SYNCHRONOUS disk path calls saver_->load_disk_to_cpu_3d from
        // prefetch_backward_chunk, i.e. through that pointer between calls. That
        // is a second, independent reason disk stays refused here.
        TORCH_CHECK(!p.boundary_on_cpu || p.cut_face_mask != 0,
                    "AcousticVRZ3D stepped backward_bs cpu boundary staging "
                    "requires a DD cut mask (cut_face_mask != 0); single-tile "
                    "cpu staging is unsupported here (use gpu-direct or a "
                    "monolithic backward)");
    }

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward_bs");
    // FIRST segment only: Python zeroes the bound adjoint once before segment 1;
    // continuation segments must carry the propagated adjoint state.
    if (first_segment && do_advance)
        zero_wavefield_state_vrz3d(adjoint);

    AcousticWavefieldTensor forward;
    // Recon steps with the NOPML kernel — u triple buffer only; psi/zeta
    // would be dead weight (use_pml=false, 3 tensors, like vrz2d/acoustic).
    // _c.py hands these over on every boundary-saving backward
    // (_forward_state_buffers over cp.forward_state_shapes =
    // cuda_layout.reconstruction_nvar, slot_table.ACOUSTIC_VRZ3D.recon = 3) and
    // the DD runner rebinds the same list.
    wavefields_required(p.forward_wavefields, 3, vp,
                        "acoustic_vrz3d/backward_bs reconstruction "
                        "(cuda_layout.reconstruction_nvar)");
    forward.bind(p.forward_wavefields, 3, false);

    // Seed the reverse reconstruction from the saved last two snapshots — FIRST
    // segment only; continuation segments carry the reconstruction state.
    if (first_segment && do_advance) {
        forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
        forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
        forward.u_next_t.zero_();
    }

    // Stepped/DD accumulate the gradient into these across segments
    // (calculate_grad does +=; Python zeroes them once before segment 1).
    const auto& gs = grad_slots(p);
    const auto& ws = workspace_slots(p);
    auto grad_vp = pool_required(gs, 1, vp, "grads_out");
    auto grad_z = pool_required(gs, 2, z, "grads_out");
    // Adjoint coeffs (C0/Cx/Cy/Cz) + split-grad scratch (c_*/e_*) come from the
    // pool (WorkspaceSlot above): under DD (phased) the driver halo-exchanges
    // the c/e grids between the build (phase 2) and divergence (phase 3) steps,
    // so they are Python-bound there, and the monolithic path binds the same
    // ten slots. BUILD_VRZ_ADJOINT_COEFFS overwrites C0..Cz on the first
    // segment; build_vrz_grad_fields overwrites every INTERIOR c_*/e_* cell each
    // step while their halo stays at the pool's zero. torch::Tensor copies share
    // storage, so .data_ptr() hits the bound buffer.
    auto C0 = pool_required(ws, COEF_C0, vp, "adjoint_workspace");
    auto Cx = pool_required(ws, COEF_CX, vp, "adjoint_workspace");
    auto Cy = pool_required(ws, COEF_CY, vp, "adjoint_workspace");
    auto Cz = pool_required(ws, COEF_CZ, vp, "adjoint_workspace");
    auto c_x = pool_required(ws, C_X, vp, "adjoint_workspace");
    auto c_y = pool_required(ws, C_Y, vp, "adjoint_workspace");
    auto c_z = pool_required(ws, C_Z, vp, "adjoint_workspace");
    auto e_x = pool_required(ws, E_X, vp, "adjoint_workspace");
    auto e_y = pool_required(ws, E_Y, vp, "adjoint_workspace");
    auto e_z = pool_required(ws, E_Z, vp, "adjoint_workspace");

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
    // ``bs.last_two`` is never read in the backward -- the reverse seeds come
    // straight from ``p.u_last_two``. Passing {} made allocate_last_two take its
    // self-allocating branch and build a full two-wavefield FP32 buffer on EVERY
    // call, in HOST memory on the staged path. Harmless for a monolithic
    // backward, ruinous under DD/stepped, which enters once per time step: the
    // skeleton records a production 3-D run going 1760 -> 166 s/iteration once
    // the backward bound the tensor instead (dev 4290248 fixed the pre-template
    // acoustic3d/backward.cu the same way; this hand-written driver never got it).
    const torch::Tensor& last_two_bound = p.u_last_two;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
            last_two_bound, p.use_pinned_memory, boundary_tangent_pad, p.boundary_staging
        );
    } else {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, last_two_bound,
            p.use_pinned_memory, boundary_tangent_pad, p.boundary_staging
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

    // A Python-owned BoundarySession, when bound, keeps the copy stream and its
    // ring events alive ACROSS calls. Under DD every time step is a separate
    // entry into the extension, so a per-call stream is destroyed and rebuilt
    // nt times and no transfer can ever be in flight. With session == nullptr
    // -- which is every gpu-direct run, since ModelParallel only builds a
    // session when storage != 'gpu' -- BoundaryScope falls into its local branch
    // and reproduces the AsyncCopyContext + BoundaryRuntime pair this replaces.
    BoundaryScope boundary_scope(
        p.boundary_session ? p.boundary_session->impl() : nullptr,
        BoundarySessionImpl::Phase::Backward,
        boundary_saver,
        3,
        true,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files
    );
    BoundaryRuntime& boundary_runtime = boundary_scope.runtime();
    // it_hi, not nt: without it every stepped call primes the TAIL chunk instead
    // of its own, which at ring_buffers=1 stamps the tail slab over the slot the
    // current restore reads -- a wrong gradient, not a slow one.
    if (do_advance)
        boundary_runtime.prefetch_initial_backward_chunk((int)p.nt, it_hi);

    // Time-invariant adjoint transpose coefficients — computed ONCE per backward
    // (first DD segment / model change) into the reused scratch, not on every
    // single-step segment.
    // Adjoint coeffs: monolithic builds on the first segment; phased builds ONLY in the
    // pre-loop coeff phase (step_phase 4 -> do_coeff), after which the driver exchanges
    // their cut halo once and phases 1/2/3 reuse them (do_coeff false there).
    if (do_coeff && first_segment) {
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

            add_source_3d_signed<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                -1.0f,   // NEGATED residual: sign bit flipped in-kernel, no negated copy
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
    auto inv_z = derived::reciprocal(p, z, "acoustic_vrz3d::backward_ckpt_impl");

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
    ctx.set_cut_mask(p.cut_face_mask);   // DD cut-aware: skip cut faces in boundary reconstruction

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward_ckpt");
    zero_wavefield_state_vrz3d(adjoint);

    // The replay state (REPLAY_STATE_NVAR above), bound through the struct's
    // full PML bind: with no psi shadows in the set the replay writes psi in
    // place (swap() rotates u only) and the propagator zeroed the set per call.
    // _c.py hands it over on every checkpoint-mode backward
    // (_forward_state_buffers over cp.forward_state_shapes, derived from
    // slot_table.ACOUSTIC_VRZ3D's forward slots without the psi shadows), so
    // there is no unbound caller to allocate for.
    AcousticWavefieldTensor forward;
    TORCH_CHECK(static_cast<int>(p.forward_wavefields.size()) == REPLAY_STATE_NVAR,
                "acoustic_vrz3d/backward_ckpt requires the propagator-bound "
                "forward_wavefields (cuda_layout.slots, the forward slots without the "
                "psi double-buffer shadows): one replay state set of ",
                REPLAY_STATE_NVAR, " tensors, got ", p.forward_wavefields.size());
    forward.bind(wavefield_set(p.forward_wavefields, 0, REPLAY_STATE_NVAR,
                               "acoustic_vrz3d ckpt replay state"), 3, true);

    const auto& gs = grad_slots(p);
    const auto& ws = workspace_slots(p);
    auto grad_vp = pool_required(gs, 1, vp, "grads_out");
    auto grad_z = pool_required(gs, 2, z, "grads_out");
    auto C0 = pool_required(ws, COEF_C0, vp, "adjoint_workspace");   // vp²       (time-invariant adjoint coeffs)
    auto Cx = pool_required(ws, COEF_CX, vp, "adjoint_workspace");   // ∂ₓb·κ
    auto Cy = pool_required(ws, COEF_CY, vp, "adjoint_workspace");   // ∂_yb·κ
    auto Cz = pool_required(ws, COEF_CZ, vp, "adjoint_workspace");   // ∂_z b·κ
    auto c_x = pool_required(ws, C_X, vp, "adjoint_workspace");      // split gradient scratch (order>=6 path)
    auto c_y = pool_required(ws, C_Y, vp, "adjoint_workspace");
    auto c_z = pool_required(ws, C_Z, vp, "adjoint_workspace");
    auto e_x = pool_required(ws, E_X, vp, "adjoint_workspace");
    auto e_y = pool_required(ws, E_Y, vp, "adjoint_workspace");
    auto e_z = pool_required(ws, E_Z, vp, "adjoint_workspace");
    auto checkpoint_steps_cpu = p.checkpoint_steps.defined()
        ? p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous()
        : torch::empty({0}, torch::TensorOptions().dtype(torch::kInt32));
    const bool recursive_checkpoint = checkpoint_steps_cpu.numel() > 0;
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        CKPT_NVAR,
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

    // The replayed segment's pressure history from p.checkpoint_replay
    // (ReplaySlot above), taken once at the full max_segment rows and reused
    // by every segment: each row the reverse pass reads was written by the
    // replay earlier in the same segment, so it is never zeroed.  An unbound
    // caller allocates it once per call, as before.
    auto chunk_forward = pool_required(p.checkpoint_replay, CHUNK_FORWARD,
                                       {max_segment_length, N, C, nz, ny, nx}, vp.options(),
                                       "checkpoint_replay (acoustic_vrz3d/backward_ckpt, "
                                       "cuda_layout.checkpoint_replay_shapes)");

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

            add_source_3d_signed<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                -1.0f,   // NEGATED residual: sign bit flipped in-kernel, no negated copy
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
