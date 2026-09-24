#include <torch/extension.h>
#include <algorithm>

#include "visco_acoustic2d.h"
#include "kernels.cuh"
#include "../acoustic2d/kernels.cuh"   // reused fused adjoint + grad kernels (ODR-safe)
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../common/adapt_inputs.h"   // *InputCore, InputArena, to_torch

namespace visco_acoustic2d {

namespace {

using derived::ViscoMode;

// ---------------------------------------------------------------------------
// Allocation contract (the forward's twin, forward.cu): the propagator's pools
// are REQUIRED -- adjoint_wavefields, forward_wavefields (the checkpoint replay
// state sets), checkpoint_replay (the chunk history), adjoint_workspace (the
// imaging carrier + the spectral scratch), derived_models (the tables),
// grads_out -- and with them no backward mode allocates anything on the device:
// the spectral terms run on the workspace slots through the cached cuFFT plan
// (kernels.cuh ViscoScratch / ViscoFFT).  illum_out is the ONE optional
// binding: the propagator hands it over only when the caller asked for
// illumination (compute_illumination), so bind_illumination keeps its
// allocating branch.
//
// Slot layouts, each through ONE helper (kernels.cuh / derived_models.h) and
// count-checked against the bound pool at entry:
//   adjoint_workspace  visco_slots(a, d, mode):
//       [CARRIER] + (spectral ? [C0, C1, (C2 if d and mode in ckpt/recursive),
//                                (R1 if d), (UPREV if a and mode == ckpt), FFT_WS]
//                             : [])
//   derived_models     derived::visco_tables(a, d, mode):
//       full:             [Gp if a, Gd1 if d, Gd2 if d]
//       ckpt / recursive: [Gp if a, dt2A if a, Gd1 if d, Gd2 if d]
// (a = amplitude damping = D_loss in eq_aux, d = phase dispersion = D_k2 +
// D_frac in eq_aux; ViscoAcoustic.cuda_layout counts the same lists.)
// CARRIER is the vp^2*Lap(u) carrier of the reverse step (one padded grid per
// shot): its kernel writes non-halo cells only and the reused grad/RTM kernels
// need a zero halo, which the pool's zero-at-entry provides.
// ---------------------------------------------------------------------------

// p.grads_out as the propagator binds it, in BackwardOutputCore.grads order,
// zeroed per backward on the Python side and accumulated here.  _c.py builds it
// unconditionally for every backward (_gradient_buffers from
// cuda_layout.grads_out_has_wavelet = true plus one slot per prepared model).
enum GradSlot : int { GRAD_WAVELET = 0, GRAD_VP, GRAD_B1, GRAD_B2, GRAD_A, N_GRADS };

struct ViscoGrads {
    Buf wavelet, vp, B1, B2, A;
};

ViscoGrads bind_grads(const BackwardInputCore& p)
{
    const auto& gs = p.grads_out;
    SWEEP_CHECK(gs.size() == N_GRADS,
                "visco_acoustic2d/backward requires the propagator-bound grads_out "
                "(cuda_layout.grads_out_has_wavelet + one slot per model = ",
                static_cast<int>(N_GRADS), " tensors: "
                "grad_wavelet, grad, grad_B1, grad_B2, grad_A), got ", gs.size());
    ViscoGrads g;
    g.wavelet = pool_required(gs, GRAD_WAVELET, p.forward_source, "grads_out");
    g.vp = pool_required(gs, GRAD_VP, p.models[0], "grads_out");
    g.B1 = pool_required(gs, GRAD_B1, p.models[1], "grads_out");
    g.B2 = pool_required(gs, GRAD_B2, p.models[2], "grads_out");
    g.A = pool_required(gs, GRAD_A, p.models[3], "grads_out");
    return g;
}

// The illumination accumulators: p.illum_out when the propagator bound them
// ({source, receiver}, zeroed per backward; cuda_layout.illum_nvar = 2), else
// allocated here only when the caller asked for illumination -- otherwise
// left undefined, which the output packing hands Python as None and _c.py's
// ``isinstance(..., torch.Tensor)`` guards skip (the acoustic family's
// eq_driver.cuh acoustic_bind_backward_outputs / init_rtm_output do the
// same).  ``always``: the rtm() entry, whose output IS the illumination.
RTMOutputCore bind_illumination(const BackwardInputCore& p, bool always)
{
    RTMOutputCore illumination;
    if (!p.illum_out.empty()) {
        SWEEP_CHECK(p.illum_out.size() == 2,
                    "illum_out must be {source_illumination, receiver_illumination}");
        illumination.source_illumination =
            pool_slot_checked(p.illum_out, 0, p.models[0], "illum_out");
        illumination.receiver_illumination =
            pool_slot_checked(p.illum_out, 1, p.models[0], "illum_out");
    } else {
        SWEEP_CHECK(!(always || p.compute_illumination),
                    "visco_acoustic2d: illumination requires the propagator-bound illum_out "
                    "{source, receiver} (cuda_layout.illum_nvar); nothing allocates it here");
    }
    return illumination;
}

void pack_outputs(BackwardOutputCore& out, const ViscoGrads& g, const RTMOutputCore& illumination)
{
    out.grads = {g.wavelet, g.vp, g.B1, g.B2, g.A};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
}

// Checkpoint replay state: p.forward_wavefields holds K sets of the 7 tensors
// (u_prev, u_now, u_next model-shaped; psix, psiz, zetax, zetaz slab-shaped
// like the checkpoint slots -- cuda_layout.checkpoint_state_nvar = 7 with
// checkpoint_slot_axes), zeroed per backward call: K = 1 (chunk mode) or
// 1 + the bisection depth (recursive mode, cuda_layout.recursive_state_depth).
// _c.py hands the sets over on every checkpoint-mode backward
// (_forward_state_buffers over cp.forward_state_shapes), and visco refuses both
// boundary saving and DD, so the checkpoint modes are the only callers -- the
// binding is required.
constexpr int CKPT_STATE_COUNT = 7;

void check_replay_state_sets(const BackwardInputCore& p, int sets, const char* what)
{
    SWEEP_CHECK(static_cast<int>(p.forward_wavefields.size()) == sets * CKPT_STATE_COUNT,
                what, " requires the propagator-bound forward_wavefields "
                "(cuda_layout.checkpoint_state_nvar, recursive_state_depth): ", sets,
                " replay state set(s) of ", CKPT_STATE_COUNT, " tensors, got ",
                p.forward_wavefields.size());
}

// The adjoint state: cuda_layout base_nvar 3 + pml_nvar 6 + adjoint_extra_nvar 2
// = 11 slots, bound by _c.py on every backward (_ensure_wavefield_buffers
// allocates them whenever the forward required a gradient).
void bind_adjoint_state(AcousticWavefieldTensor& wf, const BackwardInputCore& p, const char* mode)
{
    SWEEP_CHECK(!p.adjoint_wavefields.empty(),
                "visco_acoustic2d/", mode, " requires the propagator-bound "
                "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar + "
                "adjoint_extra_nvar tensors)");
    wf.bind(p.adjoint_wavefields, 2, true);
}

void bind_replay_state(AcousticWavefieldTensor& wf, const BackwardInputCore& p,
                       const Buf& vp, int set)
{
    wf.bind_replay_state(wavefield_set(p.forward_wavefields, set, CKPT_STATE_COUNT,
                                       "visco_acoustic2d ckpt replay state"),
                         vp, p.checkpoints, 2);
}

// ---------------------------------------------------------------------------
// Spectral terms (Zhu & Harris 2014, decoupled) — adjoint machinery.
// The ViscoSpectral bundle (kernels.cuh) carries the damping filter and the
// fractional-Laplacian dispersion remainder; the ViscoScratch bundle the
// workspace slots the pipeline runs on.
//
// Forward, step j (buffers):  u_next -= Gp ⊙ L(u_now - u_prev),  Gp = dt*A,
// followed by the halo zeroing Z.  With λ_j := adjoint of the fully-updated
// u_next of step j, the transpose across the buffer rotation contributes to
// the adjoint recursion at reverse iteration it (u_now = λ_{it+1},
// u_prev = λ_{it+2} BEFORE the fused kernel's swap):
//     λ_it += Z( L(Gp ⊙ (λ_{it+2} - λ_{it+1})) )
// L is self-adjoint (real, even |k| multiplier); Z^T = Z gates every
// accumulation into λ, preserving the "λ halo == 0" invariant the reused
// stencil kernels rely on.
//
// Bit-exactness of the slot forms below, against the torch expressions they
// replaced (kernels.cuh records the transform / promote / product argument):
//   * a product ``coef * X`` is at::mul_out(P, coef, X) into a float alias of
//     a dead complex slot -- the structured mul kernel the functional product
//     ran, operands in the same order, one IEEE multiply per cell;
//   * ``X.mul_(coef)`` is used only where the original was ``coef * X`` on a
//     staged X: the in-place mul computes X * coef, and IEEE multiplication
//     commutes exactly (the correctly rounded exact product is symmetric, the
//     sign of a zero is the XOR of the signs);
//   * a difference ``a - b`` is at::sub_out(D, a, b) / ``D.sub_(b)`` on D == a:
//     the same structured sub kernel (sub_out -> add_stub with alpha = -1) on
//     the same operand values in the same order;
//   * a sum ``m + e`` is ``R.add_(m)`` on R == e: IEEE addition commutes
//     exactly, and the summands are Lops of adjoint fields nothing writes in
//     between, so forming them in another order changes no bits;
//   * every accumulation into a gradient keeps today's add_(P, alpha) with the
//     same double alpha.  No addcmul anywhere (one fused lambda = FMA).
// ---------------------------------------------------------------------------

// ``u_a - u_b`` staged in the float alias of C1 (free between the adjoint
// step, whose terms are already added into u_next, and the gradient
// products), shaped like the adjoint field so the Lop pipeline takes it as
// is.  Was ``u_a - u_b`` (functional); the row views only relabel contiguous
// (B, nz, nx) rows as (B, 1, nz, nx).
Buf stage_difference(ViscoScratch& ws, const float* u_a, const float* u_b)
{
    const Buf D = visco_acoustic2d_real_alias(ws.C1);
    visco_ops::binary<visco_ops::SubOp>(D.data_ptr<float>(), u_a, 1, u_b, 1, D.numel());
    return D;
}

// The spectral terms of the adjoint recursion; call between the fused adjoint
// kernel (which wrote u_next = λ_it^{S^T}) and swap_aux().  Damping reads the
// lag pair (λ_{it+2} - λ_{it+1}); the dispersion term is memoryless in u_now,
// so its transpose reads λ_{it+1} alone (adj.u_now_t) through the SAME
// self-adjoint operators with the coefficient maps moved inside.
//
// Was:  m = Lop(Gp * (u_prev - u_now), kmul)                       [damping]
//       e = Lop(Gd1 * u_now, Dk2) - Lop(Gd2 * u_now, Dfrac)         [dispersion]
//       m = active ? m + e : e;   zero_halo(m);   u_next += m
// Now, on the slots: each product is staged in the float alias of C1 (the
// spectrum C1 held is dead at that point) and consumed by the Lop pipeline's
// promote before C1 is overwritten; the dispersion difference is
// R1.copy_(real(L_Dk2)) (a copy, no arithmetic) then R1.sub_(real(L_Dfrac));
// with both terms on, the damping Lop is formed AFTER the dispersion sum and
// added into it, R1.add_(real(L_damp)) = e + m = m + e.  The result (R1, or
// real(C1) with the damping alone) is halo-zeroed and added into u_next as
// before.
void adjoint_damping_extra(AcousticWavefieldTensor& adj, const ViscoSpectral& d,
                           ViscoScratch& ws, int M)
{
    if (!(d.active || d.disp)) return;
    const float* u_now = adj.u_now_t.template data_ptr<float>();
    const float* u_prev = adj.u_prev_t.template data_ptr<float>();
    float* u_next = adj.u_next_t.template data_ptr<float>();
    const int64_t n = ws.C1.numel() / 2;
    // m: the term to add, as (pointer, stride) -- R1 (contiguous), or the real
    // part of C1 (stride 2) with the damping alone.
    const float* m = nullptr;
    int m_stride = 1;
    if (d.disp) {
        const Buf X = visco_acoustic2d_real_alias(ws.C1);
        visco_ops::binary<visco_ops::MulOp>(X.data_ptr<float>(), d.Gd1.data_ptr<float>(), 1, u_now, 1, n);   // Gd1 * u_now
        visco_acoustic2d_lop_into(X.data_ptr<float>(), d.Dk2.data_ptr<float>(), ws);                          // C1.re = L_Dk2
        visco_ops::real_copy(ws.R1.data_ptr<float>(), visco_acoustic2d_complex_ptr(ws.C1), n);                          // R1 = L_Dk2
        visco_ops::binary<visco_ops::MulOp>(X.data_ptr<float>(), d.Gd2.data_ptr<float>(), 1, u_now, 1, n);   // Gd2 * u_now (L_Dk2 is in R1)
        visco_acoustic2d_lop_into(X.data_ptr<float>(), d.Dfrac.data_ptr<float>(), ws);                        // C1.re = L_Dfrac
        visco_ops::inplace<visco_ops::SubOp>(ws.R1.data_ptr<float>(), visco_acoustic2d_real_ptr(ws.C1), VISCO_REAL_STRIDE, n);   // R1 = L_Dk2 - L_Dfrac = e
        m = ws.R1.data_ptr<float>(); m_stride = 1;
    }
    if (d.active) {
        const Buf X = visco_acoustic2d_real_alias(ws.C1);
        visco_ops::binary<visco_ops::SubOp>(X.data_ptr<float>(), u_prev, 1, u_now, 1, n);                    // u_prev - u_now
        visco_ops::inplace<visco_ops::MulOp>(X.data_ptr<float>(), d.Gp.data_ptr<float>(), 1, n);             // == Gp * (u_prev - u_now)
        visco_acoustic2d_lop_into(X.data_ptr<float>(), d.kmul.data_ptr<float>(), ws);                         // C1.re = L
        if (d.disp)
            visco_ops::inplace<visco_ops::AddOp>(ws.R1.data_ptr<float>(), visco_acoustic2d_real_ptr(ws.C1), VISCO_REAL_STRIDE, n);   // e + m == m + e
        else {
            m = visco_acoustic2d_real_ptr(ws.C1); m_stride = VISCO_REAL_STRIDE;
        }
    }
    // zero_halo(m) then u_next += m.  With the damping alone m is the strided
    // real part of C1: the halo zeroing was on that view, so the add reads the
    // zeroed cells -- here the halo cells of the ADD are masked instead, which
    // adds 0 to exactly the cells the zeroed view added 0 to.
    if (m_stride == 1) {
        visco_acoustic2d_zero_halo(ws.R1, M);
        visco_ops::inplace<visco_ops::AddOp>(u_next, m, 1, n);
    } else {
        const Buf Lr = visco_acoustic2d_real_alias(ws.C1);   // C1's real parts, gathered contiguous
        visco_ops::real_copy(Lr.data_ptr<float>(), visco_acoustic2d_complex_ptr(ws.C1), n);
        visco_acoustic2d_zero_halo(Lr, M);
        visco_ops::inplace<visco_ops::AddOp>(u_next, Lr.data_ptr<float>(), 1, n);
    }
}

// grad_A += -dt^2 * λ_it ⊙ L((u[it] - u[it-1]) / dt)
//         = -dt   * λ_it ⊙ L(du),  du = u[it] - u[it-1].
// ``lam`` is the post-swap adjoint (λ_it, halo == 0 so the halo band of the
// filtered field drops out automatically); ``du`` any float view of
// lam.sizes() outside C0's storage (stage_difference puts it in C1's alias).
// Was grad_A.add_(lam * Lop(du), -dt): the Lop on the pipeline, the product
// through at::mul_out(P, lam, real(C1)) into the float alias of C0 (dead
// after the inverse transform), lam first as before, then the same add_.
void accumulate_grad_A(float* grad_A,
                       const float* lam,
                       const float* du,
                       const ViscoSpectral& d, float dt, ViscoScratch& ws)
{
    if (!d.active) return;
    const int64_t n = ws.C1.numel() / 2;
    visco_acoustic2d_lop_into(du, d.kmul.data_ptr<float>(), ws);                       // C1.re = L(du)
    const Buf P = visco_acoustic2d_real_alias(ws.C0);
    visco_ops::binary<visco_ops::MulOp>(P.data_ptr<float>(), lam, 1,
                                        visco_acoustic2d_real_ptr(ws.C1), VISCO_REAL_STRIDE, n);
    visco_ops::axpy(grad_A, P.data_ptr<float>(), 1, -dt, n);                            // grad_A.add_(P, -dt)
}

// grad_B1 += dt^2 * λ ⊙ L_{D_k2}(u[it]);  grad_B2 -= dt^2 * λ ⊙ L_{D_frac}(u[it]).
// Same (λ, u[it]) index pairing as grad_A's u_now slot; valid from it = 0
// (the dispersion term needs no u[it-1]).  Was grad_B1.add_(lam * Lop(u, Dk2),
// dt2) and grad_B2.add_(lam * Lop(u, Dfrac), -dt2), each product now through
// at::mul_out(P, lam, real(C1)) into C0's float alias, lam first as before.
void accumulate_grad_disp(float* grad_B1, float* grad_B2,
                          const float* lam,
                          const float* u_it,
                          const ViscoSpectral& d, float dt, ViscoScratch& ws)
{
    if (!d.disp || grad_B1 == nullptr) return;
    const int64_t n = ws.C1.numel() / 2;
    // add_(P, Scalar(dt2)): the double dt*dt, loaded as the float alpha
    const float dt2 = static_cast<float>(static_cast<double>(dt) * static_cast<double>(dt));
    const Buf P = visco_acoustic2d_real_alias(ws.C0);
    visco_acoustic2d_lop_into(u_it, d.Dk2.data_ptr<float>(), ws);
    visco_ops::binary<visco_ops::MulOp>(P.data_ptr<float>(), lam, 1, visco_acoustic2d_real_ptr(ws.C1), VISCO_REAL_STRIDE, n);
    visco_ops::axpy(grad_B1, P.data_ptr<float>(), 1, dt2, n);
    visco_acoustic2d_lop_into(u_it, d.Dfrac.data_ptr<float>(), ws);
    visco_ops::binary<visco_ops::MulOp>(P.data_ptr<float>(), lam, 1, visco_acoustic2d_real_ptr(ws.C1), VISCO_REAL_STRIDE, n);
    visco_ops::axpy(grad_B2, P.data_ptr<float>(), 1, -dt2, n);
}

// One fused exact-adjoint launch (reused acoustic2d kernel; the damping term
// is layered on top by adjoint_damping_extra).
inline void run_visco2d_adjoint_step(
    int order, dim3 grid, dim3 block,
    AcousticWavefieldPointer adj_view,
    const float* vp_ptr,
    LaplaceParam lap_ctx,
    GradParam grad_ctx_x, GradParam grad_ctx_z,
    AcousticCPMLPointer cpml, SolverContext ctx)
{
    SWEEP_CHECK(adj_view.zetaxn != nullptr && adj_view.psixn != nullptr,
        "fused adjoint needs the adjoint wavefield bound with psi+zeta "
        "double-buffer (11 tensors in 2D); set cuda_layout.adjoint_extra_nvar=2.");
    ACOUSTIC2D_ADJOINT_FUSED(order, grid, block,
        adj_view, vp_ptr, lap_ctx, grad_ctx_x, grad_ctx_z, cpml, ctx,
        adj_view.psixn, adj_view.psizn, adj_view.zetaxn, adj_view.zetazn,
        nullptr, nullptr);
}

// Imaging for one reverse step: recompute the vp_step-gradient carrier
// (vp^2 * Lap u) from the RAW pressure, then reuse the shared acoustic
// calculate_grad / accumulate_illumination_2d kernels.
void image_step_from_raw(
    int order, dim3 grid, dim3 block,
    const float* u_raw_ptr,
    const float* lam_ptr,
    const Buf& vp,
    const Buf& carrier,             // (B, nz, nx) scratch (the CARRIER slot), halo stays 0
    Buf* grad,
    RTMOutputCore* rtm_out,
    const LaplaceParam& lap_ctx,
    const SolverContext& ctx,
    int nx, int nz, float dt)
{
    if (grad == nullptr && rtm_out == nullptr) return;
    VISCO_ACOUSTIC2D_CARRIER(order, grid, block,
        u_raw_ptr, vp.data_ptr<float>(), carrier.data_ptr<float>(),
        lap_ctx, ctx);
    if (grad != nullptr) {
        // Imaging runs over the PHYSICAL box only: EdgePadding.backward crops the
        // pad gradients, so the skipped cells never reach the model. (This
        // branch's acoustic2d change; the shared kernel now takes the box.)
        calculate_grad<<<grid, block>>>(
            carrier.data_ptr<float>(), lam_ptr,
            vp.data_ptr<float>(), grad->data_ptr<float>(),
            nx, nz, dt,
            ctx.phys_x0(), ctx.phys_x1(), ctx.phys_z0(), ctx.phys_z1());
    }
    if (rtm_out != nullptr) {
        // ``carrier`` is vp^2*Lap(u) = u_tt, so the u_tt branch stays off. Same
        // physical box as the calculate_grad above: illumination and gradient
        // now cover the same cells.
        accumulate_illumination_2d<<<grid, block>>>(
            carrier.data_ptr<float>(), nullptr, nullptr, lam_ptr,
            rtm_out->source_illumination.data_ptr<float>(),
            rtm_out->receiver_illumination.data_ptr<float>(),
            nx, nz, dt,
            ctx.phys_x0(), ctx.phys_x1(), ctx.phys_z0(), ctx.phys_z1());
    }
}

void check_visco_backward(const BackwardInputCore& p)
{
    SWEEP_CHECK(p.models.size() == 4,
                "visco_acoustic2d expects the prepared models "
                "(vp_step, B1, B2, A); got ", p.models.size());
    SWEEP_CHECK(!p.bw_stepped() && p.step_phase == 0,
                "visco_acoustic2d does not support stepped backward segments");
    SWEEP_CHECK(p.cut_face_mask == 0,
                "visco_acoustic2d does not support domain decomposition");
    SWEEP_CHECK(!p.has_topo && !p.use_apm,
                "visco_acoustic2d does not support topography on impl='c' yet; "
                "use impl='eager'");
    SWEEP_CHECK(!p.compute_adcig,
                "visco_acoustic2d does not support ADCIG yet");
}

// Full-storage reverse sweep (the rtm() entry point it used to also serve
// was removed in 295858c).
void run_full_imaging_visco(
    const BackwardInputCore& p,
    Buf* grad,
    Buf* grad_A,
    Buf* grad_B1,
    Buf* grad_B2,
    Buf* grad_wavelet,
    RTMOutputCore* rtm_out)
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
    bind_adjoint_state(adjoint, p, "backward");

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto forward_source_config = fdtd::Geom::make(forward_nsrc, B);

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    ctx.set_cut_mask(0);
    acoustic_init_aux_slabs(ctx, adjoint);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    // Tables [Gp if a, Gd1, Gd2 if d] from p.derived_models; the carrier and
    // the spectral scratch [CARRIER, C0, C1, (R1 if d), FFT_WS] from
    // p.adjoint_workspace -- both count-checked against the flags at entry.
    ViscoSpectral spectral = visco_acoustic2d_make_spectral_from(
        p.eq_aux, p.models, p.derived_models, ViscoMode::Full, dt, nz, nx,
        "visco_acoustic2d::backward");
    ViscoScratch ws = visco_acoustic2d_bind_scratch(
        p.adjoint_workspace, adjoint.u_now_t, spectral.active, spectral.disp,
        ViscoMode::Full, "visco_acoustic2d::backward adjoint_workspace");

    for (int it = p.nt - 1; it >= 0; --it) {

        auto adj_view = adjoint.view();

        run_visco2d_adjoint_step(
            order, launch_config.grid, launch_config.block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx);

        adjoint_damping_extra(adjoint, spectral, ws, M);

        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer

        if (grad_wavelet != nullptr) {
            accumulate_source_grad_2d<<<forward_source_config.grid,
                                        forward_source_config.block>>>(
                adjoint.u_now_t.data_ptr<float>(),
                grad_wavelet->data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                ctx
            );
        }

        image_step_from_raw(
            order, launch_config.grid, launch_config.block,
            p.u_forward.select(0, it).data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp, ws.CARRIER, grad, rtm_out,
            lap_ctx, ctx, nx, nz, dt);

        if (grad_A != nullptr && spectral.active && it >= 1) {
            const Buf du = stage_difference(ws, p.u_forward.select(0, it).data_ptr<float>(),
                                            p.u_forward.select(0, it - 1).data_ptr<float>());
            accumulate_grad_A(grad_A->data_ptr<float>(), adjoint.u_now_t.template data_ptr<float>(),
                              du.data_ptr<float>(), spectral, dt, ws);
        }
        accumulate_grad_disp(grad_B1 ? grad_B1->data_ptr<float>() : nullptr,
                             grad_B2 ? grad_B2->data_ptr<float>() : nullptr,
                             adjoint.u_now_t.template data_ptr<float>(),
                             p.u_forward.select(0, it).data_ptr<float>(), spectral, dt, ws);
    }
}

} // namespace

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    check_visco_backward(in);
    SWEEP_CHECK(in.u_forward.defined() && in.u_forward.numel() > 0,
                "visco_acoustic2d backward (full) requires the raw forward "
                "wavefield history");
    BackwardOutputCore out;
    ViscoGrads grads = bind_grads(in);
    RTMOutputCore illumination = bind_illumination(in, /*always=*/false);
    run_full_imaging_visco(in, &grads.vp, &grads.A, &grads.B1, &grads.B2,
                           &grads.wavelet,
                           in.compute_illumination ? &illumination : nullptr);
    pack_outputs(out, grads, illumination);
    return out;
}

RTMOutputCore rtm_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    check_visco_backward(in);
    SWEEP_CHECK(
        in.u_forward.defined() && in.u_forward.numel() > 0,
        "visco_acoustic2d RTM requires full forward wavefields (raw pressure)."
    );
    SWEEP_CHECK(
        in.checkpoints.empty(),
        "visco_acoustic2d RTM does not support checkpoint mode."
    );

    RTMOutputCore out = bind_illumination(in, /*always=*/true);
    run_full_imaging_visco(in, nullptr, nullptr, nullptr, nullptr, nullptr, &out);
    return out;
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    SWEEP_CHECK(false,
        "visco_acoustic2d does not support boundary saving: the amplitude "
        "damping is dissipative (reverse-time reconstruction amplifies) and "
        "global (the |k| filter reads the whole padded grid, which boundary "
        "strips cannot restore).  Use memory=Ckpt() or memory=Full().");
    return {};
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    check_visco_backward(in);

    const auto& p = in;
    BackwardOutputCore out;

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "visco_acoustic2d"
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

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward_ckpt");

    // The chunk replay state: set 0 of p.forward_wavefields (the only set in
    // chunk mode), zeroed by the propagator, loaded per chunk below.
    check_replay_state_sets(p, 1, "visco_acoustic2d/backward_ckpt");
    AcousticWavefieldTensor forward;
    bind_replay_state(forward, p, vp, 0);
    // Slab geometry follows the FORWARD-state aux layout (the recompute runs
    // the forward kernel); the adjoint aux stays full-domain.
    acoustic_init_aux_slabs(ctx, forward);

    ViscoGrads grads = bind_grads(p);
    RTMOutputCore illumination = bind_illumination(p, /*always=*/false);
    RTMOutputCore* rtm_out = p.compute_illumination ? &illumination : nullptr;

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    // Tables [Gp, dt2A if a, Gd1, Gd2 if d] (the replay damps with dt2A, the
    // reverse step with Gp) from p.derived_models; the scratch
    // [CARRIER, C0, C1, (C2 if d), (R1 if d), (UPREV if a), FFT_WS] from
    // p.adjoint_workspace -- both count-checked against the flags at entry.
    ViscoSpectral spectral = visco_acoustic2d_make_spectral_from(
        p.eq_aux, p.models, p.derived_models, ViscoMode::Checkpoint, dt, nz, nx,
        "visco_acoustic2d::backward_ckpt");
    ViscoScratch ws = visco_acoustic2d_bind_scratch(
        p.adjoint_workspace, adjoint.u_now_t, spectral.active, spectral.disp,
        ViscoMode::Checkpoint, "visco_acoustic2d::backward_ckpt adjoint_workspace");

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    // RAW pressure store for the chunk (the acoustic twin stores the
    // vp^2*Lap(u) carrier; visco recomputes it from raw — see kernels.cuh).
    // Python-allocated with the checkpoint snapshots; every row is written by the replay before the reverse pass reads it.
    auto chunk_raw = pool_required(p.checkpoint_replay, 0, {chunk_size, B, nz, nx},
                                   "checkpoint_replay (visco_acoustic2d/backward_ckpt, "
                                   "cuda_layout.checkpoint_replay_shapes)");

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors());

        // u(start-1): the loaded state is (u_prev, u_now) = (u(start-1), u(start));
        // the it == start reverse step needs it for du/dt of step ``start``.
        // Kept in the UPREV slot (the replay rotates u_prev away).
        if (spectral.active)
            copy_tensor_device_to_device_async(ws.UPREV, forward.u_prev_t);

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();

            copy_tensor_device_to_device_async(chunk_raw.select(0, it - start), forward.u_now_t);

            ACOUSTIC2D(
                order,
                launch_config.grid,
                launch_config.block,
                for_view,
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

            visco_acoustic2d_apply_spectral_into(forward, spectral, ws, dt, M);

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

            run_visco2d_adjoint_step(
                order, launch_config.grid, launch_config.block,
                adj_view, vp.data_ptr<float>(),
                lap_ctx, grad_ctx_x, grad_ctx_z,
                cpml, ctx);

            adjoint_damping_extra(adjoint, spectral, ws, M);

            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                adj_view.u_next,
                p.adjoint_source.data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                ctx
            );

            adjoint.swap_aux();

            accumulate_source_grad_2d<<<fwd_source_config.grid,
                                        fwd_source_config.block>>>(
                adjoint.u_now_t.data_ptr<float>(),
                grads.wavelet.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                ctx
            );

            image_step_from_raw(
                order, launch_config.grid, launch_config.block,
                chunk_raw.select(0, it - start).data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp, ws.CARRIER, &grads.vp, rtm_out,
                lap_ctx, ctx, nx, nz, dt);

            if (spectral.active && it >= 1) {
                const Buf du = stage_difference(
                    ws, chunk_raw.select(0, it - start).data_ptr<float>(),
                    it > start ? chunk_raw.select(0, it - start - 1).data_ptr<float>() : ws.UPREV.data_ptr<float>());
                accumulate_grad_A(grads.A.data_ptr<float>(), adjoint.u_now_t.template data_ptr<float>(),
                                  du.data_ptr<float>(), spectral, dt, ws);
            }
            accumulate_grad_disp(grads.B1.data_ptr<float>(), grads.B2.data_ptr<float>(),
                                 adjoint.u_now_t.template data_ptr<float>(),
                                 chunk_raw.select(0, it - start).data_ptr<float>(), spectral, dt, ws);
        }
    }

    pack_outputs(out, grads, illumination);
    return out;
}

} // namespace visco_acoustic2d

// ---------------------------------------------------------------------------
// Recursive (binary) checkpointing.  Mirrors acoustic2d's driver; the leaf
// replays one visco forward step (stencil + spectral terms) from the interval
// state, recomputes the carrier from the raw pressure, and layers the damping
// terms onto the fused adjoint.  The checkpoint state is the same 6-tensor
// acoustic state — the amplitude damping is memoryless in (u_now, u_prev).
// ---------------------------------------------------------------------------
namespace visco_acoustic2d {
namespace {

int visco_recursive_scratch_depth(int interval_length)
{
    int depth = 0;
    while (interval_length > 1) {
        interval_length = (interval_length + 1) / 2;
        ++depth;
    }
    return depth;
}

void advance_forward_interval_visco_2d(
    AcousticWavefieldTensor& forward,
    int start,
    int end,
    int order,
    dim3 wave_grid,
    dim3 wave_block,
    dim3 source_grid,
    dim3 source_block,
    const BackwardInputCore& p,
    const Buf& vp,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const AcousticCPMLPointer& cpml,
    const SolverContext& ctx,
    int forward_nsrc,
    const ViscoSpectral& spectral,
    ViscoScratch& ws)
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

        visco_acoustic2d_apply_spectral_into(forward, spectral, ws, ctx.dt, ctx.M);

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

void process_recursive_interval_visco_2d(
    int start,
    int end,
    AcousticWavefieldTensor& start_state,
    AcousticWavefieldTensor& adjoint,
    const BackwardInputCore& p,
    const Buf& vp,
    Buf* grad,
    Buf* grad_A,
    Buf* grad_B1,
    Buf* grad_B2,
    Buf* grad_wavelet,
    RTMOutputCore* rtm_out,
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
    const ViscoSpectral& spectral,
    ViscoScratch& ws,
    int nx,
    int nz)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        // Pre-step: the state holds (u_prev, u_now) = (u(start-1), u(start));
        // the carrier is recomputed from u(start) before the replay step.
        VISCO_ACOUSTIC2D_CARRIER(order, wave_grid, wave_block,
            start_state.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            ws.CARRIER.data_ptr<float>(),
            lap_ctx, ctx);

        auto fwd_view = start_state.view();

        ACOUSTIC2D(
            order,
            wave_grid,
            wave_block,
            fwd_view,
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

        visco_acoustic2d_apply_spectral_into(start_state, spectral, ws, ctx.dt, ctx.M);

        add_source<<<forward_source_grid, forward_source_block>>>(
            fwd_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            start,
            forward_nsrc,
            ctx
        );

        start_state.swap();

        // The gradient bases used to be captured BEFORE the replay step
        // (du = u_now - u_prev and the two Lops of u_now, three fresh tensors).
        // swap() rotates (u_prev, u_now, u_next) <- (u_now, u_next, u_prev), so
        // afterwards u_prev_t IS the pre-step u_now (u(start)) and u_next_t the
        // pre-step u_prev (u(start-1)); the replay wrote only the old u_next
        // (now u_now_t) and the CPML aux, so both hold exactly the bits the
        // captures read.  They are consumed below, after the adjoint step, by
        // the same kernels in the same operand order -- later in the stream,
        // on values nothing has touched in between, i.e. bit for bit the same.
        const float* u_start = start_state.u_prev_t.template data_ptr<float>();    // u(start)
        const float* u_before = start_state.u_next_t.template data_ptr<float>();   // u(start-1)

        auto adj_view = adjoint.view();

        run_visco2d_adjoint_step(
            order, wave_grid, wave_block,
            adj_view, vp.data_ptr<float>(),
            lap_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx);

        adjoint_damping_extra(adjoint, spectral, ws, ctx.M);

        add_source<<<adj_source_grid, adj_source_block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            start,
            adjoint_nsrc,
            ctx
        );

        adjoint.swap_aux();

        if (grad_wavelet != nullptr) {
            accumulate_source_grad_2d<<<forward_source_grid, forward_source_block>>>(
                adjoint.u_now_t.data_ptr<float>(),
                grad_wavelet->data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                start,
                forward_nsrc,
                ctx
            );
        }

        if (grad != nullptr) {
            calculate_grad<<<wave_grid, wave_block>>>(
                ws.CARRIER.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                vp.data_ptr<float>(),
                grad->data_ptr<float>(),
                nx, nz, ctx.dt,
                ctx.phys_x0(), ctx.phys_x1(), ctx.phys_z0(), ctx.phys_z1());
        }
        if (rtm_out != nullptr) {
            accumulate_illumination_2d<<<wave_grid, wave_block>>>(
                ws.CARRIER.data_ptr<float>(), nullptr, nullptr,
                adjoint.u_now_t.data_ptr<float>(),
                rtm_out->source_illumination.data_ptr<float>(),
                rtm_out->receiver_illumination.data_ptr<float>(),
                nx, nz, ctx.dt,
                ctx.phys_x0(), ctx.phys_x1(), ctx.phys_z0(), ctx.phys_z1());
        }

        if (grad_A != nullptr && spectral.active && start >= 1) {
            const Buf du = stage_difference(ws, u_start, u_before);   // == pre-step u_now - u_prev
            accumulate_grad_A(grad_A->data_ptr<float>(), adjoint.u_now_t.template data_ptr<float>(),
                              du.data_ptr<float>(), spectral, ctx.dt, ws);
        }
        // == λ ⊙ Lop(pre-step u_now, D_k2 / D_frac), the former Pb / Rb
        accumulate_grad_disp(grad_B1 ? grad_B1->data_ptr<float>() : nullptr,
                             grad_B2 ? grad_B2->data_ptr<float>() : nullptr,
                             adjoint.u_now_t.template data_ptr<float>(), u_start,
                             spectral, ctx.dt, ws);
        return;
    }

    int mid = start + (end - start) / 2;

    SWEEP_CHECK(
        scratch_depth < static_cast<int>(scratch_states.size()),
        "Recursive checkpoint scratch depth exhausted."
    );
    AcousticWavefieldTensor& mid_state = scratch_states[scratch_depth];
    checkpoint_runtime.copy_state(mid_state.state_tensors(), start_state.state_tensors());
    advance_forward_interval_visco_2d(
        mid_state, start, mid, order,
        wave_grid, wave_block,
        forward_source_grid, forward_source_block,
        p, vp, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
        cpml, ctx, forward_nsrc, spectral, ws);

    process_recursive_interval_visco_2d(
        mid, end, mid_state, adjoint, p, vp,
        grad, grad_A, grad_B1, grad_B2, grad_wavelet, rtm_out,
        order, wave_grid, wave_block,
        forward_source_grid, forward_source_block,
        adj_source_grid, adj_source_block,
        lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
        cpml, ctx, forward_nsrc, adjoint_nsrc,
        checkpoint_runtime, scratch_states, scratch_depth + 1,
        spectral, ws, nx, nz);

    process_recursive_interval_visco_2d(
        start, mid, start_state, adjoint, p, vp,
        grad, grad_A, grad_B1, grad_B2, grad_wavelet, rtm_out,
        order, wave_grid, wave_block,
        forward_source_grid, forward_source_block,
        adj_source_grid, adj_source_block,
        lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
        cpml, ctx, forward_nsrc, adjoint_nsrc,
        checkpoint_runtime, scratch_states, scratch_depth + 1,
        spectral, ws, nx, nz);
}

} // namespace

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    check_visco_backward(in);
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(
        p.checkpoints.size() == 6,
        "visco_acoustic2d recursive checkpointing expects 6 checkpoint tensors"
    );

    const Buf& checkpoint_steps_cpu = p.checkpoint_steps;   // a host copy, made by the adapter
    SWEEP_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "visco_acoustic2d"
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

    SolverContext ctx{2, nx, 0, nz, B, dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};
    ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);

    AcousticWavefieldTensor adjoint;
    bind_adjoint_state(adjoint, p, "backward_recursive_ckpt");
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ViscoGrads grads = bind_grads(p);
    RTMOutputCore illumination = bind_illumination(p, /*always=*/false);
    RTMOutputCore* rtm_out = p.compute_illumination ? &illumination : nullptr;

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.bind(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    LaplaceParam lap_ctx{nx, 1, M, p.lap_coes.data_ptr<float>(), dx, 0, dz};
    GradParam grad_ctx{1, 0, nx, M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    // Tables [Gp, dt2A if a, Gd1, Gd2 if d] from p.derived_models; the scratch
    // [CARRIER, C0, C1, (C2 if d), (R1 if d), FFT_WS] from p.adjoint_workspace
    // -- both count-checked against the flags at entry.  CARRIER is zero at
    // entry (the pool's per-backward zeroing / zeros here): the carrier kernel
    // writes non-halo cells only and the halo band must stay 0 for the reused
    // grad/RTM kernels.
    ViscoSpectral spectral = visco_acoustic2d_make_spectral_from(
        p.eq_aux, p.models, p.derived_models, ViscoMode::Recursive, dt, nz, nx,
        "visco_acoustic2d::backward_recursive_ckpt");
    ViscoScratch ws = visco_acoustic2d_bind_scratch(
        p.adjoint_workspace, adjoint.u_now_t, spectral.active, spectral.disp,
        ViscoMode::Recursive, "visco_acoustic2d::backward_recursive_ckpt adjoint_workspace");

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    SWEEP_CHECK(
        p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
        "checkpoint_count does not match checkpoint_steps"
    );
    SWEEP_CHECK(
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

    // Replay state sets of p.forward_wavefields: set 0 is the segment start
    // state (zeroed or checkpoint-loaded per segment before any read), sets
    // 1..depth the bisection's scratch states (copy_state-filled from their
    // parent before any read).  The propagator hands 1 + depth sets, its depth
    // (_c.py _recursive_scratch_depth) the same halving loop on the same
    // longest segment.
    const int scratch_depth = visco_recursive_scratch_depth(max_segment_length);
    check_replay_state_sets(p, 1 + scratch_depth, "visco_acoustic2d/backward_recursive_ckpt");

    AcousticWavefieldTensor start_state;
    bind_replay_state(start_state, p, vp, 0);
    acoustic_init_aux_slabs(ctx, start_state);

    std::vector<AcousticWavefieldTensor> scratch_states(scratch_depth);
    for (int level = 0; level < scratch_depth; ++level)
        bind_replay_state(scratch_states[level], p, vp, /*set=*/level + 1);

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(), start_state.next_tensors());

        process_recursive_interval_visco_2d(
            start, end, start_state, adjoint, p, vp,
            &grads.vp, &grads.A, &grads.B1, &grads.B2,
            &grads.wavelet, rtm_out,
            order, launch_config.grid, launch_config.block,
            fwd_source_config.grid, fwd_source_config.block,
            adj_source_config.grid, adj_source_config.block,
            lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z,
            cpml, ctx, forward_nsrc, adjoint_nsrc,
            checkpoint_runtime, scratch_states, 0,
            spectral, ws, nx, nz);
    }

    pack_outputs(out, grads, illumination);
    return out;
}


BackwardOutput backward(const BackwardInput& in_torch)
{
    InputArena arena;
    const BackwardInputCore in = adapt_input(in_torch, arena);
    return to_torch(backward_core(in), in_torch);
}

RTMOutput rtm(const BackwardInput& in_torch)
{
    InputArena arena;
    const BackwardInputCore in = adapt_input(in_torch, arena);
    return to_torch(rtm_core(in), in_torch);
}

BackwardOutput backward_bs(const BackwardInput& in_torch)
{
    InputArena arena;
    const BackwardInputCore in = adapt_input(in_torch, arena);
    return to_torch(backward_bs_core(in), in_torch);
}

BackwardOutput backward_ckpt(const BackwardInput& in_torch)
{
    InputArena arena;
    const BackwardInputCore in = adapt_input(in_torch, arena);
    return to_torch(backward_ckpt_core(in), in_torch);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in_torch)
{
    InputArena arena;
    const BackwardInputCore in = adapt_input(in_torch, arena);
    return to_torch(backward_recursive_ckpt_core(in), in_torch);
}

} // namespace visco_acoustic2d
