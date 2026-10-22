#pragma once
#include "../../launch/by_order.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

// NOTE: the background-field kernels below are LSRTM-private. They MUST NOT
// reuse the global names ``acoustic2nd`` / ``acoustic2nd_nopml`` defined in
// ../acoustic2d/kernels.cuh: both headers are compiled into separate TUs of
// the same extension, and two __global__ templates with one mangled name but
// different bodies make the CUDA runtime's stub->module resolution pick a
// winner per process (ODR violation) — observed as 1-ULP bimodal forward
// output and the DD backward losing its cut-aware nopml bands. Hence the
// ``acoustic2d_single*`` names (mirrors lsrtm3d's ``acoustic3d_single*``).
#define ACOUSTIC_LSRTM2D_SINGLE(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic2d_single_by_order, order, grid, block, __VA_ARGS__)

#define ACOUSTIC_LSRTM2D_SINGLE_NOPML(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic2d_single_nopml_by_order, order, grid, block, __VA_ARGS__)

#define ACOUSTIC_LSRTM2D_COUPLED(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_lsrtm2nd_by_order, order, grid, block, __VA_ARGS__)

// Exact transpose of a field's step (acoustic_lsrtm2nd_adjoint_exact below).
#define ACOUSTIC_LSRTM2D_ADJOINT(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_lsrtm2nd_adjoint_exact_by_order, order, grid, block, __VA_ARGS__)

template <int Order>
__device__ inline float acoustic_cpml_update_2d(
    AcousticWavefieldPointer f,
    int ix,
    int iz,
    int idx,
    const AcousticCPMLPointer& cpml,
    const LaplaceParam& lap_ctx,
    const GradParam& grad_ctx,
    const GradParam& grad_ctx_x,
    const GradParam& grad_ctx_z,
    const SolverContext& solver,
    int halo
) {
    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    // Interior fast-path: ax/bx/dbxdx vanish, so w_sum reduces to lap_x+lap_z
    // and the aux fields psix/psiz/zetax/zetaz stay zero. Skip the loads/stores.
    // in_cpml: the band AND the cell past it, where dbxdx = np.gradient(bx)
    // is still nonzero (dropping it there reflected off the CPML).
    bool in_pml = solver.in_cpml_2d(ix, iz, halo);
    if (!in_pml) return lap_x + lap_z;

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);
    float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dpsizdz = gradient<2, Order, Z>(f.psiz, ix, 0, iz, grad_ctx);
    float dpsixdx = gradient<2, Order, X>(f.psix, ix, 0, iz, grad_ctx);
    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
    float daipsiz_dz = az_ * dpsizdz + dazdz * f.psiz[idx];
    float daipxix_dx = ax_ * dpsixdx + daxdx * f.psix[idx];

    float w_sum = 0.0f;

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dudx) + daipxix_dx;
    w_sum += (1.0f + bx_) * tmpx + ax_ * f.zetax[idx];
    (f.psixn ? f.psixn : f.psix)[idx] = bx_ * dudx + ax_ * f.psix[idx];   // race-free fwd double-buffer
    f.zetax[idx] = bx_ * tmpx + ax_ * f.zetax[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dudz) + daipsiz_dz;
    w_sum += (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];
    (f.psizn ? f.psizn : f.psiz)[idx] = bz_ * dudz + az_ * f.psiz[idx];   // race-free fwd double-buffer
    f.zetaz[idx] = bz_ * tmpz + az_ * f.zetaz[idx];

    return w_sum;
}

template <int Order>
__global__ void acoustic2d_single(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float w_sum = acoustic_cpml_update_2d<Order>(f, ix, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, solver, halo);
    float v = vp_b[idx];
    float utt = (v * v) * w_sum;

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * utt;
    if (save_all_wavefields && u_this_b != nullptr)
        u_this_b[idx] = utt;
}

using acoustic2d_single_fn = void (*)(AcousticWavefieldPointer, bool, float*,
    const float*, LaplaceParam, GradParam, GradParam, GradParam, AcousticCPMLPointer,
    SolverContext);
SWEEP_BY_ORDER_DECL(acoustic2d_single_fn, acoustic2d_single_by_order);

template <int Order>
__global__ void acoustic2d_single_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int M = (Order == -1) ? solver.M : (Order / 2);
    // On a physical face the reconstruction starts just inside the boundary-
    // saving restore strip (abcn + M halo + save_width = abcn + 2M + 1).  A DD
    // CUT face has neither the abcn pad nor a restore strip -- its halo is
    // filled by the NCCL exchange -- so the reconstruction must reach in to the
    // M-wide halo there; the abcn-based bound would leave abcn + M + 1 physical
    // cells beside every cut never reconstructed.  cut_mask == 0 leaves every
    // bound exactly as before.
    int halo = solver.abcn > 0 ? solver.abcn + 2 * M + 1 : 2 * M;
    int top_halo = solver.free_surface ? 2 * M : halo;
    int x_lo = solver.cut_x_lo() ? M : halo, x_hi = solver.cut_x_hi() ? M : halo;
    int z_lo = solver.cut_z_lo() ? M : top_halo, z_hi = solver.cut_z_hi() ? M : halo;
    if (ix < x_lo || ix >= solver.nx - x_hi || iz < z_lo || iz >= solver.nz - z_hi)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);
    float w_sum = lap_x + lap_z;
    float v = vp_b[idx];
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * (v * v) * w_sum;
}

using acoustic2d_single_nopml_fn = void (*)(AcousticWavefieldPointer, const float*,
    LaplaceParam, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic2d_single_nopml_fn, acoustic2d_single_nopml_by_order);

// v2_lambda = vp^2 * lambda  (per-cell, race-free pre-pass for the L* adjoint).
// (Defined in kernels.cu: a definition here is compiled into every TU that
// includes this header.)
__global__ void compute_v2_lambda_lsrtm2d(
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx, int nz, int B
);

// Exact transpose of one step of a field advanced by acoustic_cpml_update_2d
// (the product-rule CPML).  Per axis d the forward computes, with u = u_now,
//   t_d    = (1+b_d) L_d u + db_d D_d u + da_d psi_d + a_d D_d psi_d
//   psi_d' = b_d D_d u + a_d psi_d,    zeta_d' = b_d t_d + a_d zeta_d
//   u'     = 2 u - u_prev + dt^2 vp^2 sum_d [(1+b_d) t_d + a_d zeta_d].
// With g = dt^2 * v2l at every cell -- the adjoint of the operator output
// sum_d[...]: v2l = vp^2 * lambda for a field's own adjoint, and for the RWI
// background adjoint vp^2 * (lambda_bg + mp * lambda_sc), the background
// update also driving the scattered field -- and gtmp_d = (1+b_d) g + b_d zeta*_d:
//   lambda_next = 2 lambda - lambda_prev + sum_d [ L_d((1+b_d) gtmp_d) - D_d(b_d psi*_d + db_d gtmp_d) ]
//   psi*_d'     = a_d psi*_d + da_d gtmp_d - D_d(a_d gtmp_d)
//   zeta*_d'    = a_d (zeta*_d + g)
// (L symmetric, D antisymmetric; the forward's coefficients at the taps).  An
// axis none of whose taps is in_cpml (band or db leak cell) sees a = b = db = 0
// at every tap: its terms reduce to L_d(g) and its aux outputs to 0, which the
// zero-allocated double buffers already hold, so they are not written.  The
// new psi* / zeta* land in the double buffers -- the neighbours' old ones are
// read -- and the caller rotates with swap_aux().
// Replaces the approximate adjoint that ran the FORWARD CPML in the band
// (grad[mp] off by 1-10 % against the exact transpose).
template <int Order>
__global__ void acoustic_lsrtm2nd_adjoint_exact(
    AcousticWavefieldPointer wf,
    const float* __restrict__ v2l,
    LaplaceParam lap_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;
    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    const int sp = solver.nx * solver.nz;
    const int idx = iz * solver.nx + ix;
    const int sz = solver.nx;
    auto f = wf.offset(b, sp);
    const float* g_b = v2l + b * sp;
    const float dt2 = solver.dt * solver.dt;
    const float* lc = lap_ctx.coeff;     // [c0 (centre, applied as -lc[0]), c1..cM]
    const float* gc = grad_ctx_x.coeff;  // [_, c1..cM] (antisymmetric)
    const float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    const float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    const float invdx = 1.0f / lap_ctx.dx;
    const float invdz = 1.0f / lap_ctx.dz;

    // Does the tap range [i - M, i + M] meet a band face's in_cpml cells?
    const bool xfull = solver.taps_in_cpml_x(ix, halo, halo);
    const bool zfull = solver.taps_in_cpml_z(iz, halo, halo);

    if (!xfull && !zfull) {
        // No tap in the CPML: L* = lap(v^2 . lambda), the transpose of the
        // forward v^2 . lap (compile-time stencils, one dt^2 at the end).
        float lap_x = laplace<2, Order, X>(g_b, ix, 0, iz, lap_ctx);
        float lap_z = laplace<2, Order, Z>(g_b, ix, 0, iz, lap_ctx);
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + dt2 * (lap_x + lap_z);
        return;
    }

    #define LSRTM_GW(n)          ( dt2 * g_b[n] )
    #define LSRTM_GTMP_X(n, nix) ( cpml.bx[nix] * f.zetax[n] + (1.0f + cpml.bx[nix]) * LSRTM_GW(n) )
    #define LSRTM_GTMP_Z(n, niz) ( cpml.bz[niz] * f.zetaz[n] + (1.0f + cpml.bz[niz]) * LSRTM_GW(n) )

    float Lx, Dx_gg = 0.0f;
    if (xfull) {
        const float ax_ = cpml.ax[ix];
        const float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
        const float gtmpx0 = LSRTM_GTMP_X(idx, ix);
        Lx = -lc[0] * ((1.0f + cpml.bx[ix]) * gtmpx0);
        float Dx_gq = 0.0f;
        #pragma unroll
        for (int k = 1; k <= (Order > 0 ? M_static : 16); ++k) {
            if (Order <= 0 && k > halo) break;
            const int np = idx + k, nm = idx - k, nixp = ix + k, nixm = ix - k;
            const float tp = LSRTM_GTMP_X(np, nixp), tm = LSRTM_GTMP_X(nm, nixm);
            Lx += lc[k] * ((1.0f + cpml.bx[nixp]) * tp + (1.0f + cpml.bx[nixm]) * tm);
            Dx_gg += gc[k] * ((cpml.bx[nixp] * f.psix[np] + cpml.dbxdx[nixp] * tp)
                            - (cpml.bx[nixm] * f.psix[nm] + cpml.dbxdx[nixm] * tm));
            Dx_gq += gc[k] * (cpml.ax[nixp] * tp - cpml.ax[nixm] * tm);
        }
        Lx *= invdx2; Dx_gg *= invdx; Dx_gq *= invdx;
        f.psixn[idx] = ax_ * f.psix[idx] + daxdx * gtmpx0 - Dx_gq;
        f.zetaxn[idx] = ax_ * (f.zetax[idx] + LSRTM_GW(idx));
    } else {
        Lx = dt2 * laplace<2, Order, X>(g_b, ix, 0, iz, lap_ctx);
    }

    float Lz, Dz_gg = 0.0f;
    if (zfull) {
        const float az_ = cpml.az[iz];
        const float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
        const float gtmpz0 = LSRTM_GTMP_Z(idx, iz);
        Lz = -lc[0] * ((1.0f + cpml.bz[iz]) * gtmpz0);
        float Dz_gq = 0.0f;
        #pragma unroll
        for (int k = 1; k <= (Order > 0 ? M_static : 16); ++k) {
            if (Order <= 0 && k > halo) break;
            const int np = idx + k * sz, nm = idx - k * sz, nizp = iz + k, nizm = iz - k;
            const float tp = LSRTM_GTMP_Z(np, nizp), tm = LSRTM_GTMP_Z(nm, nizm);
            Lz += lc[k] * ((1.0f + cpml.bz[nizp]) * tp + (1.0f + cpml.bz[nizm]) * tm);
            Dz_gg += gc[k] * ((cpml.bz[nizp] * f.psiz[np] + cpml.dbzdz[nizp] * tp)
                            - (cpml.bz[nizm] * f.psiz[nm] + cpml.dbzdz[nizm] * tm));
            Dz_gq += gc[k] * (cpml.az[nizp] * tp - cpml.az[nizm] * tm);
        }
        Lz *= invdz2; Dz_gg *= invdz; Dz_gq *= invdz;
        f.psizn[idx] = az_ * f.psiz[idx] + dazdz * gtmpz0 - Dz_gq;
        f.zetazn[idx] = az_ * (f.zetaz[idx] + LSRTM_GW(idx));
    } else {
        Lz = dt2 * laplace<2, Order, Z>(g_b, ix, 0, iz, lap_ctx);
    }
    #undef LSRTM_GW
    #undef LSRTM_GTMP_X
    #undef LSRTM_GTMP_Z

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + (Lx + Lz - Dx_gg - Dz_gg);
}

using acoustic_lsrtm2nd_adjoint_exact_fn = void (*)(AcousticWavefieldPointer, const float*,
    LaplaceParam, GradParam, GradParam, AcousticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic_lsrtm2nd_adjoint_exact_fn, acoustic_lsrtm2nd_adjoint_exact_by_order);

template <int Order>
__global__ void acoustic_lsrtm2nd(
    AcousticWavefieldPointer bg,
    AcousticWavefieldPointer sc,
    bool save_all_wavefields,
    float* __restrict__ bg_utt,
    float* __restrict__ sc_utt,   // RWI: the scattered u_tt, for the tomographic vp gradient
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto bg_f = bg.offset(b, spatial_size);
    auto sc_f = sc.offset(b, spatial_size);
    float* bg_utt_b = bg_utt ? bg_utt + b * spatial_size : nullptr;
    float* sc_utt_b = sc_utt ? sc_utt + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;
    const float* mp_b = mp + b * spatial_size;

    float bg_w_sum = acoustic_cpml_update_2d<Order>(bg_f, ix, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, solver, halo);
    float sc_w_sum = acoustic_cpml_update_2d<Order>(sc_f, ix, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, solver, halo);

    float v = vp_b[idx];
    float bg_utt_val = (v * v) * bg_w_sum;
    float sc_utt_val = (v * v) * sc_w_sum;

    bg_f.u_next[idx] = 2.0f * bg_f.u_now[idx] - bg_f.u_prev[idx] + solver.dt * solver.dt * bg_utt_val;
    sc_f.u_next[idx] = 2.0f * sc_f.u_now[idx] - sc_f.u_prev[idx] + solver.dt * solver.dt * (sc_utt_val + mp_b[idx] * bg_utt_val);

    if (save_all_wavefields && bg_utt_b != nullptr)
        bg_utt_b[idx] = bg_utt_val;
    if (save_all_wavefields && sc_utt_b != nullptr)
        sc_utt_b[idx] = sc_utt_val;
}

using acoustic_lsrtm2nd_fn = void (*)(AcousticWavefieldPointer,
    AcousticWavefieldPointer, bool, float*, float*, const float*, const float*, LaplaceParam,
    GradParam, GradParam, GradParam, AcousticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic_lsrtm2nd_fn, acoustic_lsrtm2nd_by_order);

__global__ void calculate_grad_lsrtm_mp(
    const float* __restrict__ u_tt_bg,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
);

__global__ void calculate_grad_lsrtm_mp_utt(
    const float* __restrict__ u_forward_next,
    const float* __restrict__ u_forward_now,
    const float* __restrict__ u_forward_prev,
    const float* __restrict__ u_backward,
    const float* __restrict__ vp,
    float* __restrict__ grad_mp,
    int nx,
    int nz,
    float dt
);

// ---- RWI tomographic vp gradient (Wu & Alkhalifah 2015) ----

__global__ void compute_v2_lambda_bg_lsrtm2d(
    const float* __restrict__ vp,
    const float* __restrict__ mp,
    const float* __restrict__ lambda_bg,
    const float* __restrict__ lambda_sc,
    float* __restrict__ v2_lambda,
    int nx, int nz, int B
);

__global__ void calculate_grad_lsrtm_vp_utt(
    const float* __restrict__ bg_utt,
    const float* __restrict__ sc_utt,
    const float* __restrict__ lam_bg,
    const float* __restrict__ lam_sc,
    const float* __restrict__ mp,
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_iii,
    int nx, int nz, float dt
);

__global__ void add_lsrtm_scattered_coupling_2d(
    float* __restrict__ sc_next,
    const float* __restrict__ bg_next,
    const float* __restrict__ bg_now,
    const float* __restrict__ bg_prev,
    const float* __restrict__ mp,
    int nx, int nz, int halo
);

__global__ void calculate_grad_lsrtm_vp_2diff_2d(
    const float* __restrict__ bg_prev, const float* __restrict__ bg_now, const float* __restrict__ bg_next,
    const float* __restrict__ sc_prev, const float* __restrict__ sc_now, const float* __restrict__ sc_next,
    const float* __restrict__ lam_bg, const float* __restrict__ lam_sc,
    const float* __restrict__ mp, const float* __restrict__ vp,
    float* __restrict__ grad_vp, float* __restrict__ grad_iii,
    int nx, int nz
);
