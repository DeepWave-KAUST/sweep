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

// Proper-adjoint (transpose) of the scattered-field propagation: L* = lap(v^2.l).
#define ACOUSTIC_LSRTM2D_ADJOINT(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_lsrtm2nd_adjoint_by_order, order, grid, block, __VA_ARGS__)

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
    bool in_pml = solver.in_pml_2d(ix, iz, halo);
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
    int halo = solver.abcn > 0 ? solver.abcn + 2 * M + 1 : 2 * M;
    int top_halo = solver.free_surface ? 2 * M : halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
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

// Proper adjoint (transpose) of the lsrtm scattered-field propagation:
//   lambda_next = 2 lambda_now - lambda_pre + dt^2 * lap(v^2 * lambda_now)
// in the non-PML interior -- the transpose of the forward  v^2 * lap(lambda)
// (vp^2 * laplacian is non-self-adjoint when vp varies).  The PML band keeps
// the forward CPML formulation (reusing acoustic_cpml_update_2d); the dominant
// non-self-adjoint error is in the variable-velocity interior.  Mirrors the
// acoustic2d fix (5188031).  Requires v2_lambda = vp^2*lambda_now pre-computed.
template <int Order>
__global__ void acoustic_lsrtm2nd_adjoint(
    AcousticWavefieldPointer wf,
    const float* __restrict__ v2_lambda,
    const float* __restrict__ pml_field,   // nullptr = wf.u_now (see the PML band below)
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
    const float* vp_b = vp + b * spatial_size;
    const float* v2l_b = v2_lambda + b * spatial_size;
    float dt2 = solver.dt * solver.dt;

    bool in_pml = solver.in_pml_2d(ix, iz, halo);

    if (!in_pml) {
        // L* = lap(v^2 . lambda)  (transpose of forward v^2 . lap(lambda))
        float lap_x = laplace<2, Order, X>(v2l_b, ix, 0, iz, lap_ctx);
        float lap_z = laplace<2, Order, Z>(v2l_b, ix, 0, iz, lap_ctx);
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + dt2 * (lap_x + lap_z);
        return;
    }

    // PML band: keep the forward CPML formulation (shared lsrtm update).  It does
    // not read v2_lambda, so a field driven through v2_lambda by more than its own
    // u_now -- the RWI background adjoint, v2 = vp^2*(lambda_bg + mp*lambda_sc) --
    // passes that combined field as pml_field: the band then differentiates it
    // (and runs the CPML memory on it) instead of u_now alone, the approximation
    // the scattered adjoint makes for itself.  Without it the coupling is dropped
    // wherever mp reaches into the band (edge-padded mp: vp.grad off by percents).
    AcousticWavefieldPointer fp = f;
    if (pml_field) fp.u_now = const_cast<float*>(pml_field) + b * spatial_size;   // only read
    float w_sum = acoustic_cpml_update_2d<Order>(fp, ix, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, solver, halo);
    float v = vp_b[idx];
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + (v * v) * dt2 * w_sum;
}

using acoustic_lsrtm2nd_adjoint_fn = void (*)(AcousticWavefieldPointer, const float*,
    const float*, const float*, LaplaceParam, GradParam, GradParam, GradParam, AcousticCPMLPointer,
    SolverContext);
SWEEP_BY_ORDER_DECL(acoustic_lsrtm2nd_adjoint_fn, acoustic_lsrtm2nd_adjoint_by_order);

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
    float* __restrict__ g,
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
    const float* __restrict__ vp,
    float* __restrict__ grad_vp,
    int nx, int nz
);
