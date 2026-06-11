#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

#define ACOUSTIC_LSRTM2D_SINGLE(order, grid, block, ...)                         \
    do {                                                                         \
        if      ((order) == 2) acoustic2d_single<2><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 4) acoustic2d_single<4><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 6) acoustic2d_single<6><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 8) acoustic2d_single<8><<<grid, block>>>(__VA_ARGS__);    \
        else                   acoustic2d_single<-1><<<grid, block>>>(__VA_ARGS__);   \
    } while (0)

#define ACOUSTIC_LSRTM2D_SINGLE_NOPML(order, grid, block, ...)                   \
    do {                                                                         \
        if      ((order) == 2) acoustic2d_single_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic2d_single_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic2d_single_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic2d_single_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic2d_single_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define ACOUSTIC_LSRTM2D_COUPLED(order, grid, block, ...)                        \
    do {                                                                         \
        if      ((order) == 2) acoustic_lsrtm2nd<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_lsrtm2nd<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_lsrtm2nd<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_lsrtm2nd<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_lsrtm2nd<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

// Proper-adjoint (transpose) of the scattered-field propagation: L* = lap(v^2.l).
#define ACOUSTIC_LSRTM2D_ADJOINT(order, grid, block, ...)                                 \
    do {                                                                                   \
        if      ((order) == 2) acoustic_lsrtm2nd_adjoint<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_lsrtm2nd_adjoint<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_lsrtm2nd_adjoint<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_lsrtm2nd_adjoint<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_lsrtm2nd_adjoint<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

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
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);
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

// v2_lambda = vp^2 * lambda  (per-cell, race-free pre-pass for the L* adjoint).
static __global__ void compute_v2_lambda_lsrtm2d(
    const float* __restrict__ vp,
    const float* __restrict__ lambda,
    float* __restrict__ v2_lambda,
    int nx, int nz, int B
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= nx || iz >= nz || b >= B) return;
    int sp = nx * nz; int idx = b * sp + iz * nx + ix;
    float v = vp[idx];
    v2_lambda[idx] = v * v * lambda[idx];
}

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

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        // L* = lap(v^2 . lambda)  (transpose of forward v^2 . lap(lambda))
        float lap_x = laplace<2, Order, X>(v2l_b, ix, 0, iz, lap_ctx);
        float lap_z = laplace<2, Order, Z>(v2l_b, ix, 0, iz, lap_ctx);
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + dt2 * (lap_x + lap_z);
        return;
    }

    // PML band: keep the forward CPML formulation (shared lsrtm update).
    float w_sum = acoustic_cpml_update_2d<Order>(f, ix, iz, idx, cpml, lap_ctx, grad_ctx, grad_ctx_x, grad_ctx_z, solver, halo);
    float v = vp_b[idx];
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + (v * v) * dt2 * w_sum;
}

template <int Order>
__global__ void acoustic_lsrtm2nd(
    AcousticWavefieldPointer bg,
    AcousticWavefieldPointer sc,
    bool save_all_wavefields,
    float* __restrict__ bg_utt,
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
}

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
