#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/staggered.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"

// Soares & Sacchi 2025 first-order momentum-stress vector-reflectivity
// elastic equation. Mirrors elastic2d kernel layout (LAUNCH macros +
// templated kernels per FD order) and re-uses ElasticWavefieldPointer
// for the state layout -- the 5 base fields (vx/vz -> px/pz,
// sxx/szz/sxz unchanged) and the 10 CPML memvars use the same memory
// slots as ::elastic2d.

namespace elastic_vr2d_kernels {

// ---------------------------------------------------------------------------
// Launch macros
// ---------------------------------------------------------------------------

#define LAUNCH_EVR_MOMENTUM(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_momentum_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_momentum_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_momentum_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_momentum_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_momentum_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_EVR_STRESS(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_stress_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_stress_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_stress_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_stress_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_stress_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_EVR_MOMENTUM_NOPML(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_momentum_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_momentum_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_momentum_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_momentum_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_momentum_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_EVR_STRESS_NOPML(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_stress_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_stress_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_stress_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_stress_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_stress_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_EVR_STRESS_ADJOINT_PREPARE(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_stress_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_stress_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_stress_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_stress_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_stress_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_EVR_STRESS_ADJOINT_APPLY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_stress_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_stress_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_stress_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_stress_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_stress_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_EVR_MOMENTUM_ADJOINT_PREPARE(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_momentum_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_momentum_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_momentum_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_momentum_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_momentum_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_EVR_MOMENTUM_ADJOINT_APPLY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_momentum_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_momentum_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_momentum_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_momentum_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_momentum_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_CALCULATE_GRAD_EVR_NOBS(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::calculate_grad_evr_nobs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::calculate_grad_evr_nobs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::calculate_grad_evr_nobs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::calculate_grad_evr_nobs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::calculate_grad_evr_nobs<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

// ---------------------------------------------------------------------------
// Centered first derivatives for the cached velocity gradients
// vp_x / vp_z / vs_x / vs_z. These reuse the canonical regular-grid
// ``gradient<ND, Order, Direction>`` operator from ../../operators/gradient.cuh
// (the same ``centered_gradient_stencil`` the second-order acoustic kernels
// use) rather than a hand-rolled 4th-order stencil -- so the accuracy
// tracks the equation's ``spatial_order`` for free.
//
// Order semantics (matches the kernel template):
//   * Order in {2,4,6,8}: compile-time centered stencil of that order.
//     ``GradParam.coeff`` is unused (the stencils are hard-coded), so it is
//     safe that FirstOrderEquation ships *staggered* grad coefficients --
//     they are never read on this path.
//   * Order == -1 (spatial_order > 8): fall back to the 4th-order centered
//     stencil. The model halo at that order is >= 5 cells, so the +/-2
//     reads stay in bounds, and the velocity-gradient term is only a
//     smooth-model correction so 4th order is ample.
//
// The stride layout comes straight from the kernel's ``SGradParam grad_ctx``
// (sx = 1, sz = nx), so x uses stride sx and z uses stride sz.
// ---------------------------------------------------------------------------

template<int Order>
__device__ __forceinline__ float
evr_model_grad_x(const float* __restrict__ u, int ix, int iz, const SGradParam& g)
{
    GradParam gp{g.sx, g.sy, g.sz, g.M, g.coeff, g.dx, g.dy, g.dz};
    if constexpr (Order == -1)
        return gradient<2, 4, X>(u, ix, 0, iz, gp);
    else
        return gradient<2, Order, X>(u, ix, 0, iz, gp);
}

template<int Order>
__device__ __forceinline__ float
evr_model_grad_z(const float* __restrict__ u, int ix, int iz, const SGradParam& g)
{
    GradParam gp{g.sx, g.sy, g.sz, g.M, g.coeff, g.dx, g.dy, g.dz};
    if constexpr (Order == -1)
        return gradient<2, 4, Z>(u, ix, 0, iz, gp);
    else
        return gradient<2, Order, Z>(u, ix, 0, iz, gp);
}

// ---------------------------------------------------------------------------
// Momentum kernel  -- same as elastic_velocity but WITHOUT 1/rho.
//   d_t p_i = d_j sigma_ij                             (S&S 2025 Eq 23)
// ---------------------------------------------------------------------------

template<int Order>
__global__ void evr_momentum_kernel(
    ElasticWavefieldPointer wf,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    float dsxx_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.szz, ix, 0, iz, grad_ctx);

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.vx[idx] += solver.dt * (dsxx_dx + dsxz_dz);   // px += ...
        f.vz[idx] += solver.dt * (dsxz_dx + dszz_dz);   // pz += ...
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];
    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];

    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];
    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];

    f.vx[idx] += solver.dt * (dsxx_dx + dsxz_dz);
    f.vz[idx] += solver.dt * (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void evr_momentum_kernel_nopml(
    ElasticWavefieldPointer wf,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    float dsxx_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.szz, ix, 0, iz, grad_ctx);

    f.vx[idx] += solver.dt * (dsxx_dx + dsxz_dz);
    f.vz[idx] += solver.dt * (dsxz_dx + dszz_dz);
}

// ---------------------------------------------------------------------------
// Stress kernel  --  the gamma-based update (S&S 2025 Eq 24-26).
//
//   gamma_alpha_ij = V_alpha (d_i V_alpha) p_j - 2 V_alpha^2 R_alpha,i p_j
//                  + V_alpha^2 (d_i p_j)
//
//   d_t sigma_xx = (gamma_P_xx + gamma_P_zz) - 2 gamma_S_zz
//   d_t sigma_zz = (gamma_P_xx + gamma_P_zz) - 2 gamma_S_xx
//   d_t sigma_xz = gamma_S_xz + gamma_S_zx
//
// vp_x/vp_z/vs_x/vs_z are computed on-the-fly with the 4th-order
// central FD helper above -- the model fields are smooth so this is
// cheap and avoids extra global memory.
// ---------------------------------------------------------------------------

template<int Order>
__global__ void evr_stress_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ Rp_x,
    const float* __restrict__ Rp_z,
    const float* __restrict__ Rs_x,
    const float* __restrict__ Rs_z,
    float* __restrict__ u_this,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;

    const float* vp_b   = vp   + b * spatial_size;
    const float* vs_b   = vs   + b * spatial_size;
    const float* Rpx_b  = Rp_x + b * spatial_size;
    const float* Rpz_b  = Rp_z + b * spatial_size;
    const float* Rsx_b  = Rs_x + b * spatial_size;
    const float* Rsz_b  = Rs_z + b * spatial_size;

    // Momentum derivatives (same stagger as elastic vx/vz derivatives).
    float dpx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dpz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(f.vz, ix, 0, iz, grad_ctx);
    float dpx_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dpz_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.vz, ix, 0, iz, grad_ctx);

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (in_pml) {
        // CPML on the momentum derivatives (same memvar layout as elastic m_vxx etc.).
        float az = cpml.az[iz];
        float bz = cpml.bz[iz];
        float azh = cpml.azh[iz];
        float bzh = cpml.bzh[iz];
        float ax = cpml.ax[ix];
        float bx = cpml.bx[ix];
        float axh = cpml.axh[ix];
        float bxh = cpml.bxh[ix];

        f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dpx_dx;
        dpx_dx += f.m_vxx[idx];
        f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dpz_dz;
        dpz_dz += f.m_vzz[idx];
        f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dpx_dz;
        dpx_dz += f.m_vxz[idx];
        f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dpz_dx;
        dpz_dx += f.m_vzx[idx];
    }

    // Velocity gradients (4th-order central, cached per-cell on-the-fly).
    float dvp_dx = evr_model_grad_x<Order>(vp_b, ix, iz, grad_ctx);
    float dvp_dz = evr_model_grad_z<Order>(vp_b, ix, iz, grad_ctx);
    float dvs_dx = evr_model_grad_x<Order>(vs_b, ix, iz, grad_ctx);
    float dvs_dz = evr_model_grad_z<Order>(vs_b, ix, iz, grad_ctx);

    float Vp   = vp_b[idx];
    float Vs   = vs_b[idx];
    float Vp2  = Vp * Vp;
    float Vs2  = Vs * Vs;
    float Rpx  = Rpx_b[idx];
    float Rpz  = Rpz_b[idx];
    float Rsx  = Rsx_b[idx];
    float Rsz  = Rsz_b[idx];

    float px = f.vx[idx];
    float pz = f.vz[idx];

    // 8 gamma components. Naming: gamma_X_ij where i = derivative axis,
    // j = momentum component.
    float gamma_P_xx = Vp * dvp_dx * px - 2.f * Vp2 * Rpx * px + Vp2 * dpx_dx;
    float gamma_P_zz = Vp * dvp_dz * pz - 2.f * Vp2 * Rpz * pz + Vp2 * dpz_dz;
    float gamma_S_xx = Vs * dvs_dx * px - 2.f * Vs2 * Rsx * px + Vs2 * dpx_dx;
    float gamma_S_zz = Vs * dvs_dz * pz - 2.f * Vs2 * Rsz * pz + Vs2 * dpz_dz;
    float gamma_S_xz = Vs * dvs_dx * pz - 2.f * Vs2 * Rsx * pz + Vs2 * dpz_dx;
    float gamma_S_zx = Vs * dvs_dz * px - 2.f * Vs2 * Rsz * px + Vs2 * dpx_dz;

    float gamma_P_kk = gamma_P_xx + gamma_P_zz;

    f.sxx[idx] += solver.dt * (gamma_P_kk - 2.f * gamma_S_zz);   // -2 gamma_S_zz, NOT gamma_S_xx
    f.szz[idx] += solver.dt * (gamma_P_kk - 2.f * gamma_S_xx);
    f.sxz[idx] += solver.dt * (gamma_S_xz + gamma_S_zx);

    if (u_this_b) {
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];   // px
        u_this_b[1 * comp_stride + idx] = f.vz[idx];   // pz
    }
}

template<int Order>
__global__ void evr_stress_kernel_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ Rp_x,
    const float* __restrict__ Rp_z,
    const float* __restrict__ Rs_x,
    const float* __restrict__ Rs_z,
    float* __restrict__ u_this,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;

    const float* vp_b   = vp   + b * spatial_size;
    const float* vs_b   = vs   + b * spatial_size;
    const float* Rpx_b  = Rp_x + b * spatial_size;
    const float* Rpz_b  = Rp_z + b * spatial_size;
    const float* Rsx_b  = Rs_x + b * spatial_size;
    const float* Rsz_b  = Rs_z + b * spatial_size;

    float dpx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dpz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(f.vz, ix, 0, iz, grad_ctx);
    float dpx_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dpz_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.vz, ix, 0, iz, grad_ctx);

    float dvp_dx = evr_model_grad_x<Order>(vp_b, ix, iz, grad_ctx);
    float dvp_dz = evr_model_grad_z<Order>(vp_b, ix, iz, grad_ctx);
    float dvs_dx = evr_model_grad_x<Order>(vs_b, ix, iz, grad_ctx);
    float dvs_dz = evr_model_grad_z<Order>(vs_b, ix, iz, grad_ctx);

    float Vp = vp_b[idx], Vs = vs_b[idx];
    float Vp2 = Vp * Vp, Vs2 = Vs * Vs;
    float Rpx = Rpx_b[idx], Rpz = Rpz_b[idx];
    float Rsx = Rsx_b[idx], Rsz = Rsz_b[idx];
    float px = f.vx[idx], pz = f.vz[idx];

    float gamma_P_xx = Vp * dvp_dx * px - 2.f * Vp2 * Rpx * px + Vp2 * dpx_dx;
    float gamma_P_zz = Vp * dvp_dz * pz - 2.f * Vp2 * Rpz * pz + Vp2 * dpz_dz;
    float gamma_S_xx = Vs * dvs_dx * px - 2.f * Vs2 * Rsx * px + Vs2 * dpx_dx;
    float gamma_S_zz = Vs * dvs_dz * pz - 2.f * Vs2 * Rsz * pz + Vs2 * dpz_dz;
    float gamma_S_xz = Vs * dvs_dx * pz - 2.f * Vs2 * Rsx * pz + Vs2 * dpz_dx;
    float gamma_S_zx = Vs * dvs_dz * px - 2.f * Vs2 * Rsz * px + Vs2 * dpx_dz;

    float gamma_P_kk = gamma_P_xx + gamma_P_zz;

    f.sxx[idx] += solver.dt * (gamma_P_kk - 2.f * gamma_S_zz);
    f.szz[idx] += solver.dt * (gamma_P_kk - 2.f * gamma_S_xx);
    f.sxz[idx] += solver.dt * (gamma_S_xz + gamma_S_zx);

    if (u_this_b) {
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vz[idx];
    }
}

// ---------------------------------------------------------------------------
// ADJOINT KERNELS  --  S&S 2025 EVR backward pass
//
// Per-step adjoint splits into 4 kernels using the prepare/apply pattern that
// elastic2d uses (mandatory to avoid the documented "write field at idx,
// then sgradient reads idx+/-k in same launch" CUDA race condition).
//
// Stress adjoint:
//   evr_stress_adjoint_prepare   -- read adjoint sigma_xx/zz/xz, write
//      workspace q* fields (containing constitutive coefficient * adjoint
//      stress + CPML adjoint updates). Also writes additional "pointwise"
//      contribution to adjoint momentum from the gamma multiplicative
//      terms V*(dV)*p_j and -2 V^2 R_*,i p_j.
//   evr_stress_adjoint_apply     -- read q* workspace, apply transpose
//      staggered FD, ADD to adjoint momentum (= adj. px/pz in vx/vz slots).
//
// Momentum adjoint (mirrors elastic_velocity_adjoint_prepare/apply but
// uses inv_rho=1.f since EVR momentum equation has no /rho):
//   evr_momentum_adjoint_prepare -- write workspace p* fields from adj. px/pz
//      + CPML adjoint on f.m_sxxx/sxzz/sxzx/szzz
//   evr_momentum_adjoint_apply   -- transpose FD on p* workspace into adj.
//      sigma_xx/zz/xz.
//
// Gradient: calculate_grad_evr_nobs computes 6 model gradients per step
// (vp, vs, Rp_x, Rp_z, Rs_x, Rs_z). Chain-rule through (dV_P/dx, dV_P/dz,
// dV_S/dx, dV_S/dz) -- the central-FD transpose is the operator negated
// (anti-symmetric kernel).
//
// Constitutive coefficient table (V_P^2, lambda/rho = V_P^2 - 2 V_S^2,
// V_S^2) appears in the same positions as elastic's ((lambda+2mu),
// lambda, mu) with rho=1 -- so the d_i p_j -> adj_p_j path is structurally
// identical to elastic2d adjoint. The EVR-specific additions are
//   - pointwise gamma multiplicative contributions (V_alpha (dV) - 2 V^2 R)
//   - chain rule through cached velocity gradients for grad_vp, grad_vs
//   - no /rho in momentum_adjoint
// ---------------------------------------------------------------------------

template<int Order>
__global__ void evr_stress_adjoint_prepare(
    ElasticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ Rp_x,
    const float* __restrict__ Rp_z,
    const float* __restrict__ Rs_x,
    const float* __restrict__ Rs_z,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qzz,
    float* __restrict__ qxz,
    float* __restrict__ qzx,
    // Per-cell pointwise contribution to adjoint momentum (from V*(dV)*p
    // and -2 V^2 R p terms). Stored in extra workspace tensors so the
    // *apply* kernel can ADD them to adj_px/adj_pz after the FD pass.
    float* __restrict__ point_px,
    float* __restrict__ point_pz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    const float* vp_b  = vp   + b * spatial_size;
    const float* vs_b  = vs   + b * spatial_size;
    const float* Rpx_b = Rp_x + b * spatial_size;
    const float* Rpz_b = Rp_z + b * spatial_size;
    const float* Rsx_b = Rs_x + b * spatial_size;
    const float* Rsz_b = Rs_z + b * spatial_size;

    float* qxx_b = qxx + b * spatial_size;
    float* qzz_b = qzz + b * spatial_size;
    float* qxz_b = qxz + b * spatial_size;
    float* qzx_b = qzx + b * spatial_size;
    float* pt_px_b = point_px + b * spatial_size;
    float* pt_pz_b = point_pz + b * spatial_size;

    float Vp = vp_b[idx];
    float Vs = vs_b[idx];
    float Vp2 = Vp * Vp;
    float Vs2 = Vs * Vs;
    // EVR constitutive coefficients in the sigma update for d_i p_j paths
    // (identical to elastic2d with rho=1):
    //   sigma_xx update on dpx_dx:  +V_P^2          (= lam + 2 mu)
    //   sigma_xx update on dpz_dz:  +V_P^2 - 2 V_S^2 (= lam)
    //   sigma_zz update on dpz_dz:  +V_P^2
    //   sigma_zz update on dpx_dx:  +V_P^2 - 2 V_S^2
    //   sigma_xz update on dpz_dx:  +V_S^2          (= mu)
    //   sigma_xz update on dpx_dz:  +V_S^2
    float lam_plus_2mu = Vp2;          // V_P^2
    float lam          = Vp2 - 2.f * Vs2;
    float mu_          = Vs2;          // V_S^2

    float bar_sxx = f.sxx[idx];
    float bar_szz = f.szz[idx];
    float bar_sxz = f.sxz[idx];

    // bar_d* = dt * (constitutive coeff * adj_sigma) -- mirrors elastic2d
    // pattern with lambda + 2 mu = V_P^2, lambda = V_P^2 - 2 V_S^2, mu = V_S^2.
    float bar_dpx_dx = solver.dt * (lam_plus_2mu * bar_sxx + lam * bar_szz);
    float bar_dpz_dz = solver.dt * (lam_plus_2mu * bar_szz + lam * bar_sxx);
    float bar_dpx_dz = solver.dt * mu_ * bar_sxz;
    float bar_dpz_dx = solver.dt * mu_ * bar_sxz;

    // -- EVR-specific pointwise contributions to adj momentum (gamma's
    // multiplicative V*(dV)*p_j and -2 V^2 R p_j terms).
    //
    // Aggregate gamma adjoints (per cell):
    //   L_gamma_P_xx = dt (bar_sxx + bar_szz)
    //   L_gamma_P_zz = dt (bar_sxx + bar_szz)
    //   L_gamma_S_xx = -2 dt bar_szz
    //   L_gamma_S_zz = -2 dt bar_sxx
    //   L_gamma_S_xz = dt bar_sxz
    //   L_gamma_S_zx = dt bar_sxz
    float L_gP_xx = solver.dt * (bar_sxx + bar_szz);
    float L_gP_zz = L_gP_xx;
    float L_gS_xx = -2.f * solver.dt * bar_szz;
    float L_gS_zz = -2.f * solver.dt * bar_sxx;
    float L_gS_xz = solver.dt * bar_sxz;
    float L_gS_zx = solver.dt * bar_sxz;

    // The pointwise momentum contribution involves centered velocity
    // gradients (neighbour reads of vp/vs). The forward stress kernel only
    // applies the gamma terms at interior cells (ix,iz in [halo, n-halo)),
    // so the adjoint contribution is non-zero only there too -- and
    // restricting the reads to the interior keeps the +/-(Order/2) stencil
    // in bounds at the grid edge.
    bool interior = (ix >= halo && ix < solver.nx - halo &&
                     iz >= halo && iz < solver.nz - halo);
    if (interior) {
        float dvp_dx = evr_model_grad_x<Order>(vp_b, ix, iz, grad_ctx);
        float dvp_dz = evr_model_grad_z<Order>(vp_b, ix, iz, grad_ctx);
        float dvs_dx = evr_model_grad_x<Order>(vs_b, ix, iz, grad_ctx);
        float dvs_dz = evr_model_grad_z<Order>(vs_b, ix, iz, grad_ctx);

        float Rpx = Rpx_b[idx], Rpz = Rpz_b[idx];
        float Rsx = Rsx_b[idx], Rsz = Rsz_b[idx];

        // Multiplicative-term coefficients to p_j:
        //   factor = V_alpha (dV_alpha/d_i) - 2 V_alpha^2 R_alpha,i
        float factor_P_x = Vp * dvp_dx - 2.f * Vp2 * Rpx;
        float factor_P_z = Vp * dvp_dz - 2.f * Vp2 * Rpz;
        float factor_S_x = Vs * dvs_dx - 2.f * Vs2 * Rsx;
        float factor_S_z = Vs * dvs_dz - 2.f * Vs2 * Rsz;

        // adj_px (j=x in gamma_P_xx, gamma_S_xx, gamma_S_zx):
        pt_px_b[idx] = L_gP_xx * factor_P_x
                     + L_gS_xx * factor_S_x
                     + L_gS_zx * factor_S_z;
        // adj_pz (j=z in gamma_P_zz, gamma_S_zz, gamma_S_xz):
        pt_pz_b[idx] = L_gP_zz * factor_P_z
                     + L_gS_zz * factor_S_z
                     + L_gS_xz * factor_S_x;
    } else {
        pt_px_b[idx] = 0.f;
        pt_pz_b[idx] = 0.f;
    }

    // Position-based PML / interior split. Outside the PML band all
    // ax/az/bx/bz coefficients vanish, m_v* aux fields stay 0 -> 0.
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        qxx_b[idx] = bar_dpx_dx;
        qzz_b[idx] = bar_dpz_dz;
        qxz_b[idx] = bar_dpx_dz;
        qzx_b[idx] = bar_dpz_dx;
        return;
    }

    float az = cpml.az[iz], bz = cpml.bz[iz];
    float azh = cpml.azh[iz], bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix], bx = cpml.bx[ix];
    float axh = cpml.axh[ix], bxh = cpml.bxh[ix];

    // CPML adjoint on the momentum-derivative memvars (same coefficient
    // pairing as the forward stress kernel uses):
    //   forward: m_pxx <- ax*m_pxx + bx*dpx_dx ; dpx_dx <- dpx_dx + m_pxx
    //   adjoint: tmp = m_adj_prev + bar_dpx_dx;  q = bar_dpx_dx + bx*tmp; m_adj_new = ax*tmp
    float tmp_pxx = f.m_vxx[idx] + bar_dpx_dx;
    float tmp_pzz = f.m_vzz[idx] + bar_dpz_dz;
    float tmp_pxz = f.m_vxz[idx] + bar_dpx_dz;
    float tmp_pzx = f.m_vzx[idx] + bar_dpz_dx;

    qxx_b[idx] = bar_dpx_dx + bx  * tmp_pxx;
    qzz_b[idx] = bar_dpz_dz + bz  * tmp_pzz;
    qxz_b[idx] = bar_dpx_dz + bzh * tmp_pxz;
    qzx_b[idx] = bar_dpz_dx + bxh * tmp_pzx;

    f.m_vxx[idx] = ax  * tmp_pxx;
    f.m_vzz[idx] = az  * tmp_pzz;
    f.m_vxz[idx] = azh * tmp_pxz;
    f.m_vzx[idx] = axh * tmp_pzx;
}

template<int Order>
__global__ void evr_stress_adjoint_apply(
    ElasticWavefieldPointer wf,
    const float* __restrict__ qxx,
    const float* __restrict__ qzz,
    const float* __restrict__ qxz,
    const float* __restrict__ qzx,
    const float* __restrict__ point_px,
    const float* __restrict__ point_pz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    auto f = wf.offset(b, spatial_size);

    const float* qxx_b = qxx + b * spatial_size;
    const float* qzz_b = qzz + b * spatial_size;
    const float* qxz_b = qxz + b * spatial_size;
    const float* qzx_b = qzx + b * spatial_size;
    const float* pt_px_b = point_px + b * spatial_size;
    const float* pt_pz_b = point_pz + b * spatial_size;

    // EXACT transpose of the forward staggered FD. The forward stress kernel
    // formed the gamma divergence terms from
    //   dpx_dx = sgrad<X,BACKWARD>(px),  dpz_dz = sgrad<Z,BACKWARD>(pz),
    //   dpx_dz = sgrad<Z,FORWARD>(px),   dpz_dx = sgrad<X,FORWARD>(pz).
    // The adjoint applies the transpose of each. For the staggered operators
    // here  D_b^T = -D_f  and  D_f^T = -D_b  (verified against autograd), so
    // the transpose is the OPPOSITE-direction sgradient NEGATED. q* hold the
    // modulus-weighted adjoint stress (qxx<-dpx_dx, qzz<-dpz_dz,
    // qxz<-dpx_dz, qzx<-dpz_dx).
    //   a_px gets dpx_dx (->-sgrad<X,FORWARD>) and dpx_dz (->-sgrad<Z,BACKWARD>)
    //   a_pz gets dpz_dz (->-sgrad<Z,FORWARD>) and dpz_dx (->-sgrad<X,BACKWARD>)
    float dqxx_dx = sgradient<2, Order, X, DIFF_FORWARD>(qxx_b, ix, 0, iz, grad_ctx);
    float dqxz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(qxz_b, ix, 0, iz, grad_ctx);
    float dqzz_dz = sgradient<2, Order, Z, DIFF_FORWARD>(qzz_b, ix, 0, iz, grad_ctx);
    float dqzx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(qzx_b, ix, 0, iz, grad_ctx);

    int idx = iz * solver.nx + ix;
    f.vx[idx] += -(dqxx_dx + dqxz_dz) + pt_px_b[idx];   // exact transpose + pointwise
    f.vz[idx] += -(dqzz_dz + dqzx_dx) + pt_pz_b[idx];
}

template<int Order>
__global__ void evr_momentum_adjoint_prepare(
    ElasticWavefieldPointer wf,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ pxx,
    float* __restrict__ pzz,
    float* __restrict__ pxz,
    float* __restrict__ pzx
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    float* pxx_b = pxx + b * spatial_size;
    float* pzz_b = pzz + b * spatial_size;
    float* pxz_b = pxz + b * spatial_size;
    float* pzx_b = pzx + b * spatial_size;

    // EVR momentum equation has no /rho, so the adjoint multiplier is 1.f
    // (compared to inv_rho = 1/rho in elastic2d).
    float inv_rho = 1.f;
    float bar_dsxx_dx = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dz = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dx = solver.dt * inv_rho * f.vz[idx];
    float bar_dszz_dz = solver.dt * inv_rho * f.vz[idx];

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);
    if (!in_pml) {
        pxx_b[idx] = bar_dsxx_dx;
        pxz_b[idx] = bar_dsxz_dz;
        pzx_b[idx] = bar_dsxz_dx;
        pzz_b[idx] = bar_dszz_dz;
        return;
    }

    float az = cpml.az[iz], bz = cpml.bz[iz];
    float azh = cpml.azh[iz], bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix], bx = cpml.bx[ix];
    float axh = cpml.axh[ix], bxh = cpml.bxh[ix];

    // CPML adjoint on stress-derivative memvars (forward kernel pairing:
    //   m_sxxx with axh/bxh; m_szzz with azh/bzh;
    //   m_sxzz with az/bz;   m_sxzx with ax/bx).
    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    pxx_b[idx] = bar_dsxx_dx + bxh * tmp_sxxx;
    pxz_b[idx] = bar_dsxz_dz + bz  * tmp_sxzz;
    pzx_b[idx] = bar_dsxz_dx + bx  * tmp_sxzx;
    pzz_b[idx] = bar_dszz_dz + bzh * tmp_szzz;

    f.m_sxxx[idx] = axh * tmp_sxxx;
    f.m_sxzz[idx] = az  * tmp_sxzz;
    f.m_sxzx[idx] = ax  * tmp_sxzx;
    f.m_szzz[idx] = azh * tmp_szzz;
}

template<int Order>
__global__ void evr_momentum_adjoint_apply(
    ElasticWavefieldPointer wf,
    const float* __restrict__ pxx,
    const float* __restrict__ pzz,
    const float* __restrict__ pxz,
    const float* __restrict__ pzx,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    auto f = wf.offset(b, spatial_size);

    const float* pxx_b = pxx + b * spatial_size;
    const float* pzz_b = pzz + b * spatial_size;
    const float* pxz_b = pxz + b * spatial_size;
    const float* pzx_b = pzx + b * spatial_size;

    // EXACT transpose of the forward momentum update. The forward momentum
    // kernel used
    //   px += dt*(sgrad<X,FORWARD>(sxx) + sgrad<Z,BACKWARD>(sxz))
    //   pz += dt*(sgrad<X,BACKWARD>(sxz) + sgrad<Z,FORWARD>(szz))
    // so the transposes are the OPPOSITE-direction sgradient NEGATED
    // (D_f^T=-D_b, D_b^T=-D_f). p* hold dt*adjoint-momentum
    // (pxx<-dsxx_dx[F-x], pzz<-dszz_dz[F-z], pxz<-dsxz_dz[B-z], pzx<-dsxz_dx[B-x]).
    float dpxx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(pxx_b, ix, 0, iz, grad_ctx);  // (F-x)^T
    float dpzz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(pzz_b, ix, 0, iz, grad_ctx);  // (F-z)^T
    float dpxz_dz = sgradient<2, Order, Z, DIFF_FORWARD>(pxz_b, ix, 0, iz, grad_ctx);   // (B-z)^T
    float dpzx_dx = sgradient<2, Order, X, DIFF_FORWARD>(pzx_b, ix, 0, iz, grad_ctx);   // (B-x)^T

    int idx = iz * solver.nx + ix;
    f.sxx[idx] += -dpxx_dx;
    f.szz[idx] += -dpzz_dz;
    f.sxz[idx] += -(dpxz_dz + dpzx_dx);
}

// Gradient kernel. Two-pass design (still inside one __global__ launch but
// the chain-rule-through-central-FD requires reads of NEIGHBOURING workspace
// values written by other threads in the SAME kernel ON OTHER CELLS -- which
// is a read-write race per the documented CUDA pitfall). To stay safe, we
// emit a SEPARATE preparation kernel that just builds the per-cell
// L_(dV_*/d*) source tensors, then the gradient kernel reads neighbours
// via central FD transpose. See *_grad_prepare followed by
// *_grad_chain_apply below.
//
// For full mode (this kernel), we accept the per-cell direct grads:
//   - grad_R_*: pure pointwise, no chain rule
//   - grad_V_P, grad_V_S: pointwise contribution only
// The chain-rule contribution through (dV/dx, dV/dz) is added by a
// follow-up kernel.

template<int Order>
__global__ void calculate_grad_evr_nobs(
    ElasticWavefieldPointer adjoint,
    const float* __restrict__ fpx,
    const float* __restrict__ fpz,
    const float* __restrict__ fpx_prev,
    const float* __restrict__ fpz_prev,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ Rp_x,
    const float* __restrict__ Rp_z,
    const float* __restrict__ Rs_x,
    const float* __restrict__ Rs_z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_vs,
    float* __restrict__ grad_Rp_x,
    float* __restrict__ grad_Rp_z,
    float* __restrict__ grad_Rs_x,
    float* __restrict__ grad_Rs_z,
    // Chain-rule source tensors -- this kernel WRITES them; a subsequent
    // kernel reads neighbours via transpose central FD into grad_vp/grad_vs.
    float* __restrict__ L_dVp_dx,
    float* __restrict__ L_dVp_dz,
    float* __restrict__ L_dVs_dx,
    float* __restrict__ L_dVs_dz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* fpx_b = fpx + b * spatial_size;
    const float* fpz_b = fpz + b * spatial_size;

    auto a = adjoint.offset(b, spatial_size);

    const float* vp_b = vp + b * spatial_size;
    const float* vs_b = vs + b * spatial_size;
    const float* Rpx_b = Rp_x + b * spatial_size;
    const float* Rpz_b = Rp_z + b * spatial_size;
    const float* Rsx_b = Rs_x + b * spatial_size;
    const float* Rsz_b = Rs_z + b * spatial_size;

    float* gvp_b = grad_vp + b * spatial_size;
    float* gvs_b = grad_vs + b * spatial_size;
    float* gRpx_b = grad_Rp_x + b * spatial_size;
    float* gRpz_b = grad_Rp_z + b * spatial_size;
    float* gRsx_b = grad_Rs_x + b * spatial_size;
    float* gRsz_b = grad_Rs_z + b * spatial_size;
    float* lvpx_b = L_dVp_dx + b * spatial_size;
    float* lvpz_b = L_dVp_dz + b * spatial_size;
    float* lvsx_b = L_dVs_dx + b * spatial_size;
    float* lvsz_b = L_dVs_dz + b * spatial_size;

    // Re-compute the forward d_i p_j derivatives (from the STORED forward
    // momentum at this step) -- the stored values fpx/fpz are the momentum
    // AFTER the momentum step but BEFORE the stress step at time t (which is
    // what the gamma terms in the forward stress update consumed).
    float fpx_x = sgradient<2, Order, X, DIFF_BACKWARD>(fpx_b, ix, 0, iz, grad_ctx);
    float fpz_z = sgradient<2, Order, Z, DIFF_BACKWARD>(fpz_b, ix, 0, iz, grad_ctx);
    float fpx_z = sgradient<2, Order, Z, DIFF_FORWARD>(fpx_b, ix, 0, iz, grad_ctx);
    float fpz_x = sgradient<2, Order, X, DIFF_FORWARD>(fpz_b, ix, 0, iz, grad_ctx);

    // Forward velocity gradients (4th-order central) -- needed for the
    // V*(dV/d_i)*p_j adjoint contribution to grad_vp / grad_vs.
    float dvp_dx = evr_model_grad_x<Order>(vp_b, ix, iz, grad_ctx);
    float dvp_dz = evr_model_grad_z<Order>(vp_b, ix, iz, grad_ctx);
    float dvs_dx = evr_model_grad_x<Order>(vs_b, ix, iz, grad_ctx);
    float dvs_dz = evr_model_grad_z<Order>(vs_b, ix, iz, grad_ctx);

    float Vp = vp_b[idx], Vs = vs_b[idx];
    float Vp2 = Vp * Vp, Vs2 = Vs * Vs;
    float Rpx = Rpx_b[idx], Rpz = Rpz_b[idx];
    float Rsx = Rsx_b[idx], Rsz = Rsz_b[idx];
    float px = fpx_b[idx], pz = fpz_b[idx];

    // Aggregate gamma adjoints
    float L_gP_xx = solver.dt * (a.sxx[idx] + a.szz[idx]);
    float L_gP_zz = L_gP_xx;
    float L_gS_xx = -2.f * solver.dt * a.szz[idx];
    float L_gS_zz = -2.f * solver.dt * a.sxx[idx];
    float L_gS_xz = solver.dt * a.sxz[idx];
    float L_gS_zx = solver.dt * a.sxz[idx];

    // grad_V_alpha direct (pointwise) contributions. Each gamma_alpha_ij has
    //   d gamma / d V_alpha = (dV/d_i) p_j - 4 V R_alpha,i p_j + 2 V (d_i p_j)
    // Sum over alpha=P contributions for grad_vp; alpha=S contributions for grad_vs.
    float dgP_xx_dVp = dvp_dx * px - 4.f * Vp * Rpx * px + 2.f * Vp * fpx_x;
    float dgP_zz_dVp = dvp_dz * pz - 4.f * Vp * Rpz * pz + 2.f * Vp * fpz_z;
    float dgS_xx_dVs = dvs_dx * px - 4.f * Vs * Rsx * px + 2.f * Vs * fpx_x;
    float dgS_zz_dVs = dvs_dz * pz - 4.f * Vs * Rsz * pz + 2.f * Vs * fpz_z;
    float dgS_xz_dVs = dvs_dx * pz - 4.f * Vs * Rsx * pz + 2.f * Vs * fpz_x;
    float dgS_zx_dVs = dvs_dz * px - 4.f * Vs * Rsz * px + 2.f * Vs * fpx_z;

    // Model gradients. With the EXACT-transpose adjoint above, a.sxx/szz/sxz
    // carry +dL/dsigma (the true adjoint stress), and L_g* = dt*(adjoint-stress
    // combination). So grad_param += L_g * d(gamma)/d(param) directly -- the
    // -2 V^2 etc. signs come from the gamma derivatives themselves, NOT from a
    // global flip. Verified against eager autograd / FD (cos -> 1.0).
    gvp_b[idx] += L_gP_xx * dgP_xx_dVp + L_gP_zz * dgP_zz_dVp;
    gvs_b[idx] += L_gS_xx * dgS_xx_dVs + L_gS_zz * dgS_zz_dVs
                + L_gS_xz * dgS_xz_dVs + L_gS_zx * dgS_zx_dVs;

    // grad_R_alpha,i contributions  ( d gamma / d R_alpha,i = -2 V_alpha^2 p_j )
    gRpx_b[idx] += L_gP_xx * (-2.f * Vp2 * px);
    gRpz_b[idx] += L_gP_zz * (-2.f * Vp2 * pz);
    gRsx_b[idx] += L_gS_xx * (-2.f * Vs2 * px) + L_gS_xz * (-2.f * Vs2 * pz);
    gRsz_b[idx] += L_gS_zz * (-2.f * Vs2 * pz) + L_gS_zx * (-2.f * Vs2 * px);

    // Chain-rule source tensors for grad_vp/grad_vs through (d_i V) in the
    // V*(d_i V)*p_j gamma term. lvpx = L_g * V * p_j is the adjoint source for
    // (d_i V); evr_grad_chain_apply applies the central-FD transpose (= -A,
    // anti-symmetric) and adds to grad_vp/grad_vs.
    lvpx_b[idx] = L_gP_xx * Vp * px;
    lvpz_b[idx] = L_gP_zz * Vp * pz;
    lvsx_b[idx] = L_gS_xx * Vs * px + L_gS_xz * Vs * pz;
    lvsz_b[idx] = L_gS_zz * Vs * pz + L_gS_zx * Vs * px;
}

#define LAUNCH_EVR_GRAD_CHAIN_APPLY(order, grid, block, ...) \
    do { \
        if      ((order) == 2) elastic_vr2d_kernels::evr_grad_chain_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_vr2d_kernels::evr_grad_chain_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_vr2d_kernels::evr_grad_chain_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_vr2d_kernels::evr_grad_chain_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_vr2d_kernels::evr_grad_chain_apply<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template<int Order>
__global__ void evr_grad_chain_apply(
    const float* __restrict__ L_dVp_dx,
    const float* __restrict__ L_dVp_dz,
    const float* __restrict__ L_dVs_dx,
    const float* __restrict__ L_dVs_dz,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_vs,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    // Transpose of the order-Order centered FD reaches Order/2 cells (or 2
    // for the runtime fallback), so guard that many cells from each edge.
    constexpr bool is_runtime = (Order == -1);
    int halo = is_runtime ? 2 : (Order / 2);
    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* lvpx_b = L_dVp_dx + b * spatial_size;
    const float* lvpz_b = L_dVp_dz + b * spatial_size;
    const float* lvsx_b = L_dVs_dx + b * spatial_size;
    const float* lvsz_b = L_dVs_dz + b * spatial_size;
    float* gvp_b = grad_vp + b * spatial_size;
    float* gvs_b = grad_vs + b * spatial_size;

    // Central FD is anti-symmetric: A^T = -A. So
    //   grad_vp += -central_dx(L_(dV_P/dx)) - central_dz(L_(dV_P/dz))
    // Use the SAME canonical centered stencil (same order) as the forward
    // model-gradient pass so the discrete transpose is exact.
    float ext_vp_dx = evr_model_grad_x<Order>(lvpx_b, ix, iz, grad_ctx);
    float ext_vp_dz = evr_model_grad_z<Order>(lvpz_b, ix, iz, grad_ctx);
    float ext_vs_dx = evr_model_grad_x<Order>(lvsx_b, ix, iz, grad_ctx);
    float ext_vs_dz = evr_model_grad_z<Order>(lvsz_b, ix, iz, grad_ctx);

    gvp_b[idx] += -(ext_vp_dx + ext_vp_dz);
    gvs_b[idx] += -(ext_vs_dx + ext_vs_dz);
}

}  // namespace elastic_vr2d_kernels
