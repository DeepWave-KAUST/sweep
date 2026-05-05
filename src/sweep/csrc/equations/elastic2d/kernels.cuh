#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/staggered.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"

#define LAUNCH_ELASTIC_VELOCITY(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_ELASTIC_STRESS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_NOPML(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_NOPML(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC_BS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) calculate_grad_elastic_bs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_bs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_bs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_bs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_bs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) calculate_grad_elastic_nobs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_nobs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_nobs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_nobs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_nobs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_ADJOINT_PREPARE(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_ADJOINT_APPLY(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_ADJOINT_APPLY(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


template<int Order>
__global__ void elastic_velocity_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,

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

    const float* rho_b = rho + b * spatial_size;

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float dsxx_dx = sgradient<2, Order, X, DIFF_FORWARD> (f.sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.szz, ix, iz, grad_ctx, solver, true);

    float inv_rho = 1.f / rho_b[idx];

    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];
    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];

    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];
    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];

    f.vx[idx] += solver.dt * inv_rho *
        (dsxx_dx + dsxz_dz);

    f.vz[idx] += solver.dt * inv_rho *
        (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void elastic_stress_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
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

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD> (f.vx, ix, 0, iz, grad_ctx);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = sgradient<2, Order, X, DIFF_FORWARD>  (f.vz, ix, 0, iz, grad_ctx);

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz += f.m_vzz[idx];
    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx += f.m_vxx[idx];

    f.sxx[idx] += solver.dt *
        ((lam + 2.f*mu_) * dvx_dx +
         lam * dvz_dz);

    f.szz[idx] += solver.dt *
        ((lam + 2.f*mu_) * dvz_dz +
         lam * dvx_dx);

    f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
    dvx_dz += f.m_vxz[idx];
    f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
    dvz_dx += f.m_vzx[idx];

    f.sxz[idx] += solver.dt *
        mu_ * (dvx_dz + dvz_dx);

    if (elastic_is_top_free_surface_row(solver, iz)) {
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
    }

    if (u_this_b) {
        int comp_stride  = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vz[idx];
    }

}

template<int Order>
__global__ void elastic_velocity_kernel_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn + M+1;

    int top_halo = solver.free_surface ? M: halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    // Wavefields
    auto f = wf.offset(b, spatial_size);

    const float* rho_b = rho + b * spatial_size;

    float dsxx_dx = sgradient<2, Order, X, DIFF_FORWARD> (f.sxx, ix, 0, iz, grad_ctx);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.sxz, ix, 0, iz, grad_ctx);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.szz, ix, iz, grad_ctx, solver, true);

    float inv_rho = 1.f / rho_b[idx];

    f.vx[idx] -= solver.dt * inv_rho *
        (dsxx_dx + dsxz_dz);

    f.vz[idx] -= solver.dt * inv_rho *
        (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void elastic_stress_kernel_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,

    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn + M+1;

    int top_halo = solver.free_surface ? M: halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    // Wavefields
    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;

    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD> (f.vx, ix, 0, iz, grad_ctx);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = sgradient<2, Order, X, DIFF_FORWARD>  (f.vz, ix, 0, iz, grad_ctx);

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    f.sxx[idx] -= solver.dt *
        ((lam + 2.f*mu_) * dvx_dx +
         lam * dvz_dz);

    f.szz[idx] -= solver.dt *
        ((lam + 2.f*mu_) * dvz_dz +
         lam * dvx_dx);

    f.sxz[idx] -= solver.dt *
        mu_ * (dvx_dz + dvz_dx);
}

template<int Order>
__global__ void elastic_stress_adjoint_prepare(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qzz,
    float* __restrict__ qxz,
    float* __restrict__ qzx
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;
    float* qxx_b = qxx + b * spatial_size;
    float* qzz_b = qzz + b * spatial_size;
    float* qxz_b = qxz + b * spatial_size;
    float* qzx_b = qzx + b * spatial_size;

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    float bar_sxx = f.sxx[idx];
    float bar_szz = f.szz[idx];
    float bar_sxz = f.sxz[idx];
    if (elastic_is_top_free_surface_row(solver, iz)) {
        bar_szz = 0.f;
        bar_sxz = 0.f;
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
    }

    float bar_dvx_dx = solver.dt * ((lam + 2.f * mu_) * bar_sxx + lam * bar_szz);
    float bar_dvz_dz = solver.dt * ((lam + 2.f * mu_) * bar_szz + lam * bar_sxx);
    float bar_dvx_dz = solver.dt * mu_ * bar_sxz;
    float bar_dvz_dx = solver.dt * mu_ * bar_sxz;

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;

    qxx_b[idx] = bar_dvx_dx + bx * tmp_vxx;
    qzz_b[idx] = bar_dvz_dz + bz * tmp_vzz;
    qxz_b[idx] = bar_dvx_dz + bzh * tmp_vxz;
    qzx_b[idx] = bar_dvz_dx + bxh * tmp_vzx;

    f.m_vxx[idx] = ax * tmp_vxx;
    f.m_vzz[idx] = az * tmp_vzz;
    f.m_vxz[idx] = azh * tmp_vxz;
    f.m_vzx[idx] = axh * tmp_vzx;
}

template<int Order>
__global__ void elastic_stress_adjoint_apply(
    ElasticWavefieldPointer wf,
    const float* __restrict__ qxx,
    const float* __restrict__ qzz,
    const float* __restrict__ qxz,
    const float* __restrict__ qzx,
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

    float dqxx_dx = sgradient<2, Order, X, DIFF_FORWARD>(qxx_b, ix, 0, iz, grad_ctx);
    float dqxz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_FORWARD> (qxz_b, ix, iz, grad_ctx, solver, false);
    float dqzx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(qzx_b, ix, 0, iz, grad_ctx);
    float dqzz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_BACKWARD>(qzz_b, ix, iz, grad_ctx, solver, true);

    int idx = iz * solver.nx + ix;
    f.vx[idx] += dqxx_dx + dqxz_dz;
    f.vz[idx] += dqzx_dx + dqzz_dz;
}

template<int Order>
__global__ void elastic_velocity_adjoint_prepare(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    const float* rho_b = rho + b * spatial_size;
    float* pxx_b = pxx + b * spatial_size;
    float* pzz_b = pzz + b * spatial_size;
    float* pxz_b = pxz + b * spatial_size;
    float* pzx_b = pzx + b * spatial_size;

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float inv_rho = 1.f / rho_b[idx];
    float bar_dsxx_dx = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dz = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dx = solver.dt * inv_rho * f.vz[idx];
    float bar_dszz_dz = solver.dt * inv_rho * f.vz[idx];

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    pxx_b[idx] = bar_dsxx_dx + bxh * tmp_sxxx;
    pxz_b[idx] = bar_dsxz_dz + bz * tmp_sxzz;
    pzx_b[idx] = bar_dsxz_dx + bx * tmp_sxzx;
    pzz_b[idx] = bar_dszz_dz + bzh * tmp_szzz;

    f.m_sxxx[idx] = axh * tmp_sxxx;
    f.m_sxzz[idx] = az * tmp_sxzz;
    f.m_sxzx[idx] = ax * tmp_sxzx;
    f.m_szzz[idx] = azh * tmp_szzz;
}

template<int Order>
__global__ void elastic_velocity_adjoint_apply(
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

    float dpxx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(pxx_b, ix, 0, iz, grad_ctx);
    float dpzz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_FORWARD> (pzz_b, ix, iz, grad_ctx, solver, true);
    float dpxz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_BACKWARD>(pxz_b, ix, iz, grad_ctx, solver, true);
    float dpzx_dx = sgradient<2, Order, X, DIFF_FORWARD>(pzx_b, ix, 0, iz, grad_ctx);

    int idx = iz * solver.nx + ix;
    f.sxx[idx] += dpxx_dx;
    f.szz[idx] += dpzz_dz;
    f.sxz[idx] += dpxz_dz + dpzx_dx;
}


template<int Order>
__global__ void calculate_grad_elastic_bs(

    ElasticWavefieldPointer forward,
    ElasticWavefieldPointer adjoint,

    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvz_prev,

    const float* __restrict__ vp,        // (B, nz, nx)
    const float* __restrict__ vs,        // (B, nz, nx)
    const float* __restrict__ rho,       // (B, nz, nx)

    float* __restrict__ grad_vp,             // (B, nz, nx)
    float* __restrict__ grad_vs,             // (B, nz, nx)
    float* __restrict__ grad_rho,            // (B, nz, nx)

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

    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;

    auto f = forward.offset(b, spatial_size);
    auto a = adjoint.offset(b, spatial_size);

    const float* vp_b = vp + b * spatial_size;
    const float* vs_b = vs + b * spatial_size;
    const float* rho_b = rho + b * spatial_size;

    float* gvp = grad_vp + b * spatial_size;
    float* gvs = grad_vs + b * spatial_size;
    float* grho = grad_rho + b * spatial_size;

    float fvx_x = sgradient<2, Order, X, DIFF_BACKWARD> (f.vx, ix, 0, iz, grad_ctx);
    float fvz_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float fvx_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float fvz_x = sgradient<2, Order, X, DIFF_FORWARD> (f.vz, ix, 0, iz, grad_ctx);

    float bar_szz = elastic_is_top_free_surface_row(solver, iz) ? 0.f : a.szz[idx];
    float bar_sxz = elastic_is_top_free_surface_row(solver, iz) ? 0.f : a.sxz[idx];
    float grad_lambda = (a.sxx[idx] + bar_szz) * (fvx_x + fvz_z);
    float grad_mu = 2*(a.sxx[idx] * fvx_x + bar_szz * fvz_z) + bar_sxz * (fvx_z + fvz_x);
    
    gvp[idx] +=   -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    gvs[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                   2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

    grho[idx] += (a.vx[idx] * (f.vx[idx]-fvx_prev_b[idx]) + 
                  a.vz[idx] * (f.vz[idx]-fvz_prev_b[idx])) / rho_b[idx];
    grho[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                 grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;
}

template<int Order>
__global__ void calculate_grad_elastic_nobs(

    ElasticWavefieldPointer adjoint,

    const float* __restrict__ fvx,
    const float* __restrict__ fvz,

    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvz_prev,

    const float* __restrict__ vp,        // (B, nz, nx)
    const float* __restrict__ vs,        // (B, nz, nx)
    const float* __restrict__ rho,       // (B, nz, nx)

    float* __restrict__ grad_vp,         // (B, nz, nx)
    float* __restrict__ grad_vs,         // (B, nz, nx)
    float* __restrict__ grad_rho,         // (B, nz, nx)

    SGradParam grad_ctx,
    SolverContext solver
)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz)
        return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* fvx_b = fvx + b * spatial_size;
    const float* fvz_b = fvz + b * spatial_size;
    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;

    float*       grad_vp_b       = grad_vp       + b * spatial_size;
    float*       grad_vs_b       = grad_vs       + b * spatial_size;
    float*       grad_rho_b      = grad_rho      + b * spatial_size;

    const float* vp_b         = vp         + b * spatial_size;
    const float* vs_b         = vs         + b * spatial_size;
    const float* rho_b        = rho        + b * spatial_size;

    auto a = adjoint.offset(b, spatial_size);

    float fvx_x = sgradient<2, Order, X, DIFF_BACKWARD> (fvx_b, ix, 0, iz, grad_ctx);
    float fvz_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(fvz_b, ix, iz, grad_ctx, solver, true);
    float fvx_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (fvx_b, ix, iz, grad_ctx, solver, false);
    float fvz_x = sgradient<2, Order, X, DIFF_FORWARD>  (fvz_b, ix, 0, iz, grad_ctx);

    float bar_szz = elastic_is_top_free_surface_row(solver, iz) ? 0.f : a.szz[idx];
    float bar_sxz = elastic_is_top_free_surface_row(solver, iz) ? 0.f : a.sxz[idx];
    float grad_lambda = (a.sxx[idx] + bar_szz) * (fvx_x + fvz_z);
    float grad_mu = 2*(a.sxx[idx] * fvx_x + bar_szz * fvz_z) + bar_sxz * (fvx_z + fvz_x);
    
    grad_vp_b[idx] += -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    grad_vs_b[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                         2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

    grad_rho_b[idx] += (a.vx[idx] * (fvx_b[idx]-fvx_prev_b[idx]) + 
                        a.vz[idx] * (fvz_b[idx]-fvz_prev_b[idx])) / rho_b[idx];
    grad_rho_b[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                       grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;

}
