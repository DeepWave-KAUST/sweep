#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/context.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../operators/dim.cuh"
#include "../../operators/staggered.cuh"

#define LAUNCH_DAS2D_FIRST(order, grid, block, ...)                           \
    do {                                                                      \
        if      ((order) == 2) das2d_first_derivatives_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das2d_first_derivatives_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das2d_first_derivatives_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das2d_first_derivatives_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das2d_first_derivatives_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS2D_SECOND(order, grid, block, ...)                          \
    do {                                                                      \
        if      ((order) == 2) das2d_update_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das2d_update_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das2d_update_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das2d_update_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das2d_update_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template<int Order>
__global__ void das2d_first_derivatives_kernel(
    DasWavefieldPointer2D wf,
    float* __restrict__ tmp_sxx_x,
    float* __restrict__ tmp_szz_z,
    float* __restrict__ tmp_txx_z,
    float* __restrict__ tmp_tzz_x,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz || b >= solver.B) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo) {
        return;
    }

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    float* tmp_sxx_x_b = tmp_sxx_x + b * spatial_size;
    float* tmp_szz_z_b = tmp_szz_z + b * spatial_size;
    float* tmp_txx_z_b = tmp_txx_z + b * spatial_size;
    float* tmp_tzz_x_b = tmp_tzz_x + b * spatial_size;

    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float dsxx_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.sxx, ix, 0, iz, grad_ctx);
    f.m_sxx_xf[idx] = axh * f.m_sxx_xf[idx] + bxh * dsxx_dx;
    tmp_sxx_x_b[idx] = dsxx_dx + f.m_sxx_xf[idx];

    float dszz_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.szz, ix, 0, iz, grad_ctx);
    f.m_szz_zf[idx] = azh * f.m_szz_zf[idx] + bzh * dszz_dz;
    tmp_szz_z_b[idx] = dszz_dz + f.m_szz_zf[idx];

    float dtxx_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.txx, ix, 0, iz, grad_ctx);
    f.m_txx_zf[idx] = azh * f.m_txx_zf[idx] + bzh * dtxx_dz;
    tmp_txx_z_b[idx] = dtxx_dz + f.m_txx_zf[idx];

    float dtzz_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.tzz, ix, 0, iz, grad_ctx);
    f.m_tzz_xf[idx] = axh * f.m_tzz_xf[idx] + bxh * dtzz_dx;
    tmp_tzz_x_b[idx] = dtzz_dx + f.m_tzz_xf[idx];
}

template<int Order>
__global__ void das2d_update_kernel(
    DasWavefieldPointer2D wf,
    const float* __restrict__ tmp_sxx_x,
    const float* __restrict__ tmp_szz_z,
    const float* __restrict__ tmp_txx_z,
    const float* __restrict__ tmp_tzz_x,
    const float* __restrict__ rho,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz || b >= solver.B) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    int update_halo = 2 * halo;

    if (ix < update_halo || ix >= solver.nx - update_halo ||
        iz < update_halo || iz >= solver.nz - update_halo) {
        return;
    }

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    const float* tmp_sxx_x_b = tmp_sxx_x + b * spatial_size;
    const float* tmp_szz_z_b = tmp_szz_z + b * spatial_size;
    const float* tmp_txx_z_b = tmp_txx_z + b * spatial_size;
    const float* tmp_tzz_x_b = tmp_tzz_x + b * spatial_size;

    const float* rho_b = rho + b * spatial_size;
    const float* lambda_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float az = cpml.az[iz];
    float bz = cpml.bz[iz];

    float dxx_sxx = sgradient<2, Order, X, DIFF_BACKWARD>(tmp_sxx_x_b, ix, 0, iz, grad_ctx);
    f.m_sxx_xb[idx] = ax * f.m_sxx_xb[idx] + bx * dxx_sxx;
    dxx_sxx += f.m_sxx_xb[idx];

    float dzz_szz = sgradient<2, Order, Z, DIFF_BACKWARD>(tmp_szz_z_b, ix, 0, iz, grad_ctx);
    f.m_szz_zb[idx] = az * f.m_szz_zb[idx] + bz * dzz_szz;
    dzz_szz += f.m_szz_zb[idx];

    float dzz_txx = sgradient<2, Order, Z, DIFF_BACKWARD>(tmp_txx_z_b, ix, 0, iz, grad_ctx);
    f.m_txx_zb[idx] = az * f.m_txx_zb[idx] + bz * dzz_txx;
    dzz_txx += f.m_txx_zb[idx];

    float dxx_tzz = sgradient<2, Order, X, DIFF_BACKWARD>(tmp_tzz_x_b, ix, 0, iz, grad_ctx);
    f.m_tzz_xb[idx] = ax * f.m_tzz_xb[idx] + bx * dxx_tzz;
    dxx_tzz += f.m_tzz_xb[idx];

    float inv_rho = 1.f / rho_b[idx];
    float shear_xz = dzz_txx + dxx_tzz;

    float exx_new = f.exx[idx] + solver.dt * inv_rho * (dxx_sxx + shear_xz);
    float ezz_new = f.ezz[idx] + solver.dt * inv_rho * (dzz_szz + shear_xz);
    f.exx[idx] = exx_new;
    f.ezz[idx] = ezz_new;

    float lam = lambda_b[idx];
    float mu_ = mu_b[idx];
    f.sxx[idx] += solver.dt * ((lam + 2.f * mu_) * exx_new + lam * ezz_new);
    f.szz[idx] += solver.dt * ((lam + 2.f * mu_) * ezz_new + lam * exx_new);
    f.txx[idx] += solver.dt * mu_ * exx_new;
    f.tzz[idx] += solver.dt * mu_ * ezz_new;

    f.das35[idx] = exx_new + ezz_new;
    f.das54x[idx] = 4.f * exx_new + ezz_new;
    f.das54z[idx] = exx_new + 4.f * ezz_new;
}
