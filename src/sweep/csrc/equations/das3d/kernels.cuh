#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/context.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../operators/dim.cuh"
#include "../../operators/staggered.cuh"

#define LAUNCH_DAS3D_FIRST(order, grid, block, ...)                           \
    do {                                                                      \
        if      ((order) == 2) das3d_first_derivatives_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das3d_first_derivatives_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das3d_first_derivatives_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das3d_first_derivatives_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das3d_first_derivatives_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS3D_SECOND(order, grid, block, ...)                          \
    do {                                                                      \
        if      ((order) == 2) das3d_update_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das3d_update_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das3d_update_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das3d_update_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das3d_update_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template<int Order>
__global__ void das3d_first_derivatives_kernel(
    DasWavefieldPointer3D wf,
    float* __restrict__ tmp_sxx_x,
    float* __restrict__ tmp_syy_y,
    float* __restrict__ tmp_szz_z,
    float* __restrict__ tmp_txx_y,
    float* __restrict__ tmp_txx_z,
    float* __restrict__ tmp_tyy_x,
    float* __restrict__ tmp_tyy_z,
    float* __restrict__ tmp_tzz_x,
    float* __restrict__ tmp_tzz_y,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (ix >= solver.nx || iy >= solver.ny || iz >= solver.nz || b >= solver.B) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo) {
        return;
    }

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.ny * solver.nx + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    int shift = b * spatial_size;
    float* tmp_sxx_x_b = tmp_sxx_x + shift;
    float* tmp_syy_y_b = tmp_syy_y + shift;
    float* tmp_szz_z_b = tmp_szz_z + shift;
    float* tmp_txx_y_b = tmp_txx_y + shift;
    float* tmp_txx_z_b = tmp_txx_z + shift;
    float* tmp_tyy_x_b = tmp_tyy_x + shift;
    float* tmp_tyy_z_b = tmp_tyy_z + shift;
    float* tmp_tzz_x_b = tmp_tzz_x + shift;
    float* tmp_tzz_y_b = tmp_tzz_y + shift;

    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD>(f.sxx, ix, iy, iz, grad_ctx);
    f.m_sxx_xf[idx] = axh * f.m_sxx_xf[idx] + bxh * dsxx_dx;
    tmp_sxx_x_b[idx] = dsxx_dx + f.m_sxx_xf[idx];

    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD>(f.syy, ix, iy, iz, grad_ctx);
    f.m_syy_yf[idx] = ayh * f.m_syy_yf[idx] + byh * dsyy_dy;
    tmp_syy_y_b[idx] = dsyy_dy + f.m_syy_yf[idx];

    float dszz_dz = sgradient<3, Order, Z, DIFF_FORWARD>(f.szz, ix, iy, iz, grad_ctx);
    f.m_szz_zf[idx] = azh * f.m_szz_zf[idx] + bzh * dszz_dz;
    tmp_szz_z_b[idx] = dszz_dz + f.m_szz_zf[idx];

    float dtxx_dy = sgradient<3, Order, Y, DIFF_FORWARD>(f.txx, ix, iy, iz, grad_ctx);
    f.m_txx_yf[idx] = ayh * f.m_txx_yf[idx] + byh * dtxx_dy;
    tmp_txx_y_b[idx] = dtxx_dy + f.m_txx_yf[idx];

    float dtxx_dz = sgradient<3, Order, Z, DIFF_FORWARD>(f.txx, ix, iy, iz, grad_ctx);
    f.m_txx_zf[idx] = azh * f.m_txx_zf[idx] + bzh * dtxx_dz;
    tmp_txx_z_b[idx] = dtxx_dz + f.m_txx_zf[idx];

    float dtyy_dx = sgradient<3, Order, X, DIFF_FORWARD>(f.tyy, ix, iy, iz, grad_ctx);
    f.m_tyy_xf[idx] = axh * f.m_tyy_xf[idx] + bxh * dtyy_dx;
    tmp_tyy_x_b[idx] = dtyy_dx + f.m_tyy_xf[idx];

    float dtyy_dz = sgradient<3, Order, Z, DIFF_FORWARD>(f.tyy, ix, iy, iz, grad_ctx);
    f.m_tyy_zf[idx] = azh * f.m_tyy_zf[idx] + bzh * dtyy_dz;
    tmp_tyy_z_b[idx] = dtyy_dz + f.m_tyy_zf[idx];

    float dtzz_dx = sgradient<3, Order, X, DIFF_FORWARD>(f.tzz, ix, iy, iz, grad_ctx);
    f.m_tzz_xf[idx] = axh * f.m_tzz_xf[idx] + bxh * dtzz_dx;
    tmp_tzz_x_b[idx] = dtzz_dx + f.m_tzz_xf[idx];

    float dtzz_dy = sgradient<3, Order, Y, DIFF_FORWARD>(f.tzz, ix, iy, iz, grad_ctx);
    f.m_tzz_yf[idx] = ayh * f.m_tzz_yf[idx] + byh * dtzz_dy;
    tmp_tzz_y_b[idx] = dtzz_dy + f.m_tzz_yf[idx];
}

template<int Order>
__global__ void das3d_update_kernel(
    DasWavefieldPointer3D wf,
    const float* __restrict__ tmp_sxx_x,
    const float* __restrict__ tmp_syy_y,
    const float* __restrict__ tmp_szz_z,
    const float* __restrict__ tmp_txx_y,
    const float* __restrict__ tmp_txx_z,
    const float* __restrict__ tmp_tyy_x,
    const float* __restrict__ tmp_tyy_z,
    const float* __restrict__ tmp_tzz_x,
    const float* __restrict__ tmp_tzz_y,
    const float* __restrict__ rho,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (ix >= solver.nx || iy >= solver.ny || iz >= solver.nz || b >= solver.B) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    int update_halo = 2 * halo;

    if (ix < update_halo || ix >= solver.nx - update_halo ||
        iy < update_halo || iy >= solver.ny - update_halo ||
        iz < update_halo || iz >= solver.nz - update_halo) {
        return;
    }

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.ny * solver.nx + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    int shift = b * spatial_size;
    const float* tmp_sxx_x_b = tmp_sxx_x + shift;
    const float* tmp_syy_y_b = tmp_syy_y + shift;
    const float* tmp_szz_z_b = tmp_szz_z + shift;
    const float* tmp_txx_y_b = tmp_txx_y + shift;
    const float* tmp_txx_z_b = tmp_txx_z + shift;
    const float* tmp_tyy_x_b = tmp_tyy_x + shift;
    const float* tmp_tyy_z_b = tmp_tyy_z + shift;
    const float* tmp_tzz_x_b = tmp_tzz_x + shift;
    const float* tmp_tzz_y_b = tmp_tzz_y + shift;

    const float* rho_b = rho + shift;
    const float* lambda_b = lambda + shift;
    const float* mu_b = mu + shift;

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float az = cpml.az[iz];
    float bz = cpml.bz[iz];

    float dxx_sxx = sgradient<3, Order, X, DIFF_BACKWARD>(tmp_sxx_x_b, ix, iy, iz, grad_ctx);
    f.m_sxx_xb[idx] = ax * f.m_sxx_xb[idx] + bx * dxx_sxx;
    dxx_sxx += f.m_sxx_xb[idx];

    float dyy_syy = sgradient<3, Order, Y, DIFF_BACKWARD>(tmp_syy_y_b, ix, iy, iz, grad_ctx);
    f.m_syy_yb[idx] = ay * f.m_syy_yb[idx] + by * dyy_syy;
    dyy_syy += f.m_syy_yb[idx];

    float dzz_szz = sgradient<3, Order, Z, DIFF_BACKWARD>(tmp_szz_z_b, ix, iy, iz, grad_ctx);
    f.m_szz_zb[idx] = az * f.m_szz_zb[idx] + bz * dzz_szz;
    dzz_szz += f.m_szz_zb[idx];

    float dyy_txx = sgradient<3, Order, Y, DIFF_BACKWARD>(tmp_txx_y_b, ix, iy, iz, grad_ctx);
    f.m_txx_yb[idx] = ay * f.m_txx_yb[idx] + by * dyy_txx;
    dyy_txx += f.m_txx_yb[idx];

    float dzz_txx = sgradient<3, Order, Z, DIFF_BACKWARD>(tmp_txx_z_b, ix, iy, iz, grad_ctx);
    f.m_txx_zb[idx] = az * f.m_txx_zb[idx] + bz * dzz_txx;
    dzz_txx += f.m_txx_zb[idx];

    float dxx_tyy = sgradient<3, Order, X, DIFF_BACKWARD>(tmp_tyy_x_b, ix, iy, iz, grad_ctx);
    f.m_tyy_xb[idx] = ax * f.m_tyy_xb[idx] + bx * dxx_tyy;
    dxx_tyy += f.m_tyy_xb[idx];

    float dzz_tyy = sgradient<3, Order, Z, DIFF_BACKWARD>(tmp_tyy_z_b, ix, iy, iz, grad_ctx);
    f.m_tyy_zb[idx] = az * f.m_tyy_zb[idx] + bz * dzz_tyy;
    dzz_tyy += f.m_tyy_zb[idx];

    float dxx_tzz = sgradient<3, Order, X, DIFF_BACKWARD>(tmp_tzz_x_b, ix, iy, iz, grad_ctx);
    f.m_tzz_xb[idx] = ax * f.m_tzz_xb[idx] + bx * dxx_tzz;
    dxx_tzz += f.m_tzz_xb[idx];

    float dyy_tzz = sgradient<3, Order, Y, DIFF_BACKWARD>(tmp_tzz_y_b, ix, iy, iz, grad_ctx);
    f.m_tzz_yb[idx] = ay * f.m_tzz_yb[idx] + by * dyy_tzz;
    dyy_tzz += f.m_tzz_yb[idx];

    float inv_rho = 1.f / rho_b[idx];
    float exx_new = f.exx[idx] + solver.dt * inv_rho *
        (dxx_sxx + dyy_txx + dxx_tyy + dzz_txx + dxx_tzz);
    float eyy_new = f.eyy[idx] + solver.dt * inv_rho *
        (dyy_syy + dyy_txx + dxx_tyy + dzz_tyy + dyy_tzz);
    float ezz_new = f.ezz[idx] + solver.dt * inv_rho *
        (dzz_szz + dzz_txx + dxx_tzz + dzz_tyy + dyy_tzz);
    f.exx[idx] = exx_new;
    f.eyy[idx] = eyy_new;
    f.ezz[idx] = ezz_new;

    float lam = lambda_b[idx];
    float mu_ = mu_b[idx];
    float div_e = exx_new + eyy_new + ezz_new;
    f.sxx[idx] += solver.dt * (lam * div_e + 2.f * mu_ * exx_new);
    f.syy[idx] += solver.dt * (lam * div_e + 2.f * mu_ * eyy_new);
    f.szz[idx] += solver.dt * (lam * div_e + 2.f * mu_ * ezz_new);
    f.txx[idx] += solver.dt * mu_ * exx_new;
    f.tyy[idx] += solver.dt * mu_ * eyy_new;
    f.tzz[idx] += solver.dt * mu_ * ezz_new;

    f.das35[idx] = div_e;
    f.das54x[idx] = 4.f * exx_new + eyy_new + ezz_new;
    f.das54y[idx] = exx_new + 4.f * eyy_new + ezz_new;
    f.das54z[idx] = exx_new + eyy_new + 4.f * ezz_new;
}
