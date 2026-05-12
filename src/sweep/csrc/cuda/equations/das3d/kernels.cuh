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

#define LAUNCH_DAS3D_PROJECT_MODEL_GRAD(order, grid, block, ...)              \
    do {                                                                      \
        if      ((order) == 2) das3d_project_model_grad_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das3d_project_model_grad_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das3d_project_model_grad_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das3d_project_model_grad_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das3d_project_model_grad_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS3D_SECOND_ADJOINT(order, direction, grid, block, ...)       \
    do {                                                                      \
        if ((direction) == X) {                                                \
            if      ((order) == 2) das3d_second_derivative_adjoint_kernel<2, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das3d_second_derivative_adjoint_kernel<4, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das3d_second_derivative_adjoint_kernel<6, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das3d_second_derivative_adjoint_kernel<8, X><<<grid, block>>>(__VA_ARGS__); \
            else                   das3d_second_derivative_adjoint_kernel<-1, X><<<grid, block>>>(__VA_ARGS__); \
        } else if ((direction) == Y) {                                         \
            if      ((order) == 2) das3d_second_derivative_adjoint_kernel<2, Y><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das3d_second_derivative_adjoint_kernel<4, Y><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das3d_second_derivative_adjoint_kernel<6, Y><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das3d_second_derivative_adjoint_kernel<8, Y><<<grid, block>>>(__VA_ARGS__); \
            else                   das3d_second_derivative_adjoint_kernel<-1, Y><<<grid, block>>>(__VA_ARGS__); \
        } else {                                                              \
            if      ((order) == 2) das3d_second_derivative_adjoint_kernel<2, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das3d_second_derivative_adjoint_kernel<4, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das3d_second_derivative_adjoint_kernel<6, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das3d_second_derivative_adjoint_kernel<8, Z><<<grid, block>>>(__VA_ARGS__); \
            else                   das3d_second_derivative_adjoint_kernel<-1, Z><<<grid, block>>>(__VA_ARGS__); \
        }                                                                     \
    } while (0)

#define LAUNCH_DAS3D_FIRST_ADJOINT(order, direction, grid, block, ...)        \
    do {                                                                      \
        if ((direction) == X) {                                                \
            if      ((order) == 2) das3d_first_derivative_adjoint_kernel<2, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das3d_first_derivative_adjoint_kernel<4, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das3d_first_derivative_adjoint_kernel<6, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das3d_first_derivative_adjoint_kernel<8, X><<<grid, block>>>(__VA_ARGS__); \
            else                   das3d_first_derivative_adjoint_kernel<-1, X><<<grid, block>>>(__VA_ARGS__); \
        } else if ((direction) == Y) {                                         \
            if      ((order) == 2) das3d_first_derivative_adjoint_kernel<2, Y><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das3d_first_derivative_adjoint_kernel<4, Y><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das3d_first_derivative_adjoint_kernel<6, Y><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das3d_first_derivative_adjoint_kernel<8, Y><<<grid, block>>>(__VA_ARGS__); \
            else                   das3d_first_derivative_adjoint_kernel<-1, Y><<<grid, block>>>(__VA_ARGS__); \
        } else {                                                              \
            if      ((order) == 2) das3d_first_derivative_adjoint_kernel<2, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das3d_first_derivative_adjoint_kernel<4, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das3d_first_derivative_adjoint_kernel<6, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das3d_first_derivative_adjoint_kernel<8, Z><<<grid, block>>>(__VA_ARGS__); \
            else                   das3d_first_derivative_adjoint_kernel<-1, Z><<<grid, block>>>(__VA_ARGS__); \
        }                                                                     \
    } while (0)

template<int Order>
__device__ __forceinline__ float das3d_staggered_coeff(int m, const SGradParam& grad_ctx)
{
    if constexpr (Order == -1) {
        return grad_ctx.coeff[m];
    } else if constexpr (Order == 2) {
        return 1.f;
    } else if constexpr (Order == 4) {
        return (m == 0) ? 9.f / 8.f : -1.f / 24.f;
    } else if constexpr (Order == 6) {
        return (m == 0) ? 75.f / 64.f : ((m == 1) ? -25.f / 384.f : 3.f / 640.f);
    } else {
        return (m == 0) ? 1225.f / 1024.f
             : (m == 1) ? -245.f / 3072.f
             : (m == 2) ? 49.f / 5120.f
                        : -5.f / 7168.f;
    }
}

template<int Order, int Direction, int Type>
__device__ __forceinline__ void das3d_scatter_sgradient_adjoint(
    float bar,
    float* __restrict__ dst,
    int ix,
    int iy,
    int iz,
    SGradParam grad_ctx
)
{
    constexpr bool is_runtime = (Order == -1);
    const int M = is_runtime ? grad_ctx.M : Order / 2;
    const int idx = ix * grad_ctx.sx + iy * grad_ctx.sy + iz * grad_ctx.sz;
    const int stride = (Direction == X) ? grad_ctx.sx : ((Direction == Y) ? grad_ctx.sy : grad_ctx.sz);
    const float inv_h = (Direction == X) ? (1.f / grad_ctx.dx)
                      : (Direction == Y) ? (1.f / grad_ctx.dy)
                                         : (1.f / grad_ctx.dz);

    for (int m = 0; m < M; ++m) {
        const float value = bar * das3d_staggered_coeff<Order>(m, grad_ctx) * inv_h;
        if constexpr (Type == DIFF_FORWARD) {
            atomicAdd(dst + idx + (m + 1) * stride, value);
            atomicAdd(dst + idx - m * stride, -value);
        } else {
            atomicAdd(dst + idx + m * stride, value);
            atomicAdd(dst + idx - (m + 1) * stride, -value);
        }
    }
}

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

template<int Order>
__global__ void das3d_project_model_grad_kernel(
    DasWavefieldPointer3D adjoint,
    const float* __restrict__ exx_now,
    const float* __restrict__ eyy_now,
    const float* __restrict__ ezz_now,
    const float* __restrict__ exx_prev,
    const float* __restrict__ eyy_prev,
    const float* __restrict__ ezz_prev,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ rho,
    float* __restrict__ gvp,
    float* __restrict__ gvs,
    float* __restrict__ grho,
    float* __restrict__ q_dxx_sxx,
    float* __restrict__ q_dyy_syy,
    float* __restrict__ q_dzz_szz,
    float* __restrict__ q_dyy_txx,
    float* __restrict__ q_dzz_txx,
    float* __restrict__ q_dxx_tyy,
    float* __restrict__ q_dzz_tyy,
    float* __restrict__ q_dxx_tzz,
    float* __restrict__ q_dyy_tzz,
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
    int shift = b * spatial_size;
    auto a = adjoint.offset(b, spatial_size);
    idx += shift;

    float bar_exx = a.exx[idx - shift] +
        a.das35[idx - shift] + 4.f * a.das54x[idx - shift] +
        a.das54y[idx - shift] + a.das54z[idx - shift];
    float bar_eyy = a.eyy[idx - shift] +
        a.das35[idx - shift] + a.das54x[idx - shift] +
        4.f * a.das54y[idx - shift] + a.das54z[idx - shift];
    float bar_ezz = a.ezz[idx - shift] +
        a.das35[idx - shift] + a.das54x[idx - shift] +
        a.das54y[idx - shift] + 4.f * a.das54z[idx - shift];

    a.das35[idx - shift] = 0.f;
    a.das54x[idx - shift] = 0.f;
    a.das54y[idx - shift] = 0.f;
    a.das54z[idx - shift] = 0.f;

    float bar_sxx = a.sxx[idx - shift];
    float bar_syy = a.syy[idx - shift];
    float bar_szz = a.szz[idx - shift];
    float bar_txx = a.txx[idx - shift];
    float bar_tyy = a.tyy[idx - shift];
    float bar_tzz = a.tzz[idx - shift];

    float exx = exx_now[idx];
    float eyy = eyy_now[idx];
    float ezz = ezz_now[idx];
    float div_e = exx + eyy + ezz;
    float sum_bar_s = bar_sxx + bar_syy + bar_szz;

    float vp_ = vp[idx];
    float vs_ = vs[idx];
    float rho_ = rho[idx];
    float mu = rho_ * vs_ * vs_;
    float lambda = rho_ * (vp_ * vp_ - 2.f * vs_ * vs_);

    float grad_lambda = solver.dt * sum_bar_s * div_e;
    float grad_mu = solver.dt * (
        2.f * bar_sxx * exx + 2.f * bar_syy * eyy + 2.f * bar_szz * ezz +
        bar_txx * exx + bar_tyy * eyy + bar_tzz * ezz
    );

    bar_exx += solver.dt * (lambda * sum_bar_s + 2.f * mu * bar_sxx + mu * bar_txx);
    bar_eyy += solver.dt * (lambda * sum_bar_s + 2.f * mu * bar_syy + mu * bar_tyy);
    bar_ezz += solver.dt * (lambda * sum_bar_s + 2.f * mu * bar_szz + mu * bar_tzz);

    float dexx = exx - exx_prev[idx];
    float deyy = eyy - eyy_prev[idx];
    float dezz = ezz - ezz_prev[idx];

    gvp[idx] += 2.f * rho_ * vp_ * grad_lambda;
    gvs[idx] += -4.f * rho_ * vs_ * grad_lambda + 2.f * rho_ * vs_ * grad_mu;
    grho[idx] += (vp_ * vp_ - 2.f * vs_ * vs_) * grad_lambda +
                 (vs_ * vs_) * grad_mu -
                 (bar_exx * dexx + bar_eyy * deyy + bar_ezz * dezz) / rho_;

    a.exx[idx - shift] = bar_exx;
    a.eyy[idx - shift] = bar_eyy;
    a.ezz[idx - shift] = bar_ezz;

    const float dt_over_rho = solver.dt / rho_;
    q_dxx_sxx[idx] = dt_over_rho * bar_exx;
    q_dyy_syy[idx] = dt_over_rho * bar_eyy;
    q_dzz_szz[idx] = dt_over_rho * bar_ezz;
    q_dyy_txx[idx] = dt_over_rho * (bar_exx + bar_eyy);
    q_dzz_txx[idx] = dt_over_rho * (bar_exx + bar_ezz);
    q_dxx_tyy[idx] = dt_over_rho * (bar_exx + bar_eyy);
    q_dzz_tyy[idx] = dt_over_rho * (bar_eyy + bar_ezz);
    q_dxx_tzz[idx] = dt_over_rho * (bar_exx + bar_ezz);
    q_dyy_tzz[idx] = dt_over_rho * (bar_eyy + bar_ezz);
}

template<int Order, int Direction>
__global__ void das3d_second_derivative_adjoint_kernel(
    DasWavefieldPointer3D adjoint,
    const float* __restrict__ bar_out,
    float* __restrict__ bar_tmp,
    float* __restrict__ adj_memory,
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
    int shift = b * spatial_size;

    const float* q_b = bar_out + shift;
    float* tmp_b = bar_tmp + shift;
    float* mem_b = adj_memory + shift;

    const float q = q_b[idx];
    const float acoef = (Direction == X) ? cpml.ax[ix] : ((Direction == Y) ? cpml.ay[iy] : cpml.az[iz]);
    const float bcoef = (Direction == X) ? cpml.bx[ix] : ((Direction == Y) ? cpml.by[iy] : cpml.bz[iz]);
    const float total_memory_bar = mem_b[idx] + q;
    const float derivative_bar = q + bcoef * total_memory_bar;
    mem_b[idx] = acoef * total_memory_bar;

    das3d_scatter_sgradient_adjoint<Order, Direction, DIFF_BACKWARD>(
        derivative_bar,
        tmp_b,
        ix,
        iy,
        iz,
        grad_ctx
    );
}

template<int Order, int Direction>
__global__ void das3d_first_derivative_adjoint_kernel(
    DasWavefieldPointer3D adjoint,
    const float* __restrict__ bar_tmp,
    float* __restrict__ adj_memory,
    float* __restrict__ adj_field,
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
    int shift = b * spatial_size;

    const float* tmp_b = bar_tmp + shift;
    float* mem_b = adj_memory + shift;
    float* field_b = adj_field + shift;

    const float q = tmp_b[idx];
    const float acoef = (Direction == X) ? cpml.axh[ix] : ((Direction == Y) ? cpml.ayh[iy] : cpml.azh[iz]);
    const float bcoef = (Direction == X) ? cpml.bxh[ix] : ((Direction == Y) ? cpml.byh[iy] : cpml.bzh[iz]);
    const float total_memory_bar = mem_b[idx] + q;
    const float derivative_bar = q + bcoef * total_memory_bar;
    mem_b[idx] = acoef * total_memory_bar;

    das3d_scatter_sgradient_adjoint<Order, Direction, DIFF_FORWARD>(
        derivative_bar,
        field_b,
        ix,
        iy,
        iz,
        grad_ctx
    );
}
