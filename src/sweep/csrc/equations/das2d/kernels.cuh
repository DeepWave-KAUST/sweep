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

#define LAUNCH_DAS2D_FIRST_NOPML(order, grid, block, ...)                     \
    do {                                                                      \
        if      ((order) == 2) das2d_first_derivatives_nopml_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das2d_first_derivatives_nopml_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das2d_first_derivatives_nopml_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das2d_first_derivatives_nopml_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das2d_first_derivatives_nopml_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS2D_REVERSE_STRAIN_NOPML(order, grid, block, ...)            \
    do {                                                                      \
        if      ((order) == 2) das2d_reverse_strain_nopml_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das2d_reverse_strain_nopml_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das2d_reverse_strain_nopml_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das2d_reverse_strain_nopml_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das2d_reverse_strain_nopml_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS2D_REVERSE_STRESS_NOPML(order, grid, block, ...)            \
    do {                                                                      \
        if      ((order) == 2) das2d_reverse_stress_nopml_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das2d_reverse_stress_nopml_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das2d_reverse_stress_nopml_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das2d_reverse_stress_nopml_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das2d_reverse_stress_nopml_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS2D_PROJECT_MODEL_GRAD(order, grid, block, ...)              \
    do {                                                                      \
        if      ((order) == 2) das2d_project_model_grad_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das2d_project_model_grad_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das2d_project_model_grad_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das2d_project_model_grad_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das2d_project_model_grad_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS2D_SECOND_ADJOINT(order, direction, grid, block, ...)       \
    do {                                                                      \
        if ((direction) == X) {                                                \
            if      ((order) == 2) das2d_second_derivative_adjoint_kernel<2, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das2d_second_derivative_adjoint_kernel<4, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das2d_second_derivative_adjoint_kernel<6, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das2d_second_derivative_adjoint_kernel<8, X><<<grid, block>>>(__VA_ARGS__); \
            else                   das2d_second_derivative_adjoint_kernel<-1, X><<<grid, block>>>(__VA_ARGS__); \
        } else {                                                              \
            if      ((order) == 2) das2d_second_derivative_adjoint_kernel<2, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das2d_second_derivative_adjoint_kernel<4, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das2d_second_derivative_adjoint_kernel<6, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das2d_second_derivative_adjoint_kernel<8, Z><<<grid, block>>>(__VA_ARGS__); \
            else                   das2d_second_derivative_adjoint_kernel<-1, Z><<<grid, block>>>(__VA_ARGS__); \
        }                                                                     \
    } while (0)

#define LAUNCH_DAS2D_FIRST_ADJOINT(order, direction, grid, block, ...)        \
    do {                                                                      \
        if ((direction) == X) {                                                \
            if      ((order) == 2) das2d_first_derivative_adjoint_kernel<2, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das2d_first_derivative_adjoint_kernel<4, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das2d_first_derivative_adjoint_kernel<6, X><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das2d_first_derivative_adjoint_kernel<8, X><<<grid, block>>>(__VA_ARGS__); \
            else                   das2d_first_derivative_adjoint_kernel<-1, X><<<grid, block>>>(__VA_ARGS__); \
        } else {                                                              \
            if      ((order) == 2) das2d_first_derivative_adjoint_kernel<2, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 4) das2d_first_derivative_adjoint_kernel<4, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 6) das2d_first_derivative_adjoint_kernel<6, Z><<<grid, block>>>(__VA_ARGS__); \
            else if ((order) == 8) das2d_first_derivative_adjoint_kernel<8, Z><<<grid, block>>>(__VA_ARGS__); \
            else                   das2d_first_derivative_adjoint_kernel<-1, Z><<<grid, block>>>(__VA_ARGS__); \
        }                                                                     \
    } while (0)

template<int Order>
__device__ __forceinline__ float das2d_staggered_coeff(int m, const SGradParam& grad_ctx)
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
__device__ __forceinline__ void das2d_scatter_sgradient_adjoint(
    float bar,
    float* __restrict__ dst,
    int ix,
    int iz,
    SGradParam grad_ctx
)
{
    constexpr bool is_runtime = (Order == -1);
    const int M = is_runtime ? grad_ctx.M : Order / 2;
    const int idx = ix * grad_ctx.sx + iz * grad_ctx.sz;
    const int stride = (Direction == X) ? grad_ctx.sx : grad_ctx.sz;
    const float inv_h = (Direction == X) ? (1.f / grad_ctx.dx) : (1.f / grad_ctx.dz);

    for (int m = 0; m < M; ++m) {
        const float value = bar * das2d_staggered_coeff<Order>(m, grad_ctx) * inv_h;
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

template<int Order>
__global__ void das2d_first_derivatives_nopml_kernel(
    DasWavefieldPointer2D wf,
    float* __restrict__ tmp_sxx_x,
    float* __restrict__ tmp_szz_z,
    float* __restrict__ tmp_txx_z,
    float* __restrict__ tmp_tzz_x,
    SGradParam grad_ctx,
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

    tmp_sxx_x_b[idx] = sgradient<2, Order, X, DIFF_FORWARD>(f.sxx, ix, 0, iz, grad_ctx);
    tmp_szz_z_b[idx] = sgradient<2, Order, Z, DIFF_FORWARD>(f.szz, ix, 0, iz, grad_ctx);
    tmp_txx_z_b[idx] = sgradient<2, Order, Z, DIFF_FORWARD>(f.txx, ix, 0, iz, grad_ctx);
    tmp_tzz_x_b[idx] = sgradient<2, Order, X, DIFF_FORWARD>(f.tzz, ix, 0, iz, grad_ctx);
}

template<int Order>
__global__ void das2d_reverse_strain_nopml_kernel(
    DasWavefieldPointer2D wf,
    const float* __restrict__ tmp_sxx_x,
    const float* __restrict__ tmp_szz_z,
    const float* __restrict__ tmp_txx_z,
    const float* __restrict__ tmp_tzz_x,
    const float* __restrict__ rho,
    SGradParam grad_ctx,
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

    float dxx_sxx = sgradient<2, Order, X, DIFF_BACKWARD>(tmp_sxx_x_b, ix, 0, iz, grad_ctx);
    float dzz_szz = sgradient<2, Order, Z, DIFF_BACKWARD>(tmp_szz_z_b, ix, 0, iz, grad_ctx);
    float dzz_txx = sgradient<2, Order, Z, DIFF_BACKWARD>(tmp_txx_z_b, ix, 0, iz, grad_ctx);
    float dxx_tzz = sgradient<2, Order, X, DIFF_BACKWARD>(tmp_tzz_x_b, ix, 0, iz, grad_ctx);

    float inv_rho = 1.f / rho_b[idx];
    float shear_xz = dzz_txx + dxx_tzz;

    float exx_prev = f.exx[idx] - solver.dt * inv_rho * (dxx_sxx + shear_xz);
    float ezz_prev = f.ezz[idx] - solver.dt * inv_rho * (dzz_szz + shear_xz);
    f.exx[idx] = exx_prev;
    f.ezz[idx] = ezz_prev;
    f.das35[idx] = exx_prev + ezz_prev;
    f.das54x[idx] = 4.f * exx_prev + ezz_prev;
    f.das54z[idx] = exx_prev + 4.f * ezz_prev;
}

template<int Order>
__global__ void das2d_reverse_stress_nopml_kernel(
    DasWavefieldPointer2D wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
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

    const float* lambda_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;

    float lam = lambda_b[idx];
    float mu_ = mu_b[idx];
    float exx = f.exx[idx];
    float ezz = f.ezz[idx];

    f.sxx[idx] -= solver.dt * ((lam + 2.f * mu_) * exx + lam * ezz);
    f.szz[idx] -= solver.dt * ((lam + 2.f * mu_) * ezz + lam * exx);
    f.txx[idx] -= solver.dt * mu_ * exx;
    f.tzz[idx] -= solver.dt * mu_ * ezz;
}

template<int Order>
__global__ void das2d_project_model_grad_kernel(
    DasWavefieldPointer2D adjoint,
    const float* __restrict__ exx_now,
    const float* __restrict__ ezz_now,
    const float* __restrict__ exx_prev,
    const float* __restrict__ ezz_prev,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ rho,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_vs,
    float* __restrict__ grad_rho,
    float* __restrict__ bar_dxx_sxx,
    float* __restrict__ bar_dzz_szz,
    float* __restrict__ bar_dzz_txx,
    float* __restrict__ bar_dxx_tzz,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz || b >= solver.B) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto a = adjoint.offset(b, spatial_size);

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    int update_halo = 2 * halo;

    bool active = ix >= update_halo && ix < solver.nx - update_halo &&
                  iz >= update_halo && iz < solver.nz - update_halo;

    if (!active) {
        a.das35[idx] = 0.f;
        a.das54x[idx] = 0.f;
        a.das54z[idx] = 0.f;
        return;
    }

    const float* exx_now_b = exx_now + b * spatial_size;
    const float* ezz_now_b = ezz_now + b * spatial_size;
    const float* exx_prev_b = exx_prev + b * spatial_size;
    const float* ezz_prev_b = ezz_prev + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    const float* vs_b = vs + b * spatial_size;
    const float* rho_b = rho + b * spatial_size;
    float* gvp = grad_vp + b * spatial_size;
    float* gvs = grad_vs + b * spatial_size;
    float* grho = grad_rho + b * spatial_size;
    float* q_sxx = bar_dxx_sxx + b * spatial_size;
    float* q_szz = bar_dzz_szz + b * spatial_size;
    float* q_txx = bar_dzz_txx + b * spatial_size;
    float* q_tzz = bar_dxx_tzz + b * spatial_size;

    float bar_exx = a.exx[idx] + a.das35[idx] + 4.f * a.das54x[idx] + a.das54z[idx];
    float bar_ezz = a.ezz[idx] + a.das35[idx] + a.das54x[idx] + 4.f * a.das54z[idx];
    a.das35[idx] = 0.f;
    a.das54x[idx] = 0.f;
    a.das54z[idx] = 0.f;

    const float exx = exx_now_b[idx];
    const float ezz = ezz_now_b[idx];
    const float vp_ = vp_b[idx];
    const float vs_ = vs_b[idx];
    const float rho_ = rho_b[idx];
    const float lambda = rho_ * (vp_ * vp_ - 2.f * vs_ * vs_);
    const float mu = rho_ * vs_ * vs_;

    const float bar_sxx = a.sxx[idx];
    const float bar_szz = a.szz[idx];
    const float bar_txx = a.txx[idx];
    const float bar_tzz = a.tzz[idx];

    const float grad_lambda = solver.dt * (bar_sxx + bar_szz) * (exx + ezz);
    const float grad_mu = solver.dt * (
        2.f * bar_sxx * exx +
        2.f * bar_szz * ezz +
        bar_txx * exx +
        bar_tzz * ezz
    );

    bar_exx += solver.dt * ((lambda + 2.f * mu) * bar_sxx + lambda * bar_szz + mu * bar_txx);
    bar_ezz += solver.dt * (lambda * bar_sxx + (lambda + 2.f * mu) * bar_szz + mu * bar_tzz);

    const float dexx = exx - exx_prev_b[idx];
    const float dezz = ezz - ezz_prev_b[idx];

    gvp[idx] += 2.f * rho_ * vp_ * grad_lambda;
    gvs[idx] += -4.f * rho_ * vs_ * grad_lambda + 2.f * rho_ * vs_ * grad_mu;
    grho[idx] += (vp_ * vp_ - 2.f * vs_ * vs_) * grad_lambda +
                 (vs_ * vs_) * grad_mu -
                 (bar_exx * dexx + bar_ezz * dezz) / rho_;

    a.exx[idx] = bar_exx;
    a.ezz[idx] = bar_ezz;

    const float common = solver.dt / rho_ * (bar_exx + bar_ezz);
    q_sxx[idx] = solver.dt / rho_ * bar_exx;
    q_szz[idx] = solver.dt / rho_ * bar_ezz;
    q_txx[idx] = common;
    q_tzz[idx] = common;
}

template<int Order, int Direction>
__global__ void das2d_second_derivative_adjoint_kernel(
    DasWavefieldPointer2D adjoint,
    const float* __restrict__ bar_out,
    float* __restrict__ bar_tmp,
    float* __restrict__ adj_memory,
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

    const float* q_b = bar_out + b * spatial_size;
    float* tmp_b = bar_tmp + b * spatial_size;
    float* mem_b = adj_memory + b * spatial_size;

    const float q = q_b[idx];
    const float acoef = (Direction == X) ? cpml.ax[ix] : cpml.az[iz];
    const float bcoef = (Direction == X) ? cpml.bx[ix] : cpml.bz[iz];
    const float total_memory_bar = mem_b[idx] + q;
    const float derivative_bar = q + bcoef * total_memory_bar;
    mem_b[idx] = acoef * total_memory_bar;

    das2d_scatter_sgradient_adjoint<Order, Direction, DIFF_BACKWARD>(
        derivative_bar,
        tmp_b,
        ix,
        iz,
        grad_ctx
    );
}

template<int Order, int Direction>
__global__ void das2d_first_derivative_adjoint_kernel(
    DasWavefieldPointer2D adjoint,
    const float* __restrict__ bar_tmp,
    float* __restrict__ adj_memory,
    float* __restrict__ adj_field,
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

    const float* tmp_b = bar_tmp + b * spatial_size;
    float* mem_b = adj_memory + b * spatial_size;
    float* field_b = adj_field + b * spatial_size;

    const float q = tmp_b[idx];
    const float acoef = (Direction == X) ? cpml.axh[ix] : cpml.azh[iz];
    const float bcoef = (Direction == X) ? cpml.bxh[ix] : cpml.bzh[iz];
    const float total_memory_bar = mem_b[idx] + q;
    const float derivative_bar = q + bcoef * total_memory_bar;
    mem_b[idx] = acoef * total_memory_bar;

    das2d_scatter_sgradient_adjoint<Order, Direction, DIFF_FORWARD>(
        derivative_bar,
        field_b,
        ix,
        iz,
        grad_ctx
    );
}
