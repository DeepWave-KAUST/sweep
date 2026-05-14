#pragma once

#include "../elastic2d/kernels.cuh"

#include "../../common/context.h"
#include "../../common/das_mu.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"
#include "../../operators/staggered.cuh"

#define LAUNCH_DAS_MU2D_STRESS_STRAIN(order, grid, block, ...)                \
    do {                                                                      \
        if      ((order) == 2) das_mu2d_stress_strain_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das_mu2d_stress_strain_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das_mu2d_stress_strain_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das_mu2d_stress_strain_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das_mu2d_stress_strain_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS_MU2D_STRESS_STRAIN_ADJOINT_PREPARE(order, grid, block, ...) \
    do {                                                                       \
        if      ((order) == 2) das_mu2d_stress_strain_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das_mu2d_stress_strain_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das_mu2d_stress_strain_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das_mu2d_stress_strain_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das_mu2d_stress_strain_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template<int Order>
__global__ void das_mu2d_stress_strain_kernel(
    DasMuWavefieldPointer2D wf,
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
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD>(f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.vz, ix, 0, iz, grad_ctx);

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz += f.m_vzz[idx];
    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx += f.m_vxx[idx];

    f.sxx[idx] += solver.dt * ((lam + 2.f * mu_) * dvx_dx + lam * dvz_dz);
    f.szz[idx] += solver.dt * ((lam + 2.f * mu_) * dvz_dz + lam * dvx_dx);

    f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
    dvx_dz += f.m_vxz[idx];
    f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
    dvz_dx += f.m_vzx[idx];

    f.sxz[idx] += solver.dt * mu_ * (dvx_dz + dvz_dx);

    f.exx[idx] += solver.dt * dvx_dx;
    f.ezz[idx] += solver.dt * dvz_dz;
    f.exz[idx] += 0.5f * solver.dt * (dvx_dz + dvz_dx);

    if (elastic_is_top_free_surface_row(solver, iz)) {
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
    }

    if (u_this_b) {
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vz[idx];
    }
}

template<int Order>
__global__ void das_mu2d_stress_strain_adjoint_prepare(
    DasMuWavefieldPointer2D wf,
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
    int b = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;
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

    float bar_exx = -f.exx[idx];
    float bar_ezz = -f.ezz[idx];
    float bar_exz = -f.exz[idx];

    float bar_dvx_dx = solver.dt * (((lam + 2.f * mu_) * bar_sxx + lam * bar_szz) + bar_exx);
    float bar_dvz_dz = solver.dt * (((lam + 2.f * mu_) * bar_szz + lam * bar_sxx) + bar_ezz);
    float bar_dvx_dz = solver.dt * (mu_ * bar_sxz + 0.5f * bar_exz);
    float bar_dvz_dx = solver.dt * (mu_ * bar_sxz + 0.5f * bar_exz);

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
