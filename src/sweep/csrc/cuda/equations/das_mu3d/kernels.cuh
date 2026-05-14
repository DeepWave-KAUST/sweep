#pragma once

#include "../elastic3d/kernels.cuh"

#include "../../common/context.h"
#include "../../common/das_mu.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"
#include "../../operators/staggered.cuh"

#define LAUNCH_DAS_MU3D_STRESS_STRAIN(order, grid, block, ...)                \
    do {                                                                      \
        if      ((order) == 2) das_mu3d_stress_strain_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das_mu3d_stress_strain_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das_mu3d_stress_strain_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das_mu3d_stress_strain_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das_mu3d_stress_strain_kernel<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define LAUNCH_DAS_MU3D_STRESS_STRAIN_ADJOINT_PREPARE(order, grid, block, ...) \
    do {                                                                       \
        if      ((order) == 2) das_mu3d_stress_strain_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) das_mu3d_stress_strain_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) das_mu3d_stress_strain_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) das_mu3d_stress_strain_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   das_mu3d_stress_strain_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

template<int Order>
__global__ void das_mu3d_stress_strain_kernel(
    DasMuWavefieldPointer3D wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    float* __restrict__ u_this,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int M_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx, solver, false);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx, solver, false);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx, solver, true);

    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz += f.m_vzz[idx];
    f.m_vyy[idx] = ay * f.m_vyy[idx] + by * dvy_dy;
    dvy_dy += f.m_vyy[idx];
    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx += f.m_vxx[idx];

    f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
    dvx_dz += f.m_vxz[idx];
    f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
    dvz_dx += f.m_vzx[idx];

    f.m_vxy[idx] = ayh * f.m_vxy[idx] + byh * dvx_dy;
    dvx_dy += f.m_vxy[idx];
    f.m_vyx[idx] = axh * f.m_vyx[idx] + bxh * dvy_dx;
    dvy_dx += f.m_vyx[idx];
    f.m_vyz[idx] = azh * f.m_vyz[idx] + bzh * dvy_dz;
    dvy_dz += f.m_vyz[idx];
    f.m_vzy[idx] = ayh * f.m_vzy[idx] + byh * dvz_dy;
    dvz_dy += f.m_vzy[idx];

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];
    float div_v = dvx_dx + dvy_dy + dvz_dz;

    f.sxx[idx] += solver.dt * (lam * div_v + 2.f * mu_ * dvx_dx);
    f.syy[idx] += solver.dt * (lam * div_v + 2.f * mu_ * dvy_dy);
    f.szz[idx] += solver.dt * (lam * div_v + 2.f * mu_ * dvz_dz);
    f.sxy[idx] += solver.dt * mu_ * (dvx_dy + dvy_dx);
    f.sxz[idx] += solver.dt * mu_ * (dvx_dz + dvz_dx);
    f.syz[idx] += solver.dt * mu_ * (dvy_dz + dvz_dy);

    f.exx[idx] += solver.dt * dvx_dx;
    f.eyy[idx] += solver.dt * dvy_dy;
    f.ezz[idx] += solver.dt * dvz_dz;
    f.exy[idx] += 0.5f * solver.dt * (dvx_dy + dvy_dx);
    f.exz[idx] += 0.5f * solver.dt * (dvx_dz + dvz_dx);
    f.eyz[idx] += 0.5f * solver.dt * (dvy_dz + dvz_dy);

    if (elastic3d_is_top_free_surface_row(solver, iz)) {
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
        f.syz[idx] = 0.f;
    }

    if (u_this_b) {
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vy[idx];
        u_this_b[2 * comp_stride + idx] = f.vz[idx];
    }
}

template<int Order>
__global__ void das_mu3d_stress_strain_adjoint_prepare(
    DasMuWavefieldPointer3D wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qxy,
    float* __restrict__ qxz,
    float* __restrict__ qyx,
    float* __restrict__ qyy,
    float* __restrict__ qyz,
    float* __restrict__ qzx,
    float* __restrict__ qzy,
    float* __restrict__ qzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;
    const int out = b * spatial_size + idx;

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];
    float l2m = lam + 2.f * mu_;

    float bar_sxx = f.sxx[idx];
    float bar_syy = f.syy[idx];
    float bar_szz = f.szz[idx];
    float bar_sxy = f.sxy[idx];
    float bar_sxz = f.sxz[idx];
    float bar_syz = f.syz[idx];
    if (elastic3d_is_top_free_surface_row(solver, iz)) {
        bar_szz = 0.f;
        bar_sxz = 0.f;
        bar_syz = 0.f;
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
        f.syz[idx] = 0.f;
    }

    float bar_exx = -f.exx[idx];
    float bar_eyy = -f.eyy[idx];
    float bar_ezz = -f.ezz[idx];
    float bar_exy = -f.exy[idx];
    float bar_exz = -f.exz[idx];
    float bar_eyz = -f.eyz[idx];

    float bar_dvx_dx = solver.dt * (l2m * bar_sxx + lam * bar_syy + lam * bar_szz + bar_exx);
    float bar_dvy_dy = solver.dt * (lam * bar_sxx + l2m * bar_syy + lam * bar_szz + bar_eyy);
    float bar_dvz_dz = solver.dt * (lam * bar_sxx + lam * bar_syy + l2m * bar_szz + bar_ezz);
    float bar_dvx_dy = solver.dt * (mu_ * bar_sxy + 0.5f * bar_exy);
    float bar_dvy_dx = solver.dt * (mu_ * bar_sxy + 0.5f * bar_exy);
    float bar_dvx_dz = solver.dt * (mu_ * bar_sxz + 0.5f * bar_exz);
    float bar_dvz_dx = solver.dt * (mu_ * bar_sxz + 0.5f * bar_exz);
    float bar_dvy_dz = solver.dt * (mu_ * bar_syz + 0.5f * bar_eyz);
    float bar_dvz_dy = solver.dt * (mu_ * bar_syz + 0.5f * bar_eyz);

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vxy = f.m_vxy[idx] + bar_dvx_dy;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vyx = f.m_vyx[idx] + bar_dvy_dx;
    float tmp_vyy = f.m_vyy[idx] + bar_dvy_dy;
    float tmp_vyz = f.m_vyz[idx] + bar_dvy_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;
    float tmp_vzy = f.m_vzy[idx] + bar_dvz_dy;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();
    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        qxx[out] = bar_dvx_dx;
        qxy[out] = bar_dvx_dy;
        qxz[out] = bar_dvx_dz;
        qyx[out] = bar_dvy_dx;
        qyy[out] = bar_dvy_dy;
        qyz[out] = bar_dvy_dz;
        qzx[out] = bar_dvz_dx;
        qzy[out] = bar_dvz_dy;
        qzz[out] = bar_dvz_dz;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    qxx[out] = bar_dvx_dx + bx * tmp_vxx;
    qxy[out] = bar_dvx_dy + byh * tmp_vxy;
    qxz[out] = bar_dvx_dz + bzh * tmp_vxz;
    qyx[out] = bar_dvy_dx + bxh * tmp_vyx;
    qyy[out] = bar_dvy_dy + by * tmp_vyy;
    qyz[out] = bar_dvy_dz + bzh * tmp_vyz;
    qzx[out] = bar_dvz_dx + bxh * tmp_vzx;
    qzy[out] = bar_dvz_dy + byh * tmp_vzy;
    qzz[out] = bar_dvz_dz + bz * tmp_vzz;

    f.m_vxx[idx] = ax * tmp_vxx;
    f.m_vxy[idx] = ayh * tmp_vxy;
    f.m_vxz[idx] = azh * tmp_vxz;
    f.m_vyx[idx] = axh * tmp_vyx;
    f.m_vyy[idx] = ay * tmp_vyy;
    f.m_vyz[idx] = azh * tmp_vyz;
    f.m_vzx[idx] = axh * tmp_vzx;
    f.m_vzy[idx] = ayh * tmp_vzy;
    f.m_vzz[idx] = az * tmp_vzz;
}
