#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"
#include "../../operators/staggered.cuh"

#define LAUNCH_3DELASTIC_VELOCITY(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_3DELASTIC_STRESS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_VELOCITY_NOPML(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel_3d_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_3d_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_3d_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_3d_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_3d_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_3DELASTIC_STRESS_NOPML(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel_3d_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_3d_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_3d_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_3d_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_3d_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_VELOCITY_ADJOINT(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_STRESS_ADJOINT(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_VELOCITY_ADJOINT_PREPARE(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_prepare_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_prepare_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_prepare_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_prepare_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_prepare_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_VELOCITY_ADJOINT_APPLY(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_apply_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_apply_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_apply_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_apply_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_apply_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_STRESS_ADJOINT_PREPARE(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_prepare_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_prepare_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_prepare_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_prepare_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_prepare_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_STRESS_ADJOINT_APPLY(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_apply_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_apply_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_apply_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_apply_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_apply_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) calculate_grad_elastic3d_bs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic3d_bs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic3d_bs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic3d_bs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic3d_bs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

template<int Order>
__global__ void elastic_velocity_kernel_3d(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx =
        iz * solver.nx * solver.ny +
        iy * solver.nx +
        ix;

    auto f = wf.offset(b, spatial_size);
    const float* rho_b = rho + b * spatial_size;

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx, solver, true);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx, solver, true);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx, solver, true);

    float inv_rho = 1.f / rho_b[idx];

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        f.vx[idx] += solver.dt * inv_rho *
            (dsxx_dx + dsxy_dy + dsxz_dz);

        f.vy[idx] += solver.dt * inv_rho *
            (dsxy_dx + dsyy_dy + dsyz_dz);

        f.vz[idx] += solver.dt * inv_rho *
            (dsxz_dx + dsyz_dy + dszz_dz);
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

    // PML boundaries
    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];
    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];

    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];
    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];

    f.m_sxyy[idx] = ay * f.m_sxyy[idx] + by * dsxy_dy;
    dsxy_dy += f.m_sxyy[idx];

    f.m_sxyx[idx] = ax * f.m_sxyx[idx] + bx * dsxy_dx;
    dsxy_dx += f.m_sxyx[idx];

    f.m_syyy[idx] = ayh * f.m_syyy[idx] + byh * dsyy_dy;
    dsyy_dy += f.m_syyy[idx];

    f.m_syzz[idx] = az * f.m_syzz[idx] + bz * dsyz_dz;
    dsyz_dz += f.m_syzz[idx];

    f.m_syzy[idx] = ay * f.m_syzy[idx] + by * dsyz_dy;
    dsyz_dy += f.m_syzy[idx];

    // Updates
    f.vx[idx] += solver.dt * inv_rho *
        (dsxx_dx + dsxy_dy + dsxz_dz);

    f.vy[idx] += solver.dt * inv_rho *
        (dsxy_dx + dsyy_dy + dsyz_dz);

    f.vz[idx] += solver.dt * inv_rho *
        (dsxz_dx + dsyz_dy + dszz_dz);
}


template<int Order>
__global__ void elastic_stress_kernel_3d(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    float* __restrict__ u_this_t,
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

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx =
        iz * solver.nx * solver.ny +
        iy * solver.nx +
        ix;

    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx, solver, false);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx, solver, false);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx, solver, true);

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        float div_v = dvx_dx + dvy_dy + dvz_dz;

        f.sxx[idx] += solver.dt *
            (lam * div_v + 2.f * mu_ * dvx_dx);

        f.syy[idx] += solver.dt *
            (lam * div_v + 2.f * mu_ * dvy_dy);

        f.szz[idx] += solver.dt *
            (lam * div_v + 2.f * mu_ * dvz_dz);

        f.sxy[idx] += solver.dt *
            mu_ * (dvx_dy + dvy_dx);

        f.sxz[idx] += solver.dt *
            mu_ * (dvx_dz + dvz_dx);

        f.syz[idx] += solver.dt *
            mu_ * (dvy_dz + dvz_dy);

        if (elastic3d_is_top_free_surface_row(solver, iz)) {
            f.szz[idx] = 0.f;
            f.sxz[idx] = 0.f;
            f.syz[idx] = 0.f;
        }

        if (u_this_t) {
            float* u_this_b = u_this_t + b * spatial_size;
            int comp_stride = solver.B * spatial_size;
            u_this_b[0 * comp_stride + idx] = f.vx[idx];
            u_this_b[1 * comp_stride + idx] = f.vy[idx];
            u_this_b[2 * comp_stride + idx] = f.vz[idx];
        }
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

    // PML
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

    // Update
    float div_v = dvx_dx + dvy_dy + dvz_dz;

    f.sxx[idx] += solver.dt *
        (lam * div_v + 2.f * mu_ * dvx_dx);

    f.syy[idx] += solver.dt *
        (lam * div_v + 2.f * mu_ * dvy_dy);

    f.szz[idx] += solver.dt *
        (lam * div_v + 2.f * mu_ * dvz_dz);

    f.sxy[idx] += solver.dt *
        mu_ * (dvx_dy + dvy_dx);

    f.sxz[idx] += solver.dt *
        mu_ * (dvx_dz + dvz_dx);

    f.syz[idx] += solver.dt *
        mu_ * (dvy_dz + dvz_dy);

    if (elastic3d_is_top_free_surface_row(solver, iz)) {
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
        f.syz[idx] = 0.f;
    }

    if (u_this_t) {
        float* u_this_b = u_this_t + b * spatial_size;
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vy[idx];
        u_this_b[2 * comp_stride + idx] = f.vz[idx];
    }
}

template<int Order>
__global__ void elastic_velocity_kernel_3d_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn + 1*M+1;

    int top_halo = solver.free_surface ? M: halo;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx =
        iz * solver.nx * solver.ny +
        iy * solver.nx +
        ix;

    auto f = wf.offset(b, spatial_size);
    const float* rho_b = rho + b * spatial_size;

    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx, solver, true);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx, solver, true);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx, solver, true);

    float inv_rho = 1.f / rho_b[idx];

    f.vx[idx] -= solver.dt * inv_rho *
        (dsxx_dx + dsxy_dy + dsxz_dz);

    f.vy[idx] -= solver.dt * inv_rho *
        (dsxy_dx + dsyy_dy + dsyz_dz);

    f.vz[idx] -= solver.dt * inv_rho *
        (dsxz_dx + dsyz_dy + dszz_dz);
}


template<int Order>
__global__ void elastic_stress_kernel_3d_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn + 1*M+1;

    int top_halo = solver.free_surface ? M: halo;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx =
        iz * solver.nx * solver.ny +
        iy * solver.nx +
        ix;

    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx, solver, false);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx, solver, false);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx, solver, true);


    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    float div_v = dvx_dx + dvy_dy + dvz_dz;

    f.sxx[idx] -= solver.dt *
        (lam * div_v + 2.f * mu_ * dvx_dx);

    f.syy[idx] -= solver.dt *
        (lam * div_v + 2.f * mu_ * dvy_dy);

    f.szz[idx] -= solver.dt *
        (lam * div_v + 2.f * mu_ * dvz_dz);

    f.sxy[idx] -= solver.dt *
        mu_ * (dvx_dy + dvy_dx);

    f.sxz[idx] -= solver.dt *
        mu_ * (dvx_dz + dvz_dx);

    f.syz[idx] -= solver.dt *
        mu_ * (dvy_dz + dvz_dy);
}


template<int Order>
__global__ void elastic_velocity_adjoint_kernel_3d(
    ElasticWavefieldPointer wf,
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

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx =
        iz * solver.nx * solver.ny +
        iy * solver.nx +
        ix;

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;

    auto f = wf.offset(b, spatial_size);
    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float lam = lam_b[idx];
    float mu_  = mu_b[idx];

    float l2m = lam + 2.f * mu_;

    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxx_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsxz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);

    float dsyy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyy_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);

    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dsyz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);

    float dszz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx);
    float dszz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx);
    float dszz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx);

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        f.vx[idx] += solver.dt * (
            l2m * dsxx_dx
            + lam * dsyy_dx
            + lam * dszz_dx
            + mu_  * dsxy_dy
            + mu_  * dsxz_dz
        );

        f.vy[idx] += solver.dt * (
            lam * dsxx_dy
            + l2m * dsyy_dy
            + lam * dszz_dy
            + mu_  * dsxy_dx
            + mu_  * dsyz_dz
        );

        f.vz[idx] += solver.dt * (
            lam * dsxx_dz
            + lam * dsyy_dz
            + l2m * dszz_dz
            + mu_  * dsxz_dx
            + mu_  * dsyz_dy
        );

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


    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];
    f.m_sxxy[idx] = ayh * f.m_sxxy[idx] + byh * dsxx_dy;
    dsxx_dy += f.m_sxxy[idx];
    f.m_sxxz[idx] = azh * f.m_sxxz[idx] + bzh * dsxx_dz;
    dsxx_dz += f.m_sxxz[idx];

    f.m_sxyx[idx] = ax * f.m_sxyx[idx] + bx * dsxy_dx;
    dsxy_dx += f.m_sxyx[idx];
    f.m_sxyy[idx] = ay * f.m_sxyy[idx] + by * dsxy_dy;
    dsxy_dy += f.m_sxyy[idx];

    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];
    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];

    f.m_syyx[idx] = ax * f.m_syyx[idx] + bx * dsyy_dx;
    dsyy_dx += f.m_syyx[idx];
    f.m_syyy[idx] = ayh * f.m_syyy[idx] + byh * dsyy_dy;
    dsyy_dy += f.m_syyy[idx];
    f.m_syyz[idx] = az * f.m_syyz[idx] + bz * dsyy_dz;
    dsyy_dz += f.m_syyz[idx];

    f.m_syzy[idx] = ay * f.m_syzy[idx] + by * dsyz_dy;
    dsyz_dy += f.m_syzy[idx];
    f.m_syzz[idx] = az * f.m_syzz[idx] + bz * dsyz_dz;
    dsyz_dz += f.m_syzz[idx];

    f.m_szzx[idx] = ax * f.m_szzx[idx] + bx * dszz_dx;
    dszz_dx += f.m_szzx[idx];
    f.m_szzy[idx] = ay * f.m_szzy[idx] + by * dszz_dy;
    dszz_dy += f.m_szzy[idx];
    f.m_szzz[idx] = az * f.m_szzz[idx] + bz * dszz_dz;
    dszz_dz += f.m_szzz[idx];

    // ===== vx adjoint =====
    f.vx[idx] += solver.dt * (
        l2m * dsxx_dx
        + lam * dsyy_dx
        + lam * dszz_dx
        + mu_  * dsxy_dy
        + mu_  * dsxz_dz
    );

    // ===== vy adjoint =====
    f.vy[idx] += solver.dt * (
        lam * dsxx_dy
        + l2m * dsyy_dy
        + lam * dszz_dy
        + mu_  * dsxy_dx
        + mu_  * dsyz_dz
    );

    // ===== vz adjoint =====
    f.vz[idx] += solver.dt * (
        lam * dsxx_dz
        + lam * dsyy_dz
        + l2m * dszz_dz
        + mu_  * dsxz_dx
        + mu_  * dsyz_dy
    );

}

template<int Order>
__global__ void elastic_velocity_adjoint_prepare_3d(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ pxx,
    float* __restrict__ pxy,
    float* __restrict__ pxz,
    float* __restrict__ pyx,
    float* __restrict__ pyy,
    float* __restrict__ pyz,
    float* __restrict__ pzx,
    float* __restrict__ pzy,
    float* __restrict__ pzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    const float* rho_b = rho + b * spatial_size;
    auto f = wf.offset(b, spatial_size);
    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float inv_rho = 1.f / rho_b[idx];

    float bar_dsxx_dx = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxy_dy = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dz = solver.dt * inv_rho * f.vx[idx];

    float bar_dsxy_dx = solver.dt * inv_rho * f.vy[idx];
    float bar_dsyy_dy = solver.dt * inv_rho * f.vy[idx];
    float bar_dsyz_dz = solver.dt * inv_rho * f.vy[idx];

    float bar_dsxz_dx = solver.dt * inv_rho * f.vz[idx];
    float bar_dsyz_dy = solver.dt * inv_rho * f.vz[idx];
    float bar_dszz_dz = solver.dt * inv_rho * f.vz[idx];

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxyy = f.m_sxyy[idx] + bar_dsxy_dy;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxyx = f.m_sxyx[idx] + bar_dsxy_dx;
    float tmp_syyy = f.m_syyy[idx] + bar_dsyy_dy;
    float tmp_syzz = f.m_syzz[idx] + bar_dsyz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_syzy = f.m_syzy[idx] + bar_dsyz_dy;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        pxx[b * spatial_size + idx] = bar_dsxx_dx;
        pxy[b * spatial_size + idx] = bar_dsxy_dy;
        pxz[b * spatial_size + idx] = bar_dsxz_dz;
        pyx[b * spatial_size + idx] = bar_dsxy_dx;
        pyy[b * spatial_size + idx] = bar_dsyy_dy;
        pyz[b * spatial_size + idx] = bar_dsyz_dz;
        pzx[b * spatial_size + idx] = bar_dsxz_dx;
        pzy[b * spatial_size + idx] = bar_dsyz_dy;
        pzz[b * spatial_size + idx] = bar_dszz_dz;
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

    pxx[b * spatial_size + idx] = bar_dsxx_dx + bxh * tmp_sxxx;
    pxy[b * spatial_size + idx] = bar_dsxy_dy + by * tmp_sxyy;
    pxz[b * spatial_size + idx] = bar_dsxz_dz + bz * tmp_sxzz;
    pyx[b * spatial_size + idx] = bar_dsxy_dx + bx * tmp_sxyx;
    pyy[b * spatial_size + idx] = bar_dsyy_dy + byh * tmp_syyy;
    pyz[b * spatial_size + idx] = bar_dsyz_dz + bz * tmp_syzz;
    pzx[b * spatial_size + idx] = bar_dsxz_dx + bx * tmp_sxzx;
    pzy[b * spatial_size + idx] = bar_dsyz_dy + by * tmp_syzy;
    pzz[b * spatial_size + idx] = bar_dszz_dz + bzh * tmp_szzz;

    f.m_sxxx[idx] = axh * tmp_sxxx;
    f.m_sxyy[idx] = ay * tmp_sxyy;
    f.m_sxzz[idx] = az * tmp_sxzz;
    f.m_sxyx[idx] = ax * tmp_sxyx;
    f.m_syyy[idx] = ayh * tmp_syyy;
    f.m_syzz[idx] = az * tmp_syzz;
    f.m_sxzx[idx] = ax * tmp_sxzx;
    f.m_syzy[idx] = ay * tmp_syzy;
    f.m_szzz[idx] = azh * tmp_szzz;
}

template<int Order>
__global__ void elastic_velocity_adjoint_apply_3d(
    ElasticWavefieldPointer wf,
    const float* __restrict__ pxx,
    const float* __restrict__ pxy,
    const float* __restrict__ pxz,
    const float* __restrict__ pyx,
    const float* __restrict__ pyy,
    const float* __restrict__ pyz,
    const float* __restrict__ pzx,
    const float* __restrict__ pzy,
    const float* __restrict__ pzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    const float* pxx_b = pxx + b * spatial_size;
    const float* pxy_b = pxy + b * spatial_size;
    const float* pxz_b = pxz + b * spatial_size;
    const float* pyx_b = pyx + b * spatial_size;
    const float* pyy_b = pyy + b * spatial_size;
    const float* pyz_b = pyz + b * spatial_size;
    const float* pzx_b = pzx + b * spatial_size;
    const float* pzy_b = pzy + b * spatial_size;
    const float* pzz_b = pzz + b * spatial_size;

    float dpxx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(pxx_b, ix, iy, iz, grad_ctx);
    float dpxy_dy = sgradient<3, Order, Y, DIFF_FORWARD>(pxy_b, ix, iy, iz, grad_ctx);
    float dpxz_dz = elastic3d_top_fs_adjoint_sgradient_z<Order, DIFF_BACKWARD>(pxz_b, ix, iy, iz, grad_ctx, solver, true);

    float dpyx_dx = sgradient<3, Order, X, DIFF_FORWARD>(pyx_b, ix, iy, iz, grad_ctx);
    float dpyy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(pyy_b, ix, iy, iz, grad_ctx);
    float dpyz_dz = elastic3d_top_fs_adjoint_sgradient_z<Order, DIFF_BACKWARD>(pyz_b, ix, iy, iz, grad_ctx, solver, true);

    float dpzx_dx = sgradient<3, Order, X, DIFF_FORWARD>(pzx_b, ix, iy, iz, grad_ctx);
    float dpzy_dy = sgradient<3, Order, Y, DIFF_FORWARD>(pzy_b, ix, iy, iz, grad_ctx);
    float dpzz_dz = elastic3d_top_fs_adjoint_sgradient_z<Order, DIFF_FORWARD>(pzz_b, ix, iy, iz, grad_ctx, solver, true);

    f.sxx[idx] += dpxx_dx;
    f.sxy[idx] += dpxy_dy + dpyx_dx;
    f.sxz[idx] += dpxz_dz + dpzx_dx;
    f.syy[idx] += dpyy_dy;
    f.syz[idx] += dpyz_dz + dpzy_dy;
    f.szz[idx] += dpzz_dz;
}


template<int Order>
__global__ void elastic_stress_adjoint_kernel_3d(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx =
        iz * solver.nx * solver.ny +
        iy * solver.nx +
        ix;

    const float* rho_b = rho + b * spatial_size;


    auto f = wf.offset(b, spatial_size);
    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    float inv_rho = 1.f / rho_b[idx];

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        f.sxx[idx] += solver.dt * inv_rho * dvx_dx;
        f.syy[idx] += solver.dt * inv_rho * dvy_dy;
        f.szz[idx] += solver.dt * inv_rho * dvz_dz;

        f.sxy[idx] += solver.dt * inv_rho * (dvx_dy + dvy_dx);
        f.sxz[idx] += solver.dt * inv_rho * (dvx_dz + dvz_dx);
        f.syz[idx] += solver.dt * inv_rho * (dvy_dz + dvz_dy);

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

    // ===== Normal stresses =====
    f.sxx[idx] += solver.dt * inv_rho * dvx_dx;
    f.syy[idx] += solver.dt * inv_rho * dvy_dy;
    f.szz[idx] += solver.dt * inv_rho * dvz_dz;

    // ===== Shear stresses =====
    f.sxy[idx] += solver.dt * inv_rho * (dvx_dy + dvy_dx);
    f.sxz[idx] += solver.dt * inv_rho * (dvx_dz + dvz_dx);
    f.syz[idx] += solver.dt * inv_rho * (dvy_dz + dvz_dy);

}

template<int Order>
__global__ void elastic_stress_adjoint_prepare_3d(
    ElasticWavefieldPointer wf,
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

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b = mu + b * spatial_size;
    auto f = wf.offset(b, spatial_size);
    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

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

    float bar_dvx_dx = solver.dt * (l2m * bar_sxx + lam * bar_syy + lam * bar_szz);
    float bar_dvx_dy = solver.dt * mu_ * bar_sxy;
    float bar_dvx_dz = solver.dt * mu_ * bar_sxz;

    float bar_dvy_dx = solver.dt * mu_ * bar_sxy;
    float bar_dvy_dy = solver.dt * (lam * bar_sxx + l2m * bar_syy + lam * bar_szz);
    float bar_dvy_dz = solver.dt * mu_ * bar_syz;

    float bar_dvz_dx = solver.dt * mu_ * bar_sxz;
    float bar_dvz_dy = solver.dt * mu_ * bar_syz;
    float bar_dvz_dz = solver.dt * (lam * bar_sxx + lam * bar_syy + l2m * bar_szz);

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vxy = f.m_vxy[idx] + bar_dvx_dy;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vyx = f.m_vyx[idx] + bar_dvy_dx;
    float tmp_vyy = f.m_vyy[idx] + bar_dvy_dy;
    float tmp_vyz = f.m_vyz[idx] + bar_dvy_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;
    float tmp_vzy = f.m_vzy[idx] + bar_dvz_dy;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;

    bool interior =
        ix >= x0 + 1 && ix < x1 - 1 &&
        iy >= y0 + 1 && iy < y1 - 1 &&
        iz >= z0 + 1 && iz < z1 - 1;

    if (interior) {
        qxx[b * spatial_size + idx] = bar_dvx_dx;
        qxy[b * spatial_size + idx] = bar_dvx_dy;
        qxz[b * spatial_size + idx] = bar_dvx_dz;
        qyx[b * spatial_size + idx] = bar_dvy_dx;
        qyy[b * spatial_size + idx] = bar_dvy_dy;
        qyz[b * spatial_size + idx] = bar_dvy_dz;
        qzx[b * spatial_size + idx] = bar_dvz_dx;
        qzy[b * spatial_size + idx] = bar_dvz_dy;
        qzz[b * spatial_size + idx] = bar_dvz_dz;
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

    qxx[b * spatial_size + idx] = bar_dvx_dx + bx * tmp_vxx;
    qxy[b * spatial_size + idx] = bar_dvx_dy + byh * tmp_vxy;
    qxz[b * spatial_size + idx] = bar_dvx_dz + bzh * tmp_vxz;
    qyx[b * spatial_size + idx] = bar_dvy_dx + bxh * tmp_vyx;
    qyy[b * spatial_size + idx] = bar_dvy_dy + by * tmp_vyy;
    qyz[b * spatial_size + idx] = bar_dvy_dz + bzh * tmp_vyz;
    qzx[b * spatial_size + idx] = bar_dvz_dx + bxh * tmp_vzx;
    qzy[b * spatial_size + idx] = bar_dvz_dy + byh * tmp_vzy;
    qzz[b * spatial_size + idx] = bar_dvz_dz + bz * tmp_vzz;

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

template<int Order>
__global__ void elastic_stress_adjoint_apply_3d(
    ElasticWavefieldPointer wf,
    const float* __restrict__ qxx,
    const float* __restrict__ qxy,
    const float* __restrict__ qxz,
    const float* __restrict__ qyx,
    const float* __restrict__ qyy,
    const float* __restrict__ qyz,
    const float* __restrict__ qzx,
    const float* __restrict__ qzy,
    const float* __restrict__ qzz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    const float* qxx_b = qxx + b * spatial_size;
    const float* qxy_b = qxy + b * spatial_size;
    const float* qxz_b = qxz + b * spatial_size;
    const float* qyx_b = qyx + b * spatial_size;
    const float* qyy_b = qyy + b * spatial_size;
    const float* qyz_b = qyz + b * spatial_size;
    const float* qzx_b = qzx + b * spatial_size;
    const float* qzy_b = qzy + b * spatial_size;
    const float* qzz_b = qzz + b * spatial_size;

    float dqxx_dx = sgradient<3, Order, X, DIFF_FORWARD>(qxx_b, ix, iy, iz, grad_ctx);
    float dqxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(qxy_b, ix, iy, iz, grad_ctx);
    float dqxz_dz = elastic3d_top_fs_adjoint_sgradient_z<Order, DIFF_FORWARD>(qxz_b, ix, iy, iz, grad_ctx, solver, false);

    float dqyx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(qyx_b, ix, iy, iz, grad_ctx);
    float dqyy_dy = sgradient<3, Order, Y, DIFF_FORWARD>(qyy_b, ix, iy, iz, grad_ctx);
    float dqyz_dz = elastic3d_top_fs_adjoint_sgradient_z<Order, DIFF_FORWARD>(qyz_b, ix, iy, iz, grad_ctx, solver, false);

    float dqzx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(qzx_b, ix, iy, iz, grad_ctx);
    float dqzy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(qzy_b, ix, iy, iz, grad_ctx);
    float dqzz_dz = elastic3d_top_fs_adjoint_sgradient_z<Order, DIFF_BACKWARD>(qzz_b, ix, iy, iz, grad_ctx, solver, true);

    f.vx[idx] += dqxx_dx + dqxy_dy + dqxz_dz;
    f.vy[idx] += dqyx_dx + dqyy_dy + dqyz_dz;
    f.vz[idx] += dqzx_dx + dqzy_dy + dqzz_dz;
}


template<int Order>
__global__ void calculate_grad_elastic3d_bs(

    ElasticWavefieldPointer forward,
    ElasticWavefieldPointer adjoint,

    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvy_prev,
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
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx =
        iz * solver.nx * solver.ny +
        iy * solver.nx +
        ix;

    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvy_prev_b = fvy_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;

    auto f = forward.offset(b, spatial_size);
    auto a = adjoint.offset(b, spatial_size);

    const float* vp_b = vp + b * spatial_size;
    const float* vs_b = vs + b * spatial_size;
    const float* rho_b = rho + b * spatial_size;

    float* gvp = grad_vp + b * spatial_size;
    float* gvs = grad_vs + b * spatial_size;
    float* grho = grad_rho + b * spatial_size;

    float fvx_x = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float fvx_y = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float fvx_z = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx, solver, false);

    float fvy_x = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float fvy_y = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float fvy_z = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx, solver, false);

    float fvz_x = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float fvz_y = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float fvz_z = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx, solver, true);

    float bar_szz = elastic3d_is_top_free_surface_row(solver, iz) ? 0.f : a.szz[idx];
    float bar_sxz = elastic3d_is_top_free_surface_row(solver, iz) ? 0.f : a.sxz[idx];
    float bar_syz = elastic3d_is_top_free_surface_row(solver, iz) ? 0.f : a.syz[idx];
    float grad_lambda = (a.sxx[idx] + a.syy[idx] + bar_szz) * (fvx_x + fvy_y + fvz_z);
    float grad_mu = 2*(a.sxx[idx] * fvx_x +
                       a.syy[idx] * fvy_y +
                       bar_szz * fvz_z) +
                       bar_sxz * (fvx_z + fvz_x) +
                       a.sxy[idx] * (fvx_y + fvy_x) +
                       bar_syz * (fvy_z + fvz_y);

    gvp[idx] +=   -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    gvs[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                   2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

    grho[idx] += (a.vx[idx] * (f.vx[idx]-fvx_prev_b[idx]) +
                  a.vy[idx] * (f.vy[idx]-fvy_prev_b[idx]) +
                  a.vz[idx] * (f.vz[idx]-fvz_prev_b[idx])) / rho_b[idx];
    grho[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                 grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;
}


// ===========================================================================
// APM (Dong 2023, elastic limit) 3-D kernels
// ===========================================================================
// Category codes mirror sweep.equations._topography:
//   INTERIOR=0, AIR=1, H=2, VL=3, VR=4, OC=5, IC=6, VF=7, VB=8
// The 3-D APM step uses the Virieux SSG stencil with parameter-modified
// moduli (9 normal + 3 shear + 3 inverse densities, all pre-computed on
// the Python side).  Image-method z-derivative substitutions are disabled
// (``solver.free_surface=false`` in APM mode, so the existing
// ``elastic3d_top_fs_sgradient_z`` helpers fall through to plain
// ``sgradient``).

#define APM3D_CATEGORY_INTERIOR 0
#define APM3D_CATEGORY_AIR      1
#define APM3D_CATEGORY_H        2
#define APM3D_CATEGORY_VL       3
#define APM3D_CATEGORY_VR       4
#define APM3D_CATEGORY_OC       5
#define APM3D_CATEGORY_IC       6
#define APM3D_CATEGORY_VF       7
#define APM3D_CATEGORY_VB       8


template<int Order>
__global__ void elastic3d_velocity_kernel_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ inv_rho_x,
    const float* __restrict__ inv_rho_y,
    const float* __restrict__ inv_rho_z,
    const int*   __restrict__ category,
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
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    // AIR cell: wipe velocity components, skip update.  ``inv_rho_*`` is
    // already zero at air cells (masked on the Python side) but be
    // explicit here too in case category > AIR was set for some reason.
    if (category[idx] == APM3D_CATEGORY_AIR) {
        f.vx[idx] = 0.f;
        f.vy[idx] = 0.f;
        f.vz[idx] = 0.f;
        return;
    }

    const float* invrx_b = inv_rho_x + b * spatial_size;
    const float* invry_b = inv_rho_y + b * spatial_size;
    const float* invrz_b = inv_rho_z + b * spatial_size;

    // No image-method z-derivative substitution: solver.free_surface=false
    // in APM mode → helpers fall through to plain sgradient.
    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx, solver, true);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx, solver, true);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx, solver, true);

    float irx = invrx_b[idx];
    float iry = invry_b[idx];
    float irz = invrz_b[idx];

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.vx[idx] += solver.dt * irx * (dsxx_dx + dsxy_dy + dsxz_dz);
        f.vy[idx] += solver.dt * iry * (dsxy_dx + dsyy_dy + dsyz_dz);
        f.vz[idx] += solver.dt * irz * (dsxz_dx + dsyz_dy + dszz_dz);
        return;
    }

    // PML CPML aux update (mirror elastic_velocity_kernel_3d).
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

    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];
    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];

    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];
    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];

    f.m_sxyy[idx] = ay * f.m_sxyy[idx] + by * dsxy_dy;
    dsxy_dy += f.m_sxyy[idx];
    f.m_sxyx[idx] = ax * f.m_sxyx[idx] + bx * dsxy_dx;
    dsxy_dx += f.m_sxyx[idx];

    f.m_syyy[idx] = ayh * f.m_syyy[idx] + byh * dsyy_dy;
    dsyy_dy += f.m_syyy[idx];
    f.m_syzz[idx] = az * f.m_syzz[idx] + bz * dsyz_dz;
    dsyz_dz += f.m_syzz[idx];
    f.m_syzy[idx] = ay * f.m_syzy[idx] + by * dsyz_dy;
    dsyz_dy += f.m_syzy[idx];

    f.vx[idx] += solver.dt * irx * (dsxx_dx + dsxy_dy + dsxz_dz);
    f.vy[idx] += solver.dt * iry * (dsxy_dx + dsyy_dy + dsyz_dz);
    f.vz[idx] += solver.dt * irz * (dsxz_dx + dsyz_dy + dszz_dz);
}


template<int Order>
__global__ void elastic3d_stress_kernel_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ alpha_xx,
    const float* __restrict__ alpha_yy,
    const float* __restrict__ alpha_zz,
    const float* __restrict__ lam_xx_yy,
    const float* __restrict__ lam_xx_zz,
    const float* __restrict__ lam_yy_xx,
    const float* __restrict__ lam_yy_zz,
    const float* __restrict__ lam_zz_xx,
    const float* __restrict__ lam_zz_yy,
    const float* __restrict__ mu_xy_node,
    const float* __restrict__ mu_xz_node,
    const float* __restrict__ mu_yz_node,
    const int*   __restrict__ category,
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
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    int cat = category[idx];

    // AIR cell: wipe stresses, save vx/vy/vz checkpoints (they were
    // zeroed by the velocity kernel) and return.
    if (cat == APM3D_CATEGORY_AIR) {
        f.sxx[idx] = 0.f; f.syy[idx] = 0.f; f.szz[idx] = 0.f;
        f.sxy[idx] = 0.f; f.sxz[idx] = 0.f; f.syz[idx] = 0.f;
        if (u_this_b) {
            int comp_stride = solver.B * spatial_size;
            u_this_b[0 * comp_stride + idx] = 0.f;
            u_this_b[1 * comp_stride + idx] = 0.f;
            u_this_b[2 * comp_stride + idx] = 0.f;
        }
        return;
    }

    const float* axx_b   = alpha_xx   + b * spatial_size;
    const float* ayy_b   = alpha_yy   + b * spatial_size;
    const float* azz_b   = alpha_zz   + b * spatial_size;
    const float* lxxyy_b = lam_xx_yy  + b * spatial_size;
    const float* lxxzz_b = lam_xx_zz  + b * spatial_size;
    const float* lyyxx_b = lam_yy_xx  + b * spatial_size;
    const float* lyyzz_b = lam_yy_zz  + b * spatial_size;
    const float* lzzxx_b = lam_zz_xx  + b * spatial_size;
    const float* lzzyy_b = lam_zz_yy  + b * spatial_size;
    const float* muxy_b  = mu_xy_node + b * spatial_size;
    const float* muxz_b  = mu_xz_node + b * spatial_size;
    const float* muyz_b  = mu_yz_node + b * spatial_size;

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx, solver, false);
    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx, solver, false);
    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx, solver, true);

    float axx  = axx_b[idx],   ayy  = ayy_b[idx],   azz  = azz_b[idx];
    float lxy  = lxxyy_b[idx], lxz  = lxxzz_b[idx];
    float lyx  = lyyxx_b[idx], lyz  = lyyzz_b[idx];
    float lzx  = lzzxx_b[idx], lzy  = lzzyy_b[idx];
    float mxy  = muxy_b[idx],  mxz  = muxz_b[idx],  myz  = muyz_b[idx];

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.sxx[idx] += solver.dt * (axx * dvx_dx + lxy * dvy_dy + lxz * dvz_dz);
        f.syy[idx] += solver.dt * (lyx * dvx_dx + ayy * dvy_dy + lyz * dvz_dz);
        f.szz[idx] += solver.dt * (lzx * dvx_dx + lzy * dvy_dy + azz * dvz_dz);
        f.sxy[idx] += solver.dt * mxy * (dvx_dy + dvy_dx);
        f.sxz[idx] += solver.dt * mxz * (dvx_dz + dvz_dx);
        f.syz[idx] += solver.dt * myz * (dvy_dz + dvz_dy);
    } else {
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

        f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
        dvx_dx += f.m_vxx[idx];
        f.m_vyy[idx] = ay * f.m_vyy[idx] + by * dvy_dy;
        dvy_dy += f.m_vyy[idx];
        f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
        dvz_dz += f.m_vzz[idx];

        f.sxx[idx] += solver.dt * (axx * dvx_dx + lxy * dvy_dy + lxz * dvz_dz);
        f.syy[idx] += solver.dt * (lyx * dvx_dx + ayy * dvy_dy + lyz * dvz_dz);
        f.szz[idx] += solver.dt * (lzx * dvx_dx + lzy * dvy_dy + azz * dvz_dz);

        f.m_vxy[idx] = ayh * f.m_vxy[idx] + byh * dvx_dy;
        dvx_dy += f.m_vxy[idx];
        f.m_vyx[idx] = axh * f.m_vyx[idx] + bxh * dvy_dx;
        dvy_dx += f.m_vyx[idx];
        f.sxy[idx] += solver.dt * mxy * (dvx_dy + dvy_dx);

        f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
        dvx_dz += f.m_vxz[idx];
        f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
        dvz_dx += f.m_vzx[idx];
        f.sxz[idx] += solver.dt * mxz * (dvx_dz + dvz_dx);

        f.m_vyz[idx] = azh * f.m_vyz[idx] + bzh * dvy_dz;
        dvy_dz += f.m_vyz[idx];
        f.m_vzy[idx] = ayh * f.m_vzy[idx] + byh * dvz_dy;
        dvz_dy += f.m_vzy[idx];
        f.syz[idx] += solver.dt * myz * (dvy_dz + dvz_dy);
    }

    // APM traction-free BC: pointwise stress zero per surface category.
    // Rules mirror enforce_apm_traction_bc_3d:
    //   H        → σ_zz = σ_xz = σ_yz = 0  (free face ±z)
    //   VL/VR    → σ_xx = σ_xy = σ_xz = 0  (free face ±x)
    //   VF/VB    → σ_yy = σ_xy = σ_yz = 0  (free face ±y)
    //   OC       → all 6 stresses zero
    if (cat == APM3D_CATEGORY_H) {
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
        f.syz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_VL || cat == APM3D_CATEGORY_VR) {
        f.sxx[idx] = 0.f;
        f.sxy[idx] = 0.f;
        f.sxz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_VF || cat == APM3D_CATEGORY_VB) {
        f.syy[idx] = 0.f;
        f.sxy[idx] = 0.f;
        f.syz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_OC) {
        f.sxx[idx] = 0.f; f.syy[idx] = 0.f; f.szz[idx] = 0.f;
        f.sxy[idx] = 0.f; f.sxz[idx] = 0.f; f.syz[idx] = 0.f;
    }

    if (u_this_b) {
        int comp_stride = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vy[idx];
        u_this_b[2 * comp_stride + idx] = f.vz[idx];
    }
}


#define LAUNCH_3DELASTIC_VELOCITY_APM(order, grid, block, ...)                 \
    do {                                                                        \
        if      ((order) == 2) elastic3d_velocity_kernel_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic3d_velocity_kernel_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic3d_velocity_kernel_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic3d_velocity_kernel_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic3d_velocity_kernel_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_STRESS_APM(order, grid, block, ...)                   \
    do {                                                                        \
        if      ((order) == 2) elastic3d_stress_kernel_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic3d_stress_kernel_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic3d_stress_kernel_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic3d_stress_kernel_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic3d_stress_kernel_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


// ===========================================================================
// APM (Dong 2023, elastic limit) 3-D BACKWARD kernels
// ===========================================================================
// Mirror the 2-D APM backward + image-method 3-D backward structures.
//
//   - elastic3d_velocity_kernel_nopml_apm — reverse forward replay (bs)
//   - elastic3d_stress_kernel_nopml_apm   — reverse forward replay (bs)
//   - elastic3d_stress_adjoint_prepare_apm  — adjoint, APM moduli
//   - elastic3d_velocity_adjoint_prepare_apm — adjoint, APM 1/rho
//   - calculate_grad_elastic3d_apm_bs     — gradient (bs path)
//   - calculate_grad_elastic3d_apm_nobs   — gradient (full path)
//
// The two *_adjoint_apply_3d kernels are reused from the image path
// (they only sum spatial derivatives, no moduli involved).
//
// Chain-rule helpers below map gradients w.r.t. the 15 effective moduli
// (alpha_xx/yy/zz, lam_*_*, mu_xy/xz/yz) back to gradients w.r.t. raw
// (lambda, mu).  Derivatives of alpha = 2μ(λ+μ)/η and β = λμ/η:
//   ∂α/∂λ = 2μ²/η²,        ∂α/∂μ = 2(λ² + 2λμ + 2μ²)/η²
//   ∂β/∂λ = 2μ²/η²,        ∂β/∂μ = λ²/η²
// where η = λ + 2μ.

__device__ __forceinline__ void apm3d_chain_lammu(
    int cat, float lam_r, float mu_r,
    float g_axx, float g_ayy, float g_azz,
    float g_lxy, float g_lxz, float g_lyx, float g_lyz, float g_lzx, float g_lzy,
    float g_mxy, float g_mxz, float g_myz,
    float* grad_lam_out, float* grad_mu_out)
{
    if (cat == APM3D_CATEGORY_INTERIOR || cat == APM3D_CATEGORY_IC) {
        // alpha_NN = eta = lam + 2mu;  lam_NN_MM = lam;  mu_node = mu
        *grad_lam_out = g_axx + g_ayy + g_azz
                       + g_lxy + g_lxz + g_lyx + g_lyz + g_lzx + g_lzy;
        *grad_mu_out  = 2.f * (g_axx + g_ayy + g_azz)
                       + g_mxy + g_mxz + g_myz;
        return;
    }
    if (cat == APM3D_CATEGORY_OC || cat == APM3D_CATEGORY_AIR) {
        *grad_lam_out = 0.f;
        *grad_mu_out  = 0.f;
        return;
    }

    // H / VL / VR / VF / VB — all use alpha + beta derivatives.
    float eta = lam_r + 2.f * mu_r;
    float safe = (eta > 0.f) ? eta : 1.f;
    float inv2 = 1.f / (safe * safe);
    float dalpha_dlam = 2.f * mu_r * mu_r * inv2;
    float dalpha_dmu  = 2.f * (lam_r * lam_r + 2.f * lam_r * mu_r + 2.f * mu_r * mu_r) * inv2;
    float dbeta_dlam  = 2.f * mu_r * mu_r * inv2;            // identical to dα/dλ
    float dbeta_dmu   = lam_r * lam_r * inv2;

    float gl = 0.f, gm = 0.f;
    if (cat == APM3D_CATEGORY_H) {
        // alpha_xx = alpha_yy = α;  alpha_zz = 0
        // lam_xx_yy = lam_yy_xx = β;  others zero
        // mu_xy = μ/2;  mu_xz = mu_yz = μ
        gl = (g_axx + g_ayy) * dalpha_dlam + (g_lxy + g_lyx) * dbeta_dlam;
        gm = (g_axx + g_ayy) * dalpha_dmu  + (g_lxy + g_lyx) * dbeta_dmu
             + g_mxy * 0.5f + g_mxz + g_myz;
    } else if (cat == APM3D_CATEGORY_VL) {
        // alpha_yy = alpha_zz = α;  alpha_xx = 0
        // lam_yy_zz = lam_zz_yy = β;  others zero
        // mu_xy = μ;  mu_xz = μ;  mu_yz = μ/2
        gl = (g_ayy + g_azz) * dalpha_dlam + (g_lyz + g_lzy) * dbeta_dlam;
        gm = (g_ayy + g_azz) * dalpha_dmu  + (g_lyz + g_lzy) * dbeta_dmu
             + g_mxy + g_mxz + g_myz * 0.5f;
    } else if (cat == APM3D_CATEGORY_VR) {
        // Same modulus structure as VL but mu_xy = mu_xz = 0 (table 6.1).
        gl = (g_ayy + g_azz) * dalpha_dlam + (g_lyz + g_lzy) * dbeta_dlam;
        gm = (g_ayy + g_azz) * dalpha_dmu  + (g_lyz + g_lzy) * dbeta_dmu
             + g_myz * 0.5f;
    } else if (cat == APM3D_CATEGORY_VF) {
        // alpha_xx = alpha_zz = α;  alpha_yy = 0
        // lam_xx_zz = lam_zz_xx = β;  others zero
        // mu_xy = μ;  mu_xz = μ/2;  mu_yz = μ
        gl = (g_axx + g_azz) * dalpha_dlam + (g_lxz + g_lzx) * dbeta_dlam;
        gm = (g_axx + g_azz) * dalpha_dmu  + (g_lxz + g_lzx) * dbeta_dmu
             + g_mxy + g_mxz * 0.5f + g_myz;
    } else if (cat == APM3D_CATEGORY_VB) {
        // Same as VF but mu_xy = mu_yz = 0.
        gl = (g_axx + g_azz) * dalpha_dlam + (g_lxz + g_lzx) * dbeta_dlam;
        gm = (g_axx + g_azz) * dalpha_dmu  + (g_lxz + g_lzx) * dbeta_dmu
             + g_mxz * 0.5f;
    }
    *grad_lam_out = gl;
    *grad_mu_out  = gm;
}

__device__ __forceinline__ void apm3d_rho_jacobian(int cat, float* dx, float* dy, float* dz)
{
    // Per Dong 2023 Table 1 density rules transcribed in
    // _topography.precompute_apm_moduli_3d.
    switch (cat) {
        case APM3D_CATEGORY_INTERIOR:
        case APM3D_CATEGORY_AIR:
            *dx = 1.f; *dy = 1.f; *dz = 1.f; break;
        case APM3D_CATEGORY_H:
            *dx = 0.5f; *dy = 0.5f; *dz = 1.f; break;
        case APM3D_CATEGORY_VL:
            *dx = 1.f;  *dy = 0.5f; *dz = 0.5f; break;
        case APM3D_CATEGORY_VR:
            *dx = 0.f;  *dy = 0.5f; *dz = 0.5f; break;
        case APM3D_CATEGORY_VF:
            *dx = 0.5f; *dy = 1.f;  *dz = 0.5f; break;
        case APM3D_CATEGORY_VB:
            *dx = 0.5f; *dy = 0.f;  *dz = 0.5f; break;
        case APM3D_CATEGORY_OC:
            *dx = 0.25f; *dy = 0.25f; *dz = 0.25f; break;
        case APM3D_CATEGORY_IC:
            *dx = 0.75f; *dy = 0.75f; *dz = 0.75f; break;
        default:
            *dx = 1.f; *dy = 1.f; *dz = 1.f; break;
    }
}


// --- Reverse-time forward replay (apm_backward_bs) -------------------------

template<int Order>
__global__ void elastic3d_velocity_kernel_nopml_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ inv_rho_x,
    const float* __restrict__ inv_rho_y,
    const float* __restrict__ inv_rho_z,
    const int*   __restrict__ category,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) { M = solver.M; } else { M = Order / 2; }
    int halo = solver.abcn + M + 1;
    int top_halo = solver.free_surface ? M : halo;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < top_halo || iz >= solver.nz - halo) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    if (category[idx] == APM3D_CATEGORY_AIR) {
        f.vx[idx] = 0.f; f.vy[idx] = 0.f; f.vz[idx] = 0.f;
        return;
    }

    const float* invrx_b = inv_rho_x + b * spatial_size;
    const float* invry_b = inv_rho_y + b * spatial_size;
    const float* invrz_b = inv_rho_z + b * spatial_size;

    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx, solver, true);
    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx, solver, true);
    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx, solver, true);

    // Reverse step: subtract instead of add.
    f.vx[idx] -= solver.dt * invrx_b[idx] * (dsxx_dx + dsxy_dy + dsxz_dz);
    f.vy[idx] -= solver.dt * invry_b[idx] * (dsxy_dx + dsyy_dy + dsyz_dz);
    f.vz[idx] -= solver.dt * invrz_b[idx] * (dsxz_dx + dsyz_dy + dszz_dz);
}

template<int Order>
__global__ void elastic3d_stress_kernel_nopml_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ alpha_xx,
    const float* __restrict__ alpha_yy,
    const float* __restrict__ alpha_zz,
    const float* __restrict__ lam_xx_yy,
    const float* __restrict__ lam_xx_zz,
    const float* __restrict__ lam_yy_xx,
    const float* __restrict__ lam_yy_zz,
    const float* __restrict__ lam_zz_xx,
    const float* __restrict__ lam_zz_yy,
    const float* __restrict__ mu_xy_node,
    const float* __restrict__ mu_xz_node,
    const float* __restrict__ mu_yz_node,
    const int*   __restrict__ category,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) { M = solver.M; } else { M = Order / 2; }
    int halo = solver.abcn + M + 1;
    int top_halo = solver.free_surface ? M : halo;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < top_halo || iz >= solver.nz - halo) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    int cat = category[idx];

    if (cat == APM3D_CATEGORY_AIR) {
        f.sxx[idx] = 0.f; f.syy[idx] = 0.f; f.szz[idx] = 0.f;
        f.sxy[idx] = 0.f; f.sxz[idx] = 0.f; f.syz[idx] = 0.f;
        return;
    }

    const float* axx_b   = alpha_xx   + b * spatial_size;
    const float* ayy_b   = alpha_yy   + b * spatial_size;
    const float* azz_b   = alpha_zz   + b * spatial_size;
    const float* lxxyy_b = lam_xx_yy  + b * spatial_size;
    const float* lxxzz_b = lam_xx_zz  + b * spatial_size;
    const float* lyyxx_b = lam_yy_xx  + b * spatial_size;
    const float* lyyzz_b = lam_yy_zz  + b * spatial_size;
    const float* lzzxx_b = lam_zz_xx  + b * spatial_size;
    const float* lzzyy_b = lam_zz_yy  + b * spatial_size;
    const float* muxy_b  = mu_xy_node + b * spatial_size;
    const float* muxz_b  = mu_xz_node + b * spatial_size;
    const float* muyz_b  = mu_yz_node + b * spatial_size;

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx, solver, false);
    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx, solver, false);
    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = elastic3d_top_fs_sgradient_z<Order, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx, solver, true);

    float axx = axx_b[idx], ayy = ayy_b[idx], azz = azz_b[idx];
    float lxy = lxxyy_b[idx], lxz = lxxzz_b[idx];
    float lyx = lyyxx_b[idx], lyz = lyyzz_b[idx];
    float lzx = lzzxx_b[idx], lzy = lzzyy_b[idx];
    float mxy = muxy_b[idx], mxz = muxz_b[idx], myz = muyz_b[idx];

    // Reverse step: subtract.
    f.sxx[idx] -= solver.dt * (axx * dvx_dx + lxy * dvy_dy + lxz * dvz_dz);
    f.syy[idx] -= solver.dt * (lyx * dvx_dx + ayy * dvy_dy + lyz * dvz_dz);
    f.szz[idx] -= solver.dt * (lzx * dvx_dx + lzy * dvy_dy + azz * dvz_dz);
    f.sxy[idx] -= solver.dt * mxy * (dvx_dy + dvy_dx);
    f.sxz[idx] -= solver.dt * mxz * (dvx_dz + dvz_dx);
    f.syz[idx] -= solver.dt * myz * (dvy_dz + dvz_dy);

    // Re-apply traction BC (forward zeroed these after stress update;
    // reverse replay must keep the same invariant on the forward state).
    if (cat == APM3D_CATEGORY_H) {
        f.szz[idx] = 0.f; f.sxz[idx] = 0.f; f.syz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_VL || cat == APM3D_CATEGORY_VR) {
        f.sxx[idx] = 0.f; f.sxy[idx] = 0.f; f.sxz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_VF || cat == APM3D_CATEGORY_VB) {
        f.syy[idx] = 0.f; f.sxy[idx] = 0.f; f.syz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_OC) {
        f.sxx[idx] = 0.f; f.syy[idx] = 0.f; f.szz[idx] = 0.f;
        f.sxy[idx] = 0.f; f.sxz[idx] = 0.f; f.syz[idx] = 0.f;
    }
}

#define LAUNCH_3DELASTIC_VELOCITY_NOPML_APM(order, grid, block, ...)            \
    do {                                                                        \
        if      ((order) == 2) elastic3d_velocity_kernel_nopml_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic3d_velocity_kernel_nopml_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic3d_velocity_kernel_nopml_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic3d_velocity_kernel_nopml_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic3d_velocity_kernel_nopml_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_STRESS_NOPML_APM(order, grid, block, ...)              \
    do {                                                                        \
        if      ((order) == 2) elastic3d_stress_kernel_nopml_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic3d_stress_kernel_nopml_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic3d_stress_kernel_nopml_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic3d_stress_kernel_nopml_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic3d_stress_kernel_nopml_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


// --- Adjoint prepare kernels (APM) -----------------------------------------

template<int Order>
__global__ void elastic3d_stress_adjoint_prepare_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ alpha_xx,
    const float* __restrict__ alpha_yy,
    const float* __restrict__ alpha_zz,
    const float* __restrict__ lam_xx_yy,
    const float* __restrict__ lam_xx_zz,
    const float* __restrict__ lam_yy_xx,
    const float* __restrict__ lam_yy_zz,
    const float* __restrict__ lam_zz_xx,
    const float* __restrict__ lam_zz_yy,
    const float* __restrict__ mu_xy_node,
    const float* __restrict__ mu_xz_node,
    const float* __restrict__ mu_yz_node,
    const int*   __restrict__ category,
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
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    int cat = category[idx];

    float bar_sxx = f.sxx[idx];
    float bar_syy = f.syy[idx];
    float bar_szz = f.szz[idx];
    float bar_sxy = f.sxy[idx];
    float bar_sxz = f.sxz[idx];
    float bar_syz = f.syz[idx];

    // APM traction-BC adjoint: zero the bar_s* components per category.
    if (cat == APM3D_CATEGORY_H) {
        bar_szz = 0.f; bar_sxz = 0.f; bar_syz = 0.f;
        f.szz[idx] = 0.f; f.sxz[idx] = 0.f; f.syz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_VL || cat == APM3D_CATEGORY_VR) {
        bar_sxx = 0.f; bar_sxy = 0.f; bar_sxz = 0.f;
        f.sxx[idx] = 0.f; f.sxy[idx] = 0.f; f.sxz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_VF || cat == APM3D_CATEGORY_VB) {
        bar_syy = 0.f; bar_sxy = 0.f; bar_syz = 0.f;
        f.syy[idx] = 0.f; f.sxy[idx] = 0.f; f.syz[idx] = 0.f;
    } else if (cat == APM3D_CATEGORY_OC || cat == APM3D_CATEGORY_AIR) {
        bar_sxx = bar_syy = bar_szz = 0.f;
        bar_sxy = bar_sxz = bar_syz = 0.f;
        f.sxx[idx] = f.syy[idx] = f.szz[idx] = 0.f;
        f.sxy[idx] = f.sxz[idx] = f.syz[idx] = 0.f;
    }

    float axx = alpha_xx[b*spatial_size+idx];
    float ayy = alpha_yy[b*spatial_size+idx];
    float azz = alpha_zz[b*spatial_size+idx];
    float lxy = lam_xx_yy[b*spatial_size+idx];
    float lxz = lam_xx_zz[b*spatial_size+idx];
    float lyx = lam_yy_xx[b*spatial_size+idx];
    float lyz = lam_yy_zz[b*spatial_size+idx];
    float lzx = lam_zz_xx[b*spatial_size+idx];
    float lzy = lam_zz_yy[b*spatial_size+idx];
    float mxy = mu_xy_node[b*spatial_size+idx];
    float mxz = mu_xz_node[b*spatial_size+idx];
    float myz = mu_yz_node[b*spatial_size+idx];

    float bar_dvx_dx = solver.dt * (axx * bar_sxx + lyx * bar_syy + lzx * bar_szz);
    float bar_dvy_dy = solver.dt * (lxy * bar_sxx + ayy * bar_syy + lzy * bar_szz);
    float bar_dvz_dz = solver.dt * (lxz * bar_sxx + lyz * bar_syy + azz * bar_szz);
    float bar_dvx_dy = solver.dt * mxy * bar_sxy;
    float bar_dvy_dx = solver.dt * mxy * bar_sxy;
    float bar_dvx_dz = solver.dt * mxz * bar_sxz;
    float bar_dvz_dx = solver.dt * mxz * bar_sxz;
    float bar_dvy_dz = solver.dt * myz * bar_syz;
    float bar_dvz_dy = solver.dt * myz * bar_syz;

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
        qxx[b * spatial_size + idx] = bar_dvx_dx;
        qxy[b * spatial_size + idx] = bar_dvx_dy;
        qxz[b * spatial_size + idx] = bar_dvx_dz;
        qyx[b * spatial_size + idx] = bar_dvy_dx;
        qyy[b * spatial_size + idx] = bar_dvy_dy;
        qyz[b * spatial_size + idx] = bar_dvy_dz;
        qzx[b * spatial_size + idx] = bar_dvz_dx;
        qzy[b * spatial_size + idx] = bar_dvz_dy;
        qzz[b * spatial_size + idx] = bar_dvz_dz;
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

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vxy = f.m_vxy[idx] + bar_dvx_dy;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vyx = f.m_vyx[idx] + bar_dvy_dx;
    float tmp_vyy = f.m_vyy[idx] + bar_dvy_dy;
    float tmp_vyz = f.m_vyz[idx] + bar_dvy_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;
    float tmp_vzy = f.m_vzy[idx] + bar_dvz_dy;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;

    qxx[b * spatial_size + idx] = bar_dvx_dx + bx * tmp_vxx;
    qxy[b * spatial_size + idx] = bar_dvx_dy + byh * tmp_vxy;
    qxz[b * spatial_size + idx] = bar_dvx_dz + bzh * tmp_vxz;
    qyx[b * spatial_size + idx] = bar_dvy_dx + bxh * tmp_vyx;
    qyy[b * spatial_size + idx] = bar_dvy_dy + by * tmp_vyy;
    qyz[b * spatial_size + idx] = bar_dvy_dz + bzh * tmp_vyz;
    qzx[b * spatial_size + idx] = bar_dvz_dx + bxh * tmp_vzx;
    qzy[b * spatial_size + idx] = bar_dvz_dy + byh * tmp_vzy;
    qzz[b * spatial_size + idx] = bar_dvz_dz + bz * tmp_vzz;

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


template<int Order>
__global__ void elastic3d_velocity_adjoint_prepare_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ inv_rho_x,
    const float* __restrict__ inv_rho_y,
    const float* __restrict__ inv_rho_z,
    const int*   __restrict__ category,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ pxx,
    float* __restrict__ pxy,
    float* __restrict__ pxz,
    float* __restrict__ pyx,
    float* __restrict__ pyy,
    float* __restrict__ pyz,
    float* __restrict__ pzx,
    float* __restrict__ pzy,
    float* __restrict__ pzz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    int cat = category[idx];

    float bar_vx = f.vx[idx];
    float bar_vy = f.vy[idx];
    float bar_vz = f.vz[idx];
    if (cat == APM3D_CATEGORY_AIR) {
        bar_vx = bar_vy = bar_vz = 0.f;
        f.vx[idx] = 0.f; f.vy[idx] = 0.f; f.vz[idx] = 0.f;
    }

    float irx = inv_rho_x[b*spatial_size+idx];
    float iry = inv_rho_y[b*spatial_size+idx];
    float irz = inv_rho_z[b*spatial_size+idx];

    float bar_dsxx_dx = solver.dt * irx * bar_vx;
    float bar_dsxy_dy = solver.dt * irx * bar_vx;
    float bar_dsxz_dz = solver.dt * irx * bar_vx;
    float bar_dsxy_dx = solver.dt * iry * bar_vy;
    float bar_dsyy_dy = solver.dt * iry * bar_vy;
    float bar_dsyz_dz = solver.dt * iry * bar_vy;
    float bar_dsxz_dx = solver.dt * irz * bar_vz;
    float bar_dsyz_dy = solver.dt * irz * bar_vz;
    float bar_dszz_dz = solver.dt * irz * bar_vz;

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
        pxx[b*spatial_size+idx] = bar_dsxx_dx;
        pxy[b*spatial_size+idx] = bar_dsxy_dy;
        pxz[b*spatial_size+idx] = bar_dsxz_dz;
        pyx[b*spatial_size+idx] = bar_dsxy_dx;
        pyy[b*spatial_size+idx] = bar_dsyy_dy;
        pyz[b*spatial_size+idx] = bar_dsyz_dz;
        pzx[b*spatial_size+idx] = bar_dsxz_dx;
        pzy[b*spatial_size+idx] = bar_dsyz_dy;
        pzz[b*spatial_size+idx] = bar_dszz_dz;
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

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxyy = f.m_sxyy[idx] + bar_dsxy_dy;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxyx = f.m_sxyx[idx] + bar_dsxy_dx;
    float tmp_syyy = f.m_syyy[idx] + bar_dsyy_dy;
    float tmp_syzz = f.m_syzz[idx] + bar_dsyz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_syzy = f.m_syzy[idx] + bar_dsyz_dy;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    pxx[b*spatial_size+idx] = bar_dsxx_dx + bxh * tmp_sxxx;
    pxy[b*spatial_size+idx] = bar_dsxy_dy + by  * tmp_sxyy;
    pxz[b*spatial_size+idx] = bar_dsxz_dz + bz  * tmp_sxzz;
    pyx[b*spatial_size+idx] = bar_dsxy_dx + bx  * tmp_sxyx;
    pyy[b*spatial_size+idx] = bar_dsyy_dy + byh * tmp_syyy;
    pyz[b*spatial_size+idx] = bar_dsyz_dz + bz  * tmp_syzz;
    pzx[b*spatial_size+idx] = bar_dsxz_dx + bx  * tmp_sxzx;
    pzy[b*spatial_size+idx] = bar_dsyz_dy + by  * tmp_syzy;
    pzz[b*spatial_size+idx] = bar_dszz_dz + bzh * tmp_szzz;

    f.m_sxxx[idx] = axh * tmp_sxxx;
    f.m_sxyy[idx] = ay  * tmp_sxyy;
    f.m_sxzz[idx] = az  * tmp_sxzz;
    f.m_sxyx[idx] = ax  * tmp_sxyx;
    f.m_syyy[idx] = ayh * tmp_syyy;
    f.m_syzz[idx] = az  * tmp_syzz;
    f.m_sxzx[idx] = ax  * tmp_sxzx;
    f.m_syzy[idx] = ay  * tmp_syzy;
    f.m_szzz[idx] = azh * tmp_szzz;
}

#define LAUNCH_3DELASTIC_STRESS_ADJOINT_PREPARE_APM(order, grid, block, ...)    \
    do {                                                                        \
        if      ((order) == 2) elastic3d_stress_adjoint_prepare_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic3d_stress_adjoint_prepare_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic3d_stress_adjoint_prepare_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic3d_stress_adjoint_prepare_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic3d_stress_adjoint_prepare_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_VELOCITY_ADJOINT_PREPARE_APM(order, grid, block, ...)  \
    do {                                                                        \
        if      ((order) == 2) elastic3d_velocity_adjoint_prepare_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic3d_velocity_adjoint_prepare_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic3d_velocity_adjoint_prepare_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic3d_velocity_adjoint_prepare_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic3d_velocity_adjoint_prepare_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


// --- Gradient kernels (APM) ------------------------------------------------
// Compute per-cell effective-moduli gradients from (bar_s, dv), apply the
// traction-BC adjoint (zeroing bar_s components per category), chain via
// apm3d_chain_lammu to (grad_lam, grad_mu), and finally chain to
// (grad_vp, grad_vs, grad_rho) using the standard
// (vp,vs,rho) ↔ (lam,mu) identities.  The kinetic contribution to
// grad_rho mirrors the image-method 3-D formula — the per-category
// factor in inv_rho_* cancels with d(inv_rho_*)/d(rho) so the result is
// the same ``(a·(f-f_prev))/rho_raw`` term.

template<int Order>
__global__ void calculate_grad_elastic3d_apm_bs(
    ElasticWavefieldPointer forward,
    ElasticWavefieldPointer adjoint,
    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvy_prev,
    const float* __restrict__ fvz_prev,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ rho,
    const float* __restrict__ lam_raw,
    const float* __restrict__ mu_raw,
    const int*   __restrict__ category,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_vs,
    float* __restrict__ grad_rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int cat = category[idx];

    auto f = forward.offset(b, spatial_size);
    auto a = adjoint.offset(b, spatial_size);
    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvy_prev_b = fvy_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;
    const float* vp_b   = vp + b * spatial_size;
    const float* vs_b   = vs + b * spatial_size;
    const float* rho_b  = rho + b * spatial_size;
    const float* lam_b  = lam_raw + b * spatial_size;
    const float* mu_b   = mu_raw  + b * spatial_size;
    float* gvp_b  = grad_vp  + b * spatial_size;
    float* gvs_b  = grad_vs  + b * spatial_size;
    float* grho_b = grad_rho + b * spatial_size;

    float fvx_x = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float fvx_y = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float fvx_z = sgradient<3, Order, Z, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float fvy_x = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float fvy_y = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float fvy_z = sgradient<3, Order, Z, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float fvz_x = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float fvz_y = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float fvz_z = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

    // Adjoint stresses with APM traction-BC adjoint applied.
    float bar_sxx = a.sxx[idx];
    float bar_syy = a.syy[idx];
    float bar_szz = a.szz[idx];
    float bar_sxy = a.sxy[idx];
    float bar_sxz = a.sxz[idx];
    float bar_syz = a.syz[idx];
    if (cat == APM3D_CATEGORY_H) {
        bar_szz = bar_sxz = bar_syz = 0.f;
    } else if (cat == APM3D_CATEGORY_VL || cat == APM3D_CATEGORY_VR) {
        bar_sxx = bar_sxy = bar_sxz = 0.f;
    } else if (cat == APM3D_CATEGORY_VF || cat == APM3D_CATEGORY_VB) {
        bar_syy = bar_sxy = bar_syz = 0.f;
    } else if (cat == APM3D_CATEGORY_OC || cat == APM3D_CATEGORY_AIR) {
        bar_sxx = bar_syy = bar_szz = 0.f;
        bar_sxy = bar_sxz = bar_syz = 0.f;
    }

    // Gradients w.r.t. effective moduli (dt is multiplied at the end via
    // the standard (vp,vs,rho) chain — so leave dt out here).
    float g_axx = bar_sxx * fvx_x;
    float g_ayy = bar_syy * fvy_y;
    float g_azz = bar_szz * fvz_z;
    float g_lxy = bar_sxx * fvy_y;
    float g_lxz = bar_sxx * fvz_z;
    float g_lyx = bar_syy * fvx_x;
    float g_lyz = bar_syy * fvz_z;
    float g_lzx = bar_szz * fvx_x;
    float g_lzy = bar_szz * fvy_y;
    float g_mxy = bar_sxy * (fvx_y + fvy_x);
    float g_mxz = bar_sxz * (fvx_z + fvz_x);
    float g_myz = bar_syz * (fvy_z + fvz_y);

    float lam_v = lam_b[idx];
    float mu_v  = mu_b[idx];
    float grad_lam, grad_mu;
    apm3d_chain_lammu(cat, lam_v, mu_v,
                       g_axx, g_ayy, g_azz,
                       g_lxy, g_lxz, g_lyx, g_lyz, g_lzx, g_lzy,
                       g_mxy, g_mxz, g_myz,
                       &grad_lam, &grad_mu);

    // Kinetic contribution to grad_rho — factor_* cancellation makes
    // this identical in form to the image-method 3-D expression.
    float grad_rho_kin = (a.vx[idx] * (f.vx[idx] - fvx_prev_b[idx]) +
                          a.vy[idx] * (f.vy[idx] - fvy_prev_b[idx]) +
                          a.vz[idx] * (f.vz[idx] - fvz_prev_b[idx])) / rho_b[idx];

    float vp_v = vp_b[idx];
    float vs_v = vs_b[idx];
    float rho_v = rho_b[idx];

    gvp_b[idx]  += -2.f * rho_v * vp_v * grad_lam * solver.dt;
    gvs_b[idx]  += -(-4.f * rho_v * vs_v * grad_lam + 2.f * rho_v * vs_v * grad_mu) * solver.dt;
    grho_b[idx] += grad_rho_kin
                   - (grad_lam * (vp_v * vp_v - 2.f * vs_v * vs_v)
                      + grad_mu * (vs_v * vs_v)) * solver.dt;
}

template<int Order>
__global__ void calculate_grad_elastic3d_apm_nobs(
    ElasticWavefieldPointer adjoint,
    const float* __restrict__ fvx,
    const float* __restrict__ fvy,
    const float* __restrict__ fvz,
    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvy_prev,
    const float* __restrict__ fvz_prev,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ rho,
    const float* __restrict__ lam_raw,
    const float* __restrict__ mu_raw,
    const int*   __restrict__ category,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_vs,
    float* __restrict__ grad_rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int cat = category[idx];

    auto a = adjoint.offset(b, spatial_size);
    const float* fvx_b = fvx + b * spatial_size;
    const float* fvy_b = fvy + b * spatial_size;
    const float* fvz_b = fvz + b * spatial_size;
    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvy_prev_b = fvy_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;
    const float* vp_b  = vp + b * spatial_size;
    const float* vs_b  = vs + b * spatial_size;
    const float* rho_b = rho + b * spatial_size;
    const float* lam_b = lam_raw + b * spatial_size;
    const float* mu_b  = mu_raw  + b * spatial_size;
    float* gvp_b  = grad_vp  + b * spatial_size;
    float* gvs_b  = grad_vs  + b * spatial_size;
    float* grho_b = grad_rho + b * spatial_size;

    float fvx_x = sgradient<3, Order, X, DIFF_BACKWARD>(fvx_b, ix, iy, iz, grad_ctx);
    float fvx_y = sgradient<3, Order, Y, DIFF_FORWARD >(fvx_b, ix, iy, iz, grad_ctx);
    float fvx_z = sgradient<3, Order, Z, DIFF_FORWARD >(fvx_b, ix, iy, iz, grad_ctx);
    float fvy_x = sgradient<3, Order, X, DIFF_FORWARD >(fvy_b, ix, iy, iz, grad_ctx);
    float fvy_y = sgradient<3, Order, Y, DIFF_BACKWARD>(fvy_b, ix, iy, iz, grad_ctx);
    float fvy_z = sgradient<3, Order, Z, DIFF_FORWARD >(fvy_b, ix, iy, iz, grad_ctx);
    float fvz_x = sgradient<3, Order, X, DIFF_FORWARD >(fvz_b, ix, iy, iz, grad_ctx);
    float fvz_y = sgradient<3, Order, Y, DIFF_FORWARD >(fvz_b, ix, iy, iz, grad_ctx);
    float fvz_z = sgradient<3, Order, Z, DIFF_BACKWARD>(fvz_b, ix, iy, iz, grad_ctx);

    float bar_sxx = a.sxx[idx];
    float bar_syy = a.syy[idx];
    float bar_szz = a.szz[idx];
    float bar_sxy = a.sxy[idx];
    float bar_sxz = a.sxz[idx];
    float bar_syz = a.syz[idx];
    if (cat == APM3D_CATEGORY_H) {
        bar_szz = bar_sxz = bar_syz = 0.f;
    } else if (cat == APM3D_CATEGORY_VL || cat == APM3D_CATEGORY_VR) {
        bar_sxx = bar_sxy = bar_sxz = 0.f;
    } else if (cat == APM3D_CATEGORY_VF || cat == APM3D_CATEGORY_VB) {
        bar_syy = bar_sxy = bar_syz = 0.f;
    } else if (cat == APM3D_CATEGORY_OC || cat == APM3D_CATEGORY_AIR) {
        bar_sxx = bar_syy = bar_szz = 0.f;
        bar_sxy = bar_sxz = bar_syz = 0.f;
    }

    float g_axx = bar_sxx * fvx_x;
    float g_ayy = bar_syy * fvy_y;
    float g_azz = bar_szz * fvz_z;
    float g_lxy = bar_sxx * fvy_y;
    float g_lxz = bar_sxx * fvz_z;
    float g_lyx = bar_syy * fvx_x;
    float g_lyz = bar_syy * fvz_z;
    float g_lzx = bar_szz * fvx_x;
    float g_lzy = bar_szz * fvy_y;
    float g_mxy = bar_sxy * (fvx_y + fvy_x);
    float g_mxz = bar_sxz * (fvx_z + fvz_x);
    float g_myz = bar_syz * (fvy_z + fvz_y);

    float grad_lam, grad_mu;
    apm3d_chain_lammu(cat, lam_b[idx], mu_b[idx],
                       g_axx, g_ayy, g_azz,
                       g_lxy, g_lxz, g_lyx, g_lyz, g_lzx, g_lzy,
                       g_mxy, g_mxz, g_myz,
                       &grad_lam, &grad_mu);

    float grad_rho_kin = (a.vx[idx] * (fvx_b[idx] - fvx_prev_b[idx]) +
                          a.vy[idx] * (fvy_b[idx] - fvy_prev_b[idx]) +
                          a.vz[idx] * (fvz_b[idx] - fvz_prev_b[idx])) / rho_b[idx];

    float vp_v = vp_b[idx];
    float vs_v = vs_b[idx];
    float rho_v = rho_b[idx];

    gvp_b[idx]  += -2.f * rho_v * vp_v * grad_lam * solver.dt;
    gvs_b[idx]  += -(-4.f * rho_v * vs_v * grad_lam + 2.f * rho_v * vs_v * grad_mu) * solver.dt;
    grho_b[idx] += grad_rho_kin
                   - (grad_lam * (vp_v * vp_v - 2.f * vs_v * vs_v)
                      + grad_mu * (vs_v * vs_v)) * solver.dt;
}

#define LAUNCH_CALCULATE_GRAD_3DELASTIC_APM_BS(order, grid, block, ...)         \
    do {                                                                        \
        if      ((order) == 2) calculate_grad_elastic3d_apm_bs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic3d_apm_bs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic3d_apm_bs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic3d_apm_bs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic3d_apm_bs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_3DELASTIC_APM_NOBS(order, grid, block, ...)       \
    do {                                                                        \
        if      ((order) == 2) calculate_grad_elastic3d_apm_nobs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic3d_apm_nobs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic3d_apm_nobs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic3d_apm_nobs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic3d_apm_nobs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)
