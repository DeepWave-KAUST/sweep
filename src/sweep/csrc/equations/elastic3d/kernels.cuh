#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../common/context.h"
#include "../../common/elastic.h"
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
    
    float dsxx_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.sxx, ix, iy, iz, grad_ctx);
    float dsxy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsxz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx);

    float inv_rho = 1.f / rho_b[idx];

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

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);
    float dvx_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

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

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

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

    int top_halo = solver.free_surface ? 1*M+1: halo;

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
    float dsxz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);

    float dsxy_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxy, ix, iy, iz, grad_ctx);
    float dsyy_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.syy, ix, iy, iz, grad_ctx);
    float dsyz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);

    float dsxz_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.sxz, ix, iy, iz, grad_ctx);
    float dsyz_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.syz, ix, iy, iz, grad_ctx);
    float dszz_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.szz, ix, iy, iz, grad_ctx);
    
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

    int top_halo = solver.free_surface ? 1*M+1: halo;

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
    float dvx_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);

    float dvy_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvy_dz = sgradient<3, Order, Z, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);

    float dvz_dx = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dy = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);


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
    float dpxz_dz = sgradient<3, Order, Z, DIFF_FORWARD>(pxz_b, ix, iy, iz, grad_ctx);

    float dpyx_dx = sgradient<3, Order, X, DIFF_FORWARD>(pyx_b, ix, iy, iz, grad_ctx);
    float dpyy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(pyy_b, ix, iy, iz, grad_ctx);
    float dpyz_dz = sgradient<3, Order, Z, DIFF_FORWARD>(pyz_b, ix, iy, iz, grad_ctx);

    float dpzx_dx = sgradient<3, Order, X, DIFF_FORWARD>(pzx_b, ix, iy, iz, grad_ctx);
    float dpzy_dy = sgradient<3, Order, Y, DIFF_FORWARD>(pzy_b, ix, iy, iz, grad_ctx);
    float dpzz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(pzz_b, ix, iy, iz, grad_ctx);

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

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];
    float l2m = lam + 2.f * mu_;

    float bar_dvx_dx = solver.dt * (l2m * f.sxx[idx] + lam * f.syy[idx] + lam * f.szz[idx]);
    float bar_dvx_dy = solver.dt * mu_ * f.sxy[idx];
    float bar_dvx_dz = solver.dt * mu_ * f.sxz[idx];

    float bar_dvy_dx = solver.dt * mu_ * f.sxy[idx];
    float bar_dvy_dy = solver.dt * (lam * f.sxx[idx] + l2m * f.syy[idx] + lam * f.szz[idx]);
    float bar_dvy_dz = solver.dt * mu_ * f.syz[idx];

    float bar_dvz_dx = solver.dt * mu_ * f.sxz[idx];
    float bar_dvz_dy = solver.dt * mu_ * f.syz[idx];
    float bar_dvz_dz = solver.dt * (lam * f.sxx[idx] + lam * f.syy[idx] + l2m * f.szz[idx]);

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
    float dqxz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(qxz_b, ix, iy, iz, grad_ctx);

    float dqyx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(qyx_b, ix, iy, iz, grad_ctx);
    float dqyy_dy = sgradient<3, Order, Y, DIFF_FORWARD>(qyy_b, ix, iy, iz, grad_ctx);
    float dqyz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(qyz_b, ix, iy, iz, grad_ctx);

    float dqzx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(qzx_b, ix, iy, iz, grad_ctx);
    float dqzy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(qzy_b, ix, iy, iz, grad_ctx);
    float dqzz_dz = sgradient<3, Order, Z, DIFF_FORWARD>(qzz_b, ix, iy, iz, grad_ctx);

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
    float fvx_z = sgradient<3, Order, Z, DIFF_FORWARD >(f.vx, ix, iy, iz, grad_ctx);

    float fvy_x = sgradient<3, Order, X, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);
    float fvy_y = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float fvy_z = sgradient<3, Order, Z, DIFF_FORWARD >(f.vy, ix, iy, iz, grad_ctx);

    float fvz_x = sgradient<3, Order, X, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float fvz_y = sgradient<3, Order, Y, DIFF_FORWARD >(f.vz, ix, iy, iz, grad_ctx);
    float fvz_z = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

    float grad_lambda = (a.sxx[idx] + a.syy[idx] + a.szz[idx]) * (fvx_x + fvy_y + fvz_z);
    float grad_mu = 2*(a.sxx[idx] * fvx_x + 
                       a.syy[idx] * fvy_y + 
                       a.szz[idx] * fvz_z) + 
                       a.sxz[idx] * (fvx_z + fvz_x) + 
                       a.sxy[idx] * (fvx_y + fvy_x) +
                       a.syz[idx] * (fvy_z + fvz_y);
    
    gvp[idx] +=   -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    gvs[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                   2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

    grho[idx] += (a.vx[idx] * (f.vx[idx]-fvx_prev_b[idx]) + 
                  a.vy[idx] * (f.vy[idx]-fvy_prev_b[idx]) +
                  a.vz[idx] * (f.vz[idx]-fvz_prev_b[idx])) / rho_b[idx];
    grho[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                 grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;
}
