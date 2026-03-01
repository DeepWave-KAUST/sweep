#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/staggered_gradient3d.cuh"
#include "../../operators/gradient3d.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"

#define LAUNCH_3DELASTIC_VELOCITY(order, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_3DELASTIC_STRESS(order, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_VELOCITY_NOPML(order, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel_3d_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_3d_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_3d_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_3d_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_3d_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_3DELASTIC_STRESS_NOPML(order, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel_3d_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_3d_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_3d_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_3d_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_3d_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_VELOCITY_ADJOINT(order, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_3DELASTIC_STRESS_ADJOINT(order, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(order, ...)                       \
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

    SGradContext3d ctx {
        1,
        solver.nx,
        solver.nx * solver.ny,
        ix, iy, iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    // ===== stress derivatives =====
    float dsxx_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.sxx, ctx);
    float dsxy_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD >(f.sxy, ctx);
    float dsxz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD >(f.sxz, ctx);

    float dsxy_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD >(f.sxy, ctx);
    float dsyy_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.syy, ctx);
    float dsyz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD >(f.syz, ctx);

    float dsxz_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD >(f.sxz, ctx);
    float dsyz_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD >(f.syz, ctx);
    float dszz_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.szz, ctx);
    
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

    SGradContext3d ctx {
        1,
        solver.nx,
        solver.nx * solver.ny,
        ix, iy, iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    // ===== velocity derivatives =====
    float dvx_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vx, ctx);
    float dvx_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vx, ctx);
    float dvx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx);

    float dvy_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vy, ctx);
    float dvy_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD>(f.vy, ctx);
    float dvy_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vy, ctx);

    float dvz_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vz, ctx);
    float dvz_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vz, ctx);
    float dvz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx);

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
}

template<int Order>
__global__ void elastic_velocity_kernel_3d_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    int halo = solver.abcn + 1*M+0;

    int top_halo = solver.free_surface ? 1*M+0: halo;

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

    SGradContext3d ctx {
        1,
        solver.nx,
        solver.nx * solver.ny,
        ix, iy, iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    // ===== stress derivatives =====
    float dsxx_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.sxx, ctx);
    float dsxy_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD >(f.sxy, ctx);
    float dsxz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD >(f.sxz, ctx);

    float dsxy_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD >(f.sxy, ctx);
    float dsyy_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.syy, ctx);
    float dsyz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD >(f.syz, ctx);

    float dsxz_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD >(f.sxz, ctx);
    float dsyz_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD >(f.syz, ctx);
    float dszz_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.szz, ctx);
    
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

    int halo = solver.abcn + 1*M+0;

    int top_halo = solver.free_surface ? 1*M+0: halo;

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

    SGradContext3d ctx {
        1,
        solver.nx,
        solver.nx * solver.ny,
        ix, iy, iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    // ===== velocity derivatives =====
    float dvx_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vx, ctx);
    float dvx_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vx, ctx);
    float dvx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx);

    float dvy_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vy, ctx);
    float dvy_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD>(f.vy, ctx);
    float dvy_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vy, ctx);

    float dvz_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vz, ctx);
    float dvz_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vz, ctx);
    float dvz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx);

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

    SGradContext3d ctx {
        1,
        solver.nx,
        solver.nx * solver.ny,
        ix, iy, iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    // ===== stress derivatives =====
    float dsxx_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.sxx, ctx); //
    float dsxx_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.sxx, ctx);
    float dsxx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.sxx, ctx);

    float dsxy_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD >(f.sxy, ctx); //
    float dsxy_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD >(f.sxy, ctx); //

    float dsxz_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD >(f.sxz, ctx); //
    float dsxz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD >(f.sxz, ctx); //

    float dsyy_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.syy, ctx);
    float dsyy_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.syy, ctx); //
    float dsyy_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.syy, ctx);

    float dsyz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD >(f.syz, ctx); //
    float dsyz_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD >(f.syz, ctx);

    float dszz_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.szz, ctx);
    float dszz_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.szz, ctx);
    float dszz_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.szz, ctx); //

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
__global__ void elastic_stress_adjoint_kernel_3d(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    SGradContext3d ctx {
        1,
        solver.nx,
        solver.nx * solver.ny,
        ix, iy, iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    float inv_rho = 1.f / rho_b[idx];

    float dvx_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vx, ctx);
    float dvx_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vx, ctx);
    float dvx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx);

    float dvy_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vy, ctx);
    float dvy_dy = sgradient<Order, GRAD_Y, DIFF_BACKWARD>(f.vy, ctx);
    float dvy_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vy, ctx);

    float dvz_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vz, ctx);
    float dvz_dy = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vz, ctx);
    float dvz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx);

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
    
    SGradContext3d ctx {
        1,
        solver.nx,
        solver.nx * solver.ny,
        ix, iy, iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    // forward because velocity is staggered
    float fvx_x = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vx, ctx);
    float fvx_y = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vx, ctx);
    float fvx_z = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx);

    float fvy_x = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vy, ctx);
    float fvy_y = sgradient<Order, GRAD_Y, DIFF_BACKWARD>(f.vy, ctx);
    float fvy_z = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vy, ctx);

    float fvz_x = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vz, ctx);
    float fvz_y = sgradient<Order, GRAD_Y, DIFF_FORWARD>(f.vz, ctx);
    float fvz_z = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx);

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

    grad_rho[idx] += (a.vx[idx] * (f.vx[idx]-fvx_prev_b[idx]) + 
                      a.vy[idx] * (f.vy[idx]-fvy_prev_b[idx]) +
                      a.vz[idx] * (f.vz[idx]-fvz_prev_b[idx])) / rho_b[idx];
    grad_rho[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                     grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;
}