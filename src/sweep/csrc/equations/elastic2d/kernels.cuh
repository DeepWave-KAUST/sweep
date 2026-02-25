#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/staggered_gradient2d.cuh"
#include "../../operators/gradient2d.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"

#define LAUNCH_ELASTIC_VELOCITY(order, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_ELASTIC_STRESS(order, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_NOPML(order, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_ELASTIC_STRESS_NOPML(order, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC(order, ...)                       \
    do {                                                        \
        if      ((order) == 2) calculate_grad_elastic<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_ADJOINT(order, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_ADJOINT(order, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


template<int Order>
__global__ void elastic_velocity_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    SGradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    // backward because stress is at integer grid
    float dsxx_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.sxx, ctx2d);
    float dsxz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.sxz, ctx2d);
    float dsxz_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.sxz, ctx2d);
    float dszz_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.szz, ctx2d);

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

    SGradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    // forward because velocity is staggered
    float dvx_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vx, ctx2d);
    float dvz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx2d);
    float dvx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx2d);
    float dvz_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vz, ctx2d);

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

    if (u_this_b) {
        int comp_stride  = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = dvx_dx;
        u_this_b[1 * comp_stride + idx] = dvx_dz;
        u_this_b[2 * comp_stride + idx] = dvz_dx;
        u_this_b[3 * comp_stride + idx] = dvz_dz;
    }

}

template<int Order>
__global__ void elastic_velocity_kernel_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    int halo = solver.abcn + 1*M+0;

    int top_halo = solver.free_surface ? 1*M+0: halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    // Wavefields
    auto f = wf.offset(b, spatial_size);

    const float* rho_b = rho + b * spatial_size;

    SGradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    // backward because stress is at integer grid
    float dsxx_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.sxx, ctx2d);
    float dsxz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.sxz, ctx2d);
    float dsxz_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.sxz, ctx2d);
    float dszz_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.szz, ctx2d);

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

    int halo = solver.abcn + 1*M+0;

    int top_halo = solver.free_surface ? 1*M+0: halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    // Wavefields
    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;


    SGradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    // forward because velocity is staggered
    float dvx_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vx, ctx2d);
    float dvz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx2d);
    float dvx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx2d);
    float dvz_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vz, ctx2d);

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
__global__ void elastic_velocity_adjoint_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
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

    // Wavefields
    auto f = wf.offset(b, spatial_size);

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

    SGradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    // backward because stress is at integer grid
    float dsxx_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.sxx, ctx2d);
    float dsxz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.sxz, ctx2d);
    float dsxz_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.sxz, ctx2d);
    float dszz_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.szz, ctx2d);
    
    float dszz_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.szz, ctx2d);
    float dsxx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.sxx, ctx2d);

    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];
    f.m_sxxz[idx] = azh * f.m_sxxz[idx] + bzh * dsxx_dz; //new
    dsxx_dz += f.m_sxxz[idx];

    f.m_szzx[idx] = axh * f.m_szzx[idx] + bxh * dszz_dx; //new
    dszz_dx += f.m_szzx[idx];
    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];

    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];
    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];

    f.vx[idx] += solver.dt *
        ((lam_b[idx] + 2.f*mu_b[idx]) * dsxx_dx + lam_b[idx] * dszz_dx + mu_b[idx] * dsxz_dz);

    f.vz[idx] += solver.dt *
        ((lam_b[idx] + 2.f*mu_b[idx]) * dszz_dz + lam_b[idx] * dsxx_dz + mu_b[idx] * dsxz_dx);
}

template<int Order>
__global__ void elastic_stress_adjoint_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
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

    // Wavefields
    auto f = wf.offset(b, spatial_size);

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    const float* rho_b = rho + b * spatial_size;

    SGradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    // forward because velocity is staggered
    float dvx_dx = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vx, ctx2d);
    float dvx_dz = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx2d);
    float dvz_dz = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx2d);
    float dvz_dx = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vz, ctx2d);
    
    float inv_rho = 1.f / rho_b[idx];
    
    // Update PML memory variables
    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx += f.m_vxx[idx];
    f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
    dvx_dz += f.m_vxz[idx];
    f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
    dvz_dx += f.m_vzx[idx];
    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz += f.m_vzz[idx];

    f.sxx[idx] += solver.dt * inv_rho * dvx_dx;

    f.szz[idx] += solver.dt * inv_rho * dvz_dz;

    f.sxz[idx] += solver.dt * inv_rho * (dvx_dz + dvz_dx);
}


template<int Order>
__global__ void calculate_grad_elastic(

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
    
    SGradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    // forward because velocity is staggered
    float fvx_x = sgradient<Order, GRAD_X, DIFF_FORWARD>(f.vx, ctx2d);
    float fvx_z = sgradient<Order, GRAD_Z, DIFF_FORWARD>(f.vx, ctx2d);
    float fvz_x = sgradient<Order, GRAD_X, DIFF_BACKWARD>(f.vz, ctx2d);
    float fvz_z = sgradient<Order, GRAD_Z, DIFF_BACKWARD>(f.vz, ctx2d);

    float grad_lambda = (a.sxx[idx] + a.szz[idx]) * (fvx_x + fvz_z);
    float grad_mu = 2*(a.sxx[idx] * fvx_x + a.szz[idx] * fvz_z) + a.sxz[idx] * (fvx_z + fvz_x);
    
    gvp[idx] += -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    gvs[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                   2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

    grad_rho[idx] += (a.vx[idx] * (fvx_prev_b[idx] - f.vx[idx]) / solver.dt + 
                      a.vz[idx] * (fvz_prev_b[idx] - f.vz[idx]) / solver.dt) * solver.dt;
}

__global__ void calculate_elastic_grad(

    ElasticWavefieldPointer adjoint,

    const float* __restrict__ vx_x,
    const float* __restrict__ vx_z,
    const float* __restrict__ vz_x,
    const float* __restrict__ vz_z,

    const float* __restrict__ vp,        // (B, nz, nx)
    const float* __restrict__ vs,        // (B, nz, nx)
    const float* __restrict__ rho,       // (B, nz, nx)

    float* __restrict__ grad_vp,         // (B, nz, nx)
    float* __restrict__ grad_vs,         // (B, nz, nx)
    float* __restrict__ grad_rho,         // (B, nz, nx)

    SolverContext solver
);