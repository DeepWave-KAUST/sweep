#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/gradient2d.cuh"
#include "../../operators/laplace2d.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

#define LAUNCH_FORWARD(order, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic_forward_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_forward_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_forward_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_forward_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_forward_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_FORWARD_NOPML(order, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

template<int Order>
__global__ void acoustic_forward_kernel(
    AcousticWavefield wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    AcousticCPML cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);

    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* u_now_b  = wf.u_now  + b * spatial_size;
    const float* u_prev_b = wf.u_prev + b * spatial_size;
    float*       u_next_b = wf.u_next + b * spatial_size;
    // float*       u_this_b = u_this + b * spatial_size;
    float*       u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b     = vp     + b * spatial_size;

    float* psix_b = wf.psix + b * spatial_size;
    float* psiz_b = wf.psiz + b * spatial_size;
    float* zetax_b = wf.zetax + b * spatial_size;
    float* zetaz_b = wf.zetaz + b * spatial_size;

    float tmpx, tmpz;
    float lap_x, lap_z;
    float dudz, dudx;
    float dazdz, daxdx, dpsizdz, dpsixdx;
    float daipsiz_dz, daipxix_dx;

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float w_sum = 0.0f;

    Laplace2dContext ctx{solver.nx, ix, iz, solver.M, solver.lap_coeff, solver.dx, solver.dz};
    GradContext ctx2d {1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};
    GradContext ctx_x {1, 0, ix, 0,  solver.M, solver.grad_coeff, solver.dx, solver.dz};
    GradContext ctx_z {0, 1, 0, iz,  solver.M, solver.grad_coeff, solver.dx, solver.dz};

    lap_x    = laplace<Order, LAPLACE_X>(u_now_b,   ctx);
    lap_z    = laplace<Order, LAPLACE_Z>(u_now_b,   ctx);
    dudz     = gradient<Order, GRAD_Z>(u_now_b,     ctx2d);
    dudx     = gradient<Order, GRAD_X>(u_now_b,     ctx2d);
    dpsizdz  = gradient<Order, GRAD_Z>(psiz_b,      ctx2d);
    dpsixdx  = gradient<Order, GRAD_X>(psix_b,      ctx2d);
    dazdz    = gradient<Order, GRAD_Z>(cpml.az,     ctx_z);
    daxdx    = gradient<Order, GRAD_X>(cpml.ax,     ctx_x);

    daipsiz_dz = dazdz * psiz_b[idx] + az_ * dpsizdz;
    daipxix_dx = daxdx * psix_b[idx] + ax_ * dpsixdx;

    // X direction
    tmpx = ((1.0f+bx_)*lap_x + dbxdx_*dudx) + daipxix_dx;
    w_sum += (1.0f+bx_) * tmpx + ax_ * zetax_b[idx];
    psix_b[idx] = bx_ * dudx + ax_ * psix_b[idx];
    zetax_b[idx] = bx_ * tmpx + ax_ * zetax_b[idx];

    // Z direction
    tmpz = ((1.0f+bz_)*lap_z + dbzdz_*dudz) + daipsiz_dz;
    w_sum += (1.0f+bz_) * tmpz + az_ * zetaz_b[idx];
    psiz_b[idx] = bz_ * dudz + az_ * psiz_b[idx];
    zetaz_b[idx] = bz_ * tmpz + az_ * zetaz_b[idx];

    float u0 = u_now_b[idx];
    float v  = vp_b[idx];

    u_next_b[idx] =
        2.0f * u0 -
        u_prev_b[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    // save all u_tt for backward
    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_z);
}

template<int Order>
__global__ void acoustic_nopml(
    AcousticWavefield wf,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int halo = solver.abcn + 2*solver.M;

    int top_halo = solver.free_surface ? solver.M : halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    
    const float* u_now_b  = wf.u_now  + b * spatial_size;
    const float* u_prev_b = wf.u_prev + b * spatial_size;
    float*       u_next_b = wf.u_next + b * spatial_size;
    float*       u_this_b = u_this + b * spatial_size;
    const float* vp_b     = vp     + b * spatial_size;

    float lap_x, lap_z;

    float w_sum = 0.0f;

    Laplace2dContext ctx{solver.nx, ix, iz, solver.M, solver.lap_coeff, solver.dx, solver.dz};

    lap_x    = laplace<Order, LAPLACE_X>(u_now_b, ctx);
    lap_z    = laplace<Order, LAPLACE_Z>(u_now_b, ctx);

    w_sum = lap_x + lap_z;

    float u0 = u_now_b[idx];
    float v  = vp_b[idx];

    u_next_b[idx] =
        2.0f * u0 -
        u_prev_b[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    u_this_b[idx] = (v * v) * w_sum;
}

__global__ void calculate_grad(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz
);

__global__ void calculate_grad_utt(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt
);

__global__ void save_boundary_kernel(
    const float* __restrict__ u,   // (B, nz, nx)
    float* __restrict__ top,       // (nt, B, n, nx)
    float* __restrict__ bottom,    // (nt, B, n, nx)
    float* __restrict__ left,      // (nt, B, nz, n)
    float* __restrict__ right,     // (nt, B, nz, n)
    int it,
    SolverContext solver
);

__global__ void restore_boundary_kernel(
    float* __restrict__ u,        // (B, nz, nx)
    const float* __restrict__ top,
    const float* __restrict__ bottom,
    const float* __restrict__ left,
    const float* __restrict__ right,
    int it,
    SolverContext solver
);