#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/gradient2d.cuh"
#include "../../operators/laplace2d.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

#define LAUNCH_FORWARD(order, grid, block, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic_forward_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_forward_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_forward_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_forward_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_forward_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_FORWARD_NOPML(order, grid, block, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

template<int Order>
__global__ void acoustic_forward_kernel(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    AcousticCPMLPointer cpml,
    SolverContext solver
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static  = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float w_sum = 0.0f;

    Laplace2dContext ctx{
        solver.nx, ix, iz,
        solver.M,
        solver.lap_coeff,
        solver.dx, solver.dz
    };

    GradContext ctx2d{1, solver.nx, ix, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};
    GradContext ctx_x{1, 0, ix, 0, solver.M, solver.grad_coeff, solver.dx, solver.dz};
    GradContext ctx_z{0, 1, 0, iz, solver.M, solver.grad_coeff, solver.dx, solver.dz};

    float lap_x    = laplace<Order, LAPLACE_X>(f.u_now, ctx);
    float lap_z    = laplace<Order, LAPLACE_Z>(f.u_now, ctx);
    float dudz     = gradient<Order, GRAD_Z>(f.u_now, ctx2d);
    float dudx     = gradient<Order, GRAD_X>(f.u_now, ctx2d);
    float dpsizdz  = gradient<Order, GRAD_Z>(f.psiz, ctx2d);
    float dpsixdx  = gradient<Order, GRAD_X>(f.psix, ctx2d);
    float dazdz    = gradient<Order, GRAD_Z>(cpml.az, ctx_z);
    float daxdx    = gradient<Order, GRAD_X>(cpml.ax, ctx_x);

    float daipsiz_dz = dazdz * f.psiz[idx] + az_ * dpsizdz;
    float daipxix_dx = daxdx * f.psix[idx] + ax_ * dpsixdx;

    // X direction
    float tmpx = ((1.0f+bx_)*lap_x + dbxdx_*dudx) + daipxix_dx;
    w_sum += (1.0f+bx_) * tmpx + ax_ * f.zetax[idx];
    f.psix[idx]  = bx_ * dudx + ax_ * f.psix[idx];
    f.zetax[idx] = bx_ * tmpx + ax_ * f.zetax[idx];

    // Z direction
    float tmpz = ((1.0f+bz_)*lap_z + dbzdz_*dudz) + daipsiz_dz;
    w_sum += (1.0f+bz_) * tmpz + az_ * f.zetaz[idx];
    f.psiz[idx]  = bz_ * dudz + az_ * f.psiz[idx];
    f.zetaz[idx] = bz_ * tmpz + az_ * f.zetaz[idx];

    float v  = vp_b[idx];

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_z);
}

template<int Order>
__global__ void acoustic_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    SolverContext solver
) {
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

    int halo = solver.abcn + 2*M;

    int top_halo = solver.free_surface ? 2* M: halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    const float* vp_b     = vp     + b * spatial_size;

    float lap_x, lap_z;

    float w_sum = 0.0f;

    Laplace2dContext ctx{solver.nx, ix, iz, solver.M, solver.lap_coeff, solver.dx, solver.dz};

    lap_x    = laplace<Order, LAPLACE_X>(f.u_now, ctx);
    lap_z    = laplace<Order, LAPLACE_Z>(f.u_now, ctx);

    w_sum = lap_x + lap_z;

    float v  = vp_b[idx];

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

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