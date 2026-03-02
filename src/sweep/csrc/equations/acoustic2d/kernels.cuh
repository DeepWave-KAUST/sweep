#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/laplace.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

#define ACOUSTIC2D(order, grid, block, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic2nd<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic2nd<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic2nd<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic2nd<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic2nd<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define ACOUSTIC2D_NOPML(order, grid, block, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic2nd_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic2nd_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic2nd_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic2nd_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic2nd_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

template<int Order>
__global__ void acoustic2nd(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
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

    float lap_x    = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z    = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float dudz     = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);
    float dudx     = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dpsizdz  = gradient<2, Order, Z>(f.psiz, ix, 0, iz, grad_ctx);
    float dpsixdx  = gradient<2, Order, X>(f.psix, ix, 0, iz, grad_ctx);

    float dazdz    = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
    float daxdx    = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);

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
__global__ void acoustic2nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    LaplaceParam lap_ctx,
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

    float w_sum = 0.0f;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

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