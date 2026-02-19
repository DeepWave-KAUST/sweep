#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/gradient3d.cuh"
#include "../../operators/laplace3d.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

#define LAUNCH_FORWARD_3D(order, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic_forward_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_forward_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_forward_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_forward_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_forward_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_FORWARD_3D_NOPML(order, ...)                                  \
    do {                                                        \
        if      ((order) == 2) acoustic_nopml_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_nopml_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_nopml_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_nopml_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_nopml_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

template<int Order>
__global__ void acoustic_forward_kernel_3d(
    AcousticWavefieldPointer wf,

    bool save_all_wavefields,
    float* __restrict__ u_this,

    const float* __restrict__ vp,

    AcousticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);

    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_now_b  = wf.u_now  + spatial_size * b;
    const float* u_prev_b = wf.u_prev + spatial_size * b;
    float*       u_next_b = wf.u_next + spatial_size * b;
    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    float* psix_b = wf.psix + spatial_size * b;
    float* psiy_b = wf.psiy + spatial_size * b;
    float* psiz_b = wf.psiz + spatial_size * b;

    float* zetax_b = wf.zetax + spatial_size * b;
    float* zetay_b = wf.zetay + spatial_size * b;
    float* zetaz_b = wf.zetaz + spatial_size * b;

    Laplace3dContext ctx{solver.nx, solver.ny, ix, iy, iz, solver.M, solver.lap_coeff, solver.dx, solver.dy, solver.dz};
    GradContext3D gctx{1, solver.nx, solver.nx*solver.ny, ix, iy, iz, solver.M, solver.grad_coeff, solver.dx, solver.dy, solver.dz};
    GradContext3D gctx_x{
        1,      // sx
        0,      // sy
        0,      // sz
        ix,     // ix
        0,      // iy
        0,      // iz
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };
    GradContext3D gctx_y{
        0,
        1,
        0,
        0,
        iy,
        0,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };
    GradContext3D gctx_z{
        0,
        0,
        1,
        0,
        0,
        iz,
        solver.M,
        solver.grad_coeff,
        solver.dx, solver.dy, solver.dz
    };

    // =========================================================
    // Laplace
    // =========================================================

    float lap_x = laplace<Order, LAPLACE_X>(u_now_b, ctx);
    float lap_y = laplace<Order, LAPLACE_Y>(u_now_b, ctx);
    float lap_z = laplace<Order, LAPLACE_Z>(u_now_b, ctx);

    // =========================================================
    // Gradients
    // =========================================================

    float dudx = gradient<Order, GRAD_X>(u_now_b, gctx);
    float dudy = gradient<Order, GRAD_Y>(u_now_b, gctx);
    float dudz = gradient<Order, GRAD_Z>(u_now_b, gctx);

    float dpsixdx = gradient<Order, GRAD_X>(psix_b, gctx);
    float dpsiydy = gradient<Order, GRAD_Y>(psiy_b, gctx);
    float dpsizdz = gradient<Order, GRAD_Z>(psiz_b, gctx);

    float daxdx = gradient<Order, GRAD_X>(cpml.ax, gctx_x);
    float daydy = gradient<Order, GRAD_Y>(cpml.ay, gctx_y);
    float dazdz = gradient<Order, GRAD_Z>(cpml.az, gctx_z);

    float ax_ = cpml.ax[ix];
    float ay_ = cpml.ay[iy];
    float az_ = cpml.az[iz];

    float bx_ = cpml.bx[ix];
    float by_ = cpml.by[iy];
    float bz_ = cpml.bz[iz];

    float dbxdx_ = cpml.dbxdx[ix];
    float dbydy_ = cpml.dbydy[iy];
    float dbzdz_ = cpml.dbzdz[iz];

    float w_sum = 0.f;

    // =========================================================
    // X direction
    // =========================================================

    float tmpx = ((1.f + bx_) * lap_x + dbxdx_ * dudx)
                 + (daxdx * psix_b[idx] + ax_ * dpsixdx);

    w_sum += (1.f + bx_) * tmpx + ax_ * zetax_b[idx];

    psix_b[idx]  = bx_ * dudx + ax_ * psix_b[idx];
    zetax_b[idx] = bx_ * tmpx + ax_ * zetax_b[idx];

    // =========================================================
    // Y direction
    // =========================================================

    float tmpy = ((1.f + by_) * lap_y + dbydy_ * dudy)
                 + (daydy * psiy_b[idx] + ay_ * dpsiydy);

    w_sum += (1.f + by_) * tmpy + ay_ * zetay_b[idx];

    psiy_b[idx]  = by_ * dudy + ay_ * psiy_b[idx];
    zetay_b[idx] = by_ * tmpy + ay_ * zetay_b[idx];

    // =========================================================
    // Z direction
    // =========================================================

    float tmpz = ((1.f + bz_) * lap_z + dbzdz_ * dudz)
                 + (dazdz * psiz_b[idx] + az_ * dpsizdz);

    w_sum += (1.f + bz_) * tmpz + az_ * zetaz_b[idx];

    psiz_b[idx]  = bz_ * dudz + az_ * psiz_b[idx];
    zetaz_b[idx] = bz_ * tmpz + az_ * zetaz_b[idx];

    // =========================================================
    // Time update
    // =========================================================

    float u0 = u_now_b[idx];
    float v  = vp_b[idx];

    u_next_b[idx] =
        2.f * u0 -
        u_prev_b[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_y + lap_z);
}


template<int Order>
__global__ void acoustic_nopml_3d(
    AcousticWavefieldPointer wf,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);

    int halo = is_runtime ? solver.M : M_static;

    int top_halo = solver.free_surface ? solver.M : halo;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < top_halo || iz >= solver.nz - halo)
        return;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_now_b  = wf.u_now  + spatial_size * b;
    const float* u_prev_b = wf.u_prev + spatial_size * b;
    float*       u_next_b = wf.u_next + spatial_size * b;
    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    float lap_x, lap_y, lap_z;

    float w_sum = 0.0f;

    Laplace3dContext ctx{solver.nx, solver.ny, ix, iy, iz, solver.M, solver.lap_coeff, solver.dx, solver.dy, solver.dz};

    lap_x    = laplace<Order, LAPLACE_X>(u_now_b, ctx);
    lap_y    = laplace<Order, LAPLACE_Y>(u_now_b, ctx);
    lap_z    = laplace<Order, LAPLACE_Z>(u_now_b, ctx);

    w_sum = lap_x + lap_y + lap_z;

    float u0 = u_now_b[idx];
    float v  = vp_b[idx];

    u_next_b[idx] =
        2.0f * u0 -
        u_prev_b[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    u_this_b[idx] = (v * v) * w_sum;
}

__global__ void calculate_grad_3d(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz
);

__global__ void calculate_grad_utt_3d(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz, float dt
);

__global__ void save_boundary_kernel_3d(
    const float* __restrict__ u,    // (B, nz, ny, nx)

    float* __restrict__ front,      // (nt, B, M, ny, nx)
    float* __restrict__ back,

    float* __restrict__ top,        // (nt, B, nz, M, nx)
    float* __restrict__ bottom,

    float* __restrict__ left,       // (nt, B, nz, ny, M)
    float* __restrict__ right,

    int it,
    SolverContext solver
);

__global__ void restore_boundary_kernel_3d(
    float* __restrict__ u,        // (B, nz, ny, nx)

    const float* __restrict__ front,   // (nt, B, n, ny, nx)
    const float* __restrict__ back,

    const float* __restrict__ top,     // (nt, B, nz, n, nx)
    const float* __restrict__ bottom,

    const float* __restrict__ left,    // (nt, B, nz, ny, n)
    const float* __restrict__ right,

    int it,
    SolverContext solver
);