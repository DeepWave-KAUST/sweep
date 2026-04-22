#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"

#define ACOUSTIC3D(order, grid, block, ...)                                          \
    do {                                                                                    \
        if      ((order) == 2) acoustic_forward_kernel_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_forward_kernel_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_forward_kernel_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_forward_kernel_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_forward_kernel_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define ACOUSTIC3D_NOPML(order, grid, block, ...)                           \
    do {                                                                           \
        if      ((order) == 2) acoustic_nopml_3d<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) acoustic_nopml_3d<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) acoustic_nopml_3d<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) acoustic_nopml_3d<8><<<grid, block>>>(__VA_ARGS__); \
        else                   acoustic_nopml_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

template<int Order>
__device__ __forceinline__ float acoustic3d_grad_coeff(int m, const float* coeff)
{
    if constexpr (Order == 2) {
        return m == 1 ? 0.5f : 0.0f;
    } else if constexpr (Order == 4) {
        return m == 1 ? (8.0f / 12.0f) : (m == 2 ? (-1.0f / 12.0f) : 0.0f);
    } else if constexpr (Order == 6) {
        return m == 1 ? 0.75f : (m == 2 ? (-3.0f / 20.0f) : (m == 3 ? (1.0f / 60.0f) : 0.0f));
    } else if constexpr (Order == 8) {
        return m == 1 ? (4.0f / 5.0f)
             : (m == 2 ? (-1.0f / 5.0f)
             : (m == 3 ? (4.0f / 105.0f)
             : (m == 4 ? (-1.0f / 280.0f) : 0.0f)));
    } else {
        return coeff[m];
    }
}

template<int Order, int Direction>
__device__ __forceinline__ float acoustic_grad_product_3d(
    const float* __restrict__ a_1d,
    const float* __restrict__ psi,
    int ix,
    int iy,
    int iz,
    int nx,
    int ny,
    const GradParam& grad_ctx
)
{
    constexpr bool along_x = (Direction & X);
    constexpr bool along_y = (Direction & Y);
    constexpr bool along_z = (Direction & Z);
    static_assert(along_x || along_y || along_z, "Direction must include X, Y, or Z.");

    const int half_order = (Order == -1) ? grad_ctx.M : (Order / 2);
    const int stride = along_x ? 1 : (along_y ? nx : (nx * ny));
    const int coord = along_x ? ix : (along_y ? iy : iz);
    const float spacing = along_x ? grad_ctx.dx : (along_y ? grad_ctx.dy : grad_ctx.dz);
    const int center = iz * nx * ny + iy * nx + ix;

    float grad = 0.0f;
    for (int m = 1; m <= half_order; ++m) {
        const float c = acoustic3d_grad_coeff<Order>(m, grad_ctx.coeff);
        grad += c * (
            a_1d[coord + m] * psi[center + m * stride] -
            a_1d[coord - m] * psi[center - m * stride]
        );
    }
    return grad / spacing;
}

template<int Order>
__global__ void acoustic_forward_kernel_3d(
    AcousticWavefieldPointer wf,

    bool save_all_wavefields,
    float* __restrict__ u_this,

    const float* __restrict__ vp,

    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
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

    auto f = wf.offset(b, spatial_size);

    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    int x0 = solver.phys_x0();
    int x1 = solver.phys_x1();
    int y0 = solver.phys_y0();
    int y1 = solver.phys_y1();
    int z0 = solver.phys_z0();
    int z1 = solver.phys_z1();

    bool in_pml_x = (ix < x0) || (ix >= x1);
    bool in_pml_y = (iy < y0) || (iy >= y1);
    bool in_pml_z = (iz < z0) || (iz >= z1);

    // =========================================================
    // Laplace
    // =========================================================

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    float w_sum = 0.f;

    if (!(in_pml_x || in_pml_y || in_pml_z)) {
        w_sum = lap_x + lap_y + lap_z;

        float u0 = f.u_now[idx];
        float v  = vp_b[idx];

        f.u_next[idx] =
            2.f * u0 -
            f.u_prev[idx] +
            (v * v) * solver.dt * solver.dt * w_sum;

        if (u_this_b != nullptr)
            u_this_b[idx] = (v * v) * w_sum;

        return;
    }

    // =========================================================
    // Gradients
    // =========================================================

    // =========================================================
    // X direction
    // =========================================================
    if (in_pml_x) {
        float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
        float ax_ = cpml.ax[ix];
        float bx_ = cpml.bx[ix];
        float dbxdx_ = cpml.dbxdx[ix];

        float tmpx = ((1.f + bx_) * lap_x + dbxdx_ * dudx)
                     + acoustic_grad_product_3d<Order, X>(cpml.ax, f.psix, ix, iy, iz, solver.nx, solver.ny, grad_ctx);

        w_sum += (1.f + bx_) * tmpx + ax_ * f.zetax[idx];

        f.psix[idx]  = bx_ * dudx + ax_ * f.psix[idx];
        f.zetax[idx] = bx_ * tmpx + ax_ * f.zetax[idx];
    } else {
        w_sum += lap_x;
    }

    // =========================================================
    // Y direction
    // =========================================================
    if (in_pml_y) {
        float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
        float ay_ = cpml.ay[iy];
        float by_ = cpml.by[iy];
        float dbydy_ = cpml.dbydy[iy];

        float tmpy = ((1.f + by_) * lap_y + dbydy_ * dudy)
                     + acoustic_grad_product_3d<Order, Y>(cpml.ay, f.psiy, ix, iy, iz, solver.nx, solver.ny, grad_ctx);

        w_sum += (1.f + by_) * tmpy + ay_ * f.zetay[idx];

        f.psiy[idx]  = by_ * dudy + ay_ * f.psiy[idx];
        f.zetay[idx] = by_ * tmpy + ay_ * f.zetay[idx];
    } else {
        w_sum += lap_y;
    }

    // =========================================================
    // Z direction
    // =========================================================
    if (in_pml_z) {
        float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);
        float az_ = cpml.az[iz];
        float bz_ = cpml.bz[iz];
        float dbzdz_ = cpml.dbzdz[iz];

        float tmpz = ((1.f + bz_) * lap_z + dbzdz_ * dudz)
                     + acoustic_grad_product_3d<Order, Z>(cpml.az, f.psiz, ix, iy, iz, solver.nx, solver.ny, grad_ctx);

        w_sum += (1.f + bz_) * tmpz + az_ * f.zetaz[idx];

        f.psiz[idx]  = bz_ * dudz + az_ * f.psiz[idx];
        f.zetaz[idx] = bz_ * tmpz + az_ * f.zetaz[idx];
    } else {
        w_sum += lap_z;
    }

    // =========================================================
    // Time update
    // =========================================================

    float u0 = f.u_now[idx];
    float v  = vp_b[idx];

    f.u_next[idx] =
        2.f * u0 -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
        u_this_b[idx] = (v * v) * (lap_x + lap_y + lap_z);
}


template<int Order>
__global__ void acoustic_nopml_3d(
    AcousticWavefieldPointer wf,
    
    float* __restrict__ u_this,
    const float* __restrict__ vp,

    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    int M;
    if constexpr (Order == -1) {
        M = solver.M;
    } else {
        M = Order / 2;
    }

    int halo = solver.abcn > 0 ? solver.abcn + 2*M+1 : 2*M;

    int top_halo = solver.free_surface ? 2*M : halo;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < top_halo || iz >= solver.nz - halo)
        return;
    
    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    
    auto f = wf.offset(b, spatial_size);

    int idx = iz * stride_z + iy * stride_y + ix;

    float*       u_this_b = u_this ? u_this + spatial_size * b : nullptr;
    const float* vp_b     = vp     + spatial_size * b;

    float lap_x, lap_y, lap_z;

    float w_sum = 0.0f;

    lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    w_sum = lap_x + lap_y + lap_z;

    float u0 = f.u_now[idx];
    float v  = vp_b[idx];

    f.u_next[idx] =
        2.0f * u0 -
        f.u_prev[idx] +
        (v * v) * solver.dt * solver.dt * w_sum;

    if (u_this_b != nullptr)
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

__global__ void accumulate_rtm_image_3d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ image,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int B, int nx, int ny, int nz
);

__global__ void accumulate_source_grad_3d(
    const float* __restrict__ u_backward,
    float* __restrict__ grad_source,
    const int* __restrict__ sources_loc,
    int it,
    int nsrc,
    SolverContext solver
);
