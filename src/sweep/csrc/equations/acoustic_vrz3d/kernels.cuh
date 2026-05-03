#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

template<int Order>
__global__ void acoustic_vrz3nd(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
);

static __global__ void build_kappa_lambda_vrz3d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
);

template<int Order>
__global__ void calculate_grad_vrz3d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ kappa_lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    GradParam grad_ctx,
    LaplaceParam lap_ctx,
    SolverContext solver
);

#define ACOUSTIC_VRZ3D(order, grid, block, ...)                                             \
    do {                                                                                    \
        if      ((order) == 2) acoustic_vrz3nd<2><<<grid, block>>>(__VA_ARGS__);            \
        else if ((order) == 4) acoustic_vrz3nd<4><<<grid, block>>>(__VA_ARGS__);            \
        else if ((order) == 6) acoustic_vrz3nd<6><<<grid, block>>>(__VA_ARGS__);            \
        else if ((order) == 8) acoustic_vrz3nd<8><<<grid, block>>>(__VA_ARGS__);            \
        else                   acoustic_vrz3nd<-1><<<grid, block>>>(__VA_ARGS__);           \
    } while (0)

#define ACOUSTIC_VRZ3D_ADJOINT(order, grid, block, ...)                                     \
    do {                                                                                    \
        if      ((order) == 2) acoustic_vrz3nd<2><<<grid, block>>>(__VA_ARGS__);            \
        else if ((order) == 4) acoustic_vrz3nd<4><<<grid, block>>>(__VA_ARGS__);            \
        else if ((order) == 6) acoustic_vrz3nd<6><<<grid, block>>>(__VA_ARGS__);            \
        else if ((order) == 8) acoustic_vrz3nd<8><<<grid, block>>>(__VA_ARGS__);            \
        else                   acoustic_vrz3nd<-1><<<grid, block>>>(__VA_ARGS__);           \
    } while (0)

#define CALCULATE_GRAD_VRZ3D(order, grid, block, ...)                                       \
    do {                                                                                    \
        if      ((order) == 2) calculate_grad_vrz3d<2><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 4) calculate_grad_vrz3d<4><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 6) calculate_grad_vrz3d<6><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 8) calculate_grad_vrz3d<8><<<grid, block>>>(__VA_ARGS__);       \
        else                   calculate_grad_vrz3d<-1><<<grid, block>>>(__VA_ARGS__);      \
    } while (0)

template<int Order>
__device__ inline void vrz3d_index(
    SolverContext solver,
    int& ix,
    int& iy,
    int& iz,
    int& b
) {
    ix = blockIdx.x * blockDim.x + threadIdx.x;
    iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    b = iz_global / solver.nz;
    iz = iz_global % solver.nz;
}

template<int Order>
__device__ inline bool vrz3d_interior(SolverContext solver, int ix, int iy, int iz, int b)
{
    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return false;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    return ix >= halo && ix < solver.nx - halo
        && iy >= halo && iy < solver.ny - halo
        && iz >= halo && iz < solver.nz - halo;
}

template<int Order, int Direction>
__device__ __forceinline__ bool vrz3d_grad_product_interior(
    SolverContext solver,
    int ix,
    int iy,
    int iz
) {
    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    return ix >= halo && ix < solver.nx - halo
        && iy >= halo && iy < solver.ny - halo
        && iz >= halo && iz < solver.nz - halo;
}

template<int Order, int Direction>
__device__ __forceinline__ float vrz3d_q_gradp(
    const float* __restrict__ q,
    const float* __restrict__ p,
    int ix,
    int iy,
    int iz,
    GradParam grad_ctx,
    SolverContext solver
) {
    if (!vrz3d_grad_product_interior<Order, Direction>(solver, ix, iy, iz))
        return 0.f;

    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    return q[idx] * gradient<3, Order, Direction>(p, ix, iy, iz, grad_ctx);
}

template<int Order, int Direction>
__device__ __forceinline__ float vrz3d_grad_q_gradp(
    const float* __restrict__ q,
    const float* __restrict__ p,
    int ix,
    int iy,
    int iz,
    GradParam grad_ctx,
    SolverContext solver
) {
    int sx = 0, sy = 0, sz = 0;
    float h = 1.f;

    if constexpr (Direction & X) {
        sx = 1;
        h = grad_ctx.dx;
    } else if constexpr (Direction & Y) {
        sy = 1;
        h = grad_ctx.dy;
    } else {
        sz = 1;
        h = grad_ctx.dz;
    }

    if constexpr (Order == 2) {
        return (
            vrz3d_q_gradp<Order, Direction>(q, p, ix + sx, iy + sy, iz + sz, grad_ctx, solver)
          - vrz3d_q_gradp<Order, Direction>(q, p, ix - sx, iy - sy, iz - sz, grad_ctx, solver)
        ) / (2.f * h);
    } else if constexpr (Order == 4) {
        constexpr float c1 = 8.f / 12.f;
        constexpr float c2 = -1.f / 12.f;
        return (
            c1 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + sx, iy + sy, iz + sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - sx, iy - sy, iz - sz, grad_ctx, solver)
            )
          + c2 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + 2 * sx, iy + 2 * sy, iz + 2 * sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - 2 * sx, iy - 2 * sy, iz - 2 * sz, grad_ctx, solver)
            )
        ) / h;
    } else if constexpr (Order == 6) {
        constexpr float c1 = 3.f / 4.f;
        constexpr float c2 = -3.f / 20.f;
        constexpr float c3 = 1.f / 60.f;
        return (
            c1 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + sx, iy + sy, iz + sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - sx, iy - sy, iz - sz, grad_ctx, solver)
            )
          + c2 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + 2 * sx, iy + 2 * sy, iz + 2 * sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - 2 * sx, iy - 2 * sy, iz - 2 * sz, grad_ctx, solver)
            )
          + c3 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + 3 * sx, iy + 3 * sy, iz + 3 * sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - 3 * sx, iy - 3 * sy, iz - 3 * sz, grad_ctx, solver)
            )
        ) / h;
    } else if constexpr (Order == 8) {
        constexpr float c1 = 4.f / 5.f;
        constexpr float c2 = -1.f / 5.f;
        constexpr float c3 = 4.f / 105.f;
        constexpr float c4 = -1.f / 280.f;
        return (
            c1 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + sx, iy + sy, iz + sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - sx, iy - sy, iz - sz, grad_ctx, solver)
            )
          + c2 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + 2 * sx, iy + 2 * sy, iz + 2 * sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - 2 * sx, iy - 2 * sy, iz - 2 * sz, grad_ctx, solver)
            )
          + c3 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + 3 * sx, iy + 3 * sy, iz + 3 * sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - 3 * sx, iy - 3 * sy, iz - 3 * sz, grad_ctx, solver)
            )
          + c4 * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + 4 * sx, iy + 4 * sy, iz + 4 * sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - 4 * sx, iy - 4 * sy, iz - 4 * sz, grad_ctx, solver)
            )
        ) / h;
    } else {
        float acc = 0.f;
        #pragma unroll 1
        for (int k = 1; k <= solver.M; ++k) {
            float coeff = grad_ctx.coeff[k];
            acc += coeff * (
                vrz3d_q_gradp<Order, Direction>(q, p, ix + k * sx, iy + k * sy, iz + k * sz, grad_ctx, solver)
              - vrz3d_q_gradp<Order, Direction>(q, p, ix - k * sx, iy - k * sy, iz - k * sz, grad_ctx, solver)
            );
        }
        return acc / h;
    }
}

template<int Order>
__global__ void acoustic_vrz3nd(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<Order>(solver, ix, iy, iz, b);
    if (!vrz3d_interior<Order>(solver, ix, iy, iz, b)) return;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;

    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);

    float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
    float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
    float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float ax_ = cpml.ax[ix];
    float ay_ = cpml.ay[iy];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float by_ = cpml.by[iy];
    float bz_ = cpml.bz[iz];

    float tmpx = ((1.0f + bx_) * lap_x + cpml.dbxdx[ix] * dudx)
               + daxdx * f.psix[idx] + ax_ * dpsixdx;
    float tmpy = ((1.0f + by_) * lap_y + cpml.dbydy[iy] * dudy)
               + daydy * f.psiy[idx] + ay_ * dpsiydy;
    float tmpz = ((1.0f + bz_) * lap_z + cpml.dbzdz[iz] * dudz)
               + dazdz * f.psiz[idx] + az_ * dpsizdz;

    float psixn = bx_ * dudx + ax_ * f.psix[idx];
    float psiyn = by_ * dudy + ay_ * f.psiy[idx];
    float psizn = bz_ * dudz + az_ * f.psiz[idx];
    float zetaxn = bx_ * tmpx + ax_ * f.zetax[idx];
    float zetayn = by_ * tmpy + ay_ * f.zetay[idx];
    float zetazn = bz_ * tmpz + az_ * f.zetaz[idx];

    float px = dudx + psixn;
    float py = dudy + psiyn;
    float pz = dudz + psizn;
    float w_sum = (1.0f + bx_) * tmpx + ax_ * f.zetax[idx]
                + (1.0f + by_) * tmpy + ay_ * f.zetay[idx]
                + (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;
    float rhs = kappa * (beta * w_sum + dbdx * px + dbdy * py + dbdz * pz);

    f.psix[idx] = psixn;
    f.psiy[idx] = psiyn;
    f.psiz[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetay[idx] = zetayn;
    f.zetaz[idx] = zetazn;
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

static __global__ void build_kappa_lambda_vrz3d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<2>(solver, ix, iy, iz, b);
    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = b * spatial_size + iz * solver.nx * solver.ny + iy * solver.nx + ix;
    kappa_lambda[idx] = vp[idx] * z[idx] * lambda_now[idx];
}

template<int Order>
__global__ void calculate_grad_vrz3d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ kappa_lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    GradParam grad_ctx,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<Order>(solver, ix, iy, iz, b);
    if (!vrz3d_interior<Order>(solver, ix, iy, iz, b)) return;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * stride_z + iy * stride_y + ix;
    int shift = b * spatial_size;

    const float* p_b = f_u_now + shift;
    const float* lambda_b = lambda_now + shift;
    const float* q_b = kappa_lambda_now + shift;
    const float* vp_b = vp + shift;
    const float* z_b = z + shift;
    const float* inv_z_b = inv_z + shift;
    float* grad_vp_b = grad_vp + shift;
    float* grad_z_b = grad_z + shift;

    float lap_x = laplace<3, Order, X>(p_b, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(p_b, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(p_b, ix, iy, iz, lap_ctx);

    float dpdx = gradient<3, Order, X>(p_b, ix, iy, iz, grad_ctx);
    float dpdy = gradient<3, Order, Y>(p_b, ix, iy, iz, grad_ctx);
    float dpdz = gradient<3, Order, Z>(p_b, ix, iy, iz, grad_ctx);

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;
    float div_b_grad_p = beta * (lap_x + lap_y + lap_z)
                       + dbdx * dpdx + dbdy * dpdy + dbdz * dpdz;

    float d_q_dpdx = vrz3d_grad_q_gradp<Order, X>(q_b, p_b, ix, iy, iz, grad_ctx, solver);
    float d_q_dpdy = vrz3d_grad_q_gradp<Order, Y>(q_b, p_b, ix, iy, iz, grad_ctx, solver);
    float d_q_dpdz = vrz3d_grad_q_gradp<Order, Z>(q_b, p_b, ix, iy, iz, grad_ctx, solver);

    float dt2 = solver.dt * solver.dt;
    float g_kappa = -dt2 * lambda_b[idx] * div_b_grad_p;
    float g_beta = dt2 * (d_q_dpdx + d_q_dpdy + d_q_dpdz - q_b[idx] * (lap_x + lap_y + lap_z));

    grad_vp_b[idx] += z_b[idx] * g_kappa + inv_z0 * g_beta;
    grad_z_b[idx] += v * g_kappa - beta * inv_z0 * g_beta;
}
