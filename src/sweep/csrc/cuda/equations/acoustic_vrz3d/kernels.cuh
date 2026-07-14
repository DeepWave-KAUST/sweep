#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../common/acoustic_vrz_fused.cuh"
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

template<int Order>
__global__ void acoustic_vrz3nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
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
__global__ void build_vrz_grad_fields_3d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ c_x, float* __restrict__ c_y, float* __restrict__ c_z,
    float* __restrict__ e_x, float* __restrict__ e_y, float* __restrict__ e_z,
    GradParam grad_ctx,
    SolverContext solver
);

template<int Order>
__global__ void build_vrz_adjoint_fields_3d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ aq0, float* __restrict__ aqx,
    float* __restrict__ aqy, float* __restrict__ aqz,
    GradParam grad_ctx,
    SolverContext solver
);

template<int Order>
__global__ void acoustic_vrz3nd_adjoint(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    const float* __restrict__ aq0,
    const float* __restrict__ aqx,
    const float* __restrict__ aqy,
    const float* __restrict__ aqz,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_y,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
);

template<int Order>
__global__ void calculate_grad_vrz3d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ c_x, const float* __restrict__ c_y, const float* __restrict__ c_z,
    const float* __restrict__ e_x, const float* __restrict__ e_y, const float* __restrict__ e_z,
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

#define ACOUSTIC_VRZ3D_NOPML(order, grid, block, ...)                                      \
    do {                                                                                    \
        if      ((order) == 2) acoustic_vrz3nd_nopml<2><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 4) acoustic_vrz3nd_nopml<4><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 6) acoustic_vrz3nd_nopml<6><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 8) acoustic_vrz3nd_nopml<8><<<grid, block>>>(__VA_ARGS__);      \
        else                   acoustic_vrz3nd_nopml<-1><<<grid, block>>>(__VA_ARGS__);     \
    } while (0)

#define ACOUSTIC_VRZ3D_ADJOINT(order, grid, block, ...)                                     \
    do {                                                                                    \
        if      ((order) == 2) acoustic_vrz3nd_adjoint<2><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 4) acoustic_vrz3nd_adjoint<4><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 6) acoustic_vrz3nd_adjoint<6><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 8) acoustic_vrz3nd_adjoint<8><<<grid, block>>>(__VA_ARGS__);    \
        else                   acoustic_vrz3nd_adjoint<-1><<<grid, block>>>(__VA_ARGS__);   \
    } while (0)

#define BUILD_VRZ_ADJOINT_FIELDS_3D(order, grid, block, ...)                                \
    do {                                                                                    \
        if      ((order) == 2) build_vrz_adjoint_fields_3d<2><<<grid, block>>>(__VA_ARGS__);\
        else if ((order) == 4) build_vrz_adjoint_fields_3d<4><<<grid, block>>>(__VA_ARGS__);\
        else if ((order) == 6) build_vrz_adjoint_fields_3d<6><<<grid, block>>>(__VA_ARGS__);\
        else if ((order) == 8) build_vrz_adjoint_fields_3d<8><<<grid, block>>>(__VA_ARGS__);\
        else                   build_vrz_adjoint_fields_3d<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define BUILD_VRZ_GRAD_FIELDS_3D(order, grid, block, ...)                                   \
    do {                                                                                    \
        if      ((order) == 2) build_vrz_grad_fields_3d<2><<<grid, block>>>(__VA_ARGS__);   \
        else if ((order) == 4) build_vrz_grad_fields_3d<4><<<grid, block>>>(__VA_ARGS__);   \
        else if ((order) == 6) build_vrz_grad_fields_3d<6><<<grid, block>>>(__VA_ARGS__);   \
        else if ((order) == 8) build_vrz_grad_fields_3d<8><<<grid, block>>>(__VA_ARGS__);   \
        else                   build_vrz_grad_fields_3d<-1><<<grid, block>>>(__VA_ARGS__);  \
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
        && iz >= halo && iz < solver.nz - halo
        && ix >= solver.phys_x0() && ix < solver.phys_x1()
        && iy >= solver.phys_y0() && iy < solver.phys_y1()
        && iz >= solver.phys_z0() && iz < solver.phys_z1();
}

#if 0
// Old fused path retained for comparison. It computes d/dx_i(q * d_i p),
// and the caller subtracted q * lap(p) to recover grad(q) * grad(p).
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
struct VRZ3DGradientProductAccessor {
    const float* q;
    const float* p;
    int ix;
    int iy;
    int iz;
    int sx;
    int sy;
    int sz;
    GradParam grad_ctx;
    SolverContext solver;

    __device__ __forceinline__
    float operator()(int offset) const
    {
        return vrz3d_q_gradp<Order, Direction>(
            q,
            p,
            ix + offset * sx,
            iy + offset * sy,
            iz + offset * sz,
            grad_ctx,
            solver
        );
    }
};

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

    return centered_gradient_stencil<Order>(
        VRZ3DGradientProductAccessor<Order, Direction>{
            q, p, ix, iy, iz, sx, sy, sz, grad_ctx, solver
        },
        solver.M,
        grad_ctx.coeff,
        h
    );
}
#endif

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
    if (!vrz3d_grad_product_interior<Order, Direction>(solver, ix, iy, iz))
        return 0.f;

    float dq = gradient<3, Order, Direction>(q, ix, iy, iz, grad_ctx);
    float dp = gradient<3, Order, Direction>(p, ix, iy, iz, grad_ctx);
    return dq * dp;
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

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

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

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    // Interior fast-path: ax/bx/dbxdx all vanish so psixn/psiyn/psizn and
    // zetaxn/zetayn/zetazn collapse to 0; rhs reduces to kappa*(beta*(sum_lap)
    // + dbdx*dudx + dbdy*dudy + dbdz*dudz). Skip the 6 aux-field writes and
    // the dpsix/dpsiy/dpsiz/daxdx/daydy/dazdz gradient loads.
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        float rhs = kappa * (beta * (lap_x + lap_y + lap_z) + dbdx * dudx + dbdy * dudy + dbdz * dudz);
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
        return;
    }

    float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
    float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
    float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

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

    float rhs = kappa * (beta * w_sum + dbdx * px + dbdy * py + dbdz * pz);

    (f.psixn ? f.psixn : f.psix)[idx] = psixn;   // race-free forward psi double-buffer
    (f.psiyn ? f.psiyn : f.psiy)[idx] = psiyn;
    (f.psizn ? f.psizn : f.psiz)[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetay[idx] = zetayn;
    f.zetaz[idx] = zetazn;
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

template<int Order>
__global__ void acoustic_vrz3nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<Order>(solver, ix, iy, iz, b);

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    int x_start = solver.phys_x0() + 1;
    int x_end = solver.phys_x1() - 1;
    int y_start = solver.phys_y0() + 1;
    int y_end = solver.phys_y1() - 1;
    int z_start = solver.phys_z0() + 1;
    int z_end = solver.phys_z1() - 1;

    if (ix < x_start || ix >= x_end ||
        iy < y_start || iy >= y_end ||
        iz < z_start || iz >= z_end)
        return;

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

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    float rhs = kappa * (beta * (lap_x + lap_y + lap_z) + dbdx * dudx + dbdy * dudy + dbdz * dudz);

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

// Reverse-mode buffers for the exact discrete-adjoint gradient (3-D):
//   c_d = λ·vp·∂_d p ,   e_d = λ·vp²·z·∂_d p   (d = x, y, z)
template<int Order>
__global__ void build_vrz_grad_fields_3d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ c_x, float* __restrict__ c_y, float* __restrict__ c_z,
    float* __restrict__ e_x, float* __restrict__ e_y, float* __restrict__ e_z,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<Order>(solver, ix, iy, iz, b);
    if (!vrz3d_interior<Order>(solver, ix, iy, iz, b)) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int gidx = b * spatial_size + idx;

    const float* p_b = f_u_now + b * spatial_size;
    const float* lam_b = lambda_now + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;

    float dpdx = gradient<3, Order, X>(p_b, ix, iy, iz, grad_ctx);
    float dpdy = gradient<3, Order, Y>(p_b, ix, iy, iz, grad_ctx);
    float dpdz = gradient<3, Order, Z>(p_b, ix, iy, iz, grad_ctx);
    float v = vp_b[idx];
    float lam = lam_b[idx];
    float lam_v = lam * v;
    float lam_v2z = lam * v * v * z_b[idx];

    c_x[gidx] = lam_v * dpdx;
    c_y[gidx] = lam_v * dpdy;
    c_z[gidx] = lam_v * dpdz;
    e_x[gidx] = lam_v2z * dpdx;
    e_y[gidx] = lam_v2z * dpdy;
    e_z[gidx] = lam_v2z * dpdz;
}

// Pre-multiply λ by the model-only adjoint coefficients (race-free split):
//   aq0 = b·κ·λ = vp²·λ ,  aq_d = (∂_d b·κ)·λ   (d = x, y, z)
template<int Order>
__global__ void build_vrz_adjoint_fields_3d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ aq0, float* __restrict__ aqx,
    float* __restrict__ aqy, float* __restrict__ aqz,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<Order>(solver, ix, iy, iz, b);
    if (!vrz3d_interior<Order>(solver, ix, iy, iz, b)) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int gidx = b * spatial_size + idx;

    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;
    const float* lam_b = lambda_now + b * spatial_size;

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);
    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;
    float lam = lam_b[idx];

    aq0[gidx] = v * v * lam;
    aqx[gidx] = dbdx * kappa * lam;
    aqy[gidx] = dbdy * kappa * lam;
    aqz[gidx] = dbdz * kappa * lam;
}

// Proper adjoint propagation (3-D): transpose Aᵀλ of the forward interior
//   A u = κ·(b·∇²u + ∂ₓb·∂ₓu + ∂_yb·∂_yu + ∂_zb·∂_zu) is
//   Aᵀλ = ∇²(vp²λ) − ∂ₓ((κ∂ₓb)λ) − ∂_y((κ∂_yb)λ) − ∂_z((κ∂_zb)λ),
// applied to the pre-multiplied buffers from build_vrz_adjoint_fields_3d.
// PML branch keeps the forward formulation (as in acoustic / 2-D VRZ).
template<int Order>
__global__ void acoustic_vrz3nd_adjoint(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    const float* __restrict__ aq0,
    const float* __restrict__ aqx,
    const float* __restrict__ aqy,
    const float* __restrict__ aqz,
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

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    auto f = wf.offset(b, spatial_size);

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        const float* aq0_b = aq0 + b * spatial_size;
        const float* aqx_b = aqx + b * spatial_size;
        const float* aqy_b = aqy + b * spatial_size;
        const float* aqz_b = aqz + b * spatial_size;
        float lap_x = laplace<3, Order, X>(aq0_b, ix, iy, iz, lap_ctx);
        float lap_y = laplace<3, Order, Y>(aq0_b, ix, iy, iz, lap_ctx);
        float lap_z = laplace<3, Order, Z>(aq0_b, ix, iy, iz, lap_ctx);
        float gx = gradient<3, Order, X>(aqx_b, ix, iy, iz, grad_ctx);
        float gy = gradient<3, Order, Y>(aqy_b, ix, iy, iz, grad_ctx);
        float gz = gradient<3, Order, Z>(aqz_b, ix, iy, iz, grad_ctx);
        float rhs = (lap_x + lap_y + lap_z) - gx - gy - gz;
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
        return;
    }

    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
    float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
    float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

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

    float rhs = kappa * (beta * w_sum + dbdx * px + dbdy * py + dbdz * pz);

    // Race-free adjoint psi double-buffer: dpsi*d* above neighbour-reads psi
    // in this same launch, so the new psi must land in psi*n (caller pairs
    // with swap_pml()).  zeta is only ever read at idx — in-place is safe.
    (f.psixn ? f.psixn : f.psix)[idx] = psixn;
    (f.psiyn ? f.psiyn : f.psiy)[idx] = psiyn;
    (f.psizn ? f.psizn : f.psiz)[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetay[idx] = zetayn;
    f.zetaz[idx] = zetazn;
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

// Exact discrete-adjoint gradient (3-D), reverse-mode transpose of the forward
//   rhs = vp²·∇²p + vp·(∇vp·∇p) + vp²·z·(∇(1/z)·∇p).  Mirrors calculate_grad_vrz2d.
template<int Order>
__global__ void calculate_grad_vrz3d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ c_x, const float* __restrict__ c_y, const float* __restrict__ c_z,
    const float* __restrict__ e_x, const float* __restrict__ e_y, const float* __restrict__ e_z,
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

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int shift = b * spatial_size;

    const float* p_b = f_u_now + shift;
    const float* lambda_b = lambda_now + shift;
    const float* cx_b = c_x + shift;
    const float* cy_b = c_y + shift;
    const float* cz_b = c_z + shift;
    const float* ex_b = e_x + shift;
    const float* ey_b = e_y + shift;
    const float* ez_b = e_z + shift;
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

    float div_c = gradient<3, Order, X>(cx_b, ix, iy, iz, grad_ctx)
                + gradient<3, Order, Y>(cy_b, ix, iy, iz, grad_ctx)
                + gradient<3, Order, Z>(cz_b, ix, iy, iz, grad_ctx);
    float div_e = gradient<3, Order, X>(ex_b, ix, iy, iz, grad_ctx)
                + gradient<3, Order, Y>(ey_b, ix, iy, iz, grad_ctx)
                + gradient<3, Order, Z>(ez_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float zz = z_b[idx];
    float inv_z0 = inv_z_b[idx];
    float lam = lambda_b[idx];

    float dp_dvp   = dpdx * dvpdx + dpdy * dvpdy + dpdz * dvpdz;
    float dp_dinvz = dpdx * z1x  + dpdy * z1y  + dpdz * z1z;

    float g_vp = 2.0f * v * lam * (lap_x + lap_y + lap_z)
               + lam * dp_dvp
               - div_c
               + 2.0f * v * zz * lam * dp_dinvz;
    float g_z  = lam * v * v * dp_dinvz
               + div_e * inv_z0 * inv_z0;

    float dt2 = solver.dt * solver.dt;
    grad_vp_b[idx] += -dt2 * g_vp;
    grad_z_b[idx]  += -dt2 * g_z;
}

// ===========================================================================
// Fused backward kernels (3-D). Same functor recompute-at-tap approach as the
// 2-D fused kernels, bit-exact (mod fast-math FMA) with the split prepare+apply
// path.  Time-invariant coefficients (vp², ∂_d b·κ) precomputed once.
// ===========================================================================

template<int Order>
__global__ void build_vrz_adjoint_coeffs_3d(
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ C0, float* __restrict__ Cx,
    float* __restrict__ Cy, float* __restrict__ Cz,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix, iy, iz, b;
    vrz3d_index<Order>(solver, ix, iy, iz, b);
    if (!vrz3d_interior<Order>(solver, ix, iy, iz, b)) return;

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int gidx = b * spatial_size + idx;

    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);
    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    C0[gidx] = v * v;
    Cx[gidx] = dbdx * kappa;
    Cy[gidx] = dbdy * kappa;
    Cz[gidx] = dbdz * kappa;
}

template<int Order>
__global__ void acoustic_vrz3nd_adjoint_fused(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    const float* __restrict__ C0,
    const float* __restrict__ Cx,
    const float* __restrict__ Cy,
    const float* __restrict__ Cz,
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

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    auto f = wf.offset(b, spatial_size);

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        const float* C0_b = C0 + b * spatial_size;
        const float* Cx_b = Cx + b * spatial_size;
        const float* Cy_b = Cy + b * spatial_size;
        const float* Cz_b = Cz + b * spatial_size;
        const float* lam = f.u_now;
        int sy = solver.nx;
        int sz = solver.nx * solver.ny;

        VrzProductAccessor a0x{C0_b, lam, idx, 1};
        VrzProductAccessor a0y{C0_b, lam, idx, sy};
        VrzProductAccessor a0z{C0_b, lam, idx, sz};
        float lap_x = centered_laplace1d_stencil<Order>(a0x, lap_ctx.dx, solver.M, lap_ctx.coeff);
        float lap_y = centered_laplace1d_stencil<Order>(a0y, lap_ctx.dy, solver.M, lap_ctx.coeff);
        float lap_z = centered_laplace1d_stencil<Order>(a0z, lap_ctx.dz, solver.M, lap_ctx.coeff);

        VrzProductAccessor axx{Cx_b, lam, idx, 1};
        VrzProductAccessor ayy{Cy_b, lam, idx, sy};
        VrzProductAccessor azz{Cz_b, lam, idx, sz};
        float gx = centered_gradient_stencil<Order>(axx, solver.M, grad_ctx.coeff, grad_ctx.dx);
        float gy = centered_gradient_stencil<Order>(ayy, solver.M, grad_ctx.coeff, grad_ctx.dy);
        float gz = centered_gradient_stencil<Order>(azz, solver.M, grad_ctx.coeff, grad_ctx.dz);

        float rhs = (lap_x + lap_y + lap_z) - gx - gy - gz;
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
        return;
    }

    // PML branch: identical to acoustic_vrz3nd_adjoint (recompute from u_now).
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<3, Order, X>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_y = laplace<3, Order, Y>(f.u_now, ix, iy, iz, lap_ctx);
    float lap_z = laplace<3, Order, Z>(f.u_now, ix, iy, iz, lap_ctx);

    float dudx = gradient<3, Order, X>(f.u_now, ix, iy, iz, grad_ctx);
    float dudy = gradient<3, Order, Y>(f.u_now, ix, iy, iz, grad_ctx);
    float dudz = gradient<3, Order, Z>(f.u_now, ix, iy, iz, grad_ctx);

    float dvpdx = gradient<3, Order, X>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdy = gradient<3, Order, Y>(vp_b, ix, iy, iz, grad_ctx);
    float dvpdz = gradient<3, Order, Z>(vp_b, ix, iy, iz, grad_ctx);
    float z1x = gradient<3, Order, X>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1y = gradient<3, Order, Y>(inv_z_b, ix, iy, iz, grad_ctx);
    float z1z = gradient<3, Order, Z>(inv_z_b, ix, iy, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdy = dvpdy * inv_z0 + v * z1y;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    float dpsixdx = gradient<3, Order, X>(f.psix, ix, iy, iz, grad_ctx);
    float dpsiydy = gradient<3, Order, Y>(f.psiy, ix, iy, iz, grad_ctx);
    float dpsizdz = gradient<3, Order, Z>(f.psiz, ix, iy, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float daydy = gradient<2, Order, X>(cpml.ay, iy, 0, 0, grad_ctx_y);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

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

    float rhs = kappa * (beta * w_sum + dbdx * px + dbdy * py + dbdz * pz);

    (f.psixn ? f.psixn : f.psix)[idx] = psixn;
    (f.psiyn ? f.psiyn : f.psiy)[idx] = psiyn;
    (f.psizn ? f.psizn : f.psiz)[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetay[idx] = zetayn;
    f.zetaz[idx] = zetazn;
    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

// Accessor recomputing c_d = λ·vp·∂_d p (UseE=false) or e_d = λ·vp²·z·∂_d p
// (UseE=true) at stencil tap `off` along Dir, matching build_vrz_grad_fields_3d.
template<int Order, int Dir, bool UseE>
struct VrzGradAccessor3D {
    const float* __restrict__ p;
    const float* __restrict__ lam;
    const float* __restrict__ vp;
    const float* __restrict__ z;
    int ix, iy, iz, b;
    SolverContext solver;
    GradParam grad_ctx;

    __device__ __forceinline__ float operator()(int off) const
    {
        int jx = ix, jy = iy, jz = iz;
        if constexpr (Dir & X) jx = ix + off;
        else if constexpr (Dir & Y) jy = iy + off;
        else jz = iz + off;
        if (!vrz3d_interior<Order>(solver, jx, jy, jz, b))
            return 0.f;
        int j = jz * solver.nx * solver.ny + jy * solver.nx + jx;
        float dpd = gradient<3, Order, Dir>(p, jx, jy, jz, grad_ctx);
        float lam_j = lam[j];
        float v_j = vp[j];
        if constexpr (UseE) {
            float lam_v2z = ((lam_j * v_j) * v_j) * z[j];
            return lam_v2z * dpd;
        } else {
            float lam_v = lam_j * v_j;
            return lam_v * dpd;
        }
    }
};

template<int Order>
__global__ void calculate_grad_vrz3d_fused(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
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

    int spatial_size = solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    int shift = b * spatial_size;

    const float* p_b = f_u_now + shift;
    const float* lambda_b = lambda_now + shift;
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

    VrzGradAccessor3D<Order, X, false> cx{p_b, lambda_b, vp_b, z_b, ix, iy, iz, b, solver, grad_ctx};
    VrzGradAccessor3D<Order, Y, false> cy{p_b, lambda_b, vp_b, z_b, ix, iy, iz, b, solver, grad_ctx};
    VrzGradAccessor3D<Order, Z, false> cz{p_b, lambda_b, vp_b, z_b, ix, iy, iz, b, solver, grad_ctx};
    float div_c = centered_gradient_stencil<Order>(cx, solver.M, grad_ctx.coeff, grad_ctx.dx)
                + centered_gradient_stencil<Order>(cy, solver.M, grad_ctx.coeff, grad_ctx.dy)
                + centered_gradient_stencil<Order>(cz, solver.M, grad_ctx.coeff, grad_ctx.dz);
    VrzGradAccessor3D<Order, X, true> ex{p_b, lambda_b, vp_b, z_b, ix, iy, iz, b, solver, grad_ctx};
    VrzGradAccessor3D<Order, Y, true> ey{p_b, lambda_b, vp_b, z_b, ix, iy, iz, b, solver, grad_ctx};
    VrzGradAccessor3D<Order, Z, true> ez{p_b, lambda_b, vp_b, z_b, ix, iy, iz, b, solver, grad_ctx};
    float div_e = centered_gradient_stencil<Order>(ex, solver.M, grad_ctx.coeff, grad_ctx.dx)
                + centered_gradient_stencil<Order>(ey, solver.M, grad_ctx.coeff, grad_ctx.dy)
                + centered_gradient_stencil<Order>(ez, solver.M, grad_ctx.coeff, grad_ctx.dz);

    float v = vp_b[idx];
    float zz = z_b[idx];
    float inv_z0 = inv_z_b[idx];
    float lam = lambda_b[idx];

    float dp_dvp   = dpdx * dvpdx + dpdy * dvpdy + dpdz * dvpdz;
    float dp_dinvz = dpdx * z1x  + dpdy * z1y  + dpdz * z1z;

    float g_vp = 2.0f * v * lam * (lap_x + lap_y + lap_z)
               + lam * dp_dvp
               - div_c
               + 2.0f * v * zz * lam * dp_dinvz;
    float g_z  = lam * v * v * dp_dinvz
               + div_e * inv_z0 * inv_z0;

    float dt2 = solver.dt * solver.dt;
    grad_vp_b[idx] += -dt2 * g_vp;
    grad_z_b[idx]  += -dt2 * g_z;
}

#define BUILD_VRZ_ADJOINT_COEFFS_3D(order, grid, block, ...)                                    \
    do {                                                                                        \
        if      ((order) == 2) build_vrz_adjoint_coeffs_3d<2><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 4) build_vrz_adjoint_coeffs_3d<4><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 6) build_vrz_adjoint_coeffs_3d<6><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 8) build_vrz_adjoint_coeffs_3d<8><<<grid, block>>>(__VA_ARGS__);    \
        else                   build_vrz_adjoint_coeffs_3d<-1><<<grid, block>>>(__VA_ARGS__);   \
    } while (0)

#define ACOUSTIC_VRZ3D_ADJOINT_FUSED(order, grid, block, ...)                                   \
    do {                                                                                        \
        if      ((order) == 2) acoustic_vrz3nd_adjoint_fused<2><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 4) acoustic_vrz3nd_adjoint_fused<4><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 6) acoustic_vrz3nd_adjoint_fused<6><<<grid, block>>>(__VA_ARGS__);  \
        else if ((order) == 8) acoustic_vrz3nd_adjoint_fused<8><<<grid, block>>>(__VA_ARGS__);  \
        else                   acoustic_vrz3nd_adjoint_fused<-1><<<grid, block>>>(__VA_ARGS__); \
    } while (0)

#define CALCULATE_GRAD_VRZ3D_FUSED(order, grid, block, ...)                                     \
    do {                                                                                        \
        if      ((order) == 2) calculate_grad_vrz3d_fused<2><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 4) calculate_grad_vrz3d_fused<4><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 6) calculate_grad_vrz3d_fused<6><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 8) calculate_grad_vrz3d_fused<8><<<grid, block>>>(__VA_ARGS__);     \
        else                   calculate_grad_vrz3d_fused<-1><<<grid, block>>>(__VA_ARGS__);    \
    } while (0)

// Gradient dispatch (see 2-D note): fuse for order<=4, split for order>=6 where
// the O(M²) nested stencils regress (order 8 measured ~0.67x on RTX 6000 Ada).
#define CALCULATE_GRAD_VRZ3D_AUTO(order, grid, block, P, LAM, VP, Z, INVZ, CX, CY, CZ, EX, EY, EZ, GVP, GZ, GCTX, LCTX, SCTX) \
    do {                                                                                        \
        if ((order) == 2 || (order) == 4) {                                                     \
            CALCULATE_GRAD_VRZ3D_FUSED((order), grid, block,                                    \
                P, LAM, VP, Z, INVZ, GVP, GZ, GCTX, LCTX, SCTX);                                \
        } else {                                                                                \
            BUILD_VRZ_GRAD_FIELDS_3D((order), grid, block,                                      \
                P, LAM, VP, Z, CX, CY, CZ, EX, EY, EZ, GCTX, SCTX);                             \
            CALCULATE_GRAD_VRZ3D((order), grid, block,                                          \
                P, LAM, CX, CY, CZ, EX, EY, EZ, VP, Z, INVZ, GVP, GZ, GCTX, LCTX, SCTX);        \
        }                                                                                       \
    } while (0)
