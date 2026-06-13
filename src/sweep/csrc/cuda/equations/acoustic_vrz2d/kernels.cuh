#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

template<int Order>
__global__ void acoustic_vrz2nd(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
);

template<int Order>
__global__ void acoustic_vrz2nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    SolverContext solver
);

template<int Order>
__global__ void build_vrz_adjoint_fields(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ aq0,
    float* __restrict__ aqx,
    float* __restrict__ aqz,
    GradParam grad_ctx,
    SolverContext solver
);

template<int Order>
__global__ void acoustic_vrz2nd_adjoint(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    const float* __restrict__ aq0,   // b·κ·λ      (= vp²·λ)
    const float* __restrict__ aqx,   // (∂ₓb·κ)·λ
    const float* __restrict__ aqz,   // (∂_z b·κ)·λ
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
);

static __global__ void build_kappa_lambda_vrz2d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
);

// Pre-compute the reverse-mode buffers for the exact discrete-adjoint gradient:
//   c_d = λ·vp·∂_d p ,   e_d = λ·vp²·z·∂_d p   (d = x, z)
template<int Order>
__global__ void build_vrz_grad_fields(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ c_x,
    float* __restrict__ c_z,
    float* __restrict__ e_x,
    float* __restrict__ e_z,
    GradParam grad_ctx,
    SolverContext solver
);

template<int Order>
__global__ void calculate_grad_vrz2d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ c_x,
    const float* __restrict__ c_z,
    const float* __restrict__ e_x,
    const float* __restrict__ e_z,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    GradParam grad_ctx,
    LaplaceParam lap_ctx,
    SolverContext solver
);

#define ACOUSTIC_VRZ2D(order, grid, block, ...)                                              \
    do {                                                                                     \
        if      ((order) == 2) acoustic_vrz2nd<2><<<grid, block>>>(__VA_ARGS__);             \
        else if ((order) == 4) acoustic_vrz2nd<4><<<grid, block>>>(__VA_ARGS__);             \
        else if ((order) == 6) acoustic_vrz2nd<6><<<grid, block>>>(__VA_ARGS__);             \
        else if ((order) == 8) acoustic_vrz2nd<8><<<grid, block>>>(__VA_ARGS__);             \
        else                   acoustic_vrz2nd<-1><<<grid, block>>>(__VA_ARGS__);            \
    } while (0)

#define ACOUSTIC_VRZ2D_NOPML(order, grid, block, ...)                                        \
    do {                                                                                     \
        if      ((order) == 2) acoustic_vrz2nd_nopml<2><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 4) acoustic_vrz2nd_nopml<4><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 6) acoustic_vrz2nd_nopml<6><<<grid, block>>>(__VA_ARGS__);      \
        else if ((order) == 8) acoustic_vrz2nd_nopml<8><<<grid, block>>>(__VA_ARGS__);      \
        else                   acoustic_vrz2nd_nopml<-1><<<grid, block>>>(__VA_ARGS__);     \
    } while (0)

#define ACOUSTIC_VRZ2D_ADJOINT(order, grid, block, ...)                                      \
    do {                                                                                     \
        if      ((order) == 2) acoustic_vrz2nd_adjoint<2><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 4) acoustic_vrz2nd_adjoint<4><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 6) acoustic_vrz2nd_adjoint<6><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 8) acoustic_vrz2nd_adjoint<8><<<grid, block>>>(__VA_ARGS__);     \
        else                   acoustic_vrz2nd_adjoint<-1><<<grid, block>>>(__VA_ARGS__);    \
    } while (0)

#define CALCULATE_GRAD_VRZ2D(order, grid, block, ...)                                        \
    do {                                                                                     \
        if      ((order) == 2) calculate_grad_vrz2d<2><<<grid, block>>>(__VA_ARGS__);        \
        else if ((order) == 4) calculate_grad_vrz2d<4><<<grid, block>>>(__VA_ARGS__);        \
        else if ((order) == 6) calculate_grad_vrz2d<6><<<grid, block>>>(__VA_ARGS__);        \
        else if ((order) == 8) calculate_grad_vrz2d<8><<<grid, block>>>(__VA_ARGS__);        \
        else                   calculate_grad_vrz2d<-1><<<grid, block>>>(__VA_ARGS__);       \
    } while (0)

#define BUILD_VRZ_GRAD_FIELDS(order, grid, block, ...)                                       \
    do {                                                                                     \
        if      ((order) == 2) build_vrz_grad_fields<2><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 4) build_vrz_grad_fields<4><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 6) build_vrz_grad_fields<6><<<grid, block>>>(__VA_ARGS__);       \
        else if ((order) == 8) build_vrz_grad_fields<8><<<grid, block>>>(__VA_ARGS__);       \
        else                   build_vrz_grad_fields<-1><<<grid, block>>>(__VA_ARGS__);      \
    } while (0)

#define BUILD_VRZ_ADJOINT_FIELDS(order, grid, block, ...)                                    \
    do {                                                                                     \
        if      ((order) == 2) build_vrz_adjoint_fields<2><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 4) build_vrz_adjoint_fields<4><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 6) build_vrz_adjoint_fields<6><<<grid, block>>>(__VA_ARGS__);    \
        else if ((order) == 8) build_vrz_adjoint_fields<8><<<grid, block>>>(__VA_ARGS__);    \
        else                   build_vrz_adjoint_fields<-1><<<grid, block>>>(__VA_ARGS__);   \
    } while (0)

template<int Order, int Direction>
__device__ __forceinline__ bool vrz2d_grad_product_interior(
    SolverContext solver,
    int ix,
    int iz
) {
    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    return ix >= halo && ix < solver.nx - halo
        && iz >= halo && iz < solver.nz - halo
        && ix >= solver.phys_x0() && ix < solver.phys_x1()
        && iz >= solver.phys_z0() && iz < solver.phys_z1();
}

// Old fused path retained for comparison. It computes d/dx_i(q * d_i p),
// and the caller subtracted q * lap(p) to recover grad(q) * grad(p).
template<int Order, int Direction>
__device__ __forceinline__ float vrz2d_q_gradp(
    const float* __restrict__ q,
    const float* __restrict__ p,
    int ix,
    int iz,
    GradParam grad_ctx,
    SolverContext solver
) {
    if (!vrz2d_grad_product_interior<Order, Direction>(solver, ix, iz))
        return 0.f;

    int idx = iz * solver.nx + ix;
    return q[idx] * gradient<2, Order, Direction>(p, ix, 0, iz, grad_ctx);
}

template<int Order, int Direction>
struct VRZ2DGradientProductAccessor {
    const float* q;
    const float* p;
    int ix;
    int iz;
    int sx;
    int sz;
    GradParam grad_ctx;
    SolverContext solver;

    __device__ __forceinline__
    float operator()(int offset) const
    {
        return vrz2d_q_gradp<Order, Direction>(
            q,
            p,
            ix + offset * sx,
            iz + offset * sz,
            grad_ctx,
            solver
        );
    }
};

template<int Order, int Direction>
__device__ __forceinline__ float vrz2d_fused_grad_q_gradp(
    const float* __restrict__ q,
    const float* __restrict__ p,
    int ix,
    int iz,
    GradParam grad_ctx,
    SolverContext solver
) {
    int sx = 0, sz = 0;
    float h = 1.f;

    if constexpr (Direction & X) {
        sx = 1;
        h = grad_ctx.dx;
    } else {
        sz = 1;
        h = grad_ctx.dz;
    }

    return centered_gradient_stencil<Order>(
        VRZ2DGradientProductAccessor<Order, Direction>{
            q, p, ix, iz, sx, sz, grad_ctx, solver
        },
        solver.M,
        grad_ctx.coeff,
        h
    );
}

template<int Order, int Direction>
__device__ __forceinline__ float vrz2d_split_grad_q_gradp(
    const float* __restrict__ q,
    const float* __restrict__ p,
    int ix,
    int iz,
    GradParam grad_ctx,
    SolverContext solver
) {
    if (!vrz2d_grad_product_interior<Order, Direction>(solver, ix, iz))
        return 0.f;

    float dq = gradient<2, Order, Direction>(q, ix, 0, iz, grad_ctx);
    float dp = gradient<2, Order, Direction>(p, ix, 0, iz, grad_ctx);
    return dq * dp;
}

template<int Order>
__global__ void acoustic_vrz2nd(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    // PML / interior split. ax/bx/dbxdx vanish in the interior so psixn/psizn/
    // zetaxn/zetazn all collapse to 0; rhs reduces to kappa*(beta*(lap_x+lap_z)
    // + dbdx*dudx + dbdz*dudz). Skipping the dpsix/dpsiz/daxdx/dazdz loads and
    // four aux-field writes is the main bandwidth win here.
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        float rhs = kappa * (beta * (lap_x + lap_z) + dbdx * dudx + dbdz * dudz);
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
        if (save_all_wavefields && u_this_b != nullptr)
            u_this_b[idx] = rhs;
        return;
    }

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float dpsixdx = gradient<2, Order, X>(f.psix, ix, 0, iz, grad_ctx);
    float dpsizdz = gradient<2, Order, Z>(f.psiz, ix, 0, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dudx) + (daxdx * f.psix[idx] + ax_ * dpsixdx);
    float psixn = bx_ * dudx + ax_ * f.psix[idx];
    float zetaxn = bx_ * tmpx + ax_ * f.zetax[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dudz) + (dazdz * f.psiz[idx] + az_ * dpsizdz);
    float psizn = bz_ * dudz + az_ * f.psiz[idx];
    float zetazn = bz_ * tmpz + az_ * f.zetaz[idx];

    float px = dudx + psixn;
    float pz = dudz + psizn;
    float w_sum = (1.0f + bx_) * tmpx + ax_ * f.zetax[idx]
                + (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];

    float rhs = kappa * (beta * w_sum + dbdx * px + dbdz * pz);

    (f.psixn ? f.psixn : f.psix)[idx] = psixn;   // race-free forward psi double-buffer
    (f.psizn ? f.psizn : f.psiz)[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetaz[idx] = zetazn;

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;

    if (save_all_wavefields && u_this_b != nullptr)
        u_this_b[idx] = rhs;
}

template<int Order>
__global__ void acoustic_vrz2nd_nopml(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int x_start = solver.phys_x0() + 1;
    int x_end = solver.phys_x1() - 1;
    int z_start = solver.phys_z0() + 1;
    int z_end = solver.phys_z1() - 1;

    if (ix < x_start || ix >= x_end || iz < z_start || iz >= z_end)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float dudx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dudz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdz = dvpdz * inv_z0 + v * z1z;
    float rhs = kappa * (beta * (lap_x + lap_z) + dbdx * dudx + dbdz * dudz);

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

// Pre-multiply λ by the model-only adjoint coefficients so the transpose
// stencils in acoustic_vrz2nd_adjoint can be applied without a read-then-write
// race:  aq0 = b·κ·λ = vp²·λ,  aqx = (∂ₓb·κ)·λ,  aqz = (∂_z b·κ)·λ.
template<int Order>
__global__ void build_vrz_adjoint_fields(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ aq0,
    float* __restrict__ aqx,
    float* __restrict__ aqz,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int gidx = b * spatial_size + idx;

    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;
    const float* lam_b = lambda_now + b * spatial_size;

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);
    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdz = dvpdz * inv_z0 + v * z1z;
    float lam = lam_b[idx];

    aq0[gidx] = v * v * lam;            // b·κ·λ  (= vp²·λ)
    aqx[gidx] = dbdx * kappa * lam;     // (∂ₓb·κ)·λ
    aqz[gidx] = dbdz * kappa * lam;     // (∂_z b·κ)·λ
}

template<int Order>
__global__ void acoustic_vrz2nd_adjoint(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    const float* __restrict__ aq0,   // b·κ·λ      (= vp²·λ)
    const float* __restrict__ aqx,   // (∂ₓb·κ)·λ
    const float* __restrict__ aqz,   // (∂_z b·κ)·λ
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    // Position-based PML / interior split (mirrors the forward fast-path).
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        // Exact transpose Aᵀλ of the forward interior operator
        //   A u = κ·(b·∇²u + ∂ₓb·∂ₓu + ∂_z b·∂_z u).
        // L is symmetric and the centred first-difference antisymmetric, so
        //   Aᵀλ = ∇²(b·κ·λ) − ∂ₓ((∂ₓb·κ)·λ) − ∂_z((∂_z b·κ)·λ),
        // computed on the pre-multiplied buffers from build_vrz_adjoint_fields
        // (split into a separate launch to keep the neighbour reads race-free).
        const float* aq0_b = aq0 + b * spatial_size;
        const float* aqx_b = aqx + b * spatial_size;
        const float* aqz_b = aqz + b * spatial_size;
        float lap_x = laplace<2, Order, X>(aq0_b, ix, 0, iz, lap_ctx);
        float lap_z = laplace<2, Order, Z>(aq0_b, ix, 0, iz, lap_ctx);
        float gx = gradient<2, Order, X>(aqx_b, ix, 0, iz, grad_ctx);
        float gz = gradient<2, Order, Z>(aqz_b, ix, 0, iz, grad_ctx);
        float rhs = (lap_x + lap_z) - gx - gz;
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
        return;
    }

    // PML branch: forward formulation retained (small residual error confined
    // to the absorbing zone, mirroring the acoustic proper-adjoint kernel).
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;
    const float* inv_z_b = inv_z + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float dqdx = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dqdz = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float beta = v * inv_z0;
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdz = dvpdz * inv_z0 + v * z1z;
    float kappa = v * z_b[idx];

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float dpsixdx = gradient<2, Order, X>(f.psix, ix, 0, iz, grad_ctx);
    float dpsizdz = gradient<2, Order, Z>(f.psiz, ix, 0, iz, grad_ctx);

    float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);

    float tmpx = ((1.0f + bx_) * lap_x + dbxdx_ * dqdx) + (daxdx * f.psix[idx] + ax_ * dpsixdx);
    float psixn = bx_ * dqdx + ax_ * f.psix[idx];
    float zetaxn = bx_ * tmpx + ax_ * f.zetax[idx];

    float tmpz = ((1.0f + bz_) * lap_z + dbzdz_ * dqdz) + (dazdz * f.psiz[idx] + az_ * dpsizdz);
    float psizn = bz_ * dqdz + az_ * f.psiz[idx];
    float zetazn = bz_ * tmpz + az_ * f.zetaz[idx];

    float qx = dqdx + psixn;
    float qz = dqdz + psizn;
    float w_sum = (1.0f + bx_) * tmpx + ax_ * f.zetax[idx]
                + (1.0f + bz_) * tmpz + az_ * f.zetaz[idx];

    float rhs = kappa * (beta * w_sum + dbdx * qx + dbdz * qz);

    // Race-free adjoint psi double-buffer: dpsi*d* above neighbour-reads psi
    // in this same launch, so the new psi must land in psi*n (caller pairs
    // with swap_pml()).  zeta is only ever read at idx — in-place is safe.
    (f.psixn ? f.psixn : f.psix)[idx] = psixn;
    (f.psizn ? f.psizn : f.psiz)[idx] = psizn;
    f.zetax[idx] = zetaxn;
    f.zetaz[idx] = zetazn;

    f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + solver.dt * solver.dt * rhs;
}

static __global__ void build_kappa_lambda_vrz2d(
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ kappa_lambda,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial_size = solver.nx * solver.nz;
    int idx = b * spatial_size + iz * solver.nx + ix;
    kappa_lambda[idx] = vp[idx] * z[idx] * lambda_now[idx];
}

template<int Order>
__global__ void build_vrz_grad_fields(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    float* __restrict__ c_x,
    float* __restrict__ c_z,
    float* __restrict__ e_x,
    float* __restrict__ e_z,
    GradParam grad_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;
    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int gidx = b * spatial_size + idx;

    const float* p_b = f_u_now + b * spatial_size;
    const float* lam_b = lambda_now + b * spatial_size;
    const float* vp_b = vp + b * spatial_size;
    const float* z_b = z + b * spatial_size;

    float dpdx = gradient<2, Order, X>(p_b, ix, 0, iz, grad_ctx);
    float dpdz = gradient<2, Order, Z>(p_b, ix, 0, iz, grad_ctx);
    float v = vp_b[idx];
    float lam = lam_b[idx];
    float lam_v = lam * v;
    float lam_v2z = lam * v * v * z_b[idx];

    c_x[gidx] = lam_v * dpdx;          // λ·vp·∂ₓp
    c_z[gidx] = lam_v * dpdz;          // λ·vp·∂_z p
    e_x[gidx] = lam_v2z * dpdx;        // λ·vp²·z·∂ₓp
    e_z[gidx] = lam_v2z * dpdz;        // λ·vp²·z·∂_z p
}

// Exact discrete-adjoint gradient (reverse-mode transpose of the forward rhs
//   rhs = vp²·∇²p + vp·(∇vp·∇p) + vp²·z·(∇(1/z)·∇p) ).
// SymPy-verified term-by-term against autograd (test/vrz_sympy_grad_audit.py):
//   grad_vp = 2·vp·λ·∇²p + λ·(∇vp·∇p) - ∇·(λ·vp·∇p) + 2·vp·z·λ·(∇(1/z)·∇p)
//   grad_z  = λ·vp²·(∇(1/z)·∇p) + ∇·(λ·vp²·z·∇p)/z²
// The ∇·(...) divergences read the pre-computed c_d / e_d buffers (so the FD
// gradient stencil — which is the transpose of the forward gradient — can read
// neighbours).  The (-dt²) weight matches the forward u_next += dt²·rhs.
template<int Order>
__global__ void calculate_grad_vrz2d(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ c_x,
    const float* __restrict__ c_z,
    const float* __restrict__ e_x,
    const float* __restrict__ e_z,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_z,
    GradParam grad_ctx,
    LaplaceParam lap_ctx,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int m_static = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : m_static;

    if (ix < halo || ix >= solver.nx - halo || iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int shift = b * spatial_size;

    const float* p_b = f_u_now + shift;
    const float* lambda_b = lambda_now + shift;
    const float* cx_b = c_x + shift;
    const float* cz_b = c_z + shift;
    const float* ex_b = e_x + shift;
    const float* ez_b = e_z + shift;
    const float* vp_b = vp + shift;
    const float* z_b = z + shift;
    const float* inv_z_b = inv_z + shift;
    float* grad_vp_b = grad_vp + shift;
    float* grad_z_b = grad_z + shift;

    float lap_x = laplace<2, Order, X>(p_b, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(p_b, ix, 0, iz, lap_ctx);

    float dpdx = gradient<2, Order, X>(p_b, ix, 0, iz, grad_ctx);
    float dpdz = gradient<2, Order, Z>(p_b, ix, 0, iz, grad_ctx);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);

    // ∇·(c)  and  ∇·(e)  via the FD gradient stencil (transpose of forward grad).
    float div_c = gradient<2, Order, X>(cx_b, ix, 0, iz, grad_ctx)
                + gradient<2, Order, Z>(cz_b, ix, 0, iz, grad_ctx);
    float div_e = gradient<2, Order, X>(ex_b, ix, 0, iz, grad_ctx)
                + gradient<2, Order, Z>(ez_b, ix, 0, iz, grad_ctx);

    float v = vp_b[idx];
    float zz = z_b[idx];
    float inv_z0 = inv_z_b[idx];
    float lam = lambda_b[idx];

    float dp_dvp   = dpdx * dvpdx + dpdz * dvpdz;   // ∇p·∇vp
    float dp_dinvz = dpdx * z1x  + dpdz * z1z;      // ∇p·∇(1/z)

    float g_vp = 2.0f * v * lam * (lap_x + lap_z)
               + lam * dp_dvp
               - div_c
               + 2.0f * v * zz * lam * dp_dinvz;
    float g_z  = lam * v * v * dp_dinvz
               + div_e * inv_z0 * inv_z0;           // ∇·(e)/z²

    float dt2 = solver.dt * solver.dt;
    grad_vp_b[idx] += -dt2 * g_vp;
    grad_z_b[idx]  += -dt2 * g_z;
}
