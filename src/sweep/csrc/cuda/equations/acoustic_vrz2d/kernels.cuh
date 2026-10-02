#pragma once
#include "../../launch/by_order.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../common/acoustic.h"
#include "../../common/context.h"
#include "../../common/acoustic_vrz_fused.cuh"
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

// Defined in kernels.cu.
__global__ void build_kappa_lambda_vrz2d(
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
    const float* __restrict__ p_tt,       // U_{it+1} − 2U_it + U_{it−1}, or nullptr
    const float* __restrict__ p_prev,     // used when p_tt == nullptr
    const float* __restrict__ p_next,
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

#define ACOUSTIC_VRZ2D(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_vrz2nd_by_order, order, grid, block, __VA_ARGS__)

#define ACOUSTIC_VRZ2D_NOPML(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_vrz2nd_nopml_by_order, order, grid, block, __VA_ARGS__)

#define ACOUSTIC_VRZ2D_ADJOINT(order, grid, block, ...)                                      \
    do {                                                                                     \
        if      ((order) == 2) acoustic_vrz2nd_adjoint<2><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 4) acoustic_vrz2nd_adjoint<4><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 6) acoustic_vrz2nd_adjoint<6><<<grid, block>>>(__VA_ARGS__);     \
        else if ((order) == 8) acoustic_vrz2nd_adjoint<8><<<grid, block>>>(__VA_ARGS__);     \
        else                   acoustic_vrz2nd_adjoint<-1><<<grid, block>>>(__VA_ARGS__);    \
    } while (0)

#define CALCULATE_GRAD_VRZ2D(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(calculate_grad_vrz2d_by_order, order, grid, block, __VA_ARGS__)

#define BUILD_VRZ_GRAD_FIELDS(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(build_vrz_grad_fields_by_order, order, grid, block, __VA_ARGS__)

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
    GridBounds solver,
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
    GridBounds solver
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
    GridBounds solver;          // not SolverContext: this is inlined per tap

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
            q, p, ix, iz, sx, sz, grad_ctx, solver.bounds()
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
    GridBounds solver
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
    bool in_pml = solver.in_pml_2d(ix, iz, halo);

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

using acoustic_vrz2nd_fn = void (*)(AcousticWavefieldPointer, bool, float*,
    const float*, const float*, const float*, LaplaceParam, GradParam, GradParam,
    GradParam, AcousticCPMLPointer, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic_vrz2nd_fn, acoustic_vrz2nd_by_order);

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

using acoustic_vrz2nd_nopml_fn = void (*)(AcousticWavefieldPointer, const float*,
    const float*, const float*, LaplaceParam, GradParam, SolverContext);
SWEEP_BY_ORDER_DECL(acoustic_vrz2nd_nopml_fn, acoustic_vrz2nd_nopml_by_order);

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
    bool in_pml = solver.in_pml_2d(ix, iz, halo);

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

using build_vrz_grad_fields_fn = void (*)(const float*, const float*, const float*,
    const float*, float*, float*, float*, float*, GradParam, SolverContext);
SWEEP_BY_ORDER_DECL(build_vrz_grad_fields_fn, build_vrz_grad_fields_by_order);

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
    const float* __restrict__ p_tt,       // U_{it+1} − 2U_it + U_{it−1}, or nullptr
    const float* __restrict__ p_prev,     // used when p_tt == nullptr
    const float* __restrict__ p_next,
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

    // 2·v·λ·∇²p is replaced through the forward identity (same operators as
    // the forward kernel, so exact in the interior up to rounding):
    //   U_{it+1} − 2U_it + U_{it−1} = dt²·κ·(β·∇²p + ∇b·∇p)   (+ S at source cells)
    //   => 2vλ∇²p = 2λ·p_tt/(dt²·v) − 2λ·∇p·∇vp − 2vzλ·∇p·∇(1/z)
    // so grad_vp takes NO second spatial derivative of the stored /
    // reconstructed field (the int8 rim came from exactly that).  The source
    // term is removed by vrz2d_utt_source_correction.  p_tt is either
    // precomputed (full store, slot 5 of u_forward) or formed here from the
    // three time levels (bs, ckpt).
    float ptt = (p_tt != nullptr)
        ? p_tt[shift + idx]
        : (p_next[shift + idx] - 2.0f * p_b[idx] + p_prev[shift + idx]);

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
    float inv_z0 = inv_z_b[idx];
    float lam = lambda_b[idx];

    float dp_dvp   = dpdx * dvpdx + dpdz * dvpdz;   // ∇p·∇vp
    float dp_dinvz = dpdx * z1x  + dpdz * z1z;      // ∇p·∇(1/z)

    float dt2 = solver.dt * solver.dt;
    // grad_vp += −dt²·[2λ·p_tt/(dt²·v) − λ·∇p·∇vp − ∇·(λ·vp·∇p)]
    grad_vp_b[idx] += -2.0f * lam * ptt / v + dt2 * (lam * dp_dvp + div_c);
    float g_z  = lam * v * v * dp_dinvz
               + div_e * inv_z0 * inv_z0;           // ∇·(e)/z²
    grad_z_b[idx]  += -dt2 * g_z;
}

using calculate_grad_vrz2d_fn = void (*)(const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, float*, float*, GradParam, LaplaceParam,
    SolverContext);
SWEEP_BY_ORDER_DECL(calculate_grad_vrz2d_fn, calculate_grad_vrz2d_by_order);

// ===========================================================================
// Fused backward kernels (functor recompute-at-tap, bit-exact with the split
// prepare+apply path above).  Backward per step: build_vrz_adjoint_fields +
// acoustic_vrz2nd_adjoint -> acoustic_vrz2nd_adjoint_fused;  build_vrz_grad_fields
// + calculate_grad_vrz2d -> calculate_grad_vrz2d_fused.  The time-invariant
// adjoint coefficients (vp^2, dx b*kappa, dz b*kappa) are precomputed once by
// build_vrz_adjoint_coeffs before the timestep loop.
// ===========================================================================

// Time-invariant adjoint transpose coefficients (computed once per backward):
//   C0 = vp^2 ,  Cx = (dx b)*kappa ,  Cz = (dz b)*kappa
// so that per step aq0 = C0*lambda, aqx = Cx*lambda, aqz = Cz*lambda exactly.
template<int Order>
__global__ void build_vrz_adjoint_coeffs(
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    float* __restrict__ C0,
    float* __restrict__ Cx,
    float* __restrict__ Cz,
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

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);
    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float kappa = v * z_b[idx];
    float dbdx = dvpdx * inv_z0 + v * z1x;
    float dbdz = dvpdz * inv_z0 + v * z1z;

    C0[gidx] = v * v;            // b*kappa = vp^2
    Cx[gidx] = dbdx * kappa;     // (dx b)*kappa
    Cz[gidx] = dbdz * kappa;     // (dz b)*kappa
}

using build_vrz_adjoint_coeffs_fn = void (*)(const float*, const float*, const float*,
    float*, float*, float*, GradParam, SolverContext);
SWEEP_BY_ORDER_DECL(build_vrz_adjoint_coeffs_fn, build_vrz_adjoint_coeffs_by_order);

template<int Order>
__global__ void acoustic_vrz2nd_adjoint_fused(
    AcousticWavefieldPointer wf,
    const float* __restrict__ vp,
    const float* __restrict__ z,
    const float* __restrict__ inv_z,
    const float* __restrict__ C0,   // vp²          (time-invariant, build_vrz_adjoint_coeffs)
    const float* __restrict__ Cx,   // (∂ₓb)·κ
    const float* __restrict__ Cz,   // (∂_z b)·κ
    LaplaceParam lap_ctx,
    GradParam grad_ctx,
    GradParam grad_ctx_x,
    GradParam grad_ctx_z,
    AcousticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ psix_out,   // adjoint psi/zeta double-buffers (swap_aux)
    float* __restrict__ psiz_out,
    float* __restrict__ zetax_out,
    float* __restrict__ zetaz_out
) {
    // EXACT discrete transpose of acoustic_vrz2nd's CPML step.  Per axis d the
    // forward reads (u, psi_d, zeta_d) and writes
    //   t_d   = (1+b_d)·L_d u + db_d·D_d u + da_d·psi_d + a_d·D_d psi_d
    //   psin_d = b_d·D_d u + a_d·psi_d,   zetan_d = b_d·t_d + a_d·zeta_d
    //   rhs   = κβ·Σ_d[(1+b_d)·t_d + a_d·zeta_d] + Σ_d κ∂_d b·(D_d u + psin_d)
    //         = C0·Σ_d[...] + Σ_d C_d·((1+b_d)·D_d u + a_d·psi_d)
    // with κβ = vp² = C0 and κ∂_d b = C_d.  Transposing operation by operation
    // (Lᵀ = L, D_dᵀ = −D_d, pointwise coefficients evaluated at the TAP) gives,
    // with gw = C0·dt²·λ and gtmp_d = b_d·ζ*_d + (1+b_d)·gw:
    //   λ_next = 2λ − λ_prev + Σ_d [ L_d((1+b_d)·gtmp_d) − D_d(b_d·ψ*_d + db_d·gtmp_d + (1+b_d)·C_d·dt²·λ) ]
    //   ψ*_d'  = a_d·(ψ*_d + C_d·dt²·λ) + da_d·gtmp_d − D_d(a_d·gtmp_d)
    //   ζ*_d'  = a_d·(ζ*_d + gw)
    // Verified against the numerical transpose of the forward step
    // (test/vrz_adjoint_transpose_check.py, rel 1e-16).  Every neighbour read
    // is of the OLD λ/ψ*/ζ*: the new ψ*/ζ* land in the double-buffers.
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
    int oidx = b * spatial_size + idx;

    auto f = wf.offset(b, spatial_size);
    const float* C0_b = C0 + b * spatial_size;
    const float* Cx_b = Cx + b * spatial_size;
    const float* Cz_b = Cz + b * spatial_size;
    const float* lam = f.u_now;
    const int sz = solver.nx;
    const float dt2 = solver.dt * solver.dt;

    // Deep interior (every tap has b = 0, aux = 0): the transpose collapses to
    // ∇²(vp²λ) − ∂ₓ((∂ₓb·κ)·λ) − ∂_z((∂_z b·κ)·λ), the pre-existing exact
    // interior branch, bit-identical.  The adjoint aux there is a dead
    // accumulator (never feeds λ: every coupling carries b), so it is left 0.
    bool pure_interior =
        (ix >= (solver.cut_x_lo() ? halo : solver.padLo(2) + 2 * halo)) &&
        (ix < solver.nx - (solver.cut_x_hi() ? halo : solver.padHi(2) + 2 * halo)) &&
        (iz >= (solver.cut_z_lo() ? halo : solver.padLo(0) + 2 * halo)) &&
        (iz < solver.nz - (solver.cut_z_hi() ? halo : solver.padHi(0) + 2 * halo));
    if (pure_interior) {
        VrzProductAccessor a0x{C0_b, lam, idx, 1};
        VrzProductAccessor a0z{C0_b, lam, idx, solver.nx};
        float lap_x = centered_laplace1d_stencil<Order>(a0x, lap_ctx.dx, solver.M, lap_ctx.coeff);
        float lap_z = centered_laplace1d_stencil<Order>(a0z, lap_ctx.dz, solver.M, lap_ctx.coeff);

        VrzProductAccessor axx{Cx_b, lam, idx, 1};
        VrzProductAccessor azz{Cz_b, lam, idx, solver.nx};
        float gx = centered_gradient_stencil<Order>(axx, solver.M, grad_ctx.coeff, grad_ctx.dx);
        float gz = centered_gradient_stencil<Order>(azz, solver.M, grad_ctx.coeff, grad_ctx.dz);

        float rhs = (lap_x + lap_z) - gx - gz;
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] + dt2 * rhs;
        return;
    }

    const float* lc = lap_ctx.coeff;    // [c0(centre), c1..cM]
    const float* gc = grad_ctx.coeff;   // [_, c1..cM] (antisymmetric)
    const float invdx2 = 1.0f / (lap_ctx.dx * lap_ctx.dx);
    const float invdz2 = 1.0f / (lap_ctx.dz * lap_ctx.dz);
    const float invdx = 1.0f / grad_ctx.dx;
    const float invdz = 1.0f / grad_ctx.dz;

    // The forward runs its CPML branch only where in_pml_2d(cell) holds; every
    // other cell takes the interior formula and never touches aux.  The
    // transpose must mirror that per TAP: a coefficient enters a tap's term
    // only if the forward evaluated it there.  b is exactly 0 outside the band
    // by construction (sigma = 0), but db = np.gradient(b) leaks one cell past
    // the band edge and a = exp(-alpha dt) is not zero in the interior, so
    // both are gated.  A tap on the x axis shares this row's z answer (and
    // vice versa), so the 2-D predicate costs one compare per tap.
    const bool px0 = solver.in_pml_x(ix, halo);
    const bool pz0 = solver.in_pml_z(iz, halo);
    const bool own_pml = px0 || pz0;

    const float ax_ = cpml.ax[ix], az_ = cpml.az[iz];
    const float bx_ = cpml.bx[ix], bz_ = cpml.bz[iz];

    const float gw0 = C0_b[idx] * dt2 * lam[idx];
    const float gtmpx0 = bx_ * f.zetax[idx] + (1.0f + bx_) * gw0;
    const float gtmpz0 = bz_ * f.zetaz[idx] + (1.0f + bz_) * gw0;

    #define VRZ_GW(n)          ( C0_b[n] * dt2 * lam[n] )
    #define VRZ_GTMP_X(n, nix) ( cpml.bx[nix] * f.zetax[n] + (1.0f + cpml.bx[nix]) * VRZ_GW(n) )
    #define VRZ_GTMP_Z(n, niz) ( cpml.bz[niz] * f.zetaz[n] + (1.0f + cpml.bz[niz]) * VRZ_GW(n) )

    // X: L((1+b)·gtmp), D(b·ψ* + db·gtmp + (1+b)·Cx·dt²·λ), D(a·gtmp)
    float Lx_gl = -lc[0] * ((1.0f + bx_) * gtmpx0);
    float Dx_gg = 0.0f, Dx_gq = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k, nm = idx - k, nixp = ix + k, nixm = ix - k;
        bool Pp = pz0 || solver.in_pml_x(nixp, halo);
        bool Pm = pz0 || solver.in_pml_x(nixm, halo);
        float tp = VRZ_GTMP_X(np, nixp), tm = VRZ_GTMP_X(nm, nixm);
        float bxp = cpml.bx[nixp], bxm = cpml.bx[nixm];
        float dbp = Pp ? cpml.dbxdx[nixp] : 0.0f, dbm = Pm ? cpml.dbxdx[nixm] : 0.0f;
        float ap  = Pp ? cpml.ax[nixp]    : 0.0f, am  = Pm ? cpml.ax[nixm]    : 0.0f;
        Lx_gl += lc[k] * ((1.0f + bxp) * tp + (1.0f + bxm) * tm);
        Dx_gg += gc[k] * ((bxp * f.psix[np] + dbp * tp + (1.0f + bxp) * Cx_b[np] * dt2 * lam[np])
                        - (bxm * f.psix[nm] + dbm * tm + (1.0f + bxm) * Cx_b[nm] * dt2 * lam[nm]));
        Dx_gq += gc[k] * (ap * tp - am * tm);
    }
    Lx_gl *= invdx2; Dx_gg *= invdx; Dx_gq *= invdx;

    // Z
    float Lz_gl = -lc[0] * ((1.0f + bz_) * gtmpz0);
    float Dz_gg = 0.0f, Dz_gq = 0.0f;
    #pragma unroll
    for (int k = 1; k <= halo; ++k) {
        int np = idx + k * sz, nm = idx - k * sz, nizp = iz + k, nizm = iz - k;
        bool Pp = px0 || solver.in_pml_z(nizp, halo);
        bool Pm = px0 || solver.in_pml_z(nizm, halo);
        float tp = VRZ_GTMP_Z(np, nizp), tm = VRZ_GTMP_Z(nm, nizm);
        float bzp = cpml.bz[nizp], bzm = cpml.bz[nizm];
        float dbp = Pp ? cpml.dbzdz[nizp] : 0.0f, dbm = Pm ? cpml.dbzdz[nizm] : 0.0f;
        float ap  = Pp ? cpml.az[nizp]    : 0.0f, am  = Pm ? cpml.az[nizm]    : 0.0f;
        Lz_gl += lc[k] * ((1.0f + bzp) * tp + (1.0f + bzm) * tm);
        Dz_gg += gc[k] * ((bzp * f.psiz[np] + dbp * tp + (1.0f + bzp) * Cz_b[np] * dt2 * lam[np])
                        - (bzm * f.psiz[nm] + dbm * tm + (1.0f + bzm) * Cz_b[nm] * dt2 * lam[nm]));
        Dz_gq += gc[k] * (ap * tp - am * tm);
    }
    Lz_gl *= invdz2; Dz_gg *= invdz; Dz_gq *= invdz;

    f.u_next[idx] = 2.0f * lam[idx] - f.u_prev[idx] + (Lx_gl + Lz_gl - Dx_gg - Dz_gg);

    // Adjoint aux is alive only where the forward ran its CPML branch: a
    // cell outside the band never wrote psi/zeta, so nothing there reads
    // them back (every coupling carries b, which is 0 there).  Its
    // double-buffers were zeroed at allocation and no other cell writes them,
    // so skipping the writes keeps them exactly 0 -- same as pure interior.
    if (!own_pml) return;

    // da_d = D_d a_d, exactly as the forward computes it (product-rule form)
    const float daxdx = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    const float dazdz = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
    psix_out[oidx]  = ax_ * (f.psix[idx] + Cx_b[idx] * dt2 * lam[idx]) + daxdx * gtmpx0 - Dx_gq;
    psiz_out[oidx]  = az_ * (f.psiz[idx] + Cz_b[idx] * dt2 * lam[idx]) + dazdz * gtmpz0 - Dz_gq;
    zetax_out[oidx] = ax_ * (f.zetax[idx] + gw0);
    zetaz_out[oidx] = az_ * (f.zetaz[idx] + gw0);
    #undef VRZ_GW
    #undef VRZ_GTMP_X
    #undef VRZ_GTMP_Z
}

using acoustic_vrz2nd_adjoint_fused_fn = void (*)(AcousticWavefieldPointer,
    const float*, const float*, const float*, const float*, const float*, const float*,
    LaplaceParam, GradParam, GradParam, GradParam, AcousticCPMLPointer, SolverContext,
    float*, float*, float*, float*);
SWEEP_BY_ORDER_DECL(acoustic_vrz2nd_adjoint_fused_fn, acoustic_vrz2nd_adjoint_fused_by_order);

// Accessor recomputing c_d = λ·vp·∂_d p  (UseE=false) or e_d = λ·vp²·z·∂_d p
// (UseE=true) at stencil tap `off` along Dir, matching build_vrz_grad_fields.
template<int Order, int Dir, bool UseE>
struct VrzGradAccessor2D {
    const float* __restrict__ p;
    const float* __restrict__ lam;
    const float* __restrict__ vp;
    const float* __restrict__ z;
    int ix, iz, nx, nz, halo;
    GradParam grad_ctx;

    __device__ __forceinline__ float operator()(int off) const
    {
        int jx = ix;
        int jz = iz;
        if constexpr (Dir & X) jx = ix + off; else jz = iz + off;
        if (jx < halo || jx >= nx - halo || jz < halo || jz >= nz - halo)
            return 0.f;
        int j = jz * nx + jx;
        float dpd = gradient<2, Order, Dir>(p, jx, 0, jz, grad_ctx);
        float lam_j = lam[j];
        float v_j = vp[j];
        if constexpr (UseE) {
            float lam_v2z = ((lam_j * v_j) * v_j) * z[j];   // λ·vp²·z
            return lam_v2z * dpd;
        } else {
            float lam_v = lam_j * v_j;                       // λ·vp
            return lam_v * dpd;
        }
    }
};

template<int Order>
__global__ void calculate_grad_vrz2d_fused(
    const float* __restrict__ f_u_now,
    const float* __restrict__ lambda_now,
    const float* __restrict__ p_tt,       // U_{it+1} − 2U_it + U_{it−1}, or nullptr
    const float* __restrict__ p_prev,     // used when p_tt == nullptr
    const float* __restrict__ p_next,
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
    // Image the PHYSICAL box only.  ``EdgePadding.backward`` crops the pad
    // gradients on the way back to the model, so everything this kernel writes
    // in the PML shell and the stencil halo is discarded -- it was read,
    // multiplied and accumulated for nothing, once per time step.  The bounds
    // come from the context and are CUT-AWARE: on a DD cut face phys_x0()
    // collapses to the halo, so the cut-adjacent columns are still imaged.
    if (ix < solver.phys_x0() || ix >= solver.phys_x1() ||
        iz < solver.phys_z0() || iz >= solver.phys_z1()) return;

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
    const float* vp_b = vp + shift;
    const float* z_b = z + shift;
    const float* inv_z_b = inv_z + shift;
    float* grad_vp_b = grad_vp + shift;
    float* grad_z_b = grad_z + shift;

    // 2·v·λ·∇²p is replaced through the forward identity (same operators as
    // the forward kernel, so exact in the interior up to rounding):
    //   U_{it+1} − 2U_it + U_{it−1} = dt²·κ·(β·∇²p + ∇b·∇p)   (+ S at source cells)
    //   => 2vλ∇²p = 2λ·p_tt/(dt²·v) − 2λ·∇p·∇vp − 2vzλ·∇p·∇(1/z)
    // so grad_vp takes NO second spatial derivative of the stored /
    // reconstructed field (the int8 rim came from exactly that).  The source
    // term is removed by vrz2d_utt_source_correction.  p_tt is either
    // precomputed (full store, slot 5 of u_forward) or formed here from the
    // three time levels (bs, ckpt).
    float ptt = (p_tt != nullptr)
        ? p_tt[shift + idx]
        : (p_next[shift + idx] - 2.0f * p_b[idx] + p_prev[shift + idx]);

    float dpdx = gradient<2, Order, X>(p_b, ix, 0, iz, grad_ctx);
    float dpdz = gradient<2, Order, Z>(p_b, ix, 0, iz, grad_ctx);

    float dvpdx = gradient<2, Order, X>(vp_b, ix, 0, iz, grad_ctx);
    float dvpdz = gradient<2, Order, Z>(vp_b, ix, 0, iz, grad_ctx);
    float z1x = gradient<2, Order, X>(inv_z_b, ix, 0, iz, grad_ctx);
    float z1z = gradient<2, Order, Z>(inv_z_b, ix, 0, iz, grad_ctx);

    // ∇·(c) and ∇·(e) recomputed tap-by-tap (transpose of forward gradient).
    VrzGradAccessor2D<Order, X, false> cx{p_b, lambda_b, vp_b, z_b, ix, iz, solver.nx, solver.nz, halo, grad_ctx};
    VrzGradAccessor2D<Order, Z, false> cz{p_b, lambda_b, vp_b, z_b, ix, iz, solver.nx, solver.nz, halo, grad_ctx};
    VrzGradAccessor2D<Order, X, true>  ex{p_b, lambda_b, vp_b, z_b, ix, iz, solver.nx, solver.nz, halo, grad_ctx};
    VrzGradAccessor2D<Order, Z, true>  ez{p_b, lambda_b, vp_b, z_b, ix, iz, solver.nx, solver.nz, halo, grad_ctx};
    float div_c = centered_gradient_stencil<Order>(cx, solver.M, grad_ctx.coeff, grad_ctx.dx)
                + centered_gradient_stencil<Order>(cz, solver.M, grad_ctx.coeff, grad_ctx.dz);
    float div_e = centered_gradient_stencil<Order>(ex, solver.M, grad_ctx.coeff, grad_ctx.dx)
                + centered_gradient_stencil<Order>(ez, solver.M, grad_ctx.coeff, grad_ctx.dz);

    float v = vp_b[idx];
    float inv_z0 = inv_z_b[idx];
    float lam = lambda_b[idx];

    float dp_dvp   = dpdx * dvpdx + dpdz * dvpdz;
    float dp_dinvz = dpdx * z1x  + dpdz * z1z;

    float dt2 = solver.dt * solver.dt;
    // grad_vp += −dt²·[2λ·p_tt/(dt²·v) − λ·∇p·∇vp − ∇·(λ·vp·∇p)]
    grad_vp_b[idx] += -2.0f * lam * ptt / v + dt2 * (lam * dp_dvp + div_c);
    float g_z  = lam * v * v * dp_dinvz
               + div_e * inv_z0 * inv_z0;
    grad_z_b[idx]  += -dt2 * g_z;
}

using calculate_grad_vrz2d_fused_fn = void (*)(const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, float*,
    float*, GradParam, LaplaceParam, SolverContext);
SWEEP_BY_ORDER_DECL(calculate_grad_vrz2d_fused_fn, calculate_grad_vrz2d_fused_by_order);

// The p_tt imaging uses U_{it+1} − 2U_it + U_{it−1}, which at a source cell
// is dt²·rhs + S_it (add_source adds the wavelet to u_next after the step,
// unscaled).  The exact adjoint wants dt²·rhs alone: add back the 2·λ·S/v the
// imaging subtracted.  Same indexing as add_source.
// (Defined in kernels.cu: a definition here is compiled into every TU that
// includes this header.)
__global__ void vrz2d_utt_source_correction(
    float* __restrict__ grad_vp,
    const float* __restrict__ lambda_now,
    const float* __restrict__ vp,
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
);

#define BUILD_VRZ_ADJOINT_COEFFS(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(build_vrz_adjoint_coeffs_by_order, order, grid, block, __VA_ARGS__)

#define ACOUSTIC_VRZ2D_ADJOINT_FUSED(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(acoustic_vrz2nd_adjoint_fused_by_order, order, grid, block, __VA_ARGS__)

#define CALCULATE_GRAD_VRZ2D_FUSED(order, grid, block, ...) \
    SWEEP_LAUNCH_BY_ORDER(calculate_grad_vrz2d_fused_by_order, order, grid, block, __VA_ARGS__)

// Gradient dispatch: the fused functor kernel recomputes ∂p at every stencil tap
// (O(M²) per point), which beats the buffered split path only at low order.
// Measured on RTX 6000 Ada: order 4 ~1.3–2.0x faster, order 8 ~0.8x (slower).
// So fuse for order<=4 and fall back to the split (build+calculate) for order>=6.
#define CALCULATE_GRAD_VRZ2D_AUTO(order, grid, block, P, LAM, PTT, PPREV, PNEXT, VP, Z, INVZ, CX, CZ, EX, EZ, GVP, GZ, GCTX, LCTX, SCTX) \
    do {                                                                                      \
        if ((order) == 2 || (order) == 4) {                                                   \
            CALCULATE_GRAD_VRZ2D_FUSED((order), grid, block,                                  \
                P, LAM, PTT, PPREV, PNEXT, VP, Z, INVZ, GVP, GZ, GCTX, LCTX, SCTX);           \
        } else {                                                                              \
            BUILD_VRZ_GRAD_FIELDS((order), grid, block,                                       \
                P, LAM, VP, Z, CX, CZ, EX, EZ, GCTX, SCTX);                                   \
            CALCULATE_GRAD_VRZ2D((order), grid, block,                                        \
                P, LAM, PTT, PPREV, PNEXT, CX, CZ, EX, EZ, VP, Z, INVZ, GVP, GZ, GCTX, LCTX, SCTX); \
        }                                                                                     \
    } while (0)
