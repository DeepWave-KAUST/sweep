#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../operators/staggered.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"   // reuses ElasticCPMLPointer (8 vals in 2D)


namespace acoustic_vti_1st_2d {

// ---------------------------------------------------------------------------
// Wavefield pointer view: 4 physical fields + 4 CPML memory variables
// Layout in Python field order (matches AcousticVTI1st.FIELD_SPECS):
//     [vx, vz, sH, sV, m_sHx, m_sVz, m_vxx, m_vzz]
// ---------------------------------------------------------------------------

struct VTIWavefieldPointer {
    float* __restrict__ vx;
    float* __restrict__ vz;
    float* __restrict__ sH;
    float* __restrict__ sV;
    float* __restrict__ m_sHx;
    float* __restrict__ m_sVz;
    float* __restrict__ m_vxx;
    float* __restrict__ m_vzz;

    __device__ VTIWavefieldPointer offset(int b, int spatial_size) const
    {
        VTIWavefieldPointer out = *this;
        const int shift = b * spatial_size;
        out.vx    += shift;
        out.vz    += shift;
        out.sH    += shift;
        out.sV    += shift;
        out.m_sHx += shift;
        out.m_sVz += shift;
        out.m_vxx += shift;
        out.m_vzz += shift;
        return out;
    }
};


// ---------------------------------------------------------------------------
// LAUNCH macros (compile-time order dispatch, same pattern as elastic2d)
// ---------------------------------------------------------------------------

#define LAUNCH_VTI_VELOCITY(order, grid, block, ...)                            \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_2d::velocity_kernel<2>          \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_2d::velocity_kernel<4>          \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_2d::velocity_kernel<6>          \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_2d::velocity_kernel<8>          \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_2d::velocity_kernel<-1>         \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


#define LAUNCH_VTI_STRESS(order, grid, block, ...)                              \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_2d::stress_kernel<2>            \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_2d::stress_kernel<4>            \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_2d::stress_kernel<6>            \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_2d::stress_kernel<8>            \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_2d::stress_kernel<-1>           \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


// ---------------------------------------------------------------------------
// Velocity update kernel
//
//   ∂v_x/∂t = (1/ρ) ∂σ_H/∂x          (∂σ_H/∂x via forward staggered in x)
//   ∂v_z/∂t = (1/ρ) ∂σ_V/∂z          (∂σ_V/∂z via forward staggered in z)
//
// CPML: half-step coefficients (axh/bxh, azh/bzh).
// Reference: Duveneck et al. (2008), Eq (2).
// ---------------------------------------------------------------------------
template<int Order>
__global__ void velocity_kernel(
    VTIWavefieldPointer wf,
    const float* __restrict__ inv_rho,
    SGradParam grad_ctx,
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
    const float* inv_rho_b = inv_rho + b * spatial_size;

    // Stress spatial derivatives at velocity locations.
    //   ∂σ_H/∂x  : center→x-face  → forward staggered
    //   ∂σ_V/∂z  : center→z-face  → forward staggered
    float dsH_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.sH, ix, 0, iz, grad_ctx);
    float dsV_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.sV, ix, 0, iz, grad_ctx);

    float inv_rho_val = inv_rho_b[idx];

    // PML / interior split — outside the PML the half-step coefficients
    // axh/bxh/azh/bzh vanish so the memory variables remain zero.
    bool in_pml =
        (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.vx[idx] += solver.dt * inv_rho_val * dsH_dx;
        f.vz[idx] += solver.dt * inv_rho_val * dsV_dz;
        return;
    }

    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    f.m_sHx[idx] = axh * f.m_sHx[idx] + bxh * dsH_dx;
    dsH_dx     += f.m_sHx[idx];

    f.m_sVz[idx] = azh * f.m_sVz[idx] + bzh * dsV_dz;
    dsV_dz     += f.m_sVz[idx];

    f.vx[idx] += solver.dt * inv_rho_val * dsH_dx;
    f.vz[idx] += solver.dt * inv_rho_val * dsV_dz;
}


// ---------------------------------------------------------------------------
// Stress update kernel
//
//   ∂σ_H/∂t = c11 ∂v_x/∂x + c13 ∂v_z/∂z
//   ∂σ_V/∂t = c13 ∂v_x/∂x + c33 ∂v_z/∂z
//
// where c11 = ρ V_P² (1+2ε), c33 = ρ V_P², c13 = ρ V_P² √(1+2δ).
// CPML: cell-centre coefficients (ax/bx, az/bz).
// Reference: Duveneck et al. (2008), Eq (1)–(2).
// ---------------------------------------------------------------------------
template<int Order>
__global__ void stress_kernel(
    VTIWavefieldPointer wf,
    const float* __restrict__ c11,
    const float* __restrict__ c33,
    const float* __restrict__ c13,
    SGradParam grad_ctx,
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
    const float* c11_b = c11 + b * spatial_size;
    const float* c33_b = c33 + b * spatial_size;
    const float* c13_b = c13 + b * spatial_size;

    // Velocity spatial derivatives at stress locations (cell centre):
    //   ∂v_x/∂x  : x-face→center → backward staggered
    //   ∂v_z/∂z  : z-face→center → backward staggered
    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dvz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(f.vz, ix, 0, iz, grad_ctx);

    float C11 = c11_b[idx];
    float C33 = c33_b[idx];
    float C13 = c13_b[idx];

    bool in_pml =
        (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.sH[idx] += solver.dt * (C11 * dvx_dx + C13 * dvz_dz);
        f.sV[idx] += solver.dt * (C13 * dvx_dx + C33 * dvz_dz);
        return;
    }

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float az = cpml.az[iz];
    float bz = cpml.bz[iz];

    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx     += f.m_vxx[idx];

    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz     += f.m_vzz[idx];

    f.sH[idx] += solver.dt * (C11 * dvx_dx + C13 * dvz_dz);
    f.sV[idx] += solver.dt * (C13 * dvx_dx + C33 * dvz_dz);
}


// ===========================================================================
//   NO-PML kernels — used by backward_bs to **time-reverse** the saved
//   forward state inside the interior.  Same equations as the PML kernels
//   but:
//   (a) the CPML memory-variable fast-path is removed (the boundary cells
//       in the PML band are restored from the saver after each step), so
//       the m_* fields are never read;
//   (b) `+= dt * ...` becomes `-= dt * ...` to step the forward state
//       backward in time by one Δt.
// ===========================================================================

template<int Order>
__global__ void velocity_kernel_nopml(
    VTIWavefieldPointer wf,
    const float* __restrict__ inv_rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int M;
    if constexpr (Order == -1) { M = solver.M; } else { M = Order / 2; }
    int halo = solver.abcn + M + 1;
    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* inv_rho_b = inv_rho + b * spatial_size;

    float dsH_dx = sgradient<2, Order, X, DIFF_FORWARD>(f.sH, ix, 0, iz, grad_ctx);
    float dsV_dz = sgradient<2, Order, Z, DIFF_FORWARD>(f.sV, ix, 0, iz, grad_ctx);

    float ir = inv_rho_b[idx];
    // Time-reverse: subtract one forward step.
    f.vx[idx] -= solver.dt * ir * dsH_dx;
    f.vz[idx] -= solver.dt * ir * dsV_dz;
}

template<int Order>
__global__ void stress_kernel_nopml(
    VTIWavefieldPointer wf,
    const float* __restrict__ c11,
    const float* __restrict__ c33,
    const float* __restrict__ c13,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int M;
    if constexpr (Order == -1) { M = solver.M; } else { M = Order / 2; }
    int halo = solver.abcn + M + 1;
    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* c11_b = c11 + b * spatial_size;
    const float* c33_b = c33 + b * spatial_size;
    const float* c13_b = c13 + b * spatial_size;

    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float dvz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(f.vz, ix, 0, iz, grad_ctx);

    float C11 = c11_b[idx];
    float C33 = c33_b[idx];
    float C13 = c13_b[idx];

    // Time-reverse: subtract one forward step.
    f.sH[idx] -= solver.dt * (C11 * dvx_dx + C13 * dvz_dz);
    f.sV[idx] -= solver.dt * (C13 * dvx_dx + C33 * dvz_dz);
}


#define LAUNCH_VTI_VELOCITY_NOPML(order, grid, block, ...)                      \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_2d::velocity_kernel_nopml<2>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_2d::velocity_kernel_nopml<4>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_2d::velocity_kernel_nopml<6>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_2d::velocity_kernel_nopml<8>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_2d::velocity_kernel_nopml<-1>   \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)

#define LAUNCH_VTI_STRESS_NOPML(order, grid, block, ...)                        \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_2d::stress_kernel_nopml<2>      \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_2d::stress_kernel_nopml<4>      \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_2d::stress_kernel_nopml<6>      \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_2d::stress_kernel_nopml<8>      \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_2d::stress_kernel_nopml<-1>     \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


// ===========================================================================
//                     ADJOINT (BACKWARD) KERNELS — Phase 1
// ---------------------------------------------------------------------------
// Implements the discrete adjoint of the Duveneck (2008) VTI step.  Phase 1
// scope: full mode only (uses saved forward wavefield); adjoint PML memory
// variables are NOT tracked — interior gradients are correct, gradients
// inside the PML band are approximate.  FWI / RTM typically masks the PML
// region in the gradient, so this is fine for production use.
//
// Derivation summary
// ------------------
// Forward step (one leapfrog cycle):
//   vx += dt * inv_rho * Df_x(sH)
//   vz += dt * inv_rho * Df_z(sV)
//   sH += dt * (c11 * Db_x(vx) + c13 * Db_z(vz))     // uses NEW vx, vz
//   sV += dt * (c13 * Db_x(vx) + c33 * Db_z(vz))
//
// Adjoint state propagation (going backward in time, accumulate "+="):
//   λvx += -dt * c11 * Df_x(λsH) - dt * c13 * Df_x(λsV)   // from stress update
//   λvz += -dt * c13 * Df_z(λsH) - dt * c33 * Df_z(λsV)
//   λsH += -dt * inv_rho * Db_x(λvx)                       // from velocity update
//   λsV += -dt * inv_rho * Db_z(λvz)
//
// Material-parameter gradients per time step:
//   grad_c11    += dt * λsH * Db_x(vx_fwd)
//   grad_c13    += dt * (λsH * Db_z(vz_fwd) + λsV * Db_x(vx_fwd))
//   grad_c33    += dt * λsV * Db_z(vz_fwd)
//   grad_irho   += dt * (λvx * Df_x(sH_fwd) + λvz * Df_z(sV_fwd))
//
// Chain rule to (vp, ε, δ, ρ) is applied in calculate_grad_kernel.
// ===========================================================================


// ---------------------------------------------------------------------------
// Adjoint propagation — split into TWO kernels to avoid the read/write race
// hazard.  In kernel A we read λ_sH/λ_sV and write to λ_vx/λ_vz; in kernel B
// we read the *new* λ_vx/λ_vz and write to λ_sH/λ_sV.  Splitting forces a
// global sync between the two halves.
//
// Order matches the *reverse* of the forward step:
//   forward = velocity-update → stress-update
//   adjoint = adjoint(stress-update) → adjoint(velocity-update)
// ---------------------------------------------------------------------------

// Kernel A: λ_sH/λ_sV → contribute to λ_vx/λ_vz  (adjoint of stress update)
//   λvx += -dt * (c11 * Df_x(λsH) + c13 * Df_x(λsV))
//   λvz += -dt * (c13 * Df_z(λsH) + c33 * Df_z(λsV))
template<int Order>
__global__ void adjoint_stress_to_vel_kernel(
    VTIWavefieldPointer adj,
    const float* __restrict__ c11,
    const float* __restrict__ c33,
    const float* __restrict__ c13,
    SGradParam grad_ctx,
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

    auto a = adj.offset(b, spatial_size);
    const float* c11_b = c11 + b * spatial_size;
    const float* c33_b = c33 + b * spatial_size;
    const float* c13_b = c13 + b * spatial_size;

    float dlSH_dx = sgradient<2, Order, X, DIFF_FORWARD>(a.sH, ix, 0, iz, grad_ctx);
    float dlSV_dx = sgradient<2, Order, X, DIFF_FORWARD>(a.sV, ix, 0, iz, grad_ctx);
    float dlSH_dz = sgradient<2, Order, Z, DIFF_FORWARD>(a.sH, ix, 0, iz, grad_ctx);
    float dlSV_dz = sgradient<2, Order, Z, DIFF_FORWARD>(a.sV, ix, 0, iz, grad_ctx);

    a.vx[idx] += -solver.dt * (c11_b[idx] * dlSH_dx + c13_b[idx] * dlSV_dx);
    a.vz[idx] += -solver.dt * (c13_b[idx] * dlSH_dz + c33_b[idx] * dlSV_dz);
}


// Kernel B: λ_vx/λ_vz → contribute to λ_sH/λ_sV  (adjoint of velocity update)
//   λsH += -dt * inv_rho * Db_x(λvx)
//   λsV += -dt * inv_rho * Db_z(λvz)
template<int Order>
__global__ void adjoint_vel_to_stress_kernel(
    VTIWavefieldPointer adj,
    const float* __restrict__ inv_rho,
    SGradParam grad_ctx,
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

    auto a = adj.offset(b, spatial_size);
    const float* invr_b = inv_rho + b * spatial_size;

    float dlvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(a.vx, ix, 0, iz, grad_ctx);
    float dlvz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(a.vz, ix, 0, iz, grad_ctx);

    a.sH[idx] += -solver.dt * invr_b[idx] * dlvx_dx;
    a.sV[idx] += -solver.dt * invr_b[idx] * dlvz_dz;
}


// ---------------------------------------------------------------------------
// Gradient kernel — accumulates ∂L/∂vp, ∂L/∂ε, ∂L/∂δ, ∂L/∂ρ once per step.
//
// At backward iter `it` we process forward step `it` (which transformed
// state_{it-1} → state_it = u_forward[it]).  The chain rule uses:
//
//   * Stiffness gradients (c11, c33, c13): need vx_new, vz_new used inside
//     the stress sub-step of step it.  Those equal u_forward[it].{vx,vz}
//     (the stress sub-step doesn't change velocities).
//   * Inv-rho gradient: needs sH_old, sV_old used inside the velocity
//     sub-step of step it.  Those equal u_forward[it-1].{sH,sV} (the state
//     at the START of step it).
//
// For iter `it = 0`, u_forward[-1] does not exist — the caller should pass
// a zero buffer; this gives the correct gradient (initial state is zero).
//
// Inputs:
//   fvx_now, fvz_now      : u_forward[it].{vx, vz}    (for stiffness grad)
//   fsH_prev, fsV_prev    : u_forward[it-1].{sH, sV}  (for inv_rho grad)
//   adjoint               : current adjoint state (λ at this time slot)
//   vp, ε, δ, ρ           : models                                 (B, nz, nx)
// Outputs (accumulate into):
//   grad_vp, grad_eps, grad_delta, grad_rho                         (B, nz, nx)
// ---------------------------------------------------------------------------
template<int Order>
__global__ void calculate_grad_kernel(
    const float* __restrict__ fvx_now,
    const float* __restrict__ fvz_now,
    const float* __restrict__ fsH_prev,
    const float* __restrict__ fsV_prev,
    VTIWavefieldPointer adj,
    const float* __restrict__ vp,
    const float* __restrict__ epsilon,
    const float* __restrict__ delta,
    const float* __restrict__ rho,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_eps,
    float* __restrict__ grad_delta,
    float* __restrict__ grad_rho,
    SGradParam grad_ctx,
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

    const float* fvx_b = fvx_now  + b * spatial_size;
    const float* fvz_b = fvz_now  + b * spatial_size;
    const float* fsH_b = fsH_prev + b * spatial_size;
    const float* fsV_b = fsV_prev + b * spatial_size;
    auto a = adj.offset(b, spatial_size);

    const float* vp_b   = vp      + b * spatial_size;
    const float* eps_b  = epsilon + b * spatial_size;
    const float* dlt_b  = delta   + b * spatial_size;
    const float* rho_b  = rho     + b * spatial_size;
    float* gvp_b        = grad_vp     + b * spatial_size;
    float* geps_b       = grad_eps    + b * spatial_size;
    float* gdlt_b       = grad_delta  + b * spatial_size;
    float* grho_b       = grad_rho    + b * spatial_size;

    // Forward spatial derivatives at this point (NO PML — interior only).
    // dvx_dx, dvz_dz : at the "stress" location, use vx,vz AT END of step
    //                   (= u_forward[it])
    // dsH_dx, dsV_dz : at the "velocity" location, use sH,sV AT START of step
    //                   (= u_forward[it-1])
    float dvx_dx = sgradient<2, Order, X, DIFF_BACKWARD>(fvx_b, ix, 0, iz, grad_ctx);
    float dvz_dz = sgradient<2, Order, Z, DIFF_BACKWARD>(fvz_b, ix, 0, iz, grad_ctx);
    float dsH_dx = sgradient<2, Order, X, DIFF_FORWARD> (fsH_b, ix, 0, iz, grad_ctx);
    float dsV_dz = sgradient<2, Order, Z, DIFF_FORWARD> (fsV_b, ix, 0, iz, grad_ctx);

    float lSH = a.sH[idx];
    float lSV = a.sV[idx];
    float lvx = a.vx[idx];
    float lvz = a.vz[idx];

    // Stiffness-coefficient gradients (chain rule input)
    float dt = solver.dt;
    float grad_c11    = dt *  lSH * dvx_dx;
    float grad_c33    = dt *  lSV * dvz_dz;
    float grad_c13    = dt * (lSH * dvz_dz + lSV * dvx_dx);
    float grad_irho   = dt * (lvx * dsH_dx + lvz * dsV_dz);

    // Material values
    float vp_v   = vp_b[idx];
    float eps_v  = eps_b[idx];
    float dlt_v  = dlt_b[idx];
    float rho_v  = rho_b[idx];

    float one_p_2eps = 1.f + 2.f * eps_v;
    float one_p_2dlt = 1.f + 2.f * dlt_v;
    float sqrt_d     = sqrtf(one_p_2dlt);
    float vp2        = vp_v * vp_v;

    // Chain rule:
    //   c11 = ρ vp² (1+2ε)        c33 = ρ vp²        c13 = ρ vp² √(1+2δ)
    //   inv_rho = 1/ρ
    //
    //   ∂c11/∂vp = 2 ρ vp (1+2ε)        ∂c11/∂ε = 2 ρ vp²        ∂c11/∂ρ = vp² (1+2ε)
    //   ∂c33/∂vp = 2 ρ vp              ∂c33/∂ρ = vp²
    //   ∂c13/∂vp = 2 ρ vp √(1+2δ)      ∂c13/∂δ = ρ vp² / √(1+2δ)  ∂c13/∂ρ = vp² √(1+2δ)
    //   ∂(1/ρ)/∂ρ = -1/ρ²
    float common_vp  = grad_c11 * one_p_2eps + grad_c33 + grad_c13 * sqrt_d;
    gvp_b[idx]  += 2.f * rho_v * vp_v * common_vp;
    geps_b[idx] += grad_c11 * 2.f * rho_v * vp2;
    gdlt_b[idx] += grad_c13 * rho_v * vp2 / fmaxf(sqrt_d, 1e-12f);
    grho_b[idx] += vp2 * common_vp - grad_irho / (rho_v * rho_v);
}


#define LAUNCH_VTI_ADJOINT_STRESS_TO_VEL(order, grid, block, ...)               \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_2d::adjoint_stress_to_vel_kernel<2>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_2d::adjoint_stress_to_vel_kernel<4>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_2d::adjoint_stress_to_vel_kernel<6>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_2d::adjoint_stress_to_vel_kernel<8>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_2d::adjoint_stress_to_vel_kernel<-1> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)

#define LAUNCH_VTI_ADJOINT_VEL_TO_STRESS(order, grid, block, ...)               \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_2d::adjoint_vel_to_stress_kernel<2>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_2d::adjoint_vel_to_stress_kernel<4>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_2d::adjoint_vel_to_stress_kernel<6>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_2d::adjoint_vel_to_stress_kernel<8>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_2d::adjoint_vel_to_stress_kernel<-1> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)

#define LAUNCH_VTI_CALC_GRAD(order, grid, block, ...)                           \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_2d::calculate_grad_kernel<2>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_2d::calculate_grad_kernel<4>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_2d::calculate_grad_kernel<6>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_2d::calculate_grad_kernel<8>    \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_2d::calculate_grad_kernel<-1>   \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


// ---------------------------------------------------------------------------
// Helper: build wavefield pointer from the Python-supplied tensor list.
// Order MUST match AcousticVTI1st.FIELD_SPECS:
//   [vx, vz, sH, sV, m_sHx, m_sVz, m_vxx, m_vzz].
// ---------------------------------------------------------------------------
inline VTIWavefieldPointer make_wf_pointer(const std::vector<torch::Tensor>& w)
{
    TORCH_CHECK(w.size() == 8,
                "AcousticVTI1st 2D expects 8 wavefield tensors; got ", w.size());
    VTIWavefieldPointer p{};
    p.vx    = w[0].data_ptr<float>();
    p.vz    = w[1].data_ptr<float>();
    p.sH    = w[2].data_ptr<float>();
    p.sV    = w[3].data_ptr<float>();
    p.m_sHx = w[4].data_ptr<float>();
    p.m_sVz = w[5].data_ptr<float>();
    p.m_vxx = w[6].data_ptr<float>();
    p.m_vzz = w[7].data_ptr<float>();
    return p;
}


// ---------------------------------------------------------------------------
// Helper: map a Python field-index (0..7) to the corresponding wavefield ptr
// for source injection / receiver recording.  Returns nullptr for indices
// pointing at internal CPML memory variables (sources/receivers should not
// touch those).
// ---------------------------------------------------------------------------
inline float* vti_field_ptr(const VTIWavefieldPointer& wf, int field_index)
{
    switch (field_index) {
        case 0: return wf.vx;
        case 1: return wf.vz;
        case 2: return wf.sH;
        case 3: return wf.sV;
        default: return nullptr;  // m_sHx, m_sVz, m_vxx, m_vzz → not addressable
    }
}

}  // namespace acoustic_vti_1st_2d
