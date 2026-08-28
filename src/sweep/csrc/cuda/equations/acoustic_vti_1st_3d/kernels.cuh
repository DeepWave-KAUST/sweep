#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../operators/staggered.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"   // reuses ElasticCPMLPointer (12 vals in 3D)


namespace acoustic_vti_1st_3d {

// ===========================================================================
// AcousticVTI1st 3-D (Duveneck et al. 2008) — CUDA kernels.
//
// First-order velocity-stress acoustic VTI on a symmetric staggered grid:
//
//     ∂v_x/∂t = (1/ρ) ∂σ_H/∂x
//     ∂v_y/∂t = (1/ρ) ∂σ_H/∂y      (NEW in 3-D — water-plane component)
//     ∂v_z/∂t = (1/ρ) ∂σ_V/∂z
//     ∂σ_H/∂t = c11 (∂v_x/∂x + ∂v_y/∂y) + c13 ∂v_z/∂z
//     ∂σ_V/∂t = c13 (∂v_x/∂x + ∂v_y/∂y) + c33 ∂v_z/∂z
//
// with stiffness coefficients
//     c11 = ρ V_P² (1+2ε),    c33 = ρ V_P²,    c13 = ρ V_P² √(1+2δ)
//
// Reference: Duveneck E., Milcik P., Bakker P. M., Perkins C. (2008),
// "Acoustic VTI wave equations and their application for anisotropic
// reverse-time migration", SEG Las Vegas Annual Meeting, pp. 2186-2190,
// DOI: 10.1190/1.3059320.
//
// VTI is rotationally invariant about the vertical (z) axis, so the
// 3-D system needs no σ_xy / σ_xz / σ_yz components — σ_H is the (equal)
// horizontal normal stress in both x and y.  This is the same idea as the
// 2-D Duveneck system, just with one extra velocity component (v_y) and
// the corresponding horizontal-divergence term ∂v_y/∂y in the stress
// update.  Variable density is supported natively.
// ===========================================================================

// ---------------------------------------------------------------------------
// Wavefield pointer view (5 physical + 6 CPML memory variables)
//
// Physical fields, in Python field order (matches AcousticVTI1st3D.FIELD_SPECS):
//     [vx, vy, vz, sH, sV]
// CPML memory variables (velocity update reads stress derivatives):
//     m_sHx (∂σH/∂x), m_sHy (∂σH/∂y), m_sVz (∂σV/∂z)
// CPML memory variables (stress update reads velocity derivatives):
//     m_vxx (∂v_x/∂x), m_vyy (∂v_y/∂y), m_vzz (∂v_z/∂z)
// ---------------------------------------------------------------------------

struct VTIWavefieldPointer3D {
    float* __restrict__ vx;
    float* __restrict__ vy;
    float* __restrict__ vz;
    float* __restrict__ sH;
    float* __restrict__ sV;
    float* __restrict__ m_sHx;
    float* __restrict__ m_sHy;
    float* __restrict__ m_sVz;
    float* __restrict__ m_vxx;
    float* __restrict__ m_vyy;
    float* __restrict__ m_vzz;

    __device__ VTIWavefieldPointer3D offset(int b, int spatial_size) const
    {
        VTIWavefieldPointer3D out = *this;
        const int shift = b * spatial_size;
        out.vx    += shift;
        out.vy    += shift;
        out.vz    += shift;
        out.sH    += shift;
        out.sV    += shift;
        out.m_sHx += shift;
        out.m_sHy += shift;
        out.m_sVz += shift;
        out.m_vxx += shift;
        out.m_vyy += shift;
        out.m_vzz += shift;
        return out;
    }
};


// ---------------------------------------------------------------------------
// Forward LAUNCH macros (compile-time order dispatch, same pattern as 2D)
// ---------------------------------------------------------------------------

#define LAUNCH_VTI_3D_VELOCITY(order, grid, block, ...)                         \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_3d::velocity_kernel_3d<2>       \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_3d::velocity_kernel_3d<4>       \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_3d::velocity_kernel_3d<6>       \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_3d::velocity_kernel_3d<8>       \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_3d::velocity_kernel_3d<-1>      \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


#define LAUNCH_VTI_3D_STRESS(order, grid, block, ...)                           \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_3d::stress_kernel_3d<2>         \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_3d::stress_kernel_3d<4>         \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_3d::stress_kernel_3d<6>         \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_3d::stress_kernel_3d<8>         \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_3d::stress_kernel_3d<-1>        \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


// ---------------------------------------------------------------------------
// Velocity update kernel (3-D)
//
//   ∂v_x/∂t = (1/ρ) ∂σ_H/∂x   (∂σ_H/∂x at v_x location → forward staggered x)
//   ∂v_y/∂t = (1/ρ) ∂σ_H/∂y   (∂σ_H/∂y at v_y location → forward staggered y)
//   ∂v_z/∂t = (1/ρ) ∂σ_V/∂z   (∂σ_V/∂z at v_z location → forward staggered z)
//
// CPML half-step coefficients (axh/bxh, ayh/byh, azh/bzh).
// ---------------------------------------------------------------------------
template<int Order>
__global__ void velocity_kernel_3d(
    VTIWavefieldPointer3D wf,
    const float* __restrict__ inv_rho,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* inv_rho_b = inv_rho + b * spatial_size;

    // Stress spatial derivatives at velocity locations.
    //   ∂σ_H/∂x  : x-face       → forward staggered in x
    //   ∂σ_H/∂y  : y-face       → forward staggered in y
    //   ∂σ_V/∂z  : z-face       → forward staggered in z
    float dsH_dx = sgradient<3, Order, X, DIFF_FORWARD>(f.sH, ix, iy, iz, grad_ctx);
    float dsH_dy = sgradient<3, Order, Y, DIFF_FORWARD>(f.sH, ix, iy, iz, grad_ctx);
    float dsV_dz = sgradient<3, Order, Z, DIFF_FORWARD>(f.sV, ix, iy, iz, grad_ctx);

    float inv_rho_val = inv_rho_b[idx];

    // PML / interior split — outside the PML the half-step coefficients
    // vanish so the memory variables stay at zero.
    bool in_pml =
        (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
        (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.vx[idx] += solver.dt * inv_rho_val * dsH_dx;
        f.vy[idx] += solver.dt * inv_rho_val * dsH_dy;
        f.vz[idx] += solver.dt * inv_rho_val * dsV_dz;
        return;
    }

    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];
    float ayh = cpml.ayh[iy];
    float byh = cpml.byh[iy];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    f.m_sHx[idx] = axh * f.m_sHx[idx] + bxh * dsH_dx;
    dsH_dx     += f.m_sHx[idx];

    f.m_sHy[idx] = ayh * f.m_sHy[idx] + byh * dsH_dy;
    dsH_dy     += f.m_sHy[idx];

    f.m_sVz[idx] = azh * f.m_sVz[idx] + bzh * dsV_dz;
    dsV_dz     += f.m_sVz[idx];

    f.vx[idx] += solver.dt * inv_rho_val * dsH_dx;
    f.vy[idx] += solver.dt * inv_rho_val * dsH_dy;
    f.vz[idx] += solver.dt * inv_rho_val * dsV_dz;
}


// ---------------------------------------------------------------------------
// Stress update kernel (3-D)
//
//   ∂σ_H/∂t = c11 (∂v_x/∂x + ∂v_y/∂y) + c13 ∂v_z/∂z
//   ∂σ_V/∂t = c13 (∂v_x/∂x + ∂v_y/∂y) + c33 ∂v_z/∂z
//
// where c11 = ρ V_P² (1+2ε), c33 = ρ V_P², c13 = ρ V_P² √(1+2δ).
// CPML cell-centre coefficients (ax/bx, ay/by, az/bz).
// ---------------------------------------------------------------------------
template<int Order>
__global__ void stress_kernel_3d(
    VTIWavefieldPointer3D wf,
    const float* __restrict__ c11,
    const float* __restrict__ c33,
    const float* __restrict__ c13,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* c11_b = c11 + b * spatial_size;
    const float* c33_b = c33 + b * spatial_size;
    const float* c13_b = c13 + b * spatial_size;

    // Velocity spatial derivatives at stress locations (cell centre):
    //   ∂v_x/∂x : x-face → centre → backward staggered in x
    //   ∂v_y/∂y : y-face → centre → backward staggered in y
    //   ∂v_z/∂z : z-face → centre → backward staggered in z
    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

    float C11 = c11_b[idx];
    float C33 = c33_b[idx];
    float C13 = c13_b[idx];

    bool in_pml =
        (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
        (iy < solver.abcn + halo) || (iy >= solver.ny - solver.abcn - halo) ||
        (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        float dh = dvx_dx + dvy_dy;       // horizontal divergence
        f.sH[idx] += solver.dt * (C11 * dh + C13 * dvz_dz);
        f.sV[idx] += solver.dt * (C13 * dh + C33 * dvz_dz);
        return;
    }

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float ay = cpml.ay[iy];
    float by = cpml.by[iy];
    float az = cpml.az[iz];
    float bz = cpml.bz[iz];

    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx     += f.m_vxx[idx];

    f.m_vyy[idx] = ay * f.m_vyy[idx] + by * dvy_dy;
    dvy_dy     += f.m_vyy[idx];

    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz     += f.m_vzz[idx];

    float dh = dvx_dx + dvy_dy;
    f.sH[idx] += solver.dt * (C11 * dh + C13 * dvz_dz);
    f.sV[idx] += solver.dt * (C13 * dh + C33 * dvz_dz);
}


// ===========================================================================
//                       NO-PML kernels (time-reverse)
//
// Used by backward_bs to step the saved forward state ONE Δt backward in
// time inside the interior.  Same physics as the PML kernels but:
//   (a) the CPML memory-variable fast-path is removed (boundary cells are
//       restored by the saver between each step);
//   (b) `+= dt · ...` becomes `-= dt · ...` to undo one forward step.
// ===========================================================================

template<int Order>
__global__ void velocity_kernel_3d_nopml(
    VTIWavefieldPointer3D wf,
    const float* __restrict__ inv_rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) { M = solver.M; } else { M = Order / 2; }
    int halo = solver.abcn + M + 1;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* inv_rho_b = inv_rho + b * spatial_size;

    float dsH_dx = sgradient<3, Order, X, DIFF_FORWARD>(f.sH, ix, iy, iz, grad_ctx);
    float dsH_dy = sgradient<3, Order, Y, DIFF_FORWARD>(f.sH, ix, iy, iz, grad_ctx);
    float dsV_dz = sgradient<3, Order, Z, DIFF_FORWARD>(f.sV, ix, iy, iz, grad_ctx);

    float ir = inv_rho_b[idx];
    f.vx[idx] -= solver.dt * ir * dsH_dx;
    f.vy[idx] -= solver.dt * ir * dsH_dy;
    f.vz[idx] -= solver.dt * ir * dsV_dz;
}

template<int Order>
__global__ void stress_kernel_3d_nopml(
    VTIWavefieldPointer3D wf,
    const float* __restrict__ c11,
    const float* __restrict__ c33,
    const float* __restrict__ c13,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    int M;
    if constexpr (Order == -1) { M = solver.M; } else { M = Order / 2; }
    int halo = solver.abcn + M + 1;
    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    const float* c11_b = c11 + b * spatial_size;
    const float* c33_b = c33 + b * spatial_size;
    const float* c13_b = c13 + b * spatial_size;

    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(f.vx, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(f.vy, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(f.vz, ix, iy, iz, grad_ctx);

    float C11 = c11_b[idx];
    float C33 = c33_b[idx];
    float C13 = c13_b[idx];

    float dh = dvx_dx + dvy_dy;
    f.sH[idx] -= solver.dt * (C11 * dh + C13 * dvz_dz);
    f.sV[idx] -= solver.dt * (C13 * dh + C33 * dvz_dz);
}


#define LAUNCH_VTI_3D_VELOCITY_NOPML(order, grid, block, ...)                   \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_3d::velocity_kernel_3d_nopml<2> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_3d::velocity_kernel_3d_nopml<4> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_3d::velocity_kernel_3d_nopml<6> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_3d::velocity_kernel_3d_nopml<8> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_3d::velocity_kernel_3d_nopml<-1>\
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)

#define LAUNCH_VTI_3D_STRESS_NOPML(order, grid, block, ...)                     \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_3d::stress_kernel_3d_nopml<2>   \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_3d::stress_kernel_3d_nopml<4>   \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_3d::stress_kernel_3d_nopml<6>   \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_3d::stress_kernel_3d_nopml<8>   \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_3d::stress_kernel_3d_nopml<-1>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


// ===========================================================================
//                ADJOINT (BACKWARD) KERNELS — Phase 1, 3-D
// ---------------------------------------------------------------------------
// Continuous-form derivation of the 3-D Duveneck adjoint.  See the 2-D
// kernels.cuh for the full discussion; only the y-direction additions are
// noted here.
//
// Forward step (one leapfrog cycle):
//   vx += dt * inv_rho * Df_x(sH)
//   vy += dt * inv_rho * Df_y(sH)                ← NEW
//   vz += dt * inv_rho * Df_z(sV)
//   sH += dt * c11 * (Db_x(vx) + Db_y(vy)) + dt * c13 * Db_z(vz)   ← Db_y NEW
//   sV += dt * c13 * (Db_x(vx) + Db_y(vy)) + dt * c33 * Db_z(vz)
//
// Adjoint propagation (going backward in time, accumulate "+="):
//   λvx += -dt * (c11 * Df_x(λsH) + c13 * Df_x(λsV))
//   λvy += -dt * (c11 * Df_y(λsH) + c13 * Df_y(λsV))   ← NEW
//   λvz += -dt * (c13 * Df_z(λsH) + c33 * Df_z(λsV))
//   λsH += -dt * inv_rho * (Db_x(λvx) + Db_y(λvy))     ← Db_y NEW
//   λsV += -dt * inv_rho * Db_z(λvz)
//
// Material-parameter gradients per time step:
//   grad_c11    += dt * λsH * (Db_x(vx_fwd) + Db_y(vy_fwd))
//   grad_c13    += dt * (λsH * Db_z(vz_fwd)
//                        + λsV * (Db_x(vx_fwd) + Db_y(vy_fwd)))
//   grad_c33    += dt * λsV * Db_z(vz_fwd)
//   grad_irho   += dt * (λvx * Df_x(sH_fwd) + λvy * Df_y(sH_fwd)
//                        + λvz * Df_z(sV_fwd))
//
// Chain rule to (vp, ε, δ, ρ) is applied in calculate_grad_kernel_3d below.
//
// As in 2-D, the adjoint must be split into TWO kernels to avoid the
// read/write race on velocity neighbours.  Kernel A reads λsH/λsV and
// writes to λvx/λvy/λvz; kernel B reads the *new* λvx/λvy/λvz and writes
// to λsH/λsV.
// ===========================================================================

// Kernel A: λ_sH/λ_sV → contribute to λ_vx, λ_vy, λ_vz  (adjoint of stress update)
//   λvx += -dt * (c11 * Df_x(λsH) + c13 * Df_x(λsV))
//   λvy += -dt * (c11 * Df_y(λsH) + c13 * Df_y(λsV))
//   λvz += -dt * (c13 * Df_z(λsH) + c33 * Df_z(λsV))
// Pre-multiply for kernel A.  Forward: s += dt*(c11*(Db_x v_x + Db_y v_y) +
// c13*Db_z v_z), so as an operator on v the adjoint is -Df( c * lambda_s ) --
// the stiffness multiplies the adjoint stress AT THE STRESS LOCATION and only
// then is differentiated.  Applying it after the derivative is equivalent only
// for a homogeneous model; otherwise the two differ by a grad(c) term that
// accumulates once per step.  ph feeds the x and y derivatives, pv the z one.
template<int Dummy>
__global__ void adjoint_premultiply_stress_kernel_3d(
    VTIWavefieldPointer3D adj,
    const float* __restrict__ c11,
    const float* __restrict__ c33,
    const float* __restrict__ c13,
    float* __restrict__ ph,          // c11*lSH + c13*lSV
    float* __restrict__ pv,          // c13*lSH + c33*lSV
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    // No halo guard: kernel A's stencil reads these cells.
    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    long long u_idx = (long long)b * spatial_size + idx;

    auto a = adj.offset(b, spatial_size);
    const float lSH = a.sH[idx];
    const float lSV = a.sV[idx];
    ph[u_idx] = c11[u_idx] * lSH + c13[u_idx] * lSV;
    pv[u_idx] = c13[u_idx] * lSH + c33[u_idx] * lSV;
}


// Pre-multiply for kernel B: inv_rho sits at the VELOCITY location, inside the
// derivative -- lambda_s += -dt * Db( inv_rho * lambda_v ).
template<int Dummy>
__global__ void adjoint_premultiply_vel_kernel_3d(
    VTIWavefieldPointer3D adj,
    const float* __restrict__ inv_rho,
    float* __restrict__ qx,
    float* __restrict__ qy,
    float* __restrict__ qz,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;
    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;
    long long u_idx = (long long)b * spatial_size + idx;

    auto a = adj.offset(b, spatial_size);
    const float ir = inv_rho[u_idx];
    qx[u_idx] = ir * a.vx[idx];
    qy[u_idx] = ir * a.vy[idx];
    qz[u_idx] = ir * a.vz[idx];
}


template<int Order>
__global__ void adjoint_stress_to_vel_kernel_3d(
    VTIWavefieldPointer3D adj,
    const float* __restrict__ ph,
    const float* __restrict__ pv,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto a = adj.offset(b, spatial_size);
    const float* ph_b = ph + b * spatial_size;
    const float* pv_b = pv + b * spatial_size;

    float dph_dx = sgradient<3, Order, X, DIFF_FORWARD>(ph_b, ix, iy, iz, grad_ctx);
    float dph_dy = sgradient<3, Order, Y, DIFF_FORWARD>(ph_b, ix, iy, iz, grad_ctx);
    float dpv_dz = sgradient<3, Order, Z, DIFF_FORWARD>(pv_b, ix, iy, iz, grad_ctx);

    a.vx[idx] += -solver.dt * dph_dx;
    a.vy[idx] += -solver.dt * dph_dy;
    a.vz[idx] += -solver.dt * dpv_dz;
}


// Kernel B: λ_vx, λ_vy, λ_vz → contribute to λ_sH/λ_sV  (adjoint of velocity update)
//   λsH += -dt * inv_rho * (Db_x(λvx) + Db_y(λvy))
//   λsV += -dt * inv_rho * Db_z(λvz)
template<int Order>
__global__ void adjoint_vel_to_stress_kernel_3d(
    VTIWavefieldPointer3D adj,
    const float* __restrict__ qx,
    const float* __restrict__ qy,
    const float* __restrict__ qz,
    SGradParam grad_ctx,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    auto a = adj.offset(b, spatial_size);
    const float* qx_b = qx + b * spatial_size;
    const float* qy_b = qy + b * spatial_size;
    const float* qz_b = qz + b * spatial_size;

    float dqx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(qx_b, ix, iy, iz, grad_ctx);
    float dqy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(qy_b, ix, iy, iz, grad_ctx);
    float dqz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(qz_b, ix, iy, iz, grad_ctx);

    a.sH[idx] += -solver.dt * (dqx_dx + dqy_dy);
    a.sV[idx] += -solver.dt * dqz_dz;
}


// ---------------------------------------------------------------------------
// Gradient kernel — accumulates ∂L/∂vp, ∂L/∂ε, ∂L/∂δ, ∂L/∂ρ once per step.
//
// See the 2-D companion for the timing convention.  Inputs:
//   * fvx_now, fvy_now, fvz_now : u_forward[it].{vx,vy,vz}  (stiffness grad)
//   * fsH_prev, fsV_prev        : u_forward[it-1].{sH,sV}   (inv-ρ grad)
//   * adjoint                   : current adjoint state
//   * vp, ε, δ, ρ               : models
// Outputs (accumulate into):
//   grad_vp, grad_eps, grad_delta, grad_rho
// ---------------------------------------------------------------------------
template<int Order>
__global__ void calculate_grad_kernel_3d(
    const float* __restrict__ fvx_now,
    const float* __restrict__ fvy_now,
    const float* __restrict__ fvz_now,
    const float* __restrict__ fsH_prev,
    const float* __restrict__ fsV_prev,
    VTIWavefieldPointer3D adj,
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
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= solver.nx || iy >= solver.ny) return;
    if (iz_global >= solver.nz * solver.B) return;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iy < halo || iy >= solver.ny - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    long long spatial_size = (long long)solver.nx * solver.ny * solver.nz;
    int idx = iz * solver.nx * solver.ny + iy * solver.nx + ix;

    const float* fvx_b = fvx_now  + b * spatial_size;
    const float* fvy_b = fvy_now  + b * spatial_size;
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

    // Forward spatial derivatives (NO PML — interior only).
    // dvx_dx, dvy_dy, dvz_dz : at the stress location, use vx,vy,vz AT END
    //                          of step (= u_forward[it])
    // dsH_dx, dsH_dy, dsV_dz : at the velocity location, use sH,sV AT START
    //                          of step (= u_forward[it-1])
    float dvx_dx = sgradient<3, Order, X, DIFF_BACKWARD>(fvx_b, ix, iy, iz, grad_ctx);
    float dvy_dy = sgradient<3, Order, Y, DIFF_BACKWARD>(fvy_b, ix, iy, iz, grad_ctx);
    float dvz_dz = sgradient<3, Order, Z, DIFF_BACKWARD>(fvz_b, ix, iy, iz, grad_ctx);
    float dsH_dx = sgradient<3, Order, X, DIFF_FORWARD> (fsH_b, ix, iy, iz, grad_ctx);
    float dsH_dy = sgradient<3, Order, Y, DIFF_FORWARD> (fsH_b, ix, iy, iz, grad_ctx);
    float dsV_dz = sgradient<3, Order, Z, DIFF_FORWARD> (fsV_b, ix, iy, iz, grad_ctx);

    float lSH = a.sH[idx];
    float lSV = a.sV[idx];
    float lvx = a.vx[idx];
    float lvy = a.vy[idx];
    float lvz = a.vz[idx];

    float dh = dvx_dx + dvy_dy;        // horizontal divergence
    float dt = solver.dt;

    // Stiffness-coefficient gradients (∂L/∂c11, ∂L/∂c33, ∂L/∂c13, ∂L/∂(1/ρ)).
    // The y-direction adds Db_y(vy) into the horizontal divergence and
    // Df_y(sH) into the velocity-update product.
    float grad_c11    = dt *  lSH * dh;
    float grad_c33    = dt *  lSV * dvz_dz;
    float grad_c13    = dt * (lSH * dvz_dz + lSV * dh);
    float grad_irho   = dt * (lvx * dsH_dx + lvy * dsH_dy + lvz * dsV_dz);

    // Material values
    float vp_v   = vp_b[idx];
    float eps_v  = eps_b[idx];
    float dlt_v  = dlt_b[idx];
    float rho_v  = rho_b[idx];

    float one_p_2eps = 1.f + 2.f * eps_v;
    float one_p_2dlt = 1.f + 2.f * dlt_v;
    float sqrt_d     = sqrtf(one_p_2dlt);
    float vp2        = vp_v * vp_v;

    // Chain rule (same as 2-D):
    //   c11 = ρ vp² (1+2ε)        c33 = ρ vp²        c13 = ρ vp² √(1+2δ)
    //   inv_rho = 1/ρ
    float common_vp  = grad_c11 * one_p_2eps + grad_c33 + grad_c13 * sqrt_d;
    gvp_b[idx]  += 2.f * rho_v * vp_v * common_vp;
    geps_b[idx] += grad_c11 * 2.f * rho_v * vp2;
    gdlt_b[idx] += grad_c13 * rho_v * vp2 / fmaxf(sqrt_d, 1e-12f);
    grho_b[idx] += vp2 * common_vp - grad_irho / (rho_v * rho_v);
}


#define LAUNCH_VTI_3D_ADJOINT_STRESS_TO_VEL(order, grid, block, ...)            \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_3d::adjoint_stress_to_vel_kernel_3d<2>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_3d::adjoint_stress_to_vel_kernel_3d<4>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_3d::adjoint_stress_to_vel_kernel_3d<6>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_3d::adjoint_stress_to_vel_kernel_3d<8>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_3d::adjoint_stress_to_vel_kernel_3d<-1> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)

#define LAUNCH_VTI_3D_ADJOINT_VEL_TO_STRESS(order, grid, block, ...)            \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_3d::adjoint_vel_to_stress_kernel_3d<2>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_3d::adjoint_vel_to_stress_kernel_3d<4>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_3d::adjoint_vel_to_stress_kernel_3d<6>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_3d::adjoint_vel_to_stress_kernel_3d<8>  \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_3d::adjoint_vel_to_stress_kernel_3d<-1> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)

#define LAUNCH_VTI_3D_CALC_GRAD(order, grid, block, ...)                        \
    do {                                                                        \
        if      ((order) == 2) acoustic_vti_1st_3d::calculate_grad_kernel_3d<2> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 4) acoustic_vti_1st_3d::calculate_grad_kernel_3d<4> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 6) acoustic_vti_1st_3d::calculate_grad_kernel_3d<6> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else if ((order) == 8) acoustic_vti_1st_3d::calculate_grad_kernel_3d<8> \
                                   <<<grid, block>>>(__VA_ARGS__);              \
        else                   acoustic_vti_1st_3d::calculate_grad_kernel_3d<-1>\
                                   <<<grid, block>>>(__VA_ARGS__);              \
    } while (0)


// ---------------------------------------------------------------------------
// Helper: build wavefield pointer from a Python-supplied tensor list.
// Order MUST match AcousticVTI1st3D.FIELD_SPECS:
//   [vx, vy, vz, sH, sV, m_sHx, m_sHy, m_sVz, m_vxx, m_vyy, m_vzz].
// ---------------------------------------------------------------------------
inline VTIWavefieldPointer3D make_wf_pointer_3d(const std::vector<torch::Tensor>& w)
{
    TORCH_CHECK(w.size() == 11,
                "AcousticVTI1st 3D expects 11 wavefield tensors; got ", w.size());
    VTIWavefieldPointer3D p{};
    p.vx    = w[0].data_ptr<float>();
    p.vy    = w[1].data_ptr<float>();
    p.vz    = w[2].data_ptr<float>();
    p.sH    = w[3].data_ptr<float>();
    p.sV    = w[4].data_ptr<float>();
    p.m_sHx = w[5].data_ptr<float>();
    p.m_sHy = w[6].data_ptr<float>();
    p.m_sVz = w[7].data_ptr<float>();
    p.m_vxx = w[8].data_ptr<float>();
    p.m_vyy = w[9].data_ptr<float>();
    p.m_vzz = w[10].data_ptr<float>();
    return p;
}


// ---------------------------------------------------------------------------
// Helper: map a Python field-index (0..10) to the corresponding wavefield
// pointer for source injection / receiver recording.  Returns nullptr for
// CPML memory variables (indices 5..10).
// ---------------------------------------------------------------------------
inline float* vti_field_ptr_3d(const VTIWavefieldPointer3D& wf, int field_index)
{
    switch (field_index) {
        case 0: return wf.vx;
        case 1: return wf.vy;
        case 2: return wf.vz;
        case 3: return wf.sH;
        case 4: return wf.sV;
        default: return nullptr;
    }
}

}  // namespace acoustic_vti_1st_3d
