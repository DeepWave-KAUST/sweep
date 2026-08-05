#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../../operators/staggered.cuh"
#include "../../operators/gradient.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/elastic_free_surface.cuh"

#define LAUNCH_ELASTIC_VELOCITY(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


#define LAUNCH_ELASTIC_STRESS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_NOPML(order, grid, block, ...)                     \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_NOPML(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_kernel_nopml<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_nopml<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_nopml<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_nopml<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_nopml<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC_BS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) calculate_grad_elastic_bs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_bs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_bs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_bs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_bs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) calculate_grad_elastic_nobs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_nobs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_nobs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_nobs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_nobs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_ADJOINT_PREPARE(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_ADJOINT_APPLY(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_stress_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_prepare<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_prepare<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_prepare<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_prepare<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_prepare<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_ADJOINT_APPLY(order, grid, block, ...)                       \
    do {                                                        \
        if      ((order) == 2) elastic_velocity_adjoint_apply<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_apply<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_apply<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_apply<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_apply<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


// Forward velocity stencil launches with block(32,8)=256 threads.  Baseline
// ~48 regs / ~53% occupancy on V100 (Volta) — register/occupancy-bound, not
// spill-bound.  Capping at 256 threads/block with minBlocks=8 forces the
// compiler to <=32 regs, lifting occupancy to ~90% (a measured forward win,
// elastic2d 1.19->1.03 vs deepwave on V100).  Math-identical: no effect on
// the adjoint or gradient.
template<int Order>
__global__ void __launch_bounds__(256, 8) elastic_velocity_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,

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

    const float* rho_b = rho + b * spatial_size;

    float dsxx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD> (f.sxx, ix, iz, grad_ctx, solver, true);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.szz, ix, iz, grad_ctx, solver, true);

    float inv_rho = 1.f / rho_b[idx];

    // Position-based PML / interior split. All ax/bx CPML coefficients vanish
    // outside the PML band, so the auxiliary fields m_szzz/m_sxzx/m_sxzz/m_sxxx
    // become 0 → 0 in the interior. Skip the four reads + four writes.
    bool in_pml = (ix < solver.padLo(2) + halo) || (ix >= solver.nx - solver.padHi(2) - halo) ||
                  (iz < solver.padLo(0) + halo) ||
                  (iz >= solver.nz - solver.padHi(0) - halo);

    if (!in_pml) {
        f.vx[idx] += solver.dt * inv_rho * (dsxx_dx + dsxz_dz);
        f.vz[idx] += solver.dt * inv_rho * (dsxz_dx + dszz_dz);
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];
    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];

    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];
    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];

    f.vx[idx] += solver.dt * inv_rho *
        (dsxx_dx + dsxz_dz);

    f.vz[idx] += solver.dt * inv_rho *
        (dsxz_dx + dszz_dz);
}

// Forward stress stencil: same block(32,8)=256, same register/occupancy
// regime as elastic_velocity_kernel above.  Cap at 256/block, minBlocks=8
// (<=32 regs) to lift occupancy.  Math-identical (forward-only change).
template<int Order>
__global__ void __launch_bounds__(256, 8) elastic_stress_kernel(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    float* __restrict__ u_this,

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
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;

    float dvx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD> (f.vx, ix, iz, grad_ctx, solver, true);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>  (f.vz, ix, iz, grad_ctx, solver, false);

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    // PML / interior split. Conservative for free_surface: keep the iz == halo
    // row (the free-surface row, where szz/sxz must be reset to 0) on the full
    // PML path so the BC fires.
    // Stress kernel needs the per-column surface row INSIDE the in_pml zone
    // so the BC zero (``σ_zz = σ_xz = 0`` at the FS row) fires after the
    // PML update.  Under irregular topography the surface row varies per
    // column — ``solver.surface_row(ix)`` returns the runtime row index
    // (falls back to ``phys_z0()`` in flat mode → ``halo`` for free_surface).
    // Keep each active z free-surface row INSIDE the in_pml (full-path) zone so
    // the traction BC (σzz=σxz=0, Robertsson σxx) fires after the PML update.
    int z_lo = solver.fsLo(0) ? (solver.surface_row(ix) + 1) : (solver.padLo(0) + halo);
    int z_hi = solver.fsHi(0) ? elastic_z_bottom_surface_row(solver)
                              : (solver.nz - solver.padHi(0) - halo);
    bool in_pml = (ix < solver.padLo(2) + halo) || (ix >= solver.nx - solver.padHi(2) - halo) ||
                  (iz < z_lo) || (iz >= z_hi);

    if (!in_pml) {
        f.sxx[idx] += solver.dt * ((lam + 2.f*mu_) * dvx_dx + lam * dvz_dz);
        f.szz[idx] += solver.dt * ((lam + 2.f*mu_) * dvz_dz + lam * dvx_dx);
        f.sxz[idx] += solver.dt * mu_ * (dvx_dz + dvz_dx);
        if (u_this_b) {
            int comp_stride  = solver.B * spatial_size;
            u_this_b[0 * comp_stride + idx] = f.vx[idx];
            u_this_b[1 * comp_stride + idx] = f.vz[idx];
        }
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
    dvz_dz += f.m_vzz[idx];
    f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
    dvx_dx += f.m_vxx[idx];

    f.sxx[idx] += solver.dt *
        ((lam + 2.f*mu_) * dvx_dx +
         lam * dvz_dz);

    f.szz[idx] += solver.dt *
        ((lam + 2.f*mu_) * dvz_dz +
         lam * dvx_dx);

    f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
    dvx_dz += f.m_vxz[idx];
    f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
    dvz_dx += f.m_vzx[idx];

    f.sxz[idx] += solver.dt *
        mu_ * (dvx_dz + dvz_dx);

    // Per-face traction BC.  z faces (top/bottom): Robertsson σxx fix, then
    // σzz=σxz=0.  x faces (left/right): Robertsson σzz fix (x<->z swap), then
    // σxx=σxz=0.  Robertsson corrections first, zeroing after, so a z∩x corner
    // ends σxx=σzz=σxz=0 (matches the eager post-step zeroing order).
    bool is_z_fs = elastic_is_top_free_surface_row(solver, ix, iz);
    bool is_x_fs = elastic_is_x_free_surface_col(solver, ix);
    if (is_z_fs) f.sxx[idx] += -solver.dt * lam * (lam / (lam + 2.f*mu_) * dvx_dx + dvz_dz);
    if (is_x_fs) f.szz[idx] += -solver.dt * lam * (lam / (lam + 2.f*mu_) * dvz_dz + dvx_dx);
    if (is_z_fs) { f.szz[idx] = 0.f; f.sxz[idx] = 0.f; }
    if (is_x_fs) { f.sxx[idx] = 0.f; f.sxz[idx] = 0.f; }

    if (u_this_b) {
        int comp_stride  = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vz[idx];
    }

}

template<int Order>
__global__ void elastic_velocity_kernel_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    SGradParam grad_ctx,
    SolverContext solver
)
{
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

    int halo = solver.abcn + M+1;

    int z_lo = solver.fsLo(0) ? M : halo;
    int z_hi = solver.fsHi(0) ? M : halo;
    int x_lo = solver.fsLo(2) ? M : halo;
    int x_hi = solver.fsHi(2) ? M : halo;
    if (ix < x_lo || ix >= solver.nx - x_hi || iz < z_lo || iz >= solver.nz - z_hi)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    // Wavefields
    auto f = wf.offset(b, spatial_size);

    const float* rho_b = rho + b * spatial_size;

    float dsxx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD> (f.sxx, ix, iz, grad_ctx, solver, true);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.szz, ix, iz, grad_ctx, solver, true);

    float inv_rho = 1.f / rho_b[idx];

    f.vx[idx] -= solver.dt * inv_rho *
        (dsxx_dx + dsxz_dz);

    f.vz[idx] -= solver.dt * inv_rho *
        (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void elastic_stress_kernel_nopml(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,

    SGradParam grad_ctx,
    SolverContext solver
)
{
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

    int halo = solver.abcn + M+1;

    int z_lo = solver.fsLo(0) ? M : halo;
    int z_hi = solver.fsHi(0) ? M : halo;
    int x_lo = solver.fsLo(2) ? M : halo;
    int x_hi = solver.fsHi(2) ? M : halo;
    if (ix < x_lo || ix >= solver.nx - x_hi || iz < z_lo || iz >= solver.nz - z_hi)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    // Wavefields
    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;

    float dvx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD> (f.vx, ix, iz, grad_ctx, solver, true);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>  (f.vz, ix, iz, grad_ctx, solver, false);

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    f.sxx[idx] -= solver.dt *
        ((lam + 2.f*mu_) * dvx_dx +
         lam * dvz_dz);

    f.szz[idx] -= solver.dt *
        ((lam + 2.f*mu_) * dvz_dz +
         lam * dvx_dx);

    f.sxz[idx] -= solver.dt *
        mu_ * (dvx_dz + dvz_dx);
}

template<int Order>
__global__ void elastic_stress_adjoint_prepare(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lambda,
    const float* __restrict__ mu,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qzz,
    float* __restrict__ qxz,
    float* __restrict__ qzx,
    // grad_ctx is always forwarded (used only when imaging pointers below are
    // non-null); harmless for bs/ckpt/apm callers that pass null imaging args.
    SGradParam grad_ctx,
    // FUSED vp/vs/rho-gradient imaging (full-mode only).  When non-null, this
    // kernel folds in the per-step gradient correlation that the standalone
    // calculate_grad_elastic_nobs would otherwise compute in a separate
    // full-grid pass.  At this kernel's entry the adjoint stress AND velocity
    // are the un-mutated post-source adjoint[it] fields (velocity is only
    // touched later by elastic_stress_adjoint_apply), so the operands match
    // calculate_grad_elastic_nobs exactly.  No lag: in the full backward loop
    // calculate_grad(it) ran immediately before apply_adjoint_step(it), and
    // both consume the same post-source adjoint[it].  All-null (bs/ckpt/apm).
    const float* __restrict__ grad_fvx      = nullptr,  // u_forward[it]   vx
    const float* __restrict__ grad_fvz      = nullptr,  // u_forward[it]   vz
    const float* __restrict__ grad_fvx_prev = nullptr,  // u_forward[it+1] vx
    const float* __restrict__ grad_fvz_prev = nullptr,  // u_forward[it+1] vz
    const float* __restrict__ grad_vp_model = nullptr,  // vp    (B,nz,nx)
    const float* __restrict__ grad_vs_model = nullptr,  // vs    (B,nz,nx)
    const float* __restrict__ grad_rho_model= nullptr,  // rho   (B,nz,nx)
    float* __restrict__ grad_vp_out         = nullptr,  // grad_vp accumulator
    float* __restrict__ grad_vs_out         = nullptr,  // grad_vs accumulator
    float* __restrict__ grad_rho_out        = nullptr   // grad_rho accumulator
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    const float* lam_b = lambda + b * spatial_size;
    const float* mu_b  = mu     + b * spatial_size;
    float* qxx_b = qxx + b * spatial_size;
    float* qzz_b = qzz + b * spatial_size;
    float* qxz_b = qxz + b * spatial_size;
    float* qzx_b = qzx + b * spatial_size;

    float lam = lam_b[idx];
    float mu_ = mu_b[idx];

    bool is_z_fs = elastic_is_top_free_surface_row(solver, ix, iz);
    bool is_x_fs = elastic_is_x_free_surface_col(solver, ix);
    float bar_sxx = f.sxx[idx];
    float bar_szz = f.szz[idx];
    float bar_sxz = f.sxz[idx];
    if (is_z_fs) { bar_szz = 0.f; bar_sxz = 0.f; f.szz[idx] = 0.f; f.sxz[idx] = 0.f; }
    if (is_x_fs) { bar_sxx = 0.f; bar_sxz = 0.f; f.sxx[idx] = 0.f; f.sxz[idx] = 0.f; }

    // --- FUSED gradient imaging (full-mode only) -----------------------------
    // Equivalent to a calculate_grad_elastic_nobs launch for this reverse step.
    // calculate_grad skips the halo entirely (ix/iz in [halo, n-halo)); replicate
    // that so the folded contribution is bit-identical (halo adjoint cells carry
    // no gradient there).  bar_sxx/bar_szz/bar_sxz above already equal
    // calculate_grad's a.sxx[idx] / bar_szz / bar_sxz (same FS-row zeroing), and
    // f.vx[idx]/f.vz[idx] are the un-mutated post-source adjoint velocities.
    if (grad_vp_out != nullptr &&
        ix >= halo && ix < solver.nx - halo &&
        iz >= halo && iz < solver.nz - halo) {
        const float* fvx_b      = grad_fvx      + b * spatial_size;
        const float* fvz_b      = grad_fvz      + b * spatial_size;
        const float* fvx_prev_b = grad_fvx_prev + b * spatial_size;
        const float* fvz_prev_b = grad_fvz_prev + b * spatial_size;
        const float* vp_b       = grad_vp_model + b * spatial_size;
        const float* vs_b       = grad_vs_model + b * spatial_size;
        const float* rho_b      = grad_rho_model+ b * spatial_size;
        float* grad_vp_b        = grad_vp_out   + b * spatial_size;
        float* grad_vs_b        = grad_vs_out   + b * spatial_size;
        float* grad_rho_b       = grad_rho_out  + b * spatial_size;

        float fvx_x = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(fvx_b, ix, iz, grad_ctx, solver, true);
        float fvz_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(fvz_b, ix, iz, grad_ctx, solver, true);
        float fvx_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (fvx_b, ix, iz, grad_ctx, solver, false);
        float fvz_x = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>(fvz_b, ix, iz, grad_ctx, solver, false);

        float grad_lambda, grad_mu;
        if (is_z_fs && is_x_fs) {
            grad_lambda = 0.f; grad_mu = 0.f;   // z∩x corner: sxx=szz=sxz=0, no dependence
        } else if (is_z_fs || is_x_fs) {
            // Material derivative of the Robertsson FS normal-stress fix.  z face:
            // sxx = old + dt*C_surf*dvx_dx (bar_sxx*fvx_x); x face (x<->z swap):
            // szz = old + dt*C_surf*dvz_dz (bar_szz*fvz_z).  C_surf=4mu(lam+mu)/(lam+2mu);
            // with lam=rho(vp^2-2vs^2), mu=rho vs^2, lam+2mu=rho vp^2:
            // dC/dlam = 4 vs^4/vp^4,  dC/dmu = 4 (vp^4 - 2 vp^2 vs^2 + 2 vs^4)/vp^4.
            float vp2 = vp_b[idx] * vp_b[idx];
            float vs2 = vs_b[idx] * vs_b[idx];
            float vp4 = vp2 * vp2;
            float a = is_z_fs ? (bar_sxx * fvx_x) : (bar_szz * fvz_z);
            grad_lambda = a * 4.f * vs2 * vs2 / vp4;
            grad_mu     = a * 4.f * (vp4 - 2.f * vp2 * vs2 + 2.f * vs2 * vs2) / vp4;
        } else {
            grad_lambda = (bar_sxx + bar_szz) * (fvx_x + fvz_z);
            grad_mu = 2*(bar_sxx * fvx_x + bar_szz * fvz_z) + bar_sxz * (fvx_z + fvz_x);
        }

        grad_vp_b[idx] += -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
        grad_vs_b[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                             2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

        grad_rho_b[idx] += (f.vx[idx] * (fvx_b[idx]-fvx_prev_b[idx]) +
                            f.vz[idx] * (fvz_b[idx]-fvz_prev_b[idx])) / rho_b[idx];
        grad_rho_b[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                           grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;
    }
    // -------------------------------------------------------------------------

    // Transpose of the Robertsson FS normal-stress fix.  z face: sxx = old +
    // dt*C_surf*dvx_dx so bar_dvx_dx = dt*C_surf*bar_sxx (dvz_dz cancels, szz/sxz
    // zeroed).  x face (x<->z swap): bar_dvz_dz = dt*C_surf*bar_szz.  z∩x corner:
    // sxx=szz=sxz=0, no derivative dependence -> all bar_dv* = 0.
    float bar_dvx_dx, bar_dvz_dz, bar_dvx_dz, bar_dvz_dx;
    if (is_z_fs && is_x_fs) {
        bar_dvx_dx = 0.f; bar_dvz_dz = 0.f; bar_dvx_dz = 0.f; bar_dvz_dx = 0.f;
    } else if (is_z_fs) {
        float c_surf = 4.f * mu_ * (lam + mu_) / (lam + 2.f * mu_);
        bar_dvx_dx = solver.dt * c_surf * bar_sxx;
        bar_dvz_dz = 0.f; bar_dvx_dz = 0.f; bar_dvz_dx = 0.f;
    } else if (is_x_fs) {
        float c_surf = 4.f * mu_ * (lam + mu_) / (lam + 2.f * mu_);
        bar_dvz_dz = solver.dt * c_surf * bar_szz;
        bar_dvx_dx = 0.f; bar_dvx_dz = 0.f; bar_dvz_dx = 0.f;
    } else {
        bar_dvx_dx = solver.dt * ((lam + 2.f * mu_) * bar_sxx + lam * bar_szz);
        bar_dvz_dz = solver.dt * ((lam + 2.f * mu_) * bar_szz + lam * bar_sxx);
        bar_dvx_dz = solver.dt * mu_ * bar_sxz;
        bar_dvz_dx = solver.dt * mu_ * bar_sxz;
    }

    // Position-based PML / interior split. Outside the PML band all
    // ax/az/bx/bz coefficients vanish, m_v* aux fields stay 0, so the four
    // q* outputs collapse to bar_dv*_d* and the four m_v* writes become
    // 0 -> 0. Skip them — saves 4 reads + 4 writes per interior cell.
    // Matches the forward elastic_stress_kernel fast-path (line 230).
    bool in_pml = (ix < solver.padLo(2) + halo) || (ix >= solver.nx - solver.padHi(2) - halo) ||
                  (iz < solver.padLo(0) + halo) ||
                  (iz >= solver.nz - solver.padHi(0) - halo);
    if (!in_pml) {
        qxx_b[idx] = bar_dvx_dx;
        qzz_b[idx] = bar_dvz_dz;
        qxz_b[idx] = bar_dvx_dz;
        qzx_b[idx] = bar_dvz_dx;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;

    qxx_b[idx] = bar_dvx_dx + bx * tmp_vxx;
    qzz_b[idx] = bar_dvz_dz + bz * tmp_vzz;
    qxz_b[idx] = bar_dvx_dz + bzh * tmp_vxz;
    qzx_b[idx] = bar_dvz_dx + bxh * tmp_vzx;

    f.m_vxx[idx] = ax * tmp_vxx;
    f.m_vzz[idx] = az * tmp_vzz;
    f.m_vxz[idx] = azh * tmp_vxz;
    f.m_vzx[idx] = axh * tmp_vzx;
}

template<int Order>
__global__ void elastic_stress_adjoint_apply(
    ElasticWavefieldPointer wf,
    const float* __restrict__ qxx,
    const float* __restrict__ qzz,
    const float* __restrict__ qxz,
    const float* __restrict__ qzx,
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
    auto f = wf.offset(b, spatial_size);

    const float* qxx_b = qxx + b * spatial_size;
    const float* qzz_b = qzz + b * spatial_size;
    const float* qxz_b = qxz + b * spatial_size;
    const float* qzx_b = qzx + b * spatial_size;

    float dqxx_dx = elastic_fs_adjoint_sgradient_x_2d<Order, DIFF_BACKWARD>(qxx_b, ix, iz, grad_ctx, solver, true);
    float dqxz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_FORWARD> (qxz_b, ix, iz, grad_ctx, solver, false);
    float dqzx_dx = elastic_fs_adjoint_sgradient_x_2d<Order, DIFF_FORWARD>(qzx_b, ix, iz, grad_ctx, solver, false);
    float dqzz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_BACKWARD>(qzz_b, ix, iz, grad_ctx, solver, true);

    int idx = iz * solver.nx + ix;
    f.vx[idx] += dqxx_dx + dqxz_dz;
    f.vz[idx] += dqzx_dx + dqzz_dz;
}

template<int Order>
__global__ void elastic_velocity_adjoint_prepare(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ pxx,
    float* __restrict__ pzz,
    float* __restrict__ pxz,
    float* __restrict__ pzx
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);

    const float* rho_b = rho + b * spatial_size;
    float* pxx_b = pxx + b * spatial_size;
    float* pzz_b = pzz + b * spatial_size;
    float* pxz_b = pxz + b * spatial_size;
    float* pzx_b = pzx + b * spatial_size;

    float inv_rho = 1.f / rho_b[idx];
    float bar_dsxx_dx = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dz = solver.dt * inv_rho * f.vx[idx];
    float bar_dsxz_dx = solver.dt * inv_rho * f.vz[idx];
    float bar_dszz_dz = solver.dt * inv_rho * f.vz[idx];

    // Position-based PML / interior split — same logic as
    // elastic_stress_adjoint_prepare above: ax/az/bx/bz vanish outside the
    // PML band, m_s* aux fields stay 0, so the four p* outputs collapse to
    // the bar_ds*_d* values and the four m_s* writes become 0 -> 0.
    bool in_pml = (ix < solver.padLo(2) + halo) || (ix >= solver.nx - solver.padHi(2) - halo) ||
                  (iz < solver.padLo(0) + halo) ||
                  (iz >= solver.nz - solver.padHi(0) - halo);
    if (!in_pml) {
        pxx_b[idx] = bar_dsxx_dx;
        pxz_b[idx] = bar_dsxz_dz;
        pzx_b[idx] = bar_dsxz_dx;
        pzz_b[idx] = bar_dszz_dz;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    pxx_b[idx] = bar_dsxx_dx + bxh * tmp_sxxx;
    pxz_b[idx] = bar_dsxz_dz + bz * tmp_sxzz;
    pzx_b[idx] = bar_dsxz_dx + bx * tmp_sxzx;
    pzz_b[idx] = bar_dszz_dz + bzh * tmp_szzz;

    f.m_sxxx[idx] = axh * tmp_sxxx;
    f.m_sxzz[idx] = az * tmp_sxzz;
    f.m_sxzx[idx] = ax * tmp_sxzx;
    f.m_szzz[idx] = azh * tmp_szzz;
}

template<int Order>
__global__ void elastic_velocity_adjoint_apply(
    ElasticWavefieldPointer wf,
    const float* __restrict__ pxx,
    const float* __restrict__ pzz,
    const float* __restrict__ pxz,
    const float* __restrict__ pzx,
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
    auto f = wf.offset(b, spatial_size);

    const float* pxx_b = pxx + b * spatial_size;
    const float* pzz_b = pzz + b * spatial_size;
    const float* pxz_b = pxz + b * spatial_size;
    const float* pzx_b = pzx + b * spatial_size;

    float dpxx_dx = elastic_fs_adjoint_sgradient_x_2d<Order, DIFF_FORWARD>(pxx_b, ix, iz, grad_ctx, solver, true);
    float dpzz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_FORWARD> (pzz_b, ix, iz, grad_ctx, solver, true);
    float dpxz_dz = elastic_top_fs_adjoint_sgradient_z_2d<Order, DIFF_BACKWARD>(pxz_b, ix, iz, grad_ctx, solver, true);
    float dpzx_dx = elastic_fs_adjoint_sgradient_x_2d<Order, DIFF_BACKWARD>(pzx_b, ix, iz, grad_ctx, solver, true);

    int idx = iz * solver.nx + ix;
    f.sxx[idx] += dpxx_dx;
    f.szz[idx] += dpzz_dz;
    f.sxz[idx] += dpxz_dz + dpzx_dx;
}


template<int Order>
__global__ void calculate_grad_elastic_bs(

    ElasticWavefieldPointer forward,
    ElasticWavefieldPointer adjoint,

    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvz_prev,

    const float* __restrict__ vp,        // (B, nz, nx)
    const float* __restrict__ vs,        // (B, nz, nx)
    const float* __restrict__ rho,       // (B, nz, nx)

    float* __restrict__ grad_vp,             // (B, nz, nx)
    float* __restrict__ grad_vs,             // (B, nz, nx)
    float* __restrict__ grad_rho,            // (B, nz, nx)

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

    // Even cells strictly above the per-column surface row ("air") can carry
    // nonzero gradient under irregular topography: vp/vs/rho at an air cell
    // (iz, ix) participate in the stress update at (iz, ix), which then
    // contributes to sxx via D_x — and D_x reads sxx across columns ix±k.
    // Neighbouring columns may have a LOWER surface so (iz, ix±k) is solid
    // there, and the physical wavefield does reach the receiver.  So Python
    // autograd assigns a nonzero gradient at air cells and we must too —
    // do NOT skip air rows here.

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;

    auto f = forward.offset(b, spatial_size);
    auto a = adjoint.offset(b, spatial_size);

    const float* vp_b = vp + b * spatial_size;
    const float* vs_b = vs + b * spatial_size;
    const float* rho_b = rho + b * spatial_size;

    float* gvp = grad_vp + b * spatial_size;
    float* gvs = grad_vs + b * spatial_size;
    float* grho = grad_rho + b * spatial_size;

    float fvx_x = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(f.vx, ix, iz, grad_ctx, solver, true);
    float fvz_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float fvx_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float fvz_x = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>(f.vz, ix, iz, grad_ctx, solver, false);

    bool is_z_fs = elastic_is_top_free_surface_row(solver, ix, iz);
    bool is_x_fs = elastic_is_x_free_surface_col(solver, ix);
    float bar_sxx = is_x_fs ? 0.f : a.sxx[idx];               // x face zeroes sxx
    float bar_szz = is_z_fs ? 0.f : a.szz[idx];               // z face zeroes szz
    float bar_sxz = (is_z_fs || is_x_fs) ? 0.f : a.sxz[idx];  // any FS face zeroes sxz
    float grad_lambda, grad_mu;
    if (is_z_fs && is_x_fs) {
        grad_lambda = 0.f; grad_mu = 0.f;   // corner: sxx=szz=sxz=0
    } else if (is_z_fs || is_x_fs) {
        // Robertsson material derivative: z face bar_sxx*dvx_dx, x face bar_szz*dvz_dz.
        float vp2 = vp_b[idx] * vp_b[idx];
        float vs2 = vs_b[idx] * vs_b[idx];
        float vp4 = vp2 * vp2;
        float a = is_z_fs ? (bar_sxx * fvx_x) : (bar_szz * fvz_z);
        grad_lambda = a * 4.f * vs2 * vs2 / vp4;
        grad_mu     = a * 4.f * (vp4 - 2.f * vp2 * vs2 + 2.f * vs2 * vs2) / vp4;
    } else {
        grad_lambda = (bar_sxx + bar_szz) * (fvx_x + fvz_z);
        grad_mu = 2*(bar_sxx * fvx_x + bar_szz * fvz_z) + bar_sxz * (fvx_z + fvz_x);
    }

    gvp[idx] +=   -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    gvs[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                   2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

    grho[idx] += (a.vx[idx] * (f.vx[idx]-fvx_prev_b[idx]) + 
                  a.vz[idx] * (f.vz[idx]-fvz_prev_b[idx])) / rho_b[idx];
    grho[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                 grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;
}

template<int Order>
__global__ void calculate_grad_elastic_nobs(

    ElasticWavefieldPointer adjoint,

    const float* __restrict__ fvx,
    const float* __restrict__ fvz,

    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvz_prev,

    const float* __restrict__ vp,        // (B, nz, nx)
    const float* __restrict__ vs,        // (B, nz, nx)
    const float* __restrict__ rho,       // (B, nz, nx)

    float* __restrict__ grad_vp,         // (B, nz, nx)
    float* __restrict__ grad_vs,         // (B, nz, nx)
    float* __restrict__ grad_rho,         // (B, nz, nx)

    SGradParam grad_ctx,
    SolverContext solver
)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz)
        return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    // Even cells strictly above the per-column surface row ("air") can carry
    // nonzero gradient under irregular topography (see comment on
    // calculate_grad_elastic_bs counterpart): air-cell parameters couple to
    // solid neighbours through x-derivatives.  Don't skip them here.

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    const float* fvx_b = fvx + b * spatial_size;
    const float* fvz_b = fvz + b * spatial_size;
    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;

    float*       grad_vp_b       = grad_vp       + b * spatial_size;
    float*       grad_vs_b       = grad_vs       + b * spatial_size;
    float*       grad_rho_b      = grad_rho      + b * spatial_size;

    const float* vp_b         = vp         + b * spatial_size;
    const float* vs_b         = vs         + b * spatial_size;
    const float* rho_b        = rho        + b * spatial_size;

    auto a = adjoint.offset(b, spatial_size);

    float fvx_x = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(fvx_b, ix, iz, grad_ctx, solver, true);
    float fvz_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(fvz_b, ix, iz, grad_ctx, solver, true);
    float fvx_z = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (fvx_b, ix, iz, grad_ctx, solver, false);
    float fvz_x = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>(fvz_b, ix, iz, grad_ctx, solver, false);

    bool is_z_fs = elastic_is_top_free_surface_row(solver, ix, iz);
    bool is_x_fs = elastic_is_x_free_surface_col(solver, ix);
    float bar_sxx = is_x_fs ? 0.f : a.sxx[idx];
    float bar_szz = is_z_fs ? 0.f : a.szz[idx];
    float bar_sxz = (is_z_fs || is_x_fs) ? 0.f : a.sxz[idx];
    float grad_lambda, grad_mu;
    if (is_z_fs && is_x_fs) {
        grad_lambda = 0.f; grad_mu = 0.f;   // corner
    } else if (is_z_fs || is_x_fs) {
        // Robertsson material derivative: z face bar_sxx*dvx_dx, x face bar_szz*dvz_dz.
        float vp2 = vp_b[idx] * vp_b[idx];
        float vs2 = vs_b[idx] * vs_b[idx];
        float vp4 = vp2 * vp2;
        float a = is_z_fs ? (bar_sxx * fvx_x) : (bar_szz * fvz_z);
        grad_lambda = a * 4.f * vs2 * vs2 / vp4;
        grad_mu     = a * 4.f * (vp4 - 2.f * vp2 * vs2 + 2.f * vs2 * vs2) / vp4;
    } else {
        grad_lambda = (bar_sxx + bar_szz) * (fvx_x + fvz_z);
        grad_mu = 2*(bar_sxx * fvx_x + bar_szz * fvz_z) + bar_sxz * (fvx_z + fvz_x);
    }

    grad_vp_b[idx] += -2*rho_b[idx]*vp_b[idx]*grad_lambda* solver.dt;
    grad_vs_b[idx] += -(-4*rho_b[idx]*vs_b[idx]*grad_lambda +
                         2*rho_b[idx]*vs_b[idx]*grad_mu)* solver.dt;

    grad_rho_b[idx] += (a.vx[idx] * (fvx_b[idx]-fvx_prev_b[idx]) +
                        a.vz[idx] * (fvz_b[idx]-fvz_prev_b[idx])) / rho_b[idx];
    grad_rho_b[idx] -= grad_lambda * (vp_b[idx]*vp_b[idx] - 2*vs_b[idx]*vs_b[idx])* solver.dt +
                       grad_mu     * (vs_b[idx]*vs_b[idx]) * solver.dt;

}


// ===========================================================================
// APM (Cao & Chen 2018) variants
// ===========================================================================
// Category codes (mirror sweep.equations._topography):
//   INTERIOR = 0   no modification
//   AIR      = 1   all 5 wavefields zeroed every step
//   H        = 2   σ_zz = 0  (horizontal top surface)
//   VL       = 3   σ_xx = 0  (vertical left surface)
//   VR       = 4   σ_xx = 0  (vertical right surface)
//   OC       = 5   σ_xx = σ_zz = σ_xz = 0  (outer / convex corner)
//   IC       = 6   no modification
//
// The APM step uses the standard Virieux stencil with parameter-modified
// moduli (λ_eff, μ_eff, μ_xz_node, ρ_x_eff, ρ_z_eff) precomputed on the
// Python side and passed in via the model tensors.  Image-method
// z-derivative substitutions are NOT engaged here — the propagator runs
// with ``solver.free_surface=false`` for the APM path, so the existing
// ``elastic_top_fs_sgradient_z_2d`` helpers fall through to the plain
// ``sgradient`` call.

#define APM_CATEGORY_INTERIOR 0
#define APM_CATEGORY_AIR      1
#define APM_CATEGORY_H        2
#define APM_CATEGORY_VL       3
#define APM_CATEGORY_VR       4
#define APM_CATEGORY_OC       5
#define APM_CATEGORY_IC       6

template<int Order>
__global__ void elastic_velocity_kernel_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho_x,
    const float* __restrict__ rho_z,
    const int*   __restrict__ category,

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

    // AIR cell: wipe velocity components and skip the update.  Mirrors
    // Python's ``zero_at_air`` (applied after the step in eager; baked
    // into the kernel here).
    if (category[idx] == APM_CATEGORY_AIR) {
        f.vx[idx] = 0.f;
        f.vz[idx] = 0.f;
        return;
    }

    const float* rho_x_b = rho_x + b * spatial_size;
    const float* rho_z_b = rho_z + b * spatial_size;

    // No image-method z-derivative substitution: solver.free_surface is
    // false in APM mode, so these helpers route to the plain sgradient.
    float dsxx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD> (f.sxx, ix, iz, grad_ctx, solver, true);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.szz, ix, iz, grad_ctx, solver, true);

    float inv_rho_x = 1.f / rho_x_b[idx];
    float inv_rho_z = 1.f / rho_z_b[idx];

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.vx[idx] += solver.dt * inv_rho_x * (dsxx_dx + dsxz_dz);
        f.vz[idx] += solver.dt * inv_rho_z * (dsxz_dx + dszz_dz);
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];

    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    f.m_szzz[idx] = azh * f.m_szzz[idx] + bzh * dszz_dz;
    dszz_dz += f.m_szzz[idx];
    f.m_sxzx[idx] = ax * f.m_sxzx[idx] + bx * dsxz_dx;
    dsxz_dx += f.m_sxzx[idx];

    f.m_sxzz[idx] = az * f.m_sxzz[idx] + bz * dsxz_dz;
    dsxz_dz += f.m_sxzz[idx];
    f.m_sxxx[idx] = axh * f.m_sxxx[idx] + bxh * dsxx_dx;
    dsxx_dx += f.m_sxxx[idx];

    f.vx[idx] += solver.dt * inv_rho_x * (dsxx_dx + dsxz_dz);
    f.vz[idx] += solver.dt * inv_rho_z * (dsxz_dx + dszz_dz);
}


template<int Order>
__global__ void elastic_stress_kernel_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lam_eff,
    const float* __restrict__ mu_eff,
    const float* __restrict__ mu_xz_node,
    const int*   __restrict__ category,
    float* __restrict__ u_this,

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
    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    int cat = category[idx];

    // AIR cell: wipe all 3 stress components.  We don't return early
    // because we still want to checkpoint vx/vz under save_all_wavefields
    // (they were zeroed by the velocity kernel).
    if (cat == APM_CATEGORY_AIR) {
        f.sxx[idx] = 0.f;
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
        if (u_this_b) {
            int comp_stride  = solver.B * spatial_size;
            u_this_b[0 * comp_stride + idx] = 0.f;
            u_this_b[1 * comp_stride + idx] = 0.f;
        }
        return;
    }

    const float* lam_b   = lam_eff    + b * spatial_size;
    const float* mu_b    = mu_eff     + b * spatial_size;
    const float* muxz_b  = mu_xz_node + b * spatial_size;

    float dvx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD> (f.vx, ix, iz, grad_ctx, solver, true);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>  (f.vz, ix, iz, grad_ctx, solver, false);

    float lam   = lam_b[idx];
    float mu_   = mu_b[idx];
    float muxz  = muxz_b[idx];

    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < solver.abcn + halo) || (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.sxx[idx] += solver.dt * ((lam + 2.f*mu_) * dvx_dx + lam * dvz_dz);
        f.szz[idx] += solver.dt * ((lam + 2.f*mu_) * dvz_dz + lam * dvx_dx);
        f.sxz[idx] += solver.dt * muxz * (dvx_dz + dvz_dx);
    } else {
        float az = cpml.az[iz];
        float bz = cpml.bz[iz];
        float azh = cpml.azh[iz];
        float bzh = cpml.bzh[iz];

        float ax = cpml.ax[ix];
        float bx = cpml.bx[ix];
        float axh = cpml.axh[ix];
        float bxh = cpml.bxh[ix];

        f.m_vzz[idx] = az * f.m_vzz[idx] + bz * dvz_dz;
        dvz_dz += f.m_vzz[idx];
        f.m_vxx[idx] = ax * f.m_vxx[idx] + bx * dvx_dx;
        dvx_dx += f.m_vxx[idx];

        f.sxx[idx] += solver.dt * ((lam + 2.f*mu_) * dvx_dx + lam * dvz_dz);
        f.szz[idx] += solver.dt * ((lam + 2.f*mu_) * dvz_dz + lam * dvx_dx);

        f.m_vxz[idx] = azh * f.m_vxz[idx] + bzh * dvx_dz;
        dvx_dz += f.m_vxz[idx];
        f.m_vzx[idx] = axh * f.m_vzx[idx] + bxh * dvz_dx;
        dvz_dx += f.m_vzx[idx];

        f.sxz[idx] += solver.dt * muxz * (dvx_dz + dvz_dx);
    }

    // APM traction-free BC: pointwise stress zero per surface category.
    // Mirror Python's ``enforce_apm_traction_bc``.
    if (cat == APM_CATEGORY_H) {
        f.szz[idx] = 0.f;
    } else if (cat == APM_CATEGORY_VL || cat == APM_CATEGORY_VR) {
        f.sxx[idx] = 0.f;
    } else if (cat == APM_CATEGORY_OC) {
        f.sxx[idx] = 0.f;
        f.szz[idx] = 0.f;
        f.sxz[idx] = 0.f;
    }

    if (u_this_b) {
        int comp_stride  = solver.B * spatial_size;
        u_this_b[0 * comp_stride + idx] = f.vx[idx];
        u_this_b[1 * comp_stride + idx] = f.vz[idx];
    }
}


#define LAUNCH_ELASTIC_VELOCITY_APM(order, grid, block, ...)                 \
    do {                                                                      \
        if      ((order) == 2) elastic_velocity_kernel_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_APM(order, grid, block, ...)                   \
    do {                                                                      \
        if      ((order) == 2) elastic_stress_kernel_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)


// ===========================================================================
// APM backward kernels
// ===========================================================================
// Mirror image-method backward kernels (stress_adjoint_prepare,
// velocity_adjoint_prepare, calculate_grad_elastic_bs/nobs and the
// _nopml variants used in bs-mode reverse replay), but consume the APM
// effective moduli (lam_eff, mu_eff, mu_xz_node, rho_x_eff, rho_z_eff)
// and chain gradients back to raw (lam, mu, rho) -> (vp, vs, rho) per
// the 6-category pointwise Jacobian table from
// ``sweep.equations._topography.precompute_apm_moduli``.
//
// The ``_adjoint_apply`` kernels (stress and velocity) don't touch
// moduli, so they're reused unchanged.

// --- Reverse-time forward replay (used in apm_backward_bs) -----------------

template<int Order>
__global__ void elastic_velocity_kernel_nopml_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho_x,
    const float* __restrict__ rho_z,
    const int*   __restrict__ category,
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
    int top_halo = solver.free_surface ? M : halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);

    if (category[idx] == APM_CATEGORY_AIR) {
        // air cell wavefield stays zero (forward kernel set it to 0)
        f.vx[idx] = 0.f;
        f.vz[idx] = 0.f;
        return;
    }

    const float* rho_x_b = rho_x + b * spatial_size;
    const float* rho_z_b = rho_z + b * spatial_size;

    float dsxx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD> (f.sxx, ix, iz, grad_ctx, solver, true);
    float dsxz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dsxz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD>(f.sxz, ix, iz, grad_ctx, solver, true);
    float dszz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.szz, ix, iz, grad_ctx, solver, true);

    float inv_rho_x = 1.f / rho_x_b[idx];
    float inv_rho_z = 1.f / rho_z_b[idx];

    // Reverse step: subtract instead of add (mirror forward APM with sign flip)
    f.vx[idx] -= solver.dt * inv_rho_x * (dsxx_dx + dsxz_dz);
    f.vz[idx] -= solver.dt * inv_rho_z * (dsxz_dx + dszz_dz);
}

template<int Order>
__global__ void elastic_stress_kernel_nopml_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lam_eff,
    const float* __restrict__ mu_eff,
    const float* __restrict__ mu_xz_node,
    const int*   __restrict__ category,
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
    int top_halo = solver.free_surface ? M : halo;
    if (ix < halo || ix >= solver.nx - halo || iz < top_halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    int cat = category[idx];

    if (cat == APM_CATEGORY_AIR) {
        f.sxx[idx] = 0.f; f.szz[idx] = 0.f; f.sxz[idx] = 0.f;
        return;
    }

    const float* lam_b  = lam_eff    + b * spatial_size;
    const float* mu_b   = mu_eff     + b * spatial_size;
    const float* muxz_b = mu_xz_node + b * spatial_size;

    float dvx_dx = elastic_fs_sgradient_x_2d<Order, DIFF_BACKWARD> (f.vx, ix, iz, grad_ctx, solver, true);
    float dvz_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_BACKWARD>(f.vz, ix, iz, grad_ctx, solver, true);
    float dvx_dz = elastic_top_fs_sgradient_z_2d<Order, DIFF_FORWARD> (f.vx, ix, iz, grad_ctx, solver, false);
    float dvz_dx = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>  (f.vz, ix, iz, grad_ctx, solver, false);

    float lam  = lam_b[idx];
    float mu_  = mu_b[idx];
    float muxz = muxz_b[idx];

    // Reverse step: subtract
    f.sxx[idx] -= solver.dt * ((lam + 2.f * mu_) * dvx_dx + lam * dvz_dz);
    f.szz[idx] -= solver.dt * ((lam + 2.f * mu_) * dvz_dz + lam * dvx_dx);
    f.sxz[idx] -= solver.dt * muxz * (dvx_dz + dvz_dx);

    // Re-apply traction BC (forward kernel zeroed these after stress update;
    // reverse replay must keep the same invariant on the forward state).
    if (cat == APM_CATEGORY_H) {
        f.szz[idx] = 0.f;
    } else if (cat == APM_CATEGORY_VL || cat == APM_CATEGORY_VR) {
        f.sxx[idx] = 0.f;
    } else if (cat == APM_CATEGORY_OC) {
        f.sxx[idx] = 0.f; f.szz[idx] = 0.f; f.sxz[idx] = 0.f;
    }
}

#define LAUNCH_ELASTIC_VELOCITY_NOPML_APM(order, grid, block, ...)               \
    do {                                                                          \
        if      ((order) == 2) elastic_velocity_kernel_nopml_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_kernel_nopml_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_kernel_nopml_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_kernel_nopml_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_kernel_nopml_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_STRESS_NOPML_APM(order, grid, block, ...)                 \
    do {                                                                          \
        if      ((order) == 2) elastic_stress_kernel_nopml_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_kernel_nopml_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_kernel_nopml_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_kernel_nopml_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_kernel_nopml_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

// --- Adjoint prepare kernels (APM) -----------------------------------------

template<int Order>
__global__ void elastic_stress_adjoint_prepare_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ lam_eff,
    const float* __restrict__ mu_eff,
    const float* __restrict__ mu_xz_node,
    const int*   __restrict__ category,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ qxx,
    float* __restrict__ qzz,
    float* __restrict__ qxz,
    float* __restrict__ qzx
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;

    auto f = wf.offset(b, spatial_size);
    float* qxx_b = qxx + b * spatial_size;
    float* qzz_b = qzz + b * spatial_size;
    float* qxz_b = qxz + b * spatial_size;
    float* qzx_b = qzx + b * spatial_size;

    int cat = category[idx];

    float bar_sxx = f.sxx[idx];
    float bar_szz = f.szz[idx];
    float bar_sxz = f.sxz[idx];

    // APM traction-BC adjoint: forward sets these to 0 AFTER stress update
    // for certain categories — so the adjoint of "set to 0" is "zero the
    // bar_s* contribution" before propagating back into v.
    if (cat == APM_CATEGORY_H) {
        bar_szz = 0.f; f.szz[idx] = 0.f;
    } else if (cat == APM_CATEGORY_VL || cat == APM_CATEGORY_VR) {
        bar_sxx = 0.f; f.sxx[idx] = 0.f;
    } else if (cat == APM_CATEGORY_OC || cat == APM_CATEGORY_AIR) {
        bar_sxx = 0.f; bar_szz = 0.f; bar_sxz = 0.f;
        f.sxx[idx] = 0.f; f.szz[idx] = 0.f; f.sxz[idx] = 0.f;
    }

    const float* lam_b  = lam_eff    + b * spatial_size;
    const float* mu_b   = mu_eff     + b * spatial_size;
    const float* muxz_b = mu_xz_node + b * spatial_size;

    float lam  = lam_b[idx];
    float mu_  = mu_b[idx];
    float muxz = muxz_b[idx];

    float bar_dvx_dx = solver.dt * ((lam + 2.f * mu_) * bar_sxx + lam * bar_szz);
    float bar_dvz_dz = solver.dt * ((lam + 2.f * mu_) * bar_szz + lam * bar_sxx);
    float bar_dvx_dz = solver.dt * muxz * bar_sxz;
    float bar_dvz_dx = solver.dt * muxz * bar_sxz;

    bool in_pml = (ix < solver.padLo(2) + halo) || (ix >= solver.nx - solver.padHi(2) - halo) ||
                  (iz < solver.padLo(0) + halo) ||
                  (iz >= solver.nz - solver.padHi(0) - halo);
    if (!in_pml) {
        qxx_b[idx] = bar_dvx_dx;
        qzz_b[idx] = bar_dvz_dz;
        qxz_b[idx] = bar_dvx_dz;
        qzx_b[idx] = bar_dvz_dx;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float tmp_vxx = f.m_vxx[idx] + bar_dvx_dx;
    float tmp_vzz = f.m_vzz[idx] + bar_dvz_dz;
    float tmp_vxz = f.m_vxz[idx] + bar_dvx_dz;
    float tmp_vzx = f.m_vzx[idx] + bar_dvz_dx;

    qxx_b[idx] = bar_dvx_dx + bx  * tmp_vxx;
    qzz_b[idx] = bar_dvz_dz + bz  * tmp_vzz;
    qxz_b[idx] = bar_dvx_dz + bzh * tmp_vxz;
    qzx_b[idx] = bar_dvz_dx + bxh * tmp_vzx;

    f.m_vxx[idx] = ax  * tmp_vxx;
    f.m_vzz[idx] = az  * tmp_vzz;
    f.m_vxz[idx] = azh * tmp_vxz;
    f.m_vzx[idx] = axh * tmp_vzx;
}

template<int Order>
__global__ void elastic_velocity_adjoint_prepare_apm(
    ElasticWavefieldPointer wf,
    const float* __restrict__ rho_x,
    const float* __restrict__ rho_z,
    const int*   __restrict__ category,
    ElasticCPMLPointer cpml,
    SolverContext solver,
    float* __restrict__ pxx,
    float* __restrict__ pzz,
    float* __restrict__ pxz,
    float* __restrict__ pzx
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= solver.nx || iz >= solver.nz) return;

    constexpr bool is_runtime = (Order == -1);
    constexpr int  M_static   = is_runtime ? 0 : (Order / 2);
    int halo = is_runtime ? solver.M : M_static;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    int cat = category[idx];

    float bar_vx = f.vx[idx];
    float bar_vz = f.vz[idx];
    if (cat == APM_CATEGORY_AIR) {
        bar_vx = 0.f; bar_vz = 0.f;
        f.vx[idx] = 0.f; f.vz[idx] = 0.f;
    }

    const float* rho_x_b = rho_x + b * spatial_size;
    const float* rho_z_b = rho_z + b * spatial_size;
    float inv_rho_x = 1.f / rho_x_b[idx];
    float inv_rho_z = 1.f / rho_z_b[idx];

    float* pxx_b = pxx + b * spatial_size;
    float* pzz_b = pzz + b * spatial_size;
    float* pxz_b = pxz + b * spatial_size;
    float* pzx_b = pzx + b * spatial_size;

    float bar_dsxx_dx = solver.dt * inv_rho_x * bar_vx;
    float bar_dsxz_dz = solver.dt * inv_rho_x * bar_vx;
    float bar_dsxz_dx = solver.dt * inv_rho_z * bar_vz;
    float bar_dszz_dz = solver.dt * inv_rho_z * bar_vz;

    bool in_pml = (ix < solver.padLo(2) + halo) || (ix >= solver.nx - solver.padHi(2) - halo) ||
                  (iz < solver.padLo(0) + halo) ||
                  (iz >= solver.nz - solver.padHi(0) - halo);
    if (!in_pml) {
        pxx_b[idx] = bar_dsxx_dx;
        pxz_b[idx] = bar_dsxz_dz;
        pzx_b[idx] = bar_dsxz_dx;
        pzz_b[idx] = bar_dszz_dz;
        return;
    }

    float az = cpml.az[iz];
    float bz = cpml.bz[iz];
    float azh = cpml.azh[iz];
    float bzh = cpml.bzh[iz];
    float ax = cpml.ax[ix];
    float bx = cpml.bx[ix];
    float axh = cpml.axh[ix];
    float bxh = cpml.bxh[ix];

    float tmp_sxxx = f.m_sxxx[idx] + bar_dsxx_dx;
    float tmp_sxzz = f.m_sxzz[idx] + bar_dsxz_dz;
    float tmp_sxzx = f.m_sxzx[idx] + bar_dsxz_dx;
    float tmp_szzz = f.m_szzz[idx] + bar_dszz_dz;

    pxx_b[idx] = bar_dsxx_dx + bxh * tmp_sxxx;
    pxz_b[idx] = bar_dsxz_dz + bz  * tmp_sxzz;
    pzx_b[idx] = bar_dsxz_dx + bx  * tmp_sxzx;
    pzz_b[idx] = bar_dszz_dz + bzh * tmp_szzz;

    f.m_sxxx[idx] = axh * tmp_sxxx;
    f.m_sxzz[idx] = az  * tmp_sxzz;
    f.m_sxzx[idx] = ax  * tmp_sxzx;
    f.m_szzz[idx] = azh * tmp_szzz;
}

#define LAUNCH_ELASTIC_STRESS_ADJOINT_PREPARE_APM(order, grid, block, ...)       \
    do {                                                                          \
        if      ((order) == 2) elastic_stress_adjoint_prepare_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_stress_adjoint_prepare_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_stress_adjoint_prepare_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_stress_adjoint_prepare_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_stress_adjoint_prepare_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE_APM(order, grid, block, ...)     \
    do {                                                                          \
        if      ((order) == 2) elastic_velocity_adjoint_prepare_apm<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) elastic_velocity_adjoint_prepare_apm<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) elastic_velocity_adjoint_prepare_apm<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) elastic_velocity_adjoint_prepare_apm<8><<<grid, block>>>(__VA_ARGS__); \
        else                   elastic_velocity_adjoint_prepare_apm<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

// --- Gradient kernels (APM) ------------------------------------------------
// Chain (grad_lam_eff, grad_mu_eff, grad_mu_xz, grad_rho_x, grad_rho_z) ->
// (grad_lam, grad_mu, grad_rho) -> (grad_vp, grad_vs, grad_rho) per
// _topography.precompute_apm_moduli's pointwise category Jacobian.
//
// Per-category Jacobians (λ_eff, μ_eff, μ_xz, ρ_x, ρ_z) wrt (λ_raw, μ_raw, ρ):
//   INTERIOR: (λ, μ, μ, ρ, ρ)         — identity for λ,μ,ρ
//   IC      : (λ, μ, μ, 0.75ρ, 0.75ρ) — same stiffness; ρ scaled
//   H       : (0, α/2, μ/2, 0.5ρ, ρ)
//   VL      : (0, α/2, μ/2, ρ,   0.5ρ)
//   VR      : (0, α/2, μ/2, 0.5ρ,0.5ρ)
//   OC      : (0, 0,   0,   0.25ρ, 0.25ρ)
//   AIR     : (0, 0,   0,   ρ,   ρ)
//
//   α = 2μ(λ+μ)/(λ+2μ);  ∂(α/2)/∂λ = μ²/(λ+2μ)²;
//   ∂(α/2)/∂μ = (λ² + 2λμ + 2μ²)/(λ+2μ)²
//
// Then standard (λ, μ, ρ) -> (vp, vs, ρ) chain (λ = ρ(vp²-2vs²), μ = ρvs²).

__device__ __forceinline__ void apm_chain_lammu(
    int cat, float lam_r, float mu_r,
    float grad_lam_eff, float grad_mu_eff, float grad_mu_xz,
    float* grad_lam_out, float* grad_mu_out)
{
    if (cat == APM_CATEGORY_INTERIOR || cat == APM_CATEGORY_IC) {
        *grad_lam_out = grad_lam_eff;
        *grad_mu_out  = grad_mu_eff + grad_mu_xz;
    } else if (cat == APM_CATEGORY_H || cat == APM_CATEGORY_VL || cat == APM_CATEGORY_VR) {
        float denom = lam_r + 2.f * mu_r;
        float safe = (denom > 0.f) ? denom : 1.f;
        float inv2 = 1.f / (safe * safe);
        float dh_dlam = mu_r * mu_r * inv2;
        float dh_dmu  = (lam_r * lam_r + 2.f * lam_r * mu_r + 2.f * mu_r * mu_r) * inv2;
        *grad_lam_out = grad_mu_eff * dh_dlam;
        *grad_mu_out  = grad_mu_eff * dh_dmu + grad_mu_xz * 0.5f;
    } else {  // OC, AIR
        *grad_lam_out = 0.f;
        *grad_mu_out  = 0.f;
    }
}

__device__ __forceinline__ void apm_rho_jacobian(int cat, float* drho_x_drho, float* drho_z_drho)
{
    switch (cat) {
        case APM_CATEGORY_INTERIOR:
        case APM_CATEGORY_AIR:
            *drho_x_drho = 1.f; *drho_z_drho = 1.f; break;
        case APM_CATEGORY_H:
            *drho_x_drho = 0.5f; *drho_z_drho = 1.f; break;
        case APM_CATEGORY_VL:
            *drho_x_drho = 1.f;  *drho_z_drho = 0.5f; break;
        case APM_CATEGORY_VR:
            *drho_x_drho = 0.5f; *drho_z_drho = 0.5f; break;
        case APM_CATEGORY_OC:
            *drho_x_drho = 0.25f; *drho_z_drho = 0.25f; break;
        case APM_CATEGORY_IC:
            *drho_x_drho = 0.75f; *drho_z_drho = 0.75f; break;
        default:
            *drho_x_drho = 1.f; *drho_z_drho = 1.f; break;
    }
}

template<int Order>
__global__ void calculate_grad_elastic_apm_bs(
    ElasticWavefieldPointer forward,
    ElasticWavefieldPointer adjoint,
    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvz_prev,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ rho,
    const float* __restrict__ lam_raw,
    const float* __restrict__ mu_raw,
    const float* __restrict__ rho_x,
    const float* __restrict__ rho_z,
    const int*   __restrict__ category,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_vs,
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
    int cat = category[idx];

    auto f = forward.offset(b, spatial_size);
    auto a = adjoint.offset(b, spatial_size);

    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;
    const float* vp_b   = vp   + b * spatial_size;
    const float* vs_b   = vs   + b * spatial_size;
    const float* rho_b  = rho  + b * spatial_size;
    const float* lam_b  = lam_raw + b * spatial_size;
    const float* mu_b   = mu_raw  + b * spatial_size;
    const float* rhox_b = rho_x + b * spatial_size;
    const float* rhoz_b = rho_z + b * spatial_size;

    float* gvp  = grad_vp  + b * spatial_size;
    float* gvs  = grad_vs  + b * spatial_size;
    float* grho = grad_rho + b * spatial_size;

    float fvx_x = sgradient<2, Order, X, DIFF_BACKWARD>(f.vx, ix, 0, iz, grad_ctx);
    float fvz_z = sgradient<2, Order, Z, DIFF_BACKWARD>(f.vz, ix, 0, iz, grad_ctx);
    float fvx_z = sgradient<2, Order, Z, DIFF_FORWARD> (f.vx, ix, 0, iz, grad_ctx);
    float fvz_x = elastic_fs_sgradient_x_2d<Order, DIFF_FORWARD>(f.vz, ix, iz, grad_ctx, solver, false);

    // Adjoint stresses with APM traction-BC adjoint applied
    float bar_sxx = a.sxx[idx];
    float bar_szz = a.szz[idx];
    float bar_sxz = a.sxz[idx];
    if (cat == APM_CATEGORY_H) {
        bar_szz = 0.f;
    } else if (cat == APM_CATEGORY_VL || cat == APM_CATEGORY_VR) {
        bar_sxx = 0.f;
    } else if (cat == APM_CATEGORY_OC || cat == APM_CATEGORY_AIR) {
        bar_sxx = 0.f; bar_szz = 0.f; bar_sxz = 0.f;
    }

    // Gradients w.r.t. effective moduli (no dt — chained later)
    float grad_lam_eff = (bar_sxx + bar_szz) * (fvx_x + fvz_z);
    float grad_mu_eff_local = 2.f * (bar_sxx * fvx_x + bar_szz * fvz_z);
    float grad_mu_xz_local = bar_sxz * (fvx_z + fvz_x);

    float lam_v = lam_b[idx];
    float mu_v  = mu_b[idx];

    float grad_lam = 0.f, grad_mu = 0.f;
    apm_chain_lammu(cat, lam_v, mu_v,
                    grad_lam_eff, grad_mu_eff_local, grad_mu_xz_local,
                    &grad_lam, &grad_mu);

    // Kinetic ρ term: structure mirrors image but split by rho_x/rho_z.
    float grad_rho_x_kin = a.vx[idx] * (f.vx[idx] - fvx_prev_b[idx]) / rhox_b[idx];
    float grad_rho_z_kin = a.vz[idx] * (f.vz[idx] - fvz_prev_b[idx]) / rhoz_b[idx];

    float drho_x_drho, drho_z_drho;
    apm_rho_jacobian(cat, &drho_x_drho, &drho_z_drho);
    float grad_rho_kin = grad_rho_x_kin * drho_x_drho + grad_rho_z_kin * drho_z_drho;

    float vp_v = vp_b[idx];
    float vs_v = vs_b[idx];
    float rho_v = rho_b[idx];

    gvp[idx]  += -2.f * rho_v * vp_v * grad_lam * solver.dt;
    gvs[idx]  += -(-4.f * rho_v * vs_v * grad_lam + 2.f * rho_v * vs_v * grad_mu) * solver.dt;
    grho[idx] += grad_rho_kin
                 - (grad_lam * (vp_v * vp_v - 2.f * vs_v * vs_v)
                    + grad_mu * (vs_v * vs_v)) * solver.dt;
}

template<int Order>
__global__ void calculate_grad_elastic_apm_nobs(
    ElasticWavefieldPointer adjoint,
    const float* __restrict__ fvx,
    const float* __restrict__ fvz,
    const float* __restrict__ fvx_prev,
    const float* __restrict__ fvz_prev,
    const float* __restrict__ vp,
    const float* __restrict__ vs,
    const float* __restrict__ rho,
    const float* __restrict__ lam_raw,
    const float* __restrict__ mu_raw,
    const float* __restrict__ rho_x,
    const float* __restrict__ rho_z,
    const int*   __restrict__ category,
    float* __restrict__ grad_vp,
    float* __restrict__ grad_vs,
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
    int cat = category[idx];

    auto a = adjoint.offset(b, spatial_size);
    const float* fvx_b = fvx + b * spatial_size;
    const float* fvz_b = fvz + b * spatial_size;
    const float* fvx_prev_b = fvx_prev + b * spatial_size;
    const float* fvz_prev_b = fvz_prev + b * spatial_size;
    const float* vp_b   = vp   + b * spatial_size;
    const float* vs_b   = vs   + b * spatial_size;
    const float* rho_b  = rho  + b * spatial_size;
    const float* lam_b  = lam_raw + b * spatial_size;
    const float* mu_b   = mu_raw  + b * spatial_size;
    const float* rhox_b = rho_x + b * spatial_size;
    const float* rhoz_b = rho_z + b * spatial_size;

    float* gvp_b  = grad_vp  + b * spatial_size;
    float* gvs_b  = grad_vs  + b * spatial_size;
    float* grho_b = grad_rho + b * spatial_size;

    float fvx_x = sgradient<2, Order, X, DIFF_BACKWARD>(fvx_b, ix, 0, iz, grad_ctx);
    float fvz_z = sgradient<2, Order, Z, DIFF_BACKWARD>(fvz_b, ix, 0, iz, grad_ctx);
    float fvx_z = sgradient<2, Order, Z, DIFF_FORWARD> (fvx_b, ix, 0, iz, grad_ctx);
    float fvz_x = sgradient<2, Order, X, DIFF_FORWARD> (fvz_b, ix, 0, iz, grad_ctx);

    float bar_sxx = a.sxx[idx];
    float bar_szz = a.szz[idx];
    float bar_sxz = a.sxz[idx];
    if (cat == APM_CATEGORY_H) {
        bar_szz = 0.f;
    } else if (cat == APM_CATEGORY_VL || cat == APM_CATEGORY_VR) {
        bar_sxx = 0.f;
    } else if (cat == APM_CATEGORY_OC || cat == APM_CATEGORY_AIR) {
        bar_sxx = 0.f; bar_szz = 0.f; bar_sxz = 0.f;
    }

    float grad_lam_eff = (bar_sxx + bar_szz) * (fvx_x + fvz_z);
    float grad_mu_eff_local = 2.f * (bar_sxx * fvx_x + bar_szz * fvz_z);
    float grad_mu_xz_local = bar_sxz * (fvx_z + fvz_x);

    float lam_v = lam_b[idx];
    float mu_v  = mu_b[idx];

    float grad_lam = 0.f, grad_mu = 0.f;
    apm_chain_lammu(cat, lam_v, mu_v,
                    grad_lam_eff, grad_mu_eff_local, grad_mu_xz_local,
                    &grad_lam, &grad_mu);

    float grad_rho_x_kin = a.vx[idx] * (fvx_b[idx] - fvx_prev_b[idx]) / rhox_b[idx];
    float grad_rho_z_kin = a.vz[idx] * (fvz_b[idx] - fvz_prev_b[idx]) / rhoz_b[idx];

    float drho_x_drho, drho_z_drho;
    apm_rho_jacobian(cat, &drho_x_drho, &drho_z_drho);
    float grad_rho_kin = grad_rho_x_kin * drho_x_drho + grad_rho_z_kin * drho_z_drho;

    float vp_v = vp_b[idx];
    float vs_v = vs_b[idx];
    float rho_v = rho_b[idx];

    gvp_b[idx]  += -2.f * rho_v * vp_v * grad_lam * solver.dt;
    gvs_b[idx]  += -(-4.f * rho_v * vs_v * grad_lam + 2.f * rho_v * vs_v * grad_mu) * solver.dt;
    grho_b[idx] += grad_rho_kin
                   - (grad_lam * (vp_v * vp_v - 2.f * vs_v * vs_v)
                      + grad_mu * (vs_v * vs_v)) * solver.dt;
}

#define LAUNCH_CALCULATE_GRAD_ELASTIC_APM_BS(order, grid, block, ...)            \
    do {                                                                          \
        if      ((order) == 2) calculate_grad_elastic_apm_bs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_apm_bs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_apm_bs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_apm_bs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_apm_bs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)

#define LAUNCH_CALCULATE_GRAD_ELASTIC_APM_NOBS(order, grid, block, ...)          \
    do {                                                                          \
        if      ((order) == 2) calculate_grad_elastic_apm_nobs<2><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 4) calculate_grad_elastic_apm_nobs<4><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 6) calculate_grad_elastic_apm_nobs<6><<<grid, block>>>(__VA_ARGS__); \
        else if ((order) == 8) calculate_grad_elastic_apm_nobs<8><<<grid, block>>>(__VA_ARGS__); \
        else                   calculate_grad_elastic_apm_nobs<-1><<<grid, block>>>(__VA_ARGS__);\
    } while (0)
