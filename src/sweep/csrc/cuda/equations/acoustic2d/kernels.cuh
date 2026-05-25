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

// Pre-pass: clear air cells (above per-column topo surface).  Launched
// BEFORE acoustic2nd so the main kernel can early-return on air cells
// without writing any aux field — eliminating intra-launch RAW races
// between air-zeroing and PML stencil reads.  See sweep VTI history.
static __global__ void acoustic2d_air_clear_kernel(
    AcousticWavefieldPointer wf,
    bool save_all_wavefields,
    float* __restrict__ u_this,
    SolverContext solver
){
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= solver.nx || iz >= solver.nz) return;
    if (!solver.has_topo) return;
    if (iz >= solver.topo_rows[ix]) return;
    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    auto f = wf.offset(b, spatial_size);
    f.u_next[idx] = 0.f;
    f.psix[idx] = 0.f; f.psiz[idx] = 0.f;
    f.zetax[idx] = 0.f; f.zetaz[idx] = 0.f;
    if (u_this) u_this[b * spatial_size + idx] = 0.f;
}

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

    // Irregular free-surface topography (vacuum staircase / image method):
    // any cell strictly above the per-column surface row is air.  Mirror
    // Python's ``zero_above_topo`` — clear the wavefield and any CPML aux
    // fields, then skip the FD update.  Solid cells just below the surface
    // see these zeros through the stencil, which is what reproduces the
    // ``p=0`` boundary condition.
    // Air cells were already cleared by acoustic2d_air_clear_kernel (a
    // separate pre-pass launch).  Just early-return here — no writes —
    // so this kernel only stencil-reads psix/psiz that the prior launch
    // finished writing.
    if (solver.has_topo && iz < solver.topo_rows[ix]) {
        return;
    }

    float* u_this_b = u_this ? u_this + b * spatial_size : nullptr;
    const float* vp_b = vp + b * spatial_size;

    float lap_x = laplace<2, Order, X>(f.u_now, ix, 0, iz, lap_ctx);
    float lap_z = laplace<2, Order, Z>(f.u_now, ix, 0, iz, lap_ctx);

    float v  = vp_b[idx];
    float v2_dt2 = (v * v) * solver.dt * solver.dt;

    // Position-based PML / interior split. ax/bx/dbxdx coefficient arrays are
    // exactly zero outside the PML band, and the centered gradient of those
    // arrays vanishes once the stencil clears the band (>= abcn + halo from
    // the edge). Skipping the full PML update there is bit-equivalent and
    // avoids ~8 aux-field loads/stores per cell. The check is warp-coherent
    // (same outcome for 32 consecutive ix values), so warps diverge only at
    // the abcn boundary.
    bool in_pml = (ix < solver.abcn + halo) || (ix >= solver.nx - solver.abcn - halo) ||
                  (iz < (solver.free_surface ? halo : solver.abcn + halo)) ||
                  (iz >= solver.nz - solver.abcn - halo);

    if (!in_pml) {
        f.u_next[idx] = 2.0f * f.u_now[idx] - f.u_prev[idx] +
                        v2_dt2 * (lap_x + lap_z);
        if (u_this_b != nullptr)
            u_this_b[idx] = (v * v) * (lap_x + lap_z);
        return;
    }

    float ax_ = cpml.ax[ix];
    float az_ = cpml.az[iz];
    float bx_ = cpml.bx[ix];
    float bz_ = cpml.bz[iz];
    float dbxdx_ = cpml.dbxdx[ix];
    float dbzdz_ = cpml.dbzdz[iz];

    float w_sum = 0.0f;

    float dudz     = gradient<2, Order, Z>(f.u_now, ix, 0, iz, grad_ctx);
    float dudx     = gradient<2, Order, X>(f.u_now, ix, 0, iz, grad_ctx);
    float dpsizdz  = gradient<2, Order, Z>(f.psiz, ix, 0, iz, grad_ctx);
    float dpsixdx  = gradient<2, Order, X>(f.psix, ix, 0, iz, grad_ctx);
    float daxdx    = gradient<2, Order, X>(cpml.ax, ix, 0, 0, grad_ctx_x);
    float dazdz    = gradient<2, Order, X>(cpml.az, iz, 0, 0, grad_ctx_z);
    float daipsiz_dz = az_ * dpsizdz + dazdz * f.psiz[idx];
    float daipxix_dx = ax_ * dpsixdx + daxdx * f.psix[idx];

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

    f.u_next[idx] =
        2.0f * f.u_now[idx] -
        f.u_prev[idx] +
        v2_dt2 * w_sum;

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

    // int halo = solver.abcn > 0 ? solver.abcn + 2*M+1 : 2*M;
    int halo = solver.abcn > 0 ? solver.abcn + 2*M+1 : 2*M;

    int top_halo = solver.free_surface ? 2*M: halo;
    // top_halo = (solver.free_surface && solver.abcn > 0) ? 2*M : top_halo;
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

__global__ void accumulate_rtm_image_2d(
    const float* __restrict__ u_forward,
    const float* __restrict__ u_backward,
    float* __restrict__ image,
    float* __restrict__ source_illumination,
    float* __restrict__ receiver_illumination,
    int nx, int nz
);

__global__ void accumulate_source_grad_2d(
    const float* __restrict__ u_backward,   // (B, nz, nx)
    float* __restrict__ grad_source,        // (B, nsrc, nt)
    const int* __restrict__ sources_loc,    // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
);
