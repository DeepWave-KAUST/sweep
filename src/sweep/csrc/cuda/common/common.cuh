#pragma once
#include <cuda_runtime.h>
#include "context.h"

// Body-force (velocity-component) sources are injected RAW into the
// velocity field (add_source above), but the rho-gradient imaging kernels
// correlate the adjoint velocity with the STORED wavefield difference
// v(it) - v(it+1), which at a source cell contains the injected amplitude
// on top of the physical propagation update.  The true d(loss)/d(rho) has
// no such term (the injection itself is rho-independent), so the imaging
// over-counts by  -adj_v(it) * amp(it+1) / rho  at every source cell and
// step.  This kernel adds the compensating term.  Launched once per
// velocity-source field per reverse step, grid = (B, nsrc-blocks).
__global__ void add_body_force_rho_grad_correction(
    float* __restrict__ grad_rho,          // (B, nz, nx)
    const float* __restrict__ adj_field,   // adjoint velocity comp (B, nz, nx)
    const float* __restrict__ rho,         // (B, nz, nx)
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 2 or 3)
    int amp_it,
    int nsrc,
    int loc_dim,
    const SolverContext solver
);

// Mirror image of the kernel above, on the RECEIVER side.  The adjoint
// residual for a velocity receiver is injected into the adjoint velocity
// BEFORE the per-step rho imaging runs, so the imaging correlates the
// just-injected residual with the same stored difference v(it) - v(it+1).
// The discrete adjoint has no such term — the multiplier of d(step)/d(rho)
// is the adjoint velocity carried in from the later steps only — so the
// imaging over-counts by  resid(it) * (v(it) - v(it+1)) / rho  at every
// receiver cell and step.  This kernel subtracts it.  Launched once per
// velocity-receiver field per reverse step, grid = (B, nrec-blocks).
//
// ``fv_now`` / ``fv_next`` must be the very pointers the imaging kernel
// correlated for this step, and ``halo`` its skipped border, so a receiver
// that sits inside the halo (where no imaging ran) is left alone.
__global__ void sub_receiver_rho_grad_correction(
    float* __restrict__ grad_rho,            // (B, nz, nx)
    const float* __restrict__ fv_now,        // forward velocity comp at it
    const float* __restrict__ fv_next,       // forward velocity comp at it+1
    const float* __restrict__ rho,           // (B, nz, nx)
    const float* __restrict__ adjoint_source,// (B, nrec, nt)
    const int* __restrict__ receivers_loc,   // (B, nrec, 2 or 3)
    int it,
    int nrec,
    int loc_dim,
    int halo,
    const SolverContext solver
);

__global__ void add_source(
    float* __restrict__ u,          // (B, nz, nx)
    const float* __restrict__ source, // (B, nsrc, nt)
    const int* __restrict__ sources_loc,  // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
);

__global__ void record_kernel(
    const float* __restrict__ u,        // (B, nz, nx)
    float* __restrict__ record,          // (B, nrec, nt)
    const int* __restrict__ receivers,   // (B, nrec, 2)
    int it,
    int nrec,
    SolverContext solver
);

__global__ void add_source_3d(
    float* __restrict__ u,                 // (B, nz, ny, nx)
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 3)
    int it,
    int nsrc,
    SolverContext solver
);

__global__ void record_kernel_3d(
    const float* __restrict__ u,           // (B, nz, ny, nx)
    float* __restrict__ record,            // (B, nrec, nt)
    const int* __restrict__ receivers,     // (B, nrec, 3)
    int it,
    int nrec,
    SolverContext solver
);

// ``fs_*`` mark per-face free surfaces: such a face is NOT zeroed, its
// boundary band holds the image mirror.
// ``cut_mask`` (SolverContext::cut_mask semantics: bit0 = x_lo, bit1 = x_hi,
// bit2 = z_lo, bit3 = z_hi) skips the rim-zeroing on domain-decomposition
// cut faces; 0 (default) reproduces the legacy all-faces behaviour.
__global__ void set_boundary_zeros(
    float* __restrict__ u,           // (B, nz, nx)
    int width,
    int nx,
    int nz,
    bool fs_top,
    bool fs_bottom,
    bool fs_left,
    bool fs_right,
    int cut_mask = 0
);

__global__ void set_boundary_zeros_3d(
    float* __restrict__ u,   // (B, nz, ny, nx)
    int width,
    int nx,
    int ny,
    int nz
);