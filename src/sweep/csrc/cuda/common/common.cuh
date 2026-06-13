#pragma once
#include <cuda_runtime.h>
#include "context.h"

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

// ``cut_mask`` (SolverContext::cut_mask semantics: bit0 = x_lo, bit1 = x_hi,
// bit2 = z_lo, bit3 = z_hi) skips the rim-zeroing on domain-decomposition
// cut faces; 0 (default) reproduces the legacy all-faces behaviour.
__global__ void set_boundary_zeros(
    float* __restrict__ u,           // (B, nz, nx)
    int width,
    int nx,
    int nz,
    bool free_surface,
    int cut_mask = 0
);

__global__ void set_boundary_zeros_3d(
    float* __restrict__ u,   // (B, nz, ny, nx)
    int width,
    int nx,
    int ny,
    int nz
);