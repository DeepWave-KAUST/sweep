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

__global__ void set_boundary_zeros(
    float* __restrict__ u,           // (B, nz, nx)
    int width,
    int nx,
    int nz,
    bool fs_top,
    bool fs_bottom,
    bool fs_left,
    bool fs_right
);

__global__ void set_boundary_zeros_3d(
    float* __restrict__ u,   // (B, nz, ny, nx)
    int width,
    int nx,
    int ny,
    int nz
);