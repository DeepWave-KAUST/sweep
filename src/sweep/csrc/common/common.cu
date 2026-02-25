
#include "common.cuh"
#include "context.h"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add_source(
    float* __restrict__ u,          // (B, nz, nx)
    const float* __restrict__ source, // (B, nsrc, nt)
    const int* __restrict__ sources_loc,  // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
) {
    int b = blockIdx.x;
    int s = threadIdx.x;

    if (s >= nsrc) return;

    int base = (b * nsrc + s) * 2;
    int ix = sources_loc[base + 0];
    int iz = sources_loc[base + 1];

    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz)
        return;

    int spatial_size = solver.nx * solver.nz;
    int u_idx = b * spatial_size + iz * solver.nx + ix;
    int src_idx = (b * nsrc + s) * solver.nt + it;

    atomicAdd(&u[u_idx], source[src_idx]);

}


__global__ void record_kernel(
    const float* __restrict__ u,        // (B, nz, nx)
    float* __restrict__ record,          // (B, nrec, nt)
    const int* __restrict__ receivers,   // (B, nrec, 2)
    int it,
    int nrec,
    SolverContext solver
) {
    int b = blockIdx.x;     // shot index
    int r = threadIdx.x;    // receiver index

    if (r >= nrec) return;

    int base = (b * nrec + r) * 2;
    int ix = receivers[base + 0];
    int iz = receivers[base + 1];

    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz)
        return;

    int spatial_size = solver.nx * solver.nz;
    int u_idx = b * spatial_size + iz * solver.nx + ix;
    int rec_idx = (b * nrec + r) * solver.nt + it;

    record[rec_idx] = u[u_idx];
}

__global__ void add_source_3d(
    float* __restrict__ u,                 // (B, nz, ny, nx)
    const float* __restrict__ source,      // (B, nsrc, nt)
    const int* __restrict__ sources_loc,   // (B, nsrc, 3)
    int it,
    int nsrc,
    SolverContext solver
) {
    int b = blockIdx.x;      // shot index
    int s = threadIdx.x;     // source index

    if (s >= nsrc) return;

    int base = (b * nsrc + s) * 3;

    int ix = sources_loc[base + 0];
    int iy = sources_loc[base + 1];
    int iz = sources_loc[base + 2];

    if (ix < 0 || ix >= solver.nx ||
        iy < 0 || iy >= solver.ny ||
        iz < 0 || iz >= solver.nz)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int u_idx = b * spatial_size
              + iz * solver.ny * solver.nx
              + iy * solver.nx
              + ix;

    int src_idx = (b * nsrc + s) * solver.nt + it;

    atomicAdd(&u[u_idx], source[src_idx]);
}

__global__ void record_kernel_3d(
    const float* __restrict__ u,           // (B, nz, ny, nx)
    float* __restrict__ record,            // (B, nrec, nt)
    const int* __restrict__ receivers,     // (B, nrec, 3)
    int it,
    int nrec,
    SolverContext solver
) {
    int b = blockIdx.x;     // shot index
    int r = threadIdx.x;    // receiver index

    if (r >= nrec) return;

    int base = (b * nrec + r) * 3;

    int ix = receivers[base + 0];
    int iy = receivers[base + 1];
    int iz = receivers[base + 2];

    if (ix < 0 || ix >= solver.nx ||
        iy < 0 || iy >= solver.ny ||
        iz < 0 || iz >= solver.nz)
        return;

    int spatial_size = solver.nx * solver.ny * solver.nz;

    int u_idx = b * spatial_size
              + iz * solver.ny * solver.nx
              + iy * solver.nx
              + ix;

    int rec_idx = (b * nrec + r) * solver.nt + it;

    record[rec_idx] = u[u_idx];
}

__global__ void set_boundary_zeros(
    float* __restrict__ u,           // (B, nz, ny, nx)
    int width,
    int nx,
    int nz
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz) return;

    int spatial = nx * nz;
    float* u_b = u + b * spatial;

    int halo = width;

    if (ix < halo || ix >= nx - halo || iz < halo || iz >= nz - halo){
        int idx = iz * nx + ix;
        u_b[idx] = 0.f;
    }

}