
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
    const SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= solver.B || s >= nsrc) return;

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


__global__ void add_body_force_rho_grad_correction(
    float* __restrict__ grad_rho,
    const float* __restrict__ adj_field,
    const float* __restrict__ rho,
    const float* __restrict__ source,
    const int* __restrict__ sources_loc,
    int amp_it,
    int nsrc,
    int loc_dim,
    const SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= solver.B || s >= nsrc) return;
    if (amp_it < 0 || amp_it >= solver.nt) return;

    int base = (b * nsrc + s) * loc_dim;
    int ix = sources_loc[base + 0];
    int iz = sources_loc[base + loc_dim - 1];
    int iy = (loc_dim == 3) ? sources_loc[base + 1] : 0;

    if (ix < 0 || ix >= solver.nx || iz < 0 || iz >= solver.nz)
        return;
    if (loc_dim == 3 && (iy < 0 || iy >= solver.ny))
        return;

    int spatial_size = solver.nx * solver.nz * (loc_dim == 3 ? solver.ny : 1);
    int row = (loc_dim == 3) ? (iz * solver.ny + iy) : iz;
    int u_idx = b * spatial_size + row * solver.nx + ix;
    int src_idx = (b * nsrc + s) * solver.nt + amp_it;

    atomicAdd(&grad_rho[u_idx],
              adj_field[u_idx] * source[src_idx] / rho[u_idx]);
}


__global__ void record_kernel(
    const float* __restrict__ u,        // (B, nz, nx)
    float* __restrict__ record,          // (B, nrec, nt)
    const int* __restrict__ receivers,   // (B, nrec, 2)
    int it,
    int nrec,
    SolverContext solver
) {
    int b = blockIdx.x;  
    int r = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= solver.B || r >= nrec) return;

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
    const SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;

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
    const SolverContext solver
) {
    int b = blockIdx.x;  

    int r = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= solver.B || r >= nrec) return;

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
    float* __restrict__ u,           // (B, nz, nx)
    int width,
    int nx,
    int nz,
    bool fs_top,       // per-face free-surface flags: a free-surface face is
    bool fs_bottom,    // NOT zeroed (its boundary band holds the image mirror).
    bool fs_left,
    bool fs_right
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz) return;

    int spatial = nx * nz;
    float* u_b = u + b * spatial;

    int halo = width;

    bool left   = ix < halo;
    bool right  = ix >= nx - halo;
    bool bottom = iz >= nz - halo;
    bool top    = iz < halo;

    // Zero a boundary cell unless every face it belongs to is a free surface.
    bool zero = (left && !fs_left) || (right && !fs_right) ||
                (top && !fs_top) || (bottom && !fs_bottom);
    if (zero)
    {
        int idx = iz * nx + ix;
        u_b[idx] = 0.f;
    }
}

__global__ void set_boundary_zeros_3d(
    float* __restrict__ u,   // (B, nz, ny, nx)
    int width,
    int nx,
    int ny,
    int nz
)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / nz;
    int iz = iz_global % nz;

    if (ix >= nx || iy >= ny || iz >= nz) return;

    int spatial = nx * ny * nz;
    float* u_b = u + b * spatial;

    int halo = width;

    if (ix < halo || ix >= nx - halo ||
        iy < halo || iy >= ny - halo ||
        iz < halo || iz >= nz - halo)
    {
        int idx = iz * ny * nx + iy * nx + ix;
        u_b[idx] = 0.f;
    }
}