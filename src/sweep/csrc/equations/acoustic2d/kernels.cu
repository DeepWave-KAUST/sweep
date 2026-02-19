#include "kernels.cuh"

__global__ void calculate_grad(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;

    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

    const float* u_forward_b  = u_forward  + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 2*u_forward_b[idx] * u_backward_b[idx]/vp3;

}

__global__ void calculate_grad_utt(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int nx, int nz, float dt
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= nx || iz >= nz)
        return;

    int spatial_size = nx * nz;
    int idx = iz * nx + ix;

    const float* u_next_b  = u_forward_next  + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx]) / (dt*dt);

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 2*u_tt * u_backward_b[idx]/vp3;

}

__global__ void save_boundary_kernel(
    const float* __restrict__ u,   // (B, nz, nx)
    float* __restrict__ top,       // (nt, B, n, nx)
    float* __restrict__ bottom,    // (nt, B, n, nx)
    float* __restrict__ left,      // (nt, B, nz, n)
    float* __restrict__ right,     // (nt, B, nz, n)
    int it,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial = solver.nx * solver.nz;
    const float* u_b = u + b * spatial;

    float val = u_b[iz * solver.nx + ix];

    // ======================
    // TOP boundary
    // ======================

    int top_start = solver.free_surface ? 
                    solver.M :
                    solver.abcn + solver.M;

    int top_end = top_start + solver.M;

    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((it * solver.B + b) * solver.M + zloc) 
            * solver.nx + ix;

        top[idx] = val;
    }

    // ======================
    // BOTTOM boundary
    // ======================
    if (iz >= solver.nz - solver.abcn - 2*solver.M && iz < solver.nz - solver.abcn - solver.M)
    {
        int zloc = iz - (solver.nz - solver.abcn - 2*solver.M);

        int idx =
            ((it * solver.B + b) * solver.M + zloc) * solver.nx + ix;

        bottom[idx] = val;
    }

    // ======================
    // LEFT boundary
    // ======================
    if (ix >= solver.abcn + solver.M && ix < solver.abcn + 2*solver.M)
    {
        int xloc = ix - (solver.abcn + solver.M);

        int idx =
            ((it * solver.B + b) * solver.nz + iz) * solver.M + xloc;

        left[idx] = val;
    }

    // ======================
    // RIGHT boundary
    // ======================
    if (ix >= solver.nx - solver.abcn - 2*solver.M && ix < solver.nx - solver.abcn - solver.M)
    {
        int xloc = ix - (solver.nx - solver.abcn - 2*solver.M);

        int idx =
            ((it * solver.B + b) * solver.nz + iz) * solver.M + xloc;

        right[idx] = val;
    }
}

__global__ void restore_boundary_kernel(
    float* __restrict__ u,        // (B, nz, nx)
    const float* __restrict__ top,
    const float* __restrict__ bottom,
    const float* __restrict__ left,
    const float* __restrict__ right,
    int it,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial = solver.nx * solver.nz;
    float* u_b = u + b * spatial;

    // ======================
    // TOP
    // ======================
    int top_start = solver.free_surface ? 
                    solver.M :
                    solver.abcn + solver.M;
    int top_end   = top_start + solver.M;

    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((it * solver.B + b) * solver.M + zloc) 
            * solver.nx + ix;

        u_b[iz * solver.nx + ix] = top[idx];
    }

    // ======================
    // BOTTOM
    // ======================
    if (iz >= solver.nz - solver.abcn - 2*solver.M && iz < solver.nz - solver.abcn - solver.M)
    {
        int zloc = iz - (solver.nz - solver.abcn - 2*solver.M);

        int idx =
            ((it * solver.B + b) * solver.M + zloc) * solver.nx + ix;

        u_b[iz * solver.nx + ix] = bottom[idx];
    }

    // ======================
    // LEFT
    // ======================
    if (ix >= solver.abcn + solver.M && ix < solver.abcn + 2*solver.M)
    {
        int xloc = ix - (solver.abcn + solver.M);

        int idx =
            ((it * solver.B + b) * solver.nz + iz) * solver.M + xloc;

        u_b[iz * solver.nx + ix] = left[idx];
    }

    // ======================
    // RIGHT
    // ======================
    if (ix >= solver.nx - solver.abcn - 2*solver.M && ix < solver.nx - solver.abcn - solver.M)
    {
        int xloc = ix - (solver.nx - solver.abcn - 2*solver.M);

        int idx =
            ((it * solver.B + b) * solver.nz + iz) * solver.M + xloc;

        u_b[iz * solver.nx + ix] = right[idx];
    }
}