#include "kernels.cuh"


__global__ void calculate_grad_3d(
    const float* __restrict__ u_forward,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz)
        return;

    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = nx * ny * nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_forward_b  = u_forward  + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 2*u_forward_b[idx] * u_backward_b[idx]/vp3;

}

__global__ void calculate_grad_utt_3d(
    const float* __restrict__ u_forward_next,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_now,  // (nt, B, nz, nx)
    const float* __restrict__ u_forward_prev,  // (nt, B, nz, nx)
    const float* __restrict__ u_backward, // (nt, B, nz, nx)
    const float* __restrict__ vp,        // (B, nz, nx)
    float* __restrict__ grad,             // (B, nz, nx)
    int B, int nx, int ny, int nz, float dt
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / nz;
    int iz = iz_global % nz;

    if (b >= B || ix >= nx || iy >= ny || iz >= nz)
        return;

    int stride_y = nx;
    int stride_z = nx * ny;
    int spatial_size = nx * ny * nz;

    int idx = iz * stride_z + iy * stride_y + ix;

    const float* u_next_b  = u_forward_next  + b * spatial_size;
    const float* u_now_b = u_forward_now + b * spatial_size;
    const float* u_prev_b = u_forward_prev + b * spatial_size;
    const float* u_backward_b = u_backward + b * spatial_size;
    float*       grad_b       = grad       + b * spatial_size;
    const float* vp_b         = vp         + b * spatial_size;

    float u_tt = (u_now_b[idx] - 2*u_prev_b[idx] + u_next_b[idx])/ (dt*dt);

    float vp3 = vp_b[idx] * vp_b[idx] * vp_b[idx];

    grad_b[idx] += 2*u_tt * u_backward_b[idx]/vp3;

}

__global__ void save_boundary_kernel_3d(
    const float* __restrict__ u,    // (B, nz, ny, nx)

    float* __restrict__ top,      // (nt, B, M, ny, nx)
    float* __restrict__ bottom,

    float* __restrict__ front,        // (nt, B, nz, M, nx)
    float* __restrict__ back,

    float* __restrict__ left,       // (nt, B, nz, ny, M)
    float* __restrict__ right,

    int it,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial_size = solver.nx * solver.ny * solver.nz;

    const float* u_b = u + b * spatial_size;

    int idx3 = iz * stride_z + iy * stride_y + ix;
    float val = u_b[idx3];

    // ==================================================
    // Z- (top)
    // ==================================================
    int top_start = solver.free_surface ? 
                    solver.M :
                    solver.abcn + solver.M;
    int top_end = top_start + solver.M;

    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((((it * solver.B + b) * solver.M + zloc) * solver.ny + iy) * solver.nx + ix);

        top[idx] = val;
    }

    // ==================================================
    // Z+ (bottom)
    // ==================================================
    if (iz >= solver.nz - solver.abcn - 2*solver.M && iz < solver.nz - solver.abcn - solver.M)
    {
        int zloc = iz - (solver.nz - solver.abcn - 2*solver.M);

        int idx =
            ((((it * solver.B + b) * solver.M + zloc) * solver.ny + iy) * solver.nx + ix);

        bottom[idx] = val;
    }

    // ==================================================
    // Y- (front)
    // ==================================================
    if (iy >= solver.abcn + solver.M && iy < solver.abcn + 2*solver.M)
    {
        int yloc = iy - (solver.abcn + solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.M + yloc) * solver.nx + ix);

        front[idx] = val;
    }

    // ==================================================
    // Y+ (back)
    // ==================================================
    if (iy >= solver.ny - solver.abcn - 2*solver.M && iy < solver.ny - solver.abcn - solver.M)
    {
        int yloc = iy - (solver.ny - solver.abcn - 2*solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.M + yloc) * solver.nx + ix);

        back[idx] = val;
    }

    // ==================================================
    // X- (left)
    // ==================================================
    if (ix >= solver.abcn + solver.M && ix < solver.abcn + 2*solver.M)
    {
        int xloc = ix - (solver.abcn + solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * solver.M + xloc);

        left[idx] = val;
    }

    // ==================================================
    // X+ (right)
    // ==================================================
    if (ix >= solver.nx - solver.abcn - 2*solver.M && ix < solver.nx - solver.abcn - solver.M)
    {
        int xloc = ix - (solver.nx - solver.abcn - 2*solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * solver.M + xloc);

        right[idx] = val;
    }
}


__global__ void restore_boundary_kernel_3d(
    float* __restrict__ u,        // (B, nz, ny, nx)

    const float* __restrict__ top,   // (nt, B, n, ny, nx)
    const float* __restrict__ bottom,

    const float* __restrict__ front,     // (nt, B, nz, n, nx)
    const float* __restrict__ back,

    const float* __restrict__ left,    // (nt, B, nz, ny, n)
    const float* __restrict__ right,

    int it,
    SolverContext solver
) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / solver.nz;
    int iz = iz_global % solver.nz;

    if (b >= solver.B || ix >= solver.nx || iy >= solver.ny || iz >= solver.nz)
        return;

    int stride_y = solver.nx;
    int stride_z = solver.nx * solver.ny;
    int spatial  = solver.nx * solver.ny * solver.nz;

    float* u_b = u + b * spatial;

    int idx3 = iz * stride_z + iy * stride_y + ix;

    // ==================================================
    // Z-  (top)
    // ==================================================
    int top_start = solver.free_surface ? 
                    solver.M :
                    solver.abcn + solver.M;

    int top_end = top_start + solver.M;
    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((((it * solver.B + b) * solver.M + zloc) * solver.ny + iy) * solver.nx + ix);

        u_b[idx3] = top[idx];
    }

    // ==================================================
    // Z+  (bottom)
    // ==================================================
    if (iz >= solver.nz - solver.abcn - 2*solver.M && iz < solver.nz - solver.abcn - solver.M)
    {
        int zloc = iz - (solver.nz - solver.abcn - 2*solver.M);

        int idx =
            ((((it * solver.B + b) * solver.M + zloc) * solver.ny + iy) * solver.nx + ix);

        u_b[idx3] = bottom[idx];
    }

    // ==================================================
    // Y-  (front)
    // ==================================================
    if (iy >= solver.abcn + solver.M && iy < solver.abcn + 2*solver.M)
    {
        int yloc = iy - (solver.abcn + solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.M + yloc) * solver.nx + ix);

        u_b[idx3] = front[idx];
    }

    // ==================================================
    // Y+  (back)
    // ==================================================
    if (iy >= solver.ny - solver.abcn - 2*solver.M && iy < solver.ny - solver.abcn - solver.M)
    {
        int yloc = iy - (solver.ny - solver.abcn - 2*solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.M + yloc) * solver.nx + ix);

        u_b[idx3] = back[idx];
    }

    // ==================================================
    // X-  (left)
    // ==================================================
    if (ix >= solver.abcn + solver.M && ix < solver.abcn + 2*solver.M)
    {
        int xloc = ix - (solver.abcn + solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * solver.M + xloc);

        u_b[idx3] = left[idx];
    }

    // ==================================================
    // X+  (right)
    // ==================================================
    if (ix >= solver.nx - solver.abcn - 2*solver.M && ix < solver.nx - solver.abcn - solver.M)
    {
        int xloc = ix - (solver.nx - solver.abcn - 2*solver.M);

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * solver.M + xloc);

        u_b[idx3] = right[idx];
    }
}