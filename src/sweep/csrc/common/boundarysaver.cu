#include "context.h"
#include "boundarysaver.h"
#include <cuda_runtime.h>

__global__ void save_boundary_kernel(
    const float* __restrict__ u,   // (B, nz, nx)
    float* __restrict__ top,       // (nt, B, n, nx)
    float* __restrict__ bottom,    // (nt, B, n, nx)
    float* __restrict__ left,      // (nt, B, nz, n)
    float* __restrict__ right,     // (nt, B, nz, n)
    int it,
    int width,
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

    int thickness = width;

    // ======================
    // TOP boundary
    // ======================

    int top_start = solver.free_surface ?
                    solver.M :
                    solver.abcn + solver.M;

    int top_end = top_start + thickness;

    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((it * solver.B + b) * thickness + zloc)
            * solver.nx + ix;

        top[idx] = val;
    }

    // ======================
    // BOTTOM boundary
    // ======================
    int bot_end   = solver.nz - solver.abcn - solver.M;
    int bot_start = bot_end - thickness;

    if (iz >= bot_start && iz < bot_end)
    {
        int zloc = iz - bot_start;

        int idx =
            ((it * solver.B + b) * thickness + zloc)
            * solver.nx + ix;

        bottom[idx] = val;
    }

    // ======================
    // LEFT boundary
    // ======================
    int left_start = solver.abcn + solver.M;
    int left_end   = left_start + thickness;

    if (ix >= left_start && ix < left_end)
    {
        int xloc = ix - left_start;

        int idx =
            ((it * solver.B + b) * solver.nz + iz)
            * thickness + xloc;

        left[idx] = val;
    }

    // ======================
    // RIGHT boundary
    // ======================
    int right_end   = solver.nx - solver.abcn - solver.M;
    int right_start = right_end - thickness;

    if (ix >= right_start && ix < right_end)
    {
        int xloc = ix - right_start;

        int idx =
            ((it * solver.B + b) * solver.nz + iz)
            * thickness + xloc;

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
    int width,
    SolverContext solver
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= solver.nx || iz >= solver.nz) return;

    int spatial = solver.nx * solver.nz;
    float* u_b = u + b * spatial;

    int thickness = width;
    // ======================
    // TOP
    // ======================
    int top_start = solver.free_surface ?
                    solver.M :
                    solver.abcn + solver.M;

    int top_end = top_start + thickness;

    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((it * solver.B + b) * thickness + zloc)
            * solver.nx + ix;

        u_b[iz * solver.nx + ix] = top[idx];
    }

    // ======================
    // BOTTOM
    // ======================
    int bot_end   = solver.nz - solver.abcn - solver.M;
    int bot_start = bot_end - thickness;

    if (iz >= bot_start && iz < bot_end)
    {
        int zloc = iz - bot_start;

        int idx =
            ((it * solver.B + b) * thickness + zloc)
            * solver.nx + ix;

        u_b[iz * solver.nx + ix] = bottom[idx];
    }

    // ======================
    // LEFT
    // ======================
    int left_start = solver.abcn + solver.M;
    int left_end   = left_start + thickness;

    if (ix >= left_start && ix < left_end)
    {
        int xloc = ix - left_start;

        int idx =
            ((it * solver.B + b) * solver.nz + iz)
            * thickness + xloc;

        u_b[iz * solver.nx + ix] = left[idx];
    }

    // ======================
    // RIGHT
    // ======================
    int right_end   = solver.nx - solver.abcn - solver.M;
    int right_start = right_end - thickness;

    if (ix >= right_start && ix < right_end)
    {
        int xloc = ix - right_start;

        int idx =
            ((it * solver.B + b) * solver.nz + iz)
            * thickness + xloc;

        u_b[iz * solver.nx + ix] = right[idx];
    }
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
    int width,
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
    int top_end = top_start + width;

    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((((it * solver.B + b) * width + zloc) * solver.ny + iy) * solver.nx + ix);

        top[idx] = val;
    }

    // ==================================================
    // Z+ (bottom)
    // ==================================================
    int bot_end = solver.nz - solver.abcn - solver.M;
    int bot_start = bot_end - width;
    if (iz >= bot_start && iz < bot_end)
    {
        int zloc = iz - bot_start;

        int idx =
            ((((it * solver.B + b) * width + zloc) * solver.ny + iy) * solver.nx + ix);

        bottom[idx] = val;
    }

    // ==================================================
    // Y- (front)
    // ==================================================
    int front_start = solver.abcn + solver.M;
    int front_end = front_start + width;
    if (iy >= front_start && iy < front_end)
    {
        int yloc = iy - front_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * width + yloc) * solver.nx + ix);

        front[idx] = val;
    }

    // ==================================================
    // Y+ (back)
    // ==================================================
    int back_end = solver.ny - solver.abcn - solver.M;
    int back_start = back_end - width;
    if (iy >= back_start && iy < back_end)
    {
        int yloc = iy - back_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * width + yloc) * solver.nx + ix);

        back[idx] = val;
    }

    // ==================================================
    // X- (left)
    // ==================================================
    int left_start = solver.abcn + solver.M;
    int left_end = left_start + width;
    if (ix >= left_start && ix < left_end)
    {
        int xloc = ix - left_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * width + xloc);

        left[idx] = val;
    }

    // ==================================================
    // X+ (right)
    // ==================================================
    int right_end = solver.nx - solver.abcn - solver.M;
    int right_start = right_end - width;
    if (ix >= right_start && ix < right_end)
    {
        int xloc = ix - right_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * width + xloc);

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
    int width,
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

    int top_end = top_start + width;
    if (iz >= top_start && iz < top_end)
    {
        int zloc = iz - top_start;

        int idx =
            ((((it * solver.B + b) * width + zloc) * solver.ny + iy) * solver.nx + ix);

        u_b[idx3] = top[idx];
    }

    // ==================================================
    // Z+  (bottom)
    // ==================================================
    int bot_end = solver.nz - solver.abcn - solver.M;
    int bot_start = bot_end - width;
    if (iz >= bot_start && iz < bot_end)
    {
        int zloc = iz - bot_start;

        int idx =
            ((((it * solver.B + b) * width + zloc) * solver.ny + iy) * solver.nx + ix);

        u_b[idx3] = bottom[idx];
    }

    // ==================================================
    // Y-  (front)
    // ==================================================
    int front_start = solver.abcn + solver.M;
    int front_end = front_start + width;
    if (iy >= front_start && iy < front_end)
    {
        int yloc = iy - front_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * width + yloc) * solver.nx + ix);

        u_b[idx3] = front[idx];
    }

    // ==================================================
    // Y+  (back)
    // ==================================================
    int back_end = solver.ny - solver.abcn - solver.M;
    int back_start = back_end - width;
    if (iy >= back_start && iy < back_end)
    {
        int yloc = iy - back_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * width + yloc) * solver.nx + ix);

        u_b[idx3] = back[idx];
    }

    // ==================================================
    // X-  (left)
    // ==================================================
    int left_start = solver.abcn + solver.M;
    int left_end = left_start + width;
    if (ix >= left_start && ix < left_end)
    {
        int xloc = ix - left_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * width + xloc);

        u_b[idx3] = left[idx];
    }

    // ==================================================
    // X+  (right)
    // ==================================================
    int right_end = solver.nx - solver.abcn - solver.M;
    int right_start = right_end - width;
    if (ix >= right_start && ix < right_end)
    {
        int xloc = ix - right_start;

        int idx =
            ((((it * solver.B + b) * solver.nz + iz) * solver.ny + iy) * width + xloc);

        u_b[idx3] = right[idx];
    }
}

__global__ void save_boundary_kernel_3d_advance(
    const float* __restrict__ u,    // (B, nz, ny, nx)

    float* __restrict__ top,        // (nt,B,width,ny_phys,nx_phys)
    float* __restrict__ bottom,

    float* __restrict__ front,      // (nt,B,nz_phys,width,nx_phys)
    float* __restrict__ back,

    float* __restrict__ left,       // (nt,B,nz_phys,ny_phys,width)
    float* __restrict__ right,

    int it,
    int width,
    SolverContext solver
)
{
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
    float val = u_b[iz * stride_z + iy * stride_y + ix];

    // ======================================================
    // interior physical region
    // ======================================================
    int phys_z0 = solver.free_surface ? solver.M : solver.abcn + solver.M;
    int phys_z1 = solver.nz - solver.abcn - solver.M;

    int phys_y0 = solver.abcn + solver.M;
    int phys_y1 = solver.ny - solver.abcn - solver.M;

    int phys_x0 = solver.abcn + solver.M;
    int phys_x1 = solver.nx - solver.abcn - solver.M;

    int nz_phys = phys_z1 - phys_z0;
    int ny_phys = phys_y1 - phys_y0;
    int nx_phys = phys_x1 - phys_x0;

    // ======================================================
    // Z-  (top)
    // ======================================================
    int top_start = phys_z0;
    int top_end   = top_start + width;

    if (iz >= top_start && iz < top_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - top_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        top[idx] = val;
    }

    // ======================================================
    // Z+ (bottom)
    // ======================================================
    int bot_end   = phys_z1;
    int bot_start = bot_end - width;

    if (iz >= bot_start && iz < bot_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - bot_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        bottom[idx] = val;
    }

    // ======================================================
    // Y- (front)
    // ======================================================
    int front_start = phys_y0;
    int front_end   = front_start + width;

    if (iy >= front_start && iy < front_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - front_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        front[idx] = val;
    }

    // ======================================================
    // Y+ (back)
    // ======================================================
    int back_end   = phys_y1;
    int back_start = back_end - width;

    if (iy >= back_start && iy < back_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - back_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        back[idx] = val;
    }

    // ======================================================
    // X- (left)
    // ======================================================
    int left_start = phys_x0;
    int left_end   = left_start + width;

    if (ix >= left_start && ix < left_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - left_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        left[idx] = val;
    }

    // ======================================================
    // X+ (right)
    // ======================================================
    int right_end   = phys_x1;
    int right_start = right_end - width;

    if (ix >= right_start && ix < right_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - right_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        right[idx] = val;
    }
}

__global__ void restore_boundary_kernel_3d_advance(
    float* __restrict__ u,        // (B, nz, ny, nx)

    const float* __restrict__ top,      // (nt,B,width,ny_phys,nx_phys)
    const float* __restrict__ bottom,

    const float* __restrict__ front,    // (nt,B,nz_phys,width,nx_phys)
    const float* __restrict__ back,

    const float* __restrict__ left,     // (nt,B,nz_phys,ny_phys,width)
    const float* __restrict__ right,

    int it,
    int width,
    SolverContext solver
)
{
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

    // ======================================================
    // interior physical region
    // ======================================================
    int phys_z0 = solver.free_surface ? solver.M : solver.abcn + solver.M;
    int phys_z1 = solver.nz - solver.abcn - solver.M;

    int phys_y0 = solver.abcn + solver.M;
    int phys_y1 = solver.ny - solver.abcn - solver.M;

    int phys_x0 = solver.abcn + solver.M;
    int phys_x1 = solver.nx - solver.abcn - solver.M;

    int nz_phys = phys_z1 - phys_z0;
    int ny_phys = phys_y1 - phys_y0;
    int nx_phys = phys_x1 - phys_x0;

    // ======================================================
    // Z- (top)
    // ======================================================
    int top_start = phys_z0;
    int top_end   = top_start + width;

    if (iz >= top_start && iz < top_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - top_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        u_b[idx3] = top[idx];
    }

    // ======================================================
    // Z+ (bottom)
    // ======================================================
    int bot_end   = phys_z1;
    int bot_start = bot_end - width;

    if (iz >= bot_start && iz < bot_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - bot_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        u_b[idx3] = bottom[idx];
    }

    // ======================================================
    // Y- (front)
    // ======================================================
    int front_start = phys_y0;
    int front_end   = front_start + width;

    if (iy >= front_start && iy < front_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - front_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        u_b[idx3] = front[idx];
    }

    // ======================================================
    // Y+ (back)
    // ======================================================
    int back_end   = phys_y1;
    int back_start = back_end - width;

    if (iy >= back_start && iy < back_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - back_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        u_b[idx3] = back[idx];
    }

    // ======================================================
    // X- (left)
    // ======================================================
    int left_start = phys_x0;
    int left_end   = left_start + width;

    if (ix >= left_start && ix < left_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - left_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        u_b[idx3] = left[idx];
    }

    // ======================================================
    // X+ (right)
    // ======================================================
    int right_end   = phys_x1;
    int right_start = right_end - width;

    if (ix >= right_start && ix < right_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - right_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        u_b[idx3] = right[idx];
    }
}

__global__ void save_boundary_kernel_3d_advance2(
    const float* __restrict__ u,    // (B, nz, ny, nx)

    float* __restrict__ top,        // (nt,B,width,ny_phys,nx_phys)
    float* __restrict__ bottom,

    float* __restrict__ front,      // (nt,B,nz_phys,width,nx_phys)
    float* __restrict__ back,

    float* __restrict__ left,       // (nt,B,nz_phys,ny_phys,width)
    float* __restrict__ right,

    int it,
    int width,
    SolverContext solver
)
{
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
    float val = u_b[iz * stride_z + iy * stride_y + ix];

    // ======================================================
    // physical interior region
    // ======================================================
    int phys_z0 = solver.free_surface ? solver.M : solver.abcn + solver.M;
    int phys_z1 = solver.nz - solver.abcn - solver.M;

    int phys_y0 = solver.abcn + solver.M;
    int phys_y1 = solver.ny - solver.abcn - solver.M;

    int phys_x0 = solver.abcn + solver.M;
    int phys_x1 = solver.nx - solver.abcn - solver.M;

    int nz_phys = phys_z1 - phys_z0;
    int ny_phys = phys_y1 - phys_y0;
    int nx_phys = phys_x1 - phys_x0;

    // ======================================================
    // Z-  (top)  phys_z0 - width  → phys_z0
    // ======================================================
    int top_end   = phys_z0;
    int top_start = top_end - width;

    if (iz >= top_start && iz < top_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - top_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        top[idx] = val;
    }

    // ======================================================
    // Z+ (bottom)  phys_z1  → phys_z1 + width
    // ======================================================
    int bot_start = phys_z1;
    int bot_end   = bot_start + width;

    if (iz >= bot_start && iz < bot_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - bot_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        bottom[idx] = val;
    }

    // ======================================================
    // Y- (front)
    // ======================================================
    int front_end   = phys_y0;
    int front_start = front_end - width;

    if (iy >= front_start && iy < front_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - front_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        front[idx] = val;
    }

    // ======================================================
    // Y+ (back)
    // ======================================================
    int back_start = phys_y1;
    int back_end   = back_start + width;

    if (iy >= back_start && iy < back_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - back_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        back[idx] = val;
    }

    // ======================================================
    // X- (left)
    // ======================================================
    int left_end   = phys_x0;
    int left_start = left_end - width;

    if (ix >= left_start && ix < left_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - left_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        left[idx] = val;
    }

    // ======================================================
    // X+ (right)
    // ======================================================
    int right_start = phys_x1;
    int right_end   = right_start + width;

    if (ix >= right_start && ix < right_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - right_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        right[idx] = val;
    }
}

__global__ void restore_boundary_kernel_3d_advance2(
    float* __restrict__ u,        // (B, nz, ny, nx)

    const float* __restrict__ top,
    const float* __restrict__ bottom,

    const float* __restrict__ front,
    const float* __restrict__ back,

    const float* __restrict__ left,
    const float* __restrict__ right,

    int it,
    int width,
    SolverContext solver
)
{
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

    // ======================================================
    // physical interior definition (unchanged)
    // ======================================================
    int phys_z0 = solver.free_surface ? solver.M : solver.abcn + solver.M;
    int phys_z1 = solver.nz - solver.abcn - solver.M;

    int phys_y0 = solver.abcn + solver.M;
    int phys_y1 = solver.ny - solver.abcn - solver.M;

    int phys_x0 = solver.abcn + solver.M;
    int phys_x1 = solver.nx - solver.abcn - solver.M;

    int nz_phys = phys_z1 - phys_z0;
    int ny_phys = phys_y1 - phys_y0;
    int nx_phys = phys_x1 - phys_x0;

    // ======================================================
    // Z- (top)   [phys_z0 - width , phys_z0)
    // ======================================================
    int top_end   = phys_z0;
    int top_start = top_end - width;

    if (iz >= top_start && iz < top_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - top_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        u_b[idx3] = top[idx];
    }

    // ======================================================
    // Z+ (bottom)   [phys_z1 , phys_z1 + width)
    // ======================================================
    int bot_start = phys_z1;
    int bot_end   = bot_start + width;

    if (iz >= bot_start && iz < bot_end &&
        iy >= phys_y0 && iy < phys_y1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int zloc = iz - bot_start;
        int yloc = iy - phys_y0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * width + zloc)
              * ny_phys + yloc)
              * nx_phys + xloc);

        u_b[idx3] = bottom[idx];
    }

    // ======================================================
    // Y- (front)   [phys_y0 - width , phys_y0)
    // ======================================================
    int front_end   = phys_y0;
    int front_start = front_end - width;

    if (iy >= front_start && iy < front_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - front_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        u_b[idx3] = front[idx];
    }

    // ======================================================
    // Y+ (back)   [phys_y1 , phys_y1 + width)
    // ======================================================
    int back_start = phys_y1;
    int back_end   = back_start + width;

    if (iy >= back_start && iy < back_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        ix >= phys_x0 && ix < phys_x1)
    {
        int yloc = iy - back_start;
        int zloc = iz - phys_z0;
        int xloc = ix - phys_x0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * width + yloc)
              * nx_phys + xloc);

        u_b[idx3] = back[idx];
    }

    // ======================================================
    // X- (left)   [phys_x0 - width , phys_x0)
    // ======================================================
    int left_end   = phys_x0;
    int left_start = left_end - width;

    if (ix >= left_start && ix < left_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - left_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        u_b[idx3] = left[idx];
    }

    // ======================================================
    // X+ (right)   [phys_x1 , phys_x1 + width)
    // ======================================================
    int right_start = phys_x1;
    int right_end   = right_start + width;

    if (ix >= right_start && ix < right_end &&
        iz >= phys_z0 && iz < phys_z1 &&
        iy >= phys_y0 && iy < phys_y1)
    {
        int xloc = ix - right_start;
        int zloc = iz - phys_z0;
        int yloc = iy - phys_y0;

        int idx =
            ((((it * solver.B + b) * nz_phys + zloc)
              * ny_phys + yloc)
              * width + xloc);

        u_b[idx3] = right[idx];
    }
}