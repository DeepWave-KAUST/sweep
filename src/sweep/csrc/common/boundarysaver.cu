#include "context.h"
#include "boundary/kernels.cuh"
#include <cuda_runtime.h>

__global__ void boundary_kernel2d(
    float* __restrict__ u,

    float* __restrict__ top,
    float* __restrict__ bottom,
    float* __restrict__ left,
    float* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;

    if (ix >= ctx.nx || iz >= ctx.nz) return;

    int spatial = ctx.nx * ctx.nz;
    float* u_b = u + b * spatial;

    int x0 = ctx.phys_x0();
    int x1 = ctx.phys_x1();

    int z0 = ctx.phys_z0();
    int z1 = ctx.phys_z1();

    int nx_phys = ctx.nx_phys();
    int nz_phys = ctx.nz_phys();
    int nx_boundary = nx_phys + 2 * tangent_pad;
    int nz_boundary = nz_phys + 2 * tangent_pad;

    int x_t0 = x0 - tangent_pad;
    int x_t1 = x1 + tangent_pad;
    int z_t0 = z0 - tangent_pad;
    int z_t1 = z1 + tangent_pad;

    // ---------- symmetric boundary bands ----------

    int top_start = z0 + offset;
    int top_end   = top_start + width;

    int bot_end   = z1 - offset;
    int bot_start = bot_end - width;

    int left_start = x0 + offset;
    int left_end   = left_start + width;

    int right_end   = x1 - offset;
    int right_start = right_end - width;

    bool is_top =
        (iz >= top_start && iz < top_end) &&
        (ix >= x_t0 && ix < x_t1);

    bool is_bottom =
        (iz >= bot_start && iz < bot_end) &&
        (ix >= x_t0 && ix < x_t1);

    bool is_left =
        (ix >= left_start && ix < left_end) &&
        (iz >= z_t0 && iz < z_t1);

    bool is_right =
        (ix >= right_start && ix < right_end) &&
        (iz >= z_t0 && iz < z_t1);

    if (!(is_top || is_bottom || is_left || is_right))
        return;

    float val = u_b[iz * ctx.nx + ix];

    // TOP
    if (is_top)
    {
        int zloc = iz - top_start;
        int xloc = ix - x_t0;

        int idx =
            ((it * ctx.B + b) * width + zloc) * nx_boundary + xloc;

        if (mode == BOUNDARY_SAVE) top[idx] = val;
        else u_b[iz * ctx.nx + ix] = top[idx];
    }

    // BOTTOM
    if (is_bottom)
    {
        int zloc = iz - bot_start;
        int xloc = ix - x_t0;

        int idx =
            ((it * ctx.B + b) * width + zloc) * nx_boundary + xloc;

        if (mode == BOUNDARY_SAVE) bottom[idx] = val;
        else u_b[iz * ctx.nx + ix] = bottom[idx];
    }

    // LEFT
    if (is_left)
    {
        int xloc = ix - left_start;
        int zloc = iz - z_t0;

        int idx =
            ((it * ctx.B + b) * nz_boundary + zloc) * width + xloc;

        if (mode == BOUNDARY_SAVE) left[idx] = val;
        else u_b[iz * ctx.nx + ix] = left[idx];
    }
    
    // RIGHT
    if (is_right)
    {
        int xloc = ix - right_start;
        int zloc = iz - z_t0;

        int idx =
            ((it * ctx.B + b) * nz_boundary + zloc) * width + xloc;

        if (mode == BOUNDARY_SAVE) right[idx] = val;
        else u_b[iz * ctx.nx + ix] = right[idx];
    }
}

__global__ void boundary_kernel3d(
    float* __restrict__ u,        // (B, nz, ny, nx)

    float* __restrict__ top,      // (nt,B,width,ny_phys,nx_phys)
    float* __restrict__ bottom,

    float* __restrict__ front,    // (nt,B,nz_phys,width,nx_phys)
    float* __restrict__ back,

    float* __restrict__ left,     // (nt,B,nz_phys,ny_phys,width)
    float* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;

    int b  = iz_global / ctx.nz;
    int iz = iz_global % ctx.nz;

    if (b >= ctx.B || ix >= ctx.nx || iy >= ctx.ny || iz >= ctx.nz)
        return;

    int stride_y = ctx.nx;
    int stride_z = ctx.nx * ctx.ny;
    int spatial  = ctx.nx * ctx.ny * ctx.nz;

    float* u_b = u + b * spatial;
    int idx3 = iz * stride_z + iy * stride_y + ix;

    // --------------------------------------------------
    // physical domain
    // --------------------------------------------------

    int x0 = ctx.phys_x0();
    int x1 = ctx.phys_x1();

    int y0 = ctx.phys_y0();
    int y1 = ctx.phys_y1();

    int z0 = ctx.phys_z0();
    int z1 = ctx.phys_z1();

    int nx_phys = ctx.nx_phys();
    int ny_phys = ctx.ny_phys();
    int nz_phys = ctx.nz_phys();
    int nx_boundary = nx_phys + 2 * tangent_pad;
    int ny_boundary = ny_phys + 2 * tangent_pad;
    int nz_boundary = nz_phys + 2 * tangent_pad;

    int x_t0 = x0 - tangent_pad;
    int x_t1 = x1 + tangent_pad;
    int y_t0 = y0 - tangent_pad;
    int y_t1 = y1 + tangent_pad;
    int z_t0 = z0 - tangent_pad;
    int z_t1 = z1 + tangent_pad;

    // --------------------------------------------------
    // boundary ranges
    // --------------------------------------------------

    int top_start = z0 + offset;
    int top_end   = top_start + width;

    int bot_end   = z1 - offset;
    int bot_start = bot_end - width;

    int front_start = y0 + offset;
    int front_end   = front_start + width;

    int back_end   = y1 - offset;
    int back_start = back_end - width;

    int left_start = x0 + offset;
    int left_end   = left_start + width;

    int right_end   = x1 - offset;
    int right_start = right_end - width;

    bool is_top =
        iz >= top_start && iz < top_end &&
        iy >= y_t0 && iy < y_t1 &&
        ix >= x_t0 && ix < x_t1;

    bool is_bottom =
        iz >= bot_start && iz < bot_end &&
        iy >= y_t0 && iy < y_t1 &&
        ix >= x_t0 && ix < x_t1;

    bool is_front =
        iy >= front_start && iy < front_end &&
        iz >= z_t0 && iz < z_t1 &&
        ix >= x_t0 && ix < x_t1;

    bool is_back =
        iy >= back_start && iy < back_end &&
        iz >= z_t0 && iz < z_t1 &&
        ix >= x_t0 && ix < x_t1;

    bool is_left =
        ix >= left_start && ix < left_end &&
        iz >= z_t0 && iz < z_t1 &&
        iy >= y_t0 && iy < y_t1;

    bool is_right =
        ix >= right_start && ix < right_end &&
        iz >= z_t0 && iz < z_t1 &&
        iy >= y_t0 && iy < y_t1;

    if (!(is_top || is_bottom || is_front || is_back || is_left || is_right))
        return;

    float val = u_b[idx3];

    // ======================================================
    // Z- (top)
    // ======================================================

    if (is_top)
    {
        int zloc = iz - top_start;
        int yloc = iy - y_t0;
        int xloc = ix - x_t0;

        int idx =
            ((((it * ctx.B + b) * width + zloc)
              * ny_boundary + yloc)
              * nx_boundary + xloc);

        if (mode == BOUNDARY_SAVE)
            top[idx] = val;
        else
            u_b[idx3] = top[idx];
    }

    // ======================================================
    // Z+ (bottom)
    // ======================================================

    if (is_bottom)
    {
        int zloc = iz - bot_start;
        int yloc = iy - y_t0;
        int xloc = ix - x_t0;

        int idx =
            ((((it * ctx.B + b) * width + zloc)
              * ny_boundary + yloc)
              * nx_boundary + xloc);

        if (mode == BOUNDARY_SAVE)
            bottom[idx] = val;
        else
            u_b[idx3] = bottom[idx];
    }

    // ======================================================
    // Y- (front)
    // ======================================================

    if (is_front)
    {
        int yloc = iy - front_start;
        int zloc = iz - z_t0;
        int xloc = ix - x_t0;

        int idx =
            ((((it * ctx.B + b) * nz_boundary + zloc)
              * width + yloc)
              * nx_boundary + xloc);

        if (mode == BOUNDARY_SAVE)
            front[idx] = val;
        else
            u_b[idx3] = front[idx];
    }

    // ======================================================
    // Y+ (back)
    // ======================================================

    if (is_back)
    {
        int yloc = iy - back_start;
        int zloc = iz - z_t0;
        int xloc = ix - x_t0;

        int idx =
            ((((it * ctx.B + b) * nz_boundary + zloc)
              * width + yloc)
              * nx_boundary + xloc);

        if (mode == BOUNDARY_SAVE)
            back[idx] = val;
        else
            u_b[idx3] = back[idx];
    }

    // ======================================================
    // X- (left)
    // ======================================================

    if (is_left)
    {
        int xloc = ix - left_start;
        int zloc = iz - z_t0;
        int yloc = iy - y_t0;

        int idx =
            ((((it * ctx.B + b) * nz_boundary + zloc)
              * ny_boundary + yloc)
              * width + xloc);

        if (mode == BOUNDARY_SAVE)
            left[idx] = val;
        else
            u_b[idx3] = left[idx];
    }

    // ======================================================
    // X+ (right)
    // ======================================================

    if (is_right)
    {
        int xloc = ix - right_start;
        int zloc = iz - z_t0;
        int yloc = iy - y_t0;

        int idx =
            ((((it * ctx.B + b) * nz_boundary + zloc)
              * ny_boundary + yloc)
              * width + xloc);

        if (mode == BOUNDARY_SAVE)
            right[idx] = val;
        else
            u_b[idx3] = right[idx];
    }
}
