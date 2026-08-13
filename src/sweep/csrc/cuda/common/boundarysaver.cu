#include "context.h"
#include "boundary/kernels.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Storage-type helpers for FP32 / FP16 / BF16 boundary buffers.  Compute
// side (``u``) is always FP32; we only cast on the storage boundary.
__device__ __forceinline__ void
bndry_xfer(float* __restrict__ buf, int64_t idx,
           float* __restrict__ u_b, int idx3, int mode) {
    if (mode == BOUNDARY_SAVE) buf[idx]   = u_b[idx3];
    else                       u_b[idx3]  = buf[idx];
}
__device__ __forceinline__ void
bndry_xfer(__half* __restrict__ buf, int64_t idx,
           float* __restrict__ u_b, int idx3, int mode) {
    if (mode == BOUNDARY_SAVE) buf[idx]   = __float2half(u_b[idx3]);
    else                       u_b[idx3]  = __half2float(buf[idx]);
}
__device__ __forceinline__ void
bndry_xfer(__nv_bfloat16* __restrict__ buf, int64_t idx,
           float* __restrict__ u_b, int idx3, int mode) {
    if (mode == BOUNDARY_SAVE) buf[idx]   = __float2bfloat16(u_b[idx3]);
    else                       u_b[idx3]  = __bfloat162float(buf[idx]);
}

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

        int64_t idx =
            ((static_cast<int64_t>(it) * ctx.B + b) * width + zloc) * nx_boundary + xloc;

        if (mode == BOUNDARY_SAVE) top[idx] = val;
        else u_b[iz * ctx.nx + ix] = top[idx];
    }

    // BOTTOM
    if (is_bottom)
    {
        int zloc = iz - bot_start;
        int xloc = ix - x_t0;

        int64_t idx =
            ((static_cast<int64_t>(it) * ctx.B + b) * width + zloc) * nx_boundary + xloc;

        if (mode == BOUNDARY_SAVE) bottom[idx] = val;
        else u_b[iz * ctx.nx + ix] = bottom[idx];
    }

    // LEFT
    if (is_left)
    {
        int xloc = ix - left_start;
        int zloc = iz - z_t0;

        int64_t idx =
            ((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * width + xloc;

        if (mode == BOUNDARY_SAVE) left[idx] = val;
        else u_b[iz * ctx.nx + ix] = left[idx];
    }
    
    // RIGHT
    if (is_right)
    {
        int xloc = ix - right_start;
        int zloc = iz - z_t0;

        int64_t idx =
            ((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * width + xloc;

        if (mode == BOUNDARY_SAVE) right[idx] = val;
        else u_b[iz * ctx.nx + ix] = right[idx];
    }
}

// Templated body for boundary_kernel2d.  Identical geometry; only the
// per-face storage type differs (float vs __half).
template<typename T>
__device__ __forceinline__ void boundary_kernel2d_body(
    float* __restrict__ u,
    T* __restrict__ top,    T* __restrict__ bottom,
    T* __restrict__ left,   T* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.y * blockDim.y + threadIdx.y;
    int b  = blockIdx.z;
    if (ix >= ctx.nx || iz >= ctx.nz) return;

    int spatial = ctx.nx * ctx.nz;
    float* u_b = u + b * spatial;

    int x0 = ctx.phys_x0(), x1 = ctx.phys_x1();
    int z0 = ctx.phys_z0(), z1 = ctx.phys_z1();
    int nx_phys = ctx.nx_phys();
    int nz_phys = ctx.nz_phys();
    int nx_boundary = nx_phys + 2 * tangent_pad;
    int nz_boundary = nz_phys + 2 * tangent_pad;
    int x_t0 = x0 - tangent_pad, x_t1 = x1 + tangent_pad;
    int z_t0 = z0 - tangent_pad, z_t1 = z1 + tangent_pad;

    int top_start = z0 + offset, top_end = top_start + width;
    int bot_end   = z1 - offset, bot_start = bot_end - width;
    int left_start = x0 + offset, left_end = left_start + width;
    int right_end  = x1 - offset, right_start = right_end - width;

    bool is_top    = iz >= top_start && iz < top_end  && ix >= x_t0 && ix < x_t1;
    bool is_bottom = iz >= bot_start && iz < bot_end  && ix >= x_t0 && ix < x_t1;
    bool is_left   = ix >= left_start && ix < left_end && iz >= z_t0 && iz < z_t1;
    bool is_right  = ix >= right_start && ix < right_end && iz >= z_t0 && iz < z_t1;

    if (!(is_top || is_bottom || is_left || is_right)) return;
    int idx3 = iz * ctx.nx + ix;

    if (is_top) {
        int zloc = iz - top_start, xloc = ix - x_t0;
        int64_t idx = ((int64_t)it * ctx.B + b) * width * nx_boundary
                    + (int64_t)zloc * nx_boundary + xloc;
        bndry_xfer(top, idx, u_b, idx3, mode);
    }
    if (is_bottom) {
        int zloc = iz - bot_start, xloc = ix - x_t0;
        int64_t idx = ((int64_t)it * ctx.B + b) * width * nx_boundary
                    + (int64_t)zloc * nx_boundary + xloc;
        bndry_xfer(bottom, idx, u_b, idx3, mode);
    }
    if (is_left) {
        int xloc = ix - left_start, zloc = iz - z_t0;
        int64_t idx = ((int64_t)it * ctx.B + b) * nz_boundary * width
                    + (int64_t)zloc * width + xloc;
        bndry_xfer(left, idx, u_b, idx3, mode);
    }
    if (is_right) {
        int xloc = ix - right_start, zloc = iz - z_t0;
        int64_t idx = ((int64_t)it * ctx.B + b) * nz_boundary * width
                    + (int64_t)zloc * width + xloc;
        bndry_xfer(right, idx, u_b, idx3, mode);
    }
}

__global__ void boundary_kernel2d_fp16(
    float* __restrict__ u,
    __half* __restrict__ top,    __half* __restrict__ bottom,
    __half* __restrict__ left,   __half* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    boundary_kernel2d_body<__half>(u, top, bottom, left, right,
                                   it, width, offset, ctx, mode, tangent_pad);
}

__global__ void boundary_kernel2d_bf16(
    float* __restrict__ u,
    __nv_bfloat16* __restrict__ top,    __nv_bfloat16* __restrict__ bottom,
    __nv_bfloat16* __restrict__ left,   __nv_bfloat16* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    boundary_kernel2d_body<__nv_bfloat16>(u, top, bottom, left, right,
                                          it, width, offset, ctx, mode, tangent_pad);
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

        int64_t idx =
            ((((static_cast<int64_t>(it) * ctx.B + b) * width + zloc)
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

        int64_t idx =
            ((((static_cast<int64_t>(it) * ctx.B + b) * width + zloc)
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

        int64_t idx =
            ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc)
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

        int64_t idx =
            ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc)
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

        int64_t idx =
            ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc)
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

        int64_t idx =
            ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc)
              * ny_boundary + yloc)
              * width + xloc);

        if (mode == BOUNDARY_SAVE)
            right[idx] = val;
        else
            u_b[idx3] = right[idx];
    }
}

// Templated body for boundary_kernel3d: storage type T is float (FP32)
// or __half (FP16).  Compute on u is always FP32.
template<typename T>
__device__ __forceinline__ void boundary_kernel3d_body(
    float* __restrict__ u,
    T* __restrict__ top,
    T* __restrict__ bottom,
    T* __restrict__ front,
    T* __restrict__ back,
    T* __restrict__ left,
    T* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz_global = blockIdx.z * blockDim.z + threadIdx.z;
    int b  = iz_global / ctx.nz;
    int iz = iz_global % ctx.nz;
    if (b >= ctx.B || ix >= ctx.nx || iy >= ctx.ny || iz >= ctx.nz) return;

    int stride_y = ctx.nx;
    int stride_z = ctx.nx * ctx.ny;
    int spatial  = ctx.nx * ctx.ny * ctx.nz;
    float* u_b = u + b * spatial;
    int idx3 = iz * stride_z + iy * stride_y + ix;

    int x0 = ctx.phys_x0(), x1 = ctx.phys_x1();
    int y0 = ctx.phys_y0(), y1 = ctx.phys_y1();
    int z0 = ctx.phys_z0(), z1 = ctx.phys_z1();
    int nx_phys = ctx.nx_phys();
    int ny_phys = ctx.ny_phys();
    int nz_phys = ctx.nz_phys();
    int nx_boundary = nx_phys + 2 * tangent_pad;
    int ny_boundary = ny_phys + 2 * tangent_pad;
    int nz_boundary = nz_phys + 2 * tangent_pad;
    int x_t0 = x0 - tangent_pad, x_t1 = x1 + tangent_pad;
    int y_t0 = y0 - tangent_pad, y_t1 = y1 + tangent_pad;
    int z_t0 = z0 - tangent_pad, z_t1 = z1 + tangent_pad;

    int top_start = z0 + offset, top_end = top_start + width;
    int bot_end   = z1 - offset, bot_start = bot_end - width;
    int front_start = y0 + offset, front_end = front_start + width;
    int back_end  = y1 - offset, back_start = back_end - width;
    int left_start = x0 + offset, left_end = left_start + width;
    int right_end = x1 - offset, right_start = right_end - width;

    bool is_top    = iz >= top_start  && iz < top_end  && iy >= y_t0 && iy < y_t1 && ix >= x_t0 && ix < x_t1;
    bool is_bottom = iz >= bot_start  && iz < bot_end  && iy >= y_t0 && iy < y_t1 && ix >= x_t0 && ix < x_t1;
    bool is_front  = iy >= front_start && iy < front_end && iz >= z_t0 && iz < z_t1 && ix >= x_t0 && ix < x_t1;
    bool is_back   = iy >= back_start  && iy < back_end  && iz >= z_t0 && iz < z_t1 && ix >= x_t0 && ix < x_t1;
    bool is_left   = ix >= left_start  && ix < left_end  && iz >= z_t0 && iz < z_t1 && iy >= y_t0 && iy < y_t1;
    bool is_right  = ix >= right_start && ix < right_end && iz >= z_t0 && iz < z_t1 && iy >= y_t0 && iy < y_t1;

    if (!(is_top || is_bottom || is_front || is_back || is_left || is_right)) return;

    if (is_top) {
        int zloc = iz - top_start, yloc = iy - y_t0, xloc = ix - x_t0;
        int64_t idx = ((((int64_t)it * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc;
        bndry_xfer(top, idx, u_b, idx3, mode);
    }
    if (is_bottom) {
        int zloc = iz - bot_start, yloc = iy - y_t0, xloc = ix - x_t0;
        int64_t idx = ((((int64_t)it * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc;
        bndry_xfer(bottom, idx, u_b, idx3, mode);
    }
    if (is_front) {
        int yloc = iy - front_start, zloc = iz - z_t0, xloc = ix - x_t0;
        int64_t idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc;
        bndry_xfer(front, idx, u_b, idx3, mode);
    }
    if (is_back) {
        int yloc = iy - back_start, zloc = iz - z_t0, xloc = ix - x_t0;
        int64_t idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc;
        bndry_xfer(back, idx, u_b, idx3, mode);
    }
    if (is_left) {
        int xloc = ix - left_start, zloc = iz - z_t0, yloc = iy - y_t0;
        int64_t idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc;
        bndry_xfer(left, idx, u_b, idx3, mode);
    }
    if (is_right) {
        int xloc = ix - right_start, zloc = iz - z_t0, yloc = iy - y_t0;
        int64_t idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc;
        bndry_xfer(right, idx, u_b, idx3, mode);
    }
}

__global__ void boundary_kernel3d_fp16(
    float* __restrict__ u,
    __half* __restrict__ top, __half* __restrict__ bottom,
    __half* __restrict__ front, __half* __restrict__ back,
    __half* __restrict__ left, __half* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    boundary_kernel3d_body<__half>(u, top, bottom, front, back, left, right,
                                   it, width, offset, ctx, mode, tangent_pad);
}

__global__ void boundary_kernel3d_bf16(
    float* __restrict__ u,
    __nv_bfloat16* __restrict__ top, __nv_bfloat16* __restrict__ bottom,
    __nv_bfloat16* __restrict__ front, __nv_bfloat16* __restrict__ back,
    __nv_bfloat16* __restrict__ left, __nv_bfloat16* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    boundary_kernel3d_body<__nv_bfloat16>(u, top, bottom, front, back, left, right,
                                          it, width, offset, ctx, mode, tangent_pad);
}

__global__ void boundary_kernel3d_compact(
    float* __restrict__ u,

    float* __restrict__ top,
    float* __restrict__ bottom,

    float* __restrict__ front,
    float* __restrict__ back,

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

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

    int top_count = width * ny_boundary * nx_boundary;
    int front_count = nz_boundary * width * nx_boundary;
    int left_count = nz_boundary * ny_boundary * width;
    int per_batch = 2 * top_count + 2 * front_count + 2 * left_count;
    int total = ctx.B * per_batch;
    if (tid >= total)
        return;

    int b = tid / per_batch;
    int local = tid - b * per_batch;

    int x_t0 = x0 - tangent_pad;
    int y_t0 = y0 - tangent_pad;
    int z_t0 = z0 - tangent_pad;
    int top_start = z0 + offset;
    int bot_end = z1 - offset;
    int bot_start = bot_end - width;
    int front_start = y0 + offset;
    int back_end = y1 - offset;
    int back_start = back_end - width;
    int left_start = x0 + offset;
    int right_end = x1 - offset;
    int right_start = right_end - width;

    int ix = 0;
    int iy = 0;
    int iz = 0;
    int64_t idx = 0;
    float* boundary = nullptr;

    if (local < top_count) {
        int plane = ny_boundary * nx_boundary;
        int zloc = local / plane;
        int rem = local - zloc * plane;
        int yloc = rem / nx_boundary;
        int xloc = rem - yloc * nx_boundary;
        iz = top_start + zloc;
        iy = y_t0 + yloc;
        ix = x_t0 + xloc;
        idx = ((((static_cast<int64_t>(it) * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc);
        boundary = top;
    } else if (local < 2 * top_count) {
        local -= top_count;
        int plane = ny_boundary * nx_boundary;
        int zloc = local / plane;
        int rem = local - zloc * plane;
        int yloc = rem / nx_boundary;
        int xloc = rem - yloc * nx_boundary;
        iz = bot_start + zloc;
        iy = y_t0 + yloc;
        ix = x_t0 + xloc;
        idx = ((((static_cast<int64_t>(it) * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc);
        boundary = bottom;
    } else if (local < 2 * top_count + front_count) {
        local -= 2 * top_count;
        int plane = width * nx_boundary;
        int zloc = local / plane;
        int rem = local - zloc * plane;
        int yloc = rem / nx_boundary;
        int xloc = rem - yloc * nx_boundary;
        iz = z_t0 + zloc;
        iy = front_start + yloc;
        ix = x_t0 + xloc;
        idx = ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc);
        boundary = front;
    } else if (local < 2 * top_count + 2 * front_count) {
        local -= 2 * top_count + front_count;
        int plane = width * nx_boundary;
        int zloc = local / plane;
        int rem = local - zloc * plane;
        int yloc = rem / nx_boundary;
        int xloc = rem - yloc * nx_boundary;
        iz = z_t0 + zloc;
        iy = back_start + yloc;
        ix = x_t0 + xloc;
        idx = ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc);
        boundary = back;
    } else if (local < 2 * top_count + 2 * front_count + left_count) {
        local -= 2 * top_count + 2 * front_count;
        int plane = ny_boundary * width;
        int zloc = local / plane;
        int rem = local - zloc * plane;
        int yloc = rem / width;
        int xloc = rem - yloc * width;
        iz = z_t0 + zloc;
        iy = y_t0 + yloc;
        ix = left_start + xloc;
        idx = ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc);
        boundary = left;
    } else {
        local -= 2 * top_count + 2 * front_count + left_count;
        int plane = ny_boundary * width;
        int zloc = local / plane;
        int rem = local - zloc * plane;
        int yloc = rem / width;
        int xloc = rem - yloc * width;
        iz = z_t0 + zloc;
        iy = y_t0 + yloc;
        ix = right_start + xloc;
        idx = ((((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc);
        boundary = right;
    }

    if (ix < 0 || ix >= ctx.nx || iy < 0 || iy >= ctx.ny || iz < 0 || iz >= ctx.nz)
        return;

    int idx3 = iz * ctx.nx * ctx.ny + iy * ctx.nx + ix;
    float* u_b = u + b * ctx.nx * ctx.ny * ctx.nz;
    if (mode == BOUNDARY_SAVE)
        boundary[idx] = u_b[idx3];
    else
        u_b[idx3] = boundary[idx];
}

// Templated body for boundary_kernel3d_compact: pick one of the six faces
// based on tid, then SAVE / LOAD with FP32 ↔ T cast at the storage boundary.
template<typename T>
__device__ __forceinline__ void boundary_kernel3d_compact_body(
    float* __restrict__ u,
    T* __restrict__ top, T* __restrict__ bottom,
    T* __restrict__ front, T* __restrict__ back,
    T* __restrict__ left, T* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int x0 = ctx.phys_x0(), x1 = ctx.phys_x1();
    int y0 = ctx.phys_y0(), y1 = ctx.phys_y1();
    int z0 = ctx.phys_z0(), z1 = ctx.phys_z1();
    int nx_phys = ctx.nx_phys(), ny_phys = ctx.ny_phys(), nz_phys = ctx.nz_phys();
    int nx_boundary = nx_phys + 2 * tangent_pad;
    int ny_boundary = ny_phys + 2 * tangent_pad;
    int nz_boundary = nz_phys + 2 * tangent_pad;
    int top_count = width * ny_boundary * nx_boundary;
    int front_count = nz_boundary * width * nx_boundary;
    int left_count = nz_boundary * ny_boundary * width;
    int per_batch = 2 * top_count + 2 * front_count + 2 * left_count;
    int total = ctx.B * per_batch;
    if (tid >= total) return;
    int b = tid / per_batch;
    int local = tid - b * per_batch;
    int x_t0 = x0 - tangent_pad, y_t0 = y0 - tangent_pad, z_t0 = z0 - tangent_pad;
    int top_start = z0 + offset;
    int bot_end = z1 - offset, bot_start = bot_end - width;
    int front_start = y0 + offset;
    int back_end = y1 - offset, back_start = back_end - width;
    int left_start = x0 + offset;
    int right_end = x1 - offset, right_start = right_end - width;

    int ix = 0, iy = 0, iz = 0;
    int64_t idx = 0;
    T* boundary = nullptr;

    if (local < top_count) {
        int plane = ny_boundary * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = top_start + zloc; iy = y_t0 + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc;
        boundary = top;
    } else if (local < 2 * top_count) {
        local -= top_count;
        int plane = ny_boundary * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = bot_start + zloc; iy = y_t0 + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc;
        boundary = bottom;
    } else if (local < 2 * top_count + front_count) {
        local -= 2 * top_count;
        int plane = width * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = z_t0 + zloc; iy = front_start + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc;
        boundary = front;
    } else if (local < 2 * top_count + 2 * front_count) {
        local -= 2 * top_count + front_count;
        int plane = width * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = z_t0 + zloc; iy = back_start + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc;
        boundary = back;
    } else if (local < 2 * top_count + 2 * front_count + left_count) {
        local -= 2 * top_count + 2 * front_count;
        int plane = ny_boundary * width;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / width; int xloc = rem - yloc * width;
        iz = z_t0 + zloc; iy = y_t0 + yloc; ix = left_start + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc;
        boundary = left;
    } else {
        local -= 2 * top_count + 2 * front_count + left_count;
        int plane = ny_boundary * width;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / width; int xloc = rem - yloc * width;
        iz = z_t0 + zloc; iy = y_t0 + yloc; ix = right_start + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc;
        boundary = right;
    }

    if (ix < 0 || ix >= ctx.nx || iy < 0 || iy >= ctx.ny || iz < 0 || iz >= ctx.nz) return;
    int idx3 = iz * ctx.nx * ctx.ny + iy * ctx.nx + ix;
    float* u_b = u + b * ctx.nx * ctx.ny * ctx.nz;
    bndry_xfer(boundary, idx, u_b, idx3, mode);
}

__global__ void boundary_kernel3d_compact_fp16(
    float* __restrict__ u,
    __half* __restrict__ top, __half* __restrict__ bottom,
    __half* __restrict__ front, __half* __restrict__ back,
    __half* __restrict__ left, __half* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    boundary_kernel3d_compact_body<__half>(u, top, bottom, front, back, left, right,
                                           it, width, offset, ctx, mode, tangent_pad);
}

__global__ void boundary_kernel3d_compact_bf16(
    float* __restrict__ u,
    __nv_bfloat16* __restrict__ top, __nv_bfloat16* __restrict__ bottom,
    __nv_bfloat16* __restrict__ front, __nv_bfloat16* __restrict__ back,
    __nv_bfloat16* __restrict__ left, __nv_bfloat16* __restrict__ right,
    int it, int width, int offset,
    SolverContext ctx, int mode, int tangent_pad)
{
    boundary_kernel3d_compact_body<__nv_bfloat16>(u, top, bottom, front, back, left, right,
                                                  it, width, offset, ctx, mode, tangent_pad);
}

// ====================================================================
// INT8 per-block symmetric quantization (DeepWave-style)
//
// Each thread block of BOUNDARY_INT8_BLOCK threads handles one quant
// block: 256 contiguous cells on the flattened spatial axis of one
// timestep slot of one face.  Block-local shared-memory reduction
// finds max(|val|), then one FP32 scale is written per block and
// every cell is quantized to uint8 with offset 128.
//
// scale layout: per face, scale buffer has shape (nt × nfields ×
// n_blocks_per_step) — same outer "step" index as the main buffer so
// the scale doesn't span timesteps (preserves dynamic range across
// the wavefield's full evolution).
//
// Compression ratio = 4·B / (B + 4) where B = BOUNDARY_INT8_BLOCK.
// B=256 → ratio = 1024/260 ≈ 3.94×.
// ====================================================================

__global__ __launch_bounds__(BOUNDARY_INT8_BLOCK) void
quantize_int8_kernel(const float* __restrict__ src,
                     uint8_t* __restrict__ dst,
                     float* __restrict__ scale,
                     int64_t total_cells)
{
    int tid = threadIdx.x;
    int64_t block_idx = blockIdx.x;
    int64_t cell_idx = block_idx * BOUNDARY_INT8_BLOCK + tid;

    float val = (cell_idx < total_cells) ? src[cell_idx] : 0.0f;

    __shared__ float s_max[BOUNDARY_INT8_BLOCK];
    s_max[tid] = fabsf(val);
    __syncthreads();

    // Tree reduction.
    for (int s = BOUNDARY_INT8_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = s_max[tid + s];
            if (other > s_max[tid]) s_max[tid] = other;
        }
        __syncthreads();
    }
    float max_abs = s_max[0];

    if (tid == 0) scale[block_idx] = max_abs;

    if (cell_idx < total_cells) {
        float q_scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 0.0f;
        int q = __float2int_rn(val * q_scale) + 128;
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        dst[cell_idx] = (uint8_t)q;
    }
}

__global__ __launch_bounds__(BOUNDARY_INT8_BLOCK) void
dequantize_int8_kernel(const uint8_t* __restrict__ src,
                       const float* __restrict__ scale,
                       float* __restrict__ dst,
                       int64_t total_cells)
{
    int64_t block_idx = blockIdx.x;
    int64_t cell_idx = block_idx * BOUNDARY_INT8_BLOCK + threadIdx.x;
    if (cell_idx >= total_cells) return;
    float s = scale[block_idx] * (1.0f / 127.0f);
    dst[cell_idx] = ((float)src[cell_idx] - 128.0f) * s;
}

// Host-side launchers — compute grid from total_cells.  Caller passes
// the stream so the launch joins the propagator's existing schedule.
void launch_quantize_int8(const float* src, uint8_t* dst, float* scale,
                          int64_t total_cells, cudaStream_t stream)
{
    int64_t n_blocks = (total_cells + BOUNDARY_INT8_BLOCK - 1) / BOUNDARY_INT8_BLOCK;
    quantize_int8_kernel<<<n_blocks, BOUNDARY_INT8_BLOCK, 0, stream>>>(
        src, dst, scale, total_cells);
}

void launch_dequantize_int8(const uint8_t* src, const float* scale,
                            float* dst, int64_t total_cells,
                            cudaStream_t stream)
{
    int64_t n_blocks = (total_cells + BOUNDARY_INT8_BLOCK - 1) / BOUNDARY_INT8_BLOCK;
    dequantize_int8_kernel<<<n_blocks, BOUNDARY_INT8_BLOCK, 0, stream>>>(
        src, scale, dst, total_cells);
}

// ====================================================================
// FP16 per-block scaled storage — same two-pass flow and scale layout
// as INT8, with a __half payload instead of uint8.
//
// A bare __float2half cast cannot store elastic boundary values: the
// wavefield mixes stresses (O(1e-2) here) and velocities (O(1e-8),
// stress / (rho·vp)), and everything below 2^-24 flushes to zero —
// which wipes the velocity faces entirely and corrupts the
// reconstructed gradient.  Normalizing each block by its max |val|
// moves the payload into [-1, 1], where fp16 keeps its full 10-bit
// mantissa (rel. ~2^-11, 16× finer than int8's 1/127) and only values
// >2^24 below their block max can underflow.
// ====================================================================

__global__ __launch_bounds__(BOUNDARY_INT8_BLOCK) void
quantize_fp16_kernel(const float* __restrict__ src,
                     __half* __restrict__ dst,
                     float* __restrict__ scale,
                     int64_t total_cells)
{
    int tid = threadIdx.x;
    int64_t block_idx = blockIdx.x;
    int64_t cell_idx = block_idx * BOUNDARY_INT8_BLOCK + tid;

    float val = (cell_idx < total_cells) ? src[cell_idx] : 0.0f;

    __shared__ float s_max[BOUNDARY_INT8_BLOCK];
    s_max[tid] = fabsf(val);
    __syncthreads();

    // Tree reduction.
    for (int s = BOUNDARY_INT8_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other = s_max[tid + s];
            if (other > s_max[tid]) s_max[tid] = other;
        }
        __syncthreads();
    }
    float max_abs = s_max[0];

    if (tid == 0) scale[block_idx] = max_abs;

    if (cell_idx < total_cells) {
        float inv = (max_abs > 0.0f) ? (1.0f / max_abs) : 0.0f;
        dst[cell_idx] = __float2half(val * inv);
    }
}

__global__ __launch_bounds__(BOUNDARY_INT8_BLOCK) void
dequantize_fp16_kernel(const __half* __restrict__ src,
                       const float* __restrict__ scale,
                       float* __restrict__ dst,
                       int64_t total_cells)
{
    int64_t block_idx = blockIdx.x;
    int64_t cell_idx = block_idx * BOUNDARY_INT8_BLOCK + threadIdx.x;
    if (cell_idx >= total_cells) return;
    dst[cell_idx] = __half2float(src[cell_idx]) * scale[block_idx];
}

void launch_quantize_fp16(const float* src, __half* dst, float* scale,
                          int64_t total_cells, cudaStream_t stream)
{
    int64_t n_blocks = (total_cells + BOUNDARY_INT8_BLOCK - 1) / BOUNDARY_INT8_BLOCK;
    quantize_fp16_kernel<<<n_blocks, BOUNDARY_INT8_BLOCK, 0, stream>>>(
        src, dst, scale, total_cells);
}

void launch_dequantize_fp16(const __half* src, const float* scale,
                            float* dst, int64_t total_cells,
                            cudaStream_t stream)
{
    int64_t n_blocks = (total_cells + BOUNDARY_INT8_BLOCK - 1) / BOUNDARY_INT8_BLOCK;
    dequantize_fp16_kernel<<<n_blocks, BOUNDARY_INT8_BLOCK, 0, stream>>>(
        src, scale, dst, total_cells);
}
