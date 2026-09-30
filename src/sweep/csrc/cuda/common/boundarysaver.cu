#include "context.h"
#include "boundary/kernels.cuh"
#include "boundary/strip.cuh"
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

    // Band geometry and per-face membership come from boundary/strip.cuh, the
    // one definition shared with the strip-source un-injection
    // (sub_source_in_restore_strip), which must agree with this restore about
    // every cell.  DD cut faces (ctx.cut_mask(): bit0=x_lo/left, bit1=x_hi/right,
    // bit2=z_lo/top, bit3=z_hi/bottom) carry no band: the strip there is
    // reverse-leapfrog-computed + halo-exchanged instead of restored.
    const BsStripBands2D g = bs_strip_bands_2d(ctx, width, offset, tangent_pad);
    const BsStripFaces2D f = bs_strip_faces_2d(ctx, g, ix, iz);
    if (!f.any())
        return;

    const bool is_top = f.top, is_bottom = f.bottom, is_left = f.left, is_right = f.right;
    const int nx_boundary = g.nx_boundary, nz_boundary = g.nz_boundary;
    const int x_t0 = g.x_t0, z_t0 = g.z_t0;
    const int top_start = g.top_start, bot_start = g.bot_start;
    const int left_start = g.left_start, right_start = g.right_start;

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

    // Geometry: boundary/strip.cuh (see boundary_kernel2d).
    const BsStripBands2D g = bs_strip_bands_2d(ctx, width, offset, tangent_pad);
    const BsStripFaces2D f = bs_strip_faces_2d(ctx, g, ix, iz);
    if (!f.any()) return;
    const bool is_top = f.top, is_bottom = f.bottom, is_left = f.left, is_right = f.right;
    const int nx_boundary = g.nx_boundary, nz_boundary = g.nz_boundary;
    const int x_t0 = g.x_t0, z_t0 = g.z_t0;
    const int top_start = g.top_start, bot_start = g.bot_start;
    const int left_start = g.left_start, right_start = g.right_start;
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

    // Band geometry and per-face membership: boundary/strip.cuh, shared with
    // the strip-source un-injection (see boundary_kernel2d).  DD cut faces
    // (ctx.cut_mask()) carry no band.
    const BsStripBands3D g = bs_strip_bands_3d(ctx, width, offset, tangent_pad);
    const BsStripFaces3D f = bs_strip_faces_3d(ctx, g, ix, iy, iz);
    if (!f.any())
        return;

    const bool is_top = f.top, is_bottom = f.bottom, is_front = f.front;
    const bool is_back = f.back, is_left = f.left, is_right = f.right;
    const int nx_boundary = g.nx_boundary, ny_boundary = g.ny_boundary, nz_boundary = g.nz_boundary;
    const int x_t0 = g.x_t0, y_t0 = g.y_t0, z_t0 = g.z_t0;
    const int top_start = g.top_start, bot_start = g.bot_start;
    const int front_start = g.front_start, back_start = g.back_start;
    const int left_start = g.left_start, right_start = g.right_start;

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

    // Geometry: boundary/strip.cuh (see boundary_kernel3d).
    const BsStripBands3D g = bs_strip_bands_3d(ctx, width, offset, tangent_pad);
    const BsStripFaces3D f = bs_strip_faces_3d(ctx, g, ix, iy, iz);
    if (!f.any()) return;
    const bool is_top = f.top, is_bottom = f.bottom, is_front = f.front;
    const bool is_back = f.back, is_left = f.left, is_right = f.right;
    const int nx_boundary = g.nx_boundary, ny_boundary = g.ny_boundary, nz_boundary = g.nz_boundary;
    const int x_t0 = g.x_t0, y_t0 = g.y_t0, z_t0 = g.z_t0;
    const int top_start = g.top_start, bot_start = g.bot_start;
    const int front_start = g.front_start, back_start = g.back_start;
    const int left_start = g.left_start, right_start = g.right_start;

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

// One thread per boundary-band cell, covering exactly what the full-grid
// boundary_kernel2d covers -- including the corner cells that belong to both a
// horizontal and a vertical band.  In the scan kernel one thread writes such a
// cell into two buffers; here they are two threads carrying the same value, so
// SAVE is identical.  On RESTORE the two threads write the same wavefield cell,
// but both read the value the same SAVE wrote, so the (unordered) duplicate
// write is bit-identical either way -- which is why this path is FP32 only:
// under a lossy storage dtype the two copies could differ.
__global__ void boundary_kernel2d_compact(
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int nx_boundary = ctx.nx_phys() + 2 * tangent_pad;
    int nz_boundary = ctx.nz_phys() + 2 * tangent_pad;

    int tb_count = width * nx_boundary;
    int lr_count = nz_boundary * width;
    int per_batch = 2 * tb_count + 2 * lr_count;
    int total = ctx.B * per_batch;
    if (tid >= total)
        return;

    int b = tid / per_batch;
    int local = tid - b * per_batch;

    // Band bounds: boundary/strip.cuh.  Enumerating the in-grid cells of the
    // non-cut bands visits exactly the cells bs_in_restore_strip_2d accepts.
    const BsStripBands2D g = bs_strip_bands_2d(ctx, width, offset, tangent_pad);
    int x_t0 = g.x_t0;
    int z_t0 = g.z_t0;
    int top_start = g.top_start;
    int bot_start = g.bot_start;
    int left_start = g.left_start;
    int right_start = g.right_start;

    int ix = 0;
    int iz = 0;
    int64_t idx = 0;
    float* boundary = nullptr;

    // DD cut faces carry neighbour data supplied by HaloExchange, not saved
    // boundary values; the scan kernel drops them via !ctx.cut_*(), so the
    // matching band is skipped here.  Their buffer slots are never written,
    // so restoring from them would read uninitialised memory.
    if (local < tb_count) {
        if (ctx.cut_z_lo()) return;
        int zloc = local / nx_boundary;
        int xloc = local - zloc * nx_boundary;
        iz = top_start + zloc;
        ix = x_t0 + xloc;
        idx = ((static_cast<int64_t>(it) * ctx.B + b) * width + zloc) * nx_boundary + xloc;
        boundary = top;
    } else if (local < 2 * tb_count) {
        if (ctx.cut_z_hi()) return;
        int rem = local - tb_count;
        int zloc = rem / nx_boundary;
        int xloc = rem - zloc * nx_boundary;
        iz = bot_start + zloc;
        ix = x_t0 + xloc;
        idx = ((static_cast<int64_t>(it) * ctx.B + b) * width + zloc) * nx_boundary + xloc;
        boundary = bottom;
    } else if (local < 2 * tb_count + lr_count) {
        if (ctx.cut_x_lo()) return;
        int rem = local - 2 * tb_count;
        int zloc = rem / width;
        int xloc = rem - zloc * width;
        ix = left_start + xloc;
        iz = z_t0 + zloc;
        idx = ((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * width + xloc;
        boundary = left;
    } else {
        if (ctx.cut_x_hi()) return;
        int rem = local - 2 * tb_count - lr_count;
        int zloc = rem / width;
        int xloc = rem - zloc * width;
        ix = right_start + xloc;
        iz = z_t0 + zloc;
        idx = ((static_cast<int64_t>(it) * ctx.B + b) * nz_boundary + zloc) * width + xloc;
        boundary = right;
    }

    // The scan kernel's threads are grid-derived, so a band cell outside the
    // allocated grid simply never had a thread; with a nonzero tangent_pad the
    // band can reach past an edge, and those slots stay unwritten there.  Drop
    // the same cells here instead of running off the wavefield.
    if (ix < 0 || ix >= ctx.nx || iz < 0 || iz >= ctx.nz)
        return;

    float* u_b = u + b * (ctx.nx * ctx.nz);
    if (mode == BOUNDARY_SAVE)
        boundary[idx] = u_b[iz * ctx.nx + ix];
    else
        u_b[iz * ctx.nx + ix] = boundary[idx];
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

    int nx_boundary = ctx.nx_phys() + 2 * tangent_pad;
    int ny_boundary = ctx.ny_phys() + 2 * tangent_pad;
    int nz_boundary = ctx.nz_phys() + 2 * tangent_pad;

    int top_count = width * ny_boundary * nx_boundary;
    int front_count = nz_boundary * width * nx_boundary;
    int left_count = nz_boundary * ny_boundary * width;
    int per_batch = 2 * top_count + 2 * front_count + 2 * left_count;
    int total = ctx.B * per_batch;
    if (tid >= total)
        return;

    int b = tid / per_batch;
    int local = tid - b * per_batch;

    // Band bounds: boundary/strip.cuh.  Enumerating the in-grid cells of the
    // non-cut bands visits exactly the cells bs_in_restore_strip_3d accepts.
    const BsStripBands3D g = bs_strip_bands_3d(ctx, width, offset, tangent_pad);
    int x_t0 = g.x_t0;
    int y_t0 = g.y_t0;
    int z_t0 = g.z_t0;
    int top_start = g.top_start;
    int bot_start = g.bot_start;
    int front_start = g.front_start;
    int back_start = g.back_start;
    int left_start = g.left_start;
    int right_start = g.right_start;

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
        if (ctx.cut_z_lo()) return;
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
        if (ctx.cut_z_hi()) return;
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
        if (ctx.cut_y_lo()) return;
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
        if (ctx.cut_y_hi()) return;
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
        if (ctx.cut_x_lo()) return;
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
        if (ctx.cut_x_hi()) return;
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

    int nx_boundary = ctx.nx_phys() + 2 * tangent_pad;
    int ny_boundary = ctx.ny_phys() + 2 * tangent_pad;
    int nz_boundary = ctx.nz_phys() + 2 * tangent_pad;
    int top_count = width * ny_boundary * nx_boundary;
    int front_count = nz_boundary * width * nx_boundary;
    int left_count = nz_boundary * ny_boundary * width;
    int per_batch = 2 * top_count + 2 * front_count + 2 * left_count;
    int total = ctx.B * per_batch;
    if (tid >= total) return;
    int b = tid / per_batch;
    int local = tid - b * per_batch;
    // Band bounds: boundary/strip.cuh (see boundary_kernel3d_compact).
    const BsStripBands3D g = bs_strip_bands_3d(ctx, width, offset, tangent_pad);
    int x_t0 = g.x_t0, y_t0 = g.y_t0, z_t0 = g.z_t0;
    int top_start = g.top_start;
    int bot_start = g.bot_start;
    int front_start = g.front_start;
    int back_start = g.back_start;
    int left_start = g.left_start;
    int right_start = g.right_start;

    int ix = 0, iy = 0, iz = 0;
    int64_t idx = 0;
    T* boundary = nullptr;

    if (local < top_count) {
        int plane = ny_boundary * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = top_start + zloc; iy = y_t0 + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc;
        if (ctx.cut_z_lo()) return;
        boundary = top;
    } else if (local < 2 * top_count) {
        local -= top_count;
        int plane = ny_boundary * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = bot_start + zloc; iy = y_t0 + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * width + zloc) * ny_boundary + yloc) * nx_boundary + xloc;
        if (ctx.cut_z_hi()) return;
        boundary = bottom;
    } else if (local < 2 * top_count + front_count) {
        local -= 2 * top_count;
        int plane = width * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = z_t0 + zloc; iy = front_start + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc;
        if (ctx.cut_y_lo()) return;
        boundary = front;
    } else if (local < 2 * top_count + 2 * front_count) {
        local -= 2 * top_count + front_count;
        int plane = width * nx_boundary;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / nx_boundary; int xloc = rem - yloc * nx_boundary;
        iz = z_t0 + zloc; iy = back_start + yloc; ix = x_t0 + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * width + yloc) * nx_boundary + xloc;
        if (ctx.cut_y_hi()) return;
        boundary = back;
    } else if (local < 2 * top_count + 2 * front_count + left_count) {
        local -= 2 * top_count + 2 * front_count;
        int plane = ny_boundary * width;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / width; int xloc = rem - yloc * width;
        iz = z_t0 + zloc; iy = y_t0 + yloc; ix = left_start + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc;
        if (ctx.cut_x_lo()) return;
        boundary = left;
    } else {
        local -= 2 * top_count + 2 * front_count + left_count;
        int plane = ny_boundary * width;
        int zloc = local / plane; int rem = local - zloc * plane;
        int yloc = rem / width; int xloc = rem - yloc * width;
        iz = z_t0 + zloc; iy = y_t0 + yloc; ix = right_start + xloc;
        idx = ((((int64_t)it * ctx.B + b) * nz_boundary + zloc) * ny_boundary + yloc) * width + xloc;
        if (ctx.cut_x_hi()) return;
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
