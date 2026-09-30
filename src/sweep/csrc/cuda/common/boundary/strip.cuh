#pragma once

#include <cuda_runtime.h>

#include "../context.h"

// ====================================================================
// Boundary-saving strip geometry -- ONE definition of which cells a strip
// save / restore touches.
//
// Users:
//   * the full-grid save/restore kernels (boundary_kernel2d/3d and their
//     fp16 / bf16 bodies, boundarysaver.cu) take the per-face membership
//     from bs_strip_faces_2d/3d;
//   * the band-enumerating compact kernels (boundary_kernel2d/3d_compact*)
//     take the band bounds from bs_strip_bands_2d/3d -- they visit exactly
//     the in-grid cells of the non-cut bands, i.e. the same set;
//   * sub_source_in_restore_strip{,_3d} (common.cu) asks
//     bs_in_restore_strip_2d/3d whether a source cell was just overwritten by
//     the restore.  It must agree with the restore kernels cell for cell, so
//     it calls the very predicate the full-grid kernels branch on.
//
// Geometry: every face of the physical box carries a band ``width`` cells
// wide starting ``offset`` cells inside the face (ElasticTTI2nd passes
// offset = -M, so its band reaches M cells out into the pad), spanning the
// tangential axes ``tangent_pad`` cells past the box.  DD cut faces
// (ctx.cut_*()) carry no band: the strip there is reverse-leapfrog-computed
// + halo-exchanged instead of restored.  A free-surface face is an ordinary
// face here -- its band starts at phys_*0(), which already sits above the
// image rows.
// ====================================================================

// A saved band may be asked to reach further outward than the array allows: a
// free-surface face has no damping pad, so only the M halo cells exist outside
// the physical box, while an equation with an imaging buffer (VRZ,
// BOUNDARY_BUFFER_REACH) asks for a wider reach.  Slide the band back inside
// instead of indexing out of bounds.  No-op whenever the band already fits --
// every face with pad >= |offset|, i.e. everything except such a free-surface
// face.  From fix/vrz-imaging-and-shell (d972dc74), moved here so the save,
// the restore and the strip-source un-injection keep sharing ONE band.
__host__ __device__ __forceinline__ int bs_band_lo(int p0, int offset, int width, int n)
{
    int s = p0 + offset;
    if (s + width > n) s = n - width;
    return s < 0 ? 0 : s;
}

__host__ __device__ __forceinline__ int bs_band_hi(int p1, int offset, int width, int n)
{
    int e = p1 - offset;
    if (e - width < 0) e = width;
    return e > n ? n : e;
}

struct BsStripBands2D {
    int nx_boundary, nz_boundary;          // tangential extent of a band
    int x_t0, x_t1, z_t0, z_t1;            // tangential range [t0, t1)
    int top_start, top_end;                // z range of the top band
    int bot_start, bot_end;                // z range of the bottom band
    int left_start, left_end;              // x range of the left band
    int right_start, right_end;            // x range of the right band
};

__device__ __forceinline__ BsStripBands2D bs_strip_bands_2d(
    const SolverContext& ctx, int width, int offset, int tangent_pad)
{
    BsStripBands2D g;
    const int x0 = ctx.phys_x0(), x1 = ctx.phys_x1();
    const int z0 = ctx.phys_z0(), z1 = ctx.phys_z1();
    g.nx_boundary = ctx.nx_phys() + 2 * tangent_pad;
    g.nz_boundary = ctx.nz_phys() + 2 * tangent_pad;
    g.x_t0 = x0 - tangent_pad;  g.x_t1 = x1 + tangent_pad;
    g.z_t0 = z0 - tangent_pad;  g.z_t1 = z1 + tangent_pad;
    g.top_start   = bs_band_lo(z0, offset, width, ctx.nz);  g.top_end   = g.top_start + width;
    g.bot_end     = bs_band_hi(z1, offset, width, ctx.nz);  g.bot_start = g.bot_end - width;
    g.left_start  = bs_band_lo(x0, offset, width, ctx.nx);  g.left_end  = g.left_start + width;
    g.right_end   = bs_band_hi(x1, offset, width, ctx.nx);  g.right_start = g.right_end - width;
    return g;
}

struct BsStripFaces2D {
    bool top, bottom, left, right;
    __device__ __forceinline__ bool any() const { return top || bottom || left || right; }
};

// Per-face band membership of cell (ix, iz); a corner cell is in two bands.
// No grid-bounds test: the full-grid kernels only launch in-grid threads.
__device__ __forceinline__ BsStripFaces2D bs_strip_faces_2d(
    const SolverContext& ctx, const BsStripBands2D& g, int ix, int iz)
{
    BsStripFaces2D f;
    f.top    = iz >= g.top_start   && iz < g.top_end   && ix >= g.x_t0 && ix < g.x_t1 && !ctx.cut_z_lo();
    f.bottom = iz >= g.bot_start   && iz < g.bot_end   && ix >= g.x_t0 && ix < g.x_t1 && !ctx.cut_z_hi();
    f.left   = ix >= g.left_start  && ix < g.left_end  && iz >= g.z_t0 && iz < g.z_t1 && !ctx.cut_x_lo();
    f.right  = ix >= g.right_start && ix < g.right_end && iz >= g.z_t0 && iz < g.z_t1 && !ctx.cut_x_hi();
    return f;
}

// Does a strip restore with (width, offset, tangent_pad) overwrite (ix, iz)?
__device__ __forceinline__ bool bs_in_restore_strip_2d(
    const SolverContext& ctx, int width, int offset, int tangent_pad, int ix, int iz)
{
    if (ix < 0 || ix >= ctx.nx || iz < 0 || iz >= ctx.nz)
        return false;
    return bs_strip_faces_2d(ctx, bs_strip_bands_2d(ctx, width, offset, tangent_pad), ix, iz).any();
}

struct BsStripBands3D {
    int nx_boundary, ny_boundary, nz_boundary;
    int x_t0, x_t1, y_t0, y_t1, z_t0, z_t1;
    int top_start, top_end;                // z
    int bot_start, bot_end;                // z
    int front_start, front_end;            // y
    int back_start, back_end;              // y
    int left_start, left_end;              // x
    int right_start, right_end;            // x
};

__device__ __forceinline__ BsStripBands3D bs_strip_bands_3d(
    const SolverContext& ctx, int width, int offset, int tangent_pad)
{
    BsStripBands3D g;
    const int x0 = ctx.phys_x0(), x1 = ctx.phys_x1();
    const int y0 = ctx.phys_y0(), y1 = ctx.phys_y1();
    const int z0 = ctx.phys_z0(), z1 = ctx.phys_z1();
    g.nx_boundary = ctx.nx_phys() + 2 * tangent_pad;
    g.ny_boundary = ctx.ny_phys() + 2 * tangent_pad;
    g.nz_boundary = ctx.nz_phys() + 2 * tangent_pad;
    g.x_t0 = x0 - tangent_pad;  g.x_t1 = x1 + tangent_pad;
    g.y_t0 = y0 - tangent_pad;  g.y_t1 = y1 + tangent_pad;
    g.z_t0 = z0 - tangent_pad;  g.z_t1 = z1 + tangent_pad;
    g.top_start   = bs_band_lo(z0, offset, width, ctx.nz);  g.top_end   = g.top_start + width;
    g.bot_end     = bs_band_hi(z1, offset, width, ctx.nz);  g.bot_start = g.bot_end - width;
    g.front_start = bs_band_lo(y0, offset, width, ctx.ny);  g.front_end = g.front_start + width;
    g.back_end    = bs_band_hi(y1, offset, width, ctx.ny);  g.back_start = g.back_end - width;
    g.left_start  = bs_band_lo(x0, offset, width, ctx.nx);  g.left_end  = g.left_start + width;
    g.right_end   = bs_band_hi(x1, offset, width, ctx.nx);  g.right_start = g.right_end - width;
    return g;
}

struct BsStripFaces3D {
    bool top, bottom, front, back, left, right;
    __device__ __forceinline__ bool any() const
    { return top || bottom || front || back || left || right; }
};

__device__ __forceinline__ BsStripFaces3D bs_strip_faces_3d(
    const SolverContext& ctx, const BsStripBands3D& g, int ix, int iy, int iz)
{
    BsStripFaces3D f;
    f.top    = iz >= g.top_start   && iz < g.top_end   && iy >= g.y_t0 && iy < g.y_t1 && ix >= g.x_t0 && ix < g.x_t1 && !ctx.cut_z_lo();
    f.bottom = iz >= g.bot_start   && iz < g.bot_end   && iy >= g.y_t0 && iy < g.y_t1 && ix >= g.x_t0 && ix < g.x_t1 && !ctx.cut_z_hi();
    f.front  = iy >= g.front_start && iy < g.front_end && iz >= g.z_t0 && iz < g.z_t1 && ix >= g.x_t0 && ix < g.x_t1 && !ctx.cut_y_lo();
    f.back   = iy >= g.back_start  && iy < g.back_end  && iz >= g.z_t0 && iz < g.z_t1 && ix >= g.x_t0 && ix < g.x_t1 && !ctx.cut_y_hi();
    f.left   = ix >= g.left_start  && ix < g.left_end  && iz >= g.z_t0 && iz < g.z_t1 && iy >= g.y_t0 && iy < g.y_t1 && !ctx.cut_x_lo();
    f.right  = ix >= g.right_start && ix < g.right_end && iz >= g.z_t0 && iz < g.z_t1 && iy >= g.y_t0 && iy < g.y_t1 && !ctx.cut_x_hi();
    return f;
}

__device__ __forceinline__ bool bs_in_restore_strip_3d(
    const SolverContext& ctx, int width, int offset, int tangent_pad, int ix, int iy, int iz)
{
    if (ix < 0 || ix >= ctx.nx || iy < 0 || iy >= ctx.ny || iz < 0 || iz >= ctx.nz)
        return false;
    return bs_strip_faces_3d(ctx, bs_strip_bands_3d(ctx, width, offset, tangent_pad), ix, iy, iz).any();
}
