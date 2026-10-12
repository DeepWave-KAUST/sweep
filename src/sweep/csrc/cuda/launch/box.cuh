#pragma once
// Box launches: a stencil kernel launched over a sub-box of the padded grid
// instead of all of it, the shot index folded into z.
//
// The split forward / adjoint steps (acoustic3d, acoustic_vrz3d) launch their
// interior -- no axis in a CPML band, compiled without the CPML path -- and the
// CPML shell around it separately.  This is the geometry those launchers share;
// which cells a box holds and what a kernel does there stays with the equation.
#include <cuda_runtime.h>
#include <algorithm>
#include "config.h"
#include "../common/context.h"

namespace fdtd {

// [xlo, xhi) x [ylo, yhi) x [zlo, zhi)
struct Box { int xlo, xhi, ylo, yhi, zlo, zhi; };

__host__ __device__ inline bool empty(const Box& r)
{
    return r.xhi <= r.xlo || r.yhi <= r.ylo || r.zhi <= r.zlo;
}

inline Box intersect(const Box& a, const Box& b)
{
    return Box{std::max(a.xlo, b.xlo), std::min(a.xhi, b.xhi),
               std::max(a.ylo, b.ylo), std::min(a.yhi, b.yhi),
               std::max(a.zlo, b.zlo), std::min(a.zhi, b.zhi)};
}

// The cells a step computes, [M, n - M) on every axis, within the context's x
// range [x_base, x_end()) (a DD phase-split launch).  In 2-D y is the one row
// [0, 1).
inline Box computed_box(const SolverContext& s)
{
    const bool flat = s.ndim == 2;
    return Box{std::max(s.x_base, s.M), std::min(s.x_end(), s.nx - s.M),
               flat ? 0 : s.M, flat ? 1 : s.ny - s.M, s.M, s.nz - s.M};
}

// The physical box shrunk by `shrink` cells on every face that carries a band;
// a DD cut face carries none and stays at the halo M.  shrink 0: the cells with
// no axis in its CPML band; M: the cells whose taps stay out of every band.
inline Box phys_box(const SolverContext& s, int shrink)
{
    const bool flat = s.ndim == 2;
    return Box{s.cut_x_lo() ? s.M : s.phys_x0() + shrink, s.cut_x_hi() ? s.nx - s.M : s.phys_x1() - shrink,
               flat ? 0 : s.cut_y_lo() ? s.M : s.phys_y0() + shrink,
               flat ? 1 : s.cut_y_hi() ? s.ny - s.M : s.phys_y1() - shrink,
               s.cut_z_lo() ? s.M : s.phys_z0() + shrink, s.cut_z_hi() ? s.nz - s.M : s.phys_z1() - shrink};
}

// Blocks are 32x1x8, deep in z, so the z-taps of a block's cells are reused in
// L1; a box only a few cells wide in x (a shell's x slab) takes 8x4x8.
const dim3 kBoxBlock(32, 1, 8), kNarrowBlock(8, 4, 8);

// Launch geometry over r: grid.z spans (zhi - zlo) * B, the block traded z for
// y rows when grid.y would pass 65535.  lead_x widens every row by that many
// cells before xlo (a row-aligned launch; those threads exit).
inline LaunchConfig box_launch(const Box& r, int B, dim3 blk, int lead_x = 0)
{
    const dim3 block = fit_grid_y(blk, r.yhi - r.ylo);
    return {block, dim3(ceil_div(r.xhi - r.xlo + lead_x, (int)block.x),
                        ceil_div(r.yhi - r.ylo, (int)block.y),
                        ceil_div((r.zhi - r.zlo) * B, (int)block.z))};
}

// This thread's cell of a box launch; false past the box in x, y or the shots.
__device__ __forceinline__ bool box_cell(const Box& r, int B, int& ix, int& iy, int& iz, int& b)
{
    const int nbz = r.zhi - r.zlo;
    ix = r.xlo + blockIdx.x * blockDim.x + threadIdx.x;
    iy = r.ylo + blockIdx.y * blockDim.y + threadIdx.y;
    const int izg = blockIdx.z * blockDim.z + threadIdx.z;
    b = izg / nbz;
    iz = r.zlo + izg % nbz;
    return b < B && ix < r.xhi && iy < r.yhi;
}

// outer minus inner (inner within outer) as up to six disjoint slabs, f(slab,
// x_slab) for each non-empty one: below / above inner in z (all of outer's x
// and y), then in y (inner's z rows), then left / right of it in x (inner's y
// and z rows: x_slab).  An empty inner leaves outer as the one slab.
template<class F>
inline void for_each_slab(const Box& outer, const Box& inner, F&& f)
{
    if (empty(inner)) {
        if (!empty(outer)) f(outer, false);
        return;
    }
    const Box slabs[6] = {
        {outer.xlo, outer.xhi, outer.ylo, outer.yhi, outer.zlo, inner.zlo},
        {outer.xlo, outer.xhi, outer.ylo, outer.yhi, inner.zhi, outer.zhi},
        {outer.xlo, outer.xhi, outer.ylo, inner.ylo, inner.zlo, inner.zhi},
        {outer.xlo, outer.xhi, inner.yhi, outer.yhi, inner.zlo, inner.zhi},
        {outer.xlo, inner.xlo, inner.ylo, inner.yhi, inner.zlo, inner.zhi},
        {inner.xhi, outer.xhi, inner.ylo, inner.yhi, inner.zlo, inner.zhi},
    };
    for (int i = 0; i < 6; ++i)
        if (!empty(slabs[i])) f(slabs[i], i >= 4);
}

// The slabs of for_each_slab as ONE launch: back to back, each x fastest, then
// the shots.  A 2-D ring's four strips are too thin to pay for a launch each
// on small grids.
struct SlabSet {
    Box slab[6];
    int start[7];   // first cell of each slab; start[n] = the cells of one shot
    int n;
};

inline SlabSet slab_set(const Box& outer, const Box& inner)
{
    SlabSet s{};
    for_each_slab(outer, inner, [&](const Box& r, bool) {
        s.slab[s.n] = r;
        s.start[s.n + 1] = s.start[s.n] + (r.xhi - r.xlo) * (r.yhi - r.ylo) * (r.zhi - r.zlo);
        ++s.n;
    });
    return s;
}

// 1-D blocks of 256 over the set's cells x B; launch only a non-empty set.
inline LaunchConfig slab_set_launch(const SlabSet& s, int B)
{
    return {dim3(256), dim3((unsigned)(((long)s.start[s.n] * B + 255) / 256))};
}

// This thread's cell of a slab-set launch; false past the last shot.  The slab
// is picked by unrolled compares, so the set stays in the parameter bank.
__device__ __forceinline__ bool slab_cell(const SlabSet& s, int B, int& ix, int& iy, int& iz, int& b)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const int per = s.start[s.n];
    b = (int)(t / per);
    if (b >= B) return false;
    int r = (int)(t - (long)b * per);
    Box q = s.slab[0];
    int base = 0;
    #pragma unroll
    for (int k = 1; k < 6; ++k)
        if (k < s.n && r >= s.start[k]) { q = s.slab[k]; base = s.start[k]; }
    r -= base;
    const int wx = q.xhi - q.xlo, wy = q.yhi - q.ylo;
    ix = q.xlo + r % wx;
    r /= wx;
    iy = q.ylo + r % wy;
    iz = q.zlo + r / wy;
    return true;
}

}  // namespace fdtd
