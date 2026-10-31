#pragma once
// launch/cell_path.h -- a stencil kernel split into "decode + guards" and a
// per-cell body, so the host core can run the interior as a SIMD loop.
//
// The 2-D stencil kernels have an interior fast path: away from the PML
// bands every cell runs the same straight-line update.  A kernel that opts in
// keeps its index decode and early-return guards in the __global__ function
// and moves everything after them, unchanged and in the same order, into a
//
//     template <int Order, CellPath Path = CellPath::Any>
//     __device__ __forceinline__ void <kernel>_cell(int ix, int iz, int b, ...)
//
// which the kernel calls with Path = Any.  Under nvcc that is the same code
// after inlining (the CUDA core stays bit-for-bit; the bitgate is the proof).
// The other two paths are the same body with its run-time tests answered by
// the caller:
//   * the cell lies in the kernel's interior box (a CellBox the kernel's own
//     test is written against), so the PML branch is dead code;
//   * the kernel's optional output -- the one pointer the body tests against
//     nullptr, a u_tt store or a gradient accumulator -- is absent
//     (Interior) or present (InteriorOut), so that store is either gone or
//     unconditional.
// What is left is branch-free, which is what lets the host compiler
// vectorise a row of such cells (host/launch.h launch_rows).
//
// Nothing else changes: the PML band and every edge cell still run through
// the kernel itself, one virtual thread at a time.
#include <cuda_runtime.h>

// A bit set: Any (0) is the kernel's own path; bit 0 says the cell is in the
// interior box, bit k + 1 that optional output k is present (k = 0 is the
// one most kernels have; a kernel with two -- acoustic3d's reconstruction
// keeps u_tt and images -- numbers them).  Interior / InteriorOut are the
// one-output spellings.
enum class CellPath : unsigned { Any = 0u, Interior = 1u, InteriorOut = 3u };

// The interior path with the optional outputs in ``mask`` (bit k = output k).
constexpr CellPath interior_with_outputs(unsigned mask)
{
    return static_cast<CellPath>(1u | (mask << 1));
}

// The caller promises the cell is in the kernel's interior box.
template <CellPath P>
constexpr bool interior_path = (static_cast<unsigned>(P) & 1u) != 0u;

// Bits 8 and up carry launch-uniform switches that an interior path needs at
// compile time, because they change the cell's arithmetic or guard its
// stores (which free-surface faces are on, say).  Each kernel family numbers
// its own: flag F is bit 8 + F.  The host launcher sets them
// (host/launch.h with_flags + add_flags); they are never set on Any, where
// the cell tests the run-time value as it always did.
__host__ __device__ constexpr CellPath add_flags(CellPath p, unsigned flags)
{
    return static_cast<CellPath>(static_cast<unsigned>(p) | (flags << 8));
}

template <CellPath P, int F>
constexpr bool path_flag = ((static_cast<unsigned>(P) >> (8 + F)) & 1u) != 0u;

// Is the kernel's optional output K there?  Tested on the kernel's own path,
// a constant on the interior ones.
template <CellPath P, int K = 0, class T>
__host__ __device__ inline bool optional_output(const T* p)
{
    if constexpr (P == CellPath::Any)
        return p != nullptr;
    else
        return ((static_cast<unsigned>(P) >> (K + 1)) & 1u) != 0u;
}

// Half-open box of cells [x0, x1) x [z0, z1); empty when either side is.
struct CellBox {
    int x0, x1, z0, z1;

    __host__ __device__ inline bool contains(int ix, int iz) const
    {
        return ix >= x0 && ix < x1 && iz >= z0 && iz < z1;
    }

    // !contains(), spelled the way the kernels' "outside" tests always were:
    // nvcc forms the predicates from the expression as written, and a kernel
    // must keep the code it had before it was split (same instructions, same
    // registers -- checked with cuobjdump, see docs/dev/host_core.md).
    __host__ __device__ inline bool outside(int ix, int iz) const
    {
        return ix < x0 || ix >= x1 || iz < z0 || iz >= z1;
    }

    __host__ __device__ inline CellBox intersect(const CellBox& o) const
    {
        return {x0 > o.x0 ? x0 : o.x0, x1 < o.x1 ? x1 : o.x1,
                z0 > o.z0 ? z0 : o.z0, z1 < o.z1 ? z1 : o.z1};
    }
};

// The 3-D twin: [x0, x1) x [y0, y1) x [z0, z1).
struct CellBox3 {
    int x0, x1, y0, y1, z0, z1;

    __host__ __device__ inline bool contains(int ix, int iy, int iz) const
    {
        return ix >= x0 && ix < x1 && iy >= y0 && iy < y1 && iz >= z0 && iz < z1;
    }

    // !contains(), spelled the way the 3-D kernels' "outside" tests are.
    __host__ __device__ inline bool outside(int ix, int iy, int iz) const
    {
        return ix < x0 || ix >= x1 || iy < y0 || iy >= y1 || iz < z0 || iz >= z1;
    }

    __host__ __device__ inline CellBox3 intersect(const CellBox3& o) const
    {
        return {x0 > o.x0 ? x0 : o.x0, x1 < o.x1 ? x1 : o.x1,
                y0 > o.y0 ? y0 : o.y0, y1 < o.y1 ? y1 : o.y1,
                z0 > o.z0 ? z0 : o.z0, z1 < o.z1 ? z1 : o.z1};
    }
};
