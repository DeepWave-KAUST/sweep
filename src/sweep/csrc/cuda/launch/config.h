#pragma once
#include <cuda_runtime.h>

namespace fdtd {

constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

struct LaunchConfig {
    dim3 block;
    dim3 grid;
};

// gridDim.y caps at 65535.  A block only one or two rows tall (the 32x1x8 boxes
// of the split stencil launches) would hit that long before the 32x4x2 Wave3D
// launch does, so trade z depth for y rows -- same thread count -- until the y
// extent fits.  The block shape never changes which cells a launch computes.
inline dim3 fit_grid_y(dim3 blk, int extent_y)
{
    while (ceil_div(extent_y, (int)blk.y) > 65535 && blk.z > 1) { blk.y *= 2; blk.z /= 2; }
    return blk;
}

// --------------------------------------------------
// 2D Wave
// --------------------------------------------------

struct Wave2D {

    static inline LaunchConfig make(int nx, int nz, int B) {
        dim3 block(32, 8);
        dim3 grid(
            ceil_div(nx, block.x),
            ceil_div(nz, block.y),
            B
        );
        return {block, grid};
    }
};

// --------------------------------------------------
// 3D Wave
// --------------------------------------------------

struct Wave3D {

    static inline LaunchConfig make(int nx, int ny, int nz, int B) {
        dim3 block(32, 4, 2);
        dim3 grid(
            ceil_div(nx, block.x),
            ceil_div(ny, block.y),
            ceil_div(nz, block.z) * B
        );
        return {block, grid};
    }
};

// --------------------------------------------------
// Geometry (Source and Receiver)
// --------------------------------------------------

struct Geom {

    static inline LaunchConfig make(int npoints, int B) {
        constexpr int threads = 256;
        dim3 block(threads);
        dim3 grid(B, ceil_div(npoints, threads));
        return {block, grid};
    }
};

} // namespace fdtd