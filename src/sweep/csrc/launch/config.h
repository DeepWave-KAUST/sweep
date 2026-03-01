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
        dim3 block(16, 8, 4);
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