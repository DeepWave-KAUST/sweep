#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "dim.cuh"

struct LaplaceParam {
    int nx;
    int ny;        // 2D 时 =1
    int M;         // runtime half order
    const float* coeff;
    float dx;
    float dy;
    float dz;
};

template<int Order, int ND, int Direction>
struct LaplaceGPU;

template<int ND, int Order, int Direction>
__device__ __forceinline__
float laplace(
    const float* __restrict__ u,
    int ix, int iy, int iz,
    const LaplaceParam& p
)
{
    if constexpr (Order == -1) {
        return LaplaceGPU<-1, ND, Direction>::eval(
            u,
            p.nx, p.ny,
            ix, iy, iz,
            p.M,
            p.coeff,
            p.dx, p.dy, p.dz
        );
    } else {
        return LaplaceGPU<Order, ND, Direction>::eval(
            u,
            p.nx, p.ny,
            ix, iy, iz,
            p.dx, p.dy, p.dz
        );
    }
}

// ============================================================
// 2nd order
// ============================================================
template<int ND, int Direction>
struct LaplaceGPU<2, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        constexpr int stride_x = 1;
        int stride_y = nx;
        int stride_z = nx * ny;

        int idx = ix;

        if constexpr (ND == 3)
            idx += iy * stride_y;

        idx += iz * stride_z;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Direction & X)
            lap += (u[idx - stride_x] + u[idx + stride_x] - 2.f*u0)/(dx*dx);

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (u[idx - stride_y] + u[idx + stride_y] - 2.f * u0)
                   / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (u[idx - stride_z] + u[idx + stride_z] - 2.f * u0)
                   / (dz * dz);
        }

        return lap;
    }
};

// ============================================================
// 4th order
// ============================================================
template<int ND, int Direction>
struct LaplaceGPU<4, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        constexpr float c0 = 5.f / 2.f;
        constexpr float c1 = 4.f / 3.f;
        constexpr float c2 = -1.f / 12.f;

        constexpr int stride_x = 1;

        int stride_y = nx;
        int stride_z = nx * ny;

        int idx = ix;

        if constexpr (ND == 3)
            idx += iy * stride_y;

        idx += iz * stride_z;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Direction & X) {
            lap += (
                c1 * (u[idx - stride_x] + u[idx + stride_x]) +
                c2 * (u[idx - 2*stride_x] + u[idx + 2*stride_x]) -
                c0 * u0
            ) / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (
                c1 * (u[idx - stride_y] + u[idx + stride_y]) +
                c2 * (u[idx - 2*stride_y] + u[idx + 2*stride_y]) -
                c0 * u0
            ) / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (
                c1 * (u[idx - stride_z] + u[idx + stride_z]) +
                c2 * (u[idx - 2*stride_z] + u[idx + 2*stride_z]) -
                c0 * u0
            ) / (dz * dz);
        }

        return lap;
    }
};

// ============================================================
// 6th order
// ============================================================
template<int ND, int Direction>
struct LaplaceGPU<6, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        constexpr float c0 = 49.f / 18.f;
        constexpr float c1 = 3.f  / 2.f;
        constexpr float c2 = -3.f / 20.f;
        constexpr float c3 = 1.f  / 90.f;

        constexpr int stride_x = 1;

        int stride_y = nx;
        int stride_z = nx * ny;

        int idx = ix;

        if constexpr (ND == 3)
            idx += iy * stride_y;

        idx += iz * stride_z;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Direction & X) {
            lap += (
                c1 * (u[idx - stride_x]     + u[idx + stride_x]) +
                c2 * (u[idx - 2*stride_x]   + u[idx + 2*stride_x]) +
                c3 * (u[idx - 3*stride_x]   + u[idx + 3*stride_x]) -
                c0 * u0
            ) / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (
                c1 * (u[idx - stride_y]     + u[idx + stride_y]) +
                c2 * (u[idx - 2*stride_y]   + u[idx + 2*stride_y]) +
                c3 * (u[idx - 3*stride_y]   + u[idx + 3*stride_y]) -
                c0 * u0
            ) / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (
                c1 * (u[idx - stride_z]     + u[idx + stride_z]) +
                c2 * (u[idx - 2*stride_z]   + u[idx + 2*stride_z]) +
                c3 * (u[idx - 3*stride_z]   + u[idx + 3*stride_z]) -
                c0 * u0
            ) / (dz * dz);
        }

        return lap;
    }
};

// ============================================================
// 8th order
// ============================================================
template<int ND, int Direction>
struct LaplaceGPU<8, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        constexpr float c0 = 205.f / 72.f;
        constexpr float c1 = 8.f   / 5.f;
        constexpr float c2 = -1.f  / 5.f;
        constexpr float c3 = 8.f   / 315.f;
        constexpr float c4 = -1.f  / 560.f;

        constexpr int stride_x = 1;

        int stride_y = nx;
        int stride_z = nx * ny;

        int idx = ix;

        if constexpr (ND == 3)
            idx += iy * stride_y;

        idx += iz * stride_z;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Direction & X) {
            lap += (
                c1 * (u[idx - stride_x]     + u[idx + stride_x]) +
                c2 * (u[idx - 2*stride_x]   + u[idx + 2*stride_x]) +
                c3 * (u[idx - 3*stride_x]   + u[idx + 3*stride_x]) +
                c4 * (u[idx - 4*stride_x]   + u[idx + 4*stride_x]) -
                c0 * u0
            ) / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (
                c1 * (u[idx - stride_y]     + u[idx + stride_y]) +
                c2 * (u[idx - 2*stride_y]   + u[idx + 2*stride_y]) +
                c3 * (u[idx - 3*stride_y]   + u[idx + 3*stride_y]) +
                c4 * (u[idx - 4*stride_y]   + u[idx + 4*stride_y]) -
                c0 * u0
            ) / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (
                c1 * (u[idx - stride_z]     + u[idx + stride_z]) +
                c2 * (u[idx - 2*stride_z]   + u[idx + 2*stride_z]) +
                c3 * (u[idx - 3*stride_z]   + u[idx + 3*stride_z]) +
                c4 * (u[idx - 4*stride_z]   + u[idx + 4*stride_z]) -
                c0 * u0
            ) / (dz * dz);
        }

        return lap;
    }
};

// ============================================================
// general 2M order (runtime)
// ============================================================
template<int ND, int Direction>
struct LaplaceGPU<-1, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx, int ny,
        int ix, int iy, int iz,
        int M,
        const float* __restrict__ coeff,
        float dx, float dy, float dz
    )
    {
        constexpr int stride_x = 1;

        int stride_y = nx;
        int stride_z = nx * ny;

        int idx = ix;

        if constexpr (ND == 3)
            idx += iy * stride_y;

        idx += iz * stride_z;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Direction & X) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] * (u[idx + k*stride_x] +
                                   u[idx - k*stride_x]);

            lap += (acc - coeff[0] * u0) / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] * (u[idx + k*stride_y] +
                                   u[idx - k*stride_y]);

            lap += (acc - coeff[0] * u0) / (dy * dy);
        }

        if constexpr (Direction & Z) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] * (u[idx + k*stride_z] +
                                   u[idx - k*stride_z]);

            lap += (acc - coeff[0] * u0) / (dz * dz);
        }

        return lap;
    }
};