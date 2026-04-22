#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "dim.cuh"

struct GradParam {
    int sx;   // stride x
    int sy;   // stride y
    int sz;   // stride z

    int M;    // runtime half order
    const float* coeff;

    float dx;
    float dy;
    float dz;
};

template<int Order, int ND, int Direction>
struct GradientGPU;

template<int ND, int Order, int Direction>
__device__ __forceinline__
float gradient(
    const float* __restrict__ u,
    int ix, int iy, int iz,
    const GradParam& p
)
{
    if constexpr (Order == -1) {
        return GradientGPU<-1, ND, Direction>::eval(
            u,
            p.sx, p.sy, p.sz,
            ix, iy, iz,
            p.M, p.coeff,
            p.dx, p.dy, p.dz
        );
    } else {
        return GradientGPU<Order, ND, Direction>::eval(
            u,
            p.sx, p.sy, p.sz,
            ix, iy, iz,
            p.dx, p.dy, p.dz
        );
    }
}

// ============================================================
// 2nd order
// ============================================================
template<int ND, int Direction>
struct GradientGPU<2, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        int idx = ix*sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += (u[idx + sx] - u[idx - sx]) / (2.f * dx);

        if constexpr (ND == 3 && (Direction & Y))
            grad += (u[idx + sy] - u[idx - sy]) / (2.f * dy);

        if constexpr (Direction & Z)
            grad += (u[idx + sz] - u[idx - sz]) / (2.f * dz);

        return grad;
    }
};
// ============================================================
// 4th order
// ============================================================
template<int ND, int Direction>
struct GradientGPU<4, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        constexpr float c1 = 8.f/12.f;
        constexpr float c2 = -1.f/12.f;

        int idx = ix*sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += (c1*(u[idx+sx]-u[idx-sx])
                   + c2*(u[idx+2*sx]-u[idx-2*sx])) / dx;

        if constexpr (ND == 3 && (Direction & Y))
            grad += (c1*(u[idx+sy]-u[idx-sy])
                   + c2*(u[idx+2*sy]-u[idx-2*sy])) / dy;

        if constexpr (Direction & Z)
            grad += (c1*(u[idx+sz]-u[idx-sz])
                   + c2*(u[idx+2*sz]-u[idx-2*sz])) / dz;

        return grad;
    }
};
// ============================================================
// 6th order
// ============================================================
template<int ND, int Direction>
struct GradientGPU<6, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        constexpr float c1 = 3.f  / 4.f;
        constexpr float c2 = -3.f / 20.f;
        constexpr float c3 = 1.f  / 60.f;

        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += (c1*(u[idx+sx]-u[idx-sx]) +
                     c2*(u[idx+2*sx]-u[idx-2*sx]) +
                     c3*(u[idx+3*sx]-u[idx-3*sx])) / dx;

        if constexpr (ND == 3 && (Direction & Y))
            grad += (c1*(u[idx+sy]-u[idx-sy]) +
                     c2*(u[idx+2*sy]-u[idx-2*sy]) +
                     c3*(u[idx+3*sy]-u[idx-3*sy])) / dy;

        if constexpr (Direction & Z)
            grad += (c1*(u[idx+sz]-u[idx-sz]) +
                     c2*(u[idx+2*sz]-u[idx-2*sz]) +
                     c3*(u[idx+3*sz]-u[idx-3*sz])) / dz;

        return grad;
    }
};
// ============================================================
// 8th order
// ============================================================
template<int ND, int Direction>
struct GradientGPU<8, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        constexpr float c1 = 4.f   / 5.f;
        constexpr float c2 = -1.f  / 5.f;
        constexpr float c3 = 4.f   / 105.f;
        constexpr float c4 = -1.f  / 280.f;

        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += (c1*(u[idx+sx]-u[idx-sx]) +
                     c2*(u[idx+2*sx]-u[idx-2*sx]) +
                     c3*(u[idx+3*sx]-u[idx-3*sx]) +
                     c4*(u[idx+4*sx]-u[idx-4*sx])) / dx;

        if constexpr (ND == 3 && (Direction & Y))
            grad += (c1*(u[idx+sy]-u[idx-sy]) +
                     c2*(u[idx+2*sy]-u[idx-2*sy]) +
                     c3*(u[idx+3*sy]-u[idx-3*sy]) +
                     c4*(u[idx+4*sy]-u[idx-4*sy])) / dy;

        if constexpr (Direction & Z)
            grad += (c1*(u[idx+sz]-u[idx-sz]) +
                     c2*(u[idx+2*sz]-u[idx-2*sz]) +
                     c3*(u[idx+3*sz]-u[idx-3*sz]) +
                     c4*(u[idx+4*sz]-u[idx-4*sz])) / dz;

        return grad;
    }
};
// ============================================================
// 2n-th order
// ============================================================
template<int ND, int Direction>
struct GradientGPU<-1, ND, Direction> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        int M,
        const float* __restrict__ coeff,
        float dx, float dy, float dz
    )
    {
        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        if constexpr (Direction & X) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] *
                       (u[idx + k*sx] - u[idx - k*sx]);

            grad += acc / dx;
        }

        if constexpr (ND == 3 && (Direction & Y)) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] *
                       (u[idx + k*sy] - u[idx - k*sy]);

            grad += acc / dy;
        }

        if constexpr (Direction & Z) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] *
                       (u[idx + k*sz] - u[idx - k*sz]);

            grad += acc / dz;
        }

        return grad;
    }
};
