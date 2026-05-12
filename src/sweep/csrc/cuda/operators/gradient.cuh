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

template<int ND>
__device__ __forceinline__
int gradient_index(int sx, int sy, int sz, int ix, int iy, int iz)
{
    int idx = ix * sx;

    if constexpr (ND == 3)
        idx += iy * sy;

    idx += iz * sz;
    return idx;
}

struct GradientArrayAccessor {
    const float* __restrict__ u;
    int idx;
    int stride;

    __device__ __forceinline__
    float operator()(int offset) const
    {
        return u[idx + offset * stride];
    }
};

template<int Order, typename Accessor>
__device__ __forceinline__
float centered_gradient_stencil(
    const Accessor& value_at,
    int M,
    const float* __restrict__ coeff,
    float h
)
{
    if constexpr (Order == 2) {
        return (value_at(1) - value_at(-1)) / (2.f * h);
    } else if constexpr (Order == 4) {
        constexpr float c1 = 8.f / 12.f;
        constexpr float c2 = -1.f / 12.f;
        return (c1 * (value_at(1) - value_at(-1))
              + c2 * (value_at(2) - value_at(-2))) / h;
    } else if constexpr (Order == 6) {
        constexpr float c1 = 3.f / 4.f;
        constexpr float c2 = -3.f / 20.f;
        constexpr float c3 = 1.f / 60.f;
        return (c1 * (value_at(1) - value_at(-1))
              + c2 * (value_at(2) - value_at(-2))
              + c3 * (value_at(3) - value_at(-3))) / h;
    } else if constexpr (Order == 8) {
        constexpr float c1 = 4.f / 5.f;
        constexpr float c2 = -1.f / 5.f;
        constexpr float c3 = 4.f / 105.f;
        constexpr float c4 = -1.f / 280.f;
        return (c1 * (value_at(1) - value_at(-1))
              + c2 * (value_at(2) - value_at(-2))
              + c3 * (value_at(3) - value_at(-3))
              + c4 * (value_at(4) - value_at(-4))) / h;
    } else {
        float acc = 0.f;

        #pragma unroll 1
        for (int k = 1; k <= M; ++k)
            acc += coeff[k] * (value_at(k) - value_at(-k));

        return acc / h;
    }
}

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
        int idx = gradient_index<ND>(sx, sy, sz, ix, iy, iz);

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += centered_gradient_stencil<2>(
                GradientArrayAccessor{u, idx, sx},
                1,
                nullptr,
                dx
            );

        if constexpr (ND == 3 && (Direction & Y))
            grad += centered_gradient_stencil<2>(
                GradientArrayAccessor{u, idx, sy},
                1,
                nullptr,
                dy
            );

        if constexpr (Direction & Z)
            grad += centered_gradient_stencil<2>(
                GradientArrayAccessor{u, idx, sz},
                1,
                nullptr,
                dz
            );

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
        int idx = gradient_index<ND>(sx, sy, sz, ix, iy, iz);

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += centered_gradient_stencil<4>(
                GradientArrayAccessor{u, idx, sx},
                2,
                nullptr,
                dx
            );

        if constexpr (ND == 3 && (Direction & Y))
            grad += centered_gradient_stencil<4>(
                GradientArrayAccessor{u, idx, sy},
                2,
                nullptr,
                dy
            );

        if constexpr (Direction & Z)
            grad += centered_gradient_stencil<4>(
                GradientArrayAccessor{u, idx, sz},
                2,
                nullptr,
                dz
            );

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
        int idx = gradient_index<ND>(sx, sy, sz, ix, iy, iz);

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += centered_gradient_stencil<6>(
                GradientArrayAccessor{u, idx, sx},
                3,
                nullptr,
                dx
            );

        if constexpr (ND == 3 && (Direction & Y))
            grad += centered_gradient_stencil<6>(
                GradientArrayAccessor{u, idx, sy},
                3,
                nullptr,
                dy
            );

        if constexpr (Direction & Z)
            grad += centered_gradient_stencil<6>(
                GradientArrayAccessor{u, idx, sz},
                3,
                nullptr,
                dz
            );

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
        int idx = gradient_index<ND>(sx, sy, sz, ix, iy, iz);

        float grad = 0.f;

        if constexpr (Direction & X)
            grad += centered_gradient_stencil<8>(
                GradientArrayAccessor{u, idx, sx},
                4,
                nullptr,
                dx
            );

        if constexpr (ND == 3 && (Direction & Y))
            grad += centered_gradient_stencil<8>(
                GradientArrayAccessor{u, idx, sy},
                4,
                nullptr,
                dy
            );

        if constexpr (Direction & Z)
            grad += centered_gradient_stencil<8>(
                GradientArrayAccessor{u, idx, sz},
                4,
                nullptr,
                dz
            );

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
        int idx = gradient_index<ND>(sx, sy, sz, ix, iy, iz);

        float grad = 0.f;

        if constexpr (Direction & X) {
            grad += centered_gradient_stencil<-1>(
                GradientArrayAccessor{u, idx, sx},
                M,
                coeff,
                dx
            );
        }

        if constexpr (ND == 3 && (Direction & Y)) {
            grad += centered_gradient_stencil<-1>(
                GradientArrayAccessor{u, idx, sy},
                M,
                coeff,
                dy
            );
        }

        if constexpr (Direction & Z) {
            grad += centered_gradient_stencil<-1>(
                GradientArrayAccessor{u, idx, sz},
                M,
                coeff,
                dz
            );
        }

        return grad;
    }
};
