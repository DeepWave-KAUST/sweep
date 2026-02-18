#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

enum LaplaceDim : int {
    LAPLACE_X  = 1,
    LAPLACE_Z  = 2,
    LAPLACE_XZ = 3
};

template<int Order, int Dim>
struct LaplaceImpl;

struct Laplace2dContext {
    int nx;
    int ix;
    int iz;

    int M;
    const float* coeff;

    float dx;
    float dz;
};

template<int Order, int Dim>
__device__ __forceinline__
float laplace(
    const float* __restrict__ u,
    const Laplace2dContext& ctx
) {
    if constexpr (Order == -1) {
        return LaplaceImpl<-1, Dim>::eval(
            u,
            ctx.nx,
            ctx.ix,
            ctx.iz,
            ctx.M,
            ctx.coeff,
            ctx.dx,
            ctx.dz
        );
    } else {
        return LaplaceImpl<Order, Dim>::eval(
            u,
            ctx.nx,
            ctx.ix,
            ctx.iz,
            ctx.dx,
            ctx.dz
        );
    }
}

template<int Order, int Dim>
__device__ __forceinline__
float laplace(
    const float* __restrict__ u,
    int nx,
    int ix, int iz,
    float dx, float dz
) {
    static_assert(
        Order == 2 || Order == 4 || Order == 6 || Order == 8,
        "Unsupported Laplace order"
    );
    return LaplaceImpl<Order, Dim>::eval(u, nx, ix, iz, dx, dz);
}

template<int Dim>
__device__ __forceinline__
float laplace(
    const float* __restrict__ u,
    int nx,
    int ix, int iz,
    int M,
    const float* __restrict__ coeff,
    float dx, float dz
) {
    return LaplaceImpl<-1, Dim>::eval(
        u, nx, ix, iz,
        M, coeff,
        dx, dz
    );
}

// ============================================================
// 2nd order
// ============================================================
template<int Dim>
struct LaplaceImpl<2, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx,
        int ix, int iz,
        float dx, float dz
    ) {
        int idx = iz * nx + ix;
        float u0 = u[idx];
        float lap = 0.0f;

        if constexpr (Dim & LAPLACE_X) {
            lap += (u[idx - 1] + u[idx + 1] - 2.0f * u0) / (dx * dx);
        }
        if constexpr (Dim & LAPLACE_Z) {
            lap += (u[idx - nx] + u[idx + nx] - 2.0f * u0) / (dz * dz);
        }
        return lap;
    }
};

// ============================================================
// 4th order
// ============================================================
template<int Dim>
struct LaplaceImpl<4, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx,
        int ix, int iz,
        float dx, float dz
    ) {
        constexpr float c0 = 5.0f / 2.0f;
        constexpr float c1 = 4.0f / 3.0f;
        constexpr float c2 = -1.0f / 12.0f;

        int idx = iz * nx + ix;
        float u0 = u[idx];
        float lap = 0.0f;

        if constexpr (Dim & LAPLACE_X) {
            lap += (
                c1 * (u[idx - 1] + u[idx + 1]) +
                c2 * (u[idx - 2] + u[idx + 2]) -
                c0 * u0
            ) / (dx * dx);
        }
        
        if constexpr (Dim & LAPLACE_Z) {
            lap += (
                c1 * (u[idx - nx] + u[idx + nx]) +
                c2 * (u[idx - 2 * nx] + u[idx + 2 * nx]) -
                c0 * u0
            ) / (dz * dz);
        }
        return lap;
    }
};

// ============================================================
// 6th order
// ============================================================
template<int Dim>
struct LaplaceImpl<6, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx,
        int ix, int iz,
        float dx, float dz
    ) {
    constexpr float c0 = 49.0f / 18.0f;
    constexpr float c1 = 3.0f / 2.0f;
    constexpr float c2 = -3.0f / 20.0f;
    constexpr float c3 = 1.0f / 90.0f;

    int idx = iz * nx + ix;
    float u0 = u[idx];

    float lap = 0.0f;

    if constexpr (Dim & LAPLACE_X) {
        lap += (
            c1 * (u[idx - 1] + u[idx + 1]) +
            c2 * (u[idx - 2] + u[idx + 2]) +
            c3 * (u[idx - 3] + u[idx + 3]) -
            c0 * u0
        ) / (dx * dx);
    }

    if constexpr (Dim & LAPLACE_Z) {
        lap += (
            c1 * (u[idx - nx] + u[idx + nx]) +
            c2 * (u[idx - 2 * nx] + u[idx + 2 * nx]) +
            c3 * (u[idx - 3 * nx] + u[idx + 3 * nx]) -
            c0 * u0
        ) / (dz * dz);
    }

    return lap;
    }
};

// ============================================================
// 8th order
// ============================================================
template<int Dim>
struct LaplaceImpl<8, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx,
        int ix, int iz,
        float dx, float dz
    ) {
    constexpr float c0 = 205.0f / 72.0f;
    constexpr float c1 = 8.0f / 5.0f;
    constexpr float c2 = -1.0f / 5.0f;
    constexpr float c3 = 8.0f / 315.0f;
    constexpr float c4 = -1.0f / 560.0f;

    int idx = iz * nx + ix;
    float u0 = u[idx];

    float lap = 0.0f;

    if constexpr (Dim & LAPLACE_X) {
        lap += (
            c1 * (u[idx - 1] + u[idx + 1]) +
            c2 * (u[idx - 2] + u[idx + 2]) +
            c3 * (u[idx - 3] + u[idx + 3]) +
            c4 * (u[idx - 4] + u[idx + 4]) -
            c0 * u0
        ) / (dx * dx);
    }

    if constexpr (Dim & LAPLACE_Z) {
        lap += (
            c1 * (u[idx - nx] + u[idx + nx]) +
            c2 * (u[idx - 2 * nx] + u[idx + 2 * nx]) +
            c3 * (u[idx - 3 * nx] + u[idx + 3 * nx]) +
            c4 * (u[idx - 4 * nx] + u[idx + 4 * nx]) -
            c0 * u0
        ) / (dz * dz);
    }

    return lap;
    }
};

// ============================================================
// general 2M order (runtime)
// ============================================================
template<int Dim>
struct LaplaceImpl<-1, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int nx,
        int ix, int iz,
        int M,                    // runtime half order
        const float* __restrict__ coeff,
        float dx, float dz
    ) {
        int idx = iz * nx + ix;
        float u0 = u[idx];
        float lap = 0.0f;

        if constexpr (Dim & LAPLACE_X) {
            float acc = 0.0f;
            #pragma unroll 1
            for (int k = 1; k <= M; ++k) {
                float ck = coeff[k];
                acc += ck * (u[idx + k] + u[idx - k]);
            }
            acc = (acc - coeff[0] * u0) / (dx * dx);
            lap += acc;
        }

        if constexpr (Dim & LAPLACE_Z) {
            float acc = 0.0f;
            #pragma unroll 1
            for (int k = 1; k <= M; ++k) {
                float ck = coeff[k];
                acc += ck * (u[idx + k * nx] + u[idx - k * nx]);
            }
            acc = (acc - coeff[0] * u0) / (dz * dz);
            lap += acc;
        }

        return lap;
    }
};
