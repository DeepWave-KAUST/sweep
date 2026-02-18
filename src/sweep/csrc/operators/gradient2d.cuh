#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

enum GradDim : int {
    GRAD_X  = 1,
    GRAD_Z  = 2,
    GRAD_XZ = 3
};

template<int Order, int Dim>
struct GradientImpl;

struct GradContext {
    int sx;      // stride in x
    int sz;      // stride in z
    int ix;
    int iz;

    int M;      // runtime order (if needed)
    const float* __restrict__ coeff;

    float dx;
    float dz;
};

template<int Order, int Dim>
__device__ __forceinline__
float gradient(
    const float* __restrict__ u,
    const GradContext& ctx
) {
    if constexpr (Order == -1) {
        return GradientImpl<-1, Dim>::eval(
            u,
            ctx.sx,
            ctx.sz,
            ctx.ix,
            ctx.iz,
            ctx.M,
            ctx.coeff,
            ctx.dx,
            ctx.dz
        );
    } else {
        return GradientImpl<Order, Dim>::eval(
            u,
            ctx.sx,
            ctx.sz,
            ctx.ix,
            ctx.iz,
            ctx.dx,
            ctx.dz
        );
    }
}

template<int Order, int Dim>
__device__ __forceinline__
float gradient(
    const float* __restrict__ u,
    int stride_x,
    int stride_z,
    int ix, int iz,
    float dx, float dz
) {
    static_assert(
        Order == 2 || Order == 4 || Order == 6 || Order == 8,
        "Unsupported gradient order"
    );

    return GradientImpl<Order, Dim>::eval(
        u, stride_x, stride_z,
        ix, iz,
        dx, dz
    );
}

template<int Dim>
__device__ __forceinline__
float gradient(
    const float* __restrict__ u,
    int stride_x,
    int stride_z,
    int ix, int iz,
    int M,
    const float* __restrict__ coeff,
    float dx, float dz
) {
    return GradientImpl<-1, Dim>::eval(
        u,
        stride_x, stride_z,
        ix, iz,
        M, coeff,
        dx, dz
    );
}

// ============================================================
// 2nd order
// ============================================================
template<int Dim>
struct GradientImpl<2, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sz,
        int ix, int iz,
        float dx, float dz
    ) {
        int idx = iz * sz + ix * sx;
        float grad = 0.0f;

        if constexpr (Dim & GRAD_X) {
            grad += (
                u[idx + sx] -
                u[idx - sx]
            ) / (2.0f * dx);
        }

        if constexpr (Dim & GRAD_Z) {
            grad += (
                u[idx + sz] -
                u[idx - sz]
            ) / (2.0f * dz);
        }

        return grad;
    }
};



// ============================================================
// 4th order
// ============================================================
template<int Dim>
struct GradientImpl<4, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sz,
        int ix, int iz,
        float dx, float dz
    ) {
        constexpr float c1 = 8.0f / 12.0f;
        constexpr float c2 = -1.0f / 12.0f;

        int idx = iz * sz + ix * sx;
        float grad = 0.0f;

        if constexpr (Dim & GRAD_X) {
            grad += (
                c1 * (u[idx + sx] - u[idx - sx]) +
                c2 * (u[idx + 2*sx] - u[idx - 2*sx])
            ) / dx;
        }

        if constexpr (Dim & GRAD_Z) {
            grad += (
                c1 * (u[idx + sz] - u[idx - sz]) +
                c2 * (u[idx + 2*sz] - u[idx - 2*sz])
            ) / dz;
        }

        return grad;
    }
};


// ============================================================
// 6th order
// ============================================================
template<int Dim>
struct GradientImpl<6, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sz,
        int ix, int iz,
        float dx, float dz
    ) {
        constexpr float c1 = 3.0f / 4.0f;
        constexpr float c2 = -3.0f / 20.0f;
        constexpr float c3 = 1.0f / 60.0f;

        int idx = iz * sz + ix * sx;
        float grad = 0.0f;

        if constexpr (Dim & GRAD_X) {
            grad += (
                c1 * (u[idx + sx] - u[idx - sx]) +
                c2 * (u[idx + 2*sx] - u[idx - 2*sx]) +
                c3 * (u[idx + 3*sx] - u[idx - 3*sx])
            ) / dx;
        }

        if constexpr (Dim & GRAD_Z) {
            grad += (
                c1 * (u[idx + sz] - u[idx - sz]) +
                c2 * (u[idx + 2*sz] - u[idx - 2*sz]) +
                c3 * (u[idx + 3*sz] - u[idx - 3*sz])
            ) / dz;
        }

        return grad;
    }
};


// ============================================================
// 8th order
// ============================================================
template<int Dim>
struct GradientImpl<8, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sz,
        int ix, int iz,
        float dx, float dz
    ) {
        constexpr float c1 = 4.0f / 5.0f;
        constexpr float c2 = -1.0f / 5.0f;
        constexpr float c3 = 4.0f / 105.0f;
        constexpr float c4 = -1.0f / 280.0f;

        int idx = iz * sz + ix * sx;
        float grad = 0.0f;

        if constexpr (Dim & GRAD_X) {
            grad += (
                c1 * (u[idx + sx] - u[idx - sx]) +
                c2 * (u[idx + 2*sx] - u[idx - 2*sx]) +
                c3 * (u[idx + 3*sx] - u[idx - 3*sx]) +
                c4 * (u[idx + 4*sx] - u[idx - 4*sx])
            ) / dx;
        }

        if constexpr (Dim & GRAD_Z) {
            grad += (
                c1 * (u[idx + sz] - u[idx - sz]) +
                c2 * (u[idx + 2*sz] - u[idx - 2*sz]) +
                c3 * (u[idx + 3*sz] - u[idx - 3*sz]) +
                c4 * (u[idx + 4*sz] - u[idx - 4*sz])
            ) / dz;
        }

        return grad;
    }
};


// ============================================================
// Arbitrary order
// ============================================================
template<int Dim>
struct GradientImpl<-1, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sz,
        int ix, int iz,
        int M,
        const float* __restrict__ coeff,
        float dx, float dz
    ) {
        int idx = iz * sz + ix * sx;
        float grad = 0.0f;

        if constexpr (Dim & GRAD_X) {
            float acc = 0.0f;
            #pragma unroll 1
            for (int k = 1; k <= M; ++k) {
                float ck = coeff[k];
                acc += ck * (
                    u[idx + k*sx] -
                    u[idx - k*sx]
                );
            }
            grad += acc / dx;
        }

        if constexpr (Dim & GRAD_Z) {
            float acc = 0.0f;
            #pragma unroll 1
            for (int k = 1; k <= M; ++k) {
                float ck = coeff[k];
                acc += ck * (
                    u[idx + k*sz] -
                    u[idx - k*sz]
                );
            }
            grad += acc / dz;
        }

        return grad;
    }
};

