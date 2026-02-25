#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "gradient2d.cuh"

enum DiffType : int {
    DIFF_FORWARD  = 0,
    DIFF_BACKWARD = 1
};

struct SGradContext {
    int sx;      // stride in x
    int sz;      // stride in z
    int ix;
    int iz;

    int M;      // runtime order (if needed)
    const float* __restrict__ coeff;

    float dx;
    float dz;
};

template<int Order, int Dim, DiffType Type>
struct SGradientImpl;


template<int Order, int Dim, DiffType Type>
__device__ __forceinline__
float sgradient(
    const float* __restrict__ u,
    const SGradContext& ctx
)
{
    if constexpr (Order == -1) {

        return SGradientImpl<-1, Dim, Type>::eval(
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

        return SGradientImpl<Order, Dim, Type>::eval(
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


// ============================================================
// 2nd order
// ============================================================
template<int Dim, DiffType Type>
struct SGradientImpl<2, Dim, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sz,
        int ix, int iz,
        float dx, float dz
    ) {

        int idx = iz * sz + ix * sx;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X) {

            if constexpr (Type == DIFF_FORWARD)
                grad += (u[idx + sx] - u[idx]) / dx;
            else
                grad += (u[idx] - u[idx - sx]) / dx;
        }

        if constexpr (Dim & GRAD_Z) {

            if constexpr (Type == DIFF_FORWARD)
                grad += (u[idx + sz] - u[idx]) / dz;
            else
                grad += (u[idx] - u[idx - sz]) / dz;
        }

        return grad;
    }
};

// ============================================================
// 4th order
// ============================================================
template<int Dim, DiffType Type>
struct SGradientImpl<4, Dim, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx,
        int sz,
        int ix,
        int iz,
        float dx,
        float dz
    ) {

        constexpr float c1 = 9.f/8.f;
        constexpr float c2 = -1.f/24.f;

        int idx = iz * sz + ix * sx;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                    c2*u[idx+2*sx] +
                     c1*u[idx+sx] -
                     c1*u[idx] -
                     c2*u[idx-sx]
                ) / dx;

            } else {

                grad += (
                     c2*u[idx+sx] +
                     c1*u[idx] -
                     c1*u[idx-sx] -
                     c2*u[idx-2*sx]
                ) / dx;
            }
        }

        if constexpr (Dim & GRAD_Z) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                    c2*u[idx+2*sz] +
                     c1*u[idx+sz] -
                     c1*u[idx] -
                     c2*u[idx-sz]
                ) / dz;

            } else {

                grad += (
                     c2*u[idx+sz] +
                     c1*u[idx] -
                     c1*u[idx-sz] -
                     c2*u[idx-2*sz]
                ) / dz;
            }
        }

        return grad;
    }
};

// ============================================================
// 6th order
// ============================================================
template<int Dim, DiffType Type>
struct SGradientImpl<6, Dim, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx,
        int sz,
        int ix,
        int iz,
        float dx,
        float dz
    ) {

        constexpr float c1 = 75.f/64.f;
        constexpr float c2 = -25.f/384.f;
        constexpr float c3 = 3.f/640.f;

        int idx = iz * sz + ix * sx;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                     c3*u[idx+3*sx] +
                     c2*u[idx+2*sx] +
                     c1*u[idx+sx] -
                     c1*u[idx] -
                     c2*u[idx-sx] -
                     c3*u[idx-2*sx]
                ) / dx;

            } else {

                grad += (
                     c3*u[idx+2*sx] +
                     c2*u[idx+sx] +
                     c1*u[idx] -
                     c1*u[idx-sx] -
                     c2*u[idx-2*sx] -
                     c3*u[idx-3*sx]
                ) / dx;
            }
        }

        if constexpr (Dim & GRAD_Z) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                     c3*u[idx+3*sz] +
                     c2*u[idx+2*sz] +
                     c1*u[idx+sz] -
                     c1*u[idx] -
                     c2*u[idx-sz] -
                     c3*u[idx-2*sz]
                ) / dz;

            } else {

                grad += (
                     c3*u[idx+2*sz] +
                     c2*u[idx+sz] +
                     c1*u[idx] -
                     c1*u[idx-sz] -
                     c2*u[idx-2*sz] -
                     c3*u[idx-3*sz]
                ) / dz;
            }
        }

        return grad;
    }
};

// ============================================================
// 8th order
// ============================================================
template<int Dim, DiffType Type>
struct SGradientImpl<8, Dim, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx,
        int sz,
        int ix,
        int iz,
        float dx,
        float dz
    ) {

        constexpr float c1 = 1225.f/1024.f;
        constexpr float c2 = -245.f/3072.f;
        constexpr float c3 = 49.f/5120.f;
        constexpr float c4 = -5.f/7168.f;

        int idx = iz * sz + ix * sx;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                     c4*u[idx+4*sx] +
                     c3*u[idx+3*sx] +
                     c2*u[idx+2*sx] +
                     c1*u[idx+sx] -
                     c1*u[idx] -
                     c2*u[idx-sx] -
                     c3*u[idx-2*sx] -
                     c4*u[idx-3*sx]
                ) / dx;

            } else {

                grad += (
                     c4*u[idx+3*sx] +
                     c3*u[idx+2*sx] +
                     c2*u[idx+sx] +
                     c1*u[idx] -
                     c1*u[idx-sx] -
                     c2*u[idx-2*sx] -
                     c3*u[idx-3*sx] -
                     c4*u[idx-4*sx]
                ) / dx;
            }
        }

        if constexpr (Dim & GRAD_Z) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                     c4*u[idx+4*sz] +
                     c3*u[idx+3*sz] +
                     c2*u[idx+2*sz] +
                     c1*u[idx+sz] -
                     c1*u[idx] -
                     c2*u[idx-sz] -
                     c3*u[idx-2*sz] -
                     c4*u[idx-3*sz]
                ) / dz;

            } else {

                grad += (
                     c4*u[idx+3*sz] +
                     c3*u[idx+2*sz] +
                     c2*u[idx+sz] +
                     c1*u[idx] -
                     c1*u[idx-sz] -
                     c2*u[idx-2*sz] -
                     c3*u[idx-3*sz] -
                     c4*u[idx-4*sz]
                ) / dz;
            }
        }

        return grad;
    }
};

// ============================================================
// 2n order
// ============================================================
template<int Dim, DiffType Type>
struct SGradientImpl<-1, Dim, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx,
        int sz,
        int ix,
        int iz,
        int M,
        const float* __restrict__ coeff,
        float dx,
        float dz
    ) {

        int idx = iz * sz + ix * sx;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X) {

            float acc = 0.f;

            if constexpr (Type == DIFF_FORWARD) {

                #pragma unroll 1
                for (int k = 0; k <= M; ++k)
                    acc += coeff[k] * u[idx + k*sx];

            } else {

                #pragma unroll 1
                for (int k = 0; k <= M; ++k)
                    acc += coeff[k] * u[idx - k*sx];
            }

            grad += acc / dx;
        }

        if constexpr (Dim & GRAD_Z) {

            float acc = 0.f;

            if constexpr (Type == DIFF_FORWARD) {

                #pragma unroll 1
                for (int k = 0; k <= M; ++k)
                    acc += coeff[k] * u[idx + k*sz];

            } else {

                #pragma unroll 1
                for (int k = 0; k <= M; ++k)
                    acc += coeff[k] * u[idx - k*sz];
            }

            grad += acc / dz;
        }

        return grad;
    }
};
