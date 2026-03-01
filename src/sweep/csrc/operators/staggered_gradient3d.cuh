#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "gradient3d.cuh"

enum DiffType : int {
    DIFF_FORWARD  = 0,
    DIFF_BACKWARD = 1
};

struct SGradContext3d {
    int sx;
    int sy;
    int sz;

    int ix;
    int iy;
    int iz;

    int M;
    const float* __restrict__ coeff;

    float dx;
    float dy;
    float dz;
};

template<int Order, int Dim, DiffType Type>
struct SGradientImpl;

template<int Order, int Dim, DiffType Type>
__device__ __forceinline__
float sgradient(
    const float* __restrict__ u,
    const SGradContext3d& ctx
)
{
    if constexpr (Order == -1) {

        return SGradientImpl<-1, Dim, Type>::eval(
            u,
            ctx.sx, ctx.sy, ctx.sz,
            ctx.ix, ctx.iy, ctx.iz,
            ctx.M,
            ctx.coeff,
            ctx.dx, ctx.dy, ctx.dz
        );

    } else {

        return SGradientImpl<Order, Dim, Type>::eval(
            u,
            ctx.sx, ctx.sy, ctx.sz,
            ctx.ix, ctx.iy, ctx.iz,
            ctx.dx, ctx.dy, ctx.dz
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
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {

        int idx = iz*sz + iy*sy + ix*sx;
        float grad = 0.f;

        // X
        if constexpr (Dim & GRAD_X) {

            if constexpr (Type == DIFF_FORWARD)
                grad += (u[idx+sx] - u[idx]) / dx;
            else
                grad += (u[idx] - u[idx-sx]) / dx;
        }

        // Y
        if constexpr (Dim & GRAD_Y) {

            if constexpr (Type == DIFF_FORWARD)
                grad += (u[idx+sy] - u[idx]) / dy;
            else
                grad += (u[idx] - u[idx-sy]) / dy;
        }

        // Z
        if constexpr (Dim & GRAD_Z) {

            if constexpr (Type == DIFF_FORWARD)
                grad += (u[idx+sz] - u[idx]) / dz;
            else
                grad += (u[idx] - u[idx-sz]) / dz;
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
        int sy,
        int sz,
        int ix,
        int iy,
        int iz,
        float dx,
        float dy,
        float dz
    ) {

        constexpr float c1 = 9.f/8.f;
        constexpr float c2 = -1.f/24.f;

        int idx = iz*sz + iy*sy + ix*sx;
        float grad = 0.f;

        // ================= X =================
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

        // ================= Y =================
        if constexpr (Dim & GRAD_Y) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                    c2*u[idx+2*sy] +
                     c1*u[idx+sy] -
                     c1*u[idx] -
                     c2*u[idx-sy]
                ) / dy;

            } else {

                grad += (
                     c2*u[idx+sy] +
                     c1*u[idx] -
                     c1*u[idx-sy] -
                     c2*u[idx-2*sy]
                ) / dy;
            }
        }

        // ================= Z =================
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
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {

        constexpr float c1 = 75.f/64.f;
        constexpr float c2 = -25.f/384.f;
        constexpr float c3 = 3.f/640.f;

        int idx = iz*sz + iy*sy + ix*sx;
        float grad = 0.f;

        // ================= X =================
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

        // ================= Y =================
        if constexpr (Dim & GRAD_Y) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                     c3*u[idx+3*sy] +
                     c2*u[idx+2*sy] +
                     c1*u[idx+sy] -
                     c1*u[idx] -
                     c2*u[idx-sy] -
                     c3*u[idx-2*sy]
                ) / dy;

            } else {

                grad += (
                     c3*u[idx+2*sy] +
                     c2*u[idx+sy] +
                     c1*u[idx] -
                     c1*u[idx-sy] -
                     c2*u[idx-2*sy] -
                     c3*u[idx-3*sy]
                ) / dy;
            }
        }

        // ================= Z =================
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
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {

        constexpr float c1 = 1225.f/1024.f;
        constexpr float c2 = -245.f/3072.f;
        constexpr float c3 = 49.f/5120.f;
        constexpr float c4 = -5.f/7168.f;

        int idx = iz*sz + iy*sy + ix*sx;
        float grad = 0.f;

        // ================= X =================
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

        // ================= Y =================
        if constexpr (Dim & GRAD_Y) {

            if constexpr (Type == DIFF_FORWARD) {

                grad += (
                     c4*u[idx+4*sy] +
                     c3*u[idx+3*sy] +
                     c2*u[idx+2*sy] +
                     c1*u[idx+sy] -
                     c1*u[idx] -
                     c2*u[idx-sy] -
                     c3*u[idx-2*sy] -
                     c4*u[idx-3*sy]
                ) / dy;

            } else {

                grad += (
                     c4*u[idx+3*sy] +
                     c3*u[idx+2*sy] +
                     c2*u[idx+sy] +
                     c1*u[idx] -
                     c1*u[idx-sy] -
                     c2*u[idx-2*sy] -
                     c3*u[idx-3*sy] -
                     c4*u[idx-4*sy]
                ) / dy;
            }
        }

        // ================= Z =================
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
        int sy,
        int sz,
        int ix,
        int iy,
        int iz,
        int M,
        const float* __restrict__ coeff,
        float dx,
        float dy,
        float dz
    ) {

        int idx = iz*sz + iy*sy + ix*sx;
        float grad = 0.f;

        // ================= X =================
        if constexpr (Dim & GRAD_X) {

            float acc = 0.f;

            #pragma unroll 1
            for (int m = 1; m <= M; ++m) {

                if constexpr (Type == DIFF_FORWARD) {

                    acc += coeff[m] *
                        ( u[idx + m*sx]
                        - u[idx - (m-1)*sx] );

                } else {

                    acc += coeff[m] *
                        ( u[idx + (m-1)*sx]
                        - u[idx - m*sx] );
                }
            }

            grad += acc / dx;
        }

        // ================= Y =================
        if constexpr (Dim & GRAD_Y) {

            float acc = 0.f;

            #pragma unroll 1
            for (int m = 1; m <= M; ++m) {

                if constexpr (Type == DIFF_FORWARD) {

                    acc += coeff[m] *
                        ( u[idx + m*sy]
                        - u[idx - (m-1)*sy] );

                } else {

                    acc += coeff[m] *
                        ( u[idx + (m-1)*sy]
                        - u[idx - m*sy] );
                }
            }

            grad += acc / dy;
        }

        // ================= Z =================
        if constexpr (Dim & GRAD_Z) {

            float acc = 0.f;

            #pragma unroll 1
            for (int m = 1; m <= M; ++m) {

                if constexpr (Type == DIFF_FORWARD) {

                    acc += coeff[m] *
                        ( u[idx + m*sz]
                        - u[idx - (m-1)*sz] );

                } else {

                    acc += coeff[m] *
                        ( u[idx + (m-1)*sz]
                        - u[idx - m*sz] );
                }
            }

            grad += acc / dz;
        }

        return grad;
    }
};