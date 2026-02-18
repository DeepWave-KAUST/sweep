#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

enum GradDim : int {
    GRAD_X   = 1,
    GRAD_Y   = 2,
    GRAD_Z   = 4,
    GRAD_XY  = 3,
    GRAD_XZ  = 5,
    GRAD_YZ  = 6,
    GRAD_XYZ = 7
};

template<int Order, int Dim>
struct GradientImpl;

struct GradContext3D {
    int sx;      // stride in x
    int sy;      // stride in y
    int sz;      // stride in z

    int ix;
    int iy;
    int iz;

    int M;       // runtime order
    const float* __restrict__ coeff;

    float dx;
    float dy;
    float dz;
};

template<int Order, int Dim>
__device__ __forceinline__
float gradient(
    const float* __restrict__ u,
    const GradContext3D& ctx
) {
    if constexpr (Order == -1) {
        return GradientImpl<-1, Dim>::eval(
            u,
            ctx.sx, ctx.sy, ctx.sz,
            ctx.ix, ctx.iy, ctx.iz,
            ctx.M, ctx.coeff,
            ctx.dx, ctx.dy, ctx.dz
        );
    } else {
        return GradientImpl<Order, Dim>::eval(
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
template<int Dim>
struct GradientImpl<2, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        int idx = ix*sx + iy*sy + iz*sz;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X)
            grad += (u[idx + sx] - u[idx - sx]) / (2.f * dx);

        if constexpr (Dim & GRAD_Y)
            grad += (u[idx + sy] - u[idx - sy]) / (2.f * dy);

        if constexpr (Dim & GRAD_Z)
            grad += (u[idx + sz] - u[idx - sz]) / (2.f * dz);

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
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        constexpr float c1 = 8.f/12.f;
        constexpr float c2 = -1.f/12.f;

        int idx = ix*sx + iy*sy + iz*sz;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X)
            grad += (c1*(u[idx+sx]-u[idx-sx])
                   + c2*(u[idx+2*sx]-u[idx-2*sx])) / dx;

        if constexpr (Dim & GRAD_Y)
            grad += (c1*(u[idx+sy]-u[idx-sy])
                   + c2*(u[idx+2*sy]-u[idx-2*sy])) / dy;

        if constexpr (Dim & GRAD_Z)
            grad += (c1*(u[idx+sz]-u[idx-sz])
                   + c2*(u[idx+2*sz]-u[idx-2*sz])) / dz;

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
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        constexpr float c1 = 3.f/4.f;
        constexpr float c2 = -3.f/20.f;
        constexpr float c3 = 1.f/60.f;

        int idx = ix*sx + iy*sy + iz*sz;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X)
            grad += (c1*(u[idx+sx]-u[idx-sx])
                   + c2*(u[idx+2*sx]-u[idx-2*sx])
                   + c3*(u[idx+3*sx]-u[idx-3*sx])) / dx;

        if constexpr (Dim & GRAD_Y)
            grad += (c1*(u[idx+sy]-u[idx-sy])
                   + c2*(u[idx+2*sy]-u[idx-2*sy])
                   + c3*(u[idx+3*sy]-u[idx-3*sy])) / dy;

        if constexpr (Dim & GRAD_Z)
            grad += (c1*(u[idx+sz]-u[idx-sz])
                   + c2*(u[idx+2*sz]-u[idx-2*sz])
                   + c3*(u[idx+3*sz]-u[idx-3*sz])) / dz;

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
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        constexpr float c1 = 4.f/5.f;
        constexpr float c2 = -1.f/5.f;
        constexpr float c3 = 4.f/105.f;
        constexpr float c4 = -1.f/280.f;

        int idx = ix*sx + iy*sy + iz*sz;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X)
            grad += (c1*(u[idx+sx]-u[idx-sx])
                   + c2*(u[idx+2*sx]-u[idx-2*sx])
                   + c3*(u[idx+3*sx]-u[idx-3*sx])
                   + c4*(u[idx+4*sx]-u[idx-4*sx])) / dx;

        if constexpr (Dim & GRAD_Y)
            grad += (c1*(u[idx+sy]-u[idx-sy])
                   + c2*(u[idx+2*sy]-u[idx-2*sy])
                   + c3*(u[idx+3*sy]-u[idx-3*sy])
                   + c4*(u[idx+4*sy]-u[idx-4*sy])) / dy;

        if constexpr (Dim & GRAD_Z)
            grad += (c1*(u[idx+sz]-u[idx-sz])
                   + c2*(u[idx+2*sz]-u[idx-2*sz])
                   + c3*(u[idx+3*sz]-u[idx-3*sz])
                   + c4*(u[idx+4*sz]-u[idx-4*sz])) / dz;

        return grad;
    }
};


// ============================================================
// 2M order
// ============================================================
template<int Dim>
struct GradientImpl<-1, Dim> {
    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        int M,
        const float* __restrict__ coeff,
        float dx, float dy, float dz
    ) {
        int idx = ix*sx + iy*sy + iz*sz;
        float grad = 0.f;

        if constexpr (Dim & GRAD_X) {
            float acc = 0.f;
            for (int k=1; k<=M; ++k)
                acc += coeff[k] * (u[idx+k*sx] - u[idx-k*sx]);
            grad += acc / dx;
        }

        if constexpr (Dim & GRAD_Y) {
            float acc = 0.f;
            for (int k=1; k<=M; ++k)
                acc += coeff[k] * (u[idx+k*sy] - u[idx-k*sy]);
            grad += acc / dy;
        }

        if constexpr (Dim & GRAD_Z) {
            float acc = 0.f;
            for (int k=1; k<=M; ++k)
                acc += coeff[k] * (u[idx+k*sz] - u[idx-k*sz]);
            grad += acc / dz;
        }

        return grad;
    }
};
