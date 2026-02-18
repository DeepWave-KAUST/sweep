#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

enum LaplaceDim : int {
    LAPLACE_X   = 1,
    LAPLACE_Y   = 2,
    LAPLACE_Z   = 4,
    LAPLACE_XY  = LAPLACE_X | LAPLACE_Y,
    LAPLACE_XZ  = LAPLACE_X | LAPLACE_Z,
    LAPLACE_YZ  = LAPLACE_Y | LAPLACE_Z,
    LAPLACE_XYZ = LAPLACE_X | LAPLACE_Y | LAPLACE_Z
};


template<int Order, int Dim>
struct LaplaceImpl;

struct Laplace3dContext {
    int nx;
    int ny;

    int ix;
    int iy;
    int iz;

    int M;
    const float* coeff;

    float dx;
    float dy;
    float dz;
};


template<int Order, int Dim>
__device__ __forceinline__
float laplace(
    const float* __restrict__ u,
    const Laplace3dContext& ctx
) {
    if constexpr (Order == -1) {
        return LaplaceImpl<-1,Dim>::eval(
            u,
            ctx.nx,
            ctx.ny,
            ctx.ix,
            ctx.iy,
            ctx.iz,
            ctx.M,
            ctx.coeff,
            ctx.dx,
            ctx.dy,
            ctx.dz
        );
    } else {
        return LaplaceImpl<Order,Dim>::eval(
            u,
            ctx.nx,
            ctx.ny,
            ctx.ix,
            ctx.iy,
            ctx.iz,
            ctx.dx,
            ctx.dy,
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
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        int stride_y = nx;
        int stride_z = nx * ny;

        int idx = iz * stride_z + iy * stride_y + ix;
        float u0 = u[idx];
        float lap = 0.0f;

        if constexpr (Dim & LAPLACE_X)
            lap += (u[idx - 1] + u[idx + 1] - 2.f*u0)/(dx*dx);

        if constexpr (Dim & LAPLACE_Y)
            lap += (u[idx - stride_y] + u[idx + stride_y] - 2.f*u0)/(dy*dy);

        if constexpr (Dim & LAPLACE_Z)
            lap += (u[idx - stride_z] + u[idx + stride_z] - 2.f*u0)/(dz*dz);

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
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        constexpr float c0 = 5.f/2.f;
        constexpr float c1 = 4.f/3.f;
        constexpr float c2 = -1.f/12.f;

        int stride_y = nx;
        int stride_z = nx * ny;
        int idx = iz * stride_z + iy * stride_y + ix;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Dim & LAPLACE_X) {
            lap += (c1*(u[idx-1]+u[idx+1]) +
                    c2*(u[idx-2]+u[idx+2]) -
                    c0*u0)/(dx*dx);
        }

        if constexpr (Dim & LAPLACE_Y) {
            lap += (c1*(u[idx-stride_y]+u[idx+stride_y]) +
                    c2*(u[idx-2*stride_y]+u[idx+2*stride_y]) -
                    c0*u0)/(dy*dy);
        }

        if constexpr (Dim & LAPLACE_Z) {
            lap += (c1*(u[idx-stride_z]+u[idx+stride_z]) +
                    c2*(u[idx-2*stride_z]+u[idx+2*stride_z]) -
                    c0*u0)/(dz*dz);
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
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        constexpr float c0 = 49.f / 18.f;
        constexpr float c1 = 3.f  / 2.f;
        constexpr float c2 = -3.f / 20.f;
        constexpr float c3 = 1.f  / 90.f;

        int stride_y = nx;
        int stride_z = nx * ny;
        int idx = iz * stride_z + iy * stride_y + ix;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Dim & LAPLACE_X) {
            lap += (
                c1 * (u[idx - 1] + u[idx + 1]) +
                c2 * (u[idx - 2] + u[idx + 2]) +
                c3 * (u[idx - 3] + u[idx + 3]) -
                c0 * u0
            ) / (dx * dx);
        }

        if constexpr (Dim & LAPLACE_Y) {
            lap += (
                c1 * (u[idx - stride_y] + u[idx + stride_y]) +
                c2 * (u[idx - 2*stride_y] + u[idx + 2*stride_y]) +
                c3 * (u[idx - 3*stride_y] + u[idx + 3*stride_y]) -
                c0 * u0
            ) / (dy * dy);
        }

        if constexpr (Dim & LAPLACE_Z) {
            lap += (
                c1 * (u[idx - stride_z] + u[idx + stride_z]) +
                c2 * (u[idx - 2*stride_z] + u[idx + 2*stride_z]) +
                c3 * (u[idx - 3*stride_z] + u[idx + 3*stride_z]) -
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
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    ) {
        constexpr float c0 = 205.f / 72.f;
        constexpr float c1 = 8.f   / 5.f;
        constexpr float c2 = -1.f  / 5.f;
        constexpr float c3 = 8.f   / 315.f;
        constexpr float c4 = -1.f  / 560.f;

        int stride_y = nx;
        int stride_z = nx * ny;
        int idx = iz * stride_z + iy * stride_y + ix;

        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Dim & LAPLACE_X) {
            lap += (
                c1 * (u[idx - 1] + u[idx + 1]) +
                c2 * (u[idx - 2] + u[idx + 2]) +
                c3 * (u[idx - 3] + u[idx + 3]) +
                c4 * (u[idx - 4] + u[idx + 4]) -
                c0 * u0
            ) / (dx * dx);
        }

        if constexpr (Dim & LAPLACE_Y) {
            lap += (
                c1 * (u[idx - stride_y] + u[idx + stride_y]) +
                c2 * (u[idx - 2*stride_y] + u[idx + 2*stride_y]) +
                c3 * (u[idx - 3*stride_y] + u[idx + 3*stride_y]) +
                c4 * (u[idx - 4*stride_y] + u[idx + 4*stride_y]) -
                c0 * u0
            ) / (dy * dy);
        }

        if constexpr (Dim & LAPLACE_Z) {
            lap += (
                c1 * (u[idx - stride_z] + u[idx + stride_z]) +
                c2 * (u[idx - 2*stride_z] + u[idx + 2*stride_z]) +
                c3 * (u[idx - 3*stride_z] + u[idx + 3*stride_z]) +
                c4 * (u[idx - 4*stride_z] + u[idx + 4*stride_z]) -
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
        int nx, int ny,
        int ix, int iy, int iz,
        int M,
        const float* __restrict__ coeff,
        float dx, float dy, float dz
    ) {
        int stride_y = nx;
        int stride_z = nx * ny;

        int idx = iz * stride_z + iy * stride_y + ix;
        float u0 = u[idx];
        float lap = 0.f;

        if constexpr (Dim & LAPLACE_X) {
            float acc = 0.f;
            for(int k=1;k<=M;++k)
                acc += coeff[k]*(u[idx+k] + u[idx-k]);
            lap += (acc - coeff[0]*u0)/(dx*dx);
        }

        if constexpr (Dim & LAPLACE_Y) {
            float acc = 0.f;
            for(int k=1;k<=M;++k)
                acc += coeff[k]*(u[idx+k*stride_y] + u[idx-k*stride_y]);
            lap += (acc - coeff[0]*u0)/(dy*dy);
        }

        if constexpr (Dim & LAPLACE_Z) {
            float acc = 0.f;
            for(int k=1;k<=M;++k)
                acc += coeff[k]*(u[idx+k*stride_z] + u[idx-k*stride_z]);
            lap += (acc - coeff[0]*u0)/(dz*dz);
        }

        return lap;
    }
};

