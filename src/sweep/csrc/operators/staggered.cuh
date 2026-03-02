#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "dim.cuh"

struct SGradParam {
    int sx;   // stride x
    int sy;   // stride y
    int sz;   // stride z

    int M;    // runtime half order
    const float* coeff;

    float dx;
    float dy;
    float dz;
};

template<int Order, int ND, int Direction, int Type>
struct SGradGPU;

template<int ND, int Order, int Direction, int Type>
__device__ __forceinline__
float sgradient(
    const float* __restrict__ u,
    int ix, int iy, int iz,
    const SGradParam& p
)
{
    if constexpr (Order == -1)
    {
        return SGradGPU<-1, ND, Direction, Type>::eval(
            u,
            p.coeff,
            p.sx, p.sy, p.sz,
            ix, iy, iz,
            p.M,
            p.dx, p.dy, p.dz
        );
    }
    else
    {
        return SGradGPU<Order, ND, Direction, Type>::eval(
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
template<int ND, int Direction, int Type>
struct SGradGPU<2, ND, Direction, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {

        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        // -----------------------------------
        // X direction
        // -----------------------------------
        if constexpr (Direction & X)
        {
            float inv_dx = 1.f / dx;

            if constexpr (Type & DIFF_FORWARD)
                grad += (u[idx + sx] - u[idx]) * inv_dx;
            if constexpr (Type & DIFF_BACKWARD)
                grad += (u[idx] - u[idx - sx]) * inv_dx;
        }

        // -----------------------------------
        // Y direction (3D only)
        // -----------------------------------
        if constexpr (ND == 3 && (Direction & Y))
        {
            float inv_dy = 1.f / dy;

            if constexpr (Type & DIFF_FORWARD)
                grad += (u[idx + sy] - u[idx]) * inv_dy;
            if constexpr (Type & DIFF_BACKWARD)
                grad += (u[idx] - u[idx - sy]) * inv_dy;
        }
        

        // -----------------------------------
        // Z direction
        // -----------------------------------
        if constexpr (Direction & Z)
        {
            float inv_dz = 1.f / dz;

            if constexpr (Type & DIFF_FORWARD)
                grad += (u[idx + sz] - u[idx]) * inv_dz;
            if constexpr (Type & DIFF_BACKWARD)
                grad += (u[idx] - u[idx - sz]) * inv_dz;
        }

        return grad;
    }
};

// ============================================================
// 4th order
// ============================================================
template<int ND, int Direction, int Type>
struct SGradGPU<4, ND, Direction, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        // -----------------------------------
        // index
        // -----------------------------------
        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        constexpr float c1 = 9.f / 8.f;
        constexpr float c2 = -1.f / 24.f;

        // -----------------------------------
        // X direction
        // -----------------------------------
        if constexpr (Direction & X)
        {
            float inv_dx = 1.f / dx;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1 * (u[idx + sx]     - u[idx]) +
                    c2 * (u[idx + 2*sx]   - u[idx - sx])
                ) * inv_dx;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1 * (u[idx]          - u[idx - sx]) +
                    c2 * (u[idx + sx]     - u[idx - 2*sx])
                ) * inv_dx;
            }
        }

        // -----------------------------------
        // Y direction (3D only)
        // -----------------------------------
        if constexpr (ND == 3 && (Direction & Y))
        {
            float inv_dy = 1.f / dy;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1 * (u[idx + sy]     - u[idx]) +
                    c2 * (u[idx + 2*sy]   - u[idx - sy])
                ) * inv_dy;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1 * (u[idx]          - u[idx - sy]) +
                    c2 * (u[idx + sy]     - u[idx - 2*sy])
                ) * inv_dy;
            }
        }

        // -----------------------------------
        // Z direction
        // -----------------------------------
        if constexpr (Direction & Z)
        {
            float inv_dz = 1.f / dz;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1 * (u[idx + sz]     - u[idx]) +
                    c2 * (u[idx + 2*sz]   - u[idx - sz])
                ) * inv_dz;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1 * (u[idx]          - u[idx - sz]) +
                    c2 * (u[idx + sz]     - u[idx - 2*sz])
                ) * inv_dz;
            }
        }

        return grad;
    }
};

// ============================================================
// 6th order
// ============================================================
template<int ND, int Direction, int Type>
struct SGradGPU<6, ND, Direction, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        constexpr float c1 = 75.f/64.f;
        constexpr float c2 = -25.f/384.f;
        constexpr float c3 = 3.f/640.f;

        // ---------------- X ----------------
        if constexpr (Direction & X)
        {
            float inv_dx = 1.f / dx;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1*(u[idx+sx]   - u[idx]) +
                    c2*(u[idx+2*sx] - u[idx-sx]) +
                    c3*(u[idx+3*sx] - u[idx-2*sx])
                ) * inv_dx;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1*(u[idx]      - u[idx-sx]) +
                    c2*(u[idx+sx]   - u[idx-2*sx]) +
                    c3*(u[idx+2*sx] - u[idx-3*sx])
                ) * inv_dx;
            }
        }

        // ---------------- Y ----------------
        if constexpr (ND == 3 && (Direction & Y))
        {
            float inv_dy = 1.f / dy;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1*(u[idx+sy]   - u[idx]) +
                    c2*(u[idx+2*sy] - u[idx-sy]) +
                    c3*(u[idx+3*sy] - u[idx-2*sy])
                ) * inv_dy;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1*(u[idx]      - u[idx-sy]) +
                    c2*(u[idx+sy]   - u[idx-2*sy]) +
                    c3*(u[idx+2*sy] - u[idx-3*sy])
                ) * inv_dy;
            }
        }

        // ---------------- Z ----------------
        if constexpr (Direction & Z)
        {
            float inv_dz = 1.f / dz;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1*(u[idx+sz]   - u[idx]) +
                    c2*(u[idx+2*sz] - u[idx-sz]) +
                    c3*(u[idx+3*sz] - u[idx-2*sz])
                ) * inv_dz;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1*(u[idx]      - u[idx-sz]) +
                    c2*(u[idx+sz]   - u[idx-2*sz]) +
                    c3*(u[idx+2*sz] - u[idx-3*sz])
                ) * inv_dz;
            }
        }

        return grad;
    }
};

// ============================================================
// 8th order
// ============================================================
template<int ND, int Direction, int Type>
struct SGradGPU<8, ND, Direction, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        int sx, int sy, int sz,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        constexpr float c1 = 1225.f/1024.f;
        constexpr float c2 = -245.f/3072.f;
        constexpr float c3 = 49.f/5120.f;
        constexpr float c4 = -5.f/7168.f;

        if constexpr (Direction & X)
        {
            float inv_dx = 1.f / dx;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1*(u[idx+sx]     - u[idx]) +
                    c2*(u[idx+2*sx]   - u[idx-sx]) +
                    c3*(u[idx+3*sx]   - u[idx-2*sx]) +
                    c4*(u[idx+4*sx]   - u[idx-3*sx])
                ) * inv_dx;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1*(u[idx]        - u[idx-sx]) +
                    c2*(u[idx+sx]     - u[idx-2*sx]) +
                    c3*(u[idx+2*sx]   - u[idx-3*sx]) +
                    c4*(u[idx+3*sx]   - u[idx-4*sx])
                ) * inv_dx;
            }
        }

        if constexpr (ND == 3 && (Direction & Y))
        {
            float inv_dy = 1.f / dy;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1*(u[idx+sy]     - u[idx]) +
                    c2*(u[idx+2*sy]   - u[idx-sy]) +
                    c3*(u[idx+3*sy]   - u[idx-2*sy]) +
                    c4*(u[idx+4*sy]   - u[idx-3*sy])
                ) * inv_dy;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1*(u[idx]        - u[idx-sy]) +
                    c2*(u[idx+sy]     - u[idx-2*sy]) +
                    c3*(u[idx+2*sy]   - u[idx-3*sy]) +
                    c4*(u[idx+3*sy]   - u[idx-4*sy])
                ) * inv_dy;
            }
        }

        if constexpr (Direction & Z)
        {
            float inv_dz = 1.f / dz;

            if constexpr (Type & DIFF_FORWARD)
            {
                grad += (
                    c1*(u[idx+sz]     - u[idx]) +
                    c2*(u[idx+2*sz]   - u[idx-sz]) +
                    c3*(u[idx+3*sz]   - u[idx-2*sz]) +
                    c4*(u[idx+4*sz]   - u[idx-3*sz])
                ) * inv_dz;
            }

            if constexpr (Type & DIFF_BACKWARD)
            {
                grad += (
                    c1*(u[idx]        - u[idx-sz]) +
                    c2*(u[idx+sz]     - u[idx-2*sz]) +
                    c3*(u[idx+2*sz]   - u[idx-3*sz]) +
                    c4*(u[idx+3*sz]   - u[idx-4*sz])
                ) * inv_dz;
            }
        }

        return grad;
    }
};

// ============================================================
// 8th order
// ============================================================
template<int ND, int Direction, int Type>
struct SGradGPU<-1, ND, Direction, Type> {

    __device__ __forceinline__
    static float eval(
        const float* __restrict__ u,
        const float* __restrict__ coeff,
        int sx, int sy, int sz,
        int ix, int iy, int iz, int M,
        float dx, float dy, float dz
    )
    {
        // ---------------- index ----------------
        int idx = ix * sx;

        if constexpr (ND == 3)
            idx += iy * sy;

        idx += iz * sz;

        float grad = 0.f;

        // ====================================================
        // X direction
        // ====================================================
        if constexpr (Direction & X)
        {
            float inv_dx = 1.f / dx;
            float gx = 0.f;

            #pragma unroll
            for (int m = 0; m < M; ++m)
            {
                int offset_f = (m + 1) * sx;
                int offset_b = m * sx;

                if constexpr (Type & DIFF_FORWARD)
                    gx += coeff[m] * (u[idx + offset_f]
                                     - u[idx - offset_b]);

                if constexpr (Type & DIFF_BACKWARD)
                    gx += coeff[m] * (u[idx + offset_b]
                                     - u[idx - offset_f]);
            }

            grad += gx * inv_dx;
        }

        // ====================================================
        // Y direction (3D only)
        // ====================================================
        if constexpr (ND == 3 && (Direction & Y))
        {
            float inv_dy = 1.f / dy;
            float gy = 0.f;

            #pragma unroll
            for (int m = 0; m < M; ++m)
            {
                int offset_f = (m + 1) * sy;
                int offset_b = m * sy;

                if constexpr (Type & DIFF_FORWARD)
                    gy += coeff[m] * (u[idx + offset_f]
                                     - u[idx - offset_b]);

                if constexpr (Type & DIFF_BACKWARD)
                    gy += coeff[m] * (u[idx + offset_b]
                                     - u[idx - offset_f]);
            }

            grad += gy * inv_dy;
        }

        // ====================================================
        // Z direction
        // ====================================================
        if constexpr (Direction & Z)
        {
            float inv_dz = 1.f / dz;
            float gz = 0.f;

            #pragma unroll
            for (int m = 0; m < M; ++m)
            {
                int offset_f = (m + 1) * sz;
                int offset_b = m * sz;

                if constexpr (Type & DIFF_FORWARD)
                    gz += coeff[m] * (u[idx + offset_f]
                                     - u[idx - offset_b]);

                if constexpr (Type & DIFF_BACKWARD)
                    gz += coeff[m] * (u[idx + offset_b]
                                     - u[idx - offset_f]);
            }

            grad += gz * inv_dz;
        }

        return grad;
    }
};