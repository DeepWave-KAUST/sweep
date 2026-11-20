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

// laplace() for the adjoint kernels: the difference form at every order, order 4
// included (see the note above tap_diff).
template<int ND, int Order, int Direction>
__device__ __forceinline__
float laplace_diff(
    const float* __restrict__ u,
    int ix, int iy, int iz,
    const LaplaceParam& p
)
{
    if constexpr (Order == 4) {
        return LaplaceGPU<4, ND, Direction>::eval_diff(
            u,
            p.nx, p.ny,
            ix, iy, iz,
            p.dx, p.dy, p.dz
        );
    } else {
        return laplace<ND, Order, Direction>(u, ix, iy, iz, p);
    }
}

// The stencils are evaluated in difference form, sum_k c_k ((u_k - u_0) + (u_-k - u_0)),
// not as sum_k c_k (u_k + u_-k) - c_0 u_0: on a smooth field that sum climbs to ~c_0 u_0
// before the centre is taken off, and its float32 rounding swamps a result of order
// (k h)^2 u_0 -- records stopped converging below ~1 m at orders 6-10, and adjoint
// records below ~2.5 m at every order.  The neighbour differences are exact wherever
// adjacent taps agree to within 2x (Sterbenz).  The two forms are one operator:
// c_0 = 2 (c_1 + ... + c_M) exactly in the float32 coefficients (see order 4).
//
// The forward keeps the summed form at order 4 (LaplaceGPU<4>::eval).  At small Courant
// numbers on fine grids a leapfrog step adds only a few hundred ulp to u, and what the
// order-4 stencil adds to the order-2 one falls below u's rounding grain: the summed
// form's rounding happens to keep it, the difference form's does not (records at
// Courant 0.2, dh 0.625 m: 1.8e-5 summed, 3.2e-5 in difference form; at Courant 0.4
// the difference form is the better one).  laplace() also serves the kernels that
// rerun the forward stencil (boundary-saving reconstruction, image carriers), so they
// keep matching the forward.  qp2d's forward stencil stays summed at every order, for
// speed (see cell() there).  The adjoint kernels take the difference form at every
// order -- laplace_diff, their hand-written copies of the stencil,
// acoustic_vrz_fused.cuh, qp2d's adjoint -- through tap_diff.

// u_k - u_0 for the difference form.  __fsub_rn is never merged into an FFMA, so a
// tap formed on the fly as a product (the adjoint kernels' v^2 dt^2 lambda, (1+b) g)
// is rounded before the centre is taken off; fused, the product would meet a rounded
// centre unrounded and each difference would carry the centre's rounding error, which
// the c_k sum amplifies by c_0 -- the same float32 floor as the summed form.
__device__ __forceinline__ float tap_diff(float uk, float u0)
{
    return __fsub_rn(uk, u0);
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
            lap += (tap_diff(u[idx - stride_x], u0) + tap_diff(u[idx + stride_x], u0))/(dx*dx);

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (tap_diff(u[idx - stride_y], u0) + tap_diff(u[idx + stride_y], u0))
                   / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (tap_diff(u[idx - stride_z], u0) + tap_diff(u[idx + stride_z], u0))
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

    // The summed form: the forward's at this order (see the note above tap_diff).
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
        // The last coefficient is what makes c0 = 2 (c1 + ... + cM) exact in fp32, so the
        // stencil is zero on a constant field; it differs from its fraction by < 6e-8
        // (sweep.scalars.second_derivative_coefficients_fp32 holds the same values).
        constexpr float c2 = 0.5f * c0 - c1;                     // -1/12

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

    // The difference form, for laplace_diff (the adjoint kernels).
    __device__ __forceinline__
    static float eval_diff(
        const float* __restrict__ u,
        int nx, int ny,
        int ix, int iy, int iz,
        float dx, float dy, float dz
    )
    {
        // c2 as in eval, which makes the difference form below exactly eager's stencil
        // -c0 u0 + c1 (u1 + u-1) + c2 (u2 + u-2); c0 itself is never applied.
        constexpr float c0 = 5.f / 2.f;
        constexpr float c1 = 4.f / 3.f;
        constexpr float c2 = 0.5f * c0 - c1;                     // -1/12

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
                c1 * (tap_diff(u[idx - stride_x], u0) + tap_diff(u[idx + stride_x], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_x], u0) + tap_diff(u[idx + 2*stride_x], u0))
            ) / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (
                c1 * (tap_diff(u[idx - stride_y], u0) + tap_diff(u[idx + stride_y], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_y], u0) + tap_diff(u[idx + 2*stride_y], u0))
            ) / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (
                c1 * (tap_diff(u[idx - stride_z], u0) + tap_diff(u[idx + stride_z], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_z], u0) + tap_diff(u[idx + 2*stride_z], u0))
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
        constexpr float c3 = (0.5f * c0 - c1) - c2;              // 1/90, see order 4

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
                c1 * (tap_diff(u[idx - stride_x], u0) + tap_diff(u[idx + stride_x], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_x], u0) + tap_diff(u[idx + 2*stride_x], u0)) +
                c3 * (tap_diff(u[idx - 3*stride_x], u0) + tap_diff(u[idx + 3*stride_x], u0))
            ) / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (
                c1 * (tap_diff(u[idx - stride_y], u0) + tap_diff(u[idx + stride_y], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_y], u0) + tap_diff(u[idx + 2*stride_y], u0)) +
                c3 * (tap_diff(u[idx - 3*stride_y], u0) + tap_diff(u[idx + 3*stride_y], u0))
            ) / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (
                c1 * (tap_diff(u[idx - stride_z], u0) + tap_diff(u[idx + stride_z], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_z], u0) + tap_diff(u[idx + 2*stride_z], u0)) +
                c3 * (tap_diff(u[idx - 3*stride_z], u0) + tap_diff(u[idx + 3*stride_z], u0))
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
        constexpr float c4 = ((0.5f * c0 - c1) - c2) - c3;       // -1/560, see order 4

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
                c1 * (tap_diff(u[idx - stride_x], u0) + tap_diff(u[idx + stride_x], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_x], u0) + tap_diff(u[idx + 2*stride_x], u0)) +
                c3 * (tap_diff(u[idx - 3*stride_x], u0) + tap_diff(u[idx + 3*stride_x], u0)) +
                c4 * (tap_diff(u[idx - 4*stride_x], u0) + tap_diff(u[idx + 4*stride_x], u0))
            ) / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {
            lap += (
                c1 * (tap_diff(u[idx - stride_y], u0) + tap_diff(u[idx + stride_y], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_y], u0) + tap_diff(u[idx + 2*stride_y], u0)) +
                c3 * (tap_diff(u[idx - 3*stride_y], u0) + tap_diff(u[idx + 3*stride_y], u0)) +
                c4 * (tap_diff(u[idx - 4*stride_y], u0) + tap_diff(u[idx + 4*stride_y], u0))
            ) / (dy * dy);
        }

        if constexpr (Direction & Z) {
            lap += (
                c1 * (tap_diff(u[idx - stride_z], u0) + tap_diff(u[idx + stride_z], u0)) +
                c2 * (tap_diff(u[idx - 2*stride_z], u0) + tap_diff(u[idx + 2*stride_z], u0)) +
                c3 * (tap_diff(u[idx - 3*stride_z], u0) + tap_diff(u[idx + 3*stride_z], u0)) +
                c4 * (tap_diff(u[idx - 4*stride_z], u0) + tap_diff(u[idx + 4*stride_z], u0))
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
                acc += coeff[k] * (tap_diff(u[idx + k*stride_x], u0) +
                                   tap_diff(u[idx - k*stride_x], u0));

            lap += acc / (dx * dx);
        }

        if constexpr (ND == 3 && (Direction & Y)) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] * (tap_diff(u[idx + k*stride_y], u0) +
                                   tap_diff(u[idx - k*stride_y], u0));

            lap += acc / (dy * dy);
        }

        if constexpr (Direction & Z) {

            float acc = 0.f;

            #pragma unroll 1
            for (int k = 1; k <= M; ++k)
                acc += coeff[k] * (tap_diff(u[idx + k*stride_z], u0) +
                                   tap_diff(u[idx - k*stride_z], u0));

            lap += acc / (dz * dz);
        }

        return lap;
    }
};