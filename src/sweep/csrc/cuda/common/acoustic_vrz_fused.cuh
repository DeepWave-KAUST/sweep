#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include "../operators/gradient.cuh"
#include "../operators/laplace.cuh"

// Functor building blocks for the fused AcousticVRZ backward kernels.
//
// The split backward path materialises the adjoint transpose buffers
//   aq0 = vp^2 * lambda,  aqx = (dx b*kappa)*lambda,  aqz = (dz b*kappa)*lambda
// (and the gradient buffers c_x/c_z/e_x/e_z), then re-reads them with a laplace
// / gradient stencil.  The fused kernels instead recompute each stencil tap on
// the fly from a time-invariant coefficient field times the current wavefield,
// exactly as elastic_adjoint_fused.cuh does for the elastic solvers.
//
// Bit-exactness: gradient<>() already routes through centered_gradient_stencil
// (operators/gradient.cuh), so a matching product accessor reproduces the split
// numerics tap-for-tap.  centered_laplace1d_stencil below mirrors LaplaceGPU's
// coefficient expressions (operators/laplace.cuh) line-for-line, including the
// `/(h*h)` division order, so the fused laplace is bit-identical too.

// value_at(off) = a[idx + off*stride] * b[idx + off*stride]
struct VrzProductAccessor {
    const float* __restrict__ a;
    const float* __restrict__ b;
    int idx;
    int stride;

    __device__ __forceinline__ float operator()(int off) const
    {
        int j = idx + off * stride;
        return a[j] * b[j];
    }
};

// Second derivative along one axis, applied to a functor value_at(off).
// Coefficient expressions and division order match LaplaceGPU<Order> exactly.
template<int Order, typename Accessor>
__device__ __forceinline__ float centered_laplace1d_stencil(
    const Accessor& v,
    float h,
    int M,
    const float* __restrict__ coeff
) {
    if constexpr (Order == 2) {
        return (v(-1) + v(1) - 2.f * v(0)) / (h * h);
    } else if constexpr (Order == 4) {
        constexpr float c0 = 5.f / 2.f;
        constexpr float c1 = 4.f / 3.f;
        constexpr float c2 = -1.f / 12.f;
        return (c1 * (v(-1) + v(1))
              + c2 * (v(-2) + v(2))
              - c0 * v(0)) / (h * h);
    } else if constexpr (Order == 6) {
        constexpr float c0 = 49.f / 18.f;
        constexpr float c1 = 3.f / 2.f;
        constexpr float c2 = -3.f / 20.f;
        constexpr float c3 = 1.f / 90.f;
        return (c1 * (v(-1) + v(1))
              + c2 * (v(-2) + v(2))
              + c3 * (v(-3) + v(3))
              - c0 * v(0)) / (h * h);
    } else if constexpr (Order == 8) {
        constexpr float c0 = 205.f / 72.f;
        constexpr float c1 = 8.f / 5.f;
        constexpr float c2 = -1.f / 5.f;
        constexpr float c3 = 8.f / 315.f;
        constexpr float c4 = -1.f / 560.f;
        return (c1 * (v(-1) + v(1))
              + c2 * (v(-2) + v(2))
              + c3 * (v(-3) + v(3))
              + c4 * (v(-4) + v(4))
              - c0 * v(0)) / (h * h);
    } else {
        float acc = 0.f;
        #pragma unroll 1
        for (int k = 1; k <= M; ++k)
            acc += coeff[k] * (v(k) + v(-k));
        return (acc - coeff[0] * v(0)) / (h * h);
    }
}
