#pragma once

#include <torch/extension.h>

#include <cstdint>

namespace sweep_cpu::ops {

struct StencilCoefficients {
    int M;
    const float* lap;
    const float* grad;
};

inline bool uses_runtime_coefficients(int M)
{
    return M > 4;
}

inline bool runtime_stencil_coefficients_are_valid(
    int M,
    const torch::Tensor& lap,
    const torch::Tensor& grad,
    bool centered_gradient
)
{
    if (!uses_runtime_coefficients(M)) return true;
    const int64_t grad_min = centered_gradient ? static_cast<int64_t>(M) + 1 : static_cast<int64_t>(M);
    return lap.defined() && grad.defined()
        && lap.is_contiguous() && grad.is_contiguous()
        && lap.scalar_type() == torch::kFloat32 && grad.scalar_type() == torch::kFloat32
        && lap.numel() >= static_cast<int64_t>(M) + 1
        && grad.numel() >= grad_min;
}

#define SWEEP_CPU_DISPATCH_STENCIL(MVAL, FUNC, ...)                                      \
    switch (MVAL) {                                                                       \
        case 1: return FUNC<1>(__VA_ARGS__);                                              \
        case 2: return FUNC<2>(__VA_ARGS__);                                              \
        case 3: return FUNC<3>(__VA_ARGS__);                                              \
        case 4: return FUNC<4>(__VA_ARGS__);                                              \
        default:                                                                           \
            TORCH_CHECK((MVAL) > 4, "CPU finite-difference half-order M must be >= 1");    \
            return FUNC<-1>(__VA_ARGS__);                                                 \
    }

#define SWEEP_CPU_DISPATCH_STENCIL_BODY(MVAL, BODY_MACRO)                                \
    switch (MVAL) {                                                                       \
        case 1: { BODY_MACRO(1); }                                                        \
        case 2: { BODY_MACRO(2); }                                                        \
        case 3: { BODY_MACRO(3); }                                                        \
        case 4: { BODY_MACRO(4); }                                                        \
        default: {                                                                         \
            TORCH_CHECK((MVAL) > 4, "CPU finite-difference half-order M must be >= 1");    \
            BODY_MACRO(-1);                                                               \
        }                                                                                 \
    }

template <int M>
inline int stencil_half_order(const StencilCoefficients& stencil)
{
    if constexpr (M == -1) {
        return stencil.M;
    } else {
        return M;
    }
}

template <int M>
inline float laplace_value(
    const float* u,
    int64_t idx,
    int64_t stride,
    float inv_h2,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    const float u0 = u[idx];
    if constexpr (M == -1) {
        float acc = 0.0f;
        const int radius = stencil_half_order<M>(stencil);
        for (int m = 1; m <= radius; ++m) {
            acc += stencil.lap[m] * (u[idx - m * stride] + u[idx + m * stride]);
        }
        return (acc - stencil.lap[0] * u0) * inv_h2;
    } else if constexpr (M == 1) {
        return (u[idx - stride] + u[idx + stride] - 2.0f * u0) * inv_h2;
    } else if constexpr (M == 2) {
        constexpr float c0 = 5.0f / 2.0f;
        constexpr float c1 = 4.0f / 3.0f;
        constexpr float c2 = -1.0f / 12.0f;
        return (c1 * (u[idx - stride] + u[idx + stride])
              + c2 * (u[idx - 2 * stride] + u[idx + 2 * stride])
              - c0 * u0) * inv_h2;
    } else if constexpr (M == 3) {
        constexpr float c0 = 49.0f / 18.0f;
        constexpr float c1 = 3.0f / 2.0f;
        constexpr float c2 = -3.0f / 20.0f;
        constexpr float c3 = 1.0f / 90.0f;
        return (c1 * (u[idx - stride] + u[idx + stride])
              + c2 * (u[idx - 2 * stride] + u[idx + 2 * stride])
              + c3 * (u[idx - 3 * stride] + u[idx + 3 * stride])
              - c0 * u0) * inv_h2;
    } else {
        constexpr float c0 = 205.0f / 72.0f;
        constexpr float c1 = 8.0f / 5.0f;
        constexpr float c2 = -1.0f / 5.0f;
        constexpr float c3 = 8.0f / 315.0f;
        constexpr float c4 = -1.0f / 560.0f;
        return (c1 * (u[idx - stride] + u[idx + stride])
              + c2 * (u[idx - 2 * stride] + u[idx + 2 * stride])
              + c3 * (u[idx - 3 * stride] + u[idx + 3 * stride])
              + c4 * (u[idx - 4 * stride] + u[idx + 4 * stride])
              - c0 * u0) * inv_h2;
    }
}

template <int M>
inline float centered_coeff(int m, const StencilCoefficients& stencil = {M, nullptr, nullptr})
{
    if constexpr (M == -1) {
        return stencil.grad[m];
    } else if constexpr (M == 1) {
        return 0.5f;
    } else if constexpr (M == 2) {
        return m == 1 ? 8.0f / 12.0f : -1.0f / 12.0f;
    } else if constexpr (M == 3) {
        return m == 1 ? 3.0f / 4.0f : (m == 2 ? -3.0f / 20.0f : 1.0f / 60.0f);
    } else {
        return m == 1 ? 4.0f / 5.0f : (m == 2 ? -1.0f / 5.0f : (m == 3 ? 4.0f / 105.0f : -1.0f / 280.0f));
    }
}

template <int M>
inline float centered_grad_value(
    const float* u,
    int64_t idx,
    int64_t stride,
    float inv_h,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    float acc = 0.0f;
    const int radius = stencil_half_order<M>(stencil);
    for (int m = 1; m <= radius; ++m) {
        acc += centered_coeff<M>(m, stencil) * (u[idx + m * stride] - u[idx - m * stride]);
    }
    return acc * inv_h;
}

template <int M>
inline float centered_grad_1d(
    const float* u,
    int64_t i,
    float inv_h,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    return centered_grad_value<M>(u, i, 1, inv_h, stencil);
}

template <int M>
inline float centered_grad_weighted_value(
    const float* u,
    const float* weight,
    int64_t idx,
    int64_t stride,
    int64_t coord,
    float inv_h,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    float acc = 0.0f;
    const int radius = stencil_half_order<M>(stencil);
    for (int m = 1; m <= radius; ++m) {
        const float c = centered_coeff<M>(m, stencil);
        acc += c * (weight[coord + m] * u[idx + m * stride] - weight[coord - m] * u[idx - m * stride]);
    }
    return acc * inv_h;
}

template <int M>
inline float staggered_coeff(int m, const StencilCoefficients& stencil = {M, nullptr, nullptr})
{
    if constexpr (M == -1) {
        return stencil.grad[m - 1];
    } else if constexpr (M == 1) {
        return 1.0f;
    } else if constexpr (M == 2) {
        return m == 1 ? 9.0f / 8.0f : -1.0f / 24.0f;
    } else if constexpr (M == 3) {
        return m == 1 ? 75.0f / 64.0f : (m == 2 ? -25.0f / 384.0f : 3.0f / 640.0f);
    } else {
        return m == 1 ? 1225.0f / 1024.0f
                      : (m == 2 ? -245.0f / 3072.0f : (m == 3 ? 49.0f / 5120.0f : -5.0f / 7168.0f));
    }
}

template <int M, bool Forward>
inline float staggered_grad_value(
    const float* u,
    int64_t idx,
    int64_t stride,
    float inv_h,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    float acc = 0.0f;
    const int radius = stencil_half_order<M>(stencil);
    for (int m = 0; m < radius; ++m) {
        const float c = staggered_coeff<M>(m + 1, stencil);
        if constexpr (Forward) {
            acc += c * (u[idx + (m + 1) * stride] - u[idx - m * stride]);
        } else {
            acc += c * (u[idx + m * stride] - u[idx - (m + 1) * stride]);
        }
    }
    return acc * inv_h;
}

template <int M>
inline float sgrad_forward(
    const float* u,
    int64_t idx,
    int64_t stride,
    float inv_h,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    return staggered_grad_value<M, true>(u, idx, stride, inv_h, stencil);
}

template <int M>
inline float sgrad_backward(
    const float* u,
    int64_t idx,
    int64_t stride,
    float inv_h,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    return staggered_grad_value<M, false>(u, idx, stride, inv_h, stencil);
}

template <int M, bool Forward>
inline void scatter_sgradient_adjoint(
    float bar,
    float* dst,
    int64_t idx,
    int64_t stride,
    float inv_h,
    const StencilCoefficients& stencil = {M, nullptr, nullptr}
)
{
    const int radius = stencil_half_order<M>(stencil);
    for (int m = 0; m < radius; ++m) {
        const float value = bar * staggered_coeff<M>(m + 1, stencil) * inv_h;
        if constexpr (Forward) {
            dst[idx + (m + 1) * stride] += value;
            dst[idx - m * stride] -= value;
        } else {
            dst[idx + m * stride] += value;
            dst[idx - (m + 1) * stride] -= value;
        }
    }
}

}  // namespace sweep_cpu::ops
