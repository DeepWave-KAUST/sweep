// Fused per-call model-coefficient kernels (see derived_models.h).
//
// Every product / sum / sqrt / division is a ``__f*_rn`` intrinsic: one IEEE
// round-to-nearest operation each, in exactly the association of the torch
// expression it replaced, and immune to the FMA contraction and approximate
// sqrt/div that --use_fast_math would otherwise apply.  That is what keeps the
// coefficients bit-identical to ``rho * vs * vs`` and friends evaluated by
// torch one elementwise op at a time.
#include "derived_models.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <algorithm>

namespace {

constexpr int kThreads = 256;

inline unsigned grid_for(int64_t n)
{
    return static_cast<unsigned>(std::min<int64_t>((n + kThreads - 1) / kThreads, 65535));
}

// mu = (rho*vs)*vs ; lambda = rho*((vp*vp) - ((2*vs)*vs))
__global__ void lame_kernel(const float* __restrict__ vp, const float* __restrict__ vs,
                            const float* __restrict__ rho, float* __restrict__ mu,
                            float* __restrict__ lambda, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float v = vp[i], s = vs[i], r = rho[i];
        mu[i] = __fmul_rn(__fmul_rn(r, s), s);
        lambda[i] = __fmul_rn(r, __fsub_rn(__fmul_rn(v, v), __fmul_rn(__fmul_rn(2.0f, s), s)));
    }
}

// c33 = rho*(vp*vp) ; c11 = c33*(1+(2*eps)) ; c13 = c33*sqrt(1+(2*delta)) ; inv_rho = 1/rho
__global__ void vti_stiffness_kernel(const float* __restrict__ vp, const float* __restrict__ epsilon,
                                     const float* __restrict__ delta, const float* __restrict__ rho,
                                     float* __restrict__ c11, float* __restrict__ c13,
                                     float* __restrict__ c33, float* __restrict__ inv_rho, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float v = vp[i], r = rho[i];
        const float rho_vp2 = __fmul_rn(r, __fmul_rn(v, v));
        c33[i] = rho_vp2;
        c11[i] = __fmul_rn(rho_vp2, __fadd_rn(1.0f, __fmul_rn(2.0f, epsilon[i])));
        c13[i] = __fmul_rn(rho_vp2, __fsqrt_rn(__fadd_rn(1.0f, __fmul_rn(2.0f, delta[i]))));
        inv_rho[i] = __fdiv_rn(1.0f, r);
    }
}

__global__ void reciprocal_kernel(const float* __restrict__ z, float* __restrict__ inv_z, int64_t n)
{
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)
        inv_z[i] = __fdiv_rn(1.0f, z[i]);
}

void check_operand(const torch::Tensor& t, const torch::Tensor& like, const char* name)
{
    TORCH_CHECK(t.defined() && t.is_cuda() && t.scalar_type() == torch::kFloat && t.is_contiguous(),
                "derived_models: ", name, " must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(t.numel() == like.numel(),
                "derived_models: ", name, " has ", t.numel(), " elements, expected ", like.numel());
}

}  // namespace

namespace derived {

void derive_lame(const torch::Tensor& vp, const torch::Tensor& vs, const torch::Tensor& rho,
                 torch::Tensor& mu, torch::Tensor& lambda)
{
    check_operand(vp, vp, "vp");
    check_operand(vs, vp, "vs");
    check_operand(rho, vp, "rho");
    check_operand(mu, vp, "mu");
    check_operand(lambda, vp, "lambda");
    const int64_t n = vp.numel();
    if (n == 0) return;
    c10::cuda::CUDAGuard guard(vp.device());
    lame_kernel<<<grid_for(n), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        vp.data_ptr<float>(), vs.data_ptr<float>(), rho.data_ptr<float>(),
        mu.data_ptr<float>(), lambda.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void derive_vti_stiffness(const torch::Tensor& vp, const torch::Tensor& epsilon,
                          const torch::Tensor& delta, const torch::Tensor& rho,
                          torch::Tensor& c11, torch::Tensor& c13, torch::Tensor& c33,
                          torch::Tensor& inv_rho)
{
    check_operand(vp, vp, "vp");
    check_operand(epsilon, vp, "epsilon");
    check_operand(delta, vp, "delta");
    check_operand(rho, vp, "rho");
    check_operand(c11, vp, "c11");
    check_operand(c13, vp, "c13");
    check_operand(c33, vp, "c33");
    check_operand(inv_rho, vp, "inv_rho");
    const int64_t n = vp.numel();
    if (n == 0) return;
    c10::cuda::CUDAGuard guard(vp.device());
    vti_stiffness_kernel<<<grid_for(n), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        vp.data_ptr<float>(), epsilon.data_ptr<float>(), delta.data_ptr<float>(),
        rho.data_ptr<float>(), c11.data_ptr<float>(), c13.data_ptr<float>(),
        c33.data_ptr<float>(), inv_rho.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void derive_reciprocal(const torch::Tensor& z, torch::Tensor& inv_z)
{
    check_operand(z, z, "z");
    check_operand(inv_z, z, "inv_z");
    const int64_t n = z.numel();
    if (n == 0) return;
    c10::cuda::CUDAGuard guard(z.device());
    reciprocal_kernel<<<grid_for(n), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        z.data_ptr<float>(), inv_z.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace derived
