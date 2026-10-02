// visco_acoustic2d/spectral_ops.cu -- the kernels and launchers declared in
// spectral_ops.cuh (see its header comment), compiled once.
#include "spectral_ops.cuh"

#include "../../../core/device.h"

namespace visco_ops {

constexpr int kThreads = 256;

inline dim3 grid_for(int64_t n)
{
    // grid-stride loops below: a bounded grid whatever n is
    const int64_t blocks = (n + kThreads - 1) / kThreads;
    return dim3(static_cast<unsigned>(blocks < 65535 ? (blocks > 0 ? blocks : 1) : 65535));
}

#define VISCO_OPS_FOR(i, n) \
    for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < (n); \
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)

// C0.copy_(x): float -> complex64 is fetch_and_cast, (x, 0).
static __global__ void promote_kernel(float2* __restrict__ c, const float* __restrict__ x, int64_t n)
{
    VISCO_OPS_FOR(i, n) c[i] = make_float2(x[i], 0.f);
}

// at::mul_out(out, g, in) with g a real (nz, nx) grid broadcast over the
// batch and in a complex spectrum: (g*re - 0*im, g*im + 0*re), g first.
static __global__ void cmul_grid_kernel(float2* __restrict__ out, const float* __restrict__ g,
                                 const float2* __restrict__ in, int64_t n, int64_t ng)
{
    VISCO_OPS_FOR(i, n) {
        const float k = g[i % ng];
        const float2 v = in[i];
        out[i] = make_float2(k * v.x - 0.f * v.y, k * v.y + 0.f * v.x);
    }
}

// Tensor::mul_(Scalar s) on complex64: (s*re - 0*im, s*0 ... ) -- ATen loads
// the scalar as complex<float>(s, 0) and multiplies; same expression.
static __global__ void cscale_kernel(float2* __restrict__ c, float s, int64_t n)
{
    VISCO_OPS_FOR(i, n) {
        const float2 v = c[i];
        c[i] = make_float2(v.x * s - v.y * 0.f, v.x * 0.f + v.y * s);
    }
}

// R1.copy_(at::real(C)): the strided gather of the real parts.
static __global__ void real_copy_kernel(float* __restrict__ out, const float2* __restrict__ c, int64_t n)
{
    VISCO_OPS_FOR(i, n) out[i] = c[i].x;
}

template <class Op>
static __global__ void binary_kernel(float* __restrict__ out, const float* __restrict__ a, int sa,
                              const float* __restrict__ b, int sb, int64_t n)
{
    Op op;
    VISCO_OPS_FOR(i, n) out[i] = op(a[i * sa], b[i * sb]);
}

// y.add_(x) / y.sub_(x) / y.mul_(x) in place.
template <class Op>
static __global__ void inplace_kernel(float* __restrict__ y, const float* __restrict__ x, int sx, int64_t n)
{
    Op op;
    VISCO_OPS_FOR(i, n) y[i] = op(y[i], x[i * sx]);
}

// y.add_(x, alpha): a + alpha * b.
static __global__ void axpy_kernel(float* __restrict__ y, const float* __restrict__ x, int sx, float alpha, int64_t n)
{
    VISCO_OPS_FOR(i, n) y[i] = y[i] + alpha * x[i * sx];
}

// y.div_(Scalar d) on float: y * (1.0f / d) -- the reciprocal is the caller's.
static __global__ void scale_kernel(float* __restrict__ y, float s, int64_t n)
{
    VISCO_OPS_FOR(i, n) y[i] = y[i] * s;
}

// u.narrow(-2, 0, M).zero_() and the three other bands, on every (b) slab.
static __global__ void zero_halo_kernel(float* __restrict__ u, int64_t slabs, int nz, int nx, int M)
{
    const int64_t per = static_cast<int64_t>(nz) * nx;
    VISCO_OPS_FOR(i, slabs * per) {
        const int64_t r = i % per;
        const int iz = static_cast<int>(r / nx);
        const int ix = static_cast<int>(r % nx);
        if (iz < M || iz >= nz - M || ix < M || ix >= nx - M) u[i] = 0.f;
    }
}

#undef VISCO_OPS_FOR

// ---- launchers ------------------------------------------------------------

void promote(float2* c, const float* x, int64_t n)
{
    if (n == 0) return;
    promote_kernel<<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(c, x, n);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

void cmul_grid(float2* out, const float* g, const float2* in, int64_t n, int64_t ng)
{
    if (n == 0) return;
    cmul_grid_kernel<<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(out, g, in, n, ng);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

void cscale(float2* c, float s, int64_t n)
{
    if (n == 0) return;
    cscale_kernel<<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(c, s, n);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

void real_copy(float* out, const float2* c, int64_t n)
{
    if (n == 0) return;
    real_copy_kernel<<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(out, c, n);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

template <class Op>
void binary(float* out, const float* a, int sa, const float* b, int sb, int64_t n)
{
    if (n == 0) return;
    binary_kernel<Op><<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(out, a, sa, b, sb, n);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

template <class Op>
void inplace(float* y, const float* x, int sx, int64_t n)
{
    if (n == 0) return;
    inplace_kernel<Op><<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(y, x, sx, n);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

void axpy(float* y, const float* x, int sx, float alpha, int64_t n)
{
    if (n == 0) return;
    axpy_kernel<<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(y, x, sx, alpha, n);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

void scale(float* y, float s, int64_t n)
{
    if (n == 0) return;
    scale_kernel<<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(y, s, n);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

void zero_halo(float* u, int64_t slabs, int nz, int nx, int M)
{
    const int64_t n = slabs * static_cast<int64_t>(nz) * nx;
    if (n == 0 || M <= 0) return;
    zero_halo_kernel<<<grid_for(n), kThreads, 0, sweep::current_stream()>>>(u, slabs, nz, nx, M);
    SWEEP_KERNEL_LAUNCH_CHECK();
}

template void binary<SubOp>(float*, const float*, int, const float*, int, int64_t);
template void binary<MulOp>(float*, const float*, int, const float*, int, int64_t);
template void inplace<SubOp>(float*, const float*, int, int64_t);
template void inplace<AddOp>(float*, const float*, int, int64_t);
template void inplace<MulOp>(float*, const float*, int, int64_t);

}  // namespace visco_ops
