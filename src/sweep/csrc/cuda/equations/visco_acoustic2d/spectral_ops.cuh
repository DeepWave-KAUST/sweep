#pragma once
// ---------------------------------------------------------------------------
// visco_acoustic2d/spectral_ops.cuh -- the spectral terms' elementwise
// arithmetic as CUDA kernels.
//
// These replace the ATen calls the visco pipeline used to make on its pool
// slots (at::mul_out, at::sub_out, Tensor::add_/sub_/mul_/div_/copy_,
// at::real, narrow().zero_()).  Each kernel evaluates, per cell, the same
// IEEE expression the ATen kernel it replaces evaluated, on the same operands
// in the same order (the header note above each launcher names the call):
//
//   * a real grid promoted to complex is (x, 0) -- a cast, no arithmetic;
//   * a complex spectrum times a real filter grid k is c10::complex's
//     (a*c - b*d, a*d + b*c) with (a, b) = (k, 0): (k*c - 0*d, k*d + 0*c);
//   * a complex tensor times a real Scalar s (the inverse-FFT normalisation)
//     is the same product with (a, b) = (s, 0);
//   * Tensor::div_(Scalar dt) on float multiplies by opmath 1.0f / dt
//     (BinaryDivTrueKernel.cu's CPU-scalar path), not divides;
//   * Tensor::add_(other, alpha) is a + alpha * b in one lambda, which nvcc
//     contracts to an FMA in ATen's build as it does here.
//
// The complex slots are float32 pairs, (..., 2) contiguous -- cuFFT's
// interleaved layout and what at::view_as_complex used to alias; a kernel
// that reads "the real part" of one reads its floats at stride 2.  Every
// launch goes on sweep::current_stream(), like every other launch in the
// core.
//
// The kernels and their launchers are compiled once, in spectral_ops.cu: this
// header is included by four translation units, and a kernel defined in it is
// compiled into each of them.  binary / inplace are instantiated there for the
// operations the pipeline uses (add one there when a new one is needed).
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cuda_runtime.h>

namespace visco_ops {

// at::sub_out(out, a, b) / at::mul_out(out, a, b) on float grids; ``sa`` /
// ``sb`` are element strides (1 for a real grid, 2 for the real part of a
// complex slot -- what at::real's view handed the ATen kernel).
struct SubOp { __device__ float operator()(float a, float b) const { return a - b; } };
struct MulOp { __device__ float operator()(float a, float b) const { return a * b; } };
struct AddOp { __device__ float operator()(float a, float b) const { return a + b; } };

// ---- launchers (spectral_ops.cu) --------------------------------------------

// C0.copy_(x): (x, 0).
void promote(float2* c, const float* x, int64_t n);
// at::mul_out(out, g, in): a real (nz, nx) grid times a complex spectrum.
void cmul_grid(float2* out, const float* g, const float2* in, int64_t n, int64_t ng);
// Tensor::mul_(Scalar s) on complex64.
void cscale(float2* c, float s, int64_t n);
// R1.copy_(at::real(C)).
void real_copy(float* out, const float2* c, int64_t n);
// out = Op(a, b) elementwise; sa / sb are element strides.
template <class Op>
void binary(float* out, const float* a, int sa, const float* b, int sb, int64_t n);
// y = Op(y, x) in place.
template <class Op>
void inplace(float* y, const float* x, int sx, int64_t n);
// y.add_(x, alpha).
void axpy(float* y, const float* x, int sx, float alpha, int64_t n);
// y * s (y.div_(d) with s = 1.0f / d).
void scale(float* y, float s, int64_t n);
// Zero the M-wide bands of every (nz, nx) slab.
void zero_halo(float* u, int64_t slabs, int nz, int nx, int M);

}  // namespace visco_ops
