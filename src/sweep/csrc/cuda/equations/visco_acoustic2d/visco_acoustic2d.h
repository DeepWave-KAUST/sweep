#pragma once
#include <cstddef>
#include <cstdint>
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../../core/runner.h"

namespace visco_acoustic2d {

// Work-area size in bytes of the spectral step's cuFFT plan for a complex64
// (B, 1, nz, nx) grid on the CURRENT CUDA device; builds and caches the plan
// (fft.cu).  The Python side sizes the FFT_WS slot of forward_workspace /
// adjoint_workspace from it as ceil(bytes / 4) float32 elements.
size_t fft_workspace_bytes(int64_t B, int64_t nz, int64_t nx);

ForwardOutputCore forward_core(const ForwardInputCore& in);

BackwardOutputCore backward_core(const BackwardInputCore& in);

BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in);

RTMOutputCore rtm_core(const BackwardInputCore& in);

} // namespace visco_acoustic2d
