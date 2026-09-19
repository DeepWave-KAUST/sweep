#pragma once
#include <torch/extension.h>
#include <cstddef>
#include <cstdint>
#include "../../common/wavetypes.h"

namespace visco_acoustic2d {

// Work-area size in bytes of the spectral step's cuFFT plan for a complex64
// (B, 1, nz, nx) grid on the CURRENT CUDA device; builds and caches the plan
// (fft.cu).  The Python side sizes the FFT_WS slot of forward_workspace /
// adjoint_workspace from it as ceil(bytes / 4) float32 elements.
size_t fft_workspace_bytes(int64_t B, int64_t nz, int64_t nx);

ForwardOutput forward(const ForwardInput& in);

BackwardOutput backward(const BackwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

BackwardOutput backward_ckpt(const BackwardInput& in);

BackwardOutput backward_recursive_ckpt(const BackwardInput& in);

RTMOutput rtm(const BackwardInput& in);

} // namespace visco_acoustic2d
