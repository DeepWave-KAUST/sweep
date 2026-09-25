#pragma once
// core/dtype.h -- the boundary storage tags and mode, torch- and CUDA-free: what
// core/buf.h needs (a Buf carries a BoundaryDtype) without cuda_fp16.h, which
// the boundary saver's own header (cuda/common/boundary/types.cuh) still
// includes for its __half / __nv_bfloat16 pointers.  A torch shim compiled
// with only the pip CUDA runtime headers has no <nv/target>, so nothing on the
// shim's include path may reach cuda_fp16.h.
#include <cstdint>

enum BoundaryMode {
    BOUNDARY_SAVE = 0,
    BOUNDARY_RESTORE = 1
};

// Boundary-buffer storage dtype.  Compute always stays FP32; this only
// changes the per-cell storage of the saved boundary strip.
enum class BoundaryDtype : int {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
};

// Per-block size for INT8 symmetric quantization.  Each block stores
// one FP32 max_abs scale and BOUNDARY_INT8_BLOCK uint8 quantized cells.
// Compression ratio = 4·B / (B + 4) where B = BOUNDARY_INT8_BLOCK.
// B=256 → ratio = 1024/260 ≈ 3.94×.  Smaller B improves local dynamic-
// range adaptation at the cost of higher metadata overhead.
constexpr int BOUNDARY_INT8_BLOCK = 256;
