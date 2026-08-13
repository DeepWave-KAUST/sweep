#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>

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

// Storage pointers passed into boundary save/load kernels.  Three
// parallel pointer sets — FP32 / FP16 / BF16 — are populated depending
// on the storage dtype chosen.  ``dtype`` and ``use_fp16`` together
// drive the kernel dispatch.  Kernels cast at the storage boundary so
// arithmetic remains FP32.  Drives are wired via Python BoundaryOptions
// ``storage_dtype`` / env var SWEEP_BOUNDARY_DTYPE.
struct GeneralBoundaryPointer {
    // FP32 storage (default)
    float* __restrict__ left = nullptr;
    float* __restrict__ right = nullptr;

    float* __restrict__ front = nullptr;
    float* __restrict__ back = nullptr;

    float* __restrict__ bottom = nullptr;
    float* __restrict__ top = nullptr;

    float* __restrict__ last_two = nullptr;

    // FP16 storage (populated when dtype == FP16).  The payload is
    // per-block NORMALIZED (same two-pass save/restore and scale layout
    // as INT8, see quantize_fp16_kernel): a bare __half cast flushes
    // everything below 2^-24 to zero, which wipes the velocity faces of
    // elastic wavefields.  last_two stays FP32 — it's a wavefield
    // snapshot used to bootstrap backward, precision is critical there.
    __half* __restrict__ left_h = nullptr;
    __half* __restrict__ right_h = nullptr;

    __half* __restrict__ front_h = nullptr;
    __half* __restrict__ back_h = nullptr;

    __half* __restrict__ bottom_h = nullptr;
    __half* __restrict__ top_h = nullptr;

    // BF16 storage (populated when dtype == BF16).
    __nv_bfloat16* __restrict__ left_bf = nullptr;
    __nv_bfloat16* __restrict__ right_bf = nullptr;

    __nv_bfloat16* __restrict__ front_bf = nullptr;
    __nv_bfloat16* __restrict__ back_bf = nullptr;

    __nv_bfloat16* __restrict__ bottom_bf = nullptr;
    __nv_bfloat16* __restrict__ top_bf = nullptr;

    // INT8 storage (populated when dtype == INT8).  Each face has a
    // uint8 quantized buffer (same element count as FP32) and a float
    // per-block scale buffer of size ceil(nelem / BOUNDARY_INT8_BLOCK).
    uint8_t* __restrict__ left_q = nullptr;
    uint8_t* __restrict__ right_q = nullptr;

    uint8_t* __restrict__ front_q = nullptr;
    uint8_t* __restrict__ back_q = nullptr;

    uint8_t* __restrict__ bottom_q = nullptr;
    uint8_t* __restrict__ top_q = nullptr;

    float* __restrict__ left_scale = nullptr;
    float* __restrict__ right_scale = nullptr;

    float* __restrict__ front_scale = nullptr;
    float* __restrict__ back_scale = nullptr;

    float* __restrict__ bottom_scale = nullptr;
    float* __restrict__ top_scale = nullptr;

    BoundaryDtype dtype = BoundaryDtype::FP32;
    bool use_fp16 = false;   // == (dtype == FP16), kept for back-compat
};

// Legacy env-var gate (FP16 only).  Prefer the per-PropTorch option
// passed through BoundaryOptions.storage_dtype.
static inline bool sweep_use_fp16_boundary() {
    static bool flag = []() {
        const char* env = std::getenv("SWEEP_FP16_BOUNDARY");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return flag;
}

// New unified dtype gate, driven by env var SWEEP_BOUNDARY_DTYPE in
// {fp32, fp16, bf16, int8}.  The Python wrapper sets this per-call
// before invoking the forward kernel.  Returning FP32 when unset
// preserves the legacy default.
static inline BoundaryDtype sweep_boundary_dtype_env() {
    const char* env = std::getenv("SWEEP_BOUNDARY_DTYPE");
    if (env == nullptr) {
        // Fall back to legacy FP16 gate so old test paths still work.
        return sweep_use_fp16_boundary() ? BoundaryDtype::FP16 : BoundaryDtype::FP32;
    }
    if (env[0] == 'f' && env[1] == 'p' && env[2] == '1' && env[3] == '6') return BoundaryDtype::FP16;
    if (env[0] == 'b' && env[1] == 'f' && env[2] == '1' && env[3] == '6') return BoundaryDtype::BF16;
    if (env[0] == 'i' && env[1] == 'n' && env[2] == 't' && env[3] == '8') return BoundaryDtype::INT8;
    return BoundaryDtype::FP32;
}
