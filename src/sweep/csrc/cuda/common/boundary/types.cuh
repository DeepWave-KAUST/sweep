#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
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
};

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

    // FP16 storage (populated when dtype == FP16).  last_two stays FP32
    // — it's a wavefield snapshot used to bootstrap backward, precision
    // is critical there.
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
