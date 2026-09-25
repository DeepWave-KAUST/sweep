#pragma once

#include "../../../core/dtype.h"   // BoundaryMode, BoundaryDtype, BOUNDARY_INT8_BLOCK
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdlib>


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

