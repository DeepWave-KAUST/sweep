#pragma once

#include <cuda_runtime.h>

#include "../context.h"
#include "types.cuh"

__global__ void boundary_kernel2d(
    float* __restrict__ u,

    float* __restrict__ top,
    float* __restrict__ bottom,
    float* __restrict__ left,
    float* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

__global__ void boundary_kernel3d(
    float* __restrict__ u,

    float* __restrict__ top,
    float* __restrict__ bottom,

    float* __restrict__ front,
    float* __restrict__ back,

    float* __restrict__ left,
    float* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

__global__ void boundary_kernel3d_compact(
    float* __restrict__ u,

    float* __restrict__ top,
    float* __restrict__ bottom,

    float* __restrict__ front,
    float* __restrict__ back,

    float* __restrict__ left,
    float* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

// FP16-storage variants: storage buffers are __half*, but kernel reads
// from / writes to FP32 wavefield ``u``.  Cast happens at the storage
// boundary only — compute stays FP32 throughout.
__global__ void boundary_kernel3d_fp16(
    float* __restrict__ u,

    __half* __restrict__ top,
    __half* __restrict__ bottom,

    __half* __restrict__ front,
    __half* __restrict__ back,

    __half* __restrict__ left,
    __half* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

// 2-D FP16 variant.  Same boundary geometry as boundary_kernel2d, only
// the stored faces are __half* (cast on save / load).
__global__ void boundary_kernel2d_fp16(
    float* __restrict__ u,

    __half* __restrict__ top,
    __half* __restrict__ bottom,
    __half* __restrict__ left,
    __half* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

__global__ void boundary_kernel3d_compact_fp16(
    float* __restrict__ u,

    __half* __restrict__ top,
    __half* __restrict__ bottom,

    __half* __restrict__ front,
    __half* __restrict__ back,

    __half* __restrict__ left,
    __half* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

// BF16-storage variants — same role as the FP16 ones; switch is in
// BoundaryDtype.  Compute stays FP32.
__global__ void boundary_kernel2d_bf16(
    float* __restrict__ u,

    __nv_bfloat16* __restrict__ top,
    __nv_bfloat16* __restrict__ bottom,
    __nv_bfloat16* __restrict__ left,
    __nv_bfloat16* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

__global__ void boundary_kernel3d_bf16(
    float* __restrict__ u,

    __nv_bfloat16* __restrict__ top,
    __nv_bfloat16* __restrict__ bottom,

    __nv_bfloat16* __restrict__ front,
    __nv_bfloat16* __restrict__ back,

    __nv_bfloat16* __restrict__ left,
    __nv_bfloat16* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);

__global__ void boundary_kernel3d_compact_bf16(
    float* __restrict__ u,

    __nv_bfloat16* __restrict__ top,
    __nv_bfloat16* __restrict__ bottom,

    __nv_bfloat16* __restrict__ front,
    __nv_bfloat16* __restrict__ back,

    __nv_bfloat16* __restrict__ left,
    __nv_bfloat16* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode,
    int tangent_pad = 0
);
