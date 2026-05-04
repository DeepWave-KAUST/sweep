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
