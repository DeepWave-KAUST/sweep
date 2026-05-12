#pragma once

enum BoundaryMode {
    BOUNDARY_SAVE = 0,
    BOUNDARY_RESTORE = 1
};

struct GeneralBoundaryPointer {
    float* __restrict__ left = nullptr;
    float* __restrict__ right = nullptr;

    float* __restrict__ front = nullptr;
    float* __restrict__ back = nullptr;

    float* __restrict__ bottom = nullptr;
    float* __restrict__ top = nullptr;

    float* __restrict__ last_two = nullptr;
};
