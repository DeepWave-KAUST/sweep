#pragma once
// host/include/cuda_bf16.h -- the bfloat16 storage type of the boundary
// saver's bf16 mode, as the host core sees it.  Round-to-nearest-even, like
// __float2bfloat16 on the device, so bf16 strips hold the same bits on both
// cores.  Only the two conversions the tree uses are provided.
#include "cuda_runtime.h"
#include <cstdint>
#include <cstring>

struct __nv_bfloat16 {
    unsigned short __x;   // the upper 16 bits of an IEEE 754 binary32
};

inline __nv_bfloat16 __float2bfloat16(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    uint16_t b;
    if (((u >> 23) & 0xFFu) == 0xFFu && (u & 0x7FFFFFu)) {
        b = static_cast<uint16_t>((u >> 16) | 0x40u);  // nan stays a (quiet) nan
    } else {
        const uint32_t lsb = (u >> 16) & 1u;
        b = static_cast<uint16_t>((u + 0x7FFFu + lsb) >> 16);   // round to nearest even
    }
    return __nv_bfloat16{b};
}

inline __nv_bfloat16 __float2bfloat16_rn(float f) { return __float2bfloat16(f); }

inline float __bfloat162float(__nv_bfloat16 b)
{
    const uint32_t u = static_cast<uint32_t>(b.__x) << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
