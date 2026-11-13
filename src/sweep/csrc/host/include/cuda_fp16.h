#pragma once
// host/include/cuda_fp16.h -- the 16-bit storage type of the boundary saver's
// fp16 mode, as the host core sees it.  Compute never happens in half: the
// kernels convert at the storage boundary (boundarysaver.cu bndry_xfer), so
// only the two conversions the tree uses are provided.  Both are
// round-to-nearest-even, which is what __float2half (== __float2half_rn) does
// on the device, so an fp16 strip saved by the host core holds the same bits
// the CUDA core would store for the same float.
#include "cuda_runtime.h"
#include <cstdint>
#include <cstring>

struct __half {
    unsigned short __x;   // the raw IEEE 754 binary16 bits, as in CUDA's __half
};

inline __half __float2half(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int32_t exp = static_cast<int32_t>((u >> 23) & 0xFFu) - 127 + 15;   // rebias to binary16
    uint32_t mant = u & 0x7FFFFFu;
    uint16_t h;
    if (((u >> 23) & 0xFFu) == 0xFFu) {                 // inf / nan
        h = static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    } else if (exp >= 0x1F) {                          // overflow -> inf
        h = static_cast<uint16_t>(sign | 0x7C00u);
    } else if (exp <= 0) {                             // subnormal or zero
        if (exp < -10) {
            h = static_cast<uint16_t>(sign);           // rounds to zero
        } else {
            mant |= 0x800000u;                         // implicit leading one
            const int shift = 14 - exp;                // 24-bit mantissa -> 10-bit subnormal field
            uint32_t half_mant = mant >> shift;
            const uint32_t rem = mant & ((1u << shift) - 1u);
            const uint32_t halfway = 1u << (shift - 1);
            if (rem > halfway || (rem == halfway && (half_mant & 1u)))
                ++half_mant;                           // may carry into the exponent: still correct
            h = static_cast<uint16_t>(sign | half_mant);
        }
    } else {
        uint32_t half_mant = mant >> 13;
        const uint32_t rem = mant & 0x1FFFu;
        uint32_t bits = (static_cast<uint32_t>(exp) << 10) | half_mant;
        if (rem > 0x1000u || (rem == 0x1000u && (bits & 1u)))
            ++bits;                                    // carry into the exponent is the correct rounding
        h = static_cast<uint16_t>(sign | bits);
    }
    return __half{h};
}

inline __half __float2half_rn(float f) { return __float2half(f); }

inline float __half2float(__half h)
{
    const uint32_t x = h.__x;
    const uint32_t sign = (x & 0x8000u) << 16;
    const uint32_t exp = (x >> 10) & 0x1Fu;
    uint32_t mant = x & 0x3FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            u = sign;                                  // +-0
        } else {                                       // subnormal: normalise
            int e = -1;
            do { ++e; mant <<= 1; } while ((mant & 0x400u) == 0);
            mant &= 0x3FFu;
            u = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        u = sign | 0x7F800000u | (mant << 13);         // inf / nan
    } else {
        u = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
