#pragma once
// core/buflist.h -- the torch-free twin of std::vector<torch::Tensor>: a span
// of Bufs somebody else owns (the adapter builds them per call, in an array
// that outlives the call). Same non-owning rule as Buf. The read API is the
// subset of std::vector the drivers use: size(), empty(), operator[], begin/end.
#include <cstdint>
#include <type_traits>
#include <vector>
#include <string>
#include "buf.h"
struct BufList {
    const Buf* p = nullptr;
    int64_t n = 0;
    int64_t size() const { return n; }
    bool empty() const { return n == 0; }
    const Buf& operator[](int64_t i) const { return p[i]; }
    const Buf* begin() const { return p; }
    const Buf* end() const { return p + n; }
    // A struct bind that wants a std::vector<Buf> takes the span through this
    // (a copy of n descriptors, nothing else).
    operator std::vector<Buf>() const { return std::vector<Buf>(p, p + n); }
};
// Small spans for the few vector<int>/vector<float>/vector<string> fields.
struct IntSpan   { const int*   p = nullptr; int64_t n = 0; int64_t size() const { return n; } int   operator[](int64_t i) const { return p[i]; } const int*   begin() const { return p; } const int*   end() const { return p + n; } };
struct FloatSpan { const float* p = nullptr; int64_t n = 0; int64_t size() const { return n; } float operator[](int64_t i) const { return p[i]; } const float* begin() const { return p; } const float* end() const { return p + n; } };
struct CStrList  {
    // the std::vector<std::string> a consumer that predates the span wants
    std::vector<std::string> vec() const { std::vector<std::string> v; for (int64_t i = 0; i < n; ++i) v.emplace_back(p[i]); return v; }
 const char* const* p = nullptr; int64_t n = 0; int64_t size() const { return n; } const char* operator[](int64_t i) const { return p[i]; } };

static_assert(std::is_standard_layout<BufList>::value, "BufList crosses the C boundary");
static_assert(std::is_standard_layout<IntSpan>::value, "IntSpan crosses the C boundary");
static_assert(std::is_standard_layout<FloatSpan>::value, "FloatSpan crosses the C boundary");
static_assert(std::is_standard_layout<CStrList>::value, "CStrList crosses the C boundary");
