// Torch-free proof of the view arithmetic, against hand-computed offsets.
// Buf is a POD the adapter fills in; here it is filled in by hand for a
// contiguous {2, 3, 4} float buffer and a 7-D last_two-shaped one.
#include <cstdio>
#include <cstdlib>
#include "core/buf.h"   // compiled with -I<csrc>; no torch anywhere
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static Buf contiguous(void* p, std::initializer_list<int64_t> sz, int64_t es = 4) {
    Buf b; b.defined_ = true; b.elem_size_ = (int32_t)es; b.data_ = p; int n = 0; int64_t numel = 1;
    for (int64_t s : sz) { b.sizes_[n++] = s; numel *= s; }
    b.ndim_ = n; b.numel_ = numel; int64_t st = 1;
    for (int j = n - 1; j >= 0; --j) { b.strides_[j] = st; st *= b.sizes_[j]; }
    return b;
}
int main() {
    float base[24]; for (int i = 0; i < 24; ++i) base[i] = (float)i;
    Buf b = contiguous(base, {2, 3, 4});
    CHECK(b.is_contiguous() && b.numel() == 24 && b.stride(0) == 12 && b.stride(1) == 4 && b.stride(2) == 1);
    // select on the leading dim: contiguous sub-block at offset 12 elements
    Buf s0 = b.select(0, 1);
    CHECK(s0.dim() == 2 && s0.size(0) == 3 && s0.size(1) == 4 && s0.numel() == 12);
    CHECK(s0.data_ptr<float>() == base + 12 && s0.is_contiguous());
    CHECK(*s0.data_ptr<float>() == 12.0f);
    // select on a middle dim: strided (stride(0) stays 12 while size(1) is 4)
    Buf s1 = b.select(1, 2);
    CHECK(s1.dim() == 2 && s1.size(0) == 2 && s1.size(1) == 4 && s1.stride(0) == 12 && s1.stride(1) == 1);
    CHECK(s1.data_ptr<float>() == base + 8 && !s1.is_contiguous());
    // negative index counts from the end
    CHECK(b.select(2, -1).data_ptr<float>() == base + 3);
    // narrow keeps the dim
    Buf n1 = b.narrow(1, 1, 2);
    CHECK(n1.dim() == 3 && n1.size(1) == 2 && n1.numel() == 16 && n1.data_ptr<float>() == base + 4 && !n1.is_contiguous());
    Buf n0 = b.narrow(0, 1, 1);
    CHECK(n0.numel() == 12 && n0.data_ptr<float>() == base + 12 && n0.is_contiguous());
    // view with -1 on a contiguous source: same pointer, row-major strides
    Buf v = b.view({-1, 4});
    CHECK(v.dim() == 2 && v.size(0) == 6 && v.size(1) == 4 && v.stride(0) == 4 && v.data_ptr() == base);
    Buf flat = b.view({-1});
    CHECK(flat.dim() == 1 && flat.size(0) == 24 && flat.stride(0) == 1);
    // real_alias-style chain: flatten, take the first half, reshape
    Buf half = flat.narrow(0, 0, 12).view({3, 4});
    CHECK(half.numel() == 12 && half.is_contiguous() && half.data_ptr() == base && half.size(1) == 4);
    // sizes() span
    int64_t prod = 1; for (int64_t x : b.sizes()) prod *= x; CHECK(prod == 24 && b.sizes()[2] == 4);
    // the 7-D last_two shape now fits: {1, 2, B=3, 1, nz=5, ny=6, nx=7}
    static float big[1 * 2 * 3 * 1 * 5 * 6 * 7];
    Buf lt = contiguous(big, {1, 2, 3, 1, 5, 6, 7});
    CHECK(lt.dim() == 7 && lt.numel() == 1260 && lt.is_contiguous());
    Buf k1 = lt.select(1, 1);                // the "which of the two" select, dim 0 has size 1
    CHECK(k1.dim() == 6 && k1.numel() == 630 && k1.data_ptr<float>() == big + 630 && k1.is_contiguous());
    // is_cuda travels through views; a default Buf is host
    Buf dev = contiguous(base, {2, 3, 4}); dev.is_cuda_ = true;
    CHECK(dev.select(0, 1).is_cuda() && dev.narrow(1, 0, 2).is_cuda() && dev.view({-1}).is_cuda() && !Buf{}.is_cuda());
    // undefined stays undefined through a view
    Buf u; CHECK(!u.select(0, 0).defined() && u.select(0, 0).numel() == 0);
    std::printf("  %s (%d failures)\n", fails ? "RED" : "GREEN", fails);
    return fails ? 1 : 0;
}
