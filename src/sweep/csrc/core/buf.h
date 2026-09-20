#pragma once
// ---------------------------------------------------------------------------
// core/buf.h -- a torch-free description of a buffer somebody else owns.
//
// WHY.  sweep's compiled backend is shipped as source and JIT-compiled against
// the user's torch (see src/sweep/_jit.py) because the C++ uses torch's
// ABI-unstable C++ API everywhere.  Most of that use is not "call torch", it is
// "ask a tensor for a pointer / a size / a stride".  ``Buf`` answers exactly
// those questions with plain data, so the code that only reads them can be
// compiled once, without libtorch, and a thin torch-facing adapter
// (cuda/common/buf_torch.h) fills a ``Buf`` in at the one place the propagator
// hands its buffers over.
//
// OWNERSHIP.  A ``Buf`` NEVER owns memory.  It is a non-owning view of a buffer
// the Python propagator allocated and keeps alive; it has no allocation, no
// free and no refcount.  Whoever builds a ``Buf`` is responsible for the
// backing tensor outliving it.
//
// METHOD NAMES are deliberately torch::Tensor's, with torch's semantics, so
// existing call sites recompile unchanged.  Every deviation is documented on
// the method itself.
//
// DEPENDENCY NOTE.  The only sweep header this pulls in is the storage-dtype
// tag ``BoundaryDtype`` (cuda/common/boundary/types.cuh), which is itself
// torch-free.  That is an upward include from core/ into cuda/common/ and
// should be resolved -- by moving the dtype tag down into core/ -- when more of
// the core is carved out; it is left alone here so this pilot changes no
// existing declaration.
// ---------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "../cuda/common/boundary/types.cuh"   // BoundaryDtype

// Maximum rank a Buf can describe.  Every buffer the boundary layer binds is
// rank 3..5 (persistent face / staging ring: 5, per-block int8 scale: 3 in 3-D,
// 4 in 2-D -- see Layout in src/sweep/memory/shape.py and
// PropagatorC._int8_scale_shapes).  The one 7-D buffer in the saver
// (``last_two``, {nvar, 2, B, 1, nz[, ny], nx}) is deliberately NOT a Buf: the
// equation drivers use it as a real tensor (``select(...).copy_(...)``).
// buf_of() hard-fails rather than truncating if this is ever too small.
inline constexpr int BUF_MAX_DIMS = 6;   // inline: odr-used from inline functions in other TUs

// Byte width implied by a storage-dtype tag.  Kept next to the tag so a
// descriptor's element_size() can be validated against its dtype().
inline int64_t buf_dtype_element_size(BoundaryDtype dt)
{
    switch (dt) {
        case BoundaryDtype::INT8: return 1;
        case BoundaryDtype::FP16: return 2;
        case BoundaryDtype::BF16: return 2;
        case BoundaryDtype::FP32: return 4;
    }
    return 4;
}

// A strided view of device or host memory.  Aggregate, trivially copyable,
// cheap to pass by value; the fields are public so an adapter can fill it in
// without a constructor that would drag torch into this header.
struct Buf {
    void* data_ = nullptr;
    int64_t sizes_[BUF_MAX_DIMS] = {0, 0, 0, 0, 0, 0};
    int64_t strides_[BUF_MAX_DIMS] = {0, 0, 0, 0, 0, 0};
    int64_t numel_ = 0;
    int32_t ndim_ = 0;
    int32_t elem_size_ = 0;
    BoundaryDtype dtype_ = BoundaryDtype::FP32;
    bool defined_ = false;

    // --- torch::Tensor-compatible read API ---------------------------------

    // torch: a Tensor with no TensorImpl.  A default-constructed Buf is the
    // undefined tensor: defined() == false, dim() == 0, numel() == 0,
    // data_ptr() == nullptr.  Note torch THROWS on numel()/dim() of an
    // undefined tensor while a default Buf answers 0 -- every site in this tree
    // guards with ``!defined() || numel() == 0``, so the two agree there, and
    // the Buf answer is the safe one ("nothing to copy", "face is cut").
    bool defined() const { return defined_; }

    // torch: number of dimensions.
    int64_t dim() const { return static_cast<int64_t>(ndim_); }

    // torch: product of the sizes (0 if any dimension is 0).  COPIED from the
    // tensor at conversion time, not recomputed.
    int64_t numel() const { return numel_; }

    // torch: byte width of one element.  COPIED from the tensor, so byte
    // arithmetic is exact even for a dtype outside the four the storage tag
    // knows about.  For those four, element_size() == buf_dtype_element_size(
    // dtype()).
    int64_t element_size() const { return elem_size_; }

    // The storage-dtype TAG (BoundaryDtype), not torch::ScalarType: the tree's
    // own torch-free classification of the payload (uint8 -> INT8, half ->
    // FP16, bfloat16 -> BF16, anything else -> FP32), exactly what
    // boundary_dtype_from_tensor() computed before.  scalar_type() is the same
    // value under torch's other spelling, so a call site that only asked the
    // tensor for its dtype keeps its method name; only the CONSTANT it compares
    // against changes (torch::kHalf -> BoundaryDtype::FP16).
    BoundaryDtype dtype() const { return dtype_; }
    BoundaryDtype scalar_type() const { return dtype_; }

    // torch: size of dimension d; a negative d counts from the end
    // (d += dim()).  Out of range is a hard error in torch; here it asserts and
    // returns 0 in a release build.
    int64_t size(int64_t d) const
    {
        const int64_t i = normalize_dim(d);
        return (i < 0) ? 0 : sizes_[i];
    }

    // torch: stride of dimension d, in ELEMENTS; a negative d counts from the
    // end.  THE STRIDE IS COPIED FROM THE TENSOR, NEVER DERIVED FROM THE SIZES:
    // torch clamps the stride of a 0-size dimension to a nonzero value, and
    // this tree depends on it -- a DD cut face is allocated with its last axis
    // 0 (Layout._cut_face_shape) and the saver still compares a staging
    // numel() against such a face's stride(0) (Layout.staging_shapes, and the
    // tangent_pad guard in EffectiveBoundarySaver::allocate_int8_staging).
    // Recomputing would give 0 there and silently change both.
    int64_t stride(int64_t d) const
    {
        const int64_t i = normalize_dim(d);
        return (i < 0) ? 0 : strides_[i];
    }

    // torch: untyped data pointer.  nullptr for a default (undefined) Buf; for
    // a defined but empty buffer it is whatever the owner's allocation carries
    // (usually nullptr), the same value the tensor would have handed back.
    void* data_ptr() const { return data_; }

    // torch: typed data pointer.  torch TORCH_CHECKs that T matches the
    // tensor's scalar type on every call; here that check is an assert (so it
    // is compiled out with -DNDEBUG) on the element WIDTH, which is all the
    // pointer arithmetic downstream depends on.
    template <typename T>
    T* data_ptr() const
    {
        assert(sizeof(T) == static_cast<size_t>(elem_size_) &&
               "Buf::data_ptr<T>(): T is not the element type of this buffer");
        return static_cast<T*>(data_);
    }

    // torch's compute_contiguous(), verbatim: an empty buffer is contiguous;
    // otherwise every dimension of size != 1 must carry the stride implied by
    // the dimensions after it.  (torch THROWS for an undefined tensor; a
    // default Buf has numel() == 0 and answers true.)
    bool is_contiguous() const
    {
        if (numel_ == 0)
            return true;
        int64_t expected = 1;
        for (int64_t d = static_cast<int64_t>(ndim_) - 1; d >= 0; --d) {
            const int64_t s = sizes_[d];
            if (s != 1) {
                if (strides_[d] != expected)
                    return false;
                expected *= s;
            }
        }
        return true;
    }

    // torch's dimension wrapping, shared by size()/stride().  Returns -1 for an
    // index torch would have rejected.
    int64_t normalize_dim(int64_t d) const
    {
        const int64_t n = static_cast<int64_t>(ndim_);
        if (d < 0)
            d += n;
        assert(d >= 0 && d < n && "Buf: dimension index out of range");
        return (d >= 0 && d < n) ? d : -1;
    }
};
