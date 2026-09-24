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
#include <ostream>
#include <vector>
#include <initializer_list>

#include "../cuda/common/boundary/types.cuh"   // BoundaryDtype

// Maximum rank a Buf can describe.  The boundary layer's buffers are rank
// 3..5; ``last_two`` is 7 ({storage_nvar, 2, B, 1, nz[, ny], nx}) and was
// kept a torch::Tensor in the pilot because the drivers ``select(...).copy_``
// on it.  Those copies now go through the cudautils.h async helpers, so the
// only thing that stood between last_two and a Buf was this constant: 8 gives
// it a slot and one of headroom.  buf_of() still hard-fails rather than
// truncating if a tensor ever exceeds it.
inline constexpr int BUF_MAX_DIMS = 8;   // inline: odr-used from inline functions in other TUs

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
    int64_t sizes_[BUF_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t strides_[BUF_MAX_DIMS] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t numel_ = 0;
    int32_t ndim_ = 0;
    int32_t elem_size_ = 0;
    BoundaryDtype dtype_ = BoundaryDtype::FP32;
    bool defined_ = false;
    // Where the bytes live.  torch answers this with is_cuda()/device(); the
    // memcpy helpers pick H2D / D2H / D2D from it, and 80-odd sites assert on
    // it.  Filled in by buf_of() from the tensor; a default Buf is "not CUDA",
    // which is the safe answer for every `is_cuda() ||` guard in the tree.
    bool is_cuda_ = false;
    // CUDA device ordinal for a device buffer, -1 for host memory.  What
    // torch's device().index() answered; the entry-point device guards read it.
    int32_t device_ = -1;

    // --- torch::Tensor-compatible read API ---------------------------------

    // torch: a Tensor with no TensorImpl.  A default-constructed Buf is the
    // undefined tensor: defined() == false, dim() == 0, numel() == 0,
    // data_ptr() == nullptr.  Note torch THROWS on numel()/dim() of an
    // undefined tensor while a default Buf answers 0 -- every site in this tree
    // guards with ``!defined() || numel() == 0``, so the two agree there, and
    // the Buf answer is the safe one ("nothing to copy", "face is cut").
    bool defined() const { return defined_; }
    // torch: whether the buffer is device memory.  See is_cuda_.
    bool is_cuda() const { return is_cuda_; }
    // torch: device().index() for a CUDA buffer; -1 on the host.
    int device_index() const { return static_cast<int>(device_); }

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

    // --- torch::Tensor-compatible VIEW API ----------------------------------
    // These return a new Buf over the same memory.  Like torch they never copy
    // and never allocate; unlike torch they do not keep the source alive --
    // nothing here does, see OWNERSHIP above.

    // torch: drop dimension d, keeping index i of it (i may be negative and
    // counts from the end).  Pointer moves by i * stride(d) elements; the other
    // sizes and strides are untouched, so a select on a non-leading dim yields
    // a strided view exactly as torch's does -- is_contiguous() will say so.
    Buf select(int64_t d, int64_t i) const
    {
        if (!defined_) return Buf{};   // the undefined tensor stays undefined, no assert
        const int64_t k = normalize_dim(d);
        if (k < 0) return Buf{};
        if (i < 0) i += sizes_[k];
        assert(i >= 0 && i < sizes_[k] && "Buf::select: index out of range");
        Buf r = *this;
        r.data_ = offset_ptr(i * strides_[k]);
        for (int64_t j = k; j + 1 < static_cast<int64_t>(ndim_); ++j) {
            r.sizes_[j] = sizes_[j + 1];
            r.strides_[j] = strides_[j + 1];
        }
        r.sizes_[ndim_ - 1] = 0; r.strides_[ndim_ - 1] = 0;
        r.ndim_ = ndim_ - 1;
        r.numel_ = (sizes_[k] == 0) ? 0 : numel_ / sizes_[k];
        return r;
    }

    // torch: keep dimension d but restrict it to [start, start + len).  Pointer
    // moves by start * stride(d); the stride of d is unchanged, so the result
    // is contiguous only when the source was and d is the outermost dim of
    // size != 1 -- again exactly torch's rule, which is_contiguous() applies.
    // torch: drop dimension d, which must have size 1 (the vrz3d history
    // capture squeezes the C axis before the copy).
    Buf squeeze(int64_t d) const
    {
        if (!defined_) return Buf{};
        const int64_t k = normalize_dim(d);
        assert(sizes_[k] == 1 && "Buf::squeeze: dimension is not of size 1");
        Buf r = *this;
        for (int64_t i = k; i + 1 < static_cast<int64_t>(ndim_); ++i) {
            r.sizes_[i] = sizes_[i + 1];
            r.strides_[i] = strides_[i + 1];
        }
        r.sizes_[ndim_ - 1] = 0; r.strides_[ndim_ - 1] = 0;
        r.ndim_ = ndim_ - 1;
        return r;
    }

    Buf narrow(int64_t d, int64_t start, int64_t len) const
    {
        if (!defined_) return Buf{};   // the undefined tensor stays undefined, no assert
        const int64_t k = normalize_dim(d);
        if (k < 0) return Buf{};
        if (start < 0) start += sizes_[k];
        assert(start >= 0 && len >= 0 && start + len <= sizes_[k] && "Buf::narrow: range out of range");
        Buf r = *this;
        r.data_ = offset_ptr(start * strides_[k]);
        r.sizes_[k] = len;
        r.numel_ = (sizes_[k] == 0) ? 0 : (numel_ / sizes_[k]) * len;
        return r;
    }

    // torch: reinterpret a CONTIGUOUS buffer with new sizes whose product is
    // numel(); one entry may be -1 and is inferred.  Strides are recomputed as
    // row-major, which is the only layout a view of a contiguous buffer can
    // have.  torch throws on a non-contiguous source; here that is an assert,
    // and the result of violating it is a Buf that is_contiguous() but points
    // at strided memory -- so the assert is not decoration.
    Buf view(std::initializer_list<int64_t> new_sizes) const
    {
        if (!defined_) return Buf{};   // the undefined tensor stays undefined, no assert
        assert(is_contiguous() && "Buf::view: source must be contiguous");
        Buf r = *this;
        int64_t n = 0, infer = -1, prod = 1;
        for (int64_t sz : new_sizes) {
            assert(n < BUF_MAX_DIMS && "Buf::view: too many dimensions");
            if (sz == -1) { assert(infer < 0 && "Buf::view: only one -1"); infer = n; r.sizes_[n++] = 0; }
            else { r.sizes_[n++] = sz; prod *= sz; }
        }
        if (infer >= 0) r.sizes_[infer] = (prod == 0) ? 0 : numel_ / prod;
        for (int64_t j = n; j < BUF_MAX_DIMS; ++j) { r.sizes_[j] = 0; r.strides_[j] = 0; }
        int64_t st = 1;
        for (int64_t j = n - 1; j >= 0; --j) { r.strides_[j] = st; st *= r.sizes_[j]; }
        assert((infer >= 0 || prod == numel_) && "Buf::view: sizes do not multiply to numel");
        r.ndim_ = static_cast<int32_t>(n);
        return r;
    }

    // torch: sizes() as a lightweight span.  Enough for the ways this tree uses
    // it -- indexing, size(), iteration, passing to a shape-taking helper --
    // without pulling in c10::ArrayRef.
    struct Span {
        const int64_t* p; int64_t n;
        int64_t size() const { return n; }
        int64_t operator[](int64_t i) const { return p[i]; }
        const int64_t* begin() const { return p; }
        const int64_t* end() const { return p + n; }
        const int64_t* data() const { return p; }
        // torch: sizes().vec()
        std::vector<int64_t> vec() const { return std::vector<int64_t>(p, p + n); }
    };
    Span sizes() const { return Span{sizes_, static_cast<int64_t>(ndim_)}; }
    Span strides() const { return Span{strides_, static_cast<int64_t>(ndim_)}; }

    // Byte-exact pointer arithmetic for the views above: offsets are in
    // ELEMENTS, and the element width is what buf_of() copied from the tensor.
    void* offset_ptr(int64_t elements) const
    {
        return data_ ? static_cast<char*>(data_) + elements * elem_size_ : nullptr;
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

// The Buf twin of ``tensor.device().index()``; the torch twin lives in
// cuda/common/buf_torch.h, so an entry-point guard spells the same thing for
// either type: ``sweep::DeviceGuard g(device_index_of(models[0]))``.
inline int device_index_of(const Buf& b) { return b.device_index(); }

// torch: IntArrayRef == IntArrayRef / == vector (the layout checks).
inline bool operator==(const Buf::Span& a, const Buf::Span& b)
{
    if (a.size() != b.size()) return false;
    for (int64_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}
inline bool operator!=(const Buf::Span& a, const Buf::Span& b) { return !(a == b); }
inline bool operator==(const Buf::Span& a, const std::vector<int64_t>& b)
{
    return a == Buf::Span{b.data(), static_cast<int64_t>(b.size())};
}
inline bool operator==(const std::vector<int64_t>& a, const Buf::Span& b) { return b == a; }
inline bool operator!=(const Buf::Span& a, const std::vector<int64_t>& b) { return !(a == b); }

// sizes() in an error message, printed as c10 printed an IntArrayRef: [a, b].
inline std::ostream& operator<<(std::ostream& o, const Buf::Span& s)
{
    o << "[";
    for (int64_t i = 0; i < s.size(); ++i) {
        if (i) o << ", ";
        o << s[i];
    }
    return o << "]";
}
