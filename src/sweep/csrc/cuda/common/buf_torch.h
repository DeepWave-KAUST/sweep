#pragma once
// ---------------------------------------------------------------------------
// cuda/common/buf_torch.h -- the ONLY place a torch::Tensor becomes a Buf.
//
// core/buf.h has no torch include and must never gain one.  This header is the
// adapter on the torch side of that line: it takes a tensor the Python
// propagator allocated and copies out the handful of facts the compiled code
// actually reads (pointer, sizes, strides, element width, storage-dtype tag).
//
// The conversion is a COPY, not a reference: nothing here keeps the tensor
// alive, so the caller must hold the tensor for at least as long as the Buf it
// made from it (EffectiveBoundarySaver keeps them in ``torch_refs_``).
// ---------------------------------------------------------------------------

#include <torch/extension.h>

#include "../../core/buf.h"

// Map a boundary-buffer tensor's dtype to the storage-dtype enum.  Python
// allocates the buffers in the requested precision, so the tensor itself is
// the source of truth (uint8 == the INT8 path's quantized main buffer).
// (Moved here from boundary/saver.cuh unchanged so buf_of() can use it; every
// caller keeps its spelling.)
static inline BoundaryDtype boundary_dtype_from_tensor(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case torch::kUInt8:    return BoundaryDtype::INT8;
        case torch::kHalf:     return BoundaryDtype::FP16;
        case torch::kBFloat16: return BoundaryDtype::BF16;
        default:               return BoundaryDtype::FP32;
    }
}

// Same question asked of an already-converted descriptor: the tag was computed
// by the overload above at conversion time, so this is the identity.  It exists
// so call sites that asked a saver member for its storage dtype keep compiling
// once that member is a Buf.
static inline BoundaryDtype boundary_dtype_from_tensor(const Buf& b) {
    return b.dtype();
}

// Describe ``t`` without owning it.  An UNDEFINED tensor becomes a default Buf
// (defined() == false, numel() == 0, data_ptr() == nullptr), which is what
// every ``!defined() || numel() == 0`` guard in the boundary layer expects.
//
// Sizes and strides are copied verbatim -- in particular the nonzero stride
// torch leaves on a 0-size dimension, which a DD cut face has and the saver's
// staging check reads (see Buf::stride).
static inline Buf buf_of(const torch::Tensor& t) {
    Buf b;
    if (!t.defined())
        return b;

    const int64_t nd = t.dim();
    TORCH_CHECK(nd >= 0 && nd <= BUF_MAX_DIMS,
                "buf_of: tensor has ", nd, " dimensions but Buf carries at most ",
                BUF_MAX_DIMS, "; raise BUF_MAX_DIMS in csrc/core/buf.h.");

    b.defined_ = true;
    b.ndim_ = static_cast<int32_t>(nd);
    b.numel_ = t.numel();
    b.elem_size_ = static_cast<int32_t>(t.element_size());
    b.dtype_ = boundary_dtype_from_tensor(t);
    // The tag drives the byte arithmetic downstream, so it must match the
    // tensor's real width. Checked HERE, not in Buf: Buf's guards are asserts,
    // and the two ways sweep ships disagree about NDEBUG (the JIT keeps asserts,
    // a wheel build strips them) -- this TORCH_CHECK throws in both.
    TORCH_CHECK(b.elem_size_ == buf_dtype_element_size(b.dtype_),
                "buf_of: a ", t.scalar_type(), " tensor is ", b.elem_size_,
                " bytes wide but the boundary storage tag implies ",
                buf_dtype_element_size(b.dtype_),
                "; the boundary layer stores float32, float16, bfloat16 or uint8.");
    // Safe for an empty tensor: torch's storage_initialized() is true when
    // numel == 0, and the boundary layer's copy primitives already skip a face
    // whose numel is 0 (they did so precisely because this pointer is null).
    b.data_ = t.data_ptr();
    for (int64_t i = 0; i < nd; ++i) {
        b.sizes_[i] = t.size(i);
        b.strides_[i] = t.stride(i);
    }
    return b;
}
