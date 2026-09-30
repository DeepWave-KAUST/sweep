#pragma once

#include <cuda_runtime.h>
#include "../../core/check.h"
#include "../../core/device.h"
#include "../../core/buflist.h"   // BufList, for the pool/set twins
#include "../../core/check.h"
#include "../../core/device.h"

// ---- Buf overloads of the async copy/zero helpers (unit 1 of step 2) --------
// The torch::Tensor overloads above stay as they are; these are the same three
// operations on descriptors, so a call site whose operands became Bufs keeps
// its spelling. Direction comes from the descriptor's is_cuda bit, exactly as
// the tensor overload reads it off the tensor. A Buf is a non-owning view, so
// these do not touch lifetime -- whoever built the Buf keeps the memory alive
// until the stream has consumed it, which is the same rule the tensor
// overloads rely on through torch's own refcount.
inline void validate_cuda_copy_bufs(const Buf& dst, const Buf& src, const char* label)
{
    SWEEP_CHECK(dst.defined() && src.defined(), label, " expects defined buffers.");
    SWEEP_CHECK(dst.element_size() == src.element_size(), label, " expects matching element widths.");
    SWEEP_CHECK(dst.dtype() == src.dtype(), label, " expects matching storage dtypes.");
    SWEEP_CHECK(dst.numel() == src.numel(), label, " expects matching numel.");
    SWEEP_CHECK(dst.is_contiguous(), label, " destination must be contiguous.");
    SWEEP_CHECK(src.is_contiguous(), label, " source must be contiguous.");
}

inline void copy_tensor_cuda_async(const Buf& dst, const Buf& src)
{
    validate_cuda_copy_bufs(dst, src, "CUDA async copy");
    SWEEP_CHECK(dst.is_cuda() || src.is_cuda(), "CUDA async copy expects at least one CUDA buffer.");
    const cudaMemcpyKind kind = (dst.is_cuda() && src.is_cuda()) ? cudaMemcpyDeviceToDevice
                              : dst.is_cuda()                    ? cudaMemcpyHostToDevice
                                                                 : cudaMemcpyDeviceToHost;
    if (dst.numel() == 0) return;
    SWEEP_CUDA_CHECK(cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(),
                                   static_cast<size_t>(dst.numel() * dst.element_size()),
                                   kind, sweep::current_stream()));
}

inline void copy_tensor_device_to_device_async(const Buf& dst, const Buf& src)
{
    SWEEP_CHECK(dst.is_cuda() && src.is_cuda(), "CUDA device copy expects CUDA buffers.");
    copy_tensor_cuda_async(dst, src);
}

inline void zero_tensor_device_async(const Buf& dst)
{
    SWEEP_CHECK(dst.defined(), "CUDA zero expects a defined buffer.");
    SWEEP_CHECK(dst.is_cuda(), "CUDA zero expects a CUDA buffer.");
    SWEEP_CHECK(dst.is_contiguous(), "CUDA zero destination must be contiguous.");
    if (dst.numel() == 0) return;
    SWEEP_CUDA_CHECK(cudaMemsetAsync(dst.data_ptr(), 0,
                                   static_cast<size_t>(dst.numel() * dst.element_size()),
                                   sweep::current_stream()));
}

// ---- BufList / Buf twins of the pool and set helpers (step 2, unit 2a) ------
// Same names, same checks, same semantics, on descriptors.  Neither side
// allocates: an unbound slot is refused, which is what the propagator-
// allocates-everything policy promises and test_binding_is_mandatory pins at
// the driver level.
inline bool pool_slot_bound(const BufList& pool, int idx)
{
    return pool.size() > idx && pool[idx].defined() && pool[idx].numel() > 0;
}

inline bool same_shape(const Buf& a, const Buf& b)
{
    if (a.dim() != b.dim()) return false;
    for (int64_t d = 0; d < a.dim(); ++d) if (a.size(d) != b.size(d)) return false;
    return true;
}

// pool_required, the tree's own name for "bound or refuse": the torch overload
// above checks sizes() == like.sizes(), float32 and is_cuda; this one checks the
// same three on descriptors.
inline const Buf& pool_required(const BufList& pool, int idx, const Buf& like, const char* what)
{
    SWEEP_CHECK(pool_slot_bound(pool, idx), what, " slot ", idx,
                " is not bound; the compiled drivers no longer allocate it.");
    const Buf& b = pool[idx];
    SWEEP_CHECK(same_shape(b, like), what, " slot ", idx, " has the wrong shape for its reference buffer");
    SWEEP_CHECK(b.dtype() == BoundaryDtype::FP32, what, " slot ", idx, " must be float32");
    SWEEP_CHECK(b.is_cuda(), what, " slot ", idx, " must be a CUDA buffer");
    return b;
}

// The driver-shaped slot with the shape as a vector (the traits build it).
inline const Buf& pool_required(const BufList& pool, int idx, const std::vector<int64_t>& shape,
                                const char* what)
{
    SWEEP_CHECK(pool_slot_bound(pool, idx),
                what, "[", idx, "] must be bound by the propagator; this driver "
                "has no fallback allocation for it (", pool.size(), " slots bound)");
    const Buf& b = pool[idx];
    SWEEP_CHECK(b.sizes() == shape, what, "[", idx, "] has shape ", b.sizes(),
                " but the driver's layout is ", shape);
    SWEEP_CHECK(b.dtype() == BoundaryDtype::FP32 && b.is_contiguous(),
                what, "[", idx, "] must be a contiguous float32 buffer");
    return b;
}

// A slot checked against a geometry (no "must be bound" wording: the set it
// comes from was counted by the caller).
inline const Buf& pool_slot_checked(const BufList& pool, int idx, const Buf& like, const char* what)
{
    const Buf& b = pool[idx];
    SWEEP_CHECK(b.sizes() == like.sizes(),
                what, "[", idx, "] has shape ", b.sizes(), " but the expected geometry is ", like.sizes());
    SWEEP_CHECK(b.dtype() == BoundaryDtype::FP32,
                what, "[", idx, "] must be float32");
    SWEEP_CHECK(b.is_cuda(),
                what, "[", idx, "] must live on the GPU (a host buffer here reads as an "
                "illegal address inside the kernels)");
    return b;
}

// A single propagator-bound output buffer (record_out, u_allt_out, adcig_out).
inline const Buf& bound_required(const Buf& bound, const std::vector<int64_t>& shape, const char* what)
{
    SWEEP_CHECK(bound.defined(),
                what, " must be bound by the propagator; this driver has no "
                "fallback allocation for it");
    SWEEP_CHECK(bound.sizes() == shape, what, " has shape ", bound.sizes(),
                " but the driver's layout is ", shape);
    SWEEP_CHECK(bound.dtype() == BoundaryDtype::FP32 && bound.is_contiguous(),
                what, " must be a contiguous float32 buffer");
    return bound;
}

inline const Buf& pool_required(const BufList& pool, int idx, std::initializer_list<int64_t> shape,
                                const char* what)
{
    SWEEP_CHECK(pool_slot_bound(pool, idx), what, " slot ", idx,
                " is not bound; the compiled drivers no longer allocate it.");
    const Buf& b = pool[idx];
    bool ok = (b.dim() == static_cast<int64_t>(shape.size()));
    int64_t d = 0; for (int64_t s : shape) { ok = ok && (d < b.dim()) && (b.size(d) == s); ++d; }
    SWEEP_CHECK(ok, what, " slot ", idx, " has the wrong shape");
    SWEEP_CHECK(b.dtype() == BoundaryDtype::FP32 && b.is_contiguous(),
                what, " slot ", idx, " must be a contiguous float32 buffer");
    return b;
}

// A window of n descriptors starting at set k of a flat list -- a span, not a
// copy; the caller's BufList outlives it because the arena does.
inline BufList wavefield_set(const BufList& list, int k, int n, const char* what)
{
    SWEEP_CHECK(list.size() >= static_cast<int64_t>(k + 1) * n,
                what, " expects at least ", (k + 1) * n, " bound wavefield buffers (",
                k + 1, " sets of ", n, "), got ", list.size());
    BufList set{list.p + static_cast<int64_t>(k) * n, n};
    for (int i = 0; i < n; ++i)
        SWEEP_CHECK(set[i].defined() && set[i].is_cuda() && set[i].dtype() == BoundaryDtype::FP32
                        && set[i].is_contiguous(),
                    what, ": set ", k, " slot ", i, " must be a contiguous float32 CUDA buffer");
    return set;
}

inline float* ptr_or_null(const Buf& b)
{
    return b.defined() ? b.data_ptr<float>() : nullptr;
}

inline size_t buf_bytes(const Buf& b)
{
    return b.defined() ? static_cast<size_t>(b.numel() * b.element_size()) : 0;
}

// The pool helpers on an owned std::vector<Buf> (a trait's slot list); after
// every BufList overload so the delegations resolve to them exactly.
inline BufList as_list(const std::vector<Buf>& v) { return BufList{v.data(), static_cast<int64_t>(v.size())}; }
inline const Buf& pool_required(const std::vector<Buf>& pool, int idx, const Buf& like, const char* what)
{ return pool_required(as_list(pool), idx, like, what); }
inline const Buf& pool_required(const std::vector<Buf>& pool, int idx, const std::vector<int64_t>& shape, const char* what)
{ return pool_required(as_list(pool), idx, shape, what); }
inline const Buf& pool_required(const std::vector<Buf>& pool, int idx, std::initializer_list<int64_t> shape, const char* what)
{ return pool_required(as_list(pool), idx, shape, what); }
inline const Buf& pool_slot_checked(const std::vector<Buf>& pool, int idx, const Buf& like, const char* what)
{ return pool_slot_checked(as_list(pool), idx, like, what); }

#define SWEEP_CUDA_SYNC_CHECK(label)                                                \
    do {                                                                            \
        cudaError_t _launch_err = cudaGetLastError();                               \
        SWEEP_CHECK(                                                                 \
            _launch_err == cudaSuccess,                                             \
            label, " launch failed: ", cudaGetErrorString(_launch_err)              \
        );                                                                          \
        cudaError_t _sync_err = cudaStreamSynchronize(sweep::current_stream()); \
        SWEEP_CHECK(                                                                 \
            _sync_err == cudaSuccess,                                               \
            label, " execution failed: ", cudaGetErrorString(_sync_err)             \
        );                                                                          \
    } while (0)

struct AsyncCopyContext {
    cudaStream_t compute_stream;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t ready_event = nullptr;
    bool enabled = false;

    explicit AsyncCopyContext(bool enable)
        : compute_stream(sweep::current_stream()),
          enabled(enable)
    {
        if (!enabled) return;
        // cudaStreamCreate makes a BLOCKING stream, which is implicitly
        // synchronised against the legacy default stream. compute_stream is
        // sweep::current_stream(), and PyTorch's default IS the legacy
        // default stream -- so every operation on copy_stream serialised against
        // all compute and this "async copy stream" was never concurrent.
        // The real dependencies are stated explicitly by the BoundaryRuntime,
        // which records and waits on its own per-slot events around each staged
        // copy, so dropping the implicit sync does not affect correctness
        // (checked by the bit-exact gates).
        // This is also why non-DD gains only +22% while DD gains 15x: non-DD runs
        // the whole time loop inside one extension call and hits few
        // serialisation points, whereas DD calls in once per time step and hits
        // this implicit sync on every one of them.
        cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
    }

    ~AsyncCopyContext()
    {
        if (!enabled) return;
        cudaEventDestroy(ready_event);
        cudaStreamDestroy(copy_stream);
    }
};

// A single Python-bound output buffer (u_allt_out, record_out).  The shape
// check is what makes a Python/driver disagreement loud instead of a silent
// overrun.
// A Python-bound wavefield LIST (``forward_wavefields`` handed to a
// boundary-saving backward as its reconstruction state, sized by
// ``cuda_layout.bs_reconstruction_nvar``): true when one was bound, after the
// count and every slot's geometry/dtype were checked; false when the caller
// bound nothing, in which case the driver allocates as it always did.
inline bool wavefields_bound(const BufList& list, int n, const Buf& like, const char* what)
{
    if (list.empty()) return false;
    SWEEP_CHECK(static_cast<int>(list.size()) == n,
                what, " expects ", n, " bound wavefield tensors, got ", list.size());
    for (int i = 0; i < n; ++i)
        pool_slot_checked(list, i, like, what);
    return true;
}
inline void wavefields_required(const BufList& list, int n, const Buf& like, const char* what)
{
    SWEEP_CHECK(!list.empty(),
                what, " must be bound by the propagator; this driver has no "
                "fallback allocation for it");
    wavefields_bound(list, n, like, what);
}
