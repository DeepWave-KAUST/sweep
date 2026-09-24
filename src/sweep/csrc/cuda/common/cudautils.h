#pragma once

#include <cuda_runtime.h>
#include "../../core/check.h"
#include "../../core/device.h"
#include <torch/extension.h>
#include "buf_torch.h"   // Buf + buf_of, for the Buf overloads below
#include "../../core/buflist.h"   // BufList, for the pool/set twins
#include "../../core/check.h"
#include "../../core/device.h"

inline void validate_cuda_copy_tensors(
    const torch::Tensor& dst,
    const torch::Tensor& src,
    const char* label
)
{
    SWEEP_CHECK(dst.defined() && src.defined(), label, " expects defined tensors.");
    SWEEP_CHECK(dst.scalar_type() == src.scalar_type(), label, " expects matching dtypes.");
    SWEEP_CHECK(dst.numel() == src.numel(), label, " expects matching numel.");
    SWEEP_CHECK(dst.is_contiguous(), label, " destination must be contiguous.");
    SWEEP_CHECK(src.is_contiguous(), label, " source must be contiguous.");
}

inline void copy_tensor_device_to_device_async(const torch::Tensor& dst, const torch::Tensor& src)
{
    validate_cuda_copy_tensors(dst, src, "CUDA device copy");
    SWEEP_CHECK(dst.is_cuda() && src.is_cuda(), "CUDA device copy expects CUDA tensors.");
    SWEEP_CHECK(dst.device() == src.device(), "CUDA device copy expects tensors on the same device.");

    if (dst.numel() == 0) return;

    SWEEP_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        static_cast<size_t>(dst.nbytes()),
        cudaMemcpyDeviceToDevice,
        sweep::current_stream()
    ));
}

inline void copy_tensor_cuda_async(const torch::Tensor& dst, const torch::Tensor& src)
{
    validate_cuda_copy_tensors(dst, src, "CUDA async copy");
    SWEEP_CHECK(dst.is_cuda() || src.is_cuda(), "CUDA async copy expects at least one CUDA tensor.");

    cudaMemcpyKind kind;
    if (dst.is_cuda() && src.is_cuda()) {
        SWEEP_CHECK(dst.device() == src.device(), "CUDA async copy expects device tensors on the same device.");
        kind = cudaMemcpyDeviceToDevice;
    } else if (dst.is_cuda()) {
        SWEEP_CHECK(src.device().is_cpu(), "CUDA async copy host source must be a CPU tensor.");
        kind = cudaMemcpyHostToDevice;
    } else {
        SWEEP_CHECK(dst.device().is_cpu(), "CUDA async copy host destination must be a CPU tensor.");
        kind = cudaMemcpyDeviceToHost;
    }

    if (dst.numel() == 0) return;

    SWEEP_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        static_cast<size_t>(dst.nbytes()),
        kind,
        sweep::current_stream()
    ));
}

inline void zero_tensor_device_async(const torch::Tensor& dst)
{
    SWEEP_CHECK(dst.defined(), "CUDA zero expects a defined tensor.");
    SWEEP_CHECK(dst.is_cuda(), "CUDA zero expects a CUDA tensor.");
    SWEEP_CHECK(dst.is_contiguous(), "CUDA zero destination must be contiguous.");

    if (dst.numel() == 0) return;

    SWEEP_CUDA_CHECK(cudaMemsetAsync(
        dst.data_ptr(),
        0,
        static_cast<size_t>(dst.nbytes()),
        sweep::current_stream()
    ));
}

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
// Mixed spellings while the tree converts one file at a time: a torch tensor on
// one side and a Buf on the other should not force a site to spell buf_of().
inline void copy_tensor_cuda_async(const Buf& dst, const torch::Tensor& src) { copy_tensor_cuda_async(dst, buf_of(src)); }
inline void copy_tensor_cuda_async(const torch::Tensor& dst, const Buf& src) { copy_tensor_cuda_async(buf_of(dst), src); }

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

// A tensor from a Python-bound pool (adjoint_workspace, forward_workspace,
// grads_out).  The propagator owns the pool's lifetime and zeroes it before
// every gradient-bearing forward; nothing here allocates.  The checks are
// what make a mismatch LOUD -- consumers take data_ptr<float>() and index by
// the model's geometry, so a pool tensor of the wrong shape or dtype would
// read as garbage rather than fail.
inline bool pool_slot_bound(const std::vector<torch::Tensor>& pool, int idx)
{
    return static_cast<int>(pool.size()) > idx && pool[idx].defined() && pool[idx].numel() > 0;
}

inline const torch::Tensor& pool_slot_checked(const std::vector<torch::Tensor>& pool, int idx,
                                              const torch::Tensor& like, const char* what)
{
    SWEEP_CHECK(pool[idx].sizes() == like.sizes(),
                what, "[", idx, "] has shape ", pool[idx].sizes(),
                " but the expected geometry is ", like.sizes());
    SWEEP_CHECK(pool[idx].scalar_type() == torch::kFloat,
                what, "[", idx, "] must be float32, got ", pool[idx].scalar_type());
    SWEEP_CHECK(pool[idx].is_cuda(),
                what, "[", idx, "] must live on the GPU (a host tensor here reads as an "
                "illegal address inside the kernels), got ", pool[idx].device());
    return pool[idx];
}

// A slot the propagator binds on every call that reaches the site (its
// cuda_layout declares the pool unconditionally for that equation and that
// memory mode).  Geometry/dtype/device checks; an unbound slot is the error,
// so a driver cannot silently keep a shadow buffer alive when a caller forgets
// to bind.
inline torch::Tensor pool_required(const std::vector<torch::Tensor>& pool, int idx,
                                   const torch::Tensor& like, const char* what)
{
    SWEEP_CHECK(pool_slot_bound(pool, idx),
                what, "[", idx, "] must be bound by the propagator; this driver "
                "has no fallback allocation for it (", pool.size(), " slots bound)");
    return pool_slot_checked(pool, idx, like, what);
}

// The same for a pool whose slots are not model-shaped (checkpoint_replay):
// the driver states the shape it lays out and a bound slot must match it.
inline const torch::Tensor& pool_slot_checked(const std::vector<torch::Tensor>& pool, int idx,
                                              const std::vector<int64_t>& shape, const char* what)
{
    SWEEP_CHECK(pool[idx].sizes().vec() == shape,
                what, "[", idx, "] has shape ", pool[idx].sizes(),
                " but the driver's layout is ", shape);
    SWEEP_CHECK(pool[idx].scalar_type() == torch::kFloat && pool[idx].is_contiguous(),
                what, "[", idx, "] must be a contiguous float32 tensor");
    return pool[idx];
}

// The driver-shaped slot (checkpoint_replay): same checks.  ``options`` is
// unused; it leaves the signature with the Buf flip.
inline torch::Tensor pool_required(const std::vector<torch::Tensor>& pool, int idx,
                                   std::vector<int64_t> shape,
                                   const torch::TensorOptions& /*options*/,
                                   const char* what)
{
    SWEEP_CHECK(pool_slot_bound(pool, idx),
                what, "[", idx, "] must be bound by the propagator; this driver "
                "has no fallback allocation for it (", pool.size(), " slots bound)");
    return pool_slot_checked(pool, idx, shape, what);
}

// A single Python-bound output buffer (u_allt_out, record_out).  The shape
// check is what makes a Python/driver disagreement loud instead of a silent
// overrun.
// A Python-bound wavefield LIST (``forward_wavefields`` handed to a
// boundary-saving backward as its reconstruction state, sized by
// ``cuda_layout.bs_reconstruction_nvar``): true when one was bound, after the
// count and every slot's geometry/dtype were checked; false when the caller
// bound nothing, in which case the driver allocates as it always did.
inline bool wavefields_bound(const std::vector<torch::Tensor>& list, int n,
                             const torch::Tensor& like, const char* what)
{
    if (list.empty()) return false;
    SWEEP_CHECK(static_cast<int>(list.size()) == n,
                what, " expects ", n, " bound wavefield tensors, got ", list.size());
    for (int i = 0; i < n; ++i)
        pool_slot_checked(list, i, like, what);
    return true;
}

// REQUIRED variant: the list is bound on every call that reaches the site, so
// an empty list is a broken contract and not "the driver allocates its own".
// Same count and per-slot geometry checks; nothing is returned because there is
// no longer a second case to report.
inline void wavefields_required(const std::vector<torch::Tensor>& list, int n,
                                const torch::Tensor& like, const char* what)
{
    SWEEP_CHECK(!list.empty(),
                what, " must be bound by the propagator; this driver has no "
                "fallback allocation for it");
    wavefields_bound(list, n, like, what);
}

// One SET of a Python-bound wavefield list that holds K sets of n tensors back
// to back (the checkpoint replay state: set 0 = the replay / segment start
// state, sets 1..depth = the bisection scratch states). Count, dtype, device
// and contiguity are checked here; the geometry is the struct's bind() business
// (CPML aux slots are slab-shaped, not model-shaped).
inline std::vector<torch::Tensor> wavefield_set(const std::vector<torch::Tensor>& list, int k, int n,
                                                const char* what)
{
    SWEEP_CHECK(static_cast<int>(list.size()) >= (k + 1) * n,
                what, " expects at least ", (k + 1) * n, " bound wavefield tensors (",
                k + 1, " sets of ", n, "), got ", list.size());
    std::vector<torch::Tensor> set(list.begin() + k * n, list.begin() + (k + 1) * n);
    for (int i = 0; i < n; ++i)
        SWEEP_CHECK(set[i].defined() && set[i].is_cuda() && set[i].scalar_type() == torch::kFloat
                        && set[i].is_contiguous(),
                    what, ": set ", k, " slot ", i, " must be a contiguous float32 CUDA tensor");
    return set;
}

// Device pointer of an optional wavefield member: nullptr when the member was
// left undefined by a partial bind (a reconstruction that carries no CPML
// memory), for kernels that never touch it in that mode.
inline float* ptr_or_null(const torch::Tensor& t)
{
    return t.defined() ? t.data_ptr<float>() : nullptr;
}

inline const torch::Tensor& bound_checked(const torch::Tensor& bound,
                                          const std::vector<int64_t>& shape, const char* what)
{
    SWEEP_CHECK(bound.sizes().vec() == shape, what, " has shape ", bound.sizes(),
                " but the driver's layout is ", shape);
    SWEEP_CHECK(bound.scalar_type() == torch::kFloat && bound.is_contiguous(),
                what, " must be a contiguous float32 tensor");
    return bound;
}

// The undefined tensor that used to select an allocation is the error.
// ``options`` is unused; it leaves the signature with the Buf flip.
inline torch::Tensor bound_required(const torch::Tensor& bound, std::vector<int64_t> shape,
                                    const torch::TensorOptions& /*options*/, const char* what)
{
    SWEEP_CHECK(bound.defined(),
                what, " must be bound by the propagator; this driver has no "
                "fallback allocation for it");
    return bound_checked(bound, shape, what);
}
