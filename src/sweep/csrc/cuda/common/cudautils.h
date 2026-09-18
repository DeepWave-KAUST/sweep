#pragma once

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

inline void validate_cuda_copy_tensors(
    const torch::Tensor& dst,
    const torch::Tensor& src,
    const char* label
)
{
    TORCH_CHECK(dst.defined() && src.defined(), label, " expects defined tensors.");
    TORCH_CHECK(dst.scalar_type() == src.scalar_type(), label, " expects matching dtypes.");
    TORCH_CHECK(dst.numel() == src.numel(), label, " expects matching numel.");
    TORCH_CHECK(dst.is_contiguous(), label, " destination must be contiguous.");
    TORCH_CHECK(src.is_contiguous(), label, " source must be contiguous.");
}

inline void copy_tensor_device_to_device_async(const torch::Tensor& dst, const torch::Tensor& src)
{
    validate_cuda_copy_tensors(dst, src, "CUDA device copy");
    TORCH_CHECK(dst.is_cuda() && src.is_cuda(), "CUDA device copy expects CUDA tensors.");
    TORCH_CHECK(dst.device() == src.device(), "CUDA device copy expects tensors on the same device.");

    if (dst.numel() == 0) return;

    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        static_cast<size_t>(dst.nbytes()),
        cudaMemcpyDeviceToDevice,
        at::cuda::getCurrentCUDAStream()
    ));
}

inline void copy_tensor_cuda_async(const torch::Tensor& dst, const torch::Tensor& src)
{
    validate_cuda_copy_tensors(dst, src, "CUDA async copy");
    TORCH_CHECK(dst.is_cuda() || src.is_cuda(), "CUDA async copy expects at least one CUDA tensor.");

    cudaMemcpyKind kind;
    if (dst.is_cuda() && src.is_cuda()) {
        TORCH_CHECK(dst.device() == src.device(), "CUDA async copy expects device tensors on the same device.");
        kind = cudaMemcpyDeviceToDevice;
    } else if (dst.is_cuda()) {
        TORCH_CHECK(src.device().is_cpu(), "CUDA async copy host source must be a CPU tensor.");
        kind = cudaMemcpyHostToDevice;
    } else {
        TORCH_CHECK(dst.device().is_cpu(), "CUDA async copy host destination must be a CPU tensor.");
        kind = cudaMemcpyDeviceToHost;
    }

    if (dst.numel() == 0) return;

    C10_CUDA_CHECK(cudaMemcpyAsync(
        dst.data_ptr(),
        src.data_ptr(),
        static_cast<size_t>(dst.nbytes()),
        kind,
        at::cuda::getCurrentCUDAStream()
    ));
}

inline void zero_tensor_device_async(const torch::Tensor& dst)
{
    TORCH_CHECK(dst.defined(), "CUDA zero expects a defined tensor.");
    TORCH_CHECK(dst.is_cuda(), "CUDA zero expects a CUDA tensor.");
    TORCH_CHECK(dst.is_contiguous(), "CUDA zero destination must be contiguous.");

    if (dst.numel() == 0) return;

    C10_CUDA_CHECK(cudaMemsetAsync(
        dst.data_ptr(),
        0,
        static_cast<size_t>(dst.nbytes()),
        at::cuda::getCurrentCUDAStream()
    ));
}

#define SWEEP_CUDA_SYNC_CHECK(label)                                                \
    do {                                                                            \
        cudaError_t _launch_err = cudaGetLastError();                               \
        TORCH_CHECK(                                                                 \
            _launch_err == cudaSuccess,                                             \
            label, " launch failed: ", cudaGetErrorString(_launch_err)              \
        );                                                                          \
        cudaError_t _sync_err = cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()); \
        TORCH_CHECK(                                                                 \
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
        : compute_stream(at::cuda::getCurrentCUDAStream()),
          enabled(enable)
    {
        if (!enabled) return;
        // cudaStreamCreate makes a BLOCKING stream, which is implicitly
        // synchronised against the legacy default stream. compute_stream is
        // at::cuda::getCurrentCUDAStream(), and PyTorch's default IS the legacy
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
// grads_out) when one was bound, else a fresh
// zero tensor of the same geometry. This is what lets an equation stop
// allocating its per-backward scratch in C++: the propagator owns the pool's
// lifetime and zeroes it before every gradient-bearing forward, which is the
// same state a fresh zeros_like starts in. The two checks are what make a
// mismatch LOUD -- consumers take data_ptr<float>() and index by the model's
// geometry, so a pool tensor of the wrong shape or dtype would read as garbage
// rather than fail.
inline bool pool_slot_bound(const std::vector<torch::Tensor>& pool, int idx)
{
    return static_cast<int>(pool.size()) > idx && pool[idx].defined() && pool[idx].numel() > 0;
}

inline const torch::Tensor& pool_slot_checked(const std::vector<torch::Tensor>& pool, int idx,
                                              const torch::Tensor& like, const char* what)
{
    TORCH_CHECK(pool[idx].sizes() == like.sizes(),
                what, "[", idx, "] has shape ", pool[idx].sizes(),
                " but the expected geometry is ", like.sizes());
    TORCH_CHECK(pool[idx].scalar_type() == torch::kFloat,
                what, "[", idx, "] must be float32, got ", pool[idx].scalar_type());
    return pool[idx];
}

inline torch::Tensor pool_or_zeros(const std::vector<torch::Tensor>& pool, int idx,
                                   const torch::Tensor& like,
                                   const char* what = "adjoint_workspace")
{
    if (pool_slot_bound(pool, idx))
        return pool_slot_checked(pool, idx, like, what);
    return torch::zeros_like(like);
}

// Same binding, uninitialised fallback: for a slot the driver writes in full
// before it reads (the derived model coefficients), so neither side pays a
// memset for it.
inline torch::Tensor pool_or_empty(const std::vector<torch::Tensor>& pool, int idx,
                                   const torch::Tensor& like, const char* what)
{
    if (pool_slot_bound(pool, idx))
        return pool_slot_checked(pool, idx, like, what);
    return torch::empty_like(like);
}

// The same for a pool whose slots are not model-shaped (checkpoint_replay):
// the driver states the shape it lays out and a bound slot must match it.
inline torch::Tensor pool_or_zeros(const std::vector<torch::Tensor>& pool, int idx,
                                   std::vector<int64_t> shape, const torch::TensorOptions& options,
                                   const char* what)
{
    if (static_cast<int>(pool.size()) > idx && pool[idx].defined() && pool[idx].numel() > 0) {
        TORCH_CHECK(pool[idx].sizes().vec() == shape,
                    what, "[", idx, "] has shape ", pool[idx].sizes(),
                    " but the driver's layout is ", shape);
        TORCH_CHECK(pool[idx].scalar_type() == torch::kFloat && pool[idx].is_contiguous(),
                    what, "[", idx, "] must be a contiguous float32 tensor");
        return pool[idx];
    }
    return torch::zeros(shape, options);
}

// A single Python-bound output buffer (u_allt_out, record_out) when it was
// bound, else a fresh zero tensor of the driver's shape. The shape check is
// what makes a Python/driver disagreement loud instead of a silent overrun.
inline torch::Tensor bound_or_zeros(const torch::Tensor& bound, std::vector<int64_t> shape,
                                    const torch::TensorOptions& options, const char* what)
{
    if (!bound.defined())
        return torch::zeros(shape, options);
    TORCH_CHECK(bound.sizes().vec() == shape, what, " has shape ", bound.sizes(),
                " but the driver's layout is ", shape);
    TORCH_CHECK(bound.scalar_type() == torch::kFloat && bound.is_contiguous(),
                what, " must be a contiguous float32 tensor");
    return bound;
}
