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
        cudaStreamCreate(&copy_stream);
        cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming);
    }

    void record_compute_ready() const
    {
        if (!enabled) return;
        cudaEventRecord(ready_event, compute_stream);
    }

    void wait_for_compute() const
    {
        if (!enabled) return;
        cudaStreamWaitEvent(copy_stream, ready_event, 0);
    }

    void record_copy_ready() const
    {
        if (!enabled) return;
        cudaEventRecord(ready_event, copy_stream);
    }

    void wait_for_copy() const
    {
        if (!enabled) return;
        cudaStreamWaitEvent(compute_stream, ready_event, 0);
    }

    void synchronize_copy() const
    {
        if (!enabled) return;
        cudaStreamSynchronize(copy_stream);
    }

    ~AsyncCopyContext()
    {
        if (!enabled) return;
        cudaEventDestroy(ready_event);
        cudaStreamDestroy(copy_stream);
    }
};
