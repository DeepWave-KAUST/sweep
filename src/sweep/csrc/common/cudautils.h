#pragma once

#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

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
