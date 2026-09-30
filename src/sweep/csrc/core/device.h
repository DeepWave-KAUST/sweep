#pragma once
// core/device.h -- the core's device and stream plumbing, torch-free.
//
// The compiled drivers used c10::cuda::CUDAGuard at every entry and
// at::cuda::getCurrentCUDAStream() at every launch, memcpy and event.  Both
// answer a question torch keeps per thread: "which device, which stream".
// Here that state is the core's own:
//
//   * DeviceGuard      -- cudaSetDevice for a scope, restored on exit (what
//                         c10::cuda::CUDAGuard does with the device part).
//   * current_stream() -- the stream every launch in the core goes on.  It is
//                         thread-local; the binding sets it for the duration
//                         of each call from torch's current stream for the
//                         input's device (bindings_utils.h), so a caller on a
//                         torch side stream is honoured exactly as before.  A
//                         thread that never set one gets the legacy default
//                         stream (nullptr), which is also torch's answer for a
//                         thread that never changed streams.
//
// SWEEP_CUDA_CHECK / SWEEP_KERNEL_LAUNCH_CHECK replace C10_CUDA_CHECK and
// C10_CUDA_KERNEL_LAUNCH_CHECK: check, and throw sweep::Error with the CUDA
// error string.
#include <cuda_runtime.h>

#include "check.h"

#define SWEEP_CUDA_CHECK(expr)                                                        \
    do {                                                                              \
        const cudaError_t _sweep_err = (expr);                                        \
        SWEEP_CHECK(_sweep_err == cudaSuccess, "CUDA error: ",                        \
                    cudaGetErrorString(_sweep_err));                                  \
    } while (0)

#define SWEEP_KERNEL_LAUNCH_CHECK() SWEEP_CUDA_CHECK(cudaGetLastError())

namespace sweep {

inline cudaStream_t& current_stream_slot()
{
    thread_local cudaStream_t s = nullptr;
    return s;
}

inline cudaStream_t current_stream() { return current_stream_slot(); }

// cudaGetDevice, checked (c10::cuda::current_device()).
inline int current_device()
{
    int d = 0;
    SWEEP_CUDA_CHECK(cudaGetDevice(&d));
    return d;
}

// Sets current_stream() for a scope.  The binding opens one per entry.
struct StreamScope {
    cudaStream_t prev;
    explicit StreamScope(cudaStream_t s) : prev(current_stream_slot()) { current_stream_slot() = s; }
    ~StreamScope() { current_stream_slot() = prev; }
    StreamScope(const StreamScope&) = delete;
    StreamScope& operator=(const StreamScope&) = delete;
};

// Makes ``dev`` the current CUDA device for a scope; a negative index (a host
// buffer) is a no-op.  The previous device is restored on exit, without
// throwing (destructor).
struct DeviceGuard {
    int prev = -1;
    explicit DeviceGuard(int dev)
    {
        if (dev < 0) return;
        SWEEP_CUDA_CHECK(cudaGetDevice(&prev));
        if (dev != prev) SWEEP_CUDA_CHECK(cudaSetDevice(dev));
        else prev = -1;
    }
    ~DeviceGuard()
    {
        if (prev >= 0) (void)cudaSetDevice(prev);
    }
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};

}  // namespace sweep
