#pragma once
// The shim's per-call plumbing: torch's current stream becomes the core's
// (sweep_set_stream, core/capi.h) for the duration of every entry, so a caller
// running under ``torch.cuda.stream(s)`` gets its launches on ``s``; the output
// structs become the tuples Python reads.
#include <tuple>
#include <vector>

// c10's stream header, not ATen/cuda/CUDAContext.h: that one drags in
// cublas_v2.h -> cuda_fp16.h -> nv/target, which torch's pip wheels do not
// ship (they bring cuda_runtime_api.h and little else), so the shim would only
// compile with a full toolkit.  Same stream object either way.
#include <c10/cuda/CUDAStream.h>

#include "shared/wavetypes.h"
#include "core/capi.h"

inline cudaStream_t torch_stream_for_device(int device_index)
{
    if (device_index < 0) return nullptr;
    return c10::cuda::getCurrentCUDAStream(device_index).stream();
}

inline cudaStream_t torch_stream_for(const std::vector<torch::Tensor>& models)
{
    if (models.empty() || !models[0].is_cuda()) return nullptr;
    return torch_stream_for_device(static_cast<int>(models[0].device().index()));
}

// Sets the core's current stream for a scope (the core's own thread-local,
// reached through the C API so the shim never touches the core's inlines).
struct CoreStreamScope {
    void* prev;
    explicit CoreStreamScope(cudaStream_t s) : prev(sweep_get_stream()) { sweep_set_stream(static_cast<void*>(s)); }
    ~CoreStreamScope() { sweep_set_stream(prev); }
    CoreStreamScope(const CoreStreamScope&) = delete;
    CoreStreamScope& operator=(const CoreStreamScope&) = delete;
};

template <typename Func>
auto with_stream_forward(Func f)
{
    return [f](const ForwardInput& in) {
        CoreStreamScope stream(torch_stream_for(in.models));
        return f(in);
    };
}

template <typename Func>
auto with_stream_backward(Func f)
{
    return [f](const BackwardInput& in) {
        CoreStreamScope stream(torch_stream_for(in.models));
        return f(in);
    };
}

template <typename Func>
auto wrap_forward(Func f)
{
    return [f](const ForwardInput& in) {
        CoreStreamScope stream(torch_stream_for(in.models));
        auto out = f(in);
        return std::make_tuple(out.wavefield, out.last_two, out.record);
    };
}

template <typename Func>
auto wrap_backward(Func f)
{
    return [f](const BackwardInput& in) {
        CoreStreamScope stream(torch_stream_for(in.models));
        auto out = f(in);
        return std::make_tuple(out.checkpoints, out.grads, out.source_illumination,
                               out.receiver_illumination, out.adcig);
    };
}
