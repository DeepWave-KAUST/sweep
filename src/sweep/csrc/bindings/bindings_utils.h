#pragma once

#include <tuple>
#include <vector>

#include <ATen/cuda/CUDAContext.h>

#include "shared/wavetypes.h"
#include "core/device.h"

// The compiled core launches on sweep::current_stream(), which is thread-local
// and set by whoever enters the core.  torch's current stream is also thread-
// local, per device; every entry from Python sets the core's stream from
// torch's for the input's device, for the duration of the call, so a caller
// running under ``torch.cuda.stream(s)`` gets its launches on ``s`` exactly as
// the old at::cuda::getCurrentCUDAStream() calls gave it.

inline cudaStream_t torch_stream_for_device(int device_index)
{
    if (device_index < 0) return nullptr;
    return at::cuda::getCurrentCUDAStream(device_index).stream();
}

inline cudaStream_t torch_stream_for(const std::vector<torch::Tensor>& models)
{
    if (models.empty() || !models[0].is_cuda()) return nullptr;
    return torch_stream_for_device(static_cast<int>(models[0].device().index()));
}

template <typename Func>
auto with_stream_forward(Func f)
{
    return [f](const ForwardInput& in) {
        sweep::StreamScope stream(torch_stream_for(in.models));
        return f(in);
    };
}

template <typename Func>
auto with_stream_backward(Func f)
{
    return [f](const BackwardInput& in) {
        sweep::StreamScope stream(torch_stream_for(in.models));
        return f(in);
    };
}

template <typename Func>
auto wrap_forward(Func f)
{
    return [f](const ForwardInput& in) {
        sweep::StreamScope stream(torch_stream_for(in.models));
        auto out = f(in);
        return std::make_tuple(
            out.wavefield,
            out.last_two,
            out.record
        );
    };
}

template <typename Func>
auto wrap_backward(Func f)
{
    return [f](const BackwardInput& in) {
        sweep::StreamScope stream(torch_stream_for(in.models));
        auto out = f(in);

        return std::make_tuple(
            out.checkpoints,
            out.grads,
            out.source_illumination,
            out.receiver_illumination,
            out.adcig
        );
    };
}
