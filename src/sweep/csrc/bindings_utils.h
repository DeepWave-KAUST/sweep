#pragma once

#include <tuple>
#include "./common/wavetypes.h"

template <typename Func>
auto wrap_forward(Func f)
{
    return [f](const ForwardInput& in) {
        auto out = f(in);
        return std::make_tuple(
            out.wavefield,
            out.boundaries,
            out.last_two,
            out.record
        );
    };
}

template <typename Func>
auto wrap_backward(Func f)
{
    return [f](const BackwardInput& in) {

        auto out = f(in);

        return std::make_tuple(
            out.checkpoints,
            out.grads
        );
    };
}