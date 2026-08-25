#pragma once

#include <tuple>
#include "shared/wavetypes.h"

template <typename Func>
auto wrap_forward(Func f)
{
    return [f](const ForwardInput& in) {
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

        auto out = f(in);

        return std::make_tuple(
            out.checkpoints,
            out.grads,
            out.source_illumination,
            out.receiver_illumination,
            out.adcig,
            out.grad_split_iii   // RWI term III (acoustic_lsrtm, SWEEP_LSRTM_SPLIT_III=1); else undefined
        );
    };
}

template <typename Func>
auto wrap_rtm(Func f)
{
    return [f](const BackwardInput& in) {
        auto out = f(in);
        return std::make_tuple(
            out.image,
            out.source_illumination,
            out.receiver_illumination,
            out.adcig
        );
    };
}
