#pragma once

#include "../../common/wavetypes.h"
#include "../../core/input_core.h"
#include "../../core/outputs.h"
#include "../../core/runner.h"

namespace elastic_tti_2nd2d {

ForwardOutput forward(const ForwardInput& in);
ForwardOutputCore forward_core(const ForwardInputCore& in);

BackwardOutput backward(const BackwardInput& in);
BackwardOutputCore backward_core(const BackwardInputCore& in);

BackwardOutput backward_bs(const BackwardInput& in);
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutput backward_ckpt(const BackwardInput& in);
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

} // namespace elastic_tti_2nd2d
