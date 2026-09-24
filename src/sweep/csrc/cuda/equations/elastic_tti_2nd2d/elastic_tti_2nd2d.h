#pragma once

#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../../core/runner.h"

namespace elastic_tti_2nd2d {

ForwardOutputCore forward_core(const ForwardInputCore& in);

BackwardOutputCore backward_core(const BackwardInputCore& in);

BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

} // namespace elastic_tti_2nd2d
