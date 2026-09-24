#pragma once
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../../core/runner.h"

namespace das2d {

ForwardOutputCore forward_core(const ForwardInputCore& in);

BackwardOutputCore backward_core(const BackwardInputCore& in);

BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in);

}
