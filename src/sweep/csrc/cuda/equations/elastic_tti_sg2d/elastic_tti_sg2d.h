#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"
#include "../../core/input_core.h"
#include "../../core/outputs.h"
#include "../../core/runner.h"

namespace elastic_tti_sg2d {

ForwardOutput forward(const ForwardInput& in);
ForwardOutputCore forward_core(const ForwardInputCore& in);

// Persistent stepped runners (prologue once, run() per segment) —
// see IForwardRunner/IBackwardRunner in shared/wavetypes.h.
ForwardRunnerPtr forward_runner(const ForwardInput& in);
ForwardRunnerCorePtr forward_runner_core(const ForwardInputCore& in);
BackwardRunnerPtr backward_bs_runner(const BackwardInput& in);
BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in);

BackwardOutput backward(const BackwardInput& in);
BackwardOutputCore backward_core(const BackwardInputCore& in);

BackwardOutput backward_bs(const BackwardInput& in);
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutput backward_ckpt(const BackwardInput& in);
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

}
