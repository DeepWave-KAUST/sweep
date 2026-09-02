#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"

namespace das_mu2d {

ForwardOutput forward(const ForwardInput& in);

// Persistent stepped runners (prologue once, run() per segment) —
// see IForwardRunner/IBackwardRunner in shared/wavetypes.h.
ForwardRunnerPtr forward_runner(const ForwardInput& in);
BackwardRunnerPtr backward_bs_runner(const BackwardInput& in);

BackwardOutput backward(const BackwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

BackwardOutput backward_ckpt(const BackwardInput& in);

BackwardOutput backward_recursive_ckpt(const BackwardInput& in);

} // namespace das_mu2d
