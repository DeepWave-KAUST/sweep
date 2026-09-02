#pragma once

#include "../../common/wavetypes.h"

namespace elastic_tti_sg3d {

ForwardOutput forward(const ForwardInput& in);

// Persistent stepped runners (prologue once, run() per segment) —
// see IForwardRunner/IBackwardRunner in shared/wavetypes.h.
ForwardRunnerPtr forward_runner(const ForwardInput& in);
BackwardRunnerPtr backward_bs_runner(const BackwardInput& in);

BackwardOutput backward(const BackwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

BackwardOutput backward_ckpt(const BackwardInput& in);

} // namespace elastic_tti_sg3d
