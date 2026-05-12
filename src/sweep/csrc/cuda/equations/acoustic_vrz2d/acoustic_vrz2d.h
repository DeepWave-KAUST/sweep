#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"

namespace acoustic_vrz2d {

ForwardOutput forward(const ForwardInput& in);
BackwardOutput backward(const BackwardInput& in);
BackwardOutput backward_bs(const BackwardInput& in);
BackwardOutput backward_ckpt(const BackwardInput& in);
BackwardOutput backward_recursive_ckpt(const BackwardInput& in);

} // namespace acoustic_vrz2d
