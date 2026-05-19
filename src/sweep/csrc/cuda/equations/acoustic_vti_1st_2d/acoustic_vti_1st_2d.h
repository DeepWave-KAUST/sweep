#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"

namespace acoustic_vti_1st_2d {

ForwardOutput forward(const ForwardInput& in);

// Backward implementations are stubs until the dedicated CUDA backward
// pass for the Duveneck 2008 acoustic VTI system lands.  They throw a
// TORCH_CHECK so accidental adjoint use surfaces immediately.
BackwardOutput backward(const BackwardInput& in);
BackwardOutput backward_bs(const BackwardInput& in);
BackwardOutput backward_ckpt(const BackwardInput& in);
BackwardOutput backward_recursive_ckpt(const BackwardInput& in);

}  // namespace acoustic_vti_1st_2d
