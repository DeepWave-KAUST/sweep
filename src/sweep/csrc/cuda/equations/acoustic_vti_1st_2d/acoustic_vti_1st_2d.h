#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"
#include "../../core/input_core.h"
#include "../../core/outputs.h"
#include "../../core/runner.h"

namespace acoustic_vti_1st_2d {

ForwardOutput forward(const ForwardInput& in);
ForwardOutputCore forward_core(const ForwardInputCore& in);

// Backward implementations are stubs until the dedicated CUDA backward
// pass for the Duveneck 2008 acoustic VTI system lands.  They throw a
// SWEEP_CHECK so accidental adjoint use surfaces immediately.
BackwardOutput backward(const BackwardInput& in);
BackwardOutputCore backward_core(const BackwardInputCore& in);
BackwardOutput backward_bs(const BackwardInput& in);
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);
BackwardOutput backward_ckpt(const BackwardInput& in);
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);
BackwardOutput backward_recursive_ckpt(const BackwardInput& in);
BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in);

}  // namespace acoustic_vti_1st_2d
