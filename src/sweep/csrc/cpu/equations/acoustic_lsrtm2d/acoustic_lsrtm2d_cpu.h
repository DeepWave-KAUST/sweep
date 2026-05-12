#pragma once

#include "../../../shared/wavetypes.h"

namespace sweep_cpu::acoustic_lsrtm2d {

ForwardOutput forward(const ForwardInput& in);
BackwardOutput backward(const BackwardInput& in);
BackwardOutput backward_bs(const BackwardInput& in);
BackwardOutput backward_ckpt(const BackwardInput& in);
BackwardOutput backward_recursive_ckpt(const BackwardInput& in);

} // namespace sweep_cpu::acoustic_lsrtm2d
