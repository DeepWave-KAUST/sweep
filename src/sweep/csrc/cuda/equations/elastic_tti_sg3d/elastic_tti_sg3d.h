#pragma once

#include "../../common/wavetypes.h"

namespace elastic_tti_sg3d {

ForwardOutput forward(const ForwardInput& in);

BackwardOutput backward(const BackwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

BackwardOutput backward_ckpt(const BackwardInput& in);

} // namespace elastic_tti_sg3d
