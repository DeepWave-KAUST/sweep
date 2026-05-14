#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"

namespace elastic_tti_sg2d {

ForwardOutput forward(const ForwardInput& in);

BackwardOutput backward(const BackwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

BackwardOutput backward_ckpt(const BackwardInput& in);

}
