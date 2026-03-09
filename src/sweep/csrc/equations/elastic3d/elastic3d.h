#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"

namespace elastic3d {

ForwardOutput forward(const ForwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

}