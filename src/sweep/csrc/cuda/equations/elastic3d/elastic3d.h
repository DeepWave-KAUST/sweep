#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"

namespace elastic3d {

ForwardOutput forward(const ForwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

BackwardOutput backward_ckpt(const BackwardInput& in);

BackwardOutput backward_recursive_ckpt(const BackwardInput& in);

BackwardOutput backward(const BackwardInput& in);

// APM (Cao & Chen 2018, 3-D) — irregular topography with
// parameter-modified moduli.  Forward only in this commit;
// apm_backward / apm_backward_bs are stubs until Phase 3D.
ForwardOutput  apm_forward(const ForwardInput& in);
BackwardOutput apm_backward(const BackwardInput& in);
BackwardOutput apm_backward_bs(const BackwardInput& in);

}
