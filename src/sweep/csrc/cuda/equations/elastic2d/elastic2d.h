#pragma once
#include <torch/extension.h>
#include "../../common/wavetypes.h"

namespace elastic2d {

ForwardOutput forward(const ForwardInput& in);

BackwardOutput backward(const BackwardInput& in);

BackwardOutput backward_bs(const BackwardInput& in);

BackwardOutput backward_ckpt(const BackwardInput& in);

BackwardOutput backward_recursive_ckpt(const BackwardInput& in);

// APM (Cao & Chen 2018) parameter-modified path for irregular topography.
// Expects ``in.models`` ordered as
//   [vp, vs, rho, lame_lambda, lame_mu, lame_lambda_2mu,
//    lam_eff, mu_eff, mu_xz_node, rho_x_eff, rho_z_eff]
// and ``in.topo_category`` set to the runtime-padded ``int32`` category
// tensor (INTERIOR=0, AIR=1, H=2, VL=3, VR=4, OC=5, IC=6).
ForwardOutput  apm_forward(const ForwardInput& in);
BackwardOutput apm_backward(const BackwardInput& in);
BackwardOutput apm_backward_bs(const BackwardInput& in);

}
