#pragma once
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../../core/runner.h"

namespace elastic2d {

ForwardOutputCore forward_core(const ForwardInputCore& in);

ForwardRunnerCorePtr forward_runner_core(const ForwardInputCore& in);
BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in);

BackwardOutputCore backward_core(const BackwardInputCore& in);

BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in);

// APM (Cao & Chen 2018) parameter-modified path for irregular topography.
// Expects ``in.models`` ordered as
//   [vp, vs, rho, lame_lambda, lame_mu, lame_lambda_2mu,
//    lam_eff, mu_eff, mu_xz_node, rho_x_eff, rho_z_eff]
// and ``in.topo_category`` set to the runtime-padded ``int32`` category
// tensor (INTERIOR=0, AIR=1, H=2, VL=3, VR=4, OC=5, IC=6).
ForwardOutputCore apm_forward_core(const ForwardInputCore& in);

}
