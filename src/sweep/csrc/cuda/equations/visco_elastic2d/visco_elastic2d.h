#pragma once
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"

namespace visco_elastic2d {

// Models: the prepared set [lam_U, mu_U, rho, lam_U + 2 mu_U, P_0..P_{L-1},
// M_0..M_{L-1}] (ViscoElastic.prepare_models), L = 1..4; eq_aux: the host
// float32 weights [a_0..a_{L-1}, c_0..c_{L-1}] (ViscoElastic.c_eq_aux).
ForwardOutputCore forward_core(const ForwardInputCore& in);

BackwardOutputCore backward_core(const BackwardInputCore& in);

// Refuses: boundary saving has no stable reverse reconstruction here.
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in);

}
