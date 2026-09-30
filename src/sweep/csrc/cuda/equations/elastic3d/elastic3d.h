#pragma once
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../../core/runner.h"

namespace elastic3d {

ForwardOutputCore forward_core(const ForwardInputCore& in);

ForwardRunnerCorePtr forward_runner_core(const ForwardInputCore& in);
BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in);

BackwardOutputCore backward_bs_core(const BackwardInputCore& in);

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in);

BackwardOutputCore backward_core(const BackwardInputCore& in);

// APM (Cao & Chen 2018, 3-D) — irregular topography with
// parameter-modified moduli.  Forward only: no compiled APM backward.
ForwardOutputCore apm_forward_core(const ForwardInputCore& in);

}
