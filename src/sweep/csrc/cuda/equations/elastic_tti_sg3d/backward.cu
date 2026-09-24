// 3-D staggered-grid elastic TTI backward drivers (full / boundary-saving /
// chunk ckpt): the staggered-family skeleton in common/sg_driver.cuh
// instantiated with this equation's traits.  No recursive-checkpoint driver:
// the equation does not implement it (the forward refuses
// use_recursive_checkpoint), so that template is never instantiated.
#include "elastic_tti_sg3d.h"
#include "driver_traits.cuh"

namespace elastic_tti_sg3d {

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_core<Driver>(in);
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_bs_core<Driver>(in);
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_ckpt_core<Driver>(in);
}

BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in)
{
    return std::make_shared<eqdrv::SgBackwardBsRunner<Driver>>(in);
}

} // namespace elastic_tti_sg3d
