// 2-D velocity-reflectivity (EVR) elastic backward drivers (full /
// boundary-saving / chunk ckpt / recursive ckpt): the staggered-family
// skeleton in common/sg_driver.cuh instantiated with this equation's traits.
#include "elastic_vr2d.h"
#include "driver_traits.cuh"

namespace elastic_vr2d {

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

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_recursive_ckpt_core<Driver>(in);
}

BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in)
{
    return std::make_shared<eqdrv::SgBackwardBsRunner<Driver>>(in);
}

} // namespace elastic_vr2d
