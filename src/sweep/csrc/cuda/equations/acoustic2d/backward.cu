// 2-D acoustic backward drivers (full / boundary-saving / chunk ckpt /
// recursive ckpt): the shared skeleton in common/eq_driver.cuh instantiated
// with this equation's traits (driver_traits.cuh).
#include "acoustic2d.h"
#include "driver_traits.cuh"

namespace acoustic2d {

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_core<Driver>(in);
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_bs_core<Driver>(in);
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_ckpt_core<Driver>(in);
}

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_recursive_ckpt_core<Driver>(in);
}

BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in)
{
    return std::make_shared<eqdrv::GenericBackwardBsRunner<Driver>>(in);
}

} // namespace acoustic2d
