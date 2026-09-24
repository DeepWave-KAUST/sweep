// 2-D DAS-Mu backward drivers (full / boundary-saving / chunk ckpt /
// recursive ckpt): the staggered-family skeleton in common/sg_driver.cuh
// instantiated with this equation's traits.
#include "das_mu2d.h"
#include "driver_traits.cuh"

namespace das_mu2d {

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward<Driver>(in);
}

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_core<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_bs<Driver>(in);
}

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_bs_core<Driver>(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_ckpt<Driver>(in);
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_ckpt_core<Driver>(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_recursive_ckpt<Driver>(in);
}

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    return eqdrv::sg_generic_backward_recursive_ckpt_core<Driver>(in);
}

BackwardRunnerCorePtr backward_bs_runner_core(const BackwardInputCore& in)
{
    return std::make_shared<eqdrv::SgBackwardBsRunner<Driver>>(in);
}

BackwardRunnerPtr backward_bs_runner(const BackwardInput& in)
{
    return std::make_shared<TorchBackwardRunner<eqdrv::SgBackwardBsRunner<Driver>>>(in);
}

} // namespace das_mu2d
