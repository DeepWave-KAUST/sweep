// 2-D velocity-reflectivity (EVR) elastic forward driver: the
// staggered-family skeleton in common/sg_driver.cuh instantiated with this
// equation's traits.
#include "elastic_vr2d.h"
#include "driver_traits.cuh"

namespace elastic_vr2d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::sg_generic_forward<Driver>(in);
}

ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::sg_generic_forward_core<Driver>(in);
}

ForwardRunnerCorePtr forward_runner_core(const ForwardInputCore& in)
{
    return std::make_shared<eqdrv::SgForwardRunner<Driver>>(in);
}

ForwardRunnerPtr forward_runner(const ForwardInput& in)
{
    return std::make_shared<TorchForwardRunner<eqdrv::SgForwardRunner<Driver>>>(in);
}

} // namespace elastic_vr2d
