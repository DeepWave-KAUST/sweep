// 3-D acoustic forward driver: the shared skeleton in common/eq_driver.cuh
// instantiated with this equation's traits (driver_traits.cuh).
#include "acoustic3d.h"
#include "driver_traits.cuh"

namespace acoustic3d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::generic_forward<Driver>(in);
}

ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::generic_forward_core<Driver>(in);
}

ForwardRunnerCorePtr forward_runner_core(const ForwardInputCore& in)
{
    return std::make_shared<eqdrv::GenericForwardRunner<Driver>>(in);
}

ForwardRunnerPtr forward_runner(const ForwardInput& in)
{
    return std::make_shared<TorchForwardRunner<eqdrv::GenericForwardRunner<Driver>>>(in);
}

} // namespace acoustic3d
