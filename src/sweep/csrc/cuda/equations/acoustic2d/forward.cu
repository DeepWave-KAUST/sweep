// 2-D acoustic forward driver: the shared skeleton in common/eq_driver.cuh
// instantiated with this equation's traits (driver_traits.cuh).
#include "acoustic2d.h"
#include "driver_traits.cuh"

namespace acoustic2d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::generic_forward<Driver>(in);
}

ForwardRunnerPtr forward_runner(const ForwardInput& in)
{
    return std::make_shared<eqdrv::GenericForwardRunner<Driver>>(in);
}

} // namespace acoustic2d
