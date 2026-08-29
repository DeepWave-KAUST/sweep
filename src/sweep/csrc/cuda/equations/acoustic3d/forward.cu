// 3-D acoustic forward driver: the shared skeleton in common/eq_driver.cuh
// instantiated with this equation's traits (driver_traits.cuh).
#include "acoustic3d.h"
#include "driver_traits.cuh"

namespace acoustic3d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::generic_forward<Driver>(in);
}

} // namespace acoustic3d
