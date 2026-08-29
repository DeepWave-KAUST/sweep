// 3-D DAS-Mu forward driver: the staggered-family skeleton in
// common/sg_driver.cuh instantiated with this equation's traits.
#include "das_mu3d.h"
#include "driver_traits.cuh"

namespace das_mu3d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::sg_generic_forward<Driver>(in);
}

} // namespace das_mu3d
