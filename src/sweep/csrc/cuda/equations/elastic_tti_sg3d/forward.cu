// 3-D staggered-grid elastic TTI forward driver: the staggered-family
// skeleton in common/sg_driver.cuh instantiated with this equation's traits.
#include "elastic_tti_sg3d.h"
#include "driver_traits.cuh"

namespace elastic_tti_sg3d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::sg_generic_forward<Driver>(in);
}

} // namespace elastic_tti_sg3d
