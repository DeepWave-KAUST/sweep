// 2-D staggered-grid elastic TTI forward driver: the staggered-family
// skeleton in common/sg_driver.cuh instantiated with this equation's traits.
#include "elastic_tti_sg2d.h"
#include "driver_traits.cuh"

namespace elastic_tti_sg2d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::sg_generic_forward<Driver>(in);
}

ForwardRunnerPtr forward_runner(const ForwardInput& in)
{
    return std::make_shared<eqdrv::SgForwardRunner<Driver>>(in);
}

} // namespace elastic_tti_sg2d
