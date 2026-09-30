// 2-D staggered-grid elastic TTI forward driver: the staggered-family
// skeleton in common/sg_driver.cuh instantiated with this equation's traits.
#include "elastic_tti_sg2d.h"
#include "driver_traits.cuh"

namespace elastic_tti_sg2d {

ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::sg_generic_forward_core<Driver>(in);
}

ForwardRunnerCorePtr forward_runner_core(const ForwardInputCore& in)
{
    return std::make_shared<eqdrv::SgForwardRunner<Driver>>(in);
}

} // namespace elastic_tti_sg2d
