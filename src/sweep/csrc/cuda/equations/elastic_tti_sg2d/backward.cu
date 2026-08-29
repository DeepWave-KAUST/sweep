// 2-D staggered-grid elastic TTI backward drivers (full / boundary-saving /
// chunk ckpt): the staggered-family skeleton in common/sg_driver.cuh
// instantiated with this equation's traits.  No recursive-checkpoint driver:
// the equation does not implement it (the forward refuses
// use_recursive_checkpoint), so that template is never instantiated.
#include "elastic_tti_sg2d.h"
#include "driver_traits.cuh"

namespace elastic_tti_sg2d {

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_bs<Driver>(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_ckpt<Driver>(in);
}

} // namespace elastic_tti_sg2d
