// 3-D DAS-Mu backward drivers (full / boundary-saving / chunk ckpt /
// recursive ckpt): the staggered-family skeleton in common/sg_driver.cuh
// instantiated with this equation's traits.
#include "das_mu3d.h"
#include "driver_traits.cuh"

namespace das_mu3d {

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

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_recursive_ckpt<Driver>(in);
}

} // namespace das_mu3d
