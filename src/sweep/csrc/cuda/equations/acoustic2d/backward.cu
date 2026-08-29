// 2-D acoustic backward drivers (full / boundary-saving / chunk ckpt /
// recursive ckpt): the shared skeleton in common/eq_driver.cuh instantiated
// with this equation's traits (driver_traits.cuh).
#include "acoustic2d.h"
#include "driver_traits.cuh"

namespace acoustic2d {

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::generic_backward<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::generic_backward_bs<Driver>(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return eqdrv::generic_backward_ckpt<Driver>(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return eqdrv::generic_backward_recursive_ckpt<Driver>(in);
}

} // namespace acoustic2d
