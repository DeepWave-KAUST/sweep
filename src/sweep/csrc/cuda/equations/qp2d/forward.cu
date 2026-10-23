// qp_vti2d / qp_tti2d forward: the shared skeleton in common/eq_driver.cuh
// instantiated with the qp2d traits (driver_traits.cuh).
#include "qp2d.h"
#include "driver_traits.cuh"

namespace qp_vti2d {
ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::generic_forward_core<qp2d::Driver<false>>(in);
}
} // namespace qp_vti2d

namespace qp_tti2d {
ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::generic_forward_core<qp2d::Driver<true>>(in);
}
} // namespace qp_tti2d
