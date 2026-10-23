// qp_vti2d / qp_tti2d backward (full storage / boundary saving): the shared
// skeleton in common/eq_driver.cuh instantiated with the qp2d traits.
#include "qp2d.h"
#include "driver_traits.cuh"

namespace {
BackwardOutputCore no_ckpt(const char* name)
{
    SWEEP_CHECK(false, name, " has no checkpoint backward: the adjoint of reverse step it "
                "needs the forward state of step it + 1, which the chunk replay does not "
                "keep. Use memory=Full() or memory=BoundarySaving().");
    return {};
}
} // namespace

namespace qp_vti2d {
BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_core<qp2d::Driver<false>>(in);
}
BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_bs_core<qp2d::Driver<false>>(in);
}
BackwardOutputCore backward_ckpt_core(const BackwardInputCore&)
{
    return no_ckpt("qp_vti2d");
}
} // namespace qp_vti2d

namespace qp_tti2d {
BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_core<qp2d::Driver<true>>(in);
}
BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_bs_core<qp2d::Driver<true>>(in);
}
BackwardOutputCore backward_ckpt_core(const BackwardInputCore&)
{
    return no_ckpt("qp_tti2d");
}
} // namespace qp_tti2d
