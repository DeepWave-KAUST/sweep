// qp_vti_lsrtm2d / qp_tti_lsrtm2d: the Born pair on the shared skeleton
// (common/eq_driver.cuh) with the BornDriver traits (born_traits.cuh).
#include "qp2d.h"
#include "born_traits.cuh"

namespace {
BackwardOutputCore no_ckpt_born(const char* name)
{
    SWEEP_CHECK(false, name, " has no checkpoint backward. Use memory=Full() or "
                "memory=BoundarySaving().");
    return {};
}
} // namespace

namespace qp_vti_lsrtm2d {
ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::generic_forward_core<qp2d::BornDriver<false>>(in);
}
BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_core<qp2d::BornDriver<false>>(in);
}
BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_bs_core<qp2d::BornDriver<false>>(in);
}
BackwardOutputCore backward_ckpt_core(const BackwardInputCore&)
{
    return no_ckpt_born("qp_vti_lsrtm2d");
}
} // namespace qp_vti_lsrtm2d

namespace qp_tti_lsrtm2d {
ForwardOutputCore forward_core(const ForwardInputCore& in)
{
    return eqdrv::generic_forward_core<qp2d::BornDriver<true>>(in);
}
BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_core<qp2d::BornDriver<true>>(in);
}
BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    return eqdrv::generic_backward_bs_core<qp2d::BornDriver<true>>(in);
}
BackwardOutputCore backward_ckpt_core(const BackwardInputCore&)
{
    return no_ckpt_born("qp_tti_lsrtm2d");
}
} // namespace qp_tti_lsrtm2d
