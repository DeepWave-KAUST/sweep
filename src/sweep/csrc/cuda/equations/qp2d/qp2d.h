#pragma once
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"

// Pure-qP VTI (qp_vti2d) and TTI (qp_tti2d), 2-D: forward,
// full-storage backward, boundary-saving backward.  backward_ckpt refuses.

namespace qp_vti2d {
ForwardOutputCore forward_core(const ForwardInputCore& in);
BackwardOutputCore backward_core(const BackwardInputCore& in);
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);
} // namespace qp_vti2d

namespace qp_tti2d {
ForwardOutputCore forward_core(const ForwardInputCore& in);
BackwardOutputCore backward_core(const BackwardInputCore& in);
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);
} // namespace qp_tti2d

// Born (LSRTM) pair: qp_vti_lsrtm2d / qp_tti_lsrtm2d (born_traits.cuh).
namespace qp_vti_lsrtm2d {
ForwardOutputCore forward_core(const ForwardInputCore& in);
BackwardOutputCore backward_core(const BackwardInputCore& in);
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);
} // namespace qp_vti_lsrtm2d

namespace qp_tti_lsrtm2d {
ForwardOutputCore forward_core(const ForwardInputCore& in);
BackwardOutputCore backward_core(const BackwardInputCore& in);
BackwardOutputCore backward_bs_core(const BackwardInputCore& in);
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in);
} // namespace qp_tti_lsrtm2d
