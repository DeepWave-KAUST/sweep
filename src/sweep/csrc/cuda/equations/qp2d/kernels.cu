// The one translation unit that instantiates the qp2d kernels (see
// launch/by_order.cuh): the drivers launch through these tables.
#include "kernels.cuh"

namespace qp2d {

#define QP2D_TABLE(fn_type, table, kernel, tilt) \
    fn_type const table[5] = {&kernel<2, tilt>, &kernel<4, tilt>, &kernel<6, tilt>, &kernel<8, tilt>, &kernel<-1, tilt>}

QP2D_TABLE(qp_forward_fn, qp_forward_vti_by_order, qp_forward, false);
QP2D_TABLE(qp_forward_fn, qp_forward_tti_by_order, qp_forward, true);
QP2D_TABLE(qp_reverse_fn, qp_reverse_vti_by_order, qp_reverse, false);
QP2D_TABLE(qp_reverse_fn, qp_reverse_tti_by_order, qp_reverse, true);
QP2D_TABLE(qp_adjoint_prep_fn, qp_adjoint_prep_vti_by_order, qp_adjoint_prep, false);
QP2D_TABLE(qp_adjoint_prep_fn, qp_adjoint_prep_tti_by_order, qp_adjoint_prep, true);
QP2D_TABLE(qp_born_forward_fn, qp_born_forward_vti_by_order, qp_born_forward, false);
QP2D_TABLE(qp_born_forward_fn, qp_born_forward_tti_by_order, qp_born_forward, true);
QP2D_TABLE(qp_born_reverse_fn, qp_born_reverse_vti_by_order, qp_born_reverse, false);
QP2D_TABLE(qp_born_reverse_fn, qp_born_reverse_tti_by_order, qp_born_reverse, true);
QP2D_TABLE(qp_born_image_fn, qp_born_image_vti_by_order, qp_born_image, false);
QP2D_TABLE(qp_born_image_fn, qp_born_image_tti_by_order, qp_born_image, true);

#undef QP2D_TABLE

SWEEP_BY_ORDER_TABLE(qp_adjoint_apply_fn, qp_adjoint_apply_by_order, qp_adjoint_apply);

} // namespace qp2d
