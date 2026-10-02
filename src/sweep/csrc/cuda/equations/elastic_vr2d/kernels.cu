// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

namespace elastic_vr2d_kernels {
SWEEP_BY_ORDER_TABLE(evr_momentum_kernel_fn, evr_momentum_kernel_by_order, evr_momentum_kernel);
SWEEP_BY_ORDER_TABLE(evr_stress_kernel_fn, evr_stress_kernel_by_order, evr_stress_kernel);
SWEEP_BY_ORDER_TABLE(evr_momentum_kernel_nopml_fn, evr_momentum_kernel_nopml_by_order, evr_momentum_kernel_nopml);
SWEEP_BY_ORDER_TABLE(evr_stress_kernel_nopml_fn, evr_stress_kernel_nopml_by_order, evr_stress_kernel_nopml);
SWEEP_BY_ORDER_TABLE(evr_stress_adjoint_prepare_fn, evr_stress_adjoint_prepare_by_order, evr_stress_adjoint_prepare);
SWEEP_BY_ORDER_TABLE(evr_stress_adjoint_apply_fn, evr_stress_adjoint_apply_by_order, evr_stress_adjoint_apply);
SWEEP_BY_ORDER_TABLE(evr_momentum_adjoint_prepare_fn, evr_momentum_adjoint_prepare_by_order, evr_momentum_adjoint_prepare);
SWEEP_BY_ORDER_TABLE(evr_momentum_adjoint_apply_fn, evr_momentum_adjoint_apply_by_order, evr_momentum_adjoint_apply);
SWEEP_BY_ORDER_TABLE(calculate_grad_evr_nobs_fn, calculate_grad_evr_nobs_by_order, calculate_grad_evr_nobs);
SWEEP_BY_ORDER_TABLE(evr_grad_chain_apply_fn, evr_grad_chain_apply_by_order, evr_grad_chain_apply);
evr_adjoint_zero_top_fs_fn const evr_adjoint_zero_top_fs_0 = &evr_adjoint_zero_top_fs<0>;
} // namespace elastic_vr2d_kernels
