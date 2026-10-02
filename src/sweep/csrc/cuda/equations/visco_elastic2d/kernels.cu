// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(visco_elastic2d_stress_kernel_fn, visco_elastic2d_stress_kernel_by_order, visco_elastic2d_stress_kernel);
SWEEP_BY_ORDER_TABLE(visco_elastic2d_stress_adjoint_prepare_fn, visco_elastic2d_stress_adjoint_prepare_by_order, visco_elastic2d_stress_adjoint_prepare);
