// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(das3d_first_derivatives_kernel_fn, das3d_first_derivatives_kernel_by_order, das3d_first_derivatives_kernel);
SWEEP_BY_ORDER_TABLE(das3d_update_kernel_fn, das3d_update_kernel_by_order, das3d_update_kernel);
SWEEP_BY_ORDER_TABLE(das3d_project_model_grad_kernel_fn, das3d_project_model_grad_kernel_by_order, das3d_project_model_grad_kernel);
