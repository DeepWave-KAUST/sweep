// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(das_mu2d_stress_strain_kernel_fn, das_mu2d_stress_strain_kernel_by_order, das_mu2d_stress_strain_kernel);
SWEEP_BY_ORDER_TABLE(das_mu2d_stress_strain_adjoint_prepare_fn, das_mu2d_stress_strain_adjoint_prepare_by_order, das_mu2d_stress_strain_adjoint_prepare);
