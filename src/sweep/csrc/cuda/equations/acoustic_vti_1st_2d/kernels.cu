// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

namespace acoustic_vti_1st_2d {
SWEEP_BY_ORDER_TABLE(velocity_kernel_fn, velocity_kernel_by_order, velocity_kernel);
SWEEP_BY_ORDER_TABLE(stress_kernel_fn, stress_kernel_by_order, stress_kernel);
SWEEP_BY_ORDER_TABLE(velocity_kernel_nopml_fn, velocity_kernel_nopml_by_order, velocity_kernel_nopml);
SWEEP_BY_ORDER_TABLE(stress_kernel_nopml_fn, stress_kernel_nopml_by_order, stress_kernel_nopml);
SWEEP_BY_ORDER_TABLE(adjoint_stress_to_vel_kernel_fn, adjoint_stress_to_vel_kernel_by_order, adjoint_stress_to_vel_kernel);
SWEEP_BY_ORDER_TABLE(adjoint_vel_to_stress_kernel_fn, adjoint_vel_to_stress_kernel_by_order, adjoint_vel_to_stress_kernel);
SWEEP_BY_ORDER_TABLE(calculate_grad_kernel_fn, calculate_grad_kernel_by_order, calculate_grad_kernel);
} // namespace acoustic_vti_1st_2d
