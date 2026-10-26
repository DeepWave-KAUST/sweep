// The one translation unit that instantiates this equation's kernel
// templates (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(elastic_born_velocity_kernel_fn, elastic_born_velocity_kernel_by_order, elastic_born_velocity_kernel);
SWEEP_BY_ORDER_TABLE(elastic_born_stress_kernel_fn, elastic_born_stress_kernel_by_order, elastic_born_stress_kernel);
