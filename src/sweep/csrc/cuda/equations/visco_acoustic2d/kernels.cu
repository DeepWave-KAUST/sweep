// The one translation unit that instantiates this equation's kernel
// templates: the tables behind the order-dispatch macros in kernels.cuh
// (launch/by_order.cuh).
#include "kernels.cuh"

SWEEP_BY_ORDER_TABLE(visco_acoustic2d_carrier_fn, visco_acoustic2d_carrier_by_order, visco_acoustic2d_carrier);
