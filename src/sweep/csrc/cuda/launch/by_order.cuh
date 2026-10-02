#pragma once
// Stencil-order dispatch of a kernel template instantiated in ONE translation
// unit.
//
// A kernel<N><<<>>> written in a header -- a driver hook, a LAUNCH_* macro --
// instantiates the template (five stencil orders x every arch of the fat
// binary) in every TU that includes it, called or not, and the core carries a
// copy per such TU.  Instead the kernel family's own kernels.cu fills a table
// of the five specializations and every launch goes through the table:
//
//   kernels.cuh:  using foo_fn = void (*)(<foo's parameter types, in order>);
//                 SWEEP_BY_ORDER_DECL(foo_fn, foo_by_order);
//                 #define LAUNCH_FOO(order, grid, block, ...) \
//                     SWEEP_LAUNCH_BY_ORDER(foo_by_order, order, grid, block, __VA_ARGS__)
//   kernels.cu:   SWEEP_BY_ORDER_TABLE(foo_fn, foo_by_order, foo);
//
// The pointer type must name the kernel's parameter types exactly, or the
// table does not compile; call sites hand their arguments to the kernel as
// before, with nothing in between.  The launch is the one nvcc emits for
// foo<N><<<>>> (the execution configuration is pushed, then the
// specialization's host stub is called).  Default arguments do not pass
// through a pointer: call sites spell them out.
#include <cuda_runtime.h>

// Table slot of a stencil order: 2, 4, 6, 8, and the runtime-M variant (-1)
// for anything else, as the per-order dispatch always chose.
inline int sweep_order_slot(int order)
{
    return order == 2 ? 0 : order == 4 ? 1 : order == 6 ? 2 : order == 8 ? 3 : 4;
}

#define SWEEP_BY_ORDER_DECL(fn_type, table) extern fn_type const table[5]

#define SWEEP_BY_ORDER_TABLE(fn_type, table, kernel) \
    fn_type const table[5] = {&kernel<2>, &kernel<4>, &kernel<6>, &kernel<8>, &kernel<-1>}

#define SWEEP_LAUNCH_BY_ORDER(table, order, grid, block, ...) \
    table[sweep_order_slot(order)]<<<grid, block>>>(__VA_ARGS__)
