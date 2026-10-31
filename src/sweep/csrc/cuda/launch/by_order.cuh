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
#include "launch.h"   // SWEEP_LAUNCH

// Table slot of a stencil order: 2, 4, 6, 8, and the runtime-M variant (-1)
// for anything else, as the per-order dispatch always chose.
inline int sweep_order_slot(int order)
{
    return order == 2 ? 0 : order == 4 ? 1 : order == 6 ? 2 : order == 8 ? 3 : 4;
}

#if defined(SWEEP_HOST_CORE)
// Host core: the table holds per-order LAUNCHERS instead of kernels.  Each
// launcher names its kernel as a compile-time constant, so g++ inlines the
// kernel body into the virtual-thread loop (host/launch.h) and the by-value
// parameters -- SolverContext (240 B), the wavefield pointer bundle, the
// operator parameter blocks -- are materialised once per launch instead of
// being re-copied for every virtual thread through an opaque pointer
// (measured: 54 -> 25 ns/cell at one thread on the 2-D acoustic step, see
// docs/dev/host_core.md).  Same macros, same call sites, same kernel bodies;
// one indirect call per launch remains, which is what the CUDA side pays too.
namespace sweep {
namespace host {

template <class Fn> struct ByOrderLauncherType;
template <class... A> struct ByOrderLauncherType<void (*)(A...)> {
    using type = void (*)(dim3, dim3, A...);
};

template <class Fn, Fn Kern> struct ByOrderLauncher;
template <class... A, void (*Kern)(A...)> struct ByOrderLauncher<void (*)(A...), Kern> {
    static void run(dim3 grid, dim3 block, A... a)
    {
        launch(grid, block, [&](auto&&... x) { Kern(x...); }, a...);
    }
};

}  // namespace host
}  // namespace sweep

#define SWEEP_BY_ORDER_DECL(fn_type, table) \
    extern ::sweep::host::ByOrderLauncherType<fn_type>::type const table[5]

#define SWEEP_BY_ORDER_TABLE(fn_type, table, kernel)                                   \
    ::sweep::host::ByOrderLauncherType<fn_type>::type const table[5] = {                \
        &::sweep::host::ByOrderLauncher<fn_type, &kernel<2>>::run,                      \
        &::sweep::host::ByOrderLauncher<fn_type, &kernel<4>>::run,                      \
        &::sweep::host::ByOrderLauncher<fn_type, &kernel<6>>::run,                      \
        &::sweep::host::ByOrderLauncher<fn_type, &kernel<8>>::run,                      \
        &::sweep::host::ByOrderLauncher<fn_type, &kernel<-1>>::run}

// A kernel split per launch/cell_path.h brings its own host launcher, a
// class template ``launcher<N>`` whose static ``run`` takes (dim3 grid,
// dim3 block, <the kernel's parameters>) and goes through
// sweep::host::launch_rows; the CUDA table is the plain one.
#define SWEEP_BY_ORDER_TABLE_ROWS(fn_type, table, kernel, launcher)                     \
    ::sweep::host::ByOrderLauncherType<fn_type>::type const table[5] = {                \
        &launcher<2>::run, &launcher<4>::run, &launcher<6>::run, &launcher<8>::run,     \
        &launcher<-1>::run}

#define SWEEP_LAUNCH_BY_ORDER(table, order, grid, block, ...) \
    table[sweep_order_slot(order)](grid, block, __VA_ARGS__)
#else
#define SWEEP_BY_ORDER_DECL(fn_type, table) extern fn_type const table[5]

#define SWEEP_BY_ORDER_TABLE(fn_type, table, kernel) \
    fn_type const table[5] = {&kernel<2>, &kernel<4>, &kernel<6>, &kernel<8>, &kernel<-1>}

#define SWEEP_BY_ORDER_TABLE_ROWS(fn_type, table, kernel, launcher) \
    SWEEP_BY_ORDER_TABLE(fn_type, table, kernel)

#define SWEEP_LAUNCH_BY_ORDER(table, order, grid, block, ...) \
    SWEEP_LAUNCH(table[sweep_order_slot(order)], grid, block, __VA_ARGS__)
#endif
