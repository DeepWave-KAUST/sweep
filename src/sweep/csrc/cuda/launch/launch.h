#pragma once
// launch/launch.h -- the ONE spelling of a kernel launch in this tree.
//
//     SWEEP_LAUNCH(kern, grid, block, args...)
//     SWEEP_LAUNCH_ON(kern, grid, block, shmem_bytes, stream, args...)
//     SWEEP_LAUNCH_ROWS(kern, host_run, grid, block, args...)
//
// Under nvcc both expand to exactly the ``kern<<<grid, block>>>(args...)`` /
// ``kern<<<grid, block, shmem, stream>>>(args...)`` the call sites used to
// spell, so the CUDA core is unchanged to the bit (the bitgate is the proof).
// ``kern`` may be a plain kernel, a function-pointer table element
// (``table[slot]``) or a parenthesised template-id (``(k<2, X>)``).
//
// Under the host core (``-DSWEEP_HOST_CORE``, g++ compiling the same .cu
// sources as C++ through csrc/host/include/cuda_runtime.h) they expand to
// ``sweep::host::launch``: the kernel runs once per virtual thread of the
// same grid x block decomposition, blocks spread over OpenMP threads, with
// ``blockIdx`` / ``threadIdx`` / ``blockDim`` / ``gridDim`` in thread-local
// storage -- so the kernel BODIES compile and run unchanged on both targets.
// Shared memory and the stream are host no-ops (``shmem_bytes`` must be 0:
// the two kernels that needed __shared__ have host bodies of their own, see
// boundarysaver.cu).
//
// Why a macro and not a function: ``<<<>>>`` is the one piece of CUDA syntax
// a C++ compiler cannot parse, so it has to be hidden behind the
// preprocessor; everything else the kernels use is supplied by the host shim
// headers as ordinary C++.
//
// SWEEP_LAUNCH_ROWS is for a kernel split per launch/cell_path.h that is not
// launched through a by-order table (by_order.cuh SWEEP_BY_ORDER_TABLE_ROWS
// covers those): ``kern<<<grid, block>>>(args...)`` under nvcc, its host row
// launcher ``host_run(grid, block, args...)`` on the host core.
#if defined(SWEEP_HOST_CORE)
#include "../../host/launch.h"
// The kernel is called by name inside a generic lambda (not through a decayed
// pointer) so its default arguments apply as under <<<>>> and the body can be
// inlined; the launch arguments are evaluated once, before the loop.
#define SWEEP_LAUNCH(kern, grid, block, ...) \
    ::sweep::host::launch(grid, block, [&](auto&&... sweep_a_) { kern(sweep_a_...); }, __VA_ARGS__)
#define SWEEP_LAUNCH_ON(kern, grid, block, shmem, stream, ...) \
    ::sweep::host::launch_on(grid, block, shmem, stream, [&](auto&&... sweep_a_) { kern(sweep_a_...); }, __VA_ARGS__)
#define SWEEP_LAUNCH_ROWS(kern, host_run, grid, block, ...) \
    host_run(grid, block, __VA_ARGS__)
#else
#include <cuda_runtime.h>
#define SWEEP_LAUNCH(kern, grid, block, ...) \
    kern<<<grid, block>>>(__VA_ARGS__)
#define SWEEP_LAUNCH_ON(kern, grid, block, shmem, stream, ...) \
    kern<<<grid, block, shmem, stream>>>(__VA_ARGS__)
#define SWEEP_LAUNCH_ROWS(kern, host_run, grid, block, ...) \
    kern<<<grid, block>>>(__VA_ARGS__)
#endif
