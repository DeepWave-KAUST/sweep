# The host (CPU) core

The compiled backend has one source tree and two cores built from it: the
**CUDA core** (`nvcc`, `lib/cu12` / `lib/cu13`) and the **host core** (`g++`,
`lib/cpu` or a local build).  Both are a `libsweep_core.so` with the same C API
(`csrc/core/capi.h`), the same entry table and the same ctypes layer
(`sweep.backend.c`).  A CUDA tensor goes to the CUDA core, a CPU tensor to the
host core; no kernel, driver, boundary saver or checkpoint scheduler exists
twice.  This page is the architecture behind that, how to build and ship the
host core, what its numerics promise, and the rules a kernel must keep for the
scheme to hold.

Why not a second, hand-written CPU engine: SWEEP had one, and it was deleted
(`af58082b`) because its gradients had drifted from the eager and CUDA paths.  A
copy of an algorithm is a second place for it to be wrong.  The host core cannot
drift, because it *is* the CUDA code.

## 1. Architecture

```
csrc/core/        C ABI: Buf, BufList, Forward/BackwardInputCore, capi.h   (device-neutral)
csrc/cuda/        kernels (.cuh/.cu), driver skeletons, boundary saving,
                  checkpointing, capi.cu                                   (compiled twice)
csrc/host/        the CUDA stand-in headers + the virtual-grid launcher    (host core only)
sweep/backend/c/  ctypes layer: loader (two kinds), entries (core per call),
                  runners, adapt, abi (generated), jit (CUDA build),
                  hostcore (host build / ship / availability)
```

The same `.cu` files are compiled by `g++ -x c++ -std=c++17` with
`-DSWEEP_HOST_CORE`, `-isystem csrc/host/include` and `-include cuda_runtime.h`
forced first.  What a kernel needs from CUDA -- the qualifiers, `dim3`, the
index registers, the runtime calls the drivers make, `atomicAdd`, the
round-to-nearest intrinsics, `__half` / `__nv_bfloat16` conversion -- is
provided by `csrc/host/include/` (section 2).  A kernel launch is written once
as a macro (section 3) that nvcc expands to `<<<>>>` and g++ to the virtual-grid
launcher (section 4).  The few places that genuinely need CUDA -- block-shared
memory in the int8/fp16 boundary quantisers and the tiled VRZ gradients, cuFFT
in the visco-acoustic spectral step -- get a host body, take another path, or
are left out of the host core by name (section 5).

Two facts make this cheap.  Every kernel in the tree is "decode `(ix[, iy],
iz, b)` from `blockIdx` / `threadIdx` / `blockDim`, bounds early-return,
per-cell arithmetic, write own cell": running it once per `(block, thread)` of
the grid × block the CUDA side launches reproduces its index space exactly.
And the drivers never allocate: every wavefield, boundary ring and checkpoint
is a buffer Python bound on the input struct, so `cudaMalloc` is not among the
runtime calls to stand in for.

## 2. The stand-in headers (`csrc/host/include/`)

`cuda_runtime.h` (plus `cuda.h`, `cuda_fp16.h`, `cuda_bf16.h`) is a *small*
header that refuses to be included without `SWEEP_HOST_CORE` (`#error`), so it
can never shadow the real toolkit in an nvcc build.  It provides exactly what
the tree uses:

| Area | Stand-in |
|---|---|
| Qualifiers | `__global__`, `__device__`, `__host__` empty; `__forceinline__` → `inline __attribute__((always_inline))`; `__launch_bounds__(...)` empty; `__syncthreads()` is **deleted** -- a kernel that calls it fails to compile, by design (section 5) |
| Index registers | `inline thread_local dim3 threadIdx, blockIdx, blockDim, gridDim` (one definition for the whole library; `-ftls-model=initial-exec` keeps a TLS read to one instruction) |
| Streams / events | `CUstream_st{flags}`, `CUevent_st{steady_clock time, flags}`; create / destroy / record / synchronize / wait / elapsed; null handles answer `cudaErrorInvalidValue` |
| Memory | `cudaMemcpy[Async]` → `memmove`; `cudaMemcpy2DAsync` row by row; `cudaMemset[Async]`; `cudaLaunchHostFunc` runs the callback at once |
| Atomics | `atomicAdd(float*, float)` as a compare-and-swap loop (+ `int`, `unsigned`, `unsigned long long`) |
| Intrinsics | `__fmul_rn / __fadd_rn / __fsub_rn / __fdiv_rn / __fsqrt_rn / __dmul_rn / __dadd_rn` as the plain operation (`-ffp-contract=off` keeps them un-fused), `__float2int_rn` → `lrintf`, `max` / `min` overloads for the integer and floating widths kernels use |
| Half types | `__half` / `__nv_bfloat16` with round-to-nearest-even conversion including subnormals, inf and NaN -- verified bit-identical to torch's on 781k floats and all 65536 half patterns |

A CUDA call the tree does not make is not in the header; adding one is a
deliberate act, and the compiler names the site.

## 3. The launch macro (`csrc/cuda/launch/launch.h`)

Every kernel launch in `csrc/cuda/` is spelled

```cpp
SWEEP_LAUNCH(kernel, grid, block, args...);                 // kernel<<<grid, block>>>(args...)
SWEEP_LAUNCH_ON(kernel, grid, block, shmem, stream, args...);
```

Under nvcc the macro is textually the `<<<>>>` form.  Under `SWEEP_HOST_CORE`
it is

```cpp
::sweep::host::launch(grid, block, [&](auto&&... a) { kernel(a...); }, args...);
```

The kernel is named inside a generic lambda, not passed as a function pointer,
for two reasons: default arguments (`set_boundary_zeros`' cut mask, the
boundary kernels' tangent pad) apply exactly as under `<<<>>>`, and the
compiler can see the body to inline it.  There are 329 launch sites in 42
files; the migration from literal `<<<>>>` touched no kernel body (the rewrite
is mechanical and the CUDA machine code is unchanged -- the bitgate tiers A/C/T
are green across it).

**Rule: no literal `<<<>>>` in `csrc/cuda/`.**  A kernel launched any other way
does not exist in the host core.

## 4. The virtual-grid launcher (`csrc/host/launch.h`)

```cpp
template <class Call, class... Args>
void launch(dim3 grid, dim3 block, Call&& call, Args&&... args) {
    #pragma omp parallel for schedule(static)
    for (long long bl = 0; bl < nblocks; ++bl)
        run_block(grid, block, bl, call, args...);   // noinline, flatten
}
```

Blocks are spread over OpenMP threads with a static schedule (a thread gets a
contiguous run of blocks -- a band of rows), and the threads of one block run
sequentially, x innermost, in the calling OpenMP thread, with `threadIdx` /
`blockIdx` / `blockDim` / `gridDim` set in the thread-local registers before
each call.  A virtual thread is a real function call, so a kernel's early
`return` ends that thread only, as on the GPU.

`run_block` is `__attribute__((noinline, flatten))`, and that attribute pair is
load-bearing.  GCC outlines the body of an `omp parallel for` into a function
of its own *before* the inliner runs; `always_inline`/`flatten` on something
that is itself inlined into that outlined body does nothing.  Keeping
`run_block` as a real function (one call per block, not per cell) and
flattening it makes every call inside it -- the lambda, then the kernel
whenever its definition is visible -- inline regardless of size heuristics.
Once inlined, a kernel's by-value parameters (`SolverContext` 240 B, the
pointer bundles -- about 600 B per acoustic launch) are read where they stand
instead of being copied per virtual thread.  That one change took the 2-D
8th-order acoustic step from 54 to 25 ns/cell/step at one thread.

### By-order tables on the host (`launch/by_order.cuh`)

A family's order-templated kernels are instantiated once, in its `kernels.cu`,
and launched through a table of the five specialisations (`2, 4, 6, 8, -1`).
Under nvcc the table holds kernel pointers and `SWEEP_LAUNCH_BY_ORDER` does
`SWEEP_LAUNCH(table[slot], ...)`.  Under the host core the table holds
*launchers* -- `ByOrderLauncher<fn_type, &kernel<N>>::run` -- each naming its
kernel as a compile-time constant, so the kernel inlines into the
virtual-thread loop exactly as a directly named kernel does.  The macros and
call sites are the same on both sides; one indirect call per launch remains,
which is what the CUDA side pays too.

### What the emulation costs

Micro-benchmark on the 2-D 8th-order acoustic step (2 × Xeon Platinum 8176): the emulated
kernel runs at 1.3-1.5× a scalar hand-written loop and 4-6× a SIMD-vectorised
one -- the CUDA-style bounds guards and the per-cell call structure keep the
compiler from vectorising.  That is the price of zero kernel edits.  A kernel
that must go faster is split so its interior can run as a SIMD loop (next
subsection) -- never a second driver, never a second copy of the arithmetic.

Measured on a 2 × Xeon Platinum 8176 workstation (56 cores), every run checked for
other load before and during the timing (other processes < 0.5 cores
throughout), ns per cell per time step, lower is better.  Grids: 2-D 500 × 600
+ 30-cell PML, 1500 steps; 3-D 100 × 100 × 120 + 20-cell PML, 400 steps -- long
enough for the wavefield to decay, which is where denormals appear.  "before"
is the host core with IEEE denormals, "FTZ" with them flushed (section 8):

| equation | order | | threads | before | FTZ | speed-up |
|---|---|---|---|---|---|---|
| acoustic2d | 8 | forward | 1 | 43.7 | 28.0 | 1.6× |
| acoustic2d | 8 | forward | 8 | 10.6 | 4.6 | 2.3× |
| acoustic2d | 8 | forward | 28 | 4.9 | 1.9 | 2.6× |
| acoustic2d | 8 | forward + backward (BS) | 28 | 39.6 | 7.1 | 5.6× |
| elastic2d | 8 | forward | 28 | 12.2 | 3.6 | 3.4× |
| elastic2d | 8 | forward + backward (BS) | 28 | 62.7 | 15.6 | 4.0× |
| acoustic3d | 8 | forward | 28 | 10.0 | 3.8 | 2.6× |
| acoustic3d | 8 | forward + backward (BS) | 28 | 63.4 | 16.4 | 3.9× |
| elastic3d | 4 | forward + backward (BS) | 28 | 72.0 | 23.6 | 3.1× |

Across orders 2/4/8, 8/28 threads and both modes the gain is 1.2-5.6×,
largest for forward + backward (the adjoint field decays too) and at high
order.  Thread scaling also improves: denormal assists serialise each core,
so 8 → 28 threads gave 1.5× before and 2.5× after on the acoustic2d order-8
backward.  For reference the CUDA core (RTX 6000 Ada) runs the acoustic2d
order-8 forward at ~0.02.  `OMP_NUM_THREADS` is the knob; the eager path's
thread count is torch's.

### Interior rows as SIMD loops (`launch_rows`, `launch/cell_path.h`)

What keeps an emulated kernel scalar is its control flow, not its arithmetic:
the bounds returns, the PML-or-interior branch, the free-surface and
topography tests, the optional outputs (`if (u_this != nullptr)`).  In the
interior of the grid every cell takes the same branch, so a kernel can hand
that part to the host as a straight loop.  Every equation the host core
builds does this for its stencil kernels (section 5 lists the ones it does
not build).

* **The kernel is split, nothing is copied.**  The `__global__` function keeps
  the index decode and the bounds returns; everything after them moves,
  unchanged and in the same order, into a
  `template <int Order, CellPath Path = CellPath::Any> __device__ __forceinline__ <kernel>_cell(...)`
  that the kernel calls with `Path = Any`.  The other paths are the same body
  with its run-time tests answered by the caller, written `if constexpr`, so on
  the kernel's own path the code is the code it was:
  * `interior_path<P>`: the cell is in the kernel's *interior box* (a
    `CellBox` / `CellBox3` built from the same expressions the kernel tests),
    so the PML, air and surface-row branches are dead;
  * `optional_output<P, K>(ptr)`: optional output `K` is present or absent;
  * `path_flag<P, F>`: a launch-uniform switch that changes the arithmetic,
    made compile-time -- the elastic family's free-surface faces
    (`ELASTIC_FS_Z`, `ELASTIC_FS_X`; with a face on, the near-surface
    derivative is a different float expression, see
    `common/elastic_free_surface.cuh`: the Path-aware helpers replay the Any
    path's operation order with the mirror tests resolved).  Integer switches
    (visco_elastic2d's mechanism count) are a plain template parameter.
* **The box is a subset of the branch, never more.**  Anything the interior
  path hard-codes must hold for every cell of the box: the PML test, the
  free-surface reach (`common/host_rows.h` `elastic_fs_box_2d` /
  `elastic3d_fs_box`: forward and adjoint margins, per face), topography (rows
  below the deepest surface row over the box's columns: `below_surface_2d`,
  `wave3d_rows`), DD cut faces and `x_base`/`x_end` phases.  A box that is too
  small only costs speed; one too large is a wrong result on a geometry the
  tests may not reach.
* **The host launcher walks rows.**  `sweep::host::launch_rows` covers the same
  virtual-thread index space as `launch()`, row by row over the OpenMP threads
  (`parallel_for_runs`: one contiguous run of rows per thread, its tail
  balanced by stealing -- see *Scheduling the rows* below); in each row the
  cells inside the box go to the cell body in one `#pragma omp simd` loop, the
  others run the kernel itself.  Wave2D rows are `(iz, b)`, Wave3D rows
  `(iy, b * nz + iz)`, decoded per row because a z block straddles two shots
  when `nz` is odd.  `with_optionals` / `with_flags` instantiate the loop once
  per compile-time path; the cell lambda captures by value so the parameters
  are row-locals no store can alias.
* **One scalar copy of each kernel.**  The cells outside the span run through
  `run_through` (`noinline, flatten`: the kernel is inlined into the loop, one
  call per row segment, not per cell), shared by every path of a
  `(kernel, Order)` -- which requires the launcher to build its `call` lambda
  *above* the `with_*` dispatch, so its closure type does not depend on the
  path.  Built inside, every path compiled its own copy of the Any kernel: the
  host core took 460 s and 47.8 MB (two TUs 450 s each, 1.9 GB) instead of
  63 s and 20.2 MB (4 cores: 140 s).  The build also starts the heaviest TUs
  first (`hostcore._host_ninja`).
* **Selection** is the by-order table: `SWEEP_BY_ORDER_TABLE_ROWS(fn, table,
  kernel, Launcher)` is the plain table under nvcc and `Launcher<N>::run` on
  the host; `SWEEP_LAUNCH_ROWS(kern, host_run, grid, block, args...)` does the
  same for a kernel launched by name.  The launchers live in an anonymous
  namespace in the family's `kernels.cu`, under `SWEEP_HOST_CORE`, and hold no
  arithmetic.
* **Split steps.**  `acoustic2d`/`3d`, `acoustic_vrz2d`/`3d` and
  `acoustic_born2d`/`3d` step with an interior kernel launched over a
  `fdtd::Box` (`launch/box.cuh` `box_launch`: the virtual thread *is* a box
  cell) plus thin CPML ring / shell kernels over the band.  The interior
  kernel carries no CPML branch at all, so its host launcher gives every row
  the box's whole x extent (`common/host_rows.h` `box_launch_rows`: empty past
  the last shot and, with topography, above the deepest surface row over the
  box's columns, where the kernel's air return must run) and decodes the cell
  with `box_cell_at`; the rings and shells stay on `launch()`.  The Born
  steps' pointwise kernels (scattering source, imaging, the optional
  `grad_iii`) are split like the stencils (`acoustic_born{2,3}d/kernels.cu`,
  the `*_rows` runners at the end of the file).
* **`SWEEP_HOST_ROWS=0`** sends every cell through the kernel, walked by rows
  the same way (`launch_through`): the A/B switch for the interior paths.

**Scheduling the rows** (`host/launch.h` `parallel_for_runs`).  Each OpenMP
thread gets one contiguous run of rows, `[n t / T, n (t + 1) / T)`, cut into
about 32 chunks.  It takes its own chunks from the front; a thread done with
its run takes chunks from the *back* of the others'.  The runs are set up
before the parallel region (one packed front/back word per run, claimed with
a compare-and-swap), so a launch pays no barrier beyond the region's own.

* Contiguous, because a row's stencil reads its neighbours' rows, and because
  a balanced launch then hands every row to the same thread launch after
  launch: its rows stay in that thread's caches and in its NUMA node's
  memory.  Handed out dynamically two rows at a time (the schedule before),
  the same core ran 2-7x slower on 2-D grids at 32-64 threads on a 2-socket
  EPYC 7702 (8 NUMA nodes; 1.0-1.5x on 3-D) -- slower than the block-static
  scalar launcher.
* Stealing, because a full-grid kernel's PML band rows run the scalar kernel
  end to end: one static run per thread puts them all on the first and the
  last thread (elastic3d forward 1.65x slower at 8 threads on a Xeon 8176).
* From the back, so a thief does not interleave with the owner's walk.

Which thread runs which row changes no result (rows write disjoint cells):
the scheduler change was bit-identical on every family's configurations.

What stays scalar: the PML band and edges (the kernel itself, inlined in
`run_through`) and the split steps' rings and shells; order -1 (runtime M),
which takes the same spans with the kernels' run-time halo but whose row loop
does not vectorise (its tap loops do); acoustic_vti_1st's gradient (a `sqrtf`
keeps an errno branch); the DAS adjoints, which scatter with `atomicAdd` (a
gather rewrite would change the CUDA numerics); point and strip kernels.

**Nothing moves on either side.**  The host result is bit-identical to the
pre-port core and to `SWEEP_HOST_ROWS=0` (`-ffp-contract=off`: every lane
rounds what the scalar cell rounded) -- 846 bitgate configurations, 17
equations, every memory mode, orders 2-10, plus extra sweeps for what the
suite does not reach (B = 2, odd sizes, per-edge free surfaces, topography,
DD phases, elastic_vr2d, visco_elastic2d).  Under nvcc every kernel keeps its
registers, stack and opcode mix: `port/sass_cmp.sh`-style strict comparison
(opcodes *with* modifiers; only set-predicate comparisons normalised) -- a
cell handed a CPML pointer struct lost `LDG.E.CONSTANT` on the profile loads
until the profiles were passed as separate `__restrict__` pointers, and int
index parameters swapped compare operands until they were passed as a
by-value index struct.  The bitgate is green.

Re-verified when the port moved onto dev's split steps (2026-10-31, dev
9f660919): the six split families on the row launcher, bit-identical to the
same kernels on the scalar launcher and to `SWEEP_HOST_ROWS=0` in 378
configurations (orders default/2/8/10, every memory mode, interior, free
surface, a cut edge), the other eleven families in 174, and the scheduler
change (`parallel_for_runs`) in 282 more; on the CUDA side the bitgate's
compiled configurations (tiers A/B/C/T) are bit-identical to dev, and per
translation unit the SASS of sm_70/75/80/86/89/90 is dev's except two
kernels: `a2d_adj_interior<-1>` (the runtime-order adjoint interior, orders
10 and up), which nvcc schedules differently once its body sits behind the
cell function's boundary, whatever the body's spelling (V100 time unchanged,
H200 +0.5% on the order-10 step), and, on sm_90 only,
`elastic_stress_kernel_3d<8>` (1256 -> 1264 SASS instructions; H200 time
unchanged).

Re-verified after the rebase onto dev 5ce0acff (2026-11-07: the SEM
additions, eager speed-ups and the removal of the elastic image-method
topography): the PTX of all 92 translation units at compute_70 and
compute_90 is dev's except the same 11 with cell functions, whose SASS on
the six architectures is dev's except the same two kernels with the same
instruction counts; the bitgate's compiled configurations (tiers A/B/C/T)
are bit-identical to dev, every eager configuration that differs also
differing between two runs of dev; the row launcher is bit-identical to
`SWEEP_HOST_ROWS=0` in 282 configurations; host against CUDA as below.

**Speed.**  The row launcher against `SWEEP_HOST_ROWS=0` -- the same core, the
same row walk and scheduler, every cell through the kernel -- so the table is
what the SIMD rows alone buy.  Order 4, AMD EPYC 7702 (2 sockets x 64 cores,
8 NUMA nodes), baseline x86-64 build, threads unpinned, median of two runs of
each setting, interleaved; 2-D 2048^2 + 30-cell PML, 3-D 160^3 + 20-cell PML;
forward + backward is boundary saving, full for das3d; FS = free surface on,
as marine runs are:

| equation | grid | forward, 8 / 64 threads | forward + backward, 8 / 64 threads |
|---|---|---|---|
| acoustic2d | 2048^2 | 2.2x / 1.2x | 2.7x / 1.5x |
| vrz2d | 2048^2 | 2.1x / 1.8x | 3.5x / 2.0x |
| born2d | 2048^2 | 3.8x / 1.9x | 3.6x / 1.7x |
| das2d | 2048^2 | 2.4x / 1.0x | 1.1x / 1.0x |
| das_mu2d | 2048^2 FS | 6.2x / 3.5x | 5.2x / 1.7x |
| elastic2d | 2048^2 FS | 7.5x / 5.0x | 6.4x / 2.4x |
| acoustic_vti_1st_2d | 2048^2 | 3.4x / 1.6x | 2.3x / 1.4x |
| elastic_tti_2nd2d | 2048^2 | 2.2x / 1.1x | 2.0x / 0.8x |
| elastic_tti_sg2d | 2048^2 | 1.3x / 1.0x | 1.2x / 1.1x |
| acoustic3d | 160^3 | 1.2x / 1.1x | 1.1x / 1.0x |
| vrz3d | 160^3 | 1.2x / 1.0x | 1.7x / 1.2x |
| born3d | 160^3 | 1.1x / 1.0x | 1.2x / 1.0x |
| das3d | 160^3 | 1.2x / 1.4x | 1.4x / 1.3x |
| das_mu3d | 160^3 FS | 1.9x / 1.2x | 1.6x / 1.1x |
| elastic3d | 160^3 FS | 1.6x / 1.3x | 1.5x / 1.0x |
| acoustic_vti_1st_3d | 160^3 | 1.3x / 1.1x | 1.4x / 1.0x |
| elastic_tti_sg3d | 160^3 | 1.4x / 1.3x | 1.7x / 1.5x |

At 8 threads the 2-D grids gain 1.3-7.5x forward and 1.1-6.4x with the
backward (median 2.4x / 2.7x), the full-grid elastic and DAS-mu kernels most;
at 64 threads, where the socket's memory bandwidth bounds the stencils, 1.0-5.0x
and 0.8-2.4x (elastic_tti_2nd2d's backward is the one that loses).  The 3-D
grids gain 1.1-1.9x: half the cells of a 160^3 grid with a 20-cell PML are in
the CPML band, which runs scalar (the split steps' shells through `launch()`,
the full-grid kernels' band rows through `run_through`).  elastic_tti_sg2d is
held back by its Bond rotation (Python, before the time loop), the DAS
adjoints by their atomic scatters.  Absolute numbers at 64 threads, ns per
cell per step: acoustic2d 0.20 forward / 0.70 with the backward, elastic2d FS
0.37 / 1.7, acoustic3d 0.71 / 2.8, elastic3d FS 1.7 / 7.1.

**Against eager.**  What `'auto'` chooses between on a CPU device (section 6):
eager time over host-core time, 2 x Xeon Platinum 8176 (2 x 28 cores), order
4, 2-D n^2 + 30-cell PML, 3-D 128^3 + 20-cell PML, elastic2d with the free
surface on; full wavefield storage on both for the backward; steady state
(eager's first call on a grid also compiles, 6-12 s here; the host core's
first call runs as fast as the rest).  Eager with its defaults (`torch.compile`
on), in parentheses the best tuned setting (`cpml_strips=True`, with and
without flush-denormal) where it is better:

| equation, grid | pass | 4 threads | 8 threads | 28 threads |
|---|---|---|---|---|
| acoustic2d 1024^2 | forward | 0.86x | 2.2x | 4.2x |
| acoustic2d 1024^2 | forward + backward | 2.6x (1.6x) | 3.0x (2.8x) | 8.4x (7.7x) |
| acoustic2d 2048^2 | forward | 1.7x | 3.1x | 8.8x |
| acoustic2d 2048^2 | forward + backward | 3.7x (2.1x) | 5.1x (3.9x) | 8.5x |
| elastic2d 1024^2 | forward | 3.2x | 5.9x | 18x |
| elastic2d 1024^2 | forward + backward | 2.6x | 6.7x | 17x |
| elastic2d 2048^2 | forward | 3.4x | 8.3x | 16x |
| elastic2d 2048^2 | forward + backward | 5.8x | 9.2x (8.0x) | 9.9x (7.9x) |
| acoustic3d 128^3 | forward | 0.66x (0.52x) | 1.2x (0.88x) | 3.2x (2.4x) |
| acoustic3d 128^3 | forward + backward | 1.5x (0.84x) | 2.7x (1.7x) | 7.9x (4.5x) |

The host core is ahead at 8 threads and more everywhere but the 3-D forward
against tuned eager, and pulls further ahead with the thread count: eager's
dozens of small torch kernels per step scale worse than one OpenMP loop per
kernel.  At 4 threads compiled eager wins the 3-D forward (1.5x, 1.9x tuned)
and the 1024^2 acoustic forward (1.16x) -- the scalar CPML band again, 54% of
the cells of a 128^3 grid with a 20-cell PML.  A second run on 256^2 (8 / 28
threads) gave 1.2x / 2.0x forward and 2.4x / 18x with the backward for
acoustic2d, 1.6x / 3.3x forward for elastic2d; uncompiled eager
(`use_compile=False`) is 5-66x slower than the host core on the 2-D grids.
Single runs per setting: the settings both runs measured differ by up to 1.5x
in the ratio, so read the table to a factor, not a digit.

## 5. What is not emulated

* **Dynamic shared memory.**  `launch_on` throws on a non-zero `shmem`, and
  `__syncthreads` is deleted, so a kernel that uses block-cooperative shared
  memory cannot compile into the host core by accident.  The only such kernels
  are the int8 / fp16 boundary quantisers (`common/boundarysaver.cu`): they
  have host bodies (OpenMP over the same 256-cell blocks, cut per shot, with
  the same block max-abs and operand order) under `#if defined(SWEEP_HOST_CORE)`
  next to the CUDA ones --
  and the tiled VRZ gradients (`acoustic_vrz2d/grad_tiled.cu`,
  `acoustic_vrz3d/grad_tiled.cu`), which the host core does not build: there
  `acoustic_vrz{2,3}d_grad_is_tiled()` is false and the tiled launches report
  "did not run", so every order takes the global-memory gradient kernels the
  CUDA core keeps for order 2 and the runtime orders (same per-cell result,
  same operand order), and the boundary-saving reverse step its unfused
  reconstruction + gradient sequence.
* **CUDA libraries, and what came later.**  `visco_acoustic2d` needs cuFFT;
  `elastic_born2d`, the qP pair (`qp2d`: `qp_vti2d`, `qp_tti2d` and their Born
  twins) and the SEM kernels (`sem2d`, `sem3d`) came to the CUDA core after the
  host port and are not ported yet (`sweep.sem`'s compiled path is CUDA-only
  anyway).  They are named in `sweep.backend.c.hostcore.HOST_UNPORTED`
  (directory -> the namespaces its header declares, i.e. the entry prefixes and
  the equations' `C_NAME`), which drives three things: their translation units
  are left out, a generated `host_unported.cpp` defines every function their
  headers declare to throw `"<namespace>::<name>: <namespace> is not built into
  the host (CPU) core"`, and `equation_available()` answers no for their
  `C_NAME`.  The C API answers
  `sweep_entry_available(entry) == 0` for them and
  `sweep_visco_fft_workspace_bytes` returns 0.  The entry *table* is the same
  in both cores (the loader checks), so entry ids never diverge.
* **Host staging.**  `boundary_on_cpu`, `boundary_on_disk`,
  `checkpoint_on_cpu` and pinned memory describe moving data off the compute
  device; on the host core the compute device *is* host RAM.  They are refused
  at the C API (`sweep::refuse_host_staging`, `core/compute_device.h`) and, in
  Python, by the memory capability table (`options._MEMORY_CAPABILITIES["c@cpu"]`
  -- storage `'gpu'` only, meaning "on the compute device") before a kernel
  runs.
* **Domain decomposition on CPU tiles** goes through gloo instead of NCCL:
  `ModelParallel` steps each tile with `impl='c'` on the host core, and its
  overlapped forward (`_run_dd_loop_overlapped`) runs the same two phases
  without a CUDA stream -- a host launch has finished when it returns, so the
  cut strips are final before their P2P is posted.  A CPU tile's VRZ takes the
  untiled DD layout (no tiled gradient on the host).
  `test/test_dd_two_rank_cpu.py` runs two ranks with no GPU visible
  (`Acoustic`, `AcousticVRZ3D`, x cut) and asserts the gathered record and the
  summed gradients equal one domain bit for bit.

Residency checks in the shared code say `sweep::on_compute_device(buf)`
(`b.is_cuda() == kCoreIsCuda`), never a literal `is_cuda()`, so the same
`SWEEP_CHECK` reads "expected a tensor on the host" in one core and "on CUDA"
in the other.

## 6. The C API and the Python layer

ABI 4 (dev's 3 was the SEM entries) adds two calls to `capi.h`:
`sweep_entry_available(int)` (−1 out of range, 0 stub, 1 built) and
`const char* sweep_core_device(void)` (`"cuda"` / `"cpu"`).  The loader (`sweep.backend.c.loader`) keeps one `ctypes.CDLL` per
kind in `_libs`, asks each core which device it is rather than trusting its
path, and checks that a second core's entry table equals the first's.  The CUDA
core is loaded `RTLD_GLOBAL` (the pybind shim and `CDLL(None)` lookups bind to
it); the host core `RTLD_LOCAL`, because it exports the same `sweep_*` names
and a second global copy would shadow the first for every by-name resolver.

An entry function (`entries._make_entry`) picks its core *per call* from
`models[0].device`: `acoustic2d_forward` is one Python function serving both
cores, and every model must live on that same device (some equations bind
models with a count check only, so a stray pointer would reach the wrong
core).  Runners do the same at construction; `BoundarySession(device=...)`
takes its core from an explicit device argument (its one in-tree caller is
`ModelParallel`).  A tensor on any other device, or a CPU tensor
when no host core can be had, is refused by name before the core sees a
pointer.  When no tensor names a device (the entry table, a device-less
session), `loader.any_core_lib` takes a core already loaded, then one that is
a file on disk, then the default kind -- so a CPU run on a GPU box never
triggers an nvcc build of a core it will not call.

All of this is the **ctypes layer**.  Under `SWEEP_JIT_FULL=1`, or with an
ahead-of-time `sweep._C` extension, `sweep._C` is the pybind shim, which has
no per-call core choice and reaches the CUDA core only; an explicit `impl='c'`
on a CPU device then falls back to eager with a warning naming the shim.

Resolution (`sweep.propagator.torch._resolve_impl_with_fallback`):

| device | `impl='auto'` | `impl='c'` (explicit) |
|---|---|---|
| CUDA | `c` (CUDA core) | `c` |
| CPU, a host core that builds the equation comes up: shipped (`lib/cpu`, or `SWEEP_CORE_CPU`), a current cached build, or one built here on first use | **`c` (host core)** | `c` (host core) |
| CPU, that build or load fails | `eager` + one `UserWarning` (the first `error:` line, where `build.log` is) | the first call raises |
| CPU, eager options passed (`eager_options=`, or an eager knob at top level / in `backend_options`) | `eager` | `ValueError`, as on CUDA |
| CPU, host staging requested (`storage='cpu'` / `'disk'`, legacy dict included) | `eager` | `NotImplementedError` (below) |
| CPU, unported equation | `eager` | `eager` + `UserWarning` saying why |
| CPU, no host core at hand and no compiler that builds one | `eager`, quietly (as CUDA without nvcc) | `eager` + `UserWarning` saying why |

`'auto'` takes the host core because it is the faster CPU path in nearly
every configuration measured (section 4, "Against eager": 1.2-18x at 8-28
threads; at 4 threads compiled eager still wins a 3-D forward).  It resolves
the core as an explicit `impl='c'` does, and as a CUDA source install
resolves its core: shipped, else a current cached build, else one built here
once (~1-3 min, announced on stderr, the compiler's output kept in
`build.log` beside the build).  What `'auto'` picks is then a property of the
machine -- is there a compiler that builds the core -- not of what ran
before: a cache only saves the build.  A compiler is offered only after it
builds a small OpenMP shared object with the core's flags
(`hostcore._cxx_probe`): Apple clang, a clang++ without libomp and the macOS
linker fail there in a fraction of a second, and `'auto'` stays eager
quietly instead of starting a doomed build in every process.  A build or load
that still fails warns once and the process stays on eager
(`_host_core_ready`); an explicit `impl='c'` raises at its first call
instead.  Processes that start together (torchrun ranks, pytest workers)
build once: the others wait on the build lock and take the core when its
source stamp is this tree's, and build it themselves when it is not (the
holder failed, or built another worktree's sources into the shared drawer).
A build killed outside Python (SIGKILL, a test timeout, a restarted kernel)
leaves its lock file behind, as a CUDA build does.  The file names its holder
(PID, machine, start time), and a waiter on that machine removes it once the
holder has exited; for a holder on another machine it cannot tell, and waits.
On Linux ninja runs under a launcher that has it SIGTERMed when the holder
dies (`jit._ninja_command`): when the holder alone is killed, its compilers
stop before the next build starts, which resumes from the objects already
built (the files being compiled at the kill start over).  A kill of the
whole process group (GNU `timeout`, for one) takes ninja down with it, and
the compilers it had started then finish on their own, beside the next
build.

What would make an explicit `impl='c'` refuse the
construction keeps `'auto'` on eager instead (`_cpu_request_needs_eager`, the
rows above): such a caller ran eager before the host core existed and still
does.  The pybind shim keeps `'auto'` on eager too, quietly.

What a CPU user gets with `'auto'` once the host core is up (the wheel ships
it from 0.3.7; a source install builds it) is what a CUDA user has always got
from it: the compiled
kernels' numerics (exactly as far from eager as the CUDA core is, 1e-5
relative on acoustic2d -- section 8), boundary saving as the default memory
strategy (eager's default is chunked checkpointing), and the compiled
backward's coverage -- no
source-wavelet gradient from the elastic family, DAS, VTI 1st-order, VRZ and
Born kernels (a wavelet requiring grad is refused, `c_refuse_grads`), and
no double backward (the compiled backward is a C call, not differentiable).
`impl='eager'` keeps all of those.  The models must be float32, as on CUDA
(eager converts them).

`hostcore.available()` -- a shipped `lib/cpu` core fits, a cached local build
is current, or a compiler can build one -- answers for both; it loads nothing
and builds no core (finding a compiler runs its probe, once per process).
`sweep.backend.torch.binding.diagnostics()["host_core"]` prints the same
answer with reasons, including why each compiler was turned down.  `binding.is_available()`
keeps its meaning (can `impl='c'` run on **CUDA** here), which is what `'auto'`
keys on for a CUDA device.

## 7. Building and shipping

```bash
python -m sweep.build --cpu            # build into the cache (TORCH_EXTENSIONS_DIR/sweep_C/core_cpu/)
python -m sweep.build --cpu --core     # file it under sweep/lib/cpu/ with its core.json, for a wheel
```

Needs `g++ >= 9` (or `clang++ >= 10`) with OpenMP and `ninja`; no nvcc, no
torch headers.  `CXX` overrides the compiler.  The build stages the sources the
same way the CUDA build does (SHA-256 per file, incremental ninja), writes
`host_unported.cpp` from `HOST_UNPORTED`, and leaves a source stamp
(the tree, the flags, the unported list -- not the compiler) so an unchanged
tree never rebuilds and a core built under `module load gcc/12` stays current
on a node with no g++ on PATH: loading a `.so` needs no compiler.  Which
compiler built a cache is in its `build.ninja`; a shipped core records it in
`core.json`.  Flags (`hostcore.HOST_FLAGS`):

```
-x c++ -std=c++17 -O3 -fPIC -fopenmp -fvisibility=hidden
-ffp-contract=off -ftls-model=initial-exec -DSWEEP_HOST_CORE
-Wno-unknown-pragmas -Wno-attributes
```

linked with `-shared -fopenmp -Wl,-z,defs -Wl,-Bsymbolic-functions`
(`-z defs`: an unresolved symbol is a link error, not a crash at first call).
`-ffp-contract=off` keeps g++ from fusing `a*b+c` into an FMA where nvcc did
not, so the host core's arithmetic is the CUDA core's to the rounding of each
operation (section 8).

Resolution order at load: `SWEEP_CORE_CPU=<path>` → the wheel's `lib/cpu/`
(its `core.json` records ABI, device, machine, compiler, flags and SHA-256) →
the cached local build → build one.  The wheel ships a `lib/cpu` core from
0.3.7 ([Release cores](release_cores.md#host-cpu-core)); a clone or sdist
install has none, so its first CPU construction with `impl='auto'` or `'c'`
builds one into the cache (`python -m sweep.build --cpu` does it ahead of
time, with the compiler's output shown).  In a working tree keep to the cache:
a core filed under `src/sweep/lib/cpu/` (`--cpu --core`) is not stamp-checked,
so it keeps being taken after csrc changes, where the cache rebuilds.

### OpenMP threads

The core never calls `omp_set_num_threads`: it uses what the OpenMP runtime
gives it.  Inside a torch process that runtime is torch's: the core's
`NEEDED libgomp.so.1` resolves to the `libgomp.so.1` torch already loaded
(`torch/lib/` in the pip wheels; the loader matches the SONAME), so there is
one OpenMP runtime and one thread pool, and `torch.set_num_threads(n)` sets
the core's thread count as well as torch's (measured: acoustic2d 1024^2,
300-step forward, 0.39 s at 8, 1.46 s at 2, 2.82 s at 1).  `OMP_NUM_THREADS`,
read at start-up, sets both.  A torch on another runtime (Intel's `libiomp5`) leaves the core
on its own libgomp, which then follows `OMP_NUM_THREADS` only.  Because the
core binds to torch's copy, it must not need a newer `GOMP_*` / `OMP_*`
symbol version than that copy defines; the release script checks
([Release cores](release_cores.md#host-cpu-core)).  On a shared
workstation cap it at 8.

## 8. Numerics contract

* **Host vs CUDA: float32 ULP-level agreement, never bit-exact.**  The CUDA
  core is built `--use_fast_math` (FMA contraction, approximate division and
  square root on the GPU); the host core is not.  Measured on all 19 host-core
  equations with the bitgate's configurations (the 17 of
  `solver_gradient_mode_suite` plus ElasticVRR and ViscoElastic; interior and
  free surface; full, boundary saving, chunked and recursive checkpointing):
  record and every model gradient within **2.8e-5** relative L2 of the CUDA
  core (das3d; most below 1e-5), cosine > 1 − 1e-9, and every equation exactly
  as far from eager on the CPU as the CUDA core is (re-measured on dev
  9f660919, 2026-10-31, and on dev 5ce0acff, 2026-11-07).  Lossy boundary
  storage (fp16 /
  bf16 / int8) differs from the CUDA core by at most 1e-3 -- a few quantisation
  buckets flip on ULP-level input differences -- which is 3-50x below each
  storage's own error against fp32 boundary saving.  The tests
  (`test/test_host_core.py`) pin 1e-5 on acoustic2d (1e-4 for the checkpoint
  modes, next points) and 1e-3 on elastic2d's lossy storage.
* **Denormals are flushed to zero, as on the device.**  The CUDA core runs
  with --use_fast_math, i.e. single-precision denormals (below 1.2e-38) read
  and written as zero; the host core sets FTZ/DAZ (x86 MXCSR, AArch64
  FPCR.FZ) in every thread of every parallel region it opens
  (`sweep::host::DenormalsAsZero`, `host/launch.h`) and restores the previous
  mode on exit, so the calling Python/torch thread is untouched.  Two reasons:
  a decaying wavefield is full of denormals and x86 pays a microcode assist on
  each operation with one (measured 2.3x on acoustic2d, 2.5x on elastic2d,
  1500-step runs); and with IEEE denormals the host differed from the device
  wherever code tests ``x > 0`` to guard ``1/x`` -- the int8 / fp16 boundary
  quantisers did, and wrote NaN into elastic fp16 boundaries before
  `host_block_reciprocal` (kept as the guard for an architecture without the
  mode).  Every parallel loop of the host core goes through
  `sweep::host::parallel_for` or `parallel_for_runs` (the row launcher's),
  which apply the scope; a new host body must use one of them too.
* **Host run-to-run: bit-exact**, with one documented exception -- the DAS
  equations' scatter `atomicAdd` into the receiver trace, whose summation
  order depends on OpenMP scheduling (the same configurations the bitgate
  records a noise floor for on CUDA).
* **Checkpoint modes are the forward, bit for bit.**  The replay re-runs the
  forward kernel, which reads the PML memory psi at neighbouring cells; its
  state carries the forward's psi double-buffer (`CKPT_STATE_COUNT` 9 in 2-D,
  12 in 3-D, rotated with `swap_pml`).  It used to replay from the in-place
  psi layout -- a cell's update raced its neighbours' reads in the same launch,
  which CUDA's block order hid and the host's OpenMP row chunks did not (the
  ckpt gradient moved with `OMP_NUM_THREADS`, ~1e-6).  Fixed for both cores;
  `TestCheckpointReplay` pins thread-count independence and equality with
  full mode.
* **3-D int8 / fp16 boundary restore: one writer per cell.**  The band-only
  restore (`boundary_kernel3d_compact`) visits a cell in two or three bands (an
  edge or corner of the strip, or on a thin axis a cell both opposite bands
  cover) once per band.  int8 and scaled fp16 quantize each face with its own
  block scales, so those copies differ, and the cell kept whichever virtual
  thread ran last: block order on the GPU, `OMP_NUM_THREADS` on the host
  (acoustic3d int8 ~2e-5, elastic3d int8 up to ~2e-3 relative).  The lossy
  restore now goes through `boundary_kernel3d_compact_owned`, which writes such
  a cell only from the band `boundary_kernel3d` writes last.  Fixed for both
  cores; `TestLossyBoundary3DRestore` pins thread-count independence and
  equality with `SWEEP_BOUNDARY_NO_COMPACT=1`.
* **Eager vs host core**: as eager vs CUDA core -- the solver gradient suite's
  tolerances (`test/backend_gradient_matrix.py` has a `c-cpu` column).

## 9. Rules for kernels and equations

1. Launch through `SWEEP_LAUNCH` / `SWEEP_LAUNCH_ON` / `SWEEP_LAUNCH_BY_ORDER`.
   Never `<<<>>>`.
2. A kernel is one virtual thread's work: decode indices, bounds-return,
   compute, write own cell.  No `__shared__`, no `__syncthreads`, no warp
   intrinsics.  If you need them, write the CUDA kernel and a host body under
   `#if defined(SWEEP_HOST_CORE)` in the same `.cu`, and gate them against
   each other (`boundarysaver.cu` is the pattern).  Two virtual threads may
   write the same cell only with the same bits (the fp32 / bf16 compact
   restores); otherwise give the cell one owner
   (`boundary_kernel3d_compact_owned` is the pattern) -- an unordered duplicate
   write resolves by block order on the GPU and by `OMP_NUM_THREADS` on the
   host.
3. Residency: `sweep::on_compute_device(buf)`, not `buf.is_cuda()`.
4. A CUDA library dependency (cuFFT, cuBLAS, NCCL), or an equation not
   ported yet, puts its directory and `C_NAME`s in `hostcore.HOST_UNPORTED` --
   that one table produces the stubs, the availability answers and the
   resolver's fallback message (the stub generator checks it against the
   header).
   Prefer a host-side replacement under `SWEEP_HOST_CORE` when one is small.
5. Allocate nothing in C++ (that rule predates the host core; it is what makes
   the runtime stand-in short).
6. Rebuild both cores after touching `csrc/` (`python -m sweep.build` and
   `--cpu`), run the bitgate for the CUDA side and `test/test_host_core.py`
   for the host.
7. Splitting a kernel for `launch_rows` must not change the CUDA core's speed:
   compare `cuobjdump -res-usage` (registers, stack, local) and the SASS of the
   affected kernels before and after, and time them on an idle GPU when the
   SASS differs.  Write each run-time test the way the kernel always wrote it
   (eager or term-by-term bounds, `||` or `&&` form) and put the path-specific
   variants behind `if constexpr`: nvcc forms its code from the expression as
   written, and a box returned eagerly instead of tested lazily was enough to
   move the fused adjoint from 64 to 66 registers.  Compare opcodes with
   their modifiers: a load that loses `.CONSTANT` keeps its opcode.
8. A row-launcher port: the box is a subset of the cells where the Any path
   takes the branch the interior path hard-codes (check it against every
   geometry feature, not only the tests'); build the launcher's `call` lambda
   above the `with_*` dispatch; verify bit-identity against the pre-port core
   and against `SWEEP_HOST_ROWS=0`, and that the static-order row loops
   vectorise (`-fdump-tree-vect-details`).

## 10. Verification

| Check | What it pins |
|---|---|
| `test/test_host_core.py` | the host core is a `cpu` core at ABI 4 with the CUDA core's table; unavailable == unported; stubs throw by name; acoustic2d resolves to `c` on a CPU device, matches eager CPU and the CUDA core in all four memory modes, is run-to-run bit-exact; host staging is refused, `'auto'` takes the core explicit `c` runs on (bit-equal), builds it when none is at hand (output off the terminal) and stays eager with eager options, an unported equation falls back with the reason |
| `test/test_impl_resolution.py` | the resolution table above, with and without a host core; through a fake compiler, the compiler probe, a failed first build (one warning, `build.log`) and the build lock's waiter |
| `test/test_build_lock.py` | the build lock: a SIGKILLed holder's lock is removed and the next staging / CUDA core / host core build completes; a live holder, an unnamed one (older sweep) and one on another machine are waited on; waiters still take turns; the build stops when its holder dies |
| `test/test_backend_c.py` | the two-core loader and the per-call core choice, against fakes; `sweep._C` answers dunder lookups (pytest's collection) without loading a core |
| `test/backend_gradient_matrix.py --backends c-cpu ...` | gradients of every case vs eager and the CUDA core |
| `gate/run_gate.sh A/B/C/T` | the CUDA core is byte-for-byte unaffected by the macro migration |
| `test/test_host_core.py::TestRowLauncher` | the row launcher is bit-identical to `SWEEP_HOST_ROWS=0` (acoustic2d, full and boundary saving, orders 2/4/8, two shots, topography) |
| `test/test_host_core.py::TestLossyBoundary3DRestore` | 3-D int8 / fp16 boundary saving (acoustic3d both dtypes, acoustic_vti_1st_3d int8 for the multi-field restore): gradient bit-identical at 1 and 4 threads and with `SWEEP_BOUNDARY_NO_COMPACT=1` |
| `test/test_dd_two_rank_cpu.py` | `ModelParallel` on two CPU tiles (gloo, no GPU visible, overlapped forward): gathered record and summed gradients equal one domain bit for bit (`Acoustic`, `AcousticVRZ3D`) |
