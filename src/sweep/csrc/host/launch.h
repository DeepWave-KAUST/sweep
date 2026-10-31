#pragma once
// host/launch.h -- the host core's kernel launcher behind SWEEP_LAUNCH
// (launch/launch.h): the virtual CUDA grid, executed by OpenMP threads.
//
// A kernel in this tree is "decode (ix[, iy], iz, b) from blockIdx /
// threadIdx / blockDim; bounds early-return; per-cell arithmetic; write own
// cell".  Running it once per (block, thread) of the SAME grid x block
// decomposition the CUDA core launches with reproduces its index space
// exactly -- Geom kernels put the batch in blockIdx.x and the point index in
// blockIdx.y * blockDim.x + threadIdx.x, Wave3D folds the batch into grid.z,
// grid-stride kernels read gridDim -- without touching a single kernel body.
// Blocks are distributed over OpenMP threads (static schedule: a thread gets
// a contiguous run of blocks, i.e. a band of rows); the threads of one block
// run sequentially, x innermost, in the calling OpenMP thread.  Each virtual
// thread is a real function call, so a kernel's early ``return`` ends that
// thread only.
//
// Cost (2 x Xeon 8176, 2-D 8th-order acoustic step, see docs/dev/host_core.md):
// about 1.3-1.5x a scalar hand-written loop; 4-6x a SIMD-vectorised one,
// because the CUDA-style guards keep the compiler from vectorising.  That is
// the price of zero kernel edits.  A kernel that needs more is split per
// launch/cell_path.h -- guards in the kernel, the body in a cell function
// both cores share -- and its by-order table entry on the host goes through
// launch_rows below, which runs the interior of each row as a SIMD loop over
// that same cell body: never a second copy of the arithmetic.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <type_traits>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#include "include/cuda_runtime.h"   // dim3, the thread-local index registers
#include "../cuda/launch/cell_path.h"   // CellBox

namespace sweep {
namespace host {

// Denormals read and written as zero in the calling thread for the scope's
// lifetime; the previous mode is restored on exit.
//
// The device does this already: the CUDA core is built --use_fast_math,
// which flushes single-precision denormals to zero.  On the host they are
// IEEE by default, and on x86 every operation with a denormal operand or
// result takes a microcode assist -- tens to a hundred times a normal one.
// A decaying wavefield (PML, the wake of a wavefront, any late time step)
// is full of them: measured 2.3x (acoustic2d) and 2.5x (elastic2d) on a
// 1500-step run.  Denormals are below 1.2e-38, twenty-odd orders under any
// wavefield amplitude, so flushing them changes no result anyone can see --
// and it makes the host do what the device does (the int8/fp16 quantisers'
// "max > 0" guard is the one place that difference had been visible).
//
// Set per thread inside every parallel region the core opens (OpenMP
// threads do not share this mode) and restored on exit, so the caller's
// thread -- Python, torch -- leaves a core call in the mode it entered.
struct DenormalsAsZero {
#if defined(__x86_64__) || defined(__i386__)
    unsigned saved_;
    DenormalsAsZero() : saved_(_mm_getcsr()) { _mm_setcsr(saved_ | kFtzDaz); }
    ~DenormalsAsZero() { _mm_setcsr(saved_); }
    static constexpr unsigned kFtzDaz = 0x8040u;   // MXCSR: FTZ (bit 15) | DAZ (bit 6)
#elif defined(__aarch64__)
    unsigned long saved_;
    DenormalsAsZero()
    {
        __asm__ __volatile__("mrs %0, fpcr" : "=r"(saved_));
        __asm__ __volatile__("msr fpcr, %0" ::"r"(saved_ | kFz));
    }
    ~DenormalsAsZero() { __asm__ __volatile__("msr fpcr, %0" ::"r"(saved_)); }
    static constexpr unsigned long kFz = 1ul << 24;   // FPCR.FZ: inputs and outputs
#else
    // Another architecture keeps IEEE denormals: correct, only slower.
#endif
    DenormalsAsZero(const DenormalsAsZero&) = delete;
    DenormalsAsZero& operator=(const DenormalsAsZero&) = delete;
};

// ``fn(i)`` for i in [0, n) over the OpenMP threads (static schedule: a thread
// takes a contiguous run), every thread -- the caller's included -- inside a
// DenormalsAsZero scope.  With parallel_for_runs (the row launcher's), the
// host core's parallel loops: the kernel launcher and the host bodies
// (boundarysaver.cu) go through one of the two.
template <class Fn>
inline void parallel_for(long long n, Fn&& fn)
{
#pragma omp parallel
    {
        const DenormalsAsZero daz;
#pragma omp for schedule(static)
        for (long long i = 0; i < n; ++i)
            fn(i);
    }
}

// Number of OpenMP threads a launch spreads its blocks over (OMP_NUM_THREADS
// or the runtime default; 1 when the core was built without OpenMP).
inline int num_threads()
{
#if defined(_OPENMP)
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// One run of parallel_for_runs: its chunks [front, back), packed in one word
// (front in the low 32 bits) so the owner and a thief claim a chunk with one
// compare-and-swap each.  A cache line per run.
struct alignas(64) ChunkRun {
    std::atomic<unsigned long long> front_back;
};

// The calling thread's runs, kept from launch to launch.
inline ChunkRun* chunk_runs(int count)
{
    thread_local std::unique_ptr<ChunkRun[]> runs;
    thread_local int capacity = 0;
    if (capacity < count) {
        runs.reset(new ChunkRun[count]);
        capacity = count;
    }
    return runs.get();
}

// ``fn(i)`` for i in [0, n) in contiguous runs, one per OpenMP thread (run t
// is [n t / T, n (t + 1) / T)), every thread inside a DenormalsAsZero scope.
// A run is cut into chunks; its thread takes them from the front, and a
// thread done with its own run takes chunks from the BACK of the others'.
// launch_rows' schedule.  Contiguous, because neighbouring rows share their
// stencil halos and a balanced launch then gives every row to the same thread
// launch after launch, so its rows stay in that thread's caches and its NUMA
// node's memory: handed out dynamically two rows at a time (as they were),
// 2-D row launches ran 2-7x slower at 32-64 threads on a 2-socket EPYC 7702.
// Stealing, because a full-grid kernel's PML band rows run the scalar kernel
// end to end and cost several interior rows: in one static run per thread
// they all land on the first and the last thread (elastic3d forward 1.65x
// slower at 8 threads on a Xeon 8176).  Thieves take from the back so the
// owner's walk keeps its locality.  The runs are set before the parallel
// region, which so needs no barrier of its own.  Which thread runs which i
// changes no result: launch_rows' rows write disjoint cells.
template <class Fn>
inline void parallel_for_runs(long long n, Fn&& fn)
{
    if (n <= 0) return;
#if defined(_OPENMP)
    const int nt = num_threads();
    ChunkRun* runs = chunk_runs(nt);
    const long long per = n / nt > 0 ? n / nt : 1;
    const long long chunk = per / 32 > 1 ? per / 32 : 1;   // chunks per run: ~32
    for (int r = 0; r < nt; ++r) {
        const long long lo = n * r / nt, hi = n * (r + 1) / nt;
        const auto nchunks = static_cast<unsigned long long>((hi - lo + chunk - 1) / chunk);
        runs[r].front_back.store(nchunks << 32, std::memory_order_relaxed);
    }
    auto run_chunk = [&](int r, unsigned long long c) {
        const long long lo = n * r / nt, hi = n * (r + 1) / nt;
        const long long i0 = lo + static_cast<long long>(c) * chunk;
        const long long i1 = i0 + chunk < hi ? i0 + chunk : hi;
        for (long long i = i0; i < i1; ++i)
            fn(i);
    };
    // Claim one chunk of run r, from its front or its back; false when it is empty.
    auto claim = [&](int r, bool from_front, unsigned long long& c) {
        std::atomic<unsigned long long>& fb = runs[r].front_back;
        unsigned long long cur = fb.load(std::memory_order_relaxed);
        for (;;) {
            const unsigned long long front = cur & 0xffffffffULL, back = cur >> 32;
            if (front >= back) return false;
            const unsigned long long next = from_front ? (back << 32) | (front + 1)
                                                       : ((back - 1) << 32) | front;
            if (fb.compare_exchange_weak(cur, next, std::memory_order_relaxed)) {
                c = from_front ? front : back - 1;
                return true;
            }
        }
    };
#pragma omp parallel num_threads(nt)
    {
        const DenormalsAsZero daz;
        const int t = omp_get_thread_num(), got = omp_get_num_threads();
        if (got != nt) {
            // a smaller team than asked for (OMP_DYNAMIC, a thread limit): one
            // contiguous run per thread of it, no stealing
            for (long long i = n * t / got; i < n * (t + 1) / got; ++i)
                fn(i);
        } else {
            unsigned long long c;
            while (claim(t, true, c))
                run_chunk(t, c);
            for (int k = 1; k < nt; ++k) {
                const int v = (t + k) % nt;
                while (claim(v, false, c))
                    run_chunk(v, c);
            }
        }
    }
#else
    const DenormalsAsZero daz;
    for (long long i = 0; i < n; ++i)
        fn(i);
#endif
}

// ``call`` is the generic lambda SWEEP_LAUNCH wraps around the kernel
// expression -- ``[&](auto&&... a) { kern(a...); }`` -- so the kernel is
// invoked by NAME at the call site: default arguments (set_boundary_zeros'
// cut_mask, the boundary kernels' tangent_pad) apply exactly as they do under
// ``<<<>>>``, and the compiler may inline the body.  ``args`` are the launch
// arguments, evaluated once here like CUDA evaluates them once per launch.
//
// The per-block work sits in ``run_block`` so that the inlining the whole
// scheme depends on actually happens.  GCC outlines the body of an OpenMP
// ``parallel for`` into a function of its own, and ``flatten`` on a function
// that is itself inlined away before the flatten pass runs does nothing -- so
// ``run_block`` is kept as a real function (noinline: one call per BLOCK, not
// per cell) and flattened: every call inside it -- the lambda, then the kernel
// whenever its definition is visible, i.e. every header-defined template
// kernel -- is inlined regardless of the size heuristics that would otherwise
// leave a 300-instruction stencil as a call.  Once the kernel is inlined, its
// by-value parameters (SolverContext 240 B, the pointer bundles, ~600 B per
// acoustic launch) are read where they stand instead of being copied once per
// virtual thread -- the difference between a stencil step that spends its
// time on arithmetic and one that spends it on memcpy.  A kernel defined in
// another TU (common.cu's point kernels) stays a call; those launches are
// cheap anyway.
template <class Call, class... Args>
__attribute__((noinline, flatten)) void run_block(dim3 grid, dim3 block, long long bl,
                                                   Call& call, Args&... args)
{
    gridDim = grid;
    blockDim = block;
    blockIdx.x = static_cast<unsigned>(bl % grid.x);
    blockIdx.y = static_cast<unsigned>((bl / grid.x) % grid.y);
    blockIdx.z = static_cast<unsigned>(bl / (static_cast<unsigned long long>(grid.x) * grid.y));
    for (unsigned tz = 0; tz < block.z; ++tz) {
        for (unsigned ty = 0; ty < block.y; ++ty) {
            for (unsigned tx = 0; tx < block.x; ++tx) {
                threadIdx.x = tx;
                threadIdx.y = ty;
                threadIdx.z = tz;
                call(args...);
            }
        }
    }
}

template <class Call, class... Args>
inline void launch(dim3 grid, dim3 block, Call&& call, Args&&... args)
{
    const unsigned long long nblocks = static_cast<unsigned long long>(grid.x) * grid.y * grid.z;
    const unsigned long long nthreads = static_cast<unsigned long long>(block.x) * block.y * block.z;
    if (nblocks == 0 || nthreads == 0) return;
    parallel_for(static_cast<long long>(nblocks),
                 [&](long long bl) { run_block(grid, block, bl, call, args...); });
}

// ---------------------------------------------------------------------------
// Row launcher for a kernel split per launch/cell_path.h.
//
// Same index space as launch() -- every virtual thread (gx, gy, gz) of
// grid x block, gx = blockIdx.x * blockDim.x + threadIdx.x and likewise for
// y and z -- walked by rows: the rows (gy, gz) go over the OpenMP threads
// (in contiguous runs, see parallel_for_runs), and in a row the virtual threads
// [lo, hi) that ``span(gy, gz)`` returns go to ``cell(gx, gy, gz)`` -- the
// kernel's body with an interior Path -- in one ``omp simd`` loop; every
// other virtual thread runs the kernel itself through ``call``, as in
// launch().  The kernel's own decode turns (gx, gy, gz) into its cell (a
// Wave2D kernel: ix = gx [+ x_base], iz = gy, b = gz; a Wave3D kernel:
// iy = gy, b = gz / nz, iz = gz % nz), so ``span`` and ``cell`` use it too.
//
// ``omp simd`` lets the compiler run consecutive cells of the row at once,
// in SIMD lanes.  That is what the device does with the same cells, in no
// particular order, so a kernel it is wrong for is a race on the device
// first.  The caller owns the other half of the contract: ``span`` returns
// only virtual threads the kernel would run through its interior path.
//
// SWEEP_HOST_ROWS=0 runs every virtual thread through the kernel instead
// (launch_through below; prototype switch, for A/B).
inline bool rows_enabled()
{
    static const bool on = [] {
        const char* e = std::getenv("SWEEP_HOST_ROWS");
        return !(e != nullptr && e[0] == '0');
    }();
    return on;
}

// The virtual threads [lo, hi) of one row that take the interior path.
struct RowSpan {
    int lo = 0, hi = 0;
};

// The span of a Wave2D kernel (gy = iz, gz = b) whose interior is ``box`` in
// virtual-thread coordinates: every batch alike.
inline auto box_rows(const CellBox& box)
{
    return [box](int gy, int /*gz*/) {
        return (gy >= box.z0 && gy < box.z1) ? RowSpan{box.x0, box.x1} : RowSpan{};
    };
}

// The index registers of row (gy, gz) of grid x block: gridDim, blockDim and
// the y/z ones; run_through counts the x ones along the row.
inline void enter_row(dim3 grid, dim3 block, int gy, int gz)
{
    gridDim = grid;
    blockDim = block;
    blockIdx.y = static_cast<unsigned>(gy) / block.y;
    threadIdx.y = static_cast<unsigned>(gy) % block.y;
    blockIdx.z = static_cast<unsigned>(gz) / block.z;
    threadIdx.z = static_cast<unsigned>(gz) % block.z;
}

// Virtual threads [from, to) of the current row (enter_row) through the
// kernel, the x index registers counted along rather than divided out per
// cell.  Like run_block: noinline (one call per row segment, not per cell)
// and flattened, so the kernel is inlined into this loop and its by-value
// parameters are read in place.
//
// Templated on the kernel call and the launch arguments only -- not on the
// span or the cell -- so there is ONE copy of the kernel's Any body per
// (kernel, Order), whatever the number of interior paths.  A launcher that
// dispatches optional outputs / flags / a mechanism count runs launch_rows
// once per path; with this loop inlined into run_row the kernel body was
// compiled twice per path (before and after the span), plus once more per
// path for the SWEEP_HOST_ROWS=0 fallback: 735 copies of 50 distinct
// kernel bodies in elastic2d, >80% of the host core's code and compile time.
// The sharing needs the launcher's ``call`` to be ONE closure type for all
// its paths: write ``auto call = [&](auto&&... a) { kern<Order>(a...); };``
// once, above the with_optional / with_flags dispatch, never inside it.
template <class Call, class... Args>
__attribute__((noinline, flatten)) void run_through(dim3 block, int from, int to,
                                                     Call& call, Args&... args)
{
    unsigned bx = static_cast<unsigned>(from) / block.x;
    unsigned tx = static_cast<unsigned>(from) % block.x;
    for (int gx = from; gx < to; ++gx) {
        blockIdx.x = bx;
        threadIdx.x = tx;
        call(args...);
        if (++tx == block.x) {
            tx = 0;
            ++bx;
        }
    }
}

// One whole row through the kernel (launch_through).
template <class Call, class... Args>
__attribute__((noinline)) void run_row_kernel(dim3 grid, dim3 block, long long row,
                                               Call& call, Args&... args)
{
    const long long ny = static_cast<long long>(grid.y) * block.y;
    const int gy = static_cast<int>(row % ny);
    const int gz = static_cast<int>(row / ny);
    enter_row(grid, block, gy, gz);
    run_through(block, 0, static_cast<int>(grid.x * block.x), call, args...);
}

template <class Span, class Cell, class Call, class... Args>
__attribute__((noinline, flatten)) void run_row(dim3 grid, dim3 block, long long row,
                                                 const Span& span, Cell cell,
                                                 Call& call, Args&... args)
{
    // ``cell`` by value: a by-value capture is then a local of this row, and
    // the compiler can see that no store through the kernel's pointers
    // reaches it -- otherwise a float parameter such as dt is reloaded after
    // every float store and the loop does not vectorise.
    const long long ny = static_cast<long long>(grid.y) * block.y;
    const int gy = static_cast<int>(row % ny);
    const int gz = static_cast<int>(row / ny);
    const int width = static_cast<int>(grid.x * block.x);
    const RowSpan s = span(gy, gz);
    const int lo = s.lo < 0 ? 0 : (s.lo > width ? width : s.lo);
    const int hi = s.hi < lo ? lo : (s.hi > width ? width : s.hi);
    enter_row(grid, block, gy, gz);
    // The band cells before and after the span through the kernel, in the
    // shared run_through segment; the span in this row's own SIMD loop.
    run_through(block, 0, lo, call, args...);
#pragma omp simd
    for (int gx = lo; gx < hi; ++gx)
        cell(gx, gy, gz);
    run_through(block, hi, width, call, args...);
}

// ``f(path)`` with the interior CellPath that matches which of the kernel's
// optional outputs ``ps...`` are present (output k = the k-th pointer;
// launch/cell_path.h), as a std::integral_constant, so the cell body is
// instantiated for exactly that combination.  An optional output's
// ``if (out != nullptr)`` is the branch that would otherwise stay in a row of
// interior cells -- a conditional store, in a loop body far over the size GCC
// unswitches on its own.
template <unsigned K, unsigned Mask, class F>
inline void dispatch_optionals(F& f)
{
    f(std::integral_constant<CellPath, interior_with_outputs(Mask)>{});
}

template <unsigned K, unsigned Mask, class F, class T, class... Rest>
inline void dispatch_optionals(F& f, const T* p, const Rest*... rest)
{
    if (p != nullptr)
        dispatch_optionals<K + 1, Mask | (1u << K)>(f, rest...);
    else
        dispatch_optionals<K + 1, Mask>(f, rest...);
}

template <class F, class... T>
inline void with_optionals(F&& f, const T*... ps)
{
    dispatch_optionals<0u, 0u>(f, ps...);
}

// with_flags(f, b0, b1, ...) calls f(std::integral_constant<unsigned, M>)
// with bit F of M = bF: launch-uniform bools made compile-time, for
// add_flags(path, M) (launch/cell_path.h).
template <unsigned F, unsigned Mask, class Fn>
inline void dispatch_flags(Fn& f)
{
    f(std::integral_constant<unsigned, Mask>{});
}

template <unsigned F, unsigned Mask, class Fn, class... B>
inline void dispatch_flags(Fn& f, bool v, B... rest)
{
    if (v)
        dispatch_flags<F + 1, Mask | (1u << F)>(f, rest...);
    else
        dispatch_flags<F + 1, Mask>(f, rest...);
}

template <class Fn, class... B>
inline void with_flags(Fn&& f, B... flags)
{
    dispatch_flags<0u, 0u>(f, static_cast<bool>(flags)...);
}

// The one-output spelling: Interior or InteriorOut.
template <class T, class F>
inline void with_optional(const T* p, F&& f)
{
    with_optionals(f, p);
}

// Every virtual thread of grid x block through the kernel, walked by rows like
// launch_rows: for a kernel launch_rows runs (its virtual threads write
// disjoint cells, so the order is free), the same result as launch() -- and,
// through run_through, the same copy of the kernel body as launch_rows' band
// cells, for the same ``call``: the SWEEP_HOST_ROWS=0 fallback, or a launcher
// case without an interior path (visco_elastic2d's mechanism count outside
// 1..4), adds no copy of its own.
template <class Call, class... Args>
inline void launch_through(dim3 grid, dim3 block, Call&& call, Args&&... args)
{
    const long long nrows = static_cast<long long>(grid.y) * block.y * grid.z * block.z;
    if (nrows == 0 || grid.x == 0 || block.x == 0) return;
    parallel_for_runs(nrows, [&](long long row) { run_row_kernel(grid, block, row, call, args...); });
}

template <class Span, class Cell, class Call, class... Args>
inline void launch_rows(dim3 grid, dim3 block, Span&& span, Cell&& cell, Call&& call,
                        Args&&... args)
{
    if (!rows_enabled()) {
        launch_through(grid, block, call, args...);
        return;
    }
    const long long nrows = static_cast<long long>(grid.y) * block.y * grid.z * block.z;
    if (nrows == 0 || grid.x == 0 || block.x == 0) return;
    // Contiguous runs of rows per thread, the tail balanced by stealing
    // (parallel_for_runs).  The row index is gy + ny * gz: a Wave3D run walks
    // the y rows of a z plane, then the next plane, so it keeps the y and z
    // halos it just loaded.
    parallel_for_runs(nrows, [&](long long row) { run_row(grid, block, row, span, cell, call, args...); });
}

// The <<<grid, block, shmem, stream>>> form.  Dynamic shared memory does not
// exist on the host (the kernels that needed __shared__ have host bodies), so
// a non-zero request is a programming error, not a silent miscompute.
template <class Call, class... Args>
inline void launch_on(dim3 grid, dim3 block, size_t shmem, cudaStream_t /*stream*/, Call&& call, Args&&... args)
{
    if (shmem != 0)
        throw std::runtime_error("host core: a kernel asked for dynamic shared memory; "
                                 "give it a host body (see boundarysaver.cu)");
    launch(grid, block, call, args...);
}

}  // namespace host
}  // namespace sweep
