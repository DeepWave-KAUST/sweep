#pragma once
// host/include/cuda_runtime.h -- the host core's stand-in for CUDA's runtime
// header.  It exists so that the SAME .cu sources that nvcc compiles into
// libsweep_core.so (the CUDA core) compile with g++ into the host core: one
// source of truth for every kernel body, every driver skeleton and every C API
// entry.  The previous CPU engine was a second, hand-written copy of each
// driver (csrc/cpu, ~20k lines, removed at af58082b) and diverged from the CUDA
// core in exactly the ways a copy diverges -- a gradient scaling here, a sign
// there, a buffer layout nobody updated.  Nothing in this directory implements
// physics; it only teaches a C++ compiler the dozen CUDA spellings the tree
// uses.
//
// HOW IT IS SELECTED.  The host build passes ``-DSWEEP_HOST_CORE -isystem
// csrc/host/include`` and compiles every .cu as C++ (``-x c++``); the tree's
// ``#include <cuda_runtime.h>`` then resolves here.  nvcc never sees this
// directory.  A TU compiled without the define must not pick it up by accident.
//
// EXECUTION MODEL.  A kernel launch (SWEEP_LAUNCH, launch/launch.h) runs the
// kernel once per virtual thread of the same grid x block decomposition CUDA
// would use, blocks spread over OpenMP threads, the threads of a block
// sequentially.  ``threadIdx`` / ``blockIdx`` / ``blockDim`` / ``gridDim`` are
// thread-local variables the launcher sets before each call; kernel bodies read
// them exactly as they read CUDA's built-ins.  Every CUDA kernel in this tree
// writes only its own cell (reads of neighbours come from the previous step's
// buffers), so "one virtual thread at a time per block" is race-free wherever
// the CUDA kernel is.  Block-cooperative constructs (``__shared__``,
// ``__syncthreads``) are deliberately NOT provided: the two kernels that used
// them (the int8/fp16 quantizers in boundarysaver.cu) carry host bodies under
// SWEEP_HOST_CORE, and a third would fail to compile here rather than run wrong.
//
// SYNCHRONY.  Streams and events are tokens: every copy and every launch
// completes before the call returns, so "record / wait / synchronize" are
// no-ops and ``cudaLaunchHostFunc`` runs its callback immediately.  Host and
// disk boundary staging are refused by the host core's entry prologue (the
// compute device IS the host), so the ring/event machinery compiles but never
// runs with a transfer in flight.
//
// NUMERICS.  No fast-math here: the host core is compiled -ffp-contract=off, so
// the ``__f*_rn`` intrinsics map to plain IEEE operations (one rounding each),
// which is what they mean on the device too.  Host and CUDA results differ at
// the ULP level by construction (nvcc --use_fast_math contracts FMAs); the
// acceptance criteria compare the host core against eager and check gradient
// magnitude/cosine against the CUDA core, never bits.
#if !defined(SWEEP_HOST_CORE)
#error "csrc/host/include is the host core's CUDA stand-in; it is only valid with -DSWEEP_HOST_CORE (nvcc builds must not have it on the include path)"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(_OPENMP)
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// Function-space qualifiers.  All code is host code here.
// ---------------------------------------------------------------------------
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(...)
#define __noinline__ __attribute__((noinline))
// __restrict__ is a GCC/Clang keyword already.  __shared__ and the warp
// intrinsics are intentionally undefined (see the header comment);
// __syncthreads is declared deleted so a kernel that reaches for it fails at
// its own line instead of compiling into a silent no-op.
void __syncthreads() = delete;

// ---------------------------------------------------------------------------
// dim3 and the index registers.
// ---------------------------------------------------------------------------
struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

// C++17 inline thread_local: one definition across all TUs AND a visible
// constant initializer, so the compiler emits a direct %fs: access instead of a
// TLS-init wrapper call per read (an ``extern thread_local`` would cost six
// wrapper calls per kernel invocation).  Compile the host core with
// -ftls-model=initial-exec so a -fPIC shared object keeps that access direct.
inline thread_local dim3 threadIdx{0, 0, 0};
inline thread_local dim3 blockIdx{0, 0, 0};
inline thread_local dim3 blockDim{1, 1, 1};
inline thread_local dim3 gridDim{1, 1, 1};

// ---------------------------------------------------------------------------
// Error codes, memcpy kinds, flags.
// ---------------------------------------------------------------------------
enum cudaError_t { cudaSuccess = 0, cudaErrorInvalidValue = 1 };

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

constexpr unsigned cudaStreamDefault = 0;
constexpr unsigned cudaStreamNonBlocking = 1;
constexpr unsigned cudaEventDefault = 0;
constexpr unsigned cudaEventDisableTiming = 2;

// Streams and events are opaque tokens (see the SYNCHRONY note above):
// create() hands out a distinct non-null object so "nullptr means disabled"
// tests in the tree keep their meaning; record() stamps the clock so the
// env-gated profiling paths still print something sensible.
struct CUstream_st { unsigned flags; };
struct CUevent_st { std::chrono::steady_clock::time_point t; unsigned flags; };
typedef CUstream_st* cudaStream_t;
typedef CUevent_st* cudaEvent_t;

inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t e)
{
    return e == cudaSuccess ? "no error" : "host core: invalid value";
}

// One host "device", ordinal 0.  A Buf on the host reports device_index() ==
// -1, so sweep::DeviceGuard never reaches these; they exist for completeness.
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

// ---------------------------------------------------------------------------
// Memory operations: synchronous memcpy/memset.  The kind is irrelevant --
// every pointer the host core sees is host memory (the core's binding checks
// enforce that through on_compute_device()).
// ---------------------------------------------------------------------------
namespace sweep_host_mem {
// The drivers clear and copy whole wavefields every time step (DAS scratch
// grids, checkpoint state, recursive-replay scratch); once the stencils run
// threaded and vectorised, a single-threaded memset there is the Amdahl
// wall.  Large calls made outside a parallel region are split into chunks
// over the OpenMP threads; the bytes written are the same.
constexpr size_t kParallelBytes = size_t(1) << 20;
constexpr size_t kChunkBytes = size_t(1) << 18;

inline bool parallel_worth(size_t n)
{
#if defined(_OPENMP)
    return n >= kParallelBytes && !omp_in_parallel() && omp_get_max_threads() > 1;
#else
    (void)n;
    return false;
#endif
}

template <class Fn>
inline void chunked(size_t n, Fn&& fn)
{
    const long long chunks = static_cast<long long>((n + kChunkBytes - 1) / kChunkBytes);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (long long c = 0; c < chunks; ++c) {
        const size_t off = static_cast<size_t>(c) * kChunkBytes;
        fn(off, std::min(kChunkBytes, n - off));
    }
}
}  // namespace sweep_host_mem

// memmove, not memcpy: a device-to-device copy between two views of one ring
// buffer may overlap, which cudaMemcpy handles and memcpy would not.  Only a
// copy whose two ranges are disjoint is split over threads.
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind)
{
    if (!n) return cudaSuccess;
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    const uintptr_t da = reinterpret_cast<uintptr_t>(d), sa = reinterpret_cast<uintptr_t>(s);
    const bool overlap = da < sa + n && sa < da + n;
    if (overlap || !sweep_host_mem::parallel_worth(n)) {
        std::memmove(dst, src, n);
        return cudaSuccess;
    }
    sweep_host_mem::chunked(n, [&](size_t off, size_t len) { std::memcpy(d + off, s + off, len); });
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind k, cudaStream_t = nullptr)
{
    return cudaMemcpy(dst, src, n, k);
}
inline cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, cudaMemcpyKind, cudaStream_t = nullptr)
{
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    for (size_t h = 0; h < height; ++h)
        if (width) std::memmove(d + h * dpitch, s + h * spitch, width);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t n)
{
    if (!n) return cudaSuccess;
    if (!sweep_host_mem::parallel_worth(n)) {
        std::memset(p, v, n);
        return cudaSuccess;
    }
    char* d = static_cast<char*>(p);
    sweep_host_mem::chunked(n, [&](size_t off, size_t len) { std::memset(d + off, v, len); });
    return cudaSuccess;
}
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = nullptr)
{
    return cudaMemset(p, v, n);
}

// ---------------------------------------------------------------------------
// Stream and event calls: all no-ops except the bookkeeping named above.
// ---------------------------------------------------------------------------
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = new CUstream_st{cudaStreamDefault}; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned f) { *s = new CUstream_st{f}; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t s) { delete s; return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0) { return cudaSuccess; }

inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new CUevent_st{{}, cudaEventDefault}; return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned f) { *e = new CUevent_st{{}, f}; return cudaSuccess; }
// A null handle is CUDA's "invalid resource handle": an error code, never a
// dereference.
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = nullptr)
{
    if (e == nullptr) return cudaErrorInvalidValue;
    e->t = std::chrono::steady_clock::now();
    return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t e) { return e ? cudaSuccess : cudaErrorInvalidValue; }
inline cudaError_t cudaEventQuery(cudaEvent_t e) { return e ? cudaSuccess : cudaErrorInvalidValue; }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b)
{
    if (a == nullptr || b == nullptr) return cudaErrorInvalidValue;
    *ms = std::chrono::duration<float, std::milli>(b->t - a->t).count();
    return cudaSuccess;
}

// A stream callback runs immediately: nothing is ever in flight on the host.
typedef void (*cudaHostFn_t)(void* userData);
inline cudaError_t cudaLaunchHostFunc(cudaStream_t, cudaHostFn_t fn, void* userData)
{
    fn(userData);
    return cudaSuccess;
}
#define CUDART_CB

// ---------------------------------------------------------------------------
// Atomics.  Only float atomicAdd is used by the tree (point-source scatter,
// DAS stencil-transpose scatter); the integer ones are here so a future use
// compiles.  Lock-free CAS on the bit pattern: correct across the OpenMP
// threads of a launch, order-nondeterministic exactly like the device's.
// ---------------------------------------------------------------------------
inline float atomicAdd(float* address, float val)
{
    uint32_t* p = reinterpret_cast<uint32_t*>(address);
    uint32_t old_bits = __atomic_load_n(p, __ATOMIC_RELAXED);
    for (;;) {
        float old_val;
        std::memcpy(&old_val, &old_bits, sizeof old_val);
        const float new_val = old_val + val;
        uint32_t new_bits;
        std::memcpy(&new_bits, &new_val, sizeof new_bits);
        if (__atomic_compare_exchange_n(p, &old_bits, new_bits, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return old_val;
    }
}
inline int atomicAdd(int* p, int v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline unsigned atomicAdd(unsigned* p, unsigned v) { return __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v)
{
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}

// ---------------------------------------------------------------------------
// IEEE round-to-nearest intrinsics.  On the device they exist to defeat
// --use_fast_math; the host core is compiled -ffp-contract=off without
// fast-math, so the plain operation IS the single-rounding result.
// ---------------------------------------------------------------------------
inline float __fmul_rn(float a, float b) { return a * b; }
inline float __fadd_rn(float a, float b) { return a + b; }
inline float __fsub_rn(float a, float b) { return a - b; }
inline float __fdiv_rn(float a, float b) { return a / b; }
inline float __fsqrt_rn(float a) { return std::sqrt(a); }
inline double __dmul_rn(double a, double b) { return a * b; }
inline double __dadd_rn(double a, double b) { return a + b; }
// Round to nearest even, as __float2int_rn does (lrintf honours the current
// rounding mode, which the core never changes from the default).
inline int __float2int_rn(float x) { return static_cast<int>(std::lrintf(x)); }

// ---------------------------------------------------------------------------
// The device math overloads kernels call unqualified (``max(ix, 0)``):
// CUDA's math_functions.h puts these in the global namespace; <algorithm>
// only has std::max.  Same-type pairs, as the device provides.
// ---------------------------------------------------------------------------
inline int max(int a, int b) { return a > b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }
inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
inline long max(long a, long b) { return a > b ? a : b; }
inline long min(long a, long b) { return a < b ? a : b; }
inline unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }
inline unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
inline long long max(long long a, long long b) { return a > b ? a : b; }
inline long long min(long long a, long long b) { return a < b ? a : b; }
inline unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
inline unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
inline float max(float a, float b) { return std::fmax(a, b); }
inline float min(float a, float b) { return std::fmin(a, b); }
inline double max(double a, double b) { return std::fmax(a, b); }
inline double min(double a, double b) { return std::fmin(a, b); }
