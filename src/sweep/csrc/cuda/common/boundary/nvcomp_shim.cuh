#pragma once
// boundary/nvcomp_shim.cuh -- the boundary ring's lossless compression (nvCOMP
// Bitcomp) without a compile-time dependency on nvCOMP.
//
// The library is dlopen'ed once per process from the path Python resolves
// (``sweep.backend.c.nvcomp``: the ``nvidia-nvcomp-cu12`` wheel, or
// SWEEP_NVCOMP_LIB), and the five batched Bitcomp entry points are dlsym'ed.
// Their signatures and the two option structs are copied from nvCOMP 6
// (``nvcomp/bitcomp.h`` / ``nvcomp/shared_types.h``) and checked at load
// time through the library's version query, so a different major refuses to
// load instead of calling a function with the wrong layout.
//
// Everything else in this header is the device-side bookkeeping the runtime
// needs around a batched call: a block-wide scan over per-piece byte counts
// (prefix offsets, 8-byte padded, with the compressed sizes mirrored into the
// chunk index), the pack kernel that gathers the bound buffer's pieces into one
// contiguous block, and the kernel that turns a chunk's index row back into the
// compressed-piece pointer table the decompressor wants.  None of it is on the
// per-step path: every kernel here runs once per chunk on the copy stream.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#if !defined(SWEEP_HOST_CORE)
#include <dlfcn.h>   // the host core links without -ldl and never loads nvCOMP
#endif

#include "../../../core/check.h"

namespace sweep_nvcomp {

// ---- the nvCOMP 6 C API, as much of it as the shim calls ----------------
typedef int nvcompStatus_t;             // nvcompSuccess == 0
typedef int nvcompType_t;               // NVCOMP_TYPE_UCHAR == 1
typedef int bitcompMode_t;              // BITCOMP_LOSSLESS == 0
typedef int nvcompDecompressBackend_t;  // NVCOMP_DECOMPRESS_BACKEND_DEFAULT == 0
constexpr nvcompStatus_t kSuccess = 0;
constexpr nvcompType_t kTypeUchar = 1;
constexpr nvcompType_t kTypeUshort = 3;
constexpr nvcompType_t kTypeUint = 5;
constexpr nvcompType_t kTypeFloat16 = 10;
constexpr nvcompType_t kTypeFloat32 = 11;

struct CompressOpts {
    int algorithm;
    nvcompType_t data_type;
    double delta;
    bitcompMode_t mode;
    char reserved[44];
};
struct DecompressOpts {
    nvcompDecompressBackend_t backend;
    char reserved[60];
};
struct AlignmentRequirements {
    size_t input;
    size_t output;
    size_t temp;
};
static_assert(sizeof(CompressOpts) == 64, "nvcompBatchedBitcompCompressOpts_t is 64 bytes in nvCOMP 6");
static_assert(sizeof(DecompressOpts) == 64, "nvcompBatchedBitcompDecompressOpts_t is 64 bytes in nvCOMP 6");

typedef nvcompStatus_t (*GetMaxOutFn)(size_t, CompressOpts, size_t*);
typedef nvcompStatus_t (*GetCompAlignFn)(CompressOpts, AlignmentRequirements*);
typedef nvcompStatus_t (*GetDecompAlignFn)(DecompressOpts, AlignmentRequirements*);
typedef nvcompStatus_t (*CompressFn)(const void* const*, const size_t*, size_t, size_t,
                                     void*, size_t, void* const*, size_t*, CompressOpts,
                                     nvcompStatus_t*, cudaStream_t);
typedef nvcompStatus_t (*DecompressFn)(const void* const*, const size_t*, const size_t*,
                                       size_t*, size_t, void*, size_t, void* const*,
                                       DecompressOpts, nvcompStatus_t*, cudaStream_t);
typedef nvcompStatus_t (*DecompTempFn)(size_t, size_t, DecompressOpts, size_t*, size_t, cudaStream_t);
struct Properties {
    uint32_t version;          // major * 1000 + minor * 10 + patch (6.0.0 -> 6000)
    uint32_t cudart_version;
};
typedef nvcompStatus_t (*GetPropsFn)(Properties*);

struct Api {
    void* handle = nullptr;
    std::string path;
    GetMaxOutFn get_max_out = nullptr;
    GetCompAlignFn get_comp_align = nullptr;
    GetDecompAlignFn get_decomp_align = nullptr;
    CompressFn compress = nullptr;
    DecompressFn decompress = nullptr;
    DecompTempFn decomp_temp = nullptr;
};

// One per process (an inline variable: the drivers include this header from
// many translation units, and the loaded handle must be the same for all).
inline Api& api()
{
    static Api a;
    return a;
}
inline std::mutex& api_mutex()
{
    static std::mutex m;
    return m;
}

// Load ``path`` (a libnvcomp.so.6) once.  Idempotent for the same path; a
// second, different path is an error: the symbols are process-global.
inline void load(const char* path)
{
#if defined(SWEEP_HOST_CORE)
    SWEEP_CHECK(false, "nvCOMP (", path, ") cannot be loaded into the host core: boundary "
                "compression runs on a CUDA device only.");
#else
    std::lock_guard<std::mutex> lock(api_mutex());
    Api& a = api();
    if (a.handle != nullptr) {
        SWEEP_CHECK(a.path == path,
                    "nvCOMP is already loaded from ", a.path,
                    "; a second library (", path, ") cannot replace it in this process.");
        return;
    }
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    SWEEP_CHECK(h != nullptr, "dlopen(", path, ") failed: ", dlerror());
    auto sym = [&](const char* name) {
        void* p = dlsym(h, name);
        if (p == nullptr) {
            const char* e = dlerror();
            std::string err = e ? e : "";
            dlclose(h);
            SWEEP_CHECK(false, path, " has no ", name, " (", err,
                        "): sweep's boundary compression needs nvCOMP 6.x");
        }
        return p;
    };
    // The version first: a wrong major would make every struct below a lie.
    auto props = reinterpret_cast<GetPropsFn>(sym("nvcompGetProperties"));
    Properties v{0, 0};
    nvcompStatus_t st = props(&v);
    if (st != kSuccess || v.version / 1000 != 6) {
        dlclose(h);
        SWEEP_CHECK(false, path, ": nvcompGetProperties reports nvCOMP version ", v.version,
                    " (status ", st, "); this shim is written against nvCOMP 6.x.");
    }
    a.get_max_out = reinterpret_cast<GetMaxOutFn>(sym("nvcompBatchedBitcompCompressGetMaxOutputChunkSize"));
    a.get_comp_align = reinterpret_cast<GetCompAlignFn>(sym("nvcompBatchedBitcompCompressGetRequiredAlignments"));
    a.get_decomp_align = reinterpret_cast<GetDecompAlignFn>(sym("nvcompBatchedBitcompDecompressGetRequiredAlignments"));
    a.compress = reinterpret_cast<CompressFn>(sym("nvcompBatchedBitcompCompressAsync"));
    a.decompress = reinterpret_cast<DecompressFn>(sym("nvcompBatchedBitcompDecompressAsync"));
    a.decomp_temp = reinterpret_cast<DecompTempFn>(sym("nvcompBatchedBitcompDecompressGetTempSize"));
    a.handle = h;
    a.path = path;
#endif
}

inline bool loaded()
{
    std::lock_guard<std::mutex> lock(api_mutex());
    return api().handle != nullptr;
}

inline const Api& require()
{
    const Api& a = api();
    SWEEP_CHECK(a.handle != nullptr,
                "boundary compression was requested but nvCOMP is not loaded in this "
                "process (sweep.backend.c.nvcomp.ensure_loaded() runs before the call "
                "on the Python side; a driver reached the saver without it).");
    return a;
}

// Lossless Bitcomp, default algorithm, input typed by the ring's element
// width: the per-element modelling is what the ratio comes from, so a 2- or
// 4-byte payload is declared as such instead of as bytes.
inline CompressOpts compress_opts(int elem_size)
{
    CompressOpts o;
    std::memset(&o, 0, sizeof(o));
    o.algorithm = 0;
    o.data_type = elem_size == 1 ? kTypeUchar : elem_size == 2 ? kTypeUshort : kTypeUint;
    o.delta = 0.0;
    o.mode = 0;   // BITCOMP_LOSSLESS
    return o;
}
inline DecompressOpts decompress_opts()
{
    DecompressOpts o;
    std::memset(&o, 0, sizeof(o));
    o.backend = 0;
    return o;
}

// Pieces are 8-byte aligned everywhere (Bitcomp's stated requirement for
// both directions; checked against the library at bind time too).
constexpr int64_t kAlign = 8;
// One uncompressed piece: 64 KiB, so a 3 MB chunk is ~48 pieces.  Measured on
// the Overthrust int8 ring (RTX 6000 Ada): 16 KiB pieces give the same per-step
// time and the same ratio (1.90 vs 1.91) with four times the table entries, so
// the piece size is not what the remaining overlap cost depends on.
constexpr int64_t kPieceBytes = 64 * 1024;

inline int64_t align_up(int64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

inline int64_t max_compressed_piece_bytes(int elem_size)
{
    size_t out = 0;
    nvcompStatus_t st = require().get_max_out(static_cast<size_t>(kPieceBytes), compress_opts(elem_size), &out);
    SWEEP_CHECK(st == kSuccess, "nvcompBatchedBitcompCompressGetMaxOutputChunkSize failed: status ", st);
    return align_up(static_cast<int64_t>(out));
}

// Both directions' alignment requirements, checked once at bind against the
// constants the ring pitch / piece layout were built with.
inline void check_alignments(int elem_size)
{
    AlignmentRequirements c{0, 0, 0}, d{0, 0, 0};
    nvcompStatus_t st = require().get_comp_align(compress_opts(elem_size), &c);
    SWEEP_CHECK(st == kSuccess, "nvcompBatchedBitcompCompressGetRequiredAlignments failed: status ", st);
    st = require().get_decomp_align(decompress_opts(), &d);
    SWEEP_CHECK(st == kSuccess, "nvcompBatchedBitcompDecompressGetRequiredAlignments failed: status ", st);
    SWEEP_CHECK(static_cast<int64_t>(c.input) <= kAlign && static_cast<int64_t>(c.output) <= kAlign &&
                static_cast<int64_t>(d.input) <= kAlign && static_cast<int64_t>(d.output) <= kAlign,
                "nvCOMP Bitcomp wants alignments (compress in/out ", c.input, "/", c.output,
                ", decompress in/out ", d.input, "/", d.output, ") above the ", kAlign,
                " bytes the boundary ring and piece layout provide.");
}

// Bytes of temporary device storage the batched decompressor wants for up to
// ``max_pieces`` pieces of kPieceBytes (20 B in nvCOMP 6.0, but asked, not assumed).
inline int64_t decompress_temp_bytes(int max_pieces, int64_t max_total_bytes)
{
    size_t out = 0;
    nvcompStatus_t st = require().decomp_temp(static_cast<size_t>(max_pieces), static_cast<size_t>(kPieceBytes),
                                              decompress_opts(), &out, static_cast<size_t>(max_total_bytes), nullptr);
    SWEEP_CHECK(st == kSuccess, "nvcompBatchedBitcompDecompressGetTempSize failed: status ", st);
    return static_cast<int64_t>(out);
}

// ---- device bookkeeping ---------------------------------------------------
// The per-slot table (``tab``): 6 rows of ``max_pieces`` int64 each.
enum TabRow : int {
    kRowUncPtr = 0,     // uncompressed piece start (ring)            [host-filled at bind]
    kRowUncBytes = 1,   // uncompressed piece bytes                   [host-filled at bind]
    kRowBoundPtr = 2,   // compressed piece start in the bound buffer [host-filled at bind]
    kRowCompBytes = 3,  // compressed bytes per piece                 [nvcomp writes]
    kRowCompPtr = 4,    // compressed piece start in the packed buffer [scan writes]
    kRowActual = 5,     // actual uncompressed bytes                  [nvcomp writes]
    kTabRows = 6,
};
// The per-chunk index row: [arena offset (host-filled), total packed bytes,
// steps the chunk was compressed with, piece sizes...] -- ``kIdxSizes +
// max_pieces`` int64.  The step count is what the backward fetch selects the
// piece geometry by: a ring that is refilled per segment (ModelParallel
// segment_steps) compresses chunks cut at segment ends, shorter than
// transfer_interval, and not only at the record's end.
constexpr int kIdxOffset = 0;
constexpr int kIdxTotal = 1;
constexpr int kIdxLen = 2;
constexpr int kIdxSizes = 3;

constexpr int kScanThreads = 1024;

#if !defined(SWEEP_HOST_CORE)
// Exclusive prefix over ``n`` byte counts, each rounded up to kAlign, in one
// block: writes the offsets (as pointers off ``base``) into ``out_ptrs``, the
// total into ``*out_total``, and mirrors the raw sizes into ``out_sizes``
// (both optional).  Tiles of kScanThreads with a carried running sum, so n is
// unbounded in principle; a chunk is at most a few thousand pieces.
static __global__ void scan_piece_sizes_kernel(const int64_t* __restrict__ sizes, int n,
                                        const char* base, int64_t* __restrict__ out_ptrs,
                                        int64_t* __restrict__ out_total,
                                        int64_t* __restrict__ out_sizes,
                                        int64_t* __restrict__ out_len, int64_t len_value)
{
    if (threadIdx.x == 0 && out_len != nullptr) *out_len = len_value;
    __shared__ int64_t s[kScanThreads];
    __shared__ int64_t carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int tile = 0; tile < n; tile += kScanThreads) {
        const int i = tile + threadIdx.x;
        const int64_t raw = (i < n) ? sizes[i] : 0;
        const int64_t v = (raw + kAlign - 1) / kAlign * kAlign;
        s[threadIdx.x] = v;
        __syncthreads();
        // Hillis-Steele inclusive scan.
        for (int off = 1; off < kScanThreads; off <<= 1) {
            const int64_t add = (threadIdx.x >= off) ? s[threadIdx.x - off] : 0;
            __syncthreads();
            s[threadIdx.x] += add;
            __syncthreads();
        }
        if (i < n) {
            const int64_t excl = carry + s[threadIdx.x] - v;
            out_ptrs[i] = reinterpret_cast<int64_t>(base) + excl;
            if (out_sizes != nullptr) out_sizes[i] = raw;
        }
        __syncthreads();
        if (threadIdx.x == 0) carry += s[kScanThreads - 1];
        __syncthreads();
    }
    if (threadIdx.x == 0 && out_total != nullptr) *out_total = carry;
}

// Gather the pieces of one slot's bound buffer into the packed block: block i
// copies piece i from ``src_ptrs[i]`` (8-aligned) to ``dst_ptrs[i]`` (8-aligned,
// from the scan) -- whole words, then the tail bytes.
static __global__ void pack_pieces_kernel(const int64_t* __restrict__ src_ptrs,
                                   const int64_t* __restrict__ sizes,
                                   const int64_t* __restrict__ dst_ptrs)
{
    const int i = blockIdx.x;
    const char* src = reinterpret_cast<const char*>(src_ptrs[i]);
    char* dst = reinterpret_cast<char*>(dst_ptrs[i]);
    const int64_t n = sizes[i];
    const int64_t words = n / 8;
    const uint64_t* s8 = reinterpret_cast<const uint64_t*>(src);
    uint64_t* d8 = reinterpret_cast<uint64_t*>(dst);
    for (int64_t w = threadIdx.x; w < words; w += blockDim.x)
        d8[w] = s8[w];
    const int64_t tail = words * 8;
    for (int64_t b = tail + threadIdx.x; b < n; b += blockDim.x)
        dst[b] = src[b];
}

#endif  // !SWEEP_HOST_CORE

// Fold the first N host-visible status words into one error message.
inline void check_statuses(const int32_t* host, int n, const char* what)
{
    for (int i = 0; i < n; ++i) {
        SWEEP_CHECK(host[i] == kSuccess,
                    "nvCOMP Bitcomp ", what, " reported status ", host[i], " on piece ", i,
                    " of a boundary chunk; the boundary ring cannot be trusted. "
                    "(nvcompStatus_t: 1 ErrorInvalidValue, 2 ErrorNotSupported, "
                    "3 ErrorCannotDecompress, 4 ErrorBadChecksum, 5 ErrorCannotVerifyChecksums, "
                    "6 ErrorOutputBufferTooSmall, 7 ErrorContentsChanged, 10 ErrorCudaError, "
                    "11 ErrorInternal).");
    }
}

}  // namespace sweep_nvcomp
