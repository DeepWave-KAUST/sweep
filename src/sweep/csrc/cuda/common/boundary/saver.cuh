#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <torch/extension.h>

#include "../buf_torch.h"
#include "../context.h"
#include "../cudautils.h"   // copy_tensor_cuda_async, zero_tensor_device_async
#include "disk_io.cuh"
#include "kernels.cuh"
#include "types.cuh"

// boundary_dtype_from_tensor() now lives in ../buf_torch.h, next to buf_of()
// which needs it to tag a descriptor; the torch::Tensor overload is unchanged
// and a Buf overload was added so call sites that ask a (now Buf) saver member
// for its storage dtype keep their spelling.

// The storage-dtype guard that used to be a four-way torch::ScalarType test.
// A Buf carries the tree's BoundaryDtype tag, whose four values ARE
// float32/float16/bfloat16/uint8 -- so testing the tag alone would be a
// tautology.  What the byte copies actually depend on is that the buffer's
// element WIDTH is the width its tag implies; that is false for every dtype
// outside the four whose width differs (float64, int64, ...), which is the case
// the old check caught.  A same-width foreign dtype (int32 -> FP32 tag, 4 B)
// now passes where it was rejected; nothing in sweep allocates one, since
// Python allocates these buffers in the requested boundary precision.
static inline bool boundary_storage_dtype_ok(const Buf& b) {
    return b.element_size() == buf_dtype_element_size(b.dtype());
}

// Printable shape/stride for the bounds diagnostic below.  A Buf carries the
// numbers but not torch's ArrayRef streaming, so format them here.
static inline std::string boundary_shape_str(const Buf& b, bool strides) {
    std::string out = "[";
    for (int64_t i = 0; i < b.dim(); ++i) {
        if (i > 0) out += ", ";
        out += std::to_string(strides ? b.stride(i) : b.size(i));
    }
    return out + "]";
}

// Byte pointer into a boundary tensor at element offset `off_elems`, honoring
// the tensor's dtype width.  Used so the gpu<->cpu<->disk byte copies work for
// fp16/bf16 staging/storage, not just fp32.
// A DD tile drops its cut faces to numel 0 (Layout(cut_mask=...) on the Python
// side).  Faces come in pairs that share one stride, so the byte count is taken
// from whichever face of the pair is live and stays non-zero -- the copy goes
// out anyway carrying the empty face's null data_ptr(), and segfaults the host
// thread inside cudaMemcpyAsync.  Report an empty face as null here; the two
// copy primitives below skip it.  Non-DD never sees this: every face is backed
// there, so nothing returns null.
static inline char* boundary_byte_ptr(const Buf& t, int64_t off_elems) {
    if (!t.defined() || t.numel() == 0) return nullptr;
    return static_cast<char*>(t.data_ptr()) + off_elems * t.element_size();
}

// Staging copy that tolerates a cut-away face on either end.
static inline cudaError_t boundary_memcpy_async(
    void* dst, const void* src, size_t bytes, cudaMemcpyKind kind,
    cudaStream_t stream) {
    if (dst == nullptr || src == nullptr) return cudaSuccess;
    return cudaMemcpyAsync(dst, src, bytes, kind, stream);
}

// Diagnostic (SWEEP_BOUNDARY_BOUNDS=1): turn the segfault inside
// cudaMemcpyAsync into a named, numbered failure.  A staged copy that walks
// off one of its two buffers is invisible until the driver dereferences it,
// and the stack then only says "cudaMemcpyAsync".
static inline void boundary_bounds_check(
    const char* what, const Buf& t, int64_t off_elems, size_t bytes) {
    static const bool on = std::getenv("SWEEP_BOUNDARY_BOUNDS") != nullptr;
    if (!on || !t.defined() || t.numel() == 0) return;
    const int64_t es = t.element_size();
    const int64_t have = t.numel() * es;
    const int64_t want = off_elems * es + (int64_t)bytes;
    SWEEP_CHECK(off_elems >= 0 && want <= have,
                "boundary staging out of range on ", what,
                ": offset ", off_elems, " elems (", off_elems * es,
                " B) + ", bytes, " B = ", want,
                " B, buffer holds ", t.numel(), " elems (", have,
                " B), sizes ", boundary_shape_str(t, false),
                " strides ", boundary_shape_str(t, true));
}

struct EffectiveBoundarySaver {

    // The buffers below are DESCRIPTORS (core/buf.h): non-owning views of
    // memory the Python propagator allocated.  The tensor handles that keep
    // that memory alive live in ``torch_refs_`` / ``storage_th_`` at the bottom
    // of this struct -- the same refcount these members used to hold
    // themselves.
    Buf left_t, right_t;
    Buf front_t, back_t;
    Buf bottom_t, top_t;
    // last_two is a Buf like every other buffer here since step 2 of the
    // torch-free line: the drivers' ``select(...).copy_(...)`` on it now go
    // through the memcpy helpers, which take a Buf, and BUF_MAX_DIMS is 8, so
    // its 7 dimensions fit.  The tensor itself is kept beside it for the one
    // thing that still needs a tensor -- handing it back to Python as
    // ``out.last_two`` -- the way storage_th_ sits beside the storage faces.
    Buf last_two;
    torch::Tensor last_two_th_;

    Buf left_gpu, right_gpu;
    Buf front_gpu, back_gpu;
    Buf bottom_gpu, top_gpu;

    // INT8 path only: per-face FP32 scale buffer (one per quantization
    // block) and an FP32 staging buffer holding a single timestep's
    // worth of data per face.  Compute writes FP32 into staging via the
    // existing kernel path, then launch_quantize_int8 reduces+stores
    // into the persistent uint8 face buffer (left_t et al.) plus this
    // scale buffer.
    Buf left_scale_t, right_scale_t;
    Buf front_scale_t, back_scale_t;
    Buf bottom_scale_t, top_scale_t;

    // INT8 staged path (storage='cpu'/'disk') only: per-face FP32 scale
    // RING on the GPU, parallel to the uint8 main ring (top_gpu et al.).
    // Each chunk is flushed to / loaded from the persistent FP32 scale
    // buffer (top_scale_t et al.) alongside the uint8 main buffer.
    Buf left_scale_gpu, right_scale_gpu;
    Buf front_scale_gpu, back_scale_gpu;
    Buf bottom_scale_gpu, top_scale_gpu;

    Buf left_staging_t, right_staging_t;
    Buf front_staging_t, back_staging_t;
    Buf bottom_staging_t, top_staging_t;

    bool enabled = false;

    int dim = 3;
    int nvar = 1;

    bool store_on_gpu = false;
    int nt = 0;

    // stride for one timestep
    int64_t left_stride = 0;
    int64_t right_stride = 0;
    int64_t front_stride = 0;
    int64_t back_stride = 0;
    int64_t bottom_stride = 0;
    int64_t top_stride = 0;

    // Torch-side ownership.  A Buf never owns memory, so every tensor this
    // saver binds (or, on the self-allocating fallback, makes) is parked here
    // for the saver's lifetime -- byte for byte the refcount the torch::Tensor
    // members used to hold.  Pushing into the vector may reallocate it, which
    // moves the Tensor handles but never the storage they point at, so the Bufs
    // stay valid.
    std::vector<torch::Tensor> torch_refs_;

    // The six PERSISTENT storage faces as tensors, in bind_tensor_group order
    // (top, bottom, front, back, left, right; front/back undefined in 2-D).
    // load_from_vector() still needs real tensors for Tensor::copy_().
    torch::Tensor storage_th_[6];

    // Park the tensor, hand back a descriptor of it.
    inline Buf keep(const torch::Tensor& t)
    {
        torch_refs_.push_back(t);
        return buf_of(t);
    }

    inline void bind_tensor_group(
        const std::vector<torch::Tensor>& tensors,
        const char* role,
        Buf& top,
        Buf& bottom,
        Buf& front,
        Buf& back,
        Buf& left,
        Buf& right,
        torch::Tensor* handles = nullptr
    )
    {
        if (dim == 3) {
            SWEEP_CHECK(tensors.size() == 6, role, " must contain 6 tensors for 3D");
            top    = keep(tensors[0]);
            bottom = keep(tensors[1]);
            front  = keep(tensors[2]);
            back   = keep(tensors[3]);
            left   = keep(tensors[4]);
            right  = keep(tensors[5]);
            if (handles != nullptr) {
                handles[0] = tensors[0];
                handles[1] = tensors[1];
                handles[2] = tensors[2];
                handles[3] = tensors[3];
                handles[4] = tensors[4];
                handles[5] = tensors[5];
            }
        } else {
            SWEEP_CHECK(tensors.size() == 4, role, " must contain 4 tensors for 2D");
            top    = keep(tensors[0]);
            bottom = keep(tensors[1]);
            left   = keep(tensors[2]);
            right  = keep(tensors[3]);
            front  = Buf();
            back   = Buf();
            if (handles != nullptr) {
                handles[0] = tensors[0];
                handles[1] = tensors[1];
                handles[2] = torch::Tensor();
                handles[3] = torch::Tensor();
                handles[4] = tensors[2];
                handles[5] = tensors[3];
            }
        }
    }

    inline void bind_storage_tensors(
        const std::vector<torch::Tensor>& tensors,
        const char* role
    )
    {
        bind_tensor_group(tensors, role, top_t, bottom_t, front_t, back_t, left_t, right_t,
                          storage_th_);
    }

    inline void bind_staging_tensors(
        const std::vector<torch::Tensor>& tensors,
        const char* role
    )
    {
        bind_tensor_group(tensors, role, top_gpu, bottom_gpu, front_gpu, back_gpu, left_gpu, right_gpu);
    }

    inline void compute_time_strides()
    {
        if (dim == 3) {
            auto& left_src = store_on_gpu ? left_t : left_gpu;
            auto& right_src = store_on_gpu ? right_t : right_gpu;
            auto& front_src = store_on_gpu ? front_t : front_gpu;
            auto& back_src = store_on_gpu ? back_t : back_gpu;
            auto& bottom_src = store_on_gpu ? bottom_t : bottom_gpu;
            auto& top_src = store_on_gpu ? top_t : top_gpu;

            left_stride   = left_src.stride(0);
            right_stride  = right_src.stride(0);
            front_stride  = front_src.stride(0);
            back_stride   = back_src.stride(0);
            bottom_stride = bottom_src.stride(0);
            top_stride    = top_src.stride(0);
        } else {
            auto& left_src = store_on_gpu ? left_t : left_gpu;
            auto& right_src = store_on_gpu ? right_t : right_gpu;
            auto& bottom_src = store_on_gpu ? bottom_t : bottom_gpu;
            auto& top_src = store_on_gpu ? top_t : top_gpu;

            left_stride   = left_src.stride(1);
            right_stride  = right_src.stride(1);
            bottom_stride = bottom_src.stride(1);
            top_stride    = top_src.stride(1);
        }
    }

    // Bind externally-allocated per-block FP32 scale tensors.  Used by
    // the INT8 path so the same persistent scale buffer is visible to
    // both the forward save kernel and the backward restore kernel.
    inline void bind_int8_scales(const std::vector<torch::Tensor>& scales)
    {
        bind_tensor_group(scales, "boundary_gpu int8 scale",
                          top_scale_t, bottom_scale_t,
                          front_scale_t, back_scale_t,
                          left_scale_t, right_scale_t);
    }

    // Bind externally-allocated per-block FP32 scale RING tensors (staged
    // INT8 only).  Parallel to bind_staging_tensors for the uint8 main
    // ring; flushed to / loaded from the persistent scale buffer.
    inline void bind_int8_scale_rings(const std::vector<torch::Tensor>& scales)
    {
        bind_tensor_group(scales, "boundary_gpu int8 scale ring",
                          top_scale_gpu, bottom_scale_gpu,
                          front_scale_gpu, back_scale_gpu,
                          left_scale_gpu, right_scale_gpu);
    }

    // Per-face FP32 staging geometry for the scaled (INT8 / FP16) store: ONE
    // timestep's worth of boundary band per face, in bind_tensor_group's face
    // order (top, bottom, front, back, left, right; 2-D drops front/back).
    // The Python side derives the same list from ``Layout.staging_shapes``
    // (memory/shape.py) out of the same width / n*_boundary the persistent
    // buffers come from, so one formula lives on each side and the bound
    // tensors are checked against this one below.
    inline std::vector<std::vector<int64_t>> int8_staging_shapes(
        const SolverContext& ctx,
        int width,
        int nx_boundary,
        int ny_boundary,
        int nz_boundary) const
    {
        if (dim == 3) {
            return {
                {1, ctx.B, width, ny_boundary, nx_boundary},   // top
                {1, ctx.B, width, ny_boundary, nx_boundary},   // bottom
                {1, ctx.B, nz_boundary, width, nx_boundary},   // front
                {1, ctx.B, nz_boundary, width, nx_boundary},   // back
                {1, ctx.B, nz_boundary, ny_boundary, width},   // left
                {1, ctx.B, nz_boundary, ny_boundary, width},   // right
            };
        }
        return {
            {1, 1, ctx.B, width, nx_boundary},   // top
            {1, 1, ctx.B, width, nx_boundary},   // bottom
            {1, 1, ctx.B, nz_boundary, width},   // left
            {1, 1, ctx.B, nz_boundary, width},   // right
        };
    }

    // Bind the Python-owned FP32 staging (ForwardInput/BackwardInput::
    // boundary_staging).  Count + per-slot geometry/dtype/device checks in the
    // spirit of cudautils.h's pool_slot_checked: the save/restore kernels take
    // data_ptr<float>() and index it by ctx.B / width / n*_boundary, and
    // quantize_step copies the persistent buffer's whole per-step stride out of
    // it, so a slot of the wrong shape reads as garbage rather than failing.
    inline void bind_int8_staging(
        const std::vector<torch::Tensor>& staging,
        const std::vector<std::vector<int64_t>>& shapes)
    {
        SWEEP_CHECK(staging.size() == shapes.size(),
                    "boundary_staging must contain ", shapes.size(),
                    " tensors for ", dim, "D, got ", staging.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            SWEEP_CHECK(staging[i].sizes().vec() == shapes[i],
                        "boundary_staging[", i, "] has shape ", staging[i].sizes(),
                        " but the saver's geometry is ", shapes[i],
                        " -- the Python Layout and this driver disagree on "
                        "save_width or tangent_pad (the staging is one timestep "
                        "of the same band as the persistent buffer).");
            SWEEP_CHECK(staging[i].scalar_type() == torch::kFloat,
                        "boundary_staging[", i, "] must be float32, got ",
                        staging[i].scalar_type());
            SWEEP_CHECK(staging[i].is_cuda(),
                        "boundary_staging[", i, "] must live on the GPU (a host "
                        "tensor here reads as an illegal address inside the "
                        "kernels), got ", staging[i].device());
        }
        bind_tensor_group(staging, "boundary_staging",
                          top_staging_t, bottom_staging_t,
                          front_staging_t, back_staging_t,
                          left_staging_t, right_staging_t);
    }

    // Scaled (INT8 / FP16) path: the FP32 staging buffer, one timestep per
    // face.  Compute writes FP32 into staging via the existing FP32 boundary
    // kernel; launch_quantize_{int8,fp16} then compresses staging into the
    // persistent payload + scale buffers (bound externally), and the backward's
    // launch_dequantize_* expands back into it before the restore kernel reads
    // it.  ``staging`` is the propagator's buffer (same lifetime as the other
    // boundary buffers: per propagator, re-made when the cache key changes);
    // there is no unbound-caller fallback.
    //
    // Zeroing: the propagator zeroes its staging ONCE, at allocation, not per
    // call -- and that is enough to keep the scaled path bit-exact.  Every cell
    // the save kernel writes is overwritten before quantize_step reduces over
    // it; the only cells it never writes are the tangential pad of a
    // tangent_pad > 0 layout (VRZ: the band kernels run with tangent_pad = 0
    // while the buffers are sized with it) and a DD cut face (kernel and
    // quantize both gate on ctx.cut_*).  The pad cells still enter the
    // per-block max, so they must read 0 -- and they stay 0 forever: nothing
    // but launch_dequantize_* ever writes them, and that writes back
    // (0 - 128) * scale == 0 for a cell that was quantized from 0.
    inline void allocate_int8_staging(
        const SolverContext& ctx,
        int width,
        int nx_boundary,
        int ny_boundary,
        int nz_boundary,
        const std::vector<torch::Tensor>& staging)
    {
        const auto shapes = int8_staging_shapes(ctx, width, nx_boundary, ny_boundary, nz_boundary);
        SWEEP_CHECK(!staging.empty(),
                    "scaled (int8/fp16) boundary storage requires the propagator-bound "
                    "boundary_staging (Layout.staging_shapes); the saver no longer "
                    "allocates it.");
        bind_int8_staging(staging, shapes);
        // Defensive: quantize_step / dequantize_step copy the persistent buffer's
        // full per-step stride (top_t.stride(0) in 3D) between the staging and the
        // int8 ring.  If the caller's tangent_pad here disagrees with the pad the
        // persistent buffers were allocated with (Python boundary_tangent_pad),
        // the staging is undersized and the copy reads/writes out of bounds.  Turn
        // that silent illegal access into a clear error at allocation time.
        // (A DD cut face keeps its FULL staging band on both sides -- Python does
        // NOT collapse these the way gpu_full_shapes collapses the persistent
        // faces -- because PyTorch clamps a 0-size dim's stride to a nonzero
        // value, so a numel-0 slot would trip this very check.)
        if (dim == 3 && top_t.defined() && top_t.numel() > 0) {
            SWEEP_CHECK(top_staging_t.numel() >= top_t.stride(0) &&
                        left_staging_t.numel() >= left_t.stride(0) &&
                        front_staging_t.numel() >= front_t.stride(0),
                        "INT8 FP32 staging is smaller than the persistent boundary "
                        "buffer per-step stride -- tangent_pad mismatch between the "
                        "Python layout and the CUDA saver (top ", top_staging_t.numel(),
                        " vs ", top_t.stride(0), ").");
        }
    }

    inline void allocate_last_two(
        const SolverContext& ctx,
        int last_two_nvar,
        const torch::Tensor& last_two
    )
    {
        // Binding is mandatory, as it is for every driver buffer since the
        // propagator took over allocation: _c.py binds u_last_two whenever
        // boundary saving is on (.contiguous(), the very tensor that comes
        // back as out.last_two).  A caller that arrives without it used to get
        // a silent torch::zeros of the right shape; that fallback is what kept
        // a torch allocation inside the saver, and it hid a missing binding.
        SWEEP_CHECK(last_two.defined(),
                    "boundary saving requires the propagator-bound u_last_two "
                    "({nvar, 2, B, 1, nz[, ny], nx}); the saver no longer allocates it.");
        (void)last_two_nvar;
        last_two_th_ = last_two;
        this->last_two = keep(last_two);
    }

    void allocate(
        bool use_boundary_saving,
        int dim_,
        int nvar_,
        const SolverContext& ctx,
        const torch::Tensor& ref_tensor,
        int width = -1,
        int last_two_nvar = 2,
        bool override_storage = false,
        bool store_on_gpu_override = false,
        int transfer_interval = 1,
        const std::vector<torch::Tensor>& boundary_cpu = {},
        const std::vector<torch::Tensor>& boundary_gpu = {},
        const torch::Tensor& last_two = {},
        bool use_pinned_memory_ = false,
        int tangent_pad = 0,
        const std::vector<torch::Tensor>& boundary_staging = {}
    )
    {
        // One binding per saver. The Buf members describe the FIRST binding while
        // torch_refs_ would pin both, so a second allocate() would leave the
        // descriptors pointing at storage nothing else refers to. Every caller
        // allocates once, in its constructor; this says so out loud.
        SWEEP_CHECK(torch_refs_.empty(),
                    "EffectiveBoundarySaver::allocate() was called twice on one saver.");
        enabled = use_boundary_saving;
        dim = dim_;
        nvar = nvar_;
        nt = static_cast<int>(ctx.nt);

        if (!enabled) return;

        if (width < 0)
            width = ctx.M;

        // =========================
        // Storage strategy
        // =========================

        if (override_storage)
            store_on_gpu = store_on_gpu_override;
        else
            store_on_gpu = (dim == 2);   // default

        // FP16 / BF16 / INT8 storage gate.  Derived from the boundary buffers
        // Python hands us (allocated in the requested precision) rather than a
        // process-global env var — this prevents a per-instance storage_dtype
        // from leaking across propagator instances.  Last-two and
        // staging buffers remain FP32 — last_two is a full-wavefield snapshot
        // used to bootstrap backward, staging is the INT8 path's per-timestep
        // FP32 view before quantization.
        BoundaryDtype runtime_dtype = BoundaryDtype::FP32;
        if (!boundary_gpu.empty() && boundary_gpu[0].scalar_type() == torch::kUInt8) {
            runtime_dtype = BoundaryDtype::INT8;        // uint8 main buffer => INT8 path
        } else if (!boundary_cpu.empty()) {
            runtime_dtype = boundary_dtype_from_tensor(boundary_cpu[0]);
        } else if (!boundary_gpu.empty()) {
            runtime_dtype = boundary_dtype_from_tensor(boundary_gpu[0]);
        } else {
            SWEEP_CHECK(false, "boundary saving requires the propagator-bound boundary "
                        "buffers (boundary_cpu / boundary_gpu); the saver no longer "
                        "allocates them.");
        }
        const bool use_scaled = (runtime_dtype == BoundaryDtype::INT8 ||
                                 runtime_dtype == BoundaryDtype::FP16);
        // ref_tensor / use_pinned_memory_ chose the dtype and placement of the
        // self-allocated storage; with binding mandatory they are unused and
        // leave the signature with the Buf flip.
        (void)ref_tensor; (void)use_pinned_memory_;

        // =========================
        // Physical domain
        // =========================

        int nx_phys = ctx.nx_phys();
        int nz_phys = ctx.nz_phys();
        int ny_phys = (dim == 3) ? ctx.ny_phys() : 1;

        int nx_boundary = nx_phys + 2 * tangent_pad;
        int nz_boundary = nz_phys + 2 * tangent_pad;
        int ny_boundary = (dim == 3) ? ny_phys + 2 * tangent_pad : 1;
        
        const size_t scaled_faces = (dim == 3 ? 6 : 4);
        if (use_scaled && store_on_gpu) {
            // Scaled gpu-direct (INT8 / FP16): Python owns a persistent
            // payload main (uint8 or fp16) + FP32 per-block scale tensors
            // and passes them concatenated in ``boundary_gpu`` (first half
            // = main, second half = scale).  We bind those here, and the
            // single-timestep FP32 staging comes from ``boundary_staging``
            // (Python-owned; self-allocated only for an unbound caller).
            SWEEP_CHECK(boundary_gpu.size() == 2 * scaled_faces,
                        "Scaled (int8/fp16) boundary_gpu expects ", 2 * scaled_faces,
                        " tensors (", scaled_faces, " main + ", scaled_faces,
                        " scale), got ", boundary_gpu.size());
            std::vector<torch::Tensor> main_tensors(boundary_gpu.begin(),
                                                    boundary_gpu.begin() + scaled_faces);
            std::vector<torch::Tensor> scale_tensors(boundary_gpu.begin() + scaled_faces,
                                                     boundary_gpu.end());
            bind_storage_tensors(main_tensors, "boundary_gpu scaled main");
            bind_int8_scales(scale_tensors);
            allocate_int8_staging(ctx, width, nx_boundary, ny_boundary, nz_boundary,
                                  boundary_staging);
        } else if (use_scaled) {
            // Scaled staged (cpu/disk): Python owns a persistent payload
            // main + FP32 scale buffer (``boundary_cpu``; on disk these are
            // the cpu staging buffers) AND a payload main + FP32 scale RING
            // on the GPU (``boundary_gpu``).  Both are concatenated
            // main-then-scale.  We bind the cpu side to top_t/top_scale_t,
            // the gpu rings to top_gpu/top_scale_gpu, and take the shared
            // single-timestep FP32 staging from ``boundary_staging`` (the same
            // Python-owned buffer as gpu-direct).
            SWEEP_CHECK(boundary_cpu.size() == 2 * scaled_faces,
                        "Scaled (int8/fp16) staged boundary_cpu expects ", 2 * scaled_faces,
                        " tensors (", scaled_faces, " main + ", scaled_faces,
                        " scale), got ", boundary_cpu.size());
            SWEEP_CHECK(boundary_gpu.size() == 2 * scaled_faces,
                        "Scaled (int8/fp16) staged boundary_gpu expects ", 2 * scaled_faces,
                        " tensors (", scaled_faces, " main + ", scaled_faces,
                        " scale), got ", boundary_gpu.size());
            std::vector<torch::Tensor> cpu_main(boundary_cpu.begin(),
                                                boundary_cpu.begin() + scaled_faces);
            std::vector<torch::Tensor> cpu_scale(boundary_cpu.begin() + scaled_faces,
                                                 boundary_cpu.end());
            std::vector<torch::Tensor> gpu_main(boundary_gpu.begin(),
                                                boundary_gpu.begin() + scaled_faces);
            std::vector<torch::Tensor> gpu_scale(boundary_gpu.begin() + scaled_faces,
                                                 boundary_gpu.end());
            bind_storage_tensors(cpu_main, "boundary_cpu scaled main");
            bind_int8_scales(cpu_scale);
            bind_staging_tensors(gpu_main, "boundary_gpu scaled main ring");
            bind_int8_scale_rings(gpu_scale);
            allocate_int8_staging(ctx, width, nx_boundary, ny_boundary, nz_boundary,
                                  boundary_staging);
        } else if (!boundary_cpu.empty()) {
            bind_storage_tensors(boundary_cpu, "boundary_cpu");
        } else if (store_on_gpu && !boundary_gpu.empty()) {
            bind_storage_tensors(boundary_gpu, "boundary_gpu");
        } else {
            SWEEP_CHECK(false, "boundary saving requires the propagator-bound boundary "
                        "storage (boundary_cpu, or boundary_gpu for gpu-direct); the "
                        "saver no longer allocates it.");
        }

        if (!use_scaled && !store_on_gpu) {
            SWEEP_CHECK(!boundary_gpu.empty(),
                        "cpu/disk boundary staging requires the propagator-bound "
                        "boundary_gpu ring; the saver no longer allocates it.");
            bind_staging_tensors(boundary_gpu, "boundary_gpu");
        }

        compute_time_strides();

        // last_two is the wavefield snapshot used to bootstrap the backward
        // pass — full-precision matters here even when boundary buffers go
        // FP16 (PoC choice; can be relaxed later).
        allocate_last_two(ctx, last_two_nvar, last_two);
    }

    GeneralBoundaryPointer view()
    {
        GeneralBoundaryPointer v{};

        if (!enabled) return v;

        // last_two is always FP32 (we forced it in allocate above).
        v.last_two = last_two.data_ptr<float>();

        // Detect storage dtype on the persistent face buffer (left_t) to
        // decide which set of pointers to populate.
        auto dt = left_t.dtype();
        if (dt == BoundaryDtype::FP16) {
            // FP16 is per-block scaled storage (same two-pass flow as
            // INT8): persistent __half payload + FP32 per-block scales +
            // FP32 transient staging for the boundary kernels.
            SWEEP_CHECK(top_scale_t.defined(),
                        "fp16 boundary storage requires per-block scale "
                        "buffers (allocated by the Python wrapper).");
            v.dtype = BoundaryDtype::FP16;
            v.use_fp16 = true;
            v.top_h    = reinterpret_cast<__half*>(top_t.data_ptr());
            v.bottom_h = reinterpret_cast<__half*>(bottom_t.data_ptr());
            v.left_h   = reinterpret_cast<__half*>(left_t.data_ptr());
            v.right_h  = reinterpret_cast<__half*>(right_t.data_ptr());
            if (dim == 3) {
                v.front_h = reinterpret_cast<__half*>(front_t.data_ptr());
                v.back_h  = reinterpret_cast<__half*>(back_t.data_ptr());
            }
            v.top_scale    = top_scale_t.data_ptr<float>();
            v.bottom_scale = bottom_scale_t.data_ptr<float>();
            v.left_scale   = left_scale_t.data_ptr<float>();
            v.right_scale  = right_scale_t.data_ptr<float>();
            if (dim == 3) {
                v.front_scale = front_scale_t.data_ptr<float>();
                v.back_scale  = back_scale_t.data_ptr<float>();
            }
            v.top    = top_staging_t.data_ptr<float>();
            v.bottom = bottom_staging_t.data_ptr<float>();
            v.left   = left_staging_t.data_ptr<float>();
            v.right  = right_staging_t.data_ptr<float>();
            if (dim == 3) {
                v.front = front_staging_t.data_ptr<float>();
                v.back  = back_staging_t.data_ptr<float>();
            }
        } else if (dt == BoundaryDtype::BF16) {
            v.dtype = BoundaryDtype::BF16;
            v.use_fp16 = false;
            v.left_bf  = reinterpret_cast<__nv_bfloat16*>(left_t.data_ptr());
            v.right_bf = reinterpret_cast<__nv_bfloat16*>(right_t.data_ptr());
            if (dim == 3) {
                v.front_bf = reinterpret_cast<__nv_bfloat16*>(front_t.data_ptr());
                v.back_bf  = reinterpret_cast<__nv_bfloat16*>(back_t.data_ptr());
            }
            v.bottom_bf = reinterpret_cast<__nv_bfloat16*>(bottom_t.data_ptr());
            v.top_bf    = reinterpret_cast<__nv_bfloat16*>(top_t.data_ptr());
        } else if (dt == BoundaryDtype::INT8) {
            // Same guard the FP16 branch carries.  torch used to raise
            // "Tensor is undefined" from the first data_ptr<float>() below if
            // the scales were missing; a descriptor's typed data_ptr only
            // asserts (and is compiled out with -DNDEBUG), so state the
            // requirement here instead of handing a null scale to a kernel.
            SWEEP_CHECK(top_scale_t.defined(),
                        "int8 boundary storage requires per-block scale "
                        "buffers (allocated by the Python wrapper).");
            v.dtype = BoundaryDtype::INT8;
            v.use_fp16 = false;
            // Persistent uint8 buffers
            v.top_q    = reinterpret_cast<uint8_t*>(top_t.data_ptr());
            v.bottom_q = reinterpret_cast<uint8_t*>(bottom_t.data_ptr());
            v.left_q   = reinterpret_cast<uint8_t*>(left_t.data_ptr());
            v.right_q  = reinterpret_cast<uint8_t*>(right_t.data_ptr());
            if (dim == 3) {
                v.front_q = reinterpret_cast<uint8_t*>(front_t.data_ptr());
                v.back_q  = reinterpret_cast<uint8_t*>(back_t.data_ptr());
            }
            // Per-block FP32 scales (one per BOUNDARY_INT8_BLOCK cells).
            v.top_scale    = top_scale_t.data_ptr<float>();
            v.bottom_scale = bottom_scale_t.data_ptr<float>();
            v.left_scale   = left_scale_t.data_ptr<float>();
            v.right_scale  = right_scale_t.data_ptr<float>();
            if (dim == 3) {
                v.front_scale = front_scale_t.data_ptr<float>();
                v.back_scale  = back_scale_t.data_ptr<float>();
            }
            // FP32 staging — the existing FP32 boundary kernel writes
            // into these, then launch_quantize_int8 compresses to
            // top_q/scale.  Bind via the FP32 face fields so the kernel
            // dispatch can reuse the FP32 path unchanged.
            v.top    = top_staging_t.data_ptr<float>();
            v.bottom = bottom_staging_t.data_ptr<float>();
            v.left   = left_staging_t.data_ptr<float>();
            v.right  = right_staging_t.data_ptr<float>();
            if (dim == 3) {
                v.front = front_staging_t.data_ptr<float>();
                v.back  = back_staging_t.data_ptr<float>();
            }
        } else {
            v.dtype = BoundaryDtype::FP32;
            v.use_fp16 = false;
            v.left  = left_t.data_ptr<float>();
            v.right = right_t.data_ptr<float>();
            if (dim == 3) {
                v.front = front_t.data_ptr<float>();
                v.back  = back_t.data_ptr<float>();
            }
            v.bottom = bottom_t.data_ptr<float>();
            v.top    = top_t.data_ptr<float>();
        }

        return v;
    }

    void load_from_vector(
        const std::vector<torch::Tensor>& u_boundary,
        const torch::Tensor& ref_tensor
        )
        {
            if (!enabled)
                throw std::runtime_error("Boundary saving not enabled.");

            // Scaled paths (INT8 / FP16): the payload+scale layout cannot
            // be seeded by a plain tensor copy, and an empty u_boundary
            // from the equation backward.cu fallback is fine — bail
            // before the size check.
            if (left_t.defined() && (left_t.dtype() == BoundaryDtype::INT8 ||
                                     left_t.dtype() == BoundaryDtype::FP16))
                return;

            auto copy_to = [&](torch::Tensor& dst, const torch::Tensor& src)
            {
                if (dst.device() == src.device()) {
                    copy_tensor_cuda_async(dst, src);
                }
                else {
                    copy_tensor_cuda_async(dst, src);
                }
            };

            if (dim == 2) {

                if (u_boundary.size() != 4)
                    throw std::runtime_error("2D boundary expects 4 tensors.");
                
                // storage_th_[] is the same tensor top_t/bottom_t/... describe,
                // in bind_tensor_group order (top, bottom, front, back, left,
                // right); Tensor::copy_ needs the tensor, not the descriptor.
                copy_to( storage_th_[0], u_boundary[0]);
                copy_to( storage_th_[1], u_boundary[1]);
                copy_to( storage_th_[4], u_boundary[2]);
                copy_to( storage_th_[5], u_boundary[3]);

            } else { // 3D

                if (u_boundary.size() != 6)
                    throw std::runtime_error("3D boundary expects 6 tensors.");

                copy_to( storage_th_[0], u_boundary[0]);
                copy_to( storage_th_[1], u_boundary[1]);

                copy_to( storage_th_[2], u_boundary[2]);
                copy_to( storage_th_[3], u_boundary[3]);

                copy_to( storage_th_[4], u_boundary[4]);
                copy_to( storage_th_[5], u_boundary[5]);
            }
        }

    inline void copy_2d_chunk_async(
        void* dst,
        size_t dst_var_block,
        const void* src,
        size_t src_var_block,
        size_t bytes,
        size_t elem_size,
        cudaMemcpyKind kind,
        cudaStream_t stream
    ) const
    {
        if (dst == nullptr || src == nullptr) return;   // DD cut face
        if (nvar == 1) {
            cudaMemcpyAsync(dst, src, bytes, kind, stream);
        } else {
            cudaMemcpy2DAsync(
                dst,
                dst_var_block * elem_size,
                src,
                src_var_block * elem_size,
                bytes,
                nvar,
                kind,
                stream
            );
        }
    }

    // INT8 staged only: copy the per-block FP32 scale RING (top_scale_gpu
    // et al.) <-> the persistent FP32 scale buffer (top_scale_t et al.),
    // exactly parallel to the uint8 main copy in flush_gpu_to_cpu /
    // load_cpu_to_gpu.  ``kind`` selects direction: DeviceToHost flushes
    // ring->cpu (forward), HostToDevice loads cpu->ring (backward).
    // ``cpu_start`` indexes the persistent/staging scale buffer (== the
    // main buffer's ``start``/``stage_start``); ``gpu_start`` indexes the
    // ring.  es = 4 (scales are always FP32).
    inline void copy_int8_scales(int cpu_start, int len, int gpu_start,
                                 cudaMemcpyKind kind, cudaStream_t stream)
    {
        const size_t es = sizeof(float);
        const bool d2h = (kind == cudaMemcpyDeviceToHost);
        if (dim == 2) {
            size_t top_var_block = top_scale_t.stride(0);
            size_t top_time_block = top_scale_t.stride(1);
            size_t left_var_block = left_scale_t.stride(0);
            size_t left_time_block = left_scale_t.stride(1);
            size_t top_gpu_var_block = top_scale_gpu.stride(0);
            size_t top_gpu_time_block = top_scale_gpu.stride(1);
            size_t left_gpu_var_block = left_scale_gpu.stride(0);
            size_t left_gpu_time_block = left_scale_gpu.stride(1);
            size_t top_bytes = (size_t)len * top_time_block * es;
            size_t left_bytes = (size_t)len * left_time_block * es;
            auto face = [&](const Buf& cpu_t, const Buf& gpu_t,
                            size_t cpu_var_block, size_t cpu_time_block,
                            size_t gpu_var_block, size_t gpu_time_block, size_t bytes) {
                char* cpu_p = boundary_byte_ptr(cpu_t, (int64_t)cpu_start * cpu_time_block);
                char* gpu_p = boundary_byte_ptr(gpu_t, (int64_t)gpu_start * gpu_time_block);
                if (d2h)
                    copy_2d_chunk_async(cpu_p, cpu_var_block, gpu_p, gpu_var_block, bytes, es, kind, stream);
                else
                    copy_2d_chunk_async(gpu_p, gpu_var_block, cpu_p, cpu_var_block, bytes, es, kind, stream);
            };
            face(top_scale_t, top_scale_gpu, top_var_block, top_time_block, top_gpu_var_block, top_gpu_time_block, top_bytes);
            face(bottom_scale_t, bottom_scale_gpu, top_var_block, top_time_block, top_gpu_var_block, top_gpu_time_block, top_bytes);
            face(left_scale_t, left_scale_gpu, left_var_block, left_time_block, left_gpu_var_block, left_gpu_time_block, left_bytes);
            face(right_scale_t, right_scale_gpu, left_var_block, left_time_block, left_gpu_var_block, left_gpu_time_block, left_bytes);
            return;
        }

        size_t top_block = top_scale_t.stride(0);
        size_t front_block = front_scale_t.stride(0);
        size_t left_block = left_scale_t.stride(0);
        size_t top_gpu_block = top_scale_gpu.stride(0);
        size_t front_gpu_block = front_scale_gpu.stride(0);
        size_t left_gpu_block = left_scale_gpu.stride(0);
        size_t top_elems = (size_t)len * nvar * top_block;
        size_t front_elems = (size_t)len * nvar * front_block;
        size_t left_elems = (size_t)len * nvar * left_block;
        size_t top_gpu_offset = (size_t)gpu_start * nvar * top_gpu_block;
        size_t front_gpu_offset = (size_t)gpu_start * nvar * front_gpu_block;
        size_t left_gpu_offset = (size_t)gpu_start * nvar * left_gpu_block;
        auto face = [&](const Buf& cpu_t, const Buf& gpu_t,
                        size_t cpu_block, size_t gpu_offset, size_t elems) {
            char* cpu_p = boundary_byte_ptr(cpu_t, (int64_t)cpu_start * nvar * cpu_block);
            char* gpu_p = boundary_byte_ptr(gpu_t, (int64_t)gpu_offset);
            if (d2h)
                boundary_memcpy_async(cpu_p, gpu_p, elems * es, kind, stream);
            else
                boundary_memcpy_async(gpu_p, cpu_p, elems * es, kind, stream);
        };
        face(top_scale_t, top_scale_gpu, top_block, top_gpu_offset, top_elems);
        face(bottom_scale_t, bottom_scale_gpu, top_block, top_gpu_offset, top_elems);
        face(front_scale_t, front_scale_gpu, front_block, front_gpu_offset, front_elems);
        face(back_scale_t, back_scale_gpu, front_block, front_gpu_offset, front_elems);
        face(left_scale_t, left_scale_gpu, left_block, left_gpu_offset, left_elems);
        face(right_scale_t, right_scale_gpu, left_block, left_gpu_offset, left_elems);
    }

    inline void flush_gpu_to_cpu(int start, int len, cudaStream_t stream, int gpu_start = 0)
    {
        // Scaled staged (INT8/FP16): flush the FP32 scale ring alongside
        // the payload main.
        if (top_t.defined() && top_scale_t.defined() &&
            (top_t.dtype() == BoundaryDtype::INT8 || top_t.dtype() == BoundaryDtype::FP16))
            copy_int8_scales(start, len, gpu_start, cudaMemcpyDeviceToHost, stream);
        if (dim == 2)
        {
            size_t top_var_block = top_t.stride(0);
            size_t top_time_block = top_t.stride(1);
            size_t left_var_block = left_t.stride(0);
            size_t left_time_block = left_t.stride(1);
            size_t top_gpu_var_block = top_gpu.stride(0);
            size_t top_gpu_time_block = top_gpu.stride(1);
            size_t left_gpu_var_block = left_gpu.stride(0);
            size_t left_gpu_time_block = left_gpu.stride(1);

            size_t es = top_t.element_size();
            size_t top_bytes = (size_t)len * top_time_block * es;
            size_t left_bytes = (size_t)len * left_time_block * es;

            SWEEP_CHECK(boundary_storage_dtype_ok(top_t), "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
            copy_2d_chunk_async(
                boundary_byte_ptr(top_t, (int64_t)start * top_time_block),
                top_var_block,
                boundary_byte_ptr(top_gpu, (int64_t)gpu_start * top_gpu_time_block),
                top_gpu_var_block,
                top_bytes,
                es,
                cudaMemcpyDeviceToHost,
                stream
            );
            copy_2d_chunk_async(
                boundary_byte_ptr(bottom_t, (int64_t)start * top_time_block),
                top_var_block,
                boundary_byte_ptr(bottom_gpu, (int64_t)gpu_start * top_gpu_time_block),
                top_gpu_var_block,
                top_bytes,
                es,
                cudaMemcpyDeviceToHost,
                stream
            );
            copy_2d_chunk_async(
                boundary_byte_ptr(left_t, (int64_t)start * left_time_block),
                left_var_block,
                boundary_byte_ptr(left_gpu, (int64_t)gpu_start * left_gpu_time_block),
                left_gpu_var_block,
                left_bytes,
                es,
                cudaMemcpyDeviceToHost,
                stream
            );
            copy_2d_chunk_async(
                boundary_byte_ptr(right_t, (int64_t)start * left_time_block),
                left_var_block,
                boundary_byte_ptr(right_gpu, (int64_t)gpu_start * left_gpu_time_block),
                left_gpu_var_block,
                left_bytes,
                es,
                cudaMemcpyDeviceToHost,
                stream
            );
            return;
        }

        size_t left_block   = left_t.stride(0);
        size_t front_block  = front_t.stride(0);
        size_t bottom_block = bottom_t.stride(0);
        size_t left_gpu_block   = left_gpu.stride(0);
        size_t front_gpu_block  = front_gpu.stride(0);
        size_t bottom_gpu_block = bottom_gpu.stride(0);

        size_t left_elems   = len * nvar * left_block;
        size_t front_elems  = len * nvar * front_block;
        size_t bottom_elems = len * nvar * bottom_block;
        size_t left_gpu_offset = static_cast<size_t>(gpu_start) * nvar * left_gpu_block;
        size_t front_gpu_offset = static_cast<size_t>(gpu_start) * nvar * front_gpu_block;
        size_t bottom_gpu_offset = static_cast<size_t>(gpu_start) * nvar * bottom_gpu_block;


        SWEEP_CHECK(boundary_storage_dtype_ok(left_t), "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
        size_t es = left_t.element_size();
        boundary_memcpy_async(
            boundary_byte_ptr(left_t, (int64_t)start * nvar * left_block),
            boundary_byte_ptr(left_gpu, (int64_t)left_gpu_offset),
            left_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        boundary_memcpy_async(
            boundary_byte_ptr(right_t, (int64_t)start * nvar * left_block),
            boundary_byte_ptr(right_gpu, (int64_t)left_gpu_offset),
            left_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        boundary_memcpy_async(
            boundary_byte_ptr(front_t, (int64_t)start * nvar * front_block),
            boundary_byte_ptr(front_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        boundary_memcpy_async(
            boundary_byte_ptr(back_t, (int64_t)start * nvar * front_block),
            boundary_byte_ptr(back_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        boundary_memcpy_async(
            boundary_byte_ptr(top_t, (int64_t)start * nvar * bottom_block),
            boundary_byte_ptr(top_gpu, (int64_t)bottom_gpu_offset),
            bottom_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        boundary_memcpy_async(
            boundary_byte_ptr(bottom_t, (int64_t)start * nvar * bottom_block),
            boundary_byte_ptr(bottom_gpu, (int64_t)bottom_gpu_offset),
            bottom_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
    }

    inline void flush_gpu_to_disk_2d(
        int start,
        int len,
        const std::vector<std::string>& paths,
        cudaStream_t stream,
        int gpu_start = 0,
        int stage_start = 0)
    {
        if (dim != 2)
            throw std::runtime_error("flush_gpu_to_disk_2d only supports 2D boundaries.");
        const bool is_scaled = top_t.defined() &&
            (top_t.dtype() == BoundaryDtype::INT8 || top_t.dtype() == BoundaryDtype::FP16);
        if (paths.size() != (is_scaled ? 8u : 4u))
            throw std::runtime_error("2D disk boundary expects 4 (fp) / 8 (int8/fp16) file paths.");

        size_t top_var_block = top_t.stride(0);
        size_t top_time_block = top_t.stride(1);
        size_t left_var_block = left_t.stride(0);
        size_t left_time_block = left_t.stride(1);
        size_t top_gpu_var_block = top_gpu.stride(0);
        size_t top_gpu_time_block = top_gpu.stride(1);
        size_t left_gpu_var_block = left_gpu.stride(0);
        size_t left_gpu_time_block = left_gpu.stride(1);

        size_t es = top_t.element_size();
        size_t top_bytes = (size_t)len * top_time_block * es;
        size_t left_bytes = (size_t)len * left_time_block * es;

        copy_2d_chunk_async(
            boundary_byte_ptr(top_t, (int64_t)stage_start * top_time_block),
            top_var_block,
            boundary_byte_ptr(top_gpu, (int64_t)gpu_start * top_gpu_time_block),
            top_gpu_var_block,
            top_bytes,
            es,
            cudaMemcpyDeviceToHost,
            stream
        );
        copy_2d_chunk_async(
            boundary_byte_ptr(bottom_t, (int64_t)stage_start * top_time_block),
            top_var_block,
            boundary_byte_ptr(bottom_gpu, (int64_t)gpu_start * top_gpu_time_block),
            top_gpu_var_block,
            top_bytes,
            es,
            cudaMemcpyDeviceToHost,
            stream
        );
        copy_2d_chunk_async(
            boundary_byte_ptr(left_t, (int64_t)stage_start * left_time_block),
            left_var_block,
            boundary_byte_ptr(left_gpu, (int64_t)gpu_start * left_gpu_time_block),
            left_gpu_var_block,
            left_bytes,
            es,
            cudaMemcpyDeviceToHost,
            stream
        );
        copy_2d_chunk_async(
            boundary_byte_ptr(right_t, (int64_t)stage_start * left_time_block),
            left_var_block,
            boundary_byte_ptr(right_gpu, (int64_t)gpu_start * left_gpu_time_block),
            left_gpu_var_block,
            left_bytes,
            es,
            cudaMemcpyDeviceToHost,
            stream
        );

        auto* meta = new BoundaryDisk2DMeta();
        meta->paths = {paths[0], paths[1], paths[2], paths[3]};
        meta->top_elems = len * top_time_block;
        meta->left_elems = len * left_time_block;
        meta->start_top_offset = static_cast<size_t>(start) * top_time_block;
        meta->start_left_offset = static_cast<size_t>(start) * left_time_block;
        meta->top_var_block = top_var_block;
        meta->left_var_block = left_var_block;
        meta->top_file_var_block = static_cast<size_t>(nt) * top_time_block;
        meta->left_file_var_block = static_cast<size_t>(nt) * left_time_block;
        meta->nvar = nvar;
        meta->elem_size = es;
        meta->top = boundary_byte_ptr(top_t, (int64_t)stage_start * top_time_block);
        meta->bottom = boundary_byte_ptr(bottom_t, (int64_t)stage_start * top_time_block);
        meta->left = boundary_byte_ptr(left_t, (int64_t)stage_start * left_time_block);
        meta->right = boundary_byte_ptr(right_t, (int64_t)stage_start * left_time_block);
        cudaLaunchHostFunc(stream, write_boundary_disk_2d_callback, meta);

        if (is_scaled) {
            // Scaled (int8/fp16): flush the FP32 scale ring -> cpu scale staging (D2H), then
            // write the scale to its own files (paths[4..7]) via a second meta
            // reusing the same dtype-agnostic disk-write callback (es=4).
            copy_int8_scales(stage_start, len, gpu_start, cudaMemcpyDeviceToHost, stream);
            size_t s_es = sizeof(float);
            size_t s_top_time = top_scale_t.stride(1);
            size_t s_left_time = left_scale_t.stride(1);
            auto* smeta = new BoundaryDisk2DMeta();
            smeta->paths = {paths[4], paths[5], paths[6], paths[7]};
            smeta->top_elems = (size_t)len * s_top_time;
            smeta->left_elems = (size_t)len * s_left_time;
            smeta->start_top_offset = (size_t)start * s_top_time;
            smeta->start_left_offset = (size_t)start * s_left_time;
            smeta->top_var_block = top_scale_t.stride(0);
            smeta->left_var_block = left_scale_t.stride(0);
            smeta->top_file_var_block = (size_t)nt * s_top_time;
            smeta->left_file_var_block = (size_t)nt * s_left_time;
            smeta->nvar = nvar;
            smeta->elem_size = s_es;
            smeta->top = boundary_byte_ptr(top_scale_t, (int64_t)stage_start * s_top_time);
            smeta->bottom = boundary_byte_ptr(bottom_scale_t, (int64_t)stage_start * s_top_time);
            smeta->left = boundary_byte_ptr(left_scale_t, (int64_t)stage_start * s_left_time);
            smeta->right = boundary_byte_ptr(right_scale_t, (int64_t)stage_start * s_left_time);
            cudaLaunchHostFunc(stream, write_boundary_disk_2d_callback, smeta);
        }
    }

    inline void flush_gpu_to_disk_3d(
        int start,
        int len,
        const std::vector<std::string>& paths,
        cudaStream_t stream,
        int gpu_start = 0,
        int stage_start = 0)
    {
        if (dim != 3)
            throw std::runtime_error("flush_gpu_to_disk_3d only supports 3D boundaries.");
        const bool is_scaled = top_t.defined() &&
            (top_t.dtype() == BoundaryDtype::INT8 || top_t.dtype() == BoundaryDtype::FP16);
        if (paths.size() != (is_scaled ? 12u : 6u))
            throw std::runtime_error("3D disk boundary expects 6 (fp) / 12 (int8/fp16) file paths.");

        size_t top_block = top_t.stride(0);
        size_t front_block = front_t.stride(0);
        size_t left_block = left_t.stride(0);
        size_t top_gpu_block = top_gpu.stride(0);
        size_t front_gpu_block = front_gpu.stride(0);
        size_t left_gpu_block = left_gpu.stride(0);
        size_t es = top_t.element_size();

        size_t top_elems = len * nvar * top_block;
        size_t front_elems = len * nvar * front_block;
        size_t left_elems = len * nvar * left_block;
        size_t top_stage_offset = static_cast<size_t>(stage_start) * nvar * top_block;
        size_t front_stage_offset = static_cast<size_t>(stage_start) * nvar * front_block;
        size_t left_stage_offset = static_cast<size_t>(stage_start) * nvar * left_block;
        size_t top_gpu_offset = static_cast<size_t>(gpu_start) * nvar * top_gpu_block;
        size_t front_gpu_offset = static_cast<size_t>(gpu_start) * nvar * front_gpu_block;
        size_t left_gpu_offset = static_cast<size_t>(gpu_start) * nvar * left_gpu_block;

        boundary_memcpy_async(
            boundary_byte_ptr(top_t, (int64_t)top_stage_offset),
            boundary_byte_ptr(top_gpu, (int64_t)top_gpu_offset),
            top_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        boundary_memcpy_async(
            boundary_byte_ptr(bottom_t, (int64_t)top_stage_offset),
            boundary_byte_ptr(bottom_gpu, (int64_t)top_gpu_offset),
            top_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        boundary_memcpy_async(
            boundary_byte_ptr(front_t, (int64_t)front_stage_offset),
            boundary_byte_ptr(front_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        boundary_memcpy_async(
            boundary_byte_ptr(back_t, (int64_t)front_stage_offset),
            boundary_byte_ptr(back_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        boundary_memcpy_async(
            boundary_byte_ptr(left_t, (int64_t)left_stage_offset),
            boundary_byte_ptr(left_gpu, (int64_t)left_gpu_offset),
            left_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        boundary_memcpy_async(
            boundary_byte_ptr(right_t, (int64_t)left_stage_offset),
            boundary_byte_ptr(right_gpu, (int64_t)left_gpu_offset),
            left_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        auto* meta = new BoundaryDisk3DMeta();
        meta->paths = {paths[0], paths[1], paths[2], paths[3], paths[4], paths[5]};
        meta->top_elems = top_elems;
        meta->front_elems = front_elems;
        meta->left_elems = left_elems;
        meta->top_offset = static_cast<size_t>(start) * nvar * top_block;
        meta->front_offset = static_cast<size_t>(start) * nvar * front_block;
        meta->left_offset = static_cast<size_t>(start) * nvar * left_block;
        meta->elem_size = es;
        meta->top = boundary_byte_ptr(top_t, (int64_t)top_stage_offset);
        meta->bottom = boundary_byte_ptr(bottom_t, (int64_t)top_stage_offset);
        meta->front = boundary_byte_ptr(front_t, (int64_t)front_stage_offset);
        meta->back = boundary_byte_ptr(back_t, (int64_t)front_stage_offset);
        meta->left = boundary_byte_ptr(left_t, (int64_t)left_stage_offset);
        meta->right = boundary_byte_ptr(right_t, (int64_t)left_stage_offset);
        cudaLaunchHostFunc(stream, write_boundary_disk_3d_callback, meta);

        if (is_scaled) {
            // INT8: flush FP32 scale ring -> cpu scale staging (D2H), then
            // write scale to its own files (paths[6..11]) via a second meta
            // reusing the dtype-agnostic 3D disk-write callback (es=4).
            copy_int8_scales(stage_start, len, gpu_start, cudaMemcpyDeviceToHost, stream);
            size_t s_top_block = top_scale_t.stride(0);
            size_t s_front_block = front_scale_t.stride(0);
            size_t s_left_block = left_scale_t.stride(0);
            size_t s_top_elems = (size_t)len * nvar * s_top_block;
            size_t s_front_elems = (size_t)len * nvar * s_front_block;
            size_t s_left_elems = (size_t)len * nvar * s_left_block;
            size_t s_top_stage = (size_t)stage_start * nvar * s_top_block;
            size_t s_front_stage = (size_t)stage_start * nvar * s_front_block;
            size_t s_left_stage = (size_t)stage_start * nvar * s_left_block;
            auto* smeta = new BoundaryDisk3DMeta();
            smeta->paths = {paths[6], paths[7], paths[8], paths[9], paths[10], paths[11]};
            smeta->top_elems = s_top_elems;
            smeta->front_elems = s_front_elems;
            smeta->left_elems = s_left_elems;
            smeta->top_offset = (size_t)start * nvar * s_top_block;
            smeta->front_offset = (size_t)start * nvar * s_front_block;
            smeta->left_offset = (size_t)start * nvar * s_left_block;
            smeta->elem_size = sizeof(float);
            smeta->top = boundary_byte_ptr(top_scale_t, (int64_t)s_top_stage);
            smeta->bottom = boundary_byte_ptr(bottom_scale_t, (int64_t)s_top_stage);
            smeta->front = boundary_byte_ptr(front_scale_t, (int64_t)s_front_stage);
            smeta->back = boundary_byte_ptr(back_scale_t, (int64_t)s_front_stage);
            smeta->left = boundary_byte_ptr(left_scale_t, (int64_t)s_left_stage);
            smeta->right = boundary_byte_ptr(right_scale_t, (int64_t)s_left_stage);
            cudaLaunchHostFunc(stream, write_boundary_disk_3d_callback, smeta);
        }
    }

    inline void flush_gpu_to_disk(
        int start,
        int len,
        const std::vector<std::string>& paths,
        cudaStream_t stream,
        int gpu_start = 0,
        int stage_start = 0
    )
    {
        if (dim == 2) {
            flush_gpu_to_disk_2d(start, len, paths, stream, gpu_start, stage_start);
        } else {
            flush_gpu_to_disk_3d(start, len, paths, stream, gpu_start, stage_start);
        }
    }

    inline void load_disk_to_cpu_2d(int start, int len, const std::vector<std::string>& paths, int stage_start = 0)
    {
        if (dim != 2)
            throw std::runtime_error("load_disk_to_cpu_2d only supports 2D boundaries.");
        const bool is_scaled = top_t.defined() &&
            (top_t.dtype() == BoundaryDtype::INT8 || top_t.dtype() == BoundaryDtype::FP16);
        if (paths.size() != (is_scaled ? 8u : 4u))
            throw std::runtime_error("2D disk boundary expects 4 (fp) / 8 (int8/fp16) file paths.");

        size_t top_time_block = top_t.stride(1);
        size_t left_time_block = left_t.stride(1);
        size_t top_elems = len * top_time_block;
        size_t left_elems = len * left_time_block;
        size_t top_offset = static_cast<size_t>(start) * top_time_block;
        size_t left_offset = static_cast<size_t>(start) * left_time_block;

        size_t top_var_block = top_t.stride(0);
        size_t left_var_block = left_t.stride(0);
        size_t top_file_var_block = static_cast<size_t>(nt) * top_time_block;
        size_t left_file_var_block = static_cast<size_t>(nt) * left_time_block;
        size_t es = top_t.element_size();
        char* top_dst = boundary_byte_ptr(top_t, (int64_t)stage_start * top_time_block);
        char* bottom_dst = boundary_byte_ptr(bottom_t, (int64_t)stage_start * top_time_block);
        char* left_dst = boundary_byte_ptr(left_t, (int64_t)stage_start * left_time_block);
        char* right_dst = boundary_byte_ptr(right_t, (int64_t)stage_start * left_time_block);

        for (int v = 0; v < nvar; ++v) {
            read_boundary_file_chunk(
                paths[0],
                static_cast<size_t>(v) * top_file_var_block + top_offset,
                top_dst + static_cast<size_t>(v) * top_var_block * es,
                top_elems,
                es
            );
            read_boundary_file_chunk(
                paths[1],
                static_cast<size_t>(v) * top_file_var_block + top_offset,
                bottom_dst + static_cast<size_t>(v) * top_var_block * es,
                top_elems,
                es
            );
            read_boundary_file_chunk(
                paths[2],
                static_cast<size_t>(v) * left_file_var_block + left_offset,
                left_dst + static_cast<size_t>(v) * left_var_block * es,
                left_elems,
                es
            );
            read_boundary_file_chunk(
                paths[3],
                static_cast<size_t>(v) * left_file_var_block + left_offset,
                right_dst + static_cast<size_t>(v) * left_var_block * es,
                left_elems,
                es
            );
        }

        if (is_scaled) {
            // Scaled (int8/fp16): read the FP32 per-block scale files (paths[4..7]) into the
            // cpu scale staging; load_cpu_to_gpu then copies it to the ring.
            size_t s_top_time = top_scale_t.stride(1);
            size_t s_left_time = left_scale_t.stride(1);
            size_t s_top_elems = (size_t)len * s_top_time;
            size_t s_left_elems = (size_t)len * s_left_time;
            size_t s_top_off = (size_t)start * s_top_time;
            size_t s_left_off = (size_t)start * s_left_time;
            size_t s_top_var = top_scale_t.stride(0);
            size_t s_left_var = left_scale_t.stride(0);
            size_t s_top_file_var = (size_t)nt * s_top_time;
            size_t s_left_file_var = (size_t)nt * s_left_time;
            const size_t ses = sizeof(float);
            char* s_top_dst = boundary_byte_ptr(top_scale_t, (int64_t)stage_start * s_top_time);
            char* s_bottom_dst = boundary_byte_ptr(bottom_scale_t, (int64_t)stage_start * s_top_time);
            char* s_left_dst = boundary_byte_ptr(left_scale_t, (int64_t)stage_start * s_left_time);
            char* s_right_dst = boundary_byte_ptr(right_scale_t, (int64_t)stage_start * s_left_time);
            for (int v = 0; v < nvar; ++v) {
                read_boundary_file_chunk(paths[4], (size_t)v * s_top_file_var + s_top_off,
                                         s_top_dst + (size_t)v * s_top_var * ses, s_top_elems, ses);
                read_boundary_file_chunk(paths[5], (size_t)v * s_top_file_var + s_top_off,
                                         s_bottom_dst + (size_t)v * s_top_var * ses, s_top_elems, ses);
                read_boundary_file_chunk(paths[6], (size_t)v * s_left_file_var + s_left_off,
                                         s_left_dst + (size_t)v * s_left_var * ses, s_left_elems, ses);
                read_boundary_file_chunk(paths[7], (size_t)v * s_left_file_var + s_left_off,
                                         s_right_dst + (size_t)v * s_left_var * ses, s_left_elems, ses);
            }
        }
    }

    inline void load_disk_to_cpu_3d(int start, int len, const std::vector<std::string>& paths, int stage_start = 0)
    {
        if (dim != 3)
            throw std::runtime_error("load_disk_to_cpu_3d only supports 3D boundaries.");
        const bool is_scaled = top_t.defined() &&
            (top_t.dtype() == BoundaryDtype::INT8 || top_t.dtype() == BoundaryDtype::FP16);
        if (paths.size() != (is_scaled ? 12u : 6u))
            throw std::runtime_error("3D disk boundary expects 6 (fp) / 12 (int8/fp16) file paths.");

        size_t top_block = top_t.stride(0);
        size_t front_block = front_t.stride(0);
        size_t left_block = left_t.stride(0);

        size_t top_elems = len * nvar * top_block;
        size_t front_elems = len * nvar * front_block;
        size_t left_elems = len * nvar * left_block;

        size_t top_offset = static_cast<size_t>(start) * nvar * top_block;
        size_t front_offset = static_cast<size_t>(start) * nvar * front_block;
        size_t left_offset = static_cast<size_t>(start) * nvar * left_block;
        size_t top_stage_offset = static_cast<size_t>(stage_start) * nvar * top_block;
        size_t front_stage_offset = static_cast<size_t>(stage_start) * nvar * front_block;
        size_t left_stage_offset = static_cast<size_t>(stage_start) * nvar * left_block;

        size_t es = top_t.element_size();
        std::exception_ptr read_error = nullptr;
        std::mutex read_error_mutex;
        auto read_one = [&](std::string path, size_t offset, void* dst, size_t elems) {
            try {
                read_boundary_file_chunk(path, offset, dst, elems, es);
            } catch (...) {
                std::lock_guard<std::mutex> lock(read_error_mutex);
                if (!read_error)
                    read_error = std::current_exception();
            }
        };

        std::vector<std::thread> readers;
        readers.reserve(6);
        readers.emplace_back(read_one, paths[0], top_offset, (void*)boundary_byte_ptr(top_t, (int64_t)top_stage_offset), top_elems);
        readers.emplace_back(read_one, paths[1], top_offset, (void*)boundary_byte_ptr(bottom_t, (int64_t)top_stage_offset), top_elems);
        readers.emplace_back(read_one, paths[2], front_offset, (void*)boundary_byte_ptr(front_t, (int64_t)front_stage_offset), front_elems);
        readers.emplace_back(read_one, paths[3], front_offset, (void*)boundary_byte_ptr(back_t, (int64_t)front_stage_offset), front_elems);
        readers.emplace_back(read_one, paths[4], left_offset, (void*)boundary_byte_ptr(left_t, (int64_t)left_stage_offset), left_elems);
        readers.emplace_back(read_one, paths[5], left_offset, (void*)boundary_byte_ptr(right_t, (int64_t)left_stage_offset), left_elems);

        if (is_scaled) {
            // Scaled (int8/fp16): read the FP32 per-block scale files (paths[6..11]) into the
            // cpu scale staging in parallel; es=4 (main reads use the payload's element size).
            const size_t ses = sizeof(float);
            auto read_one_scale = [&](std::string path, size_t offset, void* dst, size_t elems) {
                try {
                    read_boundary_file_chunk(path, offset, dst, elems, ses);
                } catch (...) {
                    std::lock_guard<std::mutex> lock(read_error_mutex);
                    if (!read_error)
                        read_error = std::current_exception();
                }
            };
            size_t s_top_block = top_scale_t.stride(0);
            size_t s_front_block = front_scale_t.stride(0);
            size_t s_left_block = left_scale_t.stride(0);
            size_t s_top_elems = (size_t)len * nvar * s_top_block;
            size_t s_front_elems = (size_t)len * nvar * s_front_block;
            size_t s_left_elems = (size_t)len * nvar * s_left_block;
            size_t s_top_off = (size_t)start * nvar * s_top_block;
            size_t s_front_off = (size_t)start * nvar * s_front_block;
            size_t s_left_off = (size_t)start * nvar * s_left_block;
            size_t s_top_stage = (size_t)stage_start * nvar * s_top_block;
            size_t s_front_stage = (size_t)stage_start * nvar * s_front_block;
            size_t s_left_stage = (size_t)stage_start * nvar * s_left_block;
            readers.emplace_back(read_one_scale, paths[6], s_top_off, (void*)boundary_byte_ptr(top_scale_t, (int64_t)s_top_stage), s_top_elems);
            readers.emplace_back(read_one_scale, paths[7], s_top_off, (void*)boundary_byte_ptr(bottom_scale_t, (int64_t)s_top_stage), s_top_elems);
            readers.emplace_back(read_one_scale, paths[8], s_front_off, (void*)boundary_byte_ptr(front_scale_t, (int64_t)s_front_stage), s_front_elems);
            readers.emplace_back(read_one_scale, paths[9], s_front_off, (void*)boundary_byte_ptr(back_scale_t, (int64_t)s_front_stage), s_front_elems);
            readers.emplace_back(read_one_scale, paths[10], s_left_off, (void*)boundary_byte_ptr(left_scale_t, (int64_t)s_left_stage), s_left_elems);
            readers.emplace_back(read_one_scale, paths[11], s_left_off, (void*)boundary_byte_ptr(right_scale_t, (int64_t)s_left_stage), s_left_elems);
        }

        for (auto& reader : readers)
            reader.join();
        if (read_error)
            std::rethrow_exception(read_error);
    }

    inline void load_cpu_to_gpu(int start, int len, cudaStream_t stream, int gpu_start = 0)
    {
        // Scaled staged (INT8/FP16): load the FP32 scale ring alongside
        // the payload main.
        if (top_t.defined() && top_scale_t.defined() &&
            (top_t.dtype() == BoundaryDtype::INT8 || top_t.dtype() == BoundaryDtype::FP16))
            copy_int8_scales(start, len, gpu_start, cudaMemcpyHostToDevice, stream);
        if (dim == 2)
        {
            size_t top_var_block = top_t.stride(0);
            size_t top_time_block = top_t.stride(1);
            size_t left_var_block = left_t.stride(0);
            size_t left_time_block = left_t.stride(1);
            size_t top_gpu_var_block = top_gpu.stride(0);
            size_t left_gpu_var_block = left_gpu.stride(0);

            size_t es = top_t.element_size();
            size_t top_bytes = (size_t)len * top_time_block * es;
            size_t left_bytes = (size_t)len * left_time_block * es;
            size_t top_gpu_time_block = top_gpu.stride(1);
            size_t left_gpu_time_block = left_gpu.stride(1);

            SWEEP_CHECK(boundary_storage_dtype_ok(top_t), "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
            boundary_bounds_check("top_t", top_t, (int64_t)start * top_time_block, top_bytes);
            boundary_bounds_check("top_gpu", top_gpu, (int64_t)gpu_start * top_gpu_time_block, top_bytes);
            copy_2d_chunk_async(
                boundary_byte_ptr(top_gpu, (int64_t)gpu_start * top_gpu_time_block),
                top_gpu_var_block,
                boundary_byte_ptr(top_t, (int64_t)start * top_time_block),
                top_var_block,
                top_bytes,
                es,
                cudaMemcpyHostToDevice,
                stream
            );
            boundary_bounds_check("bottom_t", bottom_t, (int64_t)start * top_time_block, top_bytes);
            boundary_bounds_check("bottom_gpu", bottom_gpu, (int64_t)gpu_start * top_gpu_time_block, top_bytes);
            copy_2d_chunk_async(
                boundary_byte_ptr(bottom_gpu, (int64_t)gpu_start * top_gpu_time_block),
                top_gpu_var_block,
                boundary_byte_ptr(bottom_t, (int64_t)start * top_time_block),
                top_var_block,
                top_bytes,
                es,
                cudaMemcpyHostToDevice,
                stream
            );
            boundary_bounds_check("left_t", left_t, (int64_t)start * left_time_block, left_bytes);
            boundary_bounds_check("left_gpu", left_gpu, (int64_t)gpu_start * left_gpu_time_block, left_bytes);
            copy_2d_chunk_async(
                boundary_byte_ptr(left_gpu, (int64_t)gpu_start * left_gpu_time_block),
                left_gpu_var_block,
                boundary_byte_ptr(left_t, (int64_t)start * left_time_block),
                left_var_block,
                left_bytes,
                es,
                cudaMemcpyHostToDevice,
                stream
            );
            boundary_bounds_check("right_t", right_t, (int64_t)start * left_time_block, left_bytes);
            boundary_bounds_check("right_gpu", right_gpu, (int64_t)gpu_start * left_gpu_time_block, left_bytes);
            copy_2d_chunk_async(
                boundary_byte_ptr(right_gpu, (int64_t)gpu_start * left_gpu_time_block),
                left_gpu_var_block,
                boundary_byte_ptr(right_t, (int64_t)start * left_time_block),
                left_var_block,
                left_bytes,
                es,
                cudaMemcpyHostToDevice,
                stream
            );
            return;
        }

        size_t top_block    = top_t.stride(0);
        size_t front_block  = front_t.stride(0);
        size_t left_block   = left_t.stride(0);

        size_t top_elems    = len * nvar * top_block;
        size_t front_elems  = len * nvar * front_block;
        size_t left_elems   = len * nvar * left_block;
        size_t top_gpu_block = top_gpu.stride(0);
        size_t front_gpu_block = front_gpu.stride(0);
        size_t left_gpu_block = left_gpu.stride(0);
        size_t top_gpu_offset = static_cast<size_t>(gpu_start) * nvar * top_gpu_block;
        size_t front_gpu_offset = static_cast<size_t>(gpu_start) * nvar * front_gpu_block;
        size_t left_gpu_offset = static_cast<size_t>(gpu_start) * nvar * left_gpu_block;

        SWEEP_CHECK(boundary_storage_dtype_ok(top_t), "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
        size_t es = top_t.element_size();
        boundary_bounds_check("top_t", top_t, (int64_t)(start * nvar * top_block), top_elems * es);
        boundary_bounds_check("top_gpu", top_gpu, (int64_t)(top_gpu_offset), top_elems * es);
        boundary_memcpy_async(
            boundary_byte_ptr(top_gpu, (int64_t)top_gpu_offset),
            boundary_byte_ptr(top_t, (int64_t)start * nvar * top_block),
            top_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        boundary_bounds_check("bottom_t", bottom_t, (int64_t)(start * nvar * top_block), top_elems * es);
        boundary_bounds_check("bottom_gpu", bottom_gpu, (int64_t)(top_gpu_offset), top_elems * es);
        boundary_memcpy_async(
            boundary_byte_ptr(bottom_gpu, (int64_t)top_gpu_offset),
            boundary_byte_ptr(bottom_t, (int64_t)start * nvar * top_block),
            top_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        boundary_bounds_check("front_t", front_t, (int64_t)(start * nvar * front_block), front_elems * es);
        boundary_bounds_check("front_gpu", front_gpu, (int64_t)(front_gpu_offset), front_elems * es);
        boundary_memcpy_async(
            boundary_byte_ptr(front_gpu, (int64_t)front_gpu_offset),
            boundary_byte_ptr(front_t, (int64_t)start * nvar * front_block),
            front_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        boundary_bounds_check("back_t", back_t, (int64_t)(start * nvar * front_block), front_elems * es);
        boundary_bounds_check("back_gpu", back_gpu, (int64_t)(front_gpu_offset), front_elems * es);
        boundary_memcpy_async(
            boundary_byte_ptr(back_gpu, (int64_t)front_gpu_offset),
            boundary_byte_ptr(back_t, (int64_t)start * nvar * front_block),
            front_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        boundary_bounds_check("left_t", left_t, (int64_t)(start * nvar * left_block), left_elems * es);
        boundary_bounds_check("left_gpu", left_gpu, (int64_t)(left_gpu_offset), left_elems * es);
        boundary_memcpy_async(
            boundary_byte_ptr(left_gpu, (int64_t)left_gpu_offset),
            boundary_byte_ptr(left_t, (int64_t)start * nvar * left_block),
            left_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        boundary_bounds_check("right_t", right_t, (int64_t)(start * nvar * left_block), left_elems * es);
        boundary_bounds_check("right_gpu", right_gpu, (int64_t)(left_gpu_offset), left_elems * es);
        boundary_memcpy_async(
            boundary_byte_ptr(right_gpu, (int64_t)left_gpu_offset),
            boundary_byte_ptr(right_t, (int64_t)start * nvar * left_block),
            left_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );
    }

};
