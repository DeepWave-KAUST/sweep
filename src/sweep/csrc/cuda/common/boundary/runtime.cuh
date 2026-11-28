#pragma once

#include <chrono>
#include "../../launch/launch.h"   // SWEEP_LAUNCH
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

#include "saver.cuh"
#include "../../../core/check.h"

// Bind one staged boundary face to the typed pointer matching the staging
// tensor's dtype (fp32/fp16/bf16) at element offset OFF.  Mirrors view()'s
// GPU-direct binding so the existing dtype-dispatched save/restore kernels work
// for the cpu/disk staging ring too.  OFF is an element offset (dtype-agnostic).
// A macro (not a function) so it assigns directly to the __restrict__-qualified
// pointer members of GeneralBoundaryPointer (a reference param would drop the
// restrict qualifier).
#define BIND_STAGED_FACE(STG, OFF, PF, PH, PBF)                                \
    do {                                                                       \
        const Buf& _stg = (STG);                                               \
        const int64_t _off = (OFF);                                            \
        const auto _st = _stg.scalar_type();                                   \
        if (_st == BoundaryDtype::FP16)                                        \
            (PH) = reinterpret_cast<__half*>(_stg.data_ptr()) + _off;          \
        else if (_st == BoundaryDtype::BF16)                                   \
            (PBF) = reinterpret_cast<__nv_bfloat16*>(_stg.data_ptr()) + _off;  \
        else                                                                   \
            (PF) = reinterpret_cast<float*>(_stg.data_ptr()) + _off;           \
    } while (0)

// Bind one staged INT8 boundary face: the uint8 main RING at element
// (cell) offset QOFF, the FP32 per-block scale RING at block offset SOFF,
// and the shared FP32 transient staging buffer TRANS (no offset — one
// timestep, reused every step).  The two-pass save/restore in
// quantize_step/dequantize_step then compresses TRANS -> ring (QOFF/SOFF
// pre-applied, so it is called with step_idx == 0).  RING_Q / RING_S are
// the saver's ring tensors; PQ/PS/PT are GeneralBoundaryPointer members.
#define BIND_STAGED_INT8(RING_Q, QOFF, RING_S, SOFF, TRANS, PQ, PS, PT)        \
    do {                                                                       \
        (PQ) = reinterpret_cast<uint8_t*>((RING_Q).data_ptr()) + (QOFF);       \
        (PS) = (RING_S).data_ptr<float>() + (SOFF);                            \
        (PT) = (TRANS);                                                         \
    } while (0)

// Dtype-aware variant shared by the INT8 and scaled-FP16 paths: binds
// the persistent payload ring to the *_q (uint8) or *_h (__half) field
// depending on DT; scale and staging handling is identical.
#define BIND_STAGED_SCALED(DT, RING, OFF, RING_S, SOFF, TRANS, PQ, PH, PS, PT) \
    do {                                                                       \
        if ((DT) == BoundaryDtype::FP16)                                       \
            (PH) = reinterpret_cast<__half*>((RING).data_ptr()) + (OFF);       \
        else                                                                   \
            (PQ) = reinterpret_cast<uint8_t*>((RING).data_ptr()) + (OFF);      \
        (PS) = (RING_S).data_ptr<float>() + (SOFF);                            \
        (PT) = (TRANS);                                                         \
    } while (0)

struct BoundaryChunk {
    int it = 0;
    int buf_idx = 0;
    int chunk_id = 0;
    int ring_slot = 0;
    int ring_start = 0;
    int chunk_start = 0;
    int chunk_len = 0;
    bool is_chunk_end = false;
};

class BoundaryRuntime {
public:
    BoundaryRuntime(
        EffectiveBoundarySaver& saver,
        int dim,
        bool use_boundary_saving,
        bool boundary_on_cpu,
        bool boundary_on_disk,
        bool boundary_disk_async_read,
        int transfer_interval,
        int ring_buffers,
        const std::vector<std::string>& disk_files,
        cudaStream_t compute_stream,
        cudaStream_t copy_stream
    )
        : saver_(&saver),
          dim_(dim),
          enabled_(use_boundary_saving),
          staged_(boundary_on_cpu || boundary_on_disk || saver.compress),
          compress_(use_boundary_saving && saver.compress),
          compress_tier_(boundary_on_disk ? 2 : boundary_on_cpu ? 1 : 0),
          boundary_on_disk_(boundary_on_disk),
          boundary_disk_async_read_(boundary_on_disk && boundary_disk_async_read),
          transfer_interval_(transfer_interval > 0 ? transfer_interval : 1),
          ring_buffers_(ring_buffers > 0 ? ring_buffers : 1),
          disk_files_(&disk_files),
          compute_stream_(compute_stream),
          copy_stream_(copy_stream)
    {
        profile_enabled_ = std::getenv("SWEEP_BOUNDARY_PROFILE") != nullptr;
        // Escape hatch back to the full-grid scan kernels.  The compact and
        // scan paths give bit-identical results: a cell in two or three bands
        // (an edge or corner of the strip) ends up with the value of the scan
        // kernel's last face write -- every face holds the same bits for fp32 /
        // bf16 storage, and the int8 / scaled-fp16 restore writes it from that
        // face only (boundary_kernel3d_compact_owned).  So this exists to prove
        // that in a test (test_boundary_compact_kernel.py,
        // test_host_core.py::TestLossyBoundary3DRestore) and to isolate the
        // compact kernels if they are ever suspected, not to change results.
        no_compact_kernel_ = std::getenv("SWEEP_BOUNDARY_NO_COMPACT") != nullptr;
        if (!enabled_ || !staged_)
            return;

        SWEEP_CHECK(
            !boundary_disk_async_read_ || ring_buffers_ >= 2,
            "boundary_disk_async_read requires boundary_ring_buffers >= 2."
        );

        compute_ready_.resize(ring_buffers_, nullptr);
        copy_ready_.resize(ring_buffers_, nullptr);
        forward_copy_start_.resize(ring_buffers_, nullptr);
        backward_copy_start_.resize(ring_buffers_, nullptr);
        backward_h2d_start_.resize(ring_buffers_, nullptr);
        copy_pending_.resize(ring_buffers_, 0);
        forward_copy_profile_pending_.resize(ring_buffers_, 0);
        backward_copy_profile_pending_.resize(ring_buffers_, 0);
        for (int i = 0; i < ring_buffers_; ++i) {
            cudaEventCreateWithFlags(&compute_ready_[i], cudaEventDisableTiming);
            if (profile_enabled_ && boundary_on_disk_)
                cudaEventCreate(&copy_ready_[i]);
            else
                cudaEventCreateWithFlags(&copy_ready_[i], cudaEventDisableTiming);
            cudaEventCreate(&forward_copy_start_[i]);
            cudaEventCreate(&backward_copy_start_[i]);
            cudaEventCreate(&backward_h2d_start_[i]);
            cudaEventRecord(compute_ready_[i], compute_stream_);
        }

        if (compress_)
            init_compress();

        if (boundary_disk_async_read_)
            disk_reader_thread_ = std::thread(&BoundaryRuntime::disk_reader_loop, this);

        // Start from a clean write-error slot. On the per-call path this
        // runtime is a BoundaryScope local rebuilt every call, so this bounds a
        // recorded-but-never-reported failure to the run that caused it.
        reset_boundary_disk_write_error();
    }

    ~BoundaryRuntime()
    {
        stop_disk_reader_no_throw();
        print_profile_if_needed();
        for (auto event : compute_ready_) {
            if (event != nullptr)
                cudaEventDestroy(event);
        }
        for (auto event : copy_ready_) {
            if (event != nullptr)
                cudaEventDestroy(event);
        }
        for (auto event : forward_copy_start_) {
            if (event != nullptr)
                cudaEventDestroy(event);
        }
        for (auto event : backward_copy_start_) {
            if (event != nullptr)
                cudaEventDestroy(event);
        }
        for (auto event : backward_h2d_start_) {
            if (event != nullptr)
                cudaEventDestroy(event);
        }
        for (auto event : comp_pack_ready_) {
            if (event != nullptr)
                cudaEventDestroy(event);
        }
    }

    // Rebind the two objects that only exist for one call. The persistent
    // runtime calls this once on entry to forward/backward; the non-persistent
    // path never calls it, so its behaviour is unchanged.
    inline void rebind(EffectiveBoundarySaver& saver,
                       const std::vector<std::string>& disk_files)
    {
        saver_ = &saver;
        disk_files_ = &disk_files;
        // The piece tables hold raw device pointers into this call's ring and
        // workspace; a session that is re-pointed at other tensors rebuilds them.
        if (compress_ && (saver.top_gpu.data_ptr() != comp_ring_ptr_ ||
                          saver.comp_tab.data_ptr() != comp_tab_ptr_))
            build_compress_tables();
    }

    inline bool enabled() const { return enabled_; }
    inline bool staged() const { return staged_; }
    inline bool compress_enabled() const { return compress_; }

    // A new forward phase starts the arena from byte 0 (the session reuses one
    // runtime across phases; the per-call path builds a fresh one anyway).
    inline void begin_forward()
    {
        if (!compress_) return;
        pending_land_.clear();
        comp_live_.clear();
        comp_clear_live_flags();
        comp_open_valid_ = false;
        arena_fill_ = 0;
        comp_last_chunk_ = -1;
        comp_last_len_ = 0;
        comp_packed_bytes_ = 0;
        comp_raw_bytes_ = 0;
        comp_chunks_landed_ = 0;
        int64_t* stats = reinterpret_cast<int64_t*>(saver_->comp_stats_host.data_ptr());
        stats[4] = stats[5] = stats[6] = stats[7] = 0;   // the peak live set of this call starts over
    }

    inline BoundaryChunk forward_chunk(int it, int nt) const
    {
        BoundaryChunk chunk;
        chunk.it = it;
        chunk.buf_idx = it % transfer_interval_;
        chunk.chunk_id = it / transfer_interval_;
        chunk.ring_slot = chunk.chunk_id % ring_buffers_;
        chunk.ring_start = chunk.ring_slot * transfer_interval_;
        chunk.chunk_start = it - chunk.buf_idx;
        chunk.chunk_len = chunk.buf_idx + 1;
        chunk.is_chunk_end = chunk.buf_idx == transfer_interval_ - 1 || it == nt - 1;
        return chunk;
    }

    // Scaled-storage helpers: launch quantize / dequantize for one
    // timestep's worth of data per face.  ``step_idx`` is the index
    // along the step axis of the persistent payload buffer (dim 0 in 3-D,
    // already including any multi-field offset; dim 1 in 2-D).  The payload
    // is uint8 (INT8, *_q) or __half (FP16, *_h); the FP32 per-block
    // scale and the FP32 transient staging are shared by both.
    //
    // The geometry comes from the saver's persistent payload / scale tensors:
    // the step pitch is the step axis' stride, and the shots are the next
    // axis.  The propagator allocates these buffers for its capacity batch and
    // narrows them to the call's batch on that axis (_c.py
    // _slice_boundary_buffers), so size() there is this call's shot count
    // while the step pitch stays capacity-sized -- the quantization therefore
    // covers this call's shots only, each blocked on its own
    // (launch_quantize_*), and never reads the staging past them.
    // allocate_int8_staging checks that the payload, its scale and the
    // staging agree on that per-shot geometry.
    //
    // ``face_t`` is the face's persistent saver tensor, also used to skip a
    // DD cut face: ``gpu_full_shapes`` drops such a face to numel 0
    // because the boundary kernels gate it on ctx.cut_* (nothing is ever
    // written there — its halo is reconstructed in the DD backward).  The test
    // MUST be on numel, not on a stride: PyTorch clamps a 0-size dim's stride
    // to a nonzero value -- and Buf copies that stride verbatim rather than
    // deriving it -- so a stride-derived cell count would still drive a
    // launch that writes into the empty buffer.  Guarding here covers the FP16
    // and INT8 payloads in one place.
    inline bool face_is_cut(const Buf& face_t) const
    {
        return face_t.numel() == 0;
    }
    inline void quantize_face(const Buf& face_t, const Buf& scale_t,
                              float* stage, uint8_t* q, __half* h, float* scale,
                              BoundaryDtype dt, int64_t step_idx)
    {
        if (face_is_cut(face_t)) return;
        const int sd = saver_->dim == 3 ? 0 : 1;
        const int64_t cells = face_t.stride(sd), blocks = scale_t.stride(sd);
        const int64_t shots = face_t.size(sd + 1), per_shot = face_t.stride(sd + 1);
        if (dt == BoundaryDtype::FP16)
            launch_quantize_fp16(stage, h + step_idx * cells, scale + step_idx * blocks, shots, per_shot, compute_stream_);
        else
            launch_quantize_int8(stage, q + step_idx * cells, scale + step_idx * blocks, shots, per_shot, compute_stream_);
    }
    inline void dequantize_face(const Buf& face_t, const Buf& scale_t,
                                float* stage, uint8_t* q, __half* h, float* scale,
                                BoundaryDtype dt, int64_t step_idx)
    {
        if (face_is_cut(face_t)) return;
        const int sd = saver_->dim == 3 ? 0 : 1;
        const int64_t cells = face_t.stride(sd), blocks = scale_t.stride(sd);
        const int64_t shots = face_t.size(sd + 1), per_shot = face_t.stride(sd + 1);
        if (dt == BoundaryDtype::FP16)
            launch_dequantize_fp16(h + step_idx * cells, scale + step_idx * blocks, stage, shots, per_shot, compute_stream_);
        else
            launch_dequantize_int8(q + step_idx * cells, scale + step_idx * blocks, stage, shots, per_shot, compute_stream_);
    }
    inline void quantize_step_2d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        const auto* s = saver_;
        quantize_face(s->top_t,    s->top_scale_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx);
        quantize_face(s->bottom_t, s->bottom_scale_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx);
        quantize_face(s->left_t,   s->left_scale_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx);
        quantize_face(s->right_t,  s->right_scale_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx);
    }
    inline void dequantize_step_2d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        const auto* s = saver_;
        dequantize_face(s->top_t,    s->top_scale_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx);
        dequantize_face(s->bottom_t, s->bottom_scale_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx);
        dequantize_face(s->left_t,   s->left_scale_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx);
        dequantize_face(s->right_t,  s->right_scale_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx);
    }
    inline void quantize_step_3d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        const auto* s = saver_;
        quantize_face(s->top_t,    s->top_scale_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx);
        quantize_face(s->bottom_t, s->bottom_scale_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx);
        quantize_face(s->front_t,  s->front_scale_t,  b.front, b.front_q, b.front_h, b.front_scale, b.dtype, step_idx);
        quantize_face(s->back_t,   s->back_scale_t,   b.back,  b.back_q,  b.back_h,  b.back_scale,  b.dtype, step_idx);
        quantize_face(s->left_t,   s->left_scale_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx);
        quantize_face(s->right_t,  s->right_scale_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx);
    }
    inline void dequantize_step_3d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        const auto* s = saver_;
        dequantize_face(s->top_t,    s->top_scale_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx);
        dequantize_face(s->bottom_t, s->bottom_scale_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx);
        dequantize_face(s->front_t,  s->front_scale_t,  b.front, b.front_q, b.front_h, b.front_scale, b.dtype, step_idx);
        dequantize_face(s->back_t,   s->back_scale_t,   b.back,  b.back_q,  b.back_h,  b.back_scale,  b.dtype, step_idx);
        dequantize_face(s->left_t,   s->left_scale_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx);
        dequantize_face(s->right_t,  s->right_scale_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx);
    }

    inline void wait_before_forward_save(const BoundaryChunk& chunk)
    {
        if (!enabled_ || !staged_ || chunk.buf_idx != 0)
            return;
        if (!copy_pending_[chunk.ring_slot])
            return;

        if (profile_enabled_) {
            auto wait_start = Clock::now();
            cudaEventSynchronize(copy_ready_[chunk.ring_slot]);
            profile_forward_slot_reuse_wait_s_ += elapsed_seconds(wait_start, Clock::now());
        }
        accumulate_forward_copy_time(chunk.ring_slot);
        cudaStreamWaitEvent(compute_stream_, copy_ready_[chunk.ring_slot], 0);
        copy_pending_[chunk.ring_slot] = 0;
    }

    inline GeneralBoundaryPointer forward_save_ptrs(
        const BoundaryChunk& chunk,
        const GeneralBoundaryPointer& direct
    )
    {
        if (!staged_)
            return direct;

        int gpu_idx = chunk.ring_start + chunk.buf_idx;
        GeneralBoundaryPointer ptr{};
        ptr.dtype = boundary_dtype_from_tensor(saver_->top_gpu);
        ptr.last_two = direct.last_two;
        if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
            // Staged INT8: uint8 ring (cells/step) + scale ring (blocks/step)
            // + shared FP32 transient (direct.*).  Scale ring per-step stride
            // is its outer-time stride (dim 0 in 3D, dim 1 in 2D).
            const int sd = (dim_ == 3) ? 0 : 1;
            BIND_STAGED_SCALED(ptr.dtype, saver_->top_gpu,    gpu_idx * saver_->top_stride,    saver_->top_scale_gpu,    gpu_idx * saver_->top_scale_gpu.stride(sd),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
            BIND_STAGED_SCALED(ptr.dtype, saver_->bottom_gpu, gpu_idx * saver_->bottom_stride, saver_->bottom_scale_gpu, gpu_idx * saver_->bottom_scale_gpu.stride(sd), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
            BIND_STAGED_SCALED(ptr.dtype, saver_->left_gpu,   gpu_idx * saver_->left_stride,   saver_->left_scale_gpu,   gpu_idx * saver_->left_scale_gpu.stride(sd),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
            BIND_STAGED_SCALED(ptr.dtype, saver_->right_gpu,  gpu_idx * saver_->right_stride,  saver_->right_scale_gpu,  gpu_idx * saver_->right_scale_gpu.stride(sd),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            if (dim_ == 3) {
                BIND_STAGED_SCALED(ptr.dtype, saver_->front_gpu, gpu_idx * saver_->front_stride, saver_->front_scale_gpu, gpu_idx * saver_->front_scale_gpu.stride(sd), direct.front, ptr.front_q, ptr.front_h, ptr.front_scale, ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_->back_gpu,  gpu_idx * saver_->back_stride,  saver_->back_scale_gpu,  gpu_idx * saver_->back_scale_gpu.stride(sd),  direct.back,  ptr.back_q,  ptr.back_h,  ptr.back_scale,  ptr.back);
            }
            return ptr;
        }
        BIND_STAGED_FACE(saver_->top_gpu,    gpu_idx * saver_->top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
        BIND_STAGED_FACE(saver_->bottom_gpu, gpu_idx * saver_->bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
        BIND_STAGED_FACE(saver_->left_gpu,   gpu_idx * saver_->left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
        BIND_STAGED_FACE(saver_->right_gpu,  gpu_idx * saver_->right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);

        if (dim_ == 3) {
            BIND_STAGED_FACE(saver_->front_gpu, gpu_idx * saver_->front_stride, ptr.front, ptr.front_h, ptr.front_bf);
            BIND_STAGED_FACE(saver_->back_gpu,  gpu_idx * saver_->back_stride,  ptr.back,  ptr.back_h,  ptr.back_bf);
        }
        return ptr;
    }

    inline GeneralBoundaryPointer forward_save_ptrs_field(
        const BoundaryChunk& chunk,
        const GeneralBoundaryPointer& direct,
        int field_idx
    )
    {
        SWEEP_CHECK(field_idx >= 0 && field_idx < saver_->nvar, "Invalid boundary field index.");

        if (dim_ == 2) {
            GeneralBoundaryPointer ptr{};
            ptr.last_two = direct.last_two;
            ptr.use_fp16 = direct.use_fp16;
            ptr.dtype = direct.dtype;

            if (staged_) {
                int gpu_idx = chunk.ring_start + chunk.buf_idx;
                ptr.dtype = boundary_dtype_from_tensor(saver_->top_gpu);
                if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                    BIND_STAGED_SCALED(ptr.dtype, saver_->top_gpu,
                        field_idx * saver_->top_gpu.stride(0) + gpu_idx * saver_->top_gpu.stride(1),
                        saver_->top_scale_gpu,
                        field_idx * saver_->top_scale_gpu.stride(0) + gpu_idx * saver_->top_scale_gpu.stride(1),
                        direct.top, ptr.top_q, ptr.top_h, ptr.top_scale, ptr.top);
                    BIND_STAGED_SCALED(ptr.dtype, saver_->bottom_gpu,
                        field_idx * saver_->bottom_gpu.stride(0) + gpu_idx * saver_->bottom_gpu.stride(1),
                        saver_->bottom_scale_gpu,
                        field_idx * saver_->bottom_scale_gpu.stride(0) + gpu_idx * saver_->bottom_scale_gpu.stride(1),
                        direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                    BIND_STAGED_SCALED(ptr.dtype, saver_->left_gpu,
                        field_idx * saver_->left_gpu.stride(0) + gpu_idx * saver_->left_gpu.stride(1),
                        saver_->left_scale_gpu,
                        field_idx * saver_->left_scale_gpu.stride(0) + gpu_idx * saver_->left_scale_gpu.stride(1),
                        direct.left, ptr.left_q, ptr.left_h, ptr.left_scale, ptr.left);
                    BIND_STAGED_SCALED(ptr.dtype, saver_->right_gpu,
                        field_idx * saver_->right_gpu.stride(0) + gpu_idx * saver_->right_gpu.stride(1),
                        saver_->right_scale_gpu,
                        field_idx * saver_->right_scale_gpu.stride(0) + gpu_idx * saver_->right_scale_gpu.stride(1),
                        direct.right, ptr.right_q, ptr.right_h, ptr.right_scale, ptr.right);
                } else {
                BIND_STAGED_FACE(saver_->top_gpu,
                    field_idx * saver_->top_gpu.stride(0) + gpu_idx * saver_->top_gpu.stride(1),
                    ptr.top, ptr.top_h, ptr.top_bf);
                BIND_STAGED_FACE(saver_->bottom_gpu,
                    field_idx * saver_->bottom_gpu.stride(0) + gpu_idx * saver_->bottom_gpu.stride(1),
                    ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
                BIND_STAGED_FACE(saver_->left_gpu,
                    field_idx * saver_->left_gpu.stride(0) + gpu_idx * saver_->left_gpu.stride(1),
                    ptr.left, ptr.left_h, ptr.left_bf);
                BIND_STAGED_FACE(saver_->right_gpu,
                    field_idx * saver_->right_gpu.stride(0) + gpu_idx * saver_->right_gpu.stride(1),
                    ptr.right, ptr.right_h, ptr.right_bf);
                }
            } else if (direct.dtype == BoundaryDtype::FP16) {
                ptr.top_h    = direct.top_h    + field_idx * saver_->top_t.stride(0);
                ptr.bottom_h = direct.bottom_h + field_idx * saver_->bottom_t.stride(0);
                ptr.left_h   = direct.left_h   + field_idx * saver_->left_t.stride(0);
                ptr.right_h  = direct.right_h  + field_idx * saver_->right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_->top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_->bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_->left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_->right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else if (direct.dtype == BoundaryDtype::BF16) {
                ptr.top_bf    = direct.top_bf    + field_idx * saver_->top_t.stride(0);
                ptr.bottom_bf = direct.bottom_bf + field_idx * saver_->bottom_t.stride(0);
                ptr.left_bf   = direct.left_bf   + field_idx * saver_->left_t.stride(0);
                ptr.right_bf  = direct.right_bf  + field_idx * saver_->right_t.stride(0);
            } else if (direct.dtype == BoundaryDtype::INT8) {
                ptr.top_q       = direct.top_q       + field_idx * saver_->top_t.stride(0);
                ptr.bottom_q    = direct.bottom_q    + field_idx * saver_->bottom_t.stride(0);
                ptr.left_q      = direct.left_q      + field_idx * saver_->left_t.stride(0);
                ptr.right_q     = direct.right_q     + field_idx * saver_->right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_->top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_->bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_->left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_->right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else {
                ptr.top = direct.top + field_idx * saver_->top_t.stride(0);
                ptr.bottom = direct.bottom + field_idx * saver_->bottom_t.stride(0);
                ptr.left = direct.left + field_idx * saver_->left_t.stride(0);
                ptr.right = direct.right + field_idx * saver_->right_t.stride(0);
            }
            return ptr;
        }

        int time_idx = staged_ ? (chunk.ring_start + chunk.buf_idx) : chunk.it;
        int64_t flat_idx = static_cast<int64_t>(time_idx) * saver_->nvar + field_idx;

        GeneralBoundaryPointer ptr{};
        ptr.use_fp16 = direct.use_fp16;
            ptr.dtype = direct.dtype;
        if (!staged_ && direct.dtype == BoundaryDtype::FP16) {
            ptr.top_h    = direct.top_h    + flat_idx * saver_->top_stride;
            ptr.bottom_h = direct.bottom_h + flat_idx * saver_->bottom_stride;
            ptr.front_h  = direct.front_h  + flat_idx * saver_->front_stride;
            ptr.back_h   = direct.back_h   + flat_idx * saver_->back_stride;
            ptr.left_h   = direct.left_h   + flat_idx * saver_->left_stride;
            ptr.right_h  = direct.right_h  + flat_idx * saver_->right_stride;
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_->top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_->bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_->front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_->back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_->left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_->right_scale_t.stride(0);
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (!staged_ && direct.dtype == BoundaryDtype::BF16) {
            ptr.top_bf    = direct.top_bf    + flat_idx * saver_->top_stride;
            ptr.bottom_bf = direct.bottom_bf + flat_idx * saver_->bottom_stride;
            ptr.front_bf  = direct.front_bf  + flat_idx * saver_->front_stride;
            ptr.back_bf   = direct.back_bf   + flat_idx * saver_->back_stride;
            ptr.left_bf   = direct.left_bf   + flat_idx * saver_->left_stride;
            ptr.right_bf  = direct.right_bf  + flat_idx * saver_->right_stride;
        } else if (!staged_ && direct.dtype == BoundaryDtype::INT8) {
            // 3D persistent uint8 buffer: outermost index = time*nvar + field.
            // top_stride/bottom_stride/... already track per-(time×field)
            // cell counts (compute_time_strides → tensor.stride(0)).
            ptr.top_q    = direct.top_q    + flat_idx * saver_->top_stride;
            ptr.bottom_q = direct.bottom_q + flat_idx * saver_->bottom_stride;
            ptr.front_q  = direct.front_q  + flat_idx * saver_->front_stride;
            ptr.back_q   = direct.back_q   + flat_idx * saver_->back_stride;
            ptr.left_q   = direct.left_q   + flat_idx * saver_->left_stride;
            ptr.right_q  = direct.right_q  + flat_idx * saver_->right_stride;
            // Scale buffer outermost stride.
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_->top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_->bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_->front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_->back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_->left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_->right_scale_t.stride(0);
            // Staging shared across (time, field) — no offset.
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (staged_) {
            ptr.dtype = boundary_dtype_from_tensor(saver_->top_gpu);
            if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                // 3D staged INT8: uint8 ring + scale ring indexed by the same
                // flat (gpu_time*nvar + field) index; shared FP32 transient.
                BIND_STAGED_SCALED(ptr.dtype, saver_->top_gpu,    flat_idx * saver_->top_stride,    saver_->top_scale_gpu,    flat_idx * saver_->top_scale_gpu.stride(0),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
                BIND_STAGED_SCALED(ptr.dtype, saver_->bottom_gpu, flat_idx * saver_->bottom_stride, saver_->bottom_scale_gpu, flat_idx * saver_->bottom_scale_gpu.stride(0), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                BIND_STAGED_SCALED(ptr.dtype, saver_->front_gpu,  flat_idx * saver_->front_stride,  saver_->front_scale_gpu,  flat_idx * saver_->front_scale_gpu.stride(0),  direct.front,  ptr.front_q,  ptr.front_h,  ptr.front_scale,  ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_->back_gpu,   flat_idx * saver_->back_stride,   saver_->back_scale_gpu,   flat_idx * saver_->back_scale_gpu.stride(0),   direct.back,   ptr.back_q,   ptr.back_h,   ptr.back_scale,   ptr.back);
                BIND_STAGED_SCALED(ptr.dtype, saver_->left_gpu,   flat_idx * saver_->left_stride,   saver_->left_scale_gpu,   flat_idx * saver_->left_scale_gpu.stride(0),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
                BIND_STAGED_SCALED(ptr.dtype, saver_->right_gpu,  flat_idx * saver_->right_stride,  saver_->right_scale_gpu,  flat_idx * saver_->right_scale_gpu.stride(0),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            } else {
            BIND_STAGED_FACE(saver_->top_gpu,    flat_idx * saver_->top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
            BIND_STAGED_FACE(saver_->bottom_gpu, flat_idx * saver_->bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
            BIND_STAGED_FACE(saver_->front_gpu,  flat_idx * saver_->front_stride,  ptr.front,  ptr.front_h,  ptr.front_bf);
            BIND_STAGED_FACE(saver_->back_gpu,   flat_idx * saver_->back_stride,   ptr.back,   ptr.back_h,   ptr.back_bf);
            BIND_STAGED_FACE(saver_->left_gpu,   flat_idx * saver_->left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
            BIND_STAGED_FACE(saver_->right_gpu,  flat_idx * saver_->right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);
            }
        } else {
            ptr.top = direct.top + flat_idx * saver_->top_stride;
            ptr.bottom = direct.bottom + flat_idx * saver_->bottom_stride;
            ptr.front = direct.front + flat_idx * saver_->front_stride;
            ptr.back = direct.back + flat_idx * saver_->back_stride;
            ptr.left = direct.left + flat_idx * saver_->left_stride;
            ptr.right = direct.right + flat_idx * saver_->right_stride;
        }
        ptr.last_two = direct.last_two;
        return ptr;
    }

    inline int boundary_time_index(const BoundaryChunk& chunk) const
    {
        return staged_ ? 0 : chunk.it;
    }

    inline int boundary_time_index_field(const BoundaryChunk& chunk) const
    {
        return dim_ == 3 ? 0 : boundary_time_index(chunk);
    }

    inline void flush_forward_if_needed(const BoundaryChunk& chunk)
    {
        if (!enabled_ || !staged_)
            return;
        if (!chunk.is_chunk_end) {
            // A segmented ring's refill ends mid-chunk; the chunk is compressed
            // when the phase ends (synchronize) rather than never.
            if (compress_) { comp_open_ = chunk; comp_open_valid_ = true; }
            return;
        }
        comp_open_valid_ = false;

        cudaEventRecord(compute_ready_[chunk.ring_slot], compute_stream_);
        cudaStreamWaitEvent(copy_stream_, compute_ready_[chunk.ring_slot], 0);
        if (profile_enabled_)
            cudaEventRecord(forward_copy_start_[chunk.ring_slot], copy_stream_);

        if (compress_) {
            compress_chunk(chunk);
            return;
        }

        if (boundary_on_disk_) {
            saver_->flush_gpu_to_disk(
                chunk.chunk_start,
                chunk.chunk_len,
                (*disk_files_),
                copy_stream_,
                chunk.ring_start,
                chunk.ring_start
            );
        } else {
            saver_->flush_gpu_to_cpu(
                chunk.chunk_start,
                chunk.chunk_len,
                copy_stream_,
                chunk.ring_start
            );
        }

        cudaEventRecord(copy_ready_[chunk.ring_slot], copy_stream_);
        copy_pending_[chunk.ring_slot] = 1;
        if (profile_enabled_ && staged_)
            ++profile_forward_chunks_;
        if (profile_enabled_ && boundary_on_disk_)
            forward_copy_profile_pending_[chunk.ring_slot] = 1;
    }

    inline void save_forward_2d(
        int it,
        int nt,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx
    )
    {
        BoundaryChunk chunk = forward_chunk(it, nt);
        wait_before_forward_save(chunk);
        auto b = forward_save_ptrs(chunk, direct);

        if (b.dtype == BoundaryDtype::BF16) {
            SWEEP_LAUNCH(boundary_kernel2d_bf16, grid, block,
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            // INT8 2-pass save: FP32 kernel → staging (1-step), then
            // quantize → persistent uint8 + per-block FP32 scale.
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
            int64_t bt = boundary_time_index(chunk);
            quantize_step_2d(b, bt);
        } else if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_2d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            SWEEP_LAUNCH(boundary_kernel2d_compact, compact_blocks, compact_threads,
                u, b.top, b.bottom, b.left, b.right,
                boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
        } else {
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
        }

        flush_forward_if_needed(chunk);
    }

    inline void save_forward_3d(
        int it,
        int nt,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx
    )
    {
        BoundaryChunk chunk = forward_chunk(it, nt);
        wait_before_forward_save(chunk);
        auto b = forward_save_ptrs(chunk, direct);

        if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_compact_bf16, compact_blocks, compact_threads,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                SWEEP_LAUNCH(boundary_kernel3d_compact, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index(chunk));
            } else {
                SWEEP_LAUNCH(boundary_kernel3d_compact, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_bf16, grid, block,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index(chunk));
            } else {
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
            }
        }

        flush_forward_if_needed(chunk);
    }

    inline void save_forward_2d_field(
        int it,
        int nt,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx,
        int field_idx,
        bool flush_chunk,
        int tangent_pad = 0    // bands run this many cells past the box (the qp2d frame)
    )
    {
        BoundaryChunk chunk = forward_chunk(it, nt);
        wait_before_forward_save(chunk);
        auto b = forward_save_ptrs_field(chunk, direct, field_idx);

        if (b.dtype == BoundaryDtype::BF16) {
            SWEEP_LAUNCH(boundary_kernel2d_bf16, grid, block,
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE, tangent_pad);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_SAVE, tangent_pad);
            quantize_step_2d(b, boundary_time_index_field(chunk));
        } else if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_2d(ctx, width, tangent_pad);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            SWEEP_LAUNCH(boundary_kernel2d_compact, compact_blocks, compact_threads,
                u, b.top, b.bottom, b.left, b.right,
                boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE, tangent_pad);
        } else {
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE, tangent_pad);
        }

        if (flush_chunk)
            flush_forward_if_needed(chunk);
    }

    inline void save_forward_3d_field(
        int it,
        int nt,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx,
        int field_idx,
        bool flush_chunk
    )
    {
        BoundaryChunk chunk = forward_chunk(it, nt);
        wait_before_forward_save(chunk);
        auto b = forward_save_ptrs_field(chunk, direct, field_idx);

        if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_compact_bf16, compact_blocks, compact_threads,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                SWEEP_LAUNCH(boundary_kernel3d_compact, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index_field(chunk));
            } else {
                SWEEP_LAUNCH(boundary_kernel3d_compact, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_bf16, grid, block,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index_field(chunk));
            } else {
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            }
        }

        if (flush_chunk)
            flush_forward_if_needed(chunk);
    }

    // ``nt_saved`` is the ring length in SAVED-step coordinates (nt - bs_it0
    // under boundary tail truncation, nt otherwise); ``it_hi_saved`` is this
    // backward call's segment top in the same coordinates.  The first step the
    // call processes is it_hi_saved-1, whose chunk has to be in the ring before
    // the first restore.
    //
    // The monolithic backward runs the whole reverse loop in one C++ call, so
    // it_hi_saved == nt_saved and the default reproduces the old nt_saved-1
    // exactly.  The DD backward is driven one step per call from Python
    // (SteppedBackwardRunner.run_segment(it+1, it)), so each call must prime
    // the ring for ITS step -- that is what makes cpu staging work under the
    // per-step loop instead of prefetching the tail chunk every time.
    inline void prefetch_initial_backward_chunk(int nt, int it_hi = -1)
    {
        if (!enabled_ || !staged_)
            return;

        int it0 = (it_hi >= 0 ? it_hi : nt) - 1;
        int buf_idx0 = (it0 - 1) % transfer_interval_;
        int chunk_start = it0 - buf_idx0 - 1;
        int chunk_len = buf_idx0 + 1;
        backward_nt_ = nt;
        backward_active_ = true;
        // Nothing precedes step 0.  The monolithic backward primes the ring
        // once with it_hi == nt, so chunk_start lands at nt-2 and this never
        // trips.  The DD backward is driven one step per call, so the last
        // call arrives with it_hi == 1: it0 is 0, and chunk_start goes to -1 --
        // a staged copy from one chunk BEFORE the buffer, which segfaults the
        // host thread inside cudaMemcpyAsync.  The step it primes for is
        // already in the ring by then; there is nothing left to fetch.
        if (chunk_start < 0) {
            profile_compute_start_ = Clock::now();
            profile_have_compute_start_ = true;
            return;
        }
        if (compress_) comp_settle();
        prefetch_backward_chunk(chunk_start, chunk_len, backward_slot_for_chunk(chunk_start));
        profile_compute_start_ = Clock::now();
    }

    inline void wait_before_backward_restore(int it)
    {
        if (!enabled_ || !staged_)
            return;

        int buf_idx = (it - 1) % transfer_interval_;
        int chunk_start = it - 1 - buf_idx;
        int chunk_len = backward_chunk_len(chunk_start);
        if (buf_idx != chunk_len - 1)
            return;

        int chunk_slot = backward_slot_for_chunk(chunk_start);
        auto boundary_arrive = Clock::now();
        if (profile_enabled_ && profile_have_compute_start_)
            profile_compute_between_chunks_s_ += elapsed_seconds(profile_compute_start_, boundary_arrive);

        // Deferred fetch (see prefetch_next_backward_chunk_if_needed): the
        // chunk was not in the arena when its early prefetch came round.
        if (compress_ && comp_slot_chunk_[chunk_slot] != chunk_start / transfer_interval_) {
            comp_settle();
            if (boundary_disk_async_read_)
                start_async_disk_read(chunk_start, chunk_len, chunk_slot);
            else
                prefetch_backward_chunk(chunk_start, chunk_len, chunk_slot);
        }
        if (boundary_disk_async_read_) {
            wait_for_async_disk_task(chunk_start, chunk_len, chunk_slot);
        }
        if (profile_enabled_) {
            auto h2d_wait_start = Clock::now();
            cudaEventSynchronize(copy_ready_[chunk_slot]);
            profile_backward_copy_ready_wait_s_ += elapsed_seconds(h2d_wait_start, Clock::now());
            accumulate_backward_copy_time(chunk_slot);
        }
        cudaStreamWaitEvent(compute_stream_, copy_ready_[chunk_slot], 0);

        if (boundary_disk_async_read_) {
            int next_start = chunk_start - transfer_interval_;
            if (compress_) comp_settle();
            if (next_start >= 0 && !(compress_ && !*comp_live_flag(next_start / transfer_interval_))) {
                int next_len = backward_chunk_len(next_start);
                start_async_disk_read(next_start, next_len, backward_slot_for_chunk(next_start));
            }
        }
        profile_compute_start_ = Clock::now();
        profile_have_compute_start_ = true;
    }

    inline GeneralBoundaryPointer backward_restore_ptrs(
        int it,
        const GeneralBoundaryPointer& direct
    )
    {
        if (!staged_)
            return direct;

        int buf_idx = (it - 1) % transfer_interval_;
        int chunk_start = it - 1 - buf_idx;
        int slot_start = backward_slot_for_chunk(chunk_start) * transfer_interval_;
        int gpu_idx = slot_start + buf_idx;
        if (boundary_on_disk_ && !boundary_disk_async_read_)
            gpu_idx = buf_idx;

        GeneralBoundaryPointer ptr{};
        ptr.dtype = boundary_dtype_from_tensor(saver_->top_gpu);
        ptr.last_two = direct.last_two;
        if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
            const int sd = (dim_ == 3) ? 0 : 1;
            BIND_STAGED_SCALED(ptr.dtype, saver_->top_gpu,    gpu_idx * saver_->top_stride,    saver_->top_scale_gpu,    gpu_idx * saver_->top_scale_gpu.stride(sd),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
            BIND_STAGED_SCALED(ptr.dtype, saver_->bottom_gpu, gpu_idx * saver_->bottom_stride, saver_->bottom_scale_gpu, gpu_idx * saver_->bottom_scale_gpu.stride(sd), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
            BIND_STAGED_SCALED(ptr.dtype, saver_->left_gpu,   gpu_idx * saver_->left_stride,   saver_->left_scale_gpu,   gpu_idx * saver_->left_scale_gpu.stride(sd),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
            BIND_STAGED_SCALED(ptr.dtype, saver_->right_gpu,  gpu_idx * saver_->right_stride,  saver_->right_scale_gpu,  gpu_idx * saver_->right_scale_gpu.stride(sd),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            if (dim_ == 3) {
                BIND_STAGED_SCALED(ptr.dtype, saver_->front_gpu, gpu_idx * saver_->front_stride, saver_->front_scale_gpu, gpu_idx * saver_->front_scale_gpu.stride(sd), direct.front, ptr.front_q, ptr.front_h, ptr.front_scale, ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_->back_gpu,  gpu_idx * saver_->back_stride,  saver_->back_scale_gpu,  gpu_idx * saver_->back_scale_gpu.stride(sd),  direct.back,  ptr.back_q,  ptr.back_h,  ptr.back_scale,  ptr.back);
            }
            return ptr;
        }
        BIND_STAGED_FACE(saver_->top_gpu,    gpu_idx * saver_->top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
        BIND_STAGED_FACE(saver_->bottom_gpu, gpu_idx * saver_->bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
        BIND_STAGED_FACE(saver_->left_gpu,   gpu_idx * saver_->left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
        BIND_STAGED_FACE(saver_->right_gpu,  gpu_idx * saver_->right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);

        if (dim_ == 3) {
            BIND_STAGED_FACE(saver_->front_gpu, gpu_idx * saver_->front_stride, ptr.front, ptr.front_h, ptr.front_bf);
            BIND_STAGED_FACE(saver_->back_gpu,  gpu_idx * saver_->back_stride,  ptr.back,  ptr.back_h,  ptr.back_bf);
        }
        return ptr;
    }

    inline GeneralBoundaryPointer backward_restore_ptrs_field(
        int it,
        const GeneralBoundaryPointer& direct,
        int field_idx
    )
    {
        SWEEP_CHECK(field_idx >= 0 && field_idx < saver_->nvar, "Invalid boundary field index.");

        if (dim_ == 2) {
            GeneralBoundaryPointer ptr{};
            ptr.last_two = direct.last_two;
            ptr.use_fp16 = direct.use_fp16;
            ptr.dtype = direct.dtype;

            if (staged_) {
                int buf_idx = (it - 1) % transfer_interval_;
                int chunk_start = it - 1 - buf_idx;
                int slot_start = backward_slot_for_chunk(chunk_start) * transfer_interval_;
                int gpu_idx = slot_start + buf_idx;
                // Only the SYNCHRONOUS DISK path is pinned to slot 0 (see
                // backward_slot_for_chunk); host staging rotates slots and must
                // read the one the prefetch wrote.  76505cc narrowed this guard
                // in backward_restore_ptrs but missed the two nvar>1 readers,
                // which made cpu staging with ring_buffers>=2 read slot 0 while
                // the H2D landed in slot k.  Latent only because
                // cpu_ring_buffers defaults to 1.
                if (boundary_on_disk_ && !boundary_disk_async_read_)
                    gpu_idx = buf_idx;

                ptr.dtype = boundary_dtype_from_tensor(saver_->top_gpu);
                if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                    BIND_STAGED_SCALED(ptr.dtype, saver_->top_gpu,
                        field_idx * saver_->top_gpu.stride(0) + gpu_idx * saver_->top_gpu.stride(1),
                        saver_->top_scale_gpu,
                        field_idx * saver_->top_scale_gpu.stride(0) + gpu_idx * saver_->top_scale_gpu.stride(1),
                        direct.top, ptr.top_q, ptr.top_h, ptr.top_scale, ptr.top);
                    BIND_STAGED_SCALED(ptr.dtype, saver_->bottom_gpu,
                        field_idx * saver_->bottom_gpu.stride(0) + gpu_idx * saver_->bottom_gpu.stride(1),
                        saver_->bottom_scale_gpu,
                        field_idx * saver_->bottom_scale_gpu.stride(0) + gpu_idx * saver_->bottom_scale_gpu.stride(1),
                        direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                    BIND_STAGED_SCALED(ptr.dtype, saver_->left_gpu,
                        field_idx * saver_->left_gpu.stride(0) + gpu_idx * saver_->left_gpu.stride(1),
                        saver_->left_scale_gpu,
                        field_idx * saver_->left_scale_gpu.stride(0) + gpu_idx * saver_->left_scale_gpu.stride(1),
                        direct.left, ptr.left_q, ptr.left_h, ptr.left_scale, ptr.left);
                    BIND_STAGED_SCALED(ptr.dtype, saver_->right_gpu,
                        field_idx * saver_->right_gpu.stride(0) + gpu_idx * saver_->right_gpu.stride(1),
                        saver_->right_scale_gpu,
                        field_idx * saver_->right_scale_gpu.stride(0) + gpu_idx * saver_->right_scale_gpu.stride(1),
                        direct.right, ptr.right_q, ptr.right_h, ptr.right_scale, ptr.right);
                } else {
                BIND_STAGED_FACE(saver_->top_gpu,
                    field_idx * saver_->top_gpu.stride(0) + gpu_idx * saver_->top_gpu.stride(1),
                    ptr.top, ptr.top_h, ptr.top_bf);
                BIND_STAGED_FACE(saver_->bottom_gpu,
                    field_idx * saver_->bottom_gpu.stride(0) + gpu_idx * saver_->bottom_gpu.stride(1),
                    ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
                BIND_STAGED_FACE(saver_->left_gpu,
                    field_idx * saver_->left_gpu.stride(0) + gpu_idx * saver_->left_gpu.stride(1),
                    ptr.left, ptr.left_h, ptr.left_bf);
                BIND_STAGED_FACE(saver_->right_gpu,
                    field_idx * saver_->right_gpu.stride(0) + gpu_idx * saver_->right_gpu.stride(1),
                    ptr.right, ptr.right_h, ptr.right_bf);
                }
            } else if (direct.dtype == BoundaryDtype::FP16) {
                ptr.top_h    = direct.top_h    + field_idx * saver_->top_t.stride(0);
                ptr.bottom_h = direct.bottom_h + field_idx * saver_->bottom_t.stride(0);
                ptr.left_h   = direct.left_h   + field_idx * saver_->left_t.stride(0);
                ptr.right_h  = direct.right_h  + field_idx * saver_->right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_->top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_->bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_->left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_->right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else if (direct.dtype == BoundaryDtype::BF16) {
                ptr.top_bf    = direct.top_bf    + field_idx * saver_->top_t.stride(0);
                ptr.bottom_bf = direct.bottom_bf + field_idx * saver_->bottom_t.stride(0);
                ptr.left_bf   = direct.left_bf   + field_idx * saver_->left_t.stride(0);
                ptr.right_bf  = direct.right_bf  + field_idx * saver_->right_t.stride(0);
            } else if (direct.dtype == BoundaryDtype::INT8) {
                ptr.top_q       = direct.top_q       + field_idx * saver_->top_t.stride(0);
                ptr.bottom_q    = direct.bottom_q    + field_idx * saver_->bottom_t.stride(0);
                ptr.left_q      = direct.left_q      + field_idx * saver_->left_t.stride(0);
                ptr.right_q     = direct.right_q     + field_idx * saver_->right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_->top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_->bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_->left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_->right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else {
                ptr.top = direct.top + field_idx * saver_->top_t.stride(0);
                ptr.bottom = direct.bottom + field_idx * saver_->bottom_t.stride(0);
                ptr.left = direct.left + field_idx * saver_->left_t.stride(0);
                ptr.right = direct.right + field_idx * saver_->right_t.stride(0);
            }
            return ptr;
        }

        int time_idx = it - 1;
        if (staged_) {
            int buf_idx = (it - 1) % transfer_interval_;
            int chunk_start = it - 1 - buf_idx;
            int slot_start = backward_slot_for_chunk(chunk_start) * transfer_interval_;
            time_idx = slot_start + buf_idx;
            // See backward_restore_ptrs_field (2-D): only synchronous disk is
            // slot-0 pinned.
            if (boundary_on_disk_ && !boundary_disk_async_read_)
                time_idx = buf_idx;
        }
        int64_t flat_idx = static_cast<int64_t>(time_idx) * saver_->nvar + field_idx;

        GeneralBoundaryPointer ptr{};
        ptr.use_fp16 = direct.use_fp16;
            ptr.dtype = direct.dtype;
        if (!staged_ && direct.dtype == BoundaryDtype::FP16) {
            ptr.top_h    = direct.top_h    + flat_idx * saver_->top_stride;
            ptr.bottom_h = direct.bottom_h + flat_idx * saver_->bottom_stride;
            ptr.front_h  = direct.front_h  + flat_idx * saver_->front_stride;
            ptr.back_h   = direct.back_h   + flat_idx * saver_->back_stride;
            ptr.left_h   = direct.left_h   + flat_idx * saver_->left_stride;
            ptr.right_h  = direct.right_h  + flat_idx * saver_->right_stride;
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_->top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_->bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_->front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_->back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_->left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_->right_scale_t.stride(0);
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (!staged_ && direct.dtype == BoundaryDtype::BF16) {
            ptr.top_bf    = direct.top_bf    + flat_idx * saver_->top_stride;
            ptr.bottom_bf = direct.bottom_bf + flat_idx * saver_->bottom_stride;
            ptr.front_bf  = direct.front_bf  + flat_idx * saver_->front_stride;
            ptr.back_bf   = direct.back_bf   + flat_idx * saver_->back_stride;
            ptr.left_bf   = direct.left_bf   + flat_idx * saver_->left_stride;
            ptr.right_bf  = direct.right_bf  + flat_idx * saver_->right_stride;
        } else if (!staged_ && direct.dtype == BoundaryDtype::INT8) {
            // 3D persistent uint8 buffer: outermost index = time*nvar + field.
            // top_stride/bottom_stride/... already track per-(time×field)
            // cell counts (compute_time_strides → tensor.stride(0)).
            ptr.top_q    = direct.top_q    + flat_idx * saver_->top_stride;
            ptr.bottom_q = direct.bottom_q + flat_idx * saver_->bottom_stride;
            ptr.front_q  = direct.front_q  + flat_idx * saver_->front_stride;
            ptr.back_q   = direct.back_q   + flat_idx * saver_->back_stride;
            ptr.left_q   = direct.left_q   + flat_idx * saver_->left_stride;
            ptr.right_q  = direct.right_q  + flat_idx * saver_->right_stride;
            // Scale buffer outermost stride.
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_->top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_->bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_->front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_->back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_->left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_->right_scale_t.stride(0);
            // Staging shared across (time, field) — no offset.
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (staged_) {
            ptr.dtype = boundary_dtype_from_tensor(saver_->top_gpu);
            if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                BIND_STAGED_SCALED(ptr.dtype, saver_->top_gpu,    flat_idx * saver_->top_stride,    saver_->top_scale_gpu,    flat_idx * saver_->top_scale_gpu.stride(0),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
                BIND_STAGED_SCALED(ptr.dtype, saver_->bottom_gpu, flat_idx * saver_->bottom_stride, saver_->bottom_scale_gpu, flat_idx * saver_->bottom_scale_gpu.stride(0), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                BIND_STAGED_SCALED(ptr.dtype, saver_->front_gpu,  flat_idx * saver_->front_stride,  saver_->front_scale_gpu,  flat_idx * saver_->front_scale_gpu.stride(0),  direct.front,  ptr.front_q,  ptr.front_h,  ptr.front_scale,  ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_->back_gpu,   flat_idx * saver_->back_stride,   saver_->back_scale_gpu,   flat_idx * saver_->back_scale_gpu.stride(0),   direct.back,   ptr.back_q,   ptr.back_h,   ptr.back_scale,   ptr.back);
                BIND_STAGED_SCALED(ptr.dtype, saver_->left_gpu,   flat_idx * saver_->left_stride,   saver_->left_scale_gpu,   flat_idx * saver_->left_scale_gpu.stride(0),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
                BIND_STAGED_SCALED(ptr.dtype, saver_->right_gpu,  flat_idx * saver_->right_stride,  saver_->right_scale_gpu,  flat_idx * saver_->right_scale_gpu.stride(0),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            } else {
            BIND_STAGED_FACE(saver_->top_gpu,    flat_idx * saver_->top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
            BIND_STAGED_FACE(saver_->bottom_gpu, flat_idx * saver_->bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
            BIND_STAGED_FACE(saver_->front_gpu,  flat_idx * saver_->front_stride,  ptr.front,  ptr.front_h,  ptr.front_bf);
            BIND_STAGED_FACE(saver_->back_gpu,   flat_idx * saver_->back_stride,   ptr.back,   ptr.back_h,   ptr.back_bf);
            BIND_STAGED_FACE(saver_->left_gpu,   flat_idx * saver_->left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
            BIND_STAGED_FACE(saver_->right_gpu,  flat_idx * saver_->right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);
            }
        } else {
            ptr.top = direct.top + flat_idx * saver_->top_stride;
            ptr.bottom = direct.bottom + flat_idx * saver_->bottom_stride;
            ptr.front = direct.front + flat_idx * saver_->front_stride;
            ptr.back = direct.back + flat_idx * saver_->back_stride;
            ptr.left = direct.left + flat_idx * saver_->left_stride;
            ptr.right = direct.right + flat_idx * saver_->right_stride;
        }
        ptr.last_two = direct.last_two;
        return ptr;
    }

    inline int backward_time_index(int it) const
    {
        return staged_ ? 0 : it - 1;
    }

    inline int backward_time_index_field(int it) const
    {
        return dim_ == 3 ? 0 : backward_time_index(it);
    }

    inline void restore_backward_2d(
        int it,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx
    )
    {
        wait_before_backward_restore(it);
        auto b = backward_restore_ptrs(it, direct);
        if (b.dtype == BoundaryDtype::BF16) {
            SWEEP_LAUNCH(boundary_kernel2d_bf16, grid, block,
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            dequantize_step_2d(b, backward_time_index(it));
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
        } else if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_2d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            SWEEP_LAUNCH(boundary_kernel2d_compact, compact_blocks, compact_threads,
                u, b.top, b.bottom, b.left, b.right,
                backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
        } else {
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
        }
        record_backward_restore_done(it);
    }

    inline void restore_backward_3d(
        int it,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx
    )
    {
        wait_before_backward_restore(it);
        auto b = backward_restore_ptrs(it, direct);
        if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_compact_bf16, compact_blocks, compact_threads,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                // per-face lossy copies of an edge/corner cell differ: restore
                // each from its owning face (boundary/kernels.cuh)
                dequantize_step_3d(b, backward_time_index(it));
                SWEEP_LAUNCH(boundary_kernel3d_compact_owned, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                SWEEP_LAUNCH(boundary_kernel3d_compact, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_bf16, grid, block,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                dequantize_step_3d(b, backward_time_index(it));
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
            }
        }
        record_backward_restore_done(it);
    }

    inline void restore_backward_2d_field(
        int it,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx,
        int field_idx,
        bool wait_chunk,
        bool record_done,
        int tangent_pad = 0    // as save_forward_2d_field
    )
    {
        if (wait_chunk)
            wait_before_backward_restore(it);
        auto b = backward_restore_ptrs_field(it, direct, field_idx);
        if (b.dtype == BoundaryDtype::BF16) {
            SWEEP_LAUNCH(boundary_kernel2d_bf16, grid, block,
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE, tangent_pad);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            dequantize_step_2d(b, backward_time_index_field(it));
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE, tangent_pad);
        } else if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_2d(ctx, width, tangent_pad);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            SWEEP_LAUNCH(boundary_kernel2d_compact, compact_blocks, compact_threads,
                u, b.top, b.bottom, b.left, b.right,
                backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE, tangent_pad);
        } else {
            SWEEP_LAUNCH(boundary_kernel2d, grid, block,
                u, b.top, b.bottom, b.left, b.right,
                backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE, tangent_pad);
        }
        if (record_done)
            record_backward_restore_done(it);
    }

    inline void restore_backward_3d_field(
        int it,
        float* u,
        dim3 grid,
        dim3 block,
        const GeneralBoundaryPointer& direct,
        int width,
        int offset,
        SolverContext ctx,
        int field_idx,
        bool wait_chunk,
        bool record_done
    )
    {
        if (wait_chunk)
            wait_before_backward_restore(it);
        auto b = backward_restore_ptrs_field(it, direct, field_idx);
        if (use_compact_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_compact_bf16, compact_blocks, compact_threads,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                // per-face lossy copies of an edge/corner cell differ: restore
                // each from its owning face (boundary/kernels.cuh)
                dequantize_step_3d(b, backward_time_index_field(it));
                SWEEP_LAUNCH(boundary_kernel3d_compact_owned, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                SWEEP_LAUNCH(boundary_kernel3d_compact, compact_blocks, compact_threads,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                SWEEP_LAUNCH(boundary_kernel3d_bf16, grid, block,
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                dequantize_step_3d(b, backward_time_index_field(it));
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                SWEEP_LAUNCH(boundary_kernel3d, grid, block,
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
            }
        }
        if (record_done)
            record_backward_restore_done(it);
    }

    inline void prefetch_next_backward_chunk_if_needed(int it, int nt)
    {
        if (!enabled_ || !staged_ || it <= 1)
            return;
        if (boundary_disk_async_read_)
            return;

        int buf_idx = (it - 1) % transfer_interval_;
        int cur_start = it - 1 - buf_idx;

        int chunk_id = (it - 1) / transfer_interval_;
        int next_chunk = chunk_id - 1;
        if (next_chunk < 0)
            return;

        int next_start = next_chunk * transfer_interval_;

        // Issue the next chunk on ENTRY to the current one -- the same point as
        // wait_before_backward_restore -- so the copy overlaps this chunk's
        // transfer_interval steps of compute.  That is only safe when the next
        // chunk lands in a DIFFERENT ring slot; when it reuses this slot the
        // early H2D would overwrite boundary data the current chunk's restore
        // is still reading, so fall back to "compute, then issue" (no overlap,
        // but correct).
        //
        // The predicate must be the slot assignment itself, not ring_buffers_.
        // backward_slot_for_chunk() pins the SYNCHRONOUS disk path to slot 0
        // whatever ring_buffers_ says, and that path defaults to
        // disk_ring_buffers_2d = 3 / _3d = 2 (options.py) -- so keying off
        // ring_buffers_ >= 2 turned early issue on for a single-slot path whose
        // prefetch_backward_chunk() has no compute_ready_ write-after-read guard
        // (only the host-staging branch got one).  Every bs_disk gate case and
        // the three disk cases in test_boundary_storage_dtype_validation.py
        // reconstructed a wrong gradient as a result.
        const bool early =
            backward_slot_for_chunk(cur_start) != backward_slot_for_chunk(next_start);
        if (early) {
            if (buf_idx != backward_chunk_len(cur_start) - 1)
                return;
        } else {
            if (buf_idx != 0)
                return;
        }
        int remain = nt - next_start;
        int next_len = remain < transfer_interval_ ? remain : transfer_interval_;
        // A compressed chunk that is not in the arena yet belongs to a segment
        // the driver refills later (ModelParallel segment_steps): leave it, the
        // entry into that chunk fetches it (wait_before_backward_restore).
        if (compress_) {
            comp_settle();
            if (!*comp_live_flag(next_start / transfer_interval_))
                return;
        }
        prefetch_backward_chunk(next_start, next_len,
                                backward_slot_for_chunk(next_start));
    }

    inline void synchronize()
    {
        wait_for_pending_disk_reader_task();
        if (compress_)
            comp_settle();
        if (enabled_ && staged_)
            cudaStreamSynchronize(copy_stream_);
        if (boundary_on_disk_)
            wait_for_boundary_disk_writes();
        if (compress_)
            finish_compress_phase();
        accumulate_all_forward_copy_time();
        print_profile_if_needed();
    }

private:
    using Clock = std::chrono::steady_clock;

    // Pointers, not references: the persistent runtime outlives a single call
    // while saver/disk_files are per-call locals, so they must be rebindable.
    // See rebind().
    EffectiveBoundarySaver* saver_ = nullptr;
    int dim_ = 2;
    bool enabled_ = false;
    bool staged_ = false;
    bool compress_ = false;
    int compress_tier_ = 0;   // 0 gpu arena, 1 pinned host arena, 2 disk file
    bool boundary_on_disk_ = false;
    bool boundary_disk_async_read_ = false;
    int transfer_interval_ = 1;
    int ring_buffers_ = 1;
    int backward_nt_ = 0;
    const std::vector<std::string>* disk_files_ = nullptr;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    bool profile_enabled_ = false;
    bool no_compact_kernel_ = false;
    bool backward_active_ = false;
    bool profile_printed_ = false;
    bool profile_have_compute_start_ = true;
    Clock::time_point profile_compute_start_;
    double profile_disk_read_s_ = 0.0;
    double profile_backward_task_wait_s_ = 0.0;
    double profile_backward_h2d_enqueue_s_ = 0.0;
    double profile_backward_copy_stream_wait_s_ = 0.0;
    double profile_backward_h2d_s_ = 0.0;
    double profile_backward_copy_ready_wait_s_ = 0.0;
    double profile_compute_between_chunks_s_ = 0.0;
    double profile_disk_write_time_s_ = 0.0;
    double profile_forward_slot_reuse_wait_s_ = 0.0;
    int profile_chunks_ = 0;
    int profile_forward_chunks_ = 0;
    std::mutex profile_mutex_;
    std::vector<cudaEvent_t> compute_ready_;
    std::vector<cudaEvent_t> copy_ready_;
    std::vector<cudaEvent_t> forward_copy_start_;
    std::vector<cudaEvent_t> backward_copy_start_;
    std::vector<cudaEvent_t> backward_h2d_start_;
    std::vector<char> copy_pending_;
    std::vector<char> forward_copy_profile_pending_;
    std::vector<char> backward_copy_profile_pending_;

    // ---- compression state (BoundarySaving(compress='bitcomp')) ----
    // Geometry: every chunk of a slot is a fixed set of ≤ kPieceBytes pieces
    // (one region per face, per field in 2-D, split in order); table t in
    // [0, R) describes slot t at a full transfer_interval, table R the one
    // shorter tail chunk (built on first use).  The per-chunk index carries
    // the packed bytes; the arena offset is a running sum the host keeps.
    int comp_elem_size_ = 1;
    int comp_max_pieces_ = 0;
    int64_t comp_cap_ = 0;          // bytes per bound / pack slot
    int comp_pack_slots_ = 0;       // pack slots (>= ring_buffers): how long a landing may be deferred
    int64_t comp_maxout_ = 0;       // bound bytes per piece (8-aligned)
    int64_t comp_nchunks_ = 0;      // index rows: one per chunk of the WHOLE record
    int comp_last_chunk_ = -1;      // last chunk compressed in this forward phase (-1: none)
    int comp_last_len_ = 0;         // ... and the steps it held then
    std::vector<int> comp_slot_chunk_;   // per ring slot: the chunk last fetched into it (-1 none)
    int64_t comp_scale_full_bytes_ = 0;   // the scale ring's bytes for one full chunk (0 for fp32/bf16)
    int64_t comp_temp_bytes_ = 0;
    int64_t comp_persist_bytes_ = 0;
    std::vector<int> comp_npieces_;          // per table
    std::vector<int64_t> comp_raw_per_table_; // uncompressed bytes per table
    std::vector<int64_t> comp_tab_host_;     // (R+1) x kTabRows x max_pieces, rows 0..2 meaningful
    std::vector<char> comp_status_used_;     // per table: a batch ran since the last check
    int comp_tail_slot_ = -1, comp_tail_len_ = -1;
    void* comp_ring_ptr_ = nullptr;
    void* comp_tab_ptr_ = nullptr;
    std::vector<cudaEvent_t> comp_pack_ready_;
    std::deque<int> pending_land_;
    // The arena is a ring of packed chunks: live ones in order, allocated
    // circularly; a chunk whose steps the step ring has overwritten is dead.
    struct LiveChunk { int k; int64_t off; int64_t bytes; int64_t raw; int64_t packed; };  // bytes = packed + scales + alignment
    std::deque<LiveChunk> comp_live_;
    int64_t comp_ring_steps_ = 0;    // the step ring's length (Python: nt_saved); 0 = whole record
    int64_t comp_arena_bytes_ = 0;   // capacity of the circular arena (device / pinned arena, or the disk file)
    int comp_window_ = 0;            // live chunks kept: ceil(ring_steps / TI) + 1 (0 = all)
    int64_t comp_raw_step_bytes_ = 0;   // uncompressed bytes of one saved step (all faces)
    int64_t comp_basis_bytes_ = 0;   // cfg[5]: the bytes compress_budget multiplies (Python _allocate_compressed_store)
    int64_t comp_extra_bytes_ = 0;   // cfg[6]: what the arena holds beyond budget x basis (scales, alignment)
    int comp_wrap_slack_ = 0;        // cfg[7] & 1: the arena is refilled and may waste one chunk at the wrap
    BoundaryChunk comp_open_;        // the last chunk saved into but not yet compressed
    bool comp_open_valid_ = false;
    int64_t arena_fill_ = 0;
    int64_t comp_packed_bytes_ = 0;
    int64_t comp_raw_bytes_ = 0;
    int64_t comp_chunks_landed_ = 0;
    double profile_compress_land_wait_s_ = 0.0;
    int profile_compress_forced_lands_ = 0;
    std::thread disk_reader_thread_;
    std::mutex disk_reader_mutex_;
    std::condition_variable disk_reader_cv_;
    bool disk_reader_stop_ = false;
    bool disk_task_queued_ = false;
    bool disk_task_running_ = false;
    bool disk_task_done_ = false;
    int disk_task_start_ = 0;
    int disk_task_len_ = 0;
    int disk_task_slot_ = 0;
    std::exception_ptr disk_reader_exception_ = nullptr;

    inline int compact_boundary_count_3d(const SolverContext& ctx, int width, int tangent_pad) const
    {
        int nx_boundary = ctx.nx_phys() + 2 * tangent_pad;
        int ny_boundary = ctx.ny_phys() + 2 * tangent_pad;
        int nz_boundary = ctx.nz_phys() + 2 * tangent_pad;
        int top_count = width * ny_boundary * nx_boundary;
        int front_count = nz_boundary * width * nx_boundary;
        int left_count = nz_boundary * ny_boundary * width;
        return ctx.B * (2 * top_count + 2 * front_count + 2 * left_count);
    }

    inline int compact_boundary_count_2d(const SolverContext& ctx, int width, int tangent_pad) const
    {
        int nx_boundary = ctx.nx_phys() + 2 * tangent_pad;
        int nz_boundary = ctx.nz_phys() + 2 * tangent_pad;
        return ctx.B * (2 * width * nx_boundary + 2 * nz_boundary * width);
    }

    // Gate for both the 2-D and 3-D compact kernels: the synchronous disk path
    // reuses the full-grid launch geometry, everything else takes band-only
    // threads.
    inline bool use_compact_boundary_kernel() const
    {
        if (no_compact_kernel_)
            return false;
        return !boundary_on_disk_ || boundary_disk_async_read_;
    }

    inline int backward_slot_for_chunk(int chunk_start) const
    {
        // The SYNCHRONOUS disk path has to stay single-slot: its
        // load_disk_to_cpu_* and load_cpu_to_gpu(0, ...) both hard-code the
        // staging offset to 0 and reuse one CPU scratch block. Host staging has
        // no such constraint, and allowing several slots there is exactly what
        // lets the backward H2D overlap compute.
        if (boundary_on_disk_ && !boundary_disk_async_read_)
            return 0;
        int chunk_id = chunk_start / transfer_interval_;
        return chunk_id % ring_buffers_;
    }

    inline int backward_chunk_len(int chunk_start) const
    {
        int remain = backward_nt_ - 1 - chunk_start;
        return remain < transfer_interval_ ? remain : transfer_interval_;
    }

    inline void prefetch_backward_chunk(int start, int len, int slot)
    {
        if (boundary_on_disk_) {
            if (boundary_disk_async_read_) {
                start_async_disk_read(start, len, slot);
            } else {
                // The synchronous disk read below overwrites the shared CPU
                // staging buffer (boundary_cpu / saver_->*_t).  The previous
                // chunk's H2D copy reads that same buffer, so we must wait for
                // it to finish before clobbering it — otherwise the host races
                // ahead of the (possibly backlogged) copy stream and the
                // in-flight H2D picks up this chunk's freshly-read bytes,
                // scrambling the restored boundary for nvar>1.  This path is
                // pinned to a single slot by backward_slot_for_chunk(), so the
                // staging buffer is reused verbatim -- note that ring_buffers_
                // itself is NOT 1 here (disk defaults are 3 in 2-D, 2 in 3-D).
                cudaStreamSynchronize(copy_stream_);
                auto read_start = Clock::now();
                if (dim_ == 2)
                    saver_->load_disk_to_cpu_2d(start, len, (*disk_files_));
                else
                    saver_->load_disk_to_cpu_3d(start, len, (*disk_files_));
                add_disk_read_time(elapsed_seconds(read_start, Clock::now()));
                if (profile_enabled_ && boundary_on_disk_) {
                    cudaEventRecord(backward_copy_start_[slot], copy_stream_);
                    cudaEventRecord(backward_h2d_start_[slot], copy_stream_);
                }
                // Write-after-read, the same fence the host-staging branch
                // below carries.  copy_stream_ is NON-BLOCKING since 76505cc, so
                // it is no longer implicitly ordered behind the default stream
                // PyTorch computes on: without this, the H2D can start
                // overwriting slot 0 while the current chunk's last restore
                // kernels are still reading it.  nvar>1 3-D lost that race
                // deterministically (elastic3d / das_mu3d bs_disk); nvar=1 won it
                // by having a smaller H2D, which is exactly how the earlier
                // CPU-staging race in this function presented too.
                cudaStreamWaitEvent(copy_stream_, compute_ready_[slot], 0);
                auto enqueue_start = Clock::now();
                saver_->load_cpu_to_gpu(0, len, copy_stream_);
                add_backward_h2d_enqueue_time(elapsed_seconds(enqueue_start, Clock::now()));
                cudaEventRecord(copy_ready_[slot], copy_stream_);
                if (profile_enabled_ && boundary_on_disk_)
                    backward_copy_profile_pending_[slot] = 1;
            }
        } else {
            // write-after-read: this slot's previous contents may still be
            // under read by compute. With ring>=2 the next chunk lands in a
            // different slot, so this event has long completed -- no stall, real
            // overlap; with ring==1 it correctly orders the H2D after the read
            // (no overlap, but no corruption).
            if (compress_) {
                fetch_compressed_chunk(start, slot);
                if (profile_enabled_)
                    ++profile_chunks_;
                return;
            }
            cudaStreamWaitEvent(copy_stream_, compute_ready_[slot], 0);
            // ...into that slot's own GPU ring interval; the old default
            // gpu_start=0 packed every chunk into slot 0, which left
            // ring_buffers doing nothing at all on the backward path.
            saver_->load_cpu_to_gpu(start, len, copy_stream_, slot * transfer_interval_);
            cudaEventRecord(copy_ready_[slot], copy_stream_);
            if (profile_enabled_ && staged_)
                ++profile_chunks_;
        }
    }

    inline void record_backward_restore_done(int it)
    {
        // With a non-blocking copy_stream the write-after-read must be stated
        // explicitly: compute has to finish reading this slot before the next
        // H2D may write into it. Host staging used to just return here and lean
        // on the implicit sync between a blocking cudaStreamCreate stream and
        // the default stream -- that cover is gone once the stream is
        // NonBlocking.
        if (!enabled_ || !staged_)
            return;
        int buf_idx = (it - 1) % transfer_interval_;
        int chunk_start = it - 1 - buf_idx;
        int slot = backward_slot_for_chunk(chunk_start);
        cudaEventRecord(compute_ready_[slot], compute_stream_);
    }

    inline void start_async_disk_read(int start, int len, int slot)
    {
        if (compress_)
            comp_slot_chunk_[slot] = start / transfer_interval_;   // queued for this slot
        std::unique_lock<std::mutex> lock(disk_reader_mutex_);
        disk_reader_cv_.wait(lock, [this]() {
            return !disk_task_queued_ && !disk_task_running_ && !disk_task_done_;
        });
        rethrow_disk_reader_exception_locked();
        disk_task_start_ = start;
        disk_task_len_ = len;
        disk_task_slot_ = slot;
        disk_task_queued_ = true;
        disk_reader_cv_.notify_all();
    }

    inline void wait_for_async_disk_task(int start, int len, int slot)
    {
        std::unique_lock<std::mutex> lock(disk_reader_mutex_);
        auto wait_start = Clock::now();
        disk_reader_cv_.wait(lock, [this, start, len, slot]() {
            return disk_reader_exception_ ||
                   (disk_task_done_ &&
                    disk_task_start_ == start &&
                    disk_task_len_ == len &&
                    disk_task_slot_ == slot);
        });
        add_backward_task_wait_time(elapsed_seconds(wait_start, Clock::now()));
        rethrow_disk_reader_exception_locked();
        disk_task_done_ = false;
        disk_reader_cv_.notify_all();
    }

    inline void wait_for_pending_disk_reader_task()
    {
        if (!boundary_disk_async_read_)
            return;

        std::unique_lock<std::mutex> lock(disk_reader_mutex_);
        disk_reader_cv_.wait(lock, [this]() {
            return disk_reader_exception_ || (!disk_task_queued_ && !disk_task_running_);
        });
        rethrow_disk_reader_exception_locked();
        if (disk_task_done_) {
            disk_task_done_ = false;
            disk_reader_cv_.notify_all();
        }
    }

    inline void stop_disk_reader_no_throw()
    {
        if (!disk_reader_thread_.joinable())
            return;
        {
            std::lock_guard<std::mutex> lock(disk_reader_mutex_);
            disk_reader_stop_ = true;
        }
        disk_reader_cv_.notify_all();
        disk_reader_thread_.join();
    }

    inline void rethrow_disk_reader_exception_locked()
    {
        if (!disk_reader_exception_)
            return;
        auto ex = disk_reader_exception_;
        disk_reader_exception_ = nullptr;
        disk_task_queued_ = false;
        disk_task_running_ = false;
        disk_task_done_ = false;
        disk_reader_cv_.notify_all();
        std::rethrow_exception(ex);
    }

    inline void run_disk_reader_task(int start, int len, int slot)
    {
        int stage_start = slot * transfer_interval_;
        // This slot's staging region was last read by its previous H2D copy
        // (tracked by copy_ready_[slot]).  Wait for that copy to finish before
        // the disk read below overwrites the region, or the in-flight H2D
        // would pick up the new chunk's bytes and scramble the restored
        // boundary for nvar>1.  (compute_ready_ already gates the H2D's write
        // to top_gpu; this gates the disk read's write to the CPU staging.)
        cudaEventSynchronize(copy_ready_[slot]);
        if (compress_) {
            // The packed chunk goes to this slot's pinned stage, the scales to
            // their cpu staging as on the plain disk path; the copy-stream work
            // (H2D, decompress, copy_ready) is then issued from this thread.
            const int k = start / transfer_interval_;
            const int64_t* row = comp_index_host_row(k);
            const int fwd_len = static_cast<int>(row[sweep_nvcomp::kIdxLen]);
            auto read_start = Clock::now();
            const size_t nbytes = static_cast<size_t>(row[sweep_nvcomp::kIdxTotal] + comp_scale_bytes(fwd_len));
            if (nbytes > 0)
                read_boundary_file_chunk((*disk_files_)[0], static_cast<size_t>(row[sweep_nvcomp::kIdxOffset]),
                                         comp_stage_ptr(slot), nbytes, 1);
            add_disk_read_time(elapsed_seconds(read_start, Clock::now()));
            fetch_compressed_chunk(start, slot);
            if (profile_enabled_) {
                ++profile_chunks_;
            }
            return;
        }
        auto read_start = Clock::now();
        if (dim_ == 2)
            saver_->load_disk_to_cpu_2d(start, len, (*disk_files_), stage_start);
        else
            saver_->load_disk_to_cpu_3d(start, len, (*disk_files_), stage_start);
        add_disk_read_time(elapsed_seconds(read_start, Clock::now()));

        if (profile_enabled_ && boundary_on_disk_)
            cudaEventRecord(backward_copy_start_[slot], copy_stream_);
        cudaStreamWaitEvent(copy_stream_, compute_ready_[slot], 0);
        if (profile_enabled_ && boundary_on_disk_)
            cudaEventRecord(backward_h2d_start_[slot], copy_stream_);
        auto enqueue_start = Clock::now();
        saver_->load_cpu_to_gpu(stage_start, len, copy_stream_, stage_start);
        add_backward_h2d_enqueue_time(elapsed_seconds(enqueue_start, Clock::now()));
        cudaEventRecord(copy_ready_[slot], copy_stream_);
        if (profile_enabled_ && boundary_on_disk_) {
            ++profile_chunks_;
            backward_copy_profile_pending_[slot] = 1;
        }
    }

    inline void disk_reader_loop()
    {
        while (true) {
            int start = 0;
            int len = 0;
            int slot = 0;
            {
                std::unique_lock<std::mutex> lock(disk_reader_mutex_);
                disk_reader_cv_.wait(lock, [this]() {
                    return disk_reader_stop_ || disk_task_queued_;
                });
                if (disk_reader_stop_)
                    return;
                start = disk_task_start_;
                len = disk_task_len_;
                slot = disk_task_slot_;
                disk_task_queued_ = false;
                disk_task_running_ = true;
            }

            try {
                run_disk_reader_task(start, len, slot);
            } catch (...) {
                std::lock_guard<std::mutex> lock(disk_reader_mutex_);
                disk_reader_exception_ = std::current_exception();
                disk_task_running_ = false;
                disk_task_done_ = false;
                disk_reader_cv_.notify_all();
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(disk_reader_mutex_);
                disk_task_running_ = false;
                disk_task_done_ = true;
            }
            disk_reader_cv_.notify_all();
        }
    }

    inline double elapsed_seconds(Clock::time_point start, Clock::time_point end) const
    {
        return std::chrono::duration<double>(end - start).count();
    }

    inline void add_disk_read_time(double seconds)
    {
        if (!profile_enabled_)
            return;
        std::lock_guard<std::mutex> lock(profile_mutex_);
        profile_disk_read_s_ += seconds;
        if (!boundary_disk_async_read_)
            ++profile_chunks_;
    }

    inline void add_backward_task_wait_time(double seconds)
    {
        if (!profile_enabled_)
            return;
        std::lock_guard<std::mutex> lock(profile_mutex_);
        profile_backward_task_wait_s_ += seconds;
    }

    inline void add_backward_h2d_enqueue_time(double seconds)
    {
        if (!profile_enabled_)
            return;
        std::lock_guard<std::mutex> lock(profile_mutex_);
        profile_backward_h2d_enqueue_s_ += seconds;
    }

    inline void print_profile_if_needed()
    {
        if (!profile_enabled_ || profile_printed_ || !staged_)
            return;
        profile_printed_ = true;
        accumulate_all_forward_copy_time();
        std::lock_guard<std::mutex> lock(profile_mutex_);
        std::fprintf(
            stderr,
            "SWEEP_BOUNDARY_PROFILE "
            "backward_disk_read_time=%.6f "
            "backward_task_wait_time=%.6f "
            "backward_h2d_enqueue_time=%.6f "
            "backward_copy_stream_wait_time=%.6f "
            "backward_h2d_time=%.6f "
            "backward_copy_ready_wait_time=%.6f "
            "backward_chunk_cpu_time=%.6f "
            "forward_copy_write_time=%.6f "
            "forward_slot_reuse_wait_time=%.6f "
            "compress_land_wait_time=%.6f "
            "compress_forced_lands=%d "
            "chunks=%d "
            "forward_chunks=%d "
            "async=%d\n",
            profile_disk_read_s_,
            profile_backward_task_wait_s_,
            profile_backward_h2d_enqueue_s_,
            profile_backward_copy_stream_wait_s_,
            profile_backward_h2d_s_,
            profile_backward_copy_ready_wait_s_,
            profile_compute_between_chunks_s_,
            profile_disk_write_time_s_,
            profile_forward_slot_reuse_wait_s_,
            profile_compress_land_wait_s_,
            profile_compress_forced_lands_,
            profile_chunks_,
            profile_forward_chunks_,
            boundary_disk_async_read_ ? 1 : 0
        );
        std::fflush(stderr);
    }

    inline void accumulate_forward_copy_time(int slot)
    {
        if (!profile_enabled_ || !boundary_on_disk_ || !forward_copy_profile_pending_[slot])
            return;
        cudaEventSynchronize(copy_ready_[slot]);
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, forward_copy_start_[slot], copy_ready_[slot]);
        profile_disk_write_time_s_ += 0.001 * static_cast<double>(milliseconds);
        forward_copy_profile_pending_[slot] = 0;
    }

    inline void accumulate_backward_copy_time(int slot)
    {
        if (!profile_enabled_ || !boundary_on_disk_ || !backward_copy_profile_pending_[slot])
            return;
        float wait_ms = 0.0f;
        float h2d_ms = 0.0f;
        cudaEventElapsedTime(&wait_ms, backward_copy_start_[slot], backward_h2d_start_[slot]);
        cudaEventElapsedTime(&h2d_ms, backward_h2d_start_[slot], copy_ready_[slot]);
        profile_backward_copy_stream_wait_s_ += 0.001 * static_cast<double>(wait_ms);
        profile_backward_h2d_s_ += 0.001 * static_cast<double>(h2d_ms);
        backward_copy_profile_pending_[slot] = 0;
    }

    inline void accumulate_all_forward_copy_time()
    {
        if (!profile_enabled_ || !boundary_on_disk_)
            return;
        for (int i = 0; i < ring_buffers_; ++i)
            accumulate_forward_copy_time(i);
    }

    // =====================================================================
    // Lossless ring compression (nvCOMP Bitcomp).  Chunk granularity, on the
    // copy stream; the per-step save/restore kernels never see it.
    //
    // Forward, at every chunk end (compress_chunk):
    //   compress the slot's pieces -> bound[slot]; scan sizes -> packed
    //   offsets + index row; pack -> pack[slot]; D2H the index row; record
    //   copy_ready (ring slot free) and pack_ready.  The LANDING of a chunk
    //   (pack[slot] -> arena / pinned arena / disk, at its exact packed size)
    //   needs the size on the host, so it is issued at a LATER chunk end once
    //   pack_ready has passed (cudaEventQuery, no stall), and forced only when
    //   its pack slot is about to be reused or the phase ends.
    // Backward, at every chunk fetch (fetch_compressed_chunk):
    //   persist[offset] -> pack[slot]; scan the index row -> compressed piece
    //   pointers; decompress -> the ring slot; copy_ready.  dequantize / restore
    //   read the ring exactly as without compression, so the result is
    //   bit-identical to the uncompressed path (lossless).
    // =====================================================================
    inline bool comp_scaled() const
    {
        const auto dt = saver_->top_gpu.dtype();
        return dt == BoundaryDtype::INT8 || dt == BoundaryDtype::FP16;
    }
    inline int64_t* comp_tab_host(int t, int row) { return comp_tab_host_.data() + (static_cast<int64_t>(t) * sweep_nvcomp::kTabRows + row) * comp_max_pieces_; }
    inline int64_t* comp_tab_dev(int t, int row) const
    {
        return reinterpret_cast<int64_t*>(saver_->comp_tab.data_ptr())
            + (static_cast<int64_t>(t) * sweep_nvcomp::kTabRows + row) * comp_max_pieces_;
    }
    inline int32_t* comp_status_dev(int t) const { return reinterpret_cast<int32_t*>(saver_->comp_status.data_ptr()) + static_cast<int64_t>(t) * comp_max_pieces_; }
    inline int32_t* comp_status_host(int t) const { return reinterpret_cast<int32_t*>(saver_->comp_status_host.data_ptr()) + static_cast<int64_t>(t) * comp_max_pieces_; }
    // cfg[16 + k]: 1 while chunk k's packed bytes are in the arena (host memory
    // the propagator owns, written by this side only -- so no D2H races it).
    // cfg: 8 int64 words [ring steps, arena/file bytes, TI, R, nt, basis, extra, wrap slack], then one live flag per chunk
    static constexpr int kCfgHeader = 64;
    inline const int64_t* comp_cfg_words() const { return reinterpret_cast<const int64_t*>(saver_->comp_cfg.data_ptr()); }
    inline uint8_t* comp_live_flag(int k) const { return static_cast<uint8_t*>(saver_->comp_cfg.data_ptr()) + kCfgHeader + k; }
    inline void comp_clear_live_flags() { std::memset(static_cast<uint8_t*>(saver_->comp_cfg.data_ptr()) + kCfgHeader, 0, static_cast<size_t>(comp_nchunks_)); }
    inline int64_t* comp_index_dev_row(int k) const { return reinterpret_cast<int64_t*>(saver_->comp_index_dev.data_ptr()) + static_cast<int64_t>(k) * (sweep_nvcomp::kIdxSizes + comp_max_pieces_); }
    inline int64_t* comp_index_host_row(int k) const { return reinterpret_cast<int64_t*>(saver_->comp_index_host.data_ptr()) + static_cast<int64_t>(k) * (sweep_nvcomp::kIdxSizes + comp_max_pieces_); }
    inline int64_t comp_index_row_bytes() const { return static_cast<int64_t>(sweep_nvcomp::kIdxSizes + comp_max_pieces_) * 8; }
    inline char* comp_bound_ptr(int slot) const { return static_cast<char*>(saver_->comp_bound.data_ptr()) + static_cast<int64_t>(slot) * comp_cap_; }
    // A pack slot is [packed payload (cap)] [the chunk's scales]: the scales are
    // copied out of their ring slot at compress time, so the deferred landing
    // never reads the ring (which the next chunk in that ring slot rewrites).
    inline char* comp_pack_ptr(int slot) const { return static_cast<char*>(saver_->comp_pack.data_ptr()) + static_cast<int64_t>(slot) * (comp_cap_ + comp_scale_full_bytes_); }
    inline char* comp_stage_ptr(int slot) const { return static_cast<char*>(saver_->comp_persist.data_ptr()) + static_cast<int64_t>(slot) * (comp_cap_ + comp_scale_full_bytes_); }
    inline char* comp_persist_ptr() const { return static_cast<char*>(saver_->comp_persist.data_ptr()); }

    inline void init_compress()
    {
#if defined(SWEEP_HOST_CORE)
        SWEEP_CHECK(false, "boundary compression (BoundarySaving(compress=...)) is not available on the host core");
#else
        SWEEP_CHECK(compress_tier_ != 2 || boundary_disk_async_read_,
                    "boundary compression with storage='disk' needs disk_async_read=True "
                    "(the packed chunks are fetched by the reader thread).");
        const auto& S = *saver_;
        comp_elem_size_ = static_cast<int>(S.top_gpu.element_size());
        sweep_nvcomp::check_alignments(comp_elem_size_);
        const int R = ring_buffers_;
        comp_max_pieces_ = static_cast<int>(S.comp_status.numel() / 4 / (R + 1));
        SWEEP_CHECK(comp_max_pieces_ > 0 && S.comp_status.numel() == static_cast<int64_t>(comp_max_pieces_) * 4 * (R + 1),
                    "boundary_compress status buffer is not (ring_buffers + 1) x max_pieces int32");
        SWEEP_CHECK(S.comp_tab.numel() == static_cast<int64_t>(R + 1) * sweep_nvcomp::kTabRows * comp_max_pieces_ * 8,
                    "boundary_compress table buffer is not (ring_buffers + 1) x 6 x max_pieces int64");
        SWEEP_CHECK(S.comp_status_host.numel() == S.comp_status.numel(), "boundary_compress status mirror size mismatch");
        comp_cap_ = S.comp_bound.numel() / R;
        SWEEP_CHECK(comp_cap_ > 0 && S.comp_bound.numel() == comp_cap_ * R,
                    "boundary_compress bound buffer must be ring_buffers equal slots");
        comp_scale_full_bytes_ = comp_scale_bytes(transfer_interval_);
        comp_pack_slots_ = static_cast<int>(S.comp_pack.numel() / (comp_cap_ + comp_scale_full_bytes_));
        SWEEP_CHECK(comp_pack_slots_ >= R && S.comp_pack.numel() == (comp_cap_ + comp_scale_full_bytes_) * comp_pack_slots_,
                    "boundary_compress pack buffer must be >= ring_buffers slots of (bound slot + scale chunk) bytes");
        comp_nchunks_ = S.comp_index_dev.numel() / comp_index_row_bytes();
        SWEEP_CHECK(comp_nchunks_ > 0 && S.comp_index_dev.numel() == comp_nchunks_ * comp_index_row_bytes()
                    && S.comp_index_host.numel() == S.comp_index_dev.numel(),
                    "boundary_compress index buffers are not nchunks x (2 + max_pieces) int64");
        SWEEP_CHECK(S.comp_stats_host.numel() >= 64, "boundary_compress stats buffer must hold 8 int64");
        comp_maxout_ = sweep_nvcomp::max_compressed_piece_bytes(comp_elem_size_);
        comp_temp_bytes_ = sweep_nvcomp::decompress_temp_bytes(comp_max_pieces_, static_cast<int64_t>(comp_max_pieces_) * sweep_nvcomp::kPieceBytes);
        SWEEP_CHECK(comp_temp_bytes_ <= S.comp_temp.numel(),
                    "nvCOMP wants ", comp_temp_bytes_, " B of decompression scratch but boundary_compress "
                    "temp holds ", S.comp_temp.numel(), " B");
        comp_persist_bytes_ = S.comp_persist.numel();
        SWEEP_CHECK(S.comp_cfg.numel() >= kCfgHeader + comp_nchunks_, "boundary_compress cfg must hold 64 + nchunks bytes");
        const int64_t* cfg = comp_cfg_words();
        comp_ring_steps_ = cfg[0];
        // the arena the circular allocation runs in: the device / pinned arena
        // itself, or for the disk tier the payload file (cfg[1]), not the stage
        comp_arena_bytes_ = compress_tier_ == 2 ? cfg[1] : comp_persist_bytes_;
        SWEEP_CHECK(comp_arena_bytes_ > 0, "boundary_compress: the arena / packed file size is not set (cfg[1])");
        SWEEP_CHECK(cfg[2] == transfer_interval_ && cfg[3] == ring_buffers_,
                    "boundary_compress workspace was sized for transfer_interval=", cfg[2], " ring_buffers=", cfg[3],
                    " but the runtime has ", transfer_interval_, "/", ring_buffers_);
        comp_basis_bytes_ = cfg[5];
        comp_extra_bytes_ = cfg[6];
        comp_wrap_slack_ = static_cast<int>(cfg[7] & 1);   // bit 1 (disk) is for the Python report
        // Live window: the ring holds ring_steps steps, i.e. parts of at most
        // ceil(ring_steps / TI) + 1 chunks (one partial at each end when a
        // refill starts mid-chunk); older ones are overwritten and evicted.
        comp_window_ = comp_ring_steps_ > 0
            ? static_cast<int>((comp_ring_steps_ + transfer_interval_ - 1) / transfer_interval_) + 1 : 0;
        if (compress_tier_ == 2) {
            SWEEP_CHECK(comp_persist_bytes_ == (comp_cap_ + comp_scale_full_bytes_) * comp_pack_slots_,
                        "boundary_compress disk stage must be one (cap + scale chunk)-sized slot per pack slot");
            SWEEP_CHECK(disk_files_ != nullptr && !disk_files_->empty(),
                        "boundary_compress on storage='disk' needs the packed file in boundary_disk_files");
        }
        comp_pack_ready_.resize(comp_pack_slots_, nullptr);
        for (int i = 0; i < comp_pack_slots_; ++i)
            cudaEventCreateWithFlags(&comp_pack_ready_[i], cudaEventDisableTiming);
        comp_status_used_.assign(R + 1, 0);
        comp_slot_chunk_.assign(R, -1);
        build_compress_tables();
        // Not begin_forward() here: the live flags are the propagator's memory
        // and describe what the LAST forward left in the arena; a backward's
        // (per-call) runtime must read them, not clear them.  BoundaryScope
        // calls begin_forward() when a forward phase starts.
#endif
    }

    // The uncompressed pieces of slot ``slot`` holding ``len`` steps: one
    // contiguous region per face (per field in 2-D), split into kPieceBytes.
    inline std::vector<std::pair<int64_t, int64_t>> comp_pieces(int slot, int len) const
    {
        const auto& S = *saver_;
        const int64_t es = comp_elem_size_;
        std::vector<std::pair<int64_t, int64_t>> out;
        auto add_region = [&](int64_t start, int64_t bytes) {
            SWEEP_CHECK(start % sweep_nvcomp::kAlign == 0,
                        "boundary compression: a ring region starts at byte ", start, " of its buffer, "
                        "not a multiple of ", sweep_nvcomp::kAlign, " (the Python side pads the ring's "
                        "per-step pitch when compress is on; this ring was not allocated that way)");
            for (int64_t off = 0; off < bytes; off += sweep_nvcomp::kPieceBytes)
                out.emplace_back(start + off, std::min<int64_t>(sweep_nvcomp::kPieceBytes, bytes - off));
        };
        const Buf* faces3[6] = {&S.top_gpu, &S.bottom_gpu, &S.front_gpu, &S.back_gpu, &S.left_gpu, &S.right_gpu};
        const Buf* faces2[4] = {&S.top_gpu, &S.bottom_gpu, &S.left_gpu, &S.right_gpu};
        const int nf = dim_ == 3 ? 6 : 4;
        for (int f = 0; f < nf; ++f) {
            const Buf& b = dim_ == 3 ? *faces3[f] : *faces2[f];
            if (!b.defined() || b.numel() == 0) continue;   // DD cut face
            const int64_t base = reinterpret_cast<int64_t>(b.data_ptr());
            if (dim_ == 3) {
                const int64_t pitch = b.stride(0);   // one (step, field) slot
                add_region(base + static_cast<int64_t>(slot) * transfer_interval_ * S.nvar * pitch * es,
                           static_cast<int64_t>(len) * S.nvar * pitch * es);
            } else {
                const int64_t var_block = b.stride(0), pitch = b.stride(1);
                for (int v = 0; v < S.nvar; ++v)
                    add_region(base + (v * var_block + static_cast<int64_t>(slot) * transfer_interval_ * pitch) * es,
                               static_cast<int64_t>(len) * pitch * es);
            }
        }
        SWEEP_CHECK(static_cast<int>(out.size()) <= comp_max_pieces_,
                    "boundary compression: slot ", slot, " at ", len, " steps needs ", out.size(),
                    " pieces but the Python workspace holds ", comp_max_pieces_);
        return out;
    }

    // The FP32 per-block scales of a chunk (int8 / fp16 rings): one region per
    // face (per field in 2-D) of the scale RING, in the same order and
    // step-geometry as comp_pieces.  They ride along uncompressed, appended to
    // the packed payload in the arena / file, so the compressed store needs no
    // persistent scale buffer of its own -- which is what lets a ring that is
    // refilled per segment (ModelParallel segment_steps) work unchanged.
    inline std::vector<std::pair<char*, int64_t>> comp_scale_regions(int slot, int len) const
    {
        std::vector<std::pair<char*, int64_t>> out;
        if (!comp_scaled()) return out;
        const auto& S = *saver_;
        const Buf* faces3[6] = {&S.top_scale_gpu, &S.bottom_scale_gpu, &S.front_scale_gpu, &S.back_scale_gpu, &S.left_scale_gpu, &S.right_scale_gpu};
        const Buf* faces2[4] = {&S.top_scale_gpu, &S.bottom_scale_gpu, &S.left_scale_gpu, &S.right_scale_gpu};
        const int nf = dim_ == 3 ? 6 : 4;
        for (int f = 0; f < nf; ++f) {
            const Buf& b = dim_ == 3 ? *faces3[f] : *faces2[f];
            if (!b.defined() || b.numel() == 0) continue;
            char* base = static_cast<char*>(b.data_ptr());
            if (dim_ == 3) {
                const int64_t pitch = b.stride(0);
                out.emplace_back(base + static_cast<int64_t>(slot) * transfer_interval_ * S.nvar * pitch * 4,
                                 static_cast<int64_t>(len) * S.nvar * pitch * 4);
            } else {
                const int64_t var_block = b.stride(0), pitch = b.stride(1);
                for (int v = 0; v < S.nvar; ++v)
                    out.emplace_back(base + (v * var_block + static_cast<int64_t>(slot) * transfer_interval_ * pitch) * 4,
                                     static_cast<int64_t>(len) * pitch * 4);
            }
        }
        return out;
    }
    inline int64_t comp_scale_bytes(int len) const
    {
        int64_t n = 0;
        for (const auto& r : comp_scale_regions(0, len)) n += r.second;
        return sweep_nvcomp::align_up(n);
    }
    // Move a chunk's scale regions between the scale ring and a contiguous
    // run at ``dst``/``src`` (arena or pinned stage), in region order.
    inline void comp_copy_scales(char* lin, int slot, int len, bool ring_to_lin, cudaMemcpyKind kind)
    {
        int64_t off = 0;
        for (const auto& r : comp_scale_regions(slot, len)) {
            if (ring_to_lin)
                cudaMemcpyAsync(lin + off, r.first, static_cast<size_t>(r.second), kind, copy_stream_);
            else
                cudaMemcpyAsync(r.first, lin + off, static_cast<size_t>(r.second), kind, copy_stream_);
            off += r.second;
        }
    }

    inline void fill_table(int t, int bound_slot, int slot, int len, cudaStream_t stream)
    {
        const auto pieces = comp_pieces(slot, len);
        const int n = static_cast<int>(pieces.size());
        comp_npieces_[t] = n;
        int64_t raw = 0;
        int64_t* unc_ptr = comp_tab_host(t, sweep_nvcomp::kRowUncPtr);
        int64_t* unc_bytes = comp_tab_host(t, sweep_nvcomp::kRowUncBytes);
        int64_t* bound_ptr = comp_tab_host(t, sweep_nvcomp::kRowBoundPtr);
        const int64_t bound_base = reinterpret_cast<int64_t>(comp_bound_ptr(bound_slot));
        for (int i = 0; i < n; ++i) {
            unc_ptr[i] = pieces[i].first;
            unc_bytes[i] = pieces[i].second;
            bound_ptr[i] = bound_base + static_cast<int64_t>(i) * comp_maxout_;
            raw += pieces[i].second;
        }
        comp_raw_per_table_[t] = raw;
        SWEEP_CHECK(static_cast<int64_t>(n) * comp_maxout_ <= comp_cap_,
                    "boundary compression: ", n, " pieces x ", comp_maxout_, " B bound exceeds the ", comp_cap_,
                    " B workspace slot");
        // Rows 0..2 of table t, in one copy (the host vector is a member, alive
        // for the runtime; a pageable async copy is staged by the driver).
        cudaMemcpyAsync(comp_tab_dev(t, 0), comp_tab_host(t, 0),
                        static_cast<size_t>(3) * comp_max_pieces_ * 8, cudaMemcpyHostToDevice, stream);
    }

    inline void build_compress_tables()
    {
        const int R = ring_buffers_;
        comp_tab_host_.assign(static_cast<size_t>(R + 1) * sweep_nvcomp::kTabRows * comp_max_pieces_, 0);
        comp_npieces_.assign(R + 1, 0);
        comp_raw_per_table_.assign(R + 1, 0);
        for (int s = 0; s < R; ++s)
            fill_table(s, s, s, transfer_interval_, copy_stream_);
        comp_raw_step_bytes_ = comp_raw_per_table_[0] / transfer_interval_;
        comp_tail_slot_ = -1;
        comp_tail_len_ = -1;
        comp_ring_ptr_ = saver_->top_gpu.data_ptr();
        comp_tab_ptr_ = saver_->comp_tab.data_ptr();
    }

    // Table index for a chunk of ``len`` steps in ``slot``: its slot's table
    // at a full interval, else the tail table (built for the first short chunk
    // seen; there is exactly one per phase).
    inline int comp_table_for(int slot, int len)
    {
        if (len == transfer_interval_) return slot;
        const int R = ring_buffers_;
        if (comp_tail_slot_ != slot || comp_tail_len_ != len) {
            fill_table(R, slot, slot, len, copy_stream_);
            comp_tail_slot_ = slot;
            comp_tail_len_ = len;
        }
        return R;
    }

    inline void compress_chunk(const BoundaryChunk& chunk)
    {
#if !defined(SWEEP_HOST_CORE)
        const int slot = chunk.ring_slot, len = chunk.chunk_len, k = chunk.chunk_id;
        SWEEP_CHECK(k < comp_nchunks_, "boundary compression: chunk ", k, " beyond the ", comp_nchunks_,
                    "-row index (the propagator sizes it by ceil(nt / transfer_interval); "
                    "nt or transfer_interval changed under the workspace?)");
        // A chunk id at or below the last one compressed in this phase means the
        // ring is being REFILLED (a segmented ring re-runs each segment's forward
        // inside the backward): land what is pending and start the arena over.
        // The arena therefore has to hold one refill, which is what the Python
        // side sizes it from (nt_saved = the ring's steps), not the whole record.
        if (k == comp_last_chunk_ && len > comp_last_len_) {
            // The same chunk again with MORE steps: a continuation, not a
            // refill -- a per-call runtime (no session) flushes the open chunk
            // at every call's end and the next call grows it.  The previous
            // compression of this chunk is superseded: drop it from the
            // pending list, or from the live set (it is the newest, so the
            // arena simply reuses its bytes).
            for (auto it = pending_land_.begin(); it != pending_land_.end(); ++it)
                if (*it == k) { pending_land_.erase(it); break; }
            if (!comp_live_.empty() && comp_live_.back().k == k) {
                *comp_live_flag(k) = 0;
                comp_live_.pop_back();
            }
        } else if (k <= comp_last_chunk_) {
            land_pending(-1, true);
            comp_live_.clear();
            comp_clear_live_flags();
            arena_fill_ = 0;
        }
        comp_last_chunk_ = k;
        comp_last_len_ = len;
        const int t = comp_table_for(slot, len);
        const int n = comp_npieces_[t];
        // The pack slots outnumber the ring slots so a landing can usually wait
        // for its pack to finish without a host sync: the chunk that last used
        // this pack slot must be landed before it is rewritten.
        const int ps = k % comp_pack_slots_;
        land_pending(ps, false);
        // The chunk's scales go behind the payload area of the pack slot now
        // (copy stream, before copy_ready releases the ring slot); the landing
        // appends them to the packed payload from there.
        comp_copy_scales(comp_pack_ptr(ps) + comp_cap_, slot, len, true, cudaMemcpyDeviceToDevice);
        if (n > 0) {
            const auto& api = sweep_nvcomp::require();
            auto st = api.compress(
                reinterpret_cast<const void* const*>(comp_tab_dev(t, sweep_nvcomp::kRowUncPtr)),
                reinterpret_cast<const size_t*>(comp_tab_dev(t, sweep_nvcomp::kRowUncBytes)),
                static_cast<size_t>(sweep_nvcomp::kPieceBytes), static_cast<size_t>(n), nullptr, 0,
                reinterpret_cast<void* const*>(comp_tab_dev(t, sweep_nvcomp::kRowBoundPtr)),
                reinterpret_cast<size_t*>(comp_tab_dev(t, sweep_nvcomp::kRowCompBytes)),
                sweep_nvcomp::compress_opts(comp_elem_size_), comp_status_dev(t), copy_stream_);
            SWEEP_CHECK(st == sweep_nvcomp::kSuccess, "nvcompBatchedBitcompCompressAsync failed: status ", st);
            sweep_nvcomp::scan_piece_sizes_kernel<<<1, sweep_nvcomp::kScanThreads, 0, copy_stream_>>>(
                comp_tab_dev(t, sweep_nvcomp::kRowCompBytes), n, comp_pack_ptr(ps),
                comp_tab_dev(t, sweep_nvcomp::kRowCompPtr),
                comp_index_dev_row(k) + sweep_nvcomp::kIdxTotal, comp_index_dev_row(k) + sweep_nvcomp::kIdxSizes,
                comp_index_dev_row(k) + sweep_nvcomp::kIdxLen, static_cast<int64_t>(len));
            sweep_nvcomp::pack_pieces_kernel<<<n, 256, 0, copy_stream_>>>(
                comp_tab_dev(t, sweep_nvcomp::kRowBoundPtr), comp_tab_dev(t, sweep_nvcomp::kRowCompBytes),
                comp_tab_dev(t, sweep_nvcomp::kRowCompPtr));
            cudaMemcpyAsync(comp_status_host(t), comp_status_dev(t), static_cast<size_t>(n) * 4,
                            cudaMemcpyDeviceToHost, copy_stream_);
            comp_status_used_[t] = 1;
        } else {
            // Nothing to compress (every face cut): the row still records the
            // step count, with a zero total, so the backward fetch accepts it.
            sweep_nvcomp::scan_piece_sizes_kernel<<<1, sweep_nvcomp::kScanThreads, 0, copy_stream_>>>(
                comp_tab_dev(t, sweep_nvcomp::kRowCompBytes), 0, comp_pack_ptr(ps),
                comp_tab_dev(t, sweep_nvcomp::kRowCompPtr),
                comp_index_dev_row(k) + sweep_nvcomp::kIdxTotal, nullptr,
                comp_index_dev_row(k) + sweep_nvcomp::kIdxLen, static_cast<int64_t>(len));
        }
        cudaMemcpyAsync(comp_index_host_row(k), comp_index_dev_row(k), static_cast<size_t>(comp_index_row_bytes()),
                        cudaMemcpyDeviceToHost, copy_stream_);
        cudaEventRecord(copy_ready_[slot], copy_stream_);
        copy_pending_[slot] = 1;
        cudaEventRecord(comp_pack_ready_[ps], copy_stream_);
        pending_land_.push_back(k);
        comp_raw_bytes_ += comp_raw_per_table_[t];
        if (profile_enabled_) {
            ++profile_forward_chunks_;
            if (boundary_on_disk_)
                forward_copy_profile_pending_[slot] = 1;
        }
#endif
    }

    // Everything the forward has saved so far is in the arena after this: the
    // chunk still open (a ring refilled per segment stops mid-chunk) is
    // compressed, every pending landing is issued.  Called at the end of a
    // phase and before any backward fetch -- a driver that refills the ring
    // inside its backward phase never flips the phase, so the fetch itself
    // settles.  A no-op in a plain backward.
    inline void comp_settle()
    {
        if (comp_open_valid_) {
            BoundaryChunk open = comp_open_;
            comp_open_valid_ = false;
            cudaEventRecord(compute_ready_[open.ring_slot], compute_stream_);
            cudaStreamWaitEvent(copy_stream_, compute_ready_[open.ring_slot], 0);
            compress_chunk(open);
        }
        land_pending(/*force_slot=*/-1, /*force_all=*/true);
    }

    // Issue the persistent landing of every pending chunk whose pack is done
    // (in order; a not-yet-done chunk stops the sweep), forcing the one whose
    // pack slot is ``force_slot`` (about to be rewritten) or all of them.
    inline void land_pending(int force_slot, bool force_all)
    {
#if !defined(SWEEP_HOST_CORE)
        while (!pending_land_.empty()) {
            const int k = pending_land_.front();
            const int s = k % comp_pack_slots_;
            if (comp_window_ > 0 && k <= comp_last_chunk_ - comp_window_) {
                pending_land_.pop_front();   // its steps are already overwritten in the ring
                continue;
            }
            if (force_all || s == force_slot) {
                auto wait_start = Clock::now();
                cudaEventSynchronize(comp_pack_ready_[s]);
                if (profile_enabled_) {
                    profile_compress_land_wait_s_ += elapsed_seconds(wait_start, Clock::now());
                    ++profile_compress_forced_lands_;
                }
            } else {
                cudaError_t q = cudaEventQuery(comp_pack_ready_[s]);
                if (q == cudaErrorNotReady) {
                    (void)cudaGetLastError();   // not-ready is not an error
                    break;
                }
                SWEEP_CHECK(q == cudaSuccess, "cudaEventQuery failed: ", cudaGetErrorString(q));
            }
            land_chunk(k, s);
            pending_land_.pop_front();
        }
#else
        (void)force_slot; (void)force_all;
#endif
    }

    inline void land_chunk(int k, int s)
    {
#if !defined(SWEEP_HOST_CORE)
        int64_t* row = comp_index_host_row(k);
        const int64_t total = row[sweep_nvcomp::kIdxTotal];
        const int len = static_cast<int>(row[sweep_nvcomp::kIdxLen]);
        const int64_t sbytes = comp_scale_bytes(len);
        // Evict the chunks the step ring has overwritten, then allocate
        // circularly after the newest live chunk (wrapping to 0 when the tail
        // does not fit), so the arena holds one fill of the ring, not the
        // record.  For the disk file the "arena" is the file, sized the same.
        if (comp_window_ > 0)
            while (!comp_live_.empty() && comp_live_.front().k <= k - comp_window_) {
                *comp_live_flag(comp_live_.front().k) = 0;
                comp_live_.pop_front();
            }
        const int64_t need = total + sbytes;
        int64_t offset = 0;
        bool fits = need <= comp_arena_bytes_;
        if (!comp_live_.empty()) {
            const int64_t head = comp_live_.front().off;
            const int64_t tail = comp_live_.back().off + comp_live_.back().bytes;
            if (tail >= head) {            // live run [head, tail): free after tail, or before head
                if (tail + need <= comp_arena_bytes_) offset = tail;
                else if (need <= head) offset = 0;
                else fits = false;
            } else {                       // wrapped: live run [head, cap) + [0, tail)
                if (tail + need <= head) offset = tail;
                else fits = false;
            }
        }
        const int64_t raw = comp_raw_step_bytes_ * len;
        int64_t live_bytes = need, live_raw = raw, live_packed = total, max_chunk = need;
        for (const auto& c : comp_live_) {
            live_bytes += c.bytes; live_raw += c.raw; live_packed += c.packed;
            if (c.bytes > max_chunk) max_chunk = c.bytes;
        }
        if (!fits) {
            // The budget this live set needs, in the units Python sizes the
            // arena with (budget x basis + extra, one chunk of slack where the
            // arena is refilled and may waste that much at the wrap).
            const int64_t want = live_bytes + (comp_wrap_slack_ ? max_chunk : 0) - comp_extra_bytes_;
            const double budget = comp_basis_bytes_ > 0 ? static_cast<double>(want) / static_cast<double>(comp_basis_bytes_) : 0.0;
            SWEEP_CHECK(false,
                        "boundary compression: chunk ", k, " (", need, " B packed + scales) does not fit the arena of ",
                        comp_arena_bytes_, " B; the live packed chunks hold ", live_bytes - need, " B for ",
                        live_raw - raw, " B of ring (ratio ", live_raw - raw > 0 && live_bytes - need > 0
                        ? static_cast<double>(live_raw - raw) / static_cast<double>(live_bytes - need) : 0.0,
                        " including scales). This live set needs BoundarySaving(compress_budget=", budget,
                        "); set at least ", budget * 1.1, " (the late chunks of a record pack worse than its early ones), "
                        "or use storage='disk' where the file is sparse.");
        }
        comp_live_.push_back(LiveChunk{k, offset, need, raw, total});
        *comp_live_flag(k) = 1;
        row[sweep_nvcomp::kIdxOffset] = offset;
        char* src = comp_pack_ptr(s);
        char* src_scales = src + comp_cap_;
        if (compress_tier_ == 0) {
            if (total > 0)
                cudaMemcpyAsync(comp_persist_ptr() + offset, src, static_cast<size_t>(total), cudaMemcpyDeviceToDevice, copy_stream_);
            if (sbytes > 0)
                cudaMemcpyAsync(comp_persist_ptr() + offset + total, src_scales, static_cast<size_t>(sbytes), cudaMemcpyDeviceToDevice, copy_stream_);
        } else if (compress_tier_ == 1) {
            if (total > 0)
                cudaMemcpyAsync(comp_persist_ptr() + offset, src, static_cast<size_t>(total), cudaMemcpyDeviceToHost, copy_stream_);
            if (sbytes > 0)
                cudaMemcpyAsync(comp_persist_ptr() + offset + total, src_scales, static_cast<size_t>(sbytes), cudaMemcpyDeviceToHost, copy_stream_);
        } else if (total + sbytes > 0) {
            char* stage = comp_stage_ptr(s);
            if (total > 0)
                cudaMemcpyAsync(stage, src, static_cast<size_t>(total), cudaMemcpyDeviceToHost, copy_stream_);
            if (sbytes > 0)
                cudaMemcpyAsync(stage + total, src_scales, static_cast<size_t>(sbytes), cudaMemcpyDeviceToHost, copy_stream_);
            auto* meta = new BoundaryPackedMeta();
            meta->path = (*disk_files_)[0];
            meta->offset = static_cast<size_t>(offset);
            meta->bytes = static_cast<size_t>(total + sbytes);
            meta->src = stage;
            cudaLaunchHostFunc(copy_stream_, write_boundary_packed_callback, meta);
        }
        arena_fill_ = offset + need;
        comp_packed_bytes_ += total;
        ++comp_chunks_landed_;
        // The stats mirror follows every landing (host words the propagator
        // reads for last_boundary_compress_ratio): a driver that keeps one
        // session for the whole run never reaches the phase-end write.  Words
        // 4..7 track the FULLEST live set of the call (the refills of a
        // segmented ring overwrite the arena; the budget has to cover the
        // worst of them, not the last).
        int64_t* stats = reinterpret_cast<int64_t*>(saver_->comp_stats_host.data_ptr());
        stats[0] = comp_packed_bytes_;
        stats[1] = comp_raw_bytes_;
        stats[2] = comp_chunks_landed_;
        stats[3] = comp_arena_bytes_;
        if (live_bytes > stats[6]) {
            stats[4] = live_packed;   // payload bytes of the fullest live set
            stats[5] = live_raw;      // its uncompressed bytes
            stats[6] = live_bytes;    // what it occupies (payload + scales + alignment)
        }
        if (max_chunk > stats[7]) stats[7] = max_chunk;   // the largest packed chunk seen
#else
        (void)k; (void)s;
#endif
    }

    // Backward: bring chunk ``start`` back into ring slot ``slot`` -- for the
    // disk tier from the pinned stage the reader thread just filled, else
    // straight from the arena -- and decompress it there.  The geometry is the
    // one the chunk was COMPRESSED with (its step count is in the index row):
    // the backward's own chunk length is one step shorter at the top, and a
    // segmented ring cuts chunks at segment ends.
    inline void fetch_compressed_chunk(int start, int slot)
    {
#if !defined(SWEEP_HOST_CORE)
        const int k = start / transfer_interval_;
        comp_slot_chunk_[slot] = k;
        SWEEP_CHECK(k < comp_nchunks_, "boundary compression: backward asks for chunk ", k,
                    " beyond the ", comp_nchunks_, "-row index");
        const int64_t* row = comp_index_host_row(k);
        const int64_t offset = row[sweep_nvcomp::kIdxOffset], total = row[sweep_nvcomp::kIdxTotal];
        const int fwd_len = static_cast<int>(row[sweep_nvcomp::kIdxLen]);
        SWEEP_CHECK(*comp_live_flag(k) == 1, "boundary compression: the backward asks for chunk ", k,
                    " (steps ", k * transfer_interval_, "..), which is not in the compressed ring: its steps were "
                    "overwritten (the ring holds its last ", comp_ring_steps_ > 0 ? comp_ring_steps_ : static_cast<int64_t>(comp_nchunks_) * transfer_interval_,
                    " steps) or it was never compressed in the phase that filled the ring.");
        SWEEP_CHECK(fwd_len >= 1 && fwd_len <= transfer_interval_,
                    "boundary compression: chunk ", k, " was never compressed in this phase "
                    "(index row holds ", fwd_len, " steps); the forward that filled the ring "
                    "did not flush it, or the backward reads a chunk outside the refilled segment.");
        const int t = comp_table_for(slot, fwd_len);
        const int n = comp_npieces_[t];
        const int stage_start = slot * transfer_interval_;
        // write-after-read: the previous chunk's restores must be done with
        // this ring slot before the decompressor overwrites it.
        cudaStreamWaitEvent(copy_stream_, compute_ready_[slot], 0);
        if (profile_enabled_ && boundary_on_disk_) {
            cudaEventRecord(backward_copy_start_[slot], copy_stream_);
            cudaEventRecord(backward_h2d_start_[slot], copy_stream_);
        }
        if (total > 0) {
            if (compress_tier_ == 0)
                cudaMemcpyAsync(comp_pack_ptr(slot), comp_persist_ptr() + offset, static_cast<size_t>(total), cudaMemcpyDeviceToDevice, copy_stream_);
            else if (compress_tier_ == 1)
                cudaMemcpyAsync(comp_pack_ptr(slot), comp_persist_ptr() + offset, static_cast<size_t>(total), cudaMemcpyHostToDevice, copy_stream_);
            else
                cudaMemcpyAsync(comp_pack_ptr(slot), comp_stage_ptr(slot), static_cast<size_t>(total), cudaMemcpyHostToDevice, copy_stream_);
        }
        (void)stage_start;
        if (compress_tier_ == 0)
            comp_copy_scales(comp_persist_ptr() + offset + total, slot, fwd_len, false, cudaMemcpyDeviceToDevice);
        else if (compress_tier_ == 1)
            comp_copy_scales(comp_persist_ptr() + offset + total, slot, fwd_len, false, cudaMemcpyHostToDevice);
        else
            comp_copy_scales(comp_stage_ptr(slot) + total, slot, fwd_len, false, cudaMemcpyHostToDevice);
        if (n > 0) {
            sweep_nvcomp::scan_piece_sizes_kernel<<<1, sweep_nvcomp::kScanThreads, 0, copy_stream_>>>(
                comp_index_dev_row(k) + sweep_nvcomp::kIdxSizes, n, comp_pack_ptr(slot),
                comp_tab_dev(t, sweep_nvcomp::kRowCompPtr), nullptr, nullptr, nullptr, 0);
            const auto& api = sweep_nvcomp::require();
            auto st = api.decompress(
                reinterpret_cast<const void* const*>(comp_tab_dev(t, sweep_nvcomp::kRowCompPtr)),
                reinterpret_cast<const size_t*>(comp_index_dev_row(k) + sweep_nvcomp::kIdxSizes),
                reinterpret_cast<const size_t*>(comp_tab_dev(t, sweep_nvcomp::kRowUncBytes)),
                reinterpret_cast<size_t*>(comp_tab_dev(t, sweep_nvcomp::kRowActual)),
                static_cast<size_t>(n), saver_->comp_temp.data_ptr(), static_cast<size_t>(comp_temp_bytes_),
                reinterpret_cast<void* const*>(comp_tab_dev(t, sweep_nvcomp::kRowUncPtr)),
                sweep_nvcomp::decompress_opts(), comp_status_dev(t), copy_stream_);
            SWEEP_CHECK(st == sweep_nvcomp::kSuccess, "nvcompBatchedBitcompDecompressAsync failed: status ", st);
            cudaMemcpyAsync(comp_status_host(t), comp_status_dev(t), static_cast<size_t>(n) * 4,
                            cudaMemcpyDeviceToHost, copy_stream_);
            comp_status_used_[t] = 1;
        }
        cudaEventRecord(copy_ready_[slot], copy_stream_);
        if (profile_enabled_ && boundary_on_disk_)
            backward_copy_profile_pending_[slot] = 1;
#endif
    }

    // After the copy stream is drained: every batch's statuses are on the host.
    inline void finish_compress_phase()
    {
        for (int t = 0; t <= ring_buffers_; ++t) {
            if (!comp_status_used_[t]) continue;
            sweep_nvcomp::check_statuses(comp_status_host(t), comp_npieces_[t], "compress/decompress");
            comp_status_used_[t] = 0;
        }
        int64_t* stats = reinterpret_cast<int64_t*>(saver_->comp_stats_host.data_ptr());
        if (comp_chunks_landed_ > 0) {
            stats[0] = comp_packed_bytes_;
            stats[1] = comp_raw_bytes_;
            stats[2] = comp_chunks_landed_;
            stats[3] = comp_arena_bytes_;
        }
    }
};
