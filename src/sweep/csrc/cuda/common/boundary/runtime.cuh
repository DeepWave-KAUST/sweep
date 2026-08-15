#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include <string>

#include "saver.cuh"

// Bind one staged boundary face to the typed pointer matching the staging
// tensor's dtype (fp32/fp16/bf16) at element offset OFF.  Mirrors view()'s
// GPU-direct binding so the existing dtype-dispatched save/restore kernels work
// for the cpu/disk staging ring too.  OFF is an element offset (dtype-agnostic).
// A macro (not a function) so it assigns directly to the __restrict__-qualified
// pointer members of GeneralBoundaryPointer (a reference param would drop the
// restrict qualifier).
#define BIND_STAGED_FACE(STG, OFF, PF, PH, PBF)                                \
    do {                                                                       \
        const torch::Tensor& _stg = (STG);                                     \
        const int64_t _off = (OFF);                                            \
        const auto _st = _stg.scalar_type();                                   \
        if (_st == torch::kHalf)                                               \
            (PH) = reinterpret_cast<__half*>(_stg.data_ptr()) + _off;          \
        else if (_st == torch::kBFloat16)                                      \
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
        : saver_(saver),
          dim_(dim),
          enabled_(use_boundary_saving),
          staged_(boundary_on_cpu || boundary_on_disk),
          boundary_on_disk_(boundary_on_disk),
          boundary_disk_async_read_(boundary_on_disk && boundary_disk_async_read),
          transfer_interval_(transfer_interval > 0 ? transfer_interval : 1),
          ring_buffers_(ring_buffers > 0 ? ring_buffers : 1),
          disk_files_(disk_files),
          compute_stream_(compute_stream),
          copy_stream_(copy_stream)
    {
        profile_enabled_ = std::getenv("SWEEP_BOUNDARY_PROFILE") != nullptr;
        if (!enabled_ || !staged_)
            return;

        TORCH_CHECK(
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

        if (boundary_disk_async_read_)
            disk_reader_thread_ = std::thread(&BoundaryRuntime::disk_reader_loop, this);
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
    }

    inline bool enabled() const { return enabled_; }
    inline bool staged() const { return staged_; }

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
    // along the outermost dimension of the persistent payload buffer
    // (already includes any multi-field offset).  We look up the
    // per-step cell count from the saver's tensor stride.  The payload
    // is uint8 (INT8, *_q) or __half (FP16, *_h); the FP32 per-block
    // scale and the FP32 transient staging are shared by both.
    // ``face_t`` is the face's persistent saver tensor, passed only so a DD cut
    // face can be skipped: ``gpu_full_shapes`` drops such a face to numel 0
    // because the boundary kernels gate it on ctx.cut_* (nothing is ever
    // written there — its halo is reconstructed in the DD backward).  The test
    // MUST be on numel, not on ``cells``: PyTorch clamps a 0-size dim's stride
    // to a nonzero value, so the stride-derived ``cells`` would still drive a
    // launch that writes into the empty buffer.  Guarding here covers the FP16
    // and INT8 payloads in one place.
    inline bool face_is_cut(const torch::Tensor& face_t) const
    {
        return face_t.numel() == 0;
    }
    inline void quantize_face(const torch::Tensor& face_t,
                              float* stage, uint8_t* q, __half* h, float* scale,
                              BoundaryDtype dt, int64_t step_idx,
                              int64_t cells, int64_t blocks)
    {
        if (face_is_cut(face_t)) return;
        if (dt == BoundaryDtype::FP16)
            launch_quantize_fp16(stage, h + step_idx * cells, scale + step_idx * blocks, cells, compute_stream_);
        else
            launch_quantize_int8(stage, q + step_idx * cells, scale + step_idx * blocks, cells, compute_stream_);
    }
    inline void dequantize_face(const torch::Tensor& face_t,
                                float* stage, uint8_t* q, __half* h, float* scale,
                                BoundaryDtype dt, int64_t step_idx,
                                int64_t cells, int64_t blocks)
    {
        if (face_is_cut(face_t)) return;
        if (dt == BoundaryDtype::FP16)
            launch_dequantize_fp16(h + step_idx * cells, scale + step_idx * blocks, stage, cells, compute_stream_);
        else
            launch_dequantize_int8(q + step_idx * cells, scale + step_idx * blocks, stage, cells, compute_stream_);
    }
    inline void quantize_step_2d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        int64_t c_top = saver_.top_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t c_bot = saver_.bottom_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t c_lef = saver_.left_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t c_rig = saver_.right_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_top = saver_.top_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_bot = saver_.bottom_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_lef = saver_.left_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_rig = saver_.right_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        quantize_face(saver_.top_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx, c_top, s_top);
        quantize_face(saver_.bottom_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx, c_bot, s_bot);
        quantize_face(saver_.left_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx, c_lef, s_lef);
        quantize_face(saver_.right_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx, c_rig, s_rig);
    }
    inline void dequantize_step_2d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        int64_t c_top = saver_.top_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t c_bot = saver_.bottom_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t c_lef = saver_.left_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t c_rig = saver_.right_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_top = saver_.top_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_bot = saver_.bottom_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_lef = saver_.left_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        int64_t s_rig = saver_.right_scale_t.stride(saver_.dim == 3 ? 0 : 1);
        dequantize_face(saver_.top_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx, c_top, s_top);
        dequantize_face(saver_.bottom_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx, c_bot, s_bot);
        dequantize_face(saver_.left_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx, c_lef, s_lef);
        dequantize_face(saver_.right_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx, c_rig, s_rig);
    }
    inline void quantize_step_3d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        int64_t c_top = saver_.top_t.stride(0);
        int64_t c_bot = saver_.bottom_t.stride(0);
        int64_t c_fro = saver_.front_t.stride(0);
        int64_t c_bac = saver_.back_t.stride(0);
        int64_t c_lef = saver_.left_t.stride(0);
        int64_t c_rig = saver_.right_t.stride(0);
        int64_t s_top = saver_.top_scale_t.stride(0);
        int64_t s_bot = saver_.bottom_scale_t.stride(0);
        int64_t s_fro = saver_.front_scale_t.stride(0);
        int64_t s_bac = saver_.back_scale_t.stride(0);
        int64_t s_lef = saver_.left_scale_t.stride(0);
        int64_t s_rig = saver_.right_scale_t.stride(0);
        quantize_face(saver_.top_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx, c_top, s_top);
        quantize_face(saver_.bottom_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx, c_bot, s_bot);
        quantize_face(saver_.front_t,  b.front, b.front_q, b.front_h, b.front_scale, b.dtype, step_idx, c_fro, s_fro);
        quantize_face(saver_.back_t,   b.back,  b.back_q,  b.back_h,  b.back_scale,  b.dtype, step_idx, c_bac, s_bac);
        quantize_face(saver_.left_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx, c_lef, s_lef);
        quantize_face(saver_.right_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx, c_rig, s_rig);
    }
    inline void dequantize_step_3d(const GeneralBoundaryPointer& b, int64_t step_idx)
    {
        int64_t c_top = saver_.top_t.stride(0);
        int64_t c_bot = saver_.bottom_t.stride(0);
        int64_t c_fro = saver_.front_t.stride(0);
        int64_t c_bac = saver_.back_t.stride(0);
        int64_t c_lef = saver_.left_t.stride(0);
        int64_t c_rig = saver_.right_t.stride(0);
        int64_t s_top = saver_.top_scale_t.stride(0);
        int64_t s_bot = saver_.bottom_scale_t.stride(0);
        int64_t s_fro = saver_.front_scale_t.stride(0);
        int64_t s_bac = saver_.back_scale_t.stride(0);
        int64_t s_lef = saver_.left_scale_t.stride(0);
        int64_t s_rig = saver_.right_scale_t.stride(0);
        dequantize_face(saver_.top_t,    b.top,   b.top_q,   b.top_h,   b.top_scale,   b.dtype, step_idx, c_top, s_top);
        dequantize_face(saver_.bottom_t, b.bottom,b.bottom_q,b.bottom_h,b.bottom_scale,b.dtype, step_idx, c_bot, s_bot);
        dequantize_face(saver_.front_t,  b.front, b.front_q, b.front_h, b.front_scale, b.dtype, step_idx, c_fro, s_fro);
        dequantize_face(saver_.back_t,   b.back,  b.back_q,  b.back_h,  b.back_scale,  b.dtype, step_idx, c_bac, s_bac);
        dequantize_face(saver_.left_t,   b.left,  b.left_q,  b.left_h,  b.left_scale,  b.dtype, step_idx, c_lef, s_lef);
        dequantize_face(saver_.right_t,  b.right, b.right_q, b.right_h, b.right_scale, b.dtype, step_idx, c_rig, s_rig);
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
        ptr.dtype = boundary_dtype_from_tensor(saver_.top_gpu);
        ptr.last_two = direct.last_two;
        if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
            // Staged INT8: uint8 ring (cells/step) + scale ring (blocks/step)
            // + shared FP32 transient (direct.*).  Scale ring per-step stride
            // is its outer-time stride (dim 0 in 3D, dim 1 in 2D).
            const int sd = (dim_ == 3) ? 0 : 1;
            BIND_STAGED_SCALED(ptr.dtype, saver_.top_gpu,    gpu_idx * saver_.top_stride,    saver_.top_scale_gpu,    gpu_idx * saver_.top_scale_gpu.stride(sd),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
            BIND_STAGED_SCALED(ptr.dtype, saver_.bottom_gpu, gpu_idx * saver_.bottom_stride, saver_.bottom_scale_gpu, gpu_idx * saver_.bottom_scale_gpu.stride(sd), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
            BIND_STAGED_SCALED(ptr.dtype, saver_.left_gpu,   gpu_idx * saver_.left_stride,   saver_.left_scale_gpu,   gpu_idx * saver_.left_scale_gpu.stride(sd),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
            BIND_STAGED_SCALED(ptr.dtype, saver_.right_gpu,  gpu_idx * saver_.right_stride,  saver_.right_scale_gpu,  gpu_idx * saver_.right_scale_gpu.stride(sd),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            if (dim_ == 3) {
                BIND_STAGED_SCALED(ptr.dtype, saver_.front_gpu, gpu_idx * saver_.front_stride, saver_.front_scale_gpu, gpu_idx * saver_.front_scale_gpu.stride(sd), direct.front, ptr.front_q, ptr.front_h, ptr.front_scale, ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_.back_gpu,  gpu_idx * saver_.back_stride,  saver_.back_scale_gpu,  gpu_idx * saver_.back_scale_gpu.stride(sd),  direct.back,  ptr.back_q,  ptr.back_h,  ptr.back_scale,  ptr.back);
            }
            return ptr;
        }
        BIND_STAGED_FACE(saver_.top_gpu,    gpu_idx * saver_.top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
        BIND_STAGED_FACE(saver_.bottom_gpu, gpu_idx * saver_.bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
        BIND_STAGED_FACE(saver_.left_gpu,   gpu_idx * saver_.left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
        BIND_STAGED_FACE(saver_.right_gpu,  gpu_idx * saver_.right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);

        if (dim_ == 3) {
            BIND_STAGED_FACE(saver_.front_gpu, gpu_idx * saver_.front_stride, ptr.front, ptr.front_h, ptr.front_bf);
            BIND_STAGED_FACE(saver_.back_gpu,  gpu_idx * saver_.back_stride,  ptr.back,  ptr.back_h,  ptr.back_bf);
        }
        return ptr;
    }

    inline GeneralBoundaryPointer forward_save_ptrs_field(
        const BoundaryChunk& chunk,
        const GeneralBoundaryPointer& direct,
        int field_idx
    )
    {
        TORCH_CHECK(field_idx >= 0 && field_idx < saver_.nvar, "Invalid boundary field index.");

        if (dim_ == 2) {
            GeneralBoundaryPointer ptr{};
            ptr.last_two = direct.last_two;
            ptr.use_fp16 = direct.use_fp16;
            ptr.dtype = direct.dtype;

            if (staged_) {
                int gpu_idx = chunk.ring_start + chunk.buf_idx;
                ptr.dtype = boundary_dtype_from_tensor(saver_.top_gpu);
                if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                    BIND_STAGED_SCALED(ptr.dtype, saver_.top_gpu,
                        field_idx * saver_.top_gpu.stride(0) + gpu_idx * saver_.top_gpu.stride(1),
                        saver_.top_scale_gpu,
                        field_idx * saver_.top_scale_gpu.stride(0) + gpu_idx * saver_.top_scale_gpu.stride(1),
                        direct.top, ptr.top_q, ptr.top_h, ptr.top_scale, ptr.top);
                    BIND_STAGED_SCALED(ptr.dtype, saver_.bottom_gpu,
                        field_idx * saver_.bottom_gpu.stride(0) + gpu_idx * saver_.bottom_gpu.stride(1),
                        saver_.bottom_scale_gpu,
                        field_idx * saver_.bottom_scale_gpu.stride(0) + gpu_idx * saver_.bottom_scale_gpu.stride(1),
                        direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                    BIND_STAGED_SCALED(ptr.dtype, saver_.left_gpu,
                        field_idx * saver_.left_gpu.stride(0) + gpu_idx * saver_.left_gpu.stride(1),
                        saver_.left_scale_gpu,
                        field_idx * saver_.left_scale_gpu.stride(0) + gpu_idx * saver_.left_scale_gpu.stride(1),
                        direct.left, ptr.left_q, ptr.left_h, ptr.left_scale, ptr.left);
                    BIND_STAGED_SCALED(ptr.dtype, saver_.right_gpu,
                        field_idx * saver_.right_gpu.stride(0) + gpu_idx * saver_.right_gpu.stride(1),
                        saver_.right_scale_gpu,
                        field_idx * saver_.right_scale_gpu.stride(0) + gpu_idx * saver_.right_scale_gpu.stride(1),
                        direct.right, ptr.right_q, ptr.right_h, ptr.right_scale, ptr.right);
                } else {
                BIND_STAGED_FACE(saver_.top_gpu,
                    field_idx * saver_.top_gpu.stride(0) + gpu_idx * saver_.top_gpu.stride(1),
                    ptr.top, ptr.top_h, ptr.top_bf);
                BIND_STAGED_FACE(saver_.bottom_gpu,
                    field_idx * saver_.bottom_gpu.stride(0) + gpu_idx * saver_.bottom_gpu.stride(1),
                    ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
                BIND_STAGED_FACE(saver_.left_gpu,
                    field_idx * saver_.left_gpu.stride(0) + gpu_idx * saver_.left_gpu.stride(1),
                    ptr.left, ptr.left_h, ptr.left_bf);
                BIND_STAGED_FACE(saver_.right_gpu,
                    field_idx * saver_.right_gpu.stride(0) + gpu_idx * saver_.right_gpu.stride(1),
                    ptr.right, ptr.right_h, ptr.right_bf);
                }
            } else if (direct.dtype == BoundaryDtype::FP16) {
                ptr.top_h    = direct.top_h    + field_idx * saver_.top_t.stride(0);
                ptr.bottom_h = direct.bottom_h + field_idx * saver_.bottom_t.stride(0);
                ptr.left_h   = direct.left_h   + field_idx * saver_.left_t.stride(0);
                ptr.right_h  = direct.right_h  + field_idx * saver_.right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_.top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_.bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_.left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_.right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else if (direct.dtype == BoundaryDtype::BF16) {
                ptr.top_bf    = direct.top_bf    + field_idx * saver_.top_t.stride(0);
                ptr.bottom_bf = direct.bottom_bf + field_idx * saver_.bottom_t.stride(0);
                ptr.left_bf   = direct.left_bf   + field_idx * saver_.left_t.stride(0);
                ptr.right_bf  = direct.right_bf  + field_idx * saver_.right_t.stride(0);
            } else if (direct.dtype == BoundaryDtype::INT8) {
                ptr.top_q       = direct.top_q       + field_idx * saver_.top_t.stride(0);
                ptr.bottom_q    = direct.bottom_q    + field_idx * saver_.bottom_t.stride(0);
                ptr.left_q      = direct.left_q      + field_idx * saver_.left_t.stride(0);
                ptr.right_q     = direct.right_q     + field_idx * saver_.right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_.top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_.bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_.left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_.right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else {
                ptr.top = direct.top + field_idx * saver_.top_t.stride(0);
                ptr.bottom = direct.bottom + field_idx * saver_.bottom_t.stride(0);
                ptr.left = direct.left + field_idx * saver_.left_t.stride(0);
                ptr.right = direct.right + field_idx * saver_.right_t.stride(0);
            }
            return ptr;
        }

        int time_idx = staged_ ? (chunk.ring_start + chunk.buf_idx) : chunk.it;
        int64_t flat_idx = static_cast<int64_t>(time_idx) * saver_.nvar + field_idx;

        GeneralBoundaryPointer ptr{};
        ptr.use_fp16 = direct.use_fp16;
            ptr.dtype = direct.dtype;
        if (!staged_ && direct.dtype == BoundaryDtype::FP16) {
            ptr.top_h    = direct.top_h    + flat_idx * saver_.top_stride;
            ptr.bottom_h = direct.bottom_h + flat_idx * saver_.bottom_stride;
            ptr.front_h  = direct.front_h  + flat_idx * saver_.front_stride;
            ptr.back_h   = direct.back_h   + flat_idx * saver_.back_stride;
            ptr.left_h   = direct.left_h   + flat_idx * saver_.left_stride;
            ptr.right_h  = direct.right_h  + flat_idx * saver_.right_stride;
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_.top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_.bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_.front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_.back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_.left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_.right_scale_t.stride(0);
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (!staged_ && direct.dtype == BoundaryDtype::BF16) {
            ptr.top_bf    = direct.top_bf    + flat_idx * saver_.top_stride;
            ptr.bottom_bf = direct.bottom_bf + flat_idx * saver_.bottom_stride;
            ptr.front_bf  = direct.front_bf  + flat_idx * saver_.front_stride;
            ptr.back_bf   = direct.back_bf   + flat_idx * saver_.back_stride;
            ptr.left_bf   = direct.left_bf   + flat_idx * saver_.left_stride;
            ptr.right_bf  = direct.right_bf  + flat_idx * saver_.right_stride;
        } else if (!staged_ && direct.dtype == BoundaryDtype::INT8) {
            // 3D persistent uint8 buffer: outermost index = time*nvar + field.
            // top_stride/bottom_stride/... already track per-(time×field)
            // cell counts (compute_time_strides → tensor.stride(0)).
            ptr.top_q    = direct.top_q    + flat_idx * saver_.top_stride;
            ptr.bottom_q = direct.bottom_q + flat_idx * saver_.bottom_stride;
            ptr.front_q  = direct.front_q  + flat_idx * saver_.front_stride;
            ptr.back_q   = direct.back_q   + flat_idx * saver_.back_stride;
            ptr.left_q   = direct.left_q   + flat_idx * saver_.left_stride;
            ptr.right_q  = direct.right_q  + flat_idx * saver_.right_stride;
            // Scale buffer outermost stride.
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_.top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_.bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_.front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_.back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_.left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_.right_scale_t.stride(0);
            // Staging shared across (time, field) — no offset.
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (staged_) {
            ptr.dtype = boundary_dtype_from_tensor(saver_.top_gpu);
            if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                // 3D staged INT8: uint8 ring + scale ring indexed by the same
                // flat (gpu_time*nvar + field) index; shared FP32 transient.
                BIND_STAGED_SCALED(ptr.dtype, saver_.top_gpu,    flat_idx * saver_.top_stride,    saver_.top_scale_gpu,    flat_idx * saver_.top_scale_gpu.stride(0),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
                BIND_STAGED_SCALED(ptr.dtype, saver_.bottom_gpu, flat_idx * saver_.bottom_stride, saver_.bottom_scale_gpu, flat_idx * saver_.bottom_scale_gpu.stride(0), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                BIND_STAGED_SCALED(ptr.dtype, saver_.front_gpu,  flat_idx * saver_.front_stride,  saver_.front_scale_gpu,  flat_idx * saver_.front_scale_gpu.stride(0),  direct.front,  ptr.front_q,  ptr.front_h,  ptr.front_scale,  ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_.back_gpu,   flat_idx * saver_.back_stride,   saver_.back_scale_gpu,   flat_idx * saver_.back_scale_gpu.stride(0),   direct.back,   ptr.back_q,   ptr.back_h,   ptr.back_scale,   ptr.back);
                BIND_STAGED_SCALED(ptr.dtype, saver_.left_gpu,   flat_idx * saver_.left_stride,   saver_.left_scale_gpu,   flat_idx * saver_.left_scale_gpu.stride(0),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
                BIND_STAGED_SCALED(ptr.dtype, saver_.right_gpu,  flat_idx * saver_.right_stride,  saver_.right_scale_gpu,  flat_idx * saver_.right_scale_gpu.stride(0),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            } else {
            BIND_STAGED_FACE(saver_.top_gpu,    flat_idx * saver_.top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
            BIND_STAGED_FACE(saver_.bottom_gpu, flat_idx * saver_.bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
            BIND_STAGED_FACE(saver_.front_gpu,  flat_idx * saver_.front_stride,  ptr.front,  ptr.front_h,  ptr.front_bf);
            BIND_STAGED_FACE(saver_.back_gpu,   flat_idx * saver_.back_stride,   ptr.back,   ptr.back_h,   ptr.back_bf);
            BIND_STAGED_FACE(saver_.left_gpu,   flat_idx * saver_.left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
            BIND_STAGED_FACE(saver_.right_gpu,  flat_idx * saver_.right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);
            }
        } else {
            ptr.top = direct.top + flat_idx * saver_.top_stride;
            ptr.bottom = direct.bottom + flat_idx * saver_.bottom_stride;
            ptr.front = direct.front + flat_idx * saver_.front_stride;
            ptr.back = direct.back + flat_idx * saver_.back_stride;
            ptr.left = direct.left + flat_idx * saver_.left_stride;
            ptr.right = direct.right + flat_idx * saver_.right_stride;
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
        if (!enabled_ || !staged_ || !chunk.is_chunk_end)
            return;

        cudaEventRecord(compute_ready_[chunk.ring_slot], compute_stream_);
        cudaStreamWaitEvent(copy_stream_, compute_ready_[chunk.ring_slot], 0);
        if (profile_enabled_)
            cudaEventRecord(forward_copy_start_[chunk.ring_slot], copy_stream_);

        if (boundary_on_disk_) {
            saver_.flush_gpu_to_disk(
                chunk.chunk_start,
                chunk.chunk_len,
                disk_files_,
                copy_stream_,
                chunk.ring_start,
                chunk.ring_start
            );
        } else {
            saver_.flush_gpu_to_cpu(
                chunk.chunk_start,
                chunk.chunk_len,
                copy_stream_,
                chunk.ring_start
            );
        }

        cudaEventRecord(copy_ready_[chunk.ring_slot], copy_stream_);
        copy_pending_[chunk.ring_slot] = 1;
        if (profile_enabled_ && boundary_on_disk_) {
            ++profile_forward_chunks_;
            forward_copy_profile_pending_[chunk.ring_slot] = 1;
        }
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
            boundary_kernel2d_bf16<<<grid, block>>>(
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            // INT8 2-pass save: FP32 kernel → staging (1-step), then
            // quantize → persistent uint8 + per-block FP32 scale.
            boundary_kernel2d<<<grid, block>>>(
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
            int64_t bt = boundary_time_index(chunk);
            quantize_step_2d(b, bt);
        } else {
            boundary_kernel2d<<<grid, block>>>(
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

        if (use_compact_3d_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_compact_bf16<<<compact_blocks, compact_threads>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index(chunk));
            } else {
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_bf16<<<grid, block>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                boundary_kernel3d<<<grid, block>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index(chunk));
            } else {
                boundary_kernel3d<<<grid, block>>>(
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
        bool flush_chunk
    )
    {
        BoundaryChunk chunk = forward_chunk(it, nt);
        wait_before_forward_save(chunk);
        auto b = forward_save_ptrs_field(chunk, direct, field_idx);

        if (b.dtype == BoundaryDtype::BF16) {
            boundary_kernel2d_bf16<<<grid, block>>>(
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            boundary_kernel2d<<<grid, block>>>(
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
            quantize_step_2d(b, boundary_time_index_field(chunk));
        } else {
            boundary_kernel2d<<<grid, block>>>(
                u, b.top, b.bottom, b.left, b.right,
                boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
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

        if (use_compact_3d_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_compact_bf16<<<compact_blocks, compact_threads>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index_field(chunk));
            } else {
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_bf16<<<grid, block>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                boundary_kernel3d<<<grid, block>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_SAVE);
                quantize_step_3d(b, boundary_time_index_field(chunk));
            } else {
                boundary_kernel3d<<<grid, block>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    boundary_time_index_field(chunk), width, offset, ctx, BOUNDARY_SAVE);
            }
        }

        if (flush_chunk)
            flush_forward_if_needed(chunk);
    }

    inline void prefetch_initial_backward_chunk(int nt)
    {
        if (!enabled_ || !staged_)
            return;

        int it0 = nt - 1;
        int buf_idx0 = (it0 - 1) % transfer_interval_;
        int chunk_start = it0 - buf_idx0 - 1;
        int chunk_len = buf_idx0 + 1;
        backward_nt_ = nt;
        backward_active_ = true;
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
            if (next_start >= 0) {
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
        if (!boundary_disk_async_read_)
            gpu_idx = buf_idx;

        GeneralBoundaryPointer ptr{};
        ptr.dtype = boundary_dtype_from_tensor(saver_.top_gpu);
        ptr.last_two = direct.last_two;
        if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
            const int sd = (dim_ == 3) ? 0 : 1;
            BIND_STAGED_SCALED(ptr.dtype, saver_.top_gpu,    gpu_idx * saver_.top_stride,    saver_.top_scale_gpu,    gpu_idx * saver_.top_scale_gpu.stride(sd),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
            BIND_STAGED_SCALED(ptr.dtype, saver_.bottom_gpu, gpu_idx * saver_.bottom_stride, saver_.bottom_scale_gpu, gpu_idx * saver_.bottom_scale_gpu.stride(sd), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
            BIND_STAGED_SCALED(ptr.dtype, saver_.left_gpu,   gpu_idx * saver_.left_stride,   saver_.left_scale_gpu,   gpu_idx * saver_.left_scale_gpu.stride(sd),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
            BIND_STAGED_SCALED(ptr.dtype, saver_.right_gpu,  gpu_idx * saver_.right_stride,  saver_.right_scale_gpu,  gpu_idx * saver_.right_scale_gpu.stride(sd),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            if (dim_ == 3) {
                BIND_STAGED_SCALED(ptr.dtype, saver_.front_gpu, gpu_idx * saver_.front_stride, saver_.front_scale_gpu, gpu_idx * saver_.front_scale_gpu.stride(sd), direct.front, ptr.front_q, ptr.front_h, ptr.front_scale, ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_.back_gpu,  gpu_idx * saver_.back_stride,  saver_.back_scale_gpu,  gpu_idx * saver_.back_scale_gpu.stride(sd),  direct.back,  ptr.back_q,  ptr.back_h,  ptr.back_scale,  ptr.back);
            }
            return ptr;
        }
        BIND_STAGED_FACE(saver_.top_gpu,    gpu_idx * saver_.top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
        BIND_STAGED_FACE(saver_.bottom_gpu, gpu_idx * saver_.bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
        BIND_STAGED_FACE(saver_.left_gpu,   gpu_idx * saver_.left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
        BIND_STAGED_FACE(saver_.right_gpu,  gpu_idx * saver_.right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);

        if (dim_ == 3) {
            BIND_STAGED_FACE(saver_.front_gpu, gpu_idx * saver_.front_stride, ptr.front, ptr.front_h, ptr.front_bf);
            BIND_STAGED_FACE(saver_.back_gpu,  gpu_idx * saver_.back_stride,  ptr.back,  ptr.back_h,  ptr.back_bf);
        }
        return ptr;
    }

    inline GeneralBoundaryPointer backward_restore_ptrs_field(
        int it,
        const GeneralBoundaryPointer& direct,
        int field_idx
    )
    {
        TORCH_CHECK(field_idx >= 0 && field_idx < saver_.nvar, "Invalid boundary field index.");

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
                if (!boundary_disk_async_read_)
                    gpu_idx = buf_idx;

                ptr.dtype = boundary_dtype_from_tensor(saver_.top_gpu);
                if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                    BIND_STAGED_SCALED(ptr.dtype, saver_.top_gpu,
                        field_idx * saver_.top_gpu.stride(0) + gpu_idx * saver_.top_gpu.stride(1),
                        saver_.top_scale_gpu,
                        field_idx * saver_.top_scale_gpu.stride(0) + gpu_idx * saver_.top_scale_gpu.stride(1),
                        direct.top, ptr.top_q, ptr.top_h, ptr.top_scale, ptr.top);
                    BIND_STAGED_SCALED(ptr.dtype, saver_.bottom_gpu,
                        field_idx * saver_.bottom_gpu.stride(0) + gpu_idx * saver_.bottom_gpu.stride(1),
                        saver_.bottom_scale_gpu,
                        field_idx * saver_.bottom_scale_gpu.stride(0) + gpu_idx * saver_.bottom_scale_gpu.stride(1),
                        direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                    BIND_STAGED_SCALED(ptr.dtype, saver_.left_gpu,
                        field_idx * saver_.left_gpu.stride(0) + gpu_idx * saver_.left_gpu.stride(1),
                        saver_.left_scale_gpu,
                        field_idx * saver_.left_scale_gpu.stride(0) + gpu_idx * saver_.left_scale_gpu.stride(1),
                        direct.left, ptr.left_q, ptr.left_h, ptr.left_scale, ptr.left);
                    BIND_STAGED_SCALED(ptr.dtype, saver_.right_gpu,
                        field_idx * saver_.right_gpu.stride(0) + gpu_idx * saver_.right_gpu.stride(1),
                        saver_.right_scale_gpu,
                        field_idx * saver_.right_scale_gpu.stride(0) + gpu_idx * saver_.right_scale_gpu.stride(1),
                        direct.right, ptr.right_q, ptr.right_h, ptr.right_scale, ptr.right);
                } else {
                BIND_STAGED_FACE(saver_.top_gpu,
                    field_idx * saver_.top_gpu.stride(0) + gpu_idx * saver_.top_gpu.stride(1),
                    ptr.top, ptr.top_h, ptr.top_bf);
                BIND_STAGED_FACE(saver_.bottom_gpu,
                    field_idx * saver_.bottom_gpu.stride(0) + gpu_idx * saver_.bottom_gpu.stride(1),
                    ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
                BIND_STAGED_FACE(saver_.left_gpu,
                    field_idx * saver_.left_gpu.stride(0) + gpu_idx * saver_.left_gpu.stride(1),
                    ptr.left, ptr.left_h, ptr.left_bf);
                BIND_STAGED_FACE(saver_.right_gpu,
                    field_idx * saver_.right_gpu.stride(0) + gpu_idx * saver_.right_gpu.stride(1),
                    ptr.right, ptr.right_h, ptr.right_bf);
                }
            } else if (direct.dtype == BoundaryDtype::FP16) {
                ptr.top_h    = direct.top_h    + field_idx * saver_.top_t.stride(0);
                ptr.bottom_h = direct.bottom_h + field_idx * saver_.bottom_t.stride(0);
                ptr.left_h   = direct.left_h   + field_idx * saver_.left_t.stride(0);
                ptr.right_h  = direct.right_h  + field_idx * saver_.right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_.top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_.bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_.left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_.right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else if (direct.dtype == BoundaryDtype::BF16) {
                ptr.top_bf    = direct.top_bf    + field_idx * saver_.top_t.stride(0);
                ptr.bottom_bf = direct.bottom_bf + field_idx * saver_.bottom_t.stride(0);
                ptr.left_bf   = direct.left_bf   + field_idx * saver_.left_t.stride(0);
                ptr.right_bf  = direct.right_bf  + field_idx * saver_.right_t.stride(0);
            } else if (direct.dtype == BoundaryDtype::INT8) {
                ptr.top_q       = direct.top_q       + field_idx * saver_.top_t.stride(0);
                ptr.bottom_q    = direct.bottom_q    + field_idx * saver_.bottom_t.stride(0);
                ptr.left_q      = direct.left_q      + field_idx * saver_.left_t.stride(0);
                ptr.right_q     = direct.right_q     + field_idx * saver_.right_t.stride(0);
                ptr.top_scale    = direct.top_scale    + field_idx * saver_.top_scale_t.stride(0);
                ptr.bottom_scale = direct.bottom_scale + field_idx * saver_.bottom_scale_t.stride(0);
                ptr.left_scale   = direct.left_scale   + field_idx * saver_.left_scale_t.stride(0);
                ptr.right_scale  = direct.right_scale  + field_idx * saver_.right_scale_t.stride(0);
                ptr.top    = direct.top;
                ptr.bottom = direct.bottom;
                ptr.left   = direct.left;
                ptr.right  = direct.right;
            } else {
                ptr.top = direct.top + field_idx * saver_.top_t.stride(0);
                ptr.bottom = direct.bottom + field_idx * saver_.bottom_t.stride(0);
                ptr.left = direct.left + field_idx * saver_.left_t.stride(0);
                ptr.right = direct.right + field_idx * saver_.right_t.stride(0);
            }
            return ptr;
        }

        int time_idx = it - 1;
        if (staged_) {
            int buf_idx = (it - 1) % transfer_interval_;
            int chunk_start = it - 1 - buf_idx;
            int slot_start = backward_slot_for_chunk(chunk_start) * transfer_interval_;
            time_idx = slot_start + buf_idx;
            if (!boundary_disk_async_read_)
                time_idx = buf_idx;
        }
        int64_t flat_idx = static_cast<int64_t>(time_idx) * saver_.nvar + field_idx;

        GeneralBoundaryPointer ptr{};
        ptr.use_fp16 = direct.use_fp16;
            ptr.dtype = direct.dtype;
        if (!staged_ && direct.dtype == BoundaryDtype::FP16) {
            ptr.top_h    = direct.top_h    + flat_idx * saver_.top_stride;
            ptr.bottom_h = direct.bottom_h + flat_idx * saver_.bottom_stride;
            ptr.front_h  = direct.front_h  + flat_idx * saver_.front_stride;
            ptr.back_h   = direct.back_h   + flat_idx * saver_.back_stride;
            ptr.left_h   = direct.left_h   + flat_idx * saver_.left_stride;
            ptr.right_h  = direct.right_h  + flat_idx * saver_.right_stride;
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_.top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_.bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_.front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_.back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_.left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_.right_scale_t.stride(0);
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (!staged_ && direct.dtype == BoundaryDtype::BF16) {
            ptr.top_bf    = direct.top_bf    + flat_idx * saver_.top_stride;
            ptr.bottom_bf = direct.bottom_bf + flat_idx * saver_.bottom_stride;
            ptr.front_bf  = direct.front_bf  + flat_idx * saver_.front_stride;
            ptr.back_bf   = direct.back_bf   + flat_idx * saver_.back_stride;
            ptr.left_bf   = direct.left_bf   + flat_idx * saver_.left_stride;
            ptr.right_bf  = direct.right_bf  + flat_idx * saver_.right_stride;
        } else if (!staged_ && direct.dtype == BoundaryDtype::INT8) {
            // 3D persistent uint8 buffer: outermost index = time*nvar + field.
            // top_stride/bottom_stride/... already track per-(time×field)
            // cell counts (compute_time_strides → tensor.stride(0)).
            ptr.top_q    = direct.top_q    + flat_idx * saver_.top_stride;
            ptr.bottom_q = direct.bottom_q + flat_idx * saver_.bottom_stride;
            ptr.front_q  = direct.front_q  + flat_idx * saver_.front_stride;
            ptr.back_q   = direct.back_q   + flat_idx * saver_.back_stride;
            ptr.left_q   = direct.left_q   + flat_idx * saver_.left_stride;
            ptr.right_q  = direct.right_q  + flat_idx * saver_.right_stride;
            // Scale buffer outermost stride.
            ptr.top_scale    = direct.top_scale    + flat_idx * saver_.top_scale_t.stride(0);
            ptr.bottom_scale = direct.bottom_scale + flat_idx * saver_.bottom_scale_t.stride(0);
            ptr.front_scale  = direct.front_scale  + flat_idx * saver_.front_scale_t.stride(0);
            ptr.back_scale   = direct.back_scale   + flat_idx * saver_.back_scale_t.stride(0);
            ptr.left_scale   = direct.left_scale   + flat_idx * saver_.left_scale_t.stride(0);
            ptr.right_scale  = direct.right_scale  + flat_idx * saver_.right_scale_t.stride(0);
            // Staging shared across (time, field) — no offset.
            ptr.top    = direct.top;
            ptr.bottom = direct.bottom;
            ptr.front  = direct.front;
            ptr.back   = direct.back;
            ptr.left   = direct.left;
            ptr.right  = direct.right;
        } else if (staged_) {
            ptr.dtype = boundary_dtype_from_tensor(saver_.top_gpu);
            if (ptr.dtype == BoundaryDtype::INT8 || ptr.dtype == BoundaryDtype::FP16) {
                BIND_STAGED_SCALED(ptr.dtype, saver_.top_gpu,    flat_idx * saver_.top_stride,    saver_.top_scale_gpu,    flat_idx * saver_.top_scale_gpu.stride(0),    direct.top,    ptr.top_q,    ptr.top_h,    ptr.top_scale,    ptr.top);
                BIND_STAGED_SCALED(ptr.dtype, saver_.bottom_gpu, flat_idx * saver_.bottom_stride, saver_.bottom_scale_gpu, flat_idx * saver_.bottom_scale_gpu.stride(0), direct.bottom, ptr.bottom_q, ptr.bottom_h, ptr.bottom_scale, ptr.bottom);
                BIND_STAGED_SCALED(ptr.dtype, saver_.front_gpu,  flat_idx * saver_.front_stride,  saver_.front_scale_gpu,  flat_idx * saver_.front_scale_gpu.stride(0),  direct.front,  ptr.front_q,  ptr.front_h,  ptr.front_scale,  ptr.front);
                BIND_STAGED_SCALED(ptr.dtype, saver_.back_gpu,   flat_idx * saver_.back_stride,   saver_.back_scale_gpu,   flat_idx * saver_.back_scale_gpu.stride(0),   direct.back,   ptr.back_q,   ptr.back_h,   ptr.back_scale,   ptr.back);
                BIND_STAGED_SCALED(ptr.dtype, saver_.left_gpu,   flat_idx * saver_.left_stride,   saver_.left_scale_gpu,   flat_idx * saver_.left_scale_gpu.stride(0),   direct.left,   ptr.left_q,   ptr.left_h,   ptr.left_scale,   ptr.left);
                BIND_STAGED_SCALED(ptr.dtype, saver_.right_gpu,  flat_idx * saver_.right_stride,  saver_.right_scale_gpu,  flat_idx * saver_.right_scale_gpu.stride(0),  direct.right,  ptr.right_q,  ptr.right_h,  ptr.right_scale,  ptr.right);
            } else {
            BIND_STAGED_FACE(saver_.top_gpu,    flat_idx * saver_.top_stride,    ptr.top,    ptr.top_h,    ptr.top_bf);
            BIND_STAGED_FACE(saver_.bottom_gpu, flat_idx * saver_.bottom_stride, ptr.bottom, ptr.bottom_h, ptr.bottom_bf);
            BIND_STAGED_FACE(saver_.front_gpu,  flat_idx * saver_.front_stride,  ptr.front,  ptr.front_h,  ptr.front_bf);
            BIND_STAGED_FACE(saver_.back_gpu,   flat_idx * saver_.back_stride,   ptr.back,   ptr.back_h,   ptr.back_bf);
            BIND_STAGED_FACE(saver_.left_gpu,   flat_idx * saver_.left_stride,   ptr.left,   ptr.left_h,   ptr.left_bf);
            BIND_STAGED_FACE(saver_.right_gpu,  flat_idx * saver_.right_stride,  ptr.right,  ptr.right_h,  ptr.right_bf);
            }
        } else {
            ptr.top = direct.top + flat_idx * saver_.top_stride;
            ptr.bottom = direct.bottom + flat_idx * saver_.bottom_stride;
            ptr.front = direct.front + flat_idx * saver_.front_stride;
            ptr.back = direct.back + flat_idx * saver_.back_stride;
            ptr.left = direct.left + flat_idx * saver_.left_stride;
            ptr.right = direct.right + flat_idx * saver_.right_stride;
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
            boundary_kernel2d_bf16<<<grid, block>>>(
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            dequantize_step_2d(b, backward_time_index(it));
            boundary_kernel2d<<<grid, block>>>(
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
        } else {
            boundary_kernel2d<<<grid, block>>>(
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
        if (use_compact_3d_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_compact_bf16<<<compact_blocks, compact_threads>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                dequantize_step_3d(b, backward_time_index(it));
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_bf16<<<grid, block>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                dequantize_step_3d(b, backward_time_index(it));
                boundary_kernel3d<<<grid, block>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                boundary_kernel3d<<<grid, block>>>(
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
        bool record_done
    )
    {
        if (wait_chunk)
            wait_before_backward_restore(it);
        auto b = backward_restore_ptrs_field(it, direct, field_idx);
        if (b.dtype == BoundaryDtype::BF16) {
            boundary_kernel2d_bf16<<<grid, block>>>(
                u, b.top_bf, b.bottom_bf, b.left_bf, b.right_bf,
                backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
        } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
            dequantize_step_2d(b, backward_time_index_field(it));
            boundary_kernel2d<<<grid, block>>>(
                u, b.top, b.bottom, b.left, b.right,
                /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
        } else {
            boundary_kernel2d<<<grid, block>>>(
                u, b.top, b.bottom, b.left, b.right,
                backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
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
        if (use_compact_3d_boundary_kernel()) {
            int compact_total = compact_boundary_count_3d(ctx, width, 0);
            int compact_threads = 512;
            int compact_blocks = (compact_total + compact_threads - 1) / compact_threads;
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_compact_bf16<<<compact_blocks, compact_threads>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                dequantize_step_3d(b, backward_time_index_field(it));
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                boundary_kernel3d_compact<<<compact_blocks, compact_threads>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
            }
        } else {
            if (b.dtype == BoundaryDtype::BF16) {
                boundary_kernel3d_bf16<<<grid, block>>>(
                    u, b.top_bf, b.bottom_bf, b.front_bf, b.back_bf, b.left_bf, b.right_bf,
                    backward_time_index_field(it), width, offset, ctx, BOUNDARY_RESTORE);
            } else if (b.dtype == BoundaryDtype::INT8 || b.dtype == BoundaryDtype::FP16) {
                dequantize_step_3d(b, backward_time_index_field(it));
                boundary_kernel3d<<<grid, block>>>(
                    u, b.top, b.bottom, b.front, b.back, b.left, b.right,
                    /*it=*/0, width, offset, ctx, BOUNDARY_RESTORE);
            } else {
                boundary_kernel3d<<<grid, block>>>(
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
        if (buf_idx != 0)
            return;

        int chunk_id = (it - 1) / transfer_interval_;
        int next_chunk = chunk_id - 1;
        if (next_chunk < 0)
            return;

        int next_start = next_chunk * transfer_interval_;
        int remain = nt - next_start;
        int next_len = remain < transfer_interval_ ? remain : transfer_interval_;
        prefetch_backward_chunk(next_start, next_len, 0);
    }

    inline void synchronize()
    {
        wait_for_pending_disk_reader_task();
        if (enabled_ && staged_)
            cudaStreamSynchronize(copy_stream_);
        if (boundary_on_disk_)
            wait_for_boundary_disk_writes();
        accumulate_all_forward_copy_time();
        print_profile_if_needed();
    }

private:
    using Clock = std::chrono::steady_clock;

    EffectiveBoundarySaver& saver_;
    int dim_ = 2;
    bool enabled_ = false;
    bool staged_ = false;
    bool boundary_on_disk_ = false;
    bool boundary_disk_async_read_ = false;
    int transfer_interval_ = 1;
    int ring_buffers_ = 1;
    int backward_nt_ = 0;
    const std::vector<std::string>& disk_files_;
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t copy_stream_ = nullptr;
    bool profile_enabled_ = false;
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

    inline bool use_compact_3d_boundary_kernel() const
    {
        return !boundary_on_disk_ || boundary_disk_async_read_;
    }

    inline int backward_slot_for_chunk(int chunk_start) const
    {
        if (!boundary_disk_async_read_)
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
                // staging buffer (boundary_cpu / saver_.*_t).  The previous
                // chunk's H2D copy reads that same buffer, so we must wait for
                // it to finish before clobbering it — otherwise the host races
                // ahead of the (possibly backlogged) copy stream and the
                // in-flight H2D picks up this chunk's freshly-read bytes,
                // scrambling the restored boundary for nvar>1.  (ring_buffers
                // is 1 on this path, so the staging buffer is reused verbatim.)
                cudaStreamSynchronize(copy_stream_);
                auto read_start = Clock::now();
                if (dim_ == 2)
                    saver_.load_disk_to_cpu_2d(start, len, disk_files_);
                else
                    saver_.load_disk_to_cpu_3d(start, len, disk_files_);
                add_disk_read_time(elapsed_seconds(read_start, Clock::now()));
                if (profile_enabled_ && boundary_on_disk_) {
                    cudaEventRecord(backward_copy_start_[slot], copy_stream_);
                    cudaEventRecord(backward_h2d_start_[slot], copy_stream_);
                }
                auto enqueue_start = Clock::now();
                saver_.load_cpu_to_gpu(0, len, copy_stream_);
                add_backward_h2d_enqueue_time(elapsed_seconds(enqueue_start, Clock::now()));
                cudaEventRecord(copy_ready_[slot], copy_stream_);
                if (profile_enabled_ && boundary_on_disk_)
                    backward_copy_profile_pending_[slot] = 1;
            }
        } else {
            saver_.load_cpu_to_gpu(start, len, copy_stream_);
            cudaEventRecord(copy_ready_[slot], copy_stream_);
        }
    }

    inline void record_backward_restore_done(int it)
    {
        if (!boundary_disk_async_read_)
            return;
        int buf_idx = (it - 1) % transfer_interval_;
        int chunk_start = it - 1 - buf_idx;
        int slot = backward_slot_for_chunk(chunk_start);
        cudaEventRecord(compute_ready_[slot], compute_stream_);
    }

    inline void start_async_disk_read(int start, int len, int slot)
    {
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
        auto read_start = Clock::now();
        if (dim_ == 2)
            saver_.load_disk_to_cpu_2d(start, len, disk_files_, stage_start);
        else
            saver_.load_disk_to_cpu_3d(start, len, disk_files_, stage_start);
        add_disk_read_time(elapsed_seconds(read_start, Clock::now()));

        if (profile_enabled_ && boundary_on_disk_)
            cudaEventRecord(backward_copy_start_[slot], copy_stream_);
        cudaStreamWaitEvent(copy_stream_, compute_ready_[slot], 0);
        if (profile_enabled_ && boundary_on_disk_)
            cudaEventRecord(backward_h2d_start_[slot], copy_stream_);
        auto enqueue_start = Clock::now();
        saver_.load_cpu_to_gpu(stage_start, len, copy_stream_, stage_start);
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
        if (!profile_enabled_ || profile_printed_ || !boundary_on_disk_)
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
};
