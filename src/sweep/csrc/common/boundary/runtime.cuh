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
        ptr.top = saver_.top_gpu.data_ptr<float>() + gpu_idx * saver_.top_stride;
        ptr.bottom = saver_.bottom_gpu.data_ptr<float>() + gpu_idx * saver_.bottom_stride;
        ptr.left = saver_.left_gpu.data_ptr<float>() + gpu_idx * saver_.left_stride;
        ptr.right = saver_.right_gpu.data_ptr<float>() + gpu_idx * saver_.right_stride;
        ptr.last_two = direct.last_two;

        if (dim_ == 3) {
            ptr.front = saver_.front_gpu.data_ptr<float>() + gpu_idx * saver_.front_stride;
            ptr.back = saver_.back_gpu.data_ptr<float>() + gpu_idx * saver_.back_stride;
        }
        return ptr;
    }

    inline int boundary_time_index(const BoundaryChunk& chunk) const
    {
        return staged_ ? 0 : chunk.it;
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

        boundary_kernel2d<<<grid, block>>>(
            u,
            b.top,
            b.bottom,
            b.left,
            b.right,
            boundary_time_index(chunk),
            width,
            offset,
            ctx,
            BOUNDARY_SAVE
        );

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

        boundary_kernel3d<<<grid, block>>>(
            u,
            b.top,
            b.bottom,
            b.front,
            b.back,
            b.left,
            b.right,
            boundary_time_index(chunk),
            width,
            offset,
            ctx,
            BOUNDARY_SAVE
        );

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
        ptr.top = saver_.top_gpu.data_ptr<float>() + gpu_idx * saver_.top_stride;
        ptr.bottom = saver_.bottom_gpu.data_ptr<float>() + gpu_idx * saver_.bottom_stride;
        ptr.left = saver_.left_gpu.data_ptr<float>() + gpu_idx * saver_.left_stride;
        ptr.right = saver_.right_gpu.data_ptr<float>() + gpu_idx * saver_.right_stride;
        ptr.last_two = direct.last_two;

        if (dim_ == 3) {
            ptr.front = saver_.front_gpu.data_ptr<float>() + gpu_idx * saver_.front_stride;
            ptr.back = saver_.back_gpu.data_ptr<float>() + gpu_idx * saver_.back_stride;
        }
        return ptr;
    }

    inline int backward_time_index(int it) const
    {
        return staged_ ? 0 : it - 1;
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
        boundary_kernel2d<<<grid, block>>>(
            u,
            b.top,
            b.bottom,
            b.left,
            b.right,
            backward_time_index(it),
            width,
            offset,
            ctx,
            BOUNDARY_RESTORE
        );
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
        boundary_kernel3d<<<grid, block>>>(
            u,
            b.top,
            b.bottom,
            b.front,
            b.back,
            b.left,
            b.right,
            backward_time_index(it),
            width,
            offset,
            ctx,
            BOUNDARY_RESTORE
        );
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
