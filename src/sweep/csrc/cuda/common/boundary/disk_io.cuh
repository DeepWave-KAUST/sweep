#pragma once

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

struct BoundaryDisk2DMeta {
    std::array<std::string, 4> paths;
    size_t top_elems = 0;
    size_t left_elems = 0;
    size_t start_top_offset = 0;
    size_t start_left_offset = 0;
    size_t top_var_block = 0;
    size_t left_var_block = 0;
    size_t top_file_var_block = 0;
    size_t left_file_var_block = 0;
    int nvar = 1;
    // Byte size of one stored boundary element (4 = fp32, 2 = fp16/bf16).  The
    // staging-buffer pointers below are raw byte pointers so the disk path can
    // copy/seek without knowing the concrete scalar type; all *_elems / *_block
    // / *_offset counts stay in ELEMENTS and are multiplied by elem_size at use.
    size_t elem_size = sizeof(float);
    const char* top = nullptr;
    const char* bottom = nullptr;
    const char* left = nullptr;
    const char* right = nullptr;
};

struct BoundaryDisk3DMeta {
    std::array<std::string, 6> paths;
    size_t top_elems = 0;
    size_t front_elems = 0;
    size_t left_elems = 0;
    size_t top_offset = 0;
    size_t front_offset = 0;
    size_t left_offset = 0;
    size_t elem_size = sizeof(float);
    const char* top = nullptr;
    const char* bottom = nullptr;
    const char* front = nullptr;
    const char* back = nullptr;
    const char* left = nullptr;
    const char* right = nullptr;
};

struct BoundaryDisk2DWriteTask {
    std::array<std::string, 4> paths;
    size_t top_elems = 0;
    size_t left_elems = 0;
    size_t start_top_offset = 0;
    size_t start_left_offset = 0;
    size_t top_file_var_block = 0;
    size_t left_file_var_block = 0;
    int nvar = 1;
    // Byte staging buffers (size = element_count * elem_size).
    size_t elem_size = sizeof(float);
    std::vector<char> top;
    std::vector<char> bottom;
    std::vector<char> left;
    std::vector<char> right;
};

struct BoundaryDisk3DWriteTask {
    std::array<std::string, 6> paths;
    size_t top_elems = 0;
    size_t front_elems = 0;
    size_t left_elems = 0;
    size_t top_offset = 0;
    size_t front_offset = 0;
    size_t left_offset = 0;
    size_t elem_size = sizeof(float);
    std::vector<char> top;
    std::vector<char> bottom;
    std::vector<char> front;
    std::vector<char> back;
    std::vector<char> left;
    std::vector<char> right;
};

inline std::mutex& boundary_disk_writer_mutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::condition_variable& boundary_disk_writer_cv()
{
    static std::condition_variable cv;
    return cv;
}

inline int& boundary_disk_writer_pending()
{
    static int pending = 0;
    return pending;
}

inline std::exception_ptr& boundary_disk_writer_error()
{
    static std::exception_ptr error;
    return error;
}

// Neither writer can report a failure by propagating it: the write threads are
// DETACHED, and the enqueue runs inside a cudaLaunchHostFunc callback, which
// terminates the process if an exception escapes it. So a failure is recorded
// here and raised by the barrier instead. Without that, a failed write still
// decremented the pending counter, wait_for_boundary_disk_writes() returned as
// if the data had landed, and the backward reconstructed from a boundary file
// that _allocate_boundary_disk_files created SPARSE -- i.e. from zeros for
// anything never written -- for a silently wrong gradient and no exception.
// Measured on a read-only boundary file: every write failed, nothing raised,
// and the gradient came back 1.46e-02 off (cosine 0.9935) against the same
// problem on gpu-direct storage.
//
// The FIRST failure wins: later ones are almost always the same cause repeated
// once per saved step, and the first is the one with the useful context.
//
// The slot is process-global, like the pending counter it sits beside, so two
// propagators writing boundaries concurrently in one process share it -- one's
// failure surfaces at the other's barrier. That is already true of the counter;
// this does not widen it.
inline void boundary_disk_writer_record_error(std::exception_ptr error)
{
    std::lock_guard<std::mutex> lock(boundary_disk_writer_mutex());
    if (!boundary_disk_writer_error())
        boundary_disk_writer_error() = error;
}

// Drop a failure nobody observed. The slot lives for the whole process (these
// are free functions), unlike the read side's disk_reader_exception_, which is
// a BoundaryRuntime member and dies with its runtime -- so without this, a run
// that unwound for some OTHER reason before reaching its barrier would hand its
// write error to the next, unrelated run. Never touches a live batch: a nonzero
// pending count means writer threads are still running and may be about to
// record. It must NOT be done in boundary_disk_writer_begin() or on bind():
// writes are launched per chunk, so the counter legitimately passes through
// zero between chunks and a reset there would erase a real failure.
inline void reset_boundary_disk_write_error()
{
    std::lock_guard<std::mutex> lock(boundary_disk_writer_mutex());
    if (boundary_disk_writer_pending() == 0)
        boundary_disk_writer_error() = nullptr;
}

inline void boundary_disk_writer_begin()
{
    std::lock_guard<std::mutex> lock(boundary_disk_writer_mutex());
    ++boundary_disk_writer_pending();
}

inline void boundary_disk_writer_done()
{
    {
        std::lock_guard<std::mutex> lock(boundary_disk_writer_mutex());
        --boundary_disk_writer_pending();
    }
    boundary_disk_writer_cv().notify_all();
}

inline void wait_for_boundary_disk_writes()
{
    std::exception_ptr error;
    {
        std::unique_lock<std::mutex> lock(boundary_disk_writer_mutex());
        boundary_disk_writer_cv().wait(lock, []() {
            return boundary_disk_writer_pending() == 0;
        });
        // Taken, not copied: reporting a failure clears it, so it raises once
        // and the next run starts from a clean slot rather than inheriting a
        // stale error from a previous one.
        error = boundary_disk_writer_error();
        boundary_disk_writer_error() = nullptr;
    }
    if (error)
        std::rethrow_exception(error);
}

inline void write_boundary_file_chunk(const std::string& path, size_t offset_elems, const void* data, size_t elems, size_t elem_size)
{
    std::ofstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out)
        throw std::runtime_error("Failed to open boundary disk file for writing: " + path);
    out.seekp(static_cast<std::streamoff>(offset_elems * elem_size));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(elems * elem_size));
    if (!out)
        throw std::runtime_error("Failed to write boundary disk file: " + path);
}

inline void read_boundary_file_chunk(const std::string& path, size_t offset_elems, void* data, size_t elems, size_t elem_size)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to open boundary disk file for reading: " + path);
    in.seekg(static_cast<std::streamoff>(offset_elems * elem_size));
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(elems * elem_size));
    if (!in)
        throw std::runtime_error("Failed to read boundary disk file: " + path);
}

inline void write_boundary_disk_2d_task(const BoundaryDisk2DWriteTask& task)
{
    const size_t es = task.elem_size;
    for (int v = 0; v < task.nvar; ++v) {
        write_boundary_file_chunk(
            task.paths[0],
            static_cast<size_t>(v) * task.top_file_var_block + task.start_top_offset,
            task.top.data() + static_cast<size_t>(v) * task.top_elems * es,
            task.top_elems,
            es
        );
        write_boundary_file_chunk(
            task.paths[1],
            static_cast<size_t>(v) * task.top_file_var_block + task.start_top_offset,
            task.bottom.data() + static_cast<size_t>(v) * task.top_elems * es,
            task.top_elems,
            es
        );
        write_boundary_file_chunk(
            task.paths[2],
            static_cast<size_t>(v) * task.left_file_var_block + task.start_left_offset,
            task.left.data() + static_cast<size_t>(v) * task.left_elems * es,
            task.left_elems,
            es
        );
        write_boundary_file_chunk(
            task.paths[3],
            static_cast<size_t>(v) * task.left_file_var_block + task.start_left_offset,
            task.right.data() + static_cast<size_t>(v) * task.left_elems * es,
            task.left_elems,
            es
        );
    }
}

inline void write_boundary_disk_3d_task(const BoundaryDisk3DWriteTask& task)
{
    constexpr size_t kParallelWriteThresholdBytes = 8 * 1024 * 1024;
    const size_t es = task.elem_size;
    size_t write_bytes =
        2 * (task.top_elems + task.front_elems + task.left_elems) * es;
    if (write_bytes < kParallelWriteThresholdBytes) {
        write_boundary_file_chunk(task.paths[0], task.top_offset, task.top.data(), task.top_elems, es);
        write_boundary_file_chunk(task.paths[1], task.top_offset, task.bottom.data(), task.top_elems, es);
        write_boundary_file_chunk(task.paths[2], task.front_offset, task.front.data(), task.front_elems, es);
        write_boundary_file_chunk(task.paths[3], task.front_offset, task.back.data(), task.front_elems, es);
        write_boundary_file_chunk(task.paths[4], task.left_offset, task.left.data(), task.left_elems, es);
        write_boundary_file_chunk(task.paths[5], task.left_offset, task.right.data(), task.left_elems, es);
        return;
    }

    std::exception_ptr write_error = nullptr;
    std::mutex write_error_mutex;
    auto write_one = [&](std::string path, size_t offset, const void* data, size_t elems) {
        try {
            write_boundary_file_chunk(path, offset, data, elems, es);
        } catch (...) {
            std::lock_guard<std::mutex> lock(write_error_mutex);
            if (!write_error)
                write_error = std::current_exception();
        }
    };

    std::vector<std::thread> writers;
    writers.reserve(6);
    writers.emplace_back(write_one, task.paths[0], task.top_offset, task.top.data(), task.top_elems);
    writers.emplace_back(write_one, task.paths[1], task.top_offset, task.bottom.data(), task.top_elems);
    writers.emplace_back(write_one, task.paths[2], task.front_offset, task.front.data(), task.front_elems);
    writers.emplace_back(write_one, task.paths[3], task.front_offset, task.back.data(), task.front_elems);
    writers.emplace_back(write_one, task.paths[4], task.left_offset, task.left.data(), task.left_elems);
    writers.emplace_back(write_one, task.paths[5], task.left_offset, task.right.data(), task.left_elems);
    for (auto& writer : writers)
        writer.join();
    if (write_error)
        std::rethrow_exception(write_error);
}

inline void launch_boundary_disk_write_2d(std::shared_ptr<BoundaryDisk2DWriteTask> task)
{
    boundary_disk_writer_begin();
    try {
        std::thread([task]() {
            try {
                write_boundary_disk_2d_task(*task);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "Boundary disk async write failed: %s\n", e.what());
                boundary_disk_writer_record_error(std::current_exception());
            }
            boundary_disk_writer_done();
        }).detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk async write launch failed: %s\n", e.what());
        try {
            write_boundary_disk_2d_task(*task);
        } catch (const std::exception& write_error) {
            std::fprintf(stderr, "Boundary disk fallback write failed: %s\n", write_error.what());
            // Only the FALLBACK failing loses data: a thread that could not be
            // launched but whose write then succeeded inline has written it.
            boundary_disk_writer_record_error(std::current_exception());
        }
        boundary_disk_writer_done();
    }
}

inline void launch_boundary_disk_write_3d(std::shared_ptr<BoundaryDisk3DWriteTask> task)
{
    boundary_disk_writer_begin();
    try {
        std::thread([task]() {
            try {
                write_boundary_disk_3d_task(*task);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "Boundary disk async write failed: %s\n", e.what());
                boundary_disk_writer_record_error(std::current_exception());
            }
            boundary_disk_writer_done();
        }).detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk async write launch failed: %s\n", e.what());
        try {
            write_boundary_disk_3d_task(*task);
        } catch (const std::exception& write_error) {
            std::fprintf(stderr, "Boundary disk fallback write failed: %s\n", write_error.what());
            // Only the FALLBACK failing loses data: a thread that could not be
            // launched but whose write then succeeded inline has written it.
            boundary_disk_writer_record_error(std::current_exception());
        }
        boundary_disk_writer_done();
    }
}

inline void CUDART_CB write_boundary_disk_2d_callback(void* user_data)
{
    std::unique_ptr<BoundaryDisk2DMeta> meta(static_cast<BoundaryDisk2DMeta*>(user_data));
    try {
        auto task = std::make_shared<BoundaryDisk2DWriteTask>();
        task->paths = meta->paths;
        task->top_elems = meta->top_elems;
        task->left_elems = meta->left_elems;
        task->start_top_offset = meta->start_top_offset;
        task->start_left_offset = meta->start_left_offset;
        task->top_file_var_block = meta->top_file_var_block;
        task->left_file_var_block = meta->left_file_var_block;
        task->nvar = meta->nvar;
        const size_t es = meta->elem_size;
        task->elem_size = es;
        task->top.resize(static_cast<size_t>(meta->nvar) * meta->top_elems * es);
        task->bottom.resize(static_cast<size_t>(meta->nvar) * meta->top_elems * es);
        task->left.resize(static_cast<size_t>(meta->nvar) * meta->left_elems * es);
        task->right.resize(static_cast<size_t>(meta->nvar) * meta->left_elems * es);
        for (int v = 0; v < meta->nvar; ++v) {
            std::memcpy(
                task->top.data() + static_cast<size_t>(v) * meta->top_elems * es,
                meta->top + static_cast<size_t>(v) * meta->top_var_block * es,
                meta->top_elems * es
            );
            std::memcpy(
                task->bottom.data() + static_cast<size_t>(v) * meta->top_elems * es,
                meta->bottom + static_cast<size_t>(v) * meta->top_var_block * es,
                meta->top_elems * es
            );
            std::memcpy(
                task->left.data() + static_cast<size_t>(v) * meta->left_elems * es,
                meta->left + static_cast<size_t>(v) * meta->left_var_block * es,
                meta->left_elems * es
            );
            std::memcpy(
                task->right.data() + static_cast<size_t>(v) * meta->left_elems * es,
                meta->right + static_cast<size_t>(v) * meta->left_var_block * es,
                meta->left_elems * es
            );
        }
        launch_boundary_disk_write_2d(task);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk write callback failed: %s\n", e.what());
        // Thrown before launch_boundary_disk_write_* ran, so nothing incremented
        // the pending counter and the whole chunk is simply gone -- the barrier
        // would otherwise return immediately and call that success.
        boundary_disk_writer_record_error(std::current_exception());
    }
}

inline void CUDART_CB write_boundary_disk_3d_callback(void* user_data)
{
    std::unique_ptr<BoundaryDisk3DMeta> meta(static_cast<BoundaryDisk3DMeta*>(user_data));
    try {
        auto task = std::make_shared<BoundaryDisk3DWriteTask>();
        task->paths = meta->paths;
        task->top_elems = meta->top_elems;
        task->front_elems = meta->front_elems;
        task->left_elems = meta->left_elems;
        task->top_offset = meta->top_offset;
        task->front_offset = meta->front_offset;
        task->left_offset = meta->left_offset;
        const size_t es = meta->elem_size;
        task->elem_size = es;
        task->top.assign(meta->top, meta->top + meta->top_elems * es);
        task->bottom.assign(meta->bottom, meta->bottom + meta->top_elems * es);
        task->front.assign(meta->front, meta->front + meta->front_elems * es);
        task->back.assign(meta->back, meta->back + meta->front_elems * es);
        task->left.assign(meta->left, meta->left + meta->left_elems * es);
        task->right.assign(meta->right, meta->right + meta->left_elems * es);
        launch_boundary_disk_write_3d(task);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk write callback failed: %s\n", e.what());
        // Thrown before launch_boundary_disk_write_* ran, so nothing incremented
        // the pending counter and the whole chunk is simply gone -- the barrier
        // would otherwise return immediately and call that success.
        boundary_disk_writer_record_error(std::current_exception());
    }
}
