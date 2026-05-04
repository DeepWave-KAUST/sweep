#pragma once

#include <array>
#include <condition_variable>
#include <cstdio>
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
    float* top = nullptr;
    float* bottom = nullptr;
    float* left = nullptr;
    float* right = nullptr;
};

struct BoundaryDisk3DMeta {
    std::array<std::string, 6> paths;
    size_t top_elems = 0;
    size_t front_elems = 0;
    size_t left_elems = 0;
    size_t top_offset = 0;
    size_t front_offset = 0;
    size_t left_offset = 0;
    float* top = nullptr;
    float* bottom = nullptr;
    float* front = nullptr;
    float* back = nullptr;
    float* left = nullptr;
    float* right = nullptr;
};

struct BoundaryDisk2DWriteTask {
    std::array<std::string, 4> paths;
    size_t top_elems = 0;
    size_t left_elems = 0;
    size_t start_top_offset = 0;
    size_t start_left_offset = 0;
    std::vector<float> top;
    std::vector<float> bottom;
    std::vector<float> left;
    std::vector<float> right;
};

struct BoundaryDisk3DWriteTask {
    std::array<std::string, 6> paths;
    size_t top_elems = 0;
    size_t front_elems = 0;
    size_t left_elems = 0;
    size_t top_offset = 0;
    size_t front_offset = 0;
    size_t left_offset = 0;
    std::vector<float> top;
    std::vector<float> bottom;
    std::vector<float> front;
    std::vector<float> back;
    std::vector<float> left;
    std::vector<float> right;
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
    std::unique_lock<std::mutex> lock(boundary_disk_writer_mutex());
    boundary_disk_writer_cv().wait(lock, []() {
        return boundary_disk_writer_pending() == 0;
    });
}

inline void write_boundary_file_chunk(const std::string& path, size_t offset_elems, const float* data, size_t elems)
{
    std::ofstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out)
        throw std::runtime_error("Failed to open boundary disk file for writing: " + path);
    out.seekp(static_cast<std::streamoff>(offset_elems * sizeof(float)));
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(elems * sizeof(float)));
    if (!out)
        throw std::runtime_error("Failed to write boundary disk file: " + path);
}

inline void read_boundary_file_chunk(const std::string& path, size_t offset_elems, float* data, size_t elems)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to open boundary disk file for reading: " + path);
    in.seekg(static_cast<std::streamoff>(offset_elems * sizeof(float)));
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(elems * sizeof(float)));
    if (!in)
        throw std::runtime_error("Failed to read boundary disk file: " + path);
}

inline void write_boundary_disk_2d_task(const BoundaryDisk2DWriteTask& task)
{
    write_boundary_file_chunk(task.paths[0], task.start_top_offset, task.top.data(), task.top_elems);
    write_boundary_file_chunk(task.paths[1], task.start_top_offset, task.bottom.data(), task.top_elems);
    write_boundary_file_chunk(task.paths[2], task.start_left_offset, task.left.data(), task.left_elems);
    write_boundary_file_chunk(task.paths[3], task.start_left_offset, task.right.data(), task.left_elems);
}

inline void write_boundary_disk_3d_task(const BoundaryDisk3DWriteTask& task)
{
    write_boundary_file_chunk(task.paths[0], task.top_offset, task.top.data(), task.top_elems);
    write_boundary_file_chunk(task.paths[1], task.top_offset, task.bottom.data(), task.top_elems);
    write_boundary_file_chunk(task.paths[2], task.front_offset, task.front.data(), task.front_elems);
    write_boundary_file_chunk(task.paths[3], task.front_offset, task.back.data(), task.front_elems);
    write_boundary_file_chunk(task.paths[4], task.left_offset, task.left.data(), task.left_elems);
    write_boundary_file_chunk(task.paths[5], task.left_offset, task.right.data(), task.left_elems);
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
            }
            boundary_disk_writer_done();
        }).detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk async write launch failed: %s\n", e.what());
        try {
            write_boundary_disk_2d_task(*task);
        } catch (const std::exception& write_error) {
            std::fprintf(stderr, "Boundary disk fallback write failed: %s\n", write_error.what());
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
            }
            boundary_disk_writer_done();
        }).detach();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk async write launch failed: %s\n", e.what());
        try {
            write_boundary_disk_3d_task(*task);
        } catch (const std::exception& write_error) {
            std::fprintf(stderr, "Boundary disk fallback write failed: %s\n", write_error.what());
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
        task->top.assign(meta->top, meta->top + meta->top_elems);
        task->bottom.assign(meta->bottom, meta->bottom + meta->top_elems);
        task->left.assign(meta->left, meta->left + meta->left_elems);
        task->right.assign(meta->right, meta->right + meta->left_elems);
        launch_boundary_disk_write_2d(task);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk write callback failed: %s\n", e.what());
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
        task->top.assign(meta->top, meta->top + meta->top_elems);
        task->bottom.assign(meta->bottom, meta->bottom + meta->top_elems);
        task->front.assign(meta->front, meta->front + meta->front_elems);
        task->back.assign(meta->back, meta->back + meta->front_elems);
        task->left.assign(meta->left, meta->left + meta->left_elems);
        task->right.assign(meta->right, meta->right + meta->left_elems);
        launch_boundary_disk_write_3d(task);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk write callback failed: %s\n", e.what());
    }
}
