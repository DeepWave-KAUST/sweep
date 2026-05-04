#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
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

inline void CUDART_CB write_boundary_disk_2d_callback(void* user_data)
{
    std::unique_ptr<BoundaryDisk2DMeta> meta(static_cast<BoundaryDisk2DMeta*>(user_data));
    try {
        write_boundary_file_chunk(meta->paths[0], meta->start_top_offset, meta->top, meta->top_elems);
        write_boundary_file_chunk(meta->paths[1], meta->start_top_offset, meta->bottom, meta->top_elems);
        write_boundary_file_chunk(meta->paths[2], meta->start_left_offset, meta->left, meta->left_elems);
        write_boundary_file_chunk(meta->paths[3], meta->start_left_offset, meta->right, meta->left_elems);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk write callback failed: %s\n", e.what());
    }
}

inline void CUDART_CB write_boundary_disk_3d_callback(void* user_data)
{
    std::unique_ptr<BoundaryDisk3DMeta> meta(static_cast<BoundaryDisk3DMeta*>(user_data));
    try {
        write_boundary_file_chunk(meta->paths[0], meta->top_offset, meta->top, meta->top_elems);
        write_boundary_file_chunk(meta->paths[1], meta->top_offset, meta->bottom, meta->top_elems);
        write_boundary_file_chunk(meta->paths[2], meta->front_offset, meta->front, meta->front_elems);
        write_boundary_file_chunk(meta->paths[3], meta->front_offset, meta->back, meta->front_elems);
        write_boundary_file_chunk(meta->paths[4], meta->left_offset, meta->left, meta->left_elems);
        write_boundary_file_chunk(meta->paths[5], meta->left_offset, meta->right, meta->left_elems);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Boundary disk write callback failed: %s\n", e.what());
    }
}
