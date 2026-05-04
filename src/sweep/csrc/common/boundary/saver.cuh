#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#include <torch/extension.h>

#include "../context.h"
#include "disk_io.cuh"
#include "kernels.cuh"
#include "types.cuh"

struct EffectiveBoundarySaver {

    torch::Tensor left_t, right_t;
    torch::Tensor front_t, back_t;
    torch::Tensor bottom_t, top_t;
    torch::Tensor last_two_t;

    torch::Tensor left_gpu, right_gpu;
    torch::Tensor front_gpu, back_gpu;
    torch::Tensor bottom_gpu, top_gpu;

    bool enabled = false;

    int dim = 3;
    int nvar = 1;

    bool store_on_gpu = false;

    // stride for one timestep
    int64_t left_stride = 0;
    int64_t right_stride = 0;
    int64_t front_stride = 0;
    int64_t back_stride = 0;
    int64_t bottom_stride = 0;
    int64_t top_stride = 0;

    inline void bind_tensor_group(
        const std::vector<torch::Tensor>& tensors,
        const char* role,
        torch::Tensor& top,
        torch::Tensor& bottom,
        torch::Tensor& front,
        torch::Tensor& back,
        torch::Tensor& left,
        torch::Tensor& right
    )
    {
        if (dim == 3) {
            TORCH_CHECK(tensors.size() == 6, role, " must contain 6 tensors for 3D");
            top    = tensors[0];
            bottom = tensors[1];
            front  = tensors[2];
            back   = tensors[3];
            left   = tensors[4];
            right  = tensors[5];
        } else {
            TORCH_CHECK(tensors.size() == 4, role, " must contain 4 tensors for 2D");
            top    = tensors[0];
            bottom = tensors[1];
            left   = tensors[2];
            right  = tensors[3];
            front  = torch::Tensor();
            back   = torch::Tensor();
        }
    }

    inline void bind_storage_tensors(
        const std::vector<torch::Tensor>& tensors,
        const char* role
    )
    {
        bind_tensor_group(tensors, role, top_t, bottom_t, front_t, back_t, left_t, right_t);
    }

    inline void bind_staging_tensors(
        const std::vector<torch::Tensor>& tensors,
        const char* role
    )
    {
        bind_tensor_group(tensors, role, top_gpu, bottom_gpu, front_gpu, back_gpu, left_gpu, right_gpu);
    }

    inline void allocate_full_storage(
        const SolverContext& ctx,
        int width,
        int nx_boundary,
        int ny_boundary,
        int nz_boundary,
        const torch::TensorOptions& options
    )
    {
        if (dim == 3) {
            left_t  = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, ny_boundary, width}, options);
            right_t = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, ny_boundary, width}, options);

            front_t = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, width, nx_boundary}, options);
            back_t  = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, width, nx_boundary}, options);

            bottom_t = torch::zeros({nvar * ctx.nt, ctx.B, width, ny_boundary, nx_boundary}, options);
            top_t    = torch::zeros({nvar * ctx.nt, ctx.B, width, ny_boundary, nx_boundary}, options);
        } else {
            left_t  = torch::zeros({nvar, ctx.nt, ctx.B, nz_boundary, width}, options);
            right_t = torch::zeros({nvar, ctx.nt, ctx.B, nz_boundary, width}, options);

            bottom_t = torch::zeros({nvar, ctx.nt, ctx.B, width, nx_boundary}, options);
            top_t    = torch::zeros({nvar, ctx.nt, ctx.B, width, nx_boundary}, options);

            front_t = torch::Tensor();
            back_t  = torch::Tensor();
        }
    }

    inline void allocate_staging_storage(
        const SolverContext& ctx,
        int width,
        int transfer_interval,
        int nx_boundary,
        int ny_boundary,
        int nz_boundary,
        const torch::TensorOptions& options
    )
    {
        if (dim == 3) {
            left_gpu  = torch::zeros({nvar * transfer_interval, ctx.B, nz_boundary, ny_boundary, width}, options);
            right_gpu = torch::zeros({nvar * transfer_interval, ctx.B, nz_boundary, ny_boundary, width}, options);

            front_gpu = torch::zeros({nvar * transfer_interval, ctx.B, nz_boundary, width, nx_boundary}, options);
            back_gpu  = torch::zeros({nvar * transfer_interval, ctx.B, nz_boundary, width, nx_boundary}, options);

            bottom_gpu = torch::zeros({nvar * transfer_interval, ctx.B, width, ny_boundary, nx_boundary}, options);
            top_gpu    = torch::zeros({nvar * transfer_interval, ctx.B, width, ny_boundary, nx_boundary}, options);
        } else {
            left_gpu  = torch::zeros({nvar, transfer_interval, ctx.B, nz_boundary, width}, options);
            right_gpu = torch::zeros({nvar, transfer_interval, ctx.B, nz_boundary, width}, options);

            bottom_gpu = torch::zeros({nvar, transfer_interval, ctx.B, width, nx_boundary}, options);
            top_gpu    = torch::zeros({nvar, transfer_interval, ctx.B, width, nx_boundary}, options);
        }
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

    inline void allocate_last_two(
        const SolverContext& ctx,
        int last_two_nvar,
        const torch::Tensor& last_two,
        const torch::TensorOptions& options
    )
    {
        if (last_two.defined()) {
            last_two_t = last_two;
        } else if (dim == 3) {
            last_two_t = torch::zeros({nvar, last_two_nvar, ctx.B, 1, ctx.nz, ctx.ny, ctx.nx}, options);
        } else {
            last_two_t = torch::zeros({nvar, last_two_nvar, ctx.B, 1, ctx.nz, ctx.nx}, options);
        }
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
        int tangent_pad = 0
    )
    {
        enabled = use_boundary_saving;
        dim = dim_;
        nvar = nvar_;

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

        auto gpu_options = ref_tensor.options().dtype(torch::kFloat32);

        auto pinned_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU)
            .pinned_memory(use_pinned_memory_);

        auto storage_options = store_on_gpu ? gpu_options : pinned_options;

        // =========================
        // Physical domain
        // =========================

        int nx_phys = ctx.nx_phys();
        int nz_phys = ctx.nz_phys();
        int ny_phys = (dim == 3) ? ctx.ny_phys() : 1;

        int nx_boundary = nx_phys + 2 * tangent_pad;
        int nz_boundary = nz_phys + 2 * tangent_pad;
        int ny_boundary = (dim == 3) ? ny_phys + 2 * tangent_pad : 1;
        
        if (!boundary_cpu.empty()) {
            bind_storage_tensors(boundary_cpu, "boundary_cpu");
        } else if (store_on_gpu && !boundary_gpu.empty()) {
            bind_storage_tensors(boundary_gpu, "boundary_gpu");
        } else {
            allocate_full_storage(ctx, width, nx_boundary, ny_boundary, nz_boundary, storage_options);
        }

        if (!store_on_gpu) {
            if (!boundary_gpu.empty())
                bind_staging_tensors(boundary_gpu, "boundary_gpu");
            else
                allocate_staging_storage(ctx, width, transfer_interval, nx_boundary, ny_boundary, nz_boundary, gpu_options);
        }

        compute_time_strides();

        auto last_two_options = store_on_gpu ? gpu_options : pinned_options;
        allocate_last_two(ctx, last_two_nvar, last_two, last_two_options);
    }

    GeneralBoundaryPointer view()
    {
        GeneralBoundaryPointer v{};

        if (!enabled) return v;

        v.left  = left_t.data_ptr<float>();
        v.right = right_t.data_ptr<float>();

        if (dim == 3) {
            v.front = front_t.data_ptr<float>();
            v.back  = back_t.data_ptr<float>();
        } else {
            v.front = nullptr;
            v.back  = nullptr;
        }

        v.bottom = bottom_t.data_ptr<float>();
        v.top    = top_t.data_ptr<float>();

        v.last_two = last_two_t.data_ptr<float>();

        return v;

    }

    void load_from_vector(
        const std::vector<torch::Tensor>& u_boundary,
        const torch::Tensor& ref_tensor
        )
        {
            if (!enabled)
                throw std::runtime_error("Boundary saving not enabled.");


            auto copy_to = [&](torch::Tensor& dst, const torch::Tensor& src)
            {
                if (dst.device() == src.device()) {
                    dst.copy_(src);
                }
                else {
                    dst.copy_(src, /*non_blocking=*/true);
                }
            };

            if (dim == 2) {

                if (u_boundary.size() != 4)
                    throw std::runtime_error("2D boundary expects 4 tensors.");
                
                copy_to( top_t, u_boundary[0]);
                copy_to( bottom_t, u_boundary[1]);
                copy_to( left_t, u_boundary[2]);
                copy_to( right_t, u_boundary[3]);

            } else { // 3D

                if (u_boundary.size() != 6)
                    throw std::runtime_error("3D boundary expects 6 tensors.");

                copy_to( top_t, u_boundary[0]);
                copy_to( bottom_t, u_boundary[1]);

                copy_to( front_t, u_boundary[2]);
                copy_to( back_t, u_boundary[3]);

                copy_to( left_t, u_boundary[4]);
                copy_to( right_t, u_boundary[5]);
            }
        }

    inline void copy_2d_chunk_async(
        float* dst,
        size_t dst_var_block,
        const float* src,
        size_t src_var_block,
        size_t bytes,
        cudaMemcpyKind kind,
        cudaStream_t stream
    ) const
    {
        if (nvar == 1) {
            cudaMemcpyAsync(dst, src, bytes, kind, stream);
        } else {
            cudaMemcpy2DAsync(
                dst,
                dst_var_block * sizeof(float),
                src,
                src_var_block * sizeof(float),
                bytes,
                nvar,
                kind,
                stream
            );
        }
    }

    inline void flush_gpu_to_cpu(int start, int len, cudaStream_t stream, int gpu_start = 0)
    {
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

            size_t top_bytes = len * top_time_block * sizeof(float);
            size_t left_bytes = len * left_time_block * sizeof(float);

            TORCH_CHECK(top_t.dtype() == torch::kFloat32, "Boundary storage must be float32.");
            copy_2d_chunk_async(
                top_t.data_ptr<float>() + start * top_time_block,
                top_var_block,
                top_gpu.data_ptr<float>() + gpu_start * top_gpu_time_block,
                top_gpu_var_block,
                top_bytes,
                cudaMemcpyDeviceToHost,
                stream
            );
            copy_2d_chunk_async(
                bottom_t.data_ptr<float>() + start * top_time_block,
                top_var_block,
                bottom_gpu.data_ptr<float>() + gpu_start * top_gpu_time_block,
                top_gpu_var_block,
                top_bytes,
                cudaMemcpyDeviceToHost,
                stream
            );
            copy_2d_chunk_async(
                left_t.data_ptr<float>() + start * left_time_block,
                left_var_block,
                left_gpu.data_ptr<float>() + gpu_start * left_gpu_time_block,
                left_gpu_var_block,
                left_bytes,
                cudaMemcpyDeviceToHost,
                stream
            );
            copy_2d_chunk_async(
                right_t.data_ptr<float>() + start * left_time_block,
                left_var_block,
                right_gpu.data_ptr<float>() + gpu_start * left_gpu_time_block,
                left_gpu_var_block,
                left_bytes,
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


        TORCH_CHECK(left_t.dtype() == torch::kFloat32, "Boundary storage must be float32.");
        cudaMemcpyAsync(
            left_t.data_ptr<float>() + start * nvar * left_block,
            left_gpu.data_ptr<float>() + left_gpu_offset,
            left_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            right_t.data_ptr<float>() + start * nvar * left_block,
            right_gpu.data_ptr<float>() + left_gpu_offset,
            left_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            front_t.data_ptr<float>() + start * nvar * front_block,
            front_gpu.data_ptr<float>() + front_gpu_offset,
            front_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            back_t.data_ptr<float>() + start * nvar * front_block,
            back_gpu.data_ptr<float>() + front_gpu_offset,
            front_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            top_t.data_ptr<float>() + start * nvar * bottom_block,
            top_gpu.data_ptr<float>() + bottom_gpu_offset,
            bottom_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            bottom_t.data_ptr<float>() + start * nvar * bottom_block,
            bottom_gpu.data_ptr<float>() + bottom_gpu_offset,
            bottom_elems * sizeof(float),
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
        if (paths.size() != 4)
            throw std::runtime_error("2D disk boundary expects 4 file paths.");

        size_t top_var_block = top_t.stride(0);
        size_t top_time_block = top_t.stride(1);
        size_t left_var_block = left_t.stride(0);
        size_t left_time_block = left_t.stride(1);
        size_t top_gpu_var_block = top_gpu.stride(0);
        size_t top_gpu_time_block = top_gpu.stride(1);
        size_t left_gpu_var_block = left_gpu.stride(0);
        size_t left_gpu_time_block = left_gpu.stride(1);

        size_t top_bytes = len * top_time_block * sizeof(float);
        size_t left_bytes = len * left_time_block * sizeof(float);

        copy_2d_chunk_async(
            top_t.data_ptr<float>() + stage_start * top_time_block,
            top_var_block,
            top_gpu.data_ptr<float>() + gpu_start * top_gpu_time_block,
            top_gpu_var_block,
            top_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        copy_2d_chunk_async(
            bottom_t.data_ptr<float>() + stage_start * top_time_block,
            top_var_block,
            bottom_gpu.data_ptr<float>() + gpu_start * top_gpu_time_block,
            top_gpu_var_block,
            top_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        copy_2d_chunk_async(
            left_t.data_ptr<float>() + stage_start * left_time_block,
            left_var_block,
            left_gpu.data_ptr<float>() + gpu_start * left_gpu_time_block,
            left_gpu_var_block,
            left_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );
        copy_2d_chunk_async(
            right_t.data_ptr<float>() + stage_start * left_time_block,
            left_var_block,
            right_gpu.data_ptr<float>() + gpu_start * left_gpu_time_block,
            left_gpu_var_block,
            left_bytes,
            cudaMemcpyDeviceToHost,
            stream
        );

        auto* meta = new BoundaryDisk2DMeta();
        meta->paths = {paths[0], paths[1], paths[2], paths[3]};
        meta->top_elems = len * top_time_block;
        meta->left_elems = len * left_time_block;
        meta->start_top_offset = static_cast<size_t>(start) * top_time_block;
        meta->start_left_offset = static_cast<size_t>(start) * left_time_block;
        meta->top = top_t.data_ptr<float>() + stage_start * top_time_block;
        meta->bottom = bottom_t.data_ptr<float>() + stage_start * top_time_block;
        meta->left = left_t.data_ptr<float>() + stage_start * left_time_block;
        meta->right = right_t.data_ptr<float>() + stage_start * left_time_block;
        cudaLaunchHostFunc(stream, write_boundary_disk_2d_callback, meta);
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
        if (paths.size() != 6)
            throw std::runtime_error("3D disk boundary expects 6 file paths.");

        size_t top_block = top_t.stride(0);
        size_t front_block = front_t.stride(0);
        size_t left_block = left_t.stride(0);
        size_t top_gpu_block = top_gpu.stride(0);
        size_t front_gpu_block = front_gpu.stride(0);
        size_t left_gpu_block = left_gpu.stride(0);

        size_t top_elems = len * nvar * top_block;
        size_t front_elems = len * nvar * front_block;
        size_t left_elems = len * nvar * left_block;
        size_t top_stage_offset = static_cast<size_t>(stage_start) * nvar * top_block;
        size_t front_stage_offset = static_cast<size_t>(stage_start) * nvar * front_block;
        size_t left_stage_offset = static_cast<size_t>(stage_start) * nvar * left_block;
        size_t top_gpu_offset = static_cast<size_t>(gpu_start) * nvar * top_gpu_block;
        size_t front_gpu_offset = static_cast<size_t>(gpu_start) * nvar * front_gpu_block;
        size_t left_gpu_offset = static_cast<size_t>(gpu_start) * nvar * left_gpu_block;

        cudaMemcpyAsync(
            top_t.data_ptr<float>() + top_stage_offset,
            top_gpu.data_ptr<float>() + top_gpu_offset,
            top_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            bottom_t.data_ptr<float>() + top_stage_offset,
            bottom_gpu.data_ptr<float>() + top_gpu_offset,
            top_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            front_t.data_ptr<float>() + front_stage_offset,
            front_gpu.data_ptr<float>() + front_gpu_offset,
            front_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            back_t.data_ptr<float>() + front_stage_offset,
            back_gpu.data_ptr<float>() + front_gpu_offset,
            front_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            left_t.data_ptr<float>() + left_stage_offset,
            left_gpu.data_ptr<float>() + left_gpu_offset,
            left_elems * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            right_t.data_ptr<float>() + left_stage_offset,
            right_gpu.data_ptr<float>() + left_gpu_offset,
            left_elems * sizeof(float),
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
        meta->top = top_t.data_ptr<float>() + top_stage_offset;
        meta->bottom = bottom_t.data_ptr<float>() + top_stage_offset;
        meta->front = front_t.data_ptr<float>() + front_stage_offset;
        meta->back = back_t.data_ptr<float>() + front_stage_offset;
        meta->left = left_t.data_ptr<float>() + left_stage_offset;
        meta->right = right_t.data_ptr<float>() + left_stage_offset;
        cudaLaunchHostFunc(stream, write_boundary_disk_3d_callback, meta);
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
        if (paths.size() != 4)
            throw std::runtime_error("2D disk boundary expects 4 file paths.");

        size_t top_time_block = top_t.stride(1);
        size_t left_time_block = left_t.stride(1);
        size_t top_elems = len * top_time_block;
        size_t left_elems = len * left_time_block;
        size_t top_offset = static_cast<size_t>(start) * top_time_block;
        size_t left_offset = static_cast<size_t>(start) * left_time_block;

        read_boundary_file_chunk(paths[0], top_offset, top_t.data_ptr<float>() + stage_start * top_time_block, top_elems);
        read_boundary_file_chunk(paths[1], top_offset, bottom_t.data_ptr<float>() + stage_start * top_time_block, top_elems);
        read_boundary_file_chunk(paths[2], left_offset, left_t.data_ptr<float>() + stage_start * left_time_block, left_elems);
        read_boundary_file_chunk(paths[3], left_offset, right_t.data_ptr<float>() + stage_start * left_time_block, left_elems);
    }

    inline void load_disk_to_cpu_3d(int start, int len, const std::vector<std::string>& paths, int stage_start = 0)
    {
        if (dim != 3)
            throw std::runtime_error("load_disk_to_cpu_3d only supports 3D boundaries.");
        if (paths.size() != 6)
            throw std::runtime_error("3D disk boundary expects 6 file paths.");

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

        read_boundary_file_chunk(paths[0], top_offset, top_t.data_ptr<float>() + top_stage_offset, top_elems);
        read_boundary_file_chunk(paths[1], top_offset, bottom_t.data_ptr<float>() + top_stage_offset, top_elems);
        read_boundary_file_chunk(paths[2], front_offset, front_t.data_ptr<float>() + front_stage_offset, front_elems);
        read_boundary_file_chunk(paths[3], front_offset, back_t.data_ptr<float>() + front_stage_offset, front_elems);
        read_boundary_file_chunk(paths[4], left_offset, left_t.data_ptr<float>() + left_stage_offset, left_elems);
        read_boundary_file_chunk(paths[5], left_offset, right_t.data_ptr<float>() + left_stage_offset, left_elems);
    }

    inline void load_cpu_to_gpu(int start, int len, cudaStream_t stream, int gpu_start = 0)
    {
        if (dim == 2)
        {
            size_t top_var_block = top_t.stride(0);
            size_t top_time_block = top_t.stride(1);
            size_t left_var_block = left_t.stride(0);
            size_t left_time_block = left_t.stride(1);
            size_t top_gpu_var_block = top_gpu.stride(0);
            size_t left_gpu_var_block = left_gpu.stride(0);

            size_t top_bytes = len * top_time_block * sizeof(float);
            size_t left_bytes = len * left_time_block * sizeof(float);
            size_t top_gpu_time_block = top_gpu.stride(1);
            size_t left_gpu_time_block = left_gpu.stride(1);

            TORCH_CHECK(top_t.dtype() == torch::kFloat32, "Boundary storage must be float32.");
            copy_2d_chunk_async(
                top_gpu.data_ptr<float>() + gpu_start * top_gpu_time_block,
                top_gpu_var_block,
                top_t.data_ptr<float>() + start * top_time_block,
                top_var_block,
                top_bytes,
                cudaMemcpyHostToDevice,
                stream
            );
            copy_2d_chunk_async(
                bottom_gpu.data_ptr<float>() + gpu_start * top_gpu_time_block,
                top_gpu_var_block,
                bottom_t.data_ptr<float>() + start * top_time_block,
                top_var_block,
                top_bytes,
                cudaMemcpyHostToDevice,
                stream
            );
            copy_2d_chunk_async(
                left_gpu.data_ptr<float>() + gpu_start * left_gpu_time_block,
                left_gpu_var_block,
                left_t.data_ptr<float>() + start * left_time_block,
                left_var_block,
                left_bytes,
                cudaMemcpyHostToDevice,
                stream
            );
            copy_2d_chunk_async(
                right_gpu.data_ptr<float>() + gpu_start * left_gpu_time_block,
                left_gpu_var_block,
                right_t.data_ptr<float>() + start * left_time_block,
                left_var_block,
                left_bytes,
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

        TORCH_CHECK(top_t.dtype() == torch::kFloat32, "Boundary storage must be float32.");
        cudaMemcpyAsync(
            top_gpu.data_ptr<float>() + top_gpu_offset,
            top_t.data_ptr<float>() + start * nvar * top_block,
            top_elems * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            bottom_gpu.data_ptr<float>() + top_gpu_offset,
            bottom_t.data_ptr<float>() + start * nvar * top_block,
            top_elems * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            front_gpu.data_ptr<float>() + front_gpu_offset,
            front_t.data_ptr<float>() + start * nvar * front_block,
            front_elems * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            back_gpu.data_ptr<float>() + front_gpu_offset,
            back_t.data_ptr<float>() + start * nvar * front_block,
            front_elems * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            left_gpu.data_ptr<float>() + left_gpu_offset,
            left_t.data_ptr<float>() + start * nvar * left_block,
            left_elems * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            right_gpu.data_ptr<float>() + left_gpu_offset,
            right_t.data_ptr<float>() + start * nvar * left_block,
            left_elems * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        );
    }

};
