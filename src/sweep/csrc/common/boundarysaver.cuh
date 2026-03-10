#pragma once
#include "context.h"
#include <torch/extension.h>

enum BoundaryMode {
    BOUNDARY_SAVE = 0,
    BOUNDARY_RESTORE = 1
};

struct GeneralBoundaryPointer {

    float* __restrict__ left;
    float* __restrict__ right;

    float* __restrict__ front;
    float* __restrict__ back;

    float* __restrict__ bottom;
    float* __restrict__ top;

    float* __restrict__ last_two;
};

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

    size_t left_bytes;
    size_t front_bytes;
    size_t bottom_bytes;


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
        const std::vector<torch::Tensor>& boundary_gpu = {}
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
        
        auto gpu_options = ref_tensor.options();

        auto pinned_options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU);
            // .pinned_memory(true);

        auto storage_options = store_on_gpu ? gpu_options : pinned_options;

        // =========================
        // Physical domain
        // =========================

        int nx_phys = ctx.nx_phys();
        int nz_phys = ctx.nz_phys();
        int ny_phys = (dim == 3) ? ctx.ny_phys() : 1;
        
        // =========================
        // Allocate boundary storage
        // =========================

        if (!boundary_cpu.empty())
        {
            TORCH_CHECK(boundary_cpu.size() == 6,
                "boundary_cpu must contain 6 tensors");
            top_t    = boundary_cpu[0];
            bottom_t = boundary_cpu[1];
            front_t  = boundary_cpu[2];
            back_t   = boundary_cpu[3];
            left_t   = boundary_cpu[4];
            right_t  = boundary_cpu[5];
        }
        else
        {   
            if (dim == 3)
                {
                    left_t  = torch::zeros({nvar * ctx.nt, ctx.B, nz_phys, ny_phys, width}, storage_options);
                    right_t = torch::zeros({nvar * ctx.nt, ctx.B, nz_phys, ny_phys, width}, storage_options);

                    front_t = torch::zeros({nvar * ctx.nt, ctx.B, nz_phys, width, nx_phys}, storage_options);
                    back_t  = torch::zeros({nvar * ctx.nt, ctx.B, nz_phys, width, nx_phys}, storage_options);

                    bottom_t = torch::zeros({nvar * ctx.nt, ctx.B, width, ny_phys, nx_phys}, storage_options);
                    top_t    = torch::zeros({nvar * ctx.nt, ctx.B, width, ny_phys, nx_phys}, storage_options);

                }
            else
                {
                    left_t  = torch::zeros({nvar, ctx.nt, ctx.B, nz_phys, width}, storage_options);
                    right_t = torch::zeros({nvar, ctx.nt, ctx.B, nz_phys, width}, storage_options);

                    bottom_t = torch::zeros({nvar, ctx.nt, ctx.B, width, nx_phys}, storage_options);
                    top_t    = torch::zeros({nvar, ctx.nt, ctx.B, width, nx_phys}, storage_options);

                    front_t = torch::Tensor();
                    back_t  = torch::Tensor();
                }
        }
        // =========================
        // GPU staging buffer
        // (only when storing on CPU)
        // =========================

        if (!store_on_gpu)
        {

            if (!boundary_gpu.empty())
            {
                TORCH_CHECK(boundary_gpu.size() == 6,
                    "boundary_gpu must contain 6 tensors");
                top_gpu    = boundary_gpu[0];
                bottom_gpu = boundary_gpu[1];
                front_gpu  = boundary_gpu[2];
                back_gpu   = boundary_gpu[3];
                left_gpu   = boundary_gpu[4];
                right_gpu  = boundary_gpu[5];
            }
            else
            {
                if (dim == 3)
                {
                    left_gpu  = torch::zeros({nvar * transfer_interval, ctx.B, nz_phys, ny_phys, width}, gpu_options);
                    right_gpu = torch::zeros({nvar * transfer_interval, ctx.B, nz_phys, ny_phys, width}, gpu_options);

                    front_gpu = torch::zeros({nvar * transfer_interval, ctx.B, nz_phys, width, nx_phys}, gpu_options);
                    back_gpu  = torch::zeros({nvar * transfer_interval, ctx.B, nz_phys, width, nx_phys}, gpu_options);

                    bottom_gpu = torch::zeros({nvar * transfer_interval, ctx.B, width, ny_phys, nx_phys}, gpu_options);
                    top_gpu    = torch::zeros({nvar * transfer_interval, ctx.B, width, ny_phys, nx_phys}, gpu_options);
                }
                else
                {
                    left_gpu  = torch::zeros({nvar,transfer_interval,ctx.B,nz_phys,width}, gpu_options);
                    right_gpu = torch::zeros({nvar,transfer_interval,ctx.B,nz_phys,width}, gpu_options);

                    bottom_gpu = torch::zeros({nvar,transfer_interval,ctx.B,width,nx_phys}, gpu_options);
                    top_gpu    = torch::zeros({nvar,transfer_interval,ctx.B,width,nx_phys}, gpu_options);
                }
            }
            // =========================
            // Compute strides (elements per timestep)
            // =========================
            
            left_stride   = left_gpu.stride(1);
            right_stride  = right_gpu.stride(1);

            if (dim == 3)
            {
                front_stride  = front_gpu.stride(1);
                back_stride   = back_gpu.stride(1);
            }

            bottom_stride = bottom_gpu.stride(1);
            top_stride    = top_gpu.stride(1);

        }


        // =========================
        // last two wavefields
        // =========================

        auto last_two_options = store_on_gpu ? gpu_options : pinned_options;

        if (dim == 3)
        {
            last_two_t = torch::zeros(
                {nvar, last_two_nvar, ctx.B, 1, ctx.nz, ctx.ny, ctx.nx},
                last_two_options);
        }
        else
        {
            last_two_t = torch::zeros(
                {nvar, last_two_nvar, ctx.B, 1, ctx.nz, ctx.nx},
                last_two_options);
        }
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

    inline void flush_gpu_to_cpu(int start, int len)
    {
        size_t left_block   = left_t.stride(0);
        size_t front_block  = front_t.stride(0);
        size_t bottom_block = bottom_t.stride(0);

        size_t left_bytes   = len * nvar * left_block * sizeof(float);
        size_t front_bytes  = len * nvar * front_block * sizeof(float);
        size_t bottom_bytes = len * nvar * bottom_block * sizeof(float);

        cudaMemcpy(
            left_t.data_ptr<float>() + start * nvar * left_block,
            left_gpu.data_ptr<float>(),
            left_bytes,
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            right_t.data_ptr<float>() + start * nvar * left_block,
            right_gpu.data_ptr<float>(),
            left_bytes,
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            front_t.data_ptr<float>() + start * nvar * front_block,
            front_gpu.data_ptr<float>(),
            front_bytes,
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            back_t.data_ptr<float>() + start * nvar * front_block,
            back_gpu.data_ptr<float>(),
            front_bytes,
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            top_t.data_ptr<float>() + start * nvar * bottom_block,
            top_gpu.data_ptr<float>(),
            front_bytes,
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            bottom_t.data_ptr<float>() + start * nvar * bottom_block,
            bottom_gpu.data_ptr<float>(),
            bottom_bytes,
            cudaMemcpyDeviceToHost
        );

    }

    inline void load_cpu_to_gpu(int start, int len)
    {

        size_t top_block    = top_t.stride(0);
        size_t front_block  = front_t.stride(0);
        size_t left_block   = left_t.stride(0);

        size_t top_bytes    = len * nvar * top_block * sizeof(float);
        size_t front_bytes  = len * nvar * front_block * sizeof(float);
        size_t left_bytes   = len * nvar * left_block * sizeof(float);

        cudaMemcpy(
            top_gpu.data_ptr<float>(),
            top_t.data_ptr<float>() + start * nvar * top_block,
            top_bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            bottom_gpu.data_ptr<float>(),
            bottom_t.data_ptr<float>() + start * nvar * top_block,
            top_bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            front_gpu.data_ptr<float>(),
            front_t.data_ptr<float>() + start * nvar * front_block,
            front_bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            back_gpu.data_ptr<float>(),
            back_t.data_ptr<float>() + start * nvar * front_block,
            front_bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            left_gpu.data_ptr<float>(),
            left_t.data_ptr<float>() + start * nvar * left_block,
            left_bytes,
            cudaMemcpyHostToDevice
        );

        cudaMemcpy(
            right_gpu.data_ptr<float>(),
            right_t.data_ptr<float>() + start * nvar * left_block,
            left_bytes,
            cudaMemcpyHostToDevice
        );

    }

};

__global__ void boundary_kernel2d(
    float* __restrict__ u,

    float* __restrict__ top,
    float* __restrict__ bottom,
    float* __restrict__ left,
    float* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode
);

__global__ void boundary_kernel3d(
    float* __restrict__ u,        // (B, nz, ny, nx)

    float* __restrict__ top,      // (nt,B,width,ny_phys,nx_phys)
    float* __restrict__ bottom,

    float* __restrict__ front,    // (nt,B,nz_phys,width,nx_phys)
    float* __restrict__ back,

    float* __restrict__ left,     // (nt,B,nz_phys,ny_phys,width)
    float* __restrict__ right,

    int it,
    int width,
    int offset,
    SolverContext ctx,
    int mode
);