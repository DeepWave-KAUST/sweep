#pragma once
#include <cuda_runtime.h>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <torch/extension.h>

#include "../context.h"
#include "disk_io.cuh"
#include "kernels.cuh"
#include "types.cuh"

// Map a boundary-buffer tensor's dtype to the storage-dtype enum.  Python
// allocates the buffers in the requested precision, so the tensor itself is
// the source of truth (uint8 == the INT8 path's quantized main buffer).
static inline BoundaryDtype boundary_dtype_from_tensor(const torch::Tensor& t) {
    switch (t.scalar_type()) {
        case torch::kUInt8:    return BoundaryDtype::INT8;
        case torch::kHalf:     return BoundaryDtype::FP16;
        case torch::kBFloat16: return BoundaryDtype::BF16;
        default:               return BoundaryDtype::FP32;
    }
}

// Byte pointer into a boundary tensor at element offset `off_elems`, honoring
// the tensor's dtype width.  Used so the gpu<->cpu<->disk byte copies work for
// fp16/bf16 staging/storage, not just fp32.
static inline char* boundary_byte_ptr(const torch::Tensor& t, int64_t off_elems) {
    return static_cast<char*>(t.data_ptr()) + off_elems * t.element_size();
}

struct EffectiveBoundarySaver {

    torch::Tensor left_t, right_t;
    torch::Tensor front_t, back_t;
    torch::Tensor bottom_t, top_t;
    torch::Tensor last_two_t;

    torch::Tensor left_gpu, right_gpu;
    torch::Tensor front_gpu, back_gpu;
    torch::Tensor bottom_gpu, top_gpu;

    // INT8 path only: per-face FP32 scale buffer (one per quantization
    // block) and an FP32 staging buffer holding a single timestep's
    // worth of data per face.  Compute writes FP32 into staging via the
    // existing kernel path, then launch_quantize_int8 reduces+stores
    // into the persistent uint8 face buffer (left_t et al.) plus this
    // scale buffer.
    torch::Tensor left_scale_t, right_scale_t;
    torch::Tensor front_scale_t, back_scale_t;
    torch::Tensor bottom_scale_t, top_scale_t;

    // INT8 staged path (storage='cpu'/'disk') only: per-face FP32 scale
    // RING on the GPU, parallel to the uint8 main ring (top_gpu et al.).
    // Each chunk is flushed to / loaded from the persistent FP32 scale
    // buffer (top_scale_t et al.) alongside the uint8 main buffer.
    torch::Tensor left_scale_gpu, right_scale_gpu;
    torch::Tensor front_scale_gpu, back_scale_gpu;
    torch::Tensor bottom_scale_gpu, top_scale_gpu;

    torch::Tensor left_staging_t, right_staging_t;
    torch::Tensor front_staging_t, back_staging_t;
    torch::Tensor bottom_staging_t, top_staging_t;

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

    // INT8 path: allocate transient FP32 staging buffer (one timestep
    // per face).  Compute writes FP32 into staging via the existing
    // FP32 boundary kernel; launch_quantize_int8 then compresses staging
    // into the persistent uint8 + scale buffers (bound externally).
    inline void allocate_int8_staging(
        const SolverContext& ctx,
        int width,
        int nx_boundary,
        int ny_boundary,
        int nz_boundary,
        const torch::TensorOptions& fp32_options)
    {
        if (dim == 3) {
            top_staging_t    = torch::zeros({1, ctx.B, width, ny_boundary, nx_boundary}, fp32_options);
            bottom_staging_t = torch::zeros({1, ctx.B, width, ny_boundary, nx_boundary}, fp32_options);
            front_staging_t  = torch::zeros({1, ctx.B, nz_boundary, width, nx_boundary}, fp32_options);
            back_staging_t   = torch::zeros({1, ctx.B, nz_boundary, width, nx_boundary}, fp32_options);
            left_staging_t   = torch::zeros({1, ctx.B, nz_boundary, ny_boundary, width}, fp32_options);
            right_staging_t  = torch::zeros({1, ctx.B, nz_boundary, ny_boundary, width}, fp32_options);
        } else {
            top_staging_t    = torch::zeros({1, 1, ctx.B, width, nx_boundary}, fp32_options);
            bottom_staging_t = torch::zeros({1, 1, ctx.B, width, nx_boundary}, fp32_options);
            left_staging_t   = torch::zeros({1, 1, ctx.B, nz_boundary, width}, fp32_options);
            right_staging_t  = torch::zeros({1, 1, ctx.B, nz_boundary, width}, fp32_options);
        }
        // Defensive: quantize_step / dequantize_step copy the persistent buffer's
        // full per-step stride (top_t.stride(0) in 3D) between the staging and the
        // int8 ring.  If the caller's tangent_pad here disagrees with the pad the
        // persistent buffers were allocated with (Python boundary_tangent_pad),
        // the staging is undersized and the copy reads/writes out of bounds.  Turn
        // that silent illegal access into a clear error at allocation time.
        if (dim == 3 && top_t.defined() && top_t.numel() > 0) {
            TORCH_CHECK(top_staging_t.numel() >= top_t.stride(0) &&
                        left_staging_t.numel() >= left_t.stride(0) &&
                        front_staging_t.numel() >= front_t.stride(0),
                        "INT8 FP32 staging is smaller than the persistent boundary "
                        "buffer per-step stride -- tangent_pad mismatch between the "
                        "Python layout and the CUDA saver (top ", top_staging_t.numel(),
                        " vs ", top_t.stride(0), ").");
        }
    }

    // INT8 path (legacy, fully-internal allocation — used only if
    // Python doesn't pre-allocate; saver-internal data is lost between
    // forward and backward so this isn't used in the production flow).
    inline void allocate_int8_main_and_scale(
        const SolverContext& ctx,
        int width,
        int nx_boundary,
        int ny_boundary,
        int nz_boundary,
        const torch::TensorOptions& gpu_options)
    {
        auto u8_options = gpu_options.dtype(torch::kUInt8);
        auto scale_options = gpu_options.dtype(torch::kFloat32);

        auto blocks_per_step = [](int64_t cells_per_step) -> int64_t {
            return (cells_per_step + BOUNDARY_INT8_BLOCK - 1) / BOUNDARY_INT8_BLOCK;
        };

        if (dim == 3) {
            int64_t top_step = (int64_t)width * ny_boundary * nx_boundary * ctx.B;
            int64_t side_step_lr = (int64_t)nz_boundary * ny_boundary * width * ctx.B;
            int64_t side_step_fb = (int64_t)nz_boundary * width * nx_boundary * ctx.B;

            top_t    = torch::zeros({nvar * ctx.nt, ctx.B, width, ny_boundary, nx_boundary}, u8_options);
            bottom_t = torch::zeros({nvar * ctx.nt, ctx.B, width, ny_boundary, nx_boundary}, u8_options);
            front_t  = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, width, nx_boundary}, u8_options);
            back_t   = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, width, nx_boundary}, u8_options);
            left_t   = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, ny_boundary, width}, u8_options);
            right_t  = torch::zeros({nvar * ctx.nt, ctx.B, nz_boundary, ny_boundary, width}, u8_options);

            top_scale_t    = torch::zeros({nvar * ctx.nt, blocks_per_step(top_step)}, scale_options);
            bottom_scale_t = torch::zeros({nvar * ctx.nt, blocks_per_step(top_step)}, scale_options);
            front_scale_t  = torch::zeros({nvar * ctx.nt, blocks_per_step(side_step_fb)}, scale_options);
            back_scale_t   = torch::zeros({nvar * ctx.nt, blocks_per_step(side_step_fb)}, scale_options);
            left_scale_t   = torch::zeros({nvar * ctx.nt, blocks_per_step(side_step_lr)}, scale_options);
            right_scale_t  = torch::zeros({nvar * ctx.nt, blocks_per_step(side_step_lr)}, scale_options);

            // FP32 staging: one timestep's worth per face (no nt dim).
            top_staging_t    = torch::zeros({1, ctx.B, width, ny_boundary, nx_boundary}, scale_options);
            bottom_staging_t = torch::zeros({1, ctx.B, width, ny_boundary, nx_boundary}, scale_options);
            front_staging_t  = torch::zeros({1, ctx.B, nz_boundary, width, nx_boundary}, scale_options);
            back_staging_t   = torch::zeros({1, ctx.B, nz_boundary, width, nx_boundary}, scale_options);
            left_staging_t   = torch::zeros({1, ctx.B, nz_boundary, ny_boundary, width}, scale_options);
            right_staging_t  = torch::zeros({1, ctx.B, nz_boundary, ny_boundary, width}, scale_options);
        } else {
            int64_t top_step = (int64_t)width * nx_boundary * ctx.B;
            int64_t side_step = (int64_t)nz_boundary * width * ctx.B;

            top_t    = torch::zeros({nvar, ctx.nt, ctx.B, width, nx_boundary}, u8_options);
            bottom_t = torch::zeros({nvar, ctx.nt, ctx.B, width, nx_boundary}, u8_options);
            left_t   = torch::zeros({nvar, ctx.nt, ctx.B, nz_boundary, width}, u8_options);
            right_t  = torch::zeros({nvar, ctx.nt, ctx.B, nz_boundary, width}, u8_options);
            front_t  = torch::Tensor();
            back_t   = torch::Tensor();

            top_scale_t    = torch::zeros({nvar, ctx.nt, blocks_per_step(top_step)}, scale_options);
            bottom_scale_t = torch::zeros({nvar, ctx.nt, blocks_per_step(top_step)}, scale_options);
            left_scale_t   = torch::zeros({nvar, ctx.nt, blocks_per_step(side_step)}, scale_options);
            right_scale_t  = torch::zeros({nvar, ctx.nt, blocks_per_step(side_step)}, scale_options);

            top_staging_t    = torch::zeros({1, 1, ctx.B, width, nx_boundary}, scale_options);
            bottom_staging_t = torch::zeros({1, 1, ctx.B, width, nx_boundary}, scale_options);
            left_staging_t   = torch::zeros({1, 1, ctx.B, nz_boundary, width}, scale_options);
            right_staging_t  = torch::zeros({1, 1, ctx.B, nz_boundary, width}, scale_options);
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
        // from leaking across propagator instances.  The SWEEP_BOUNDARY_DTYPE
        // env is consulted only for the (normally unreachable) C++-side
        // self-allocation fallback where no buffer is passed.  Last-two and
        // staging buffers remain FP32 — last_two is a full-wavefield snapshot
        // used to bootstrap backward, staging is the INT8 path's per-timestep
        // FP32 view before quantization.
        BoundaryDtype runtime_dtype;
        if (!boundary_gpu.empty() && boundary_gpu[0].scalar_type() == torch::kUInt8) {
            runtime_dtype = BoundaryDtype::INT8;        // uint8 main buffer => INT8 path
        } else if (!boundary_cpu.empty()) {
            runtime_dtype = boundary_dtype_from_tensor(boundary_cpu[0]);
        } else if (!boundary_gpu.empty()) {
            runtime_dtype = boundary_dtype_from_tensor(boundary_gpu[0]);
        } else {
            runtime_dtype = sweep_boundary_dtype_env();
        }
        const bool use_scaled = (runtime_dtype == BoundaryDtype::INT8 ||
                                 runtime_dtype == BoundaryDtype::FP16);
        torch::Dtype dtype_storage;
        switch (runtime_dtype) {
            case BoundaryDtype::FP16: dtype_storage = torch::kHalf; break;
            case BoundaryDtype::BF16: dtype_storage = torch::kBFloat16; break;
            case BoundaryDtype::INT8: dtype_storage = torch::kUInt8; break;
            default:                  dtype_storage = torch::kFloat32; break;
        }

        auto gpu_options = ref_tensor.options().dtype(dtype_storage);

        auto pinned_options = torch::TensorOptions()
            .dtype(dtype_storage)
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
        
        const size_t scaled_faces = (dim == 3 ? 6 : 4);
        if (use_scaled && store_on_gpu) {
            // Scaled gpu-direct (INT8 / FP16): Python owns a persistent
            // payload main (uint8 or fp16) + FP32 per-block scale tensors
            // and passes them concatenated in ``boundary_gpu`` (first half
            // = main, second half = scale).  We bind those here and
            // allocate FP32 staging buffers internally (single timestep,
            // transient).
            TORCH_CHECK(boundary_gpu.size() == 2 * scaled_faces,
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
                                  gpu_options.dtype(torch::kFloat32));
        } else if (use_scaled) {
            // Scaled staged (cpu/disk): Python owns a persistent payload
            // main + FP32 scale buffer (``boundary_cpu``; on disk these are
            // the cpu staging buffers) AND a payload main + FP32 scale RING
            // on the GPU (``boundary_gpu``).  Both are concatenated
            // main-then-scale.  We bind the cpu side to top_t/top_scale_t,
            // the gpu rings to top_gpu/top_scale_gpu, and allocate the
            // shared FP32 transient staging internally (a single timestep,
            // like gpu-direct).
            TORCH_CHECK(boundary_cpu.size() == 2 * scaled_faces,
                        "Scaled (int8/fp16) staged boundary_cpu expects ", 2 * scaled_faces,
                        " tensors (", scaled_faces, " main + ", scaled_faces,
                        " scale), got ", boundary_cpu.size());
            TORCH_CHECK(boundary_gpu.size() == 2 * scaled_faces,
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
                                  gpu_options.dtype(torch::kFloat32));
        } else if (!boundary_cpu.empty()) {
            bind_storage_tensors(boundary_cpu, "boundary_cpu");
        } else if (store_on_gpu && !boundary_gpu.empty()) {
            bind_storage_tensors(boundary_gpu, "boundary_gpu");
        } else {
            allocate_full_storage(ctx, width, nx_boundary, ny_boundary, nz_boundary, storage_options);
        }

        if (!use_scaled && !store_on_gpu) {
            if (!boundary_gpu.empty())
                bind_staging_tensors(boundary_gpu, "boundary_gpu");
            else
                allocate_staging_storage(ctx, width, transfer_interval, nx_boundary, ny_boundary, nz_boundary, gpu_options);
        }

        compute_time_strides();

        // last_two is the wavefield snapshot used to bootstrap the backward
        // pass — full-precision matters here even when boundary buffers go
        // FP16 (PoC choice; can be relaxed later).
        auto last_two_options = (store_on_gpu ? gpu_options : pinned_options).dtype(torch::kFloat32);
        allocate_last_two(ctx, last_two_nvar, last_two, last_two_options);
    }

    GeneralBoundaryPointer view()
    {
        GeneralBoundaryPointer v{};

        if (!enabled) return v;

        // last_two is always FP32 (we forced it in allocate above).
        v.last_two = last_two_t.data_ptr<float>();

        // Detect storage dtype on the persistent face buffer (left_t) to
        // decide which set of pointers to populate.
        auto dt = left_t.dtype();
        if (dt == torch::kHalf) {
            // FP16 is per-block scaled storage (same two-pass flow as
            // INT8): persistent __half payload + FP32 per-block scales +
            // FP32 transient staging for the boundary kernels.
            TORCH_CHECK(top_scale_t.defined(),
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
        } else if (dt == torch::kBFloat16) {
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
        } else if (dt == torch::kUInt8) {
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
            if (left_t.defined() && (left_t.dtype() == torch::kUInt8 ||
                                     left_t.dtype() == torch::kHalf))
                return;

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
            auto face = [&](const torch::Tensor& cpu_t, const torch::Tensor& gpu_t,
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
        auto face = [&](const torch::Tensor& cpu_t, const torch::Tensor& gpu_t,
                        size_t cpu_block, size_t gpu_offset, size_t elems) {
            char* cpu_p = boundary_byte_ptr(cpu_t, (int64_t)cpu_start * nvar * cpu_block);
            char* gpu_p = boundary_byte_ptr(gpu_t, (int64_t)gpu_offset);
            if (d2h)
                cudaMemcpyAsync(cpu_p, gpu_p, elems * es, kind, stream);
            else
                cudaMemcpyAsync(gpu_p, cpu_p, elems * es, kind, stream);
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
            (top_t.dtype() == torch::kUInt8 || top_t.dtype() == torch::kHalf))
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

            TORCH_CHECK(top_t.dtype() == torch::kFloat32 || top_t.dtype() == torch::kHalf || top_t.dtype() == torch::kBFloat16 || top_t.dtype() == torch::kUInt8, "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
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


        TORCH_CHECK(left_t.dtype() == torch::kFloat32 || left_t.dtype() == torch::kHalf || left_t.dtype() == torch::kBFloat16 || left_t.dtype() == torch::kUInt8, "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
        size_t es = left_t.element_size();
        cudaMemcpyAsync(
            boundary_byte_ptr(left_t, (int64_t)start * nvar * left_block),
            boundary_byte_ptr(left_gpu, (int64_t)left_gpu_offset),
            left_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(right_t, (int64_t)start * nvar * left_block),
            boundary_byte_ptr(right_gpu, (int64_t)left_gpu_offset),
            left_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(front_t, (int64_t)start * nvar * front_block),
            boundary_byte_ptr(front_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(back_t, (int64_t)start * nvar * front_block),
            boundary_byte_ptr(back_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(top_t, (int64_t)start * nvar * bottom_block),
            boundary_byte_ptr(top_gpu, (int64_t)bottom_gpu_offset),
            bottom_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaMemcpyAsync(
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
            (top_t.dtype() == torch::kUInt8 || top_t.dtype() == torch::kHalf);
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
            (top_t.dtype() == torch::kUInt8 || top_t.dtype() == torch::kHalf);
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

        cudaMemcpyAsync(
            boundary_byte_ptr(top_t, (int64_t)top_stage_offset),
            boundary_byte_ptr(top_gpu, (int64_t)top_gpu_offset),
            top_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            boundary_byte_ptr(bottom_t, (int64_t)top_stage_offset),
            boundary_byte_ptr(bottom_gpu, (int64_t)top_gpu_offset),
            top_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            boundary_byte_ptr(front_t, (int64_t)front_stage_offset),
            boundary_byte_ptr(front_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            boundary_byte_ptr(back_t, (int64_t)front_stage_offset),
            boundary_byte_ptr(back_gpu, (int64_t)front_gpu_offset),
            front_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            boundary_byte_ptr(left_t, (int64_t)left_stage_offset),
            boundary_byte_ptr(left_gpu, (int64_t)left_gpu_offset),
            left_elems * es,
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
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
            (top_t.dtype() == torch::kUInt8 || top_t.dtype() == torch::kHalf);
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
            (top_t.dtype() == torch::kUInt8 || top_t.dtype() == torch::kHalf);
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
            (top_t.dtype() == torch::kUInt8 || top_t.dtype() == torch::kHalf))
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

            TORCH_CHECK(top_t.dtype() == torch::kFloat32 || top_t.dtype() == torch::kHalf || top_t.dtype() == torch::kBFloat16 || top_t.dtype() == torch::kUInt8, "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
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

        TORCH_CHECK(top_t.dtype() == torch::kFloat32 || top_t.dtype() == torch::kHalf || top_t.dtype() == torch::kBFloat16 || top_t.dtype() == torch::kUInt8, "Boundary storage must be float32 / float16 / bfloat16 / uint8.");
        size_t es = top_t.element_size();
        cudaMemcpyAsync(
            boundary_byte_ptr(top_gpu, (int64_t)top_gpu_offset),
            boundary_byte_ptr(top_t, (int64_t)start * nvar * top_block),
            top_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(bottom_gpu, (int64_t)top_gpu_offset),
            boundary_byte_ptr(bottom_t, (int64_t)start * nvar * top_block),
            top_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(front_gpu, (int64_t)front_gpu_offset),
            boundary_byte_ptr(front_t, (int64_t)start * nvar * front_block),
            front_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(back_gpu, (int64_t)front_gpu_offset),
            boundary_byte_ptr(back_t, (int64_t)start * nvar * front_block),
            front_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(left_gpu, (int64_t)left_gpu_offset),
            boundary_byte_ptr(left_t, (int64_t)start * nvar * left_block),
            left_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );

        cudaMemcpyAsync(
            boundary_byte_ptr(right_gpu, (int64_t)left_gpu_offset),
            boundary_byte_ptr(right_t, (int64_t)start * nvar * left_block),
            left_elems * es,
            cudaMemcpyHostToDevice,
            stream
        );
    }

};
