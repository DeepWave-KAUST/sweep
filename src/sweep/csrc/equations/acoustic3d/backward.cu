#include <torch/extension.h>
#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../common/boundarysaver.cuh"
#include "../../launch/config.h"

namespace acoustic3d {

BackwardOutput backward(const BackwardInput& in)
{

    const auto& p = in;
    BackwardOutput out;

    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);

    int B     = N * C;

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);

    auto grad = torch::zeros_like(vp);

    float* u_thist = nullptr;

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();
    
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, true, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};


    for (int it = p.nt - 1; it >= 0; --it) {

        auto adj_view = adjoint.view();

        ACOUSTIC3D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            u_thist,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );
        
        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        // rotate pointers: u_prev <- u_now <- u_next
        adjoint.swap();

        calculate_grad_3d<<<launch_config.grid, launch_config.block>>>(
            p.u_forward[it].data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            B, nx, ny, nz
        );

    }

    out.grads = {grad};
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{

    const auto& p = in;
    BackwardOutput out;

    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    // Assign the last two wavefields from forward to u_prev and u_now
    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, nullptr, nullptr, dx, dy, dz};

    auto f_this = torch::zeros_like(vp); // for gradient calculation

    AcousticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, 3, true);
    else
        adjoint.allocate(vp, 3, true);
    AcousticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, 3, false);
    else
        forward.allocate(vp, 3, false);
    forward.u_prev_t.copy_(p.u_last_two.select(1,1).squeeze(0));
    forward.u_now_t.copy_(p.u_last_two.select(1,0).squeeze(0));
    
    auto grad = torch::zeros_like(vp);

    // For checking wavefields
    // torch::Tensor u_allt = torch::zeros({p.nt, B, 1, nz, ny, nx}, vp.options());

    // PML coefficients
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    // Boundary wavefields (for saving all wavefields)
    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
            {}, false, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            true, 3, 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, {}, false, p.use_pinned_memory
        );
        if (!p.boundary_gpu.empty()) {
            // Boundaries are already provided as external GPU buffers.
        } else {
            boundary_saver.load_from_vector(p.u_boundary, vp);
        }
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto for_view = forward.view();
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.u_now, ctx.p.abcn+ctx.p.M, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.u_prev, ctx.p.abcn+ctx.p.M, nx, ny, nz);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

    int interval = p.transfer_interval;
    int buf_idx = 0;
    int gpu_idx = 0;

    AsyncCopyContext async_copy(staged_boundary);
    if (staged_boundary) {
        int it0 = p.nt - 1;
        int buf_idx0 = (it0 - 1) % interval;
        int chunk_start = it0 - buf_idx0 - 1;
        int chunk_len = buf_idx0 + 1;

        boundary_saver.load_cpu_to_gpu(chunk_start, chunk_len, async_copy.copy_stream);
        async_copy.record_copy_ready();
    }


    for (int it = p.nt - 1; it >= 1; --it) {
        buf_idx = (it - 1) % interval;

        // u_allt[it].copy_(forward.u_now_t);

        auto adj_view = adjoint.view();
        auto for_view = forward.view();

        // adjoint modeling
        ACOUSTIC3D(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );
        
        add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            ctx
        );

        // rotate pointers: u_prev <- u_now <- u_next
        adjoint.swap();
        
        
        ACOUSTIC3D_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            f_this.data_ptr<float>(),
            vp.data_ptr<float>(),
            lap_ctx,
            ctx
        );

        add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            ctx
        );

        float* top_ptr = nullptr;
        float* bottom_ptr = nullptr;
        float* front_ptr = nullptr;
        float* back_ptr = nullptr;
        float* left_ptr = nullptr;
        float* right_ptr = nullptr;

        if (staged_boundary && buf_idx == interval - 1)
            async_copy.wait_for_copy();

        if (staged_boundary) {
            gpu_idx = buf_idx;
            top_ptr = boundary_saver.top_gpu.data_ptr<float>() + gpu_idx * boundary_saver.top_stride;
            bottom_ptr = boundary_saver.bottom_gpu.data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride;
            front_ptr = boundary_saver.front_gpu.data_ptr<float>() + gpu_idx * boundary_saver.front_stride;
            back_ptr = boundary_saver.back_gpu.data_ptr<float>() + gpu_idx * boundary_saver.back_stride;
            left_ptr = boundary_saver.left_gpu.data_ptr<float>() + gpu_idx * boundary_saver.left_stride;
            right_ptr = boundary_saver.right_gpu.data_ptr<float>() + gpu_idx * boundary_saver.right_stride;
        } else {
            top_ptr = bs.top;
            bottom_ptr = bs.bottom;
            front_ptr = bs.front;
            back_ptr = bs.back;
            left_ptr = bs.left;
            right_ptr = bs.right;
        }

        boundary_kernel3d<<<launch_config.grid, launch_config.block>>>(
            for_view.u_next,
            top_ptr,
            bottom_ptr,
            front_ptr,
            back_ptr,
            left_ptr,
            right_ptr,
            staged_boundary ? 0 : it - 1,
            save_width,
            0,
            ctx,
            BOUNDARY_RESTORE
        );        
        // rotate pointers for forward wavefields
        forward.swap();

        if (staged_boundary && buf_idx == 0 && it > 1) {
            int chunk_id = (it - 1) / interval;
            int next_chunk = chunk_id - 1;

            if (next_chunk >= 0) {
                int next_start = next_chunk * interval;
                int remain = static_cast<int>(p.nt) - next_start;
                int next_len = std::min(interval, remain);

                boundary_saver.load_cpu_to_gpu(next_start, next_len, async_copy.copy_stream);
                async_copy.record_copy_ready();
            }
        }

        calculate_grad_utt_3d<<<launch_config.grid, launch_config.block>>>(
            forward.u_next_t.data_ptr<float>(),
            forward.u_now_t.data_ptr<float>(),
            forward.u_prev_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            grad.data_ptr<float>(),
            B, nx, ny, nz, p.dt
        );

    }

    out.grads = {grad};
    return out;

}

}
