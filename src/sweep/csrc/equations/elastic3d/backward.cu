#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "kernels.cuh"
#include "elastic3d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"

namespace elastic3d {

BackwardOutput backward_bs(const BackwardInput& in)
{

    cudaEvent_t start, stop, flush_start, flush_end;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&flush_start);
    cudaEventCreate(&flush_end);

    cudaEventRecord(start);

    const auto& p = in;
    BackwardOutput out;

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);

    int B = N * C;

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    
    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 3);

    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, false);
    else
        forward.allocate(vp, 3, false);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Copy last step of forward wavefield from u_last_two
    forward.vx_t.copy_(p.u_last_two.select(0,0).select(0,0));
    forward.vy_t.copy_(p.u_last_two.select(0,1).select(0,0));
    forward.vz_t.copy_(p.u_last_two.select(0,2).select(0,0));
    forward.sxx_t.copy_(p.u_last_two.select(0,3).select(0,0));
    forward.syy_t.copy_(p.u_last_two.select(0,4).select(0,0));
    forward.szz_t.copy_(p.u_last_two.select(0,5).select(0,0));
    forward.sxy_t.copy_(p.u_last_two.select(0,6).select(0,0));
    forward.sxz_t.copy_(p.u_last_two.select(0,7).select(0,0));
    forward.syz_t.copy_(p.u_last_two.select(0,8).select(0,0));

    auto neg_forward_source = -p.forward_source;

    // Generate pointer views
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    // // Set PML part to zero
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.vx, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.vy, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.vz, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.sxx, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.syy, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.szz, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.sxy, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.sxz, solver.abcn+0, nx, ny, nz);
    // set_boundary_zeros_3d<<<launch_config.grid, launch_config.block>>>(for_view.syz, solver.abcn+0, nx, ny, nz);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    // GeneralBoundarySaverMore boundary_saver;
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    // boundary_saver.allocate(true, 3, 9, solver, vp, save_width, 1, true, false, p.transfer_interval);
    // boundary_saver.load_from_vector(p.u_boundary, vp);
    boundary_saver.allocate(true, 3, 9, solver, vp, save_width, 1, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu);
    // auto bs = boundary_saver.view();

    auto fvx_prev = torch::zeros_like(vp);
    auto fvy_prev = torch::zeros_like(vp);
    auto fvz_prev = torch::zeros_like(vp);

    // auto u_all_t = torch::zeros({2, B, 1, nz, ny, nx}, vp.options());
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    int interval = p.transfer_interval;
    int buf_idx = 0;

    int gpu_idx = 0;

    // Copy the last block boundaries
    cudaEvent_t copy_done_event;
    cudaStream_t copy_stream;
    cudaStreamCreate(&copy_stream);
    cudaEventCreateWithFlags(&copy_done_event, cudaEventDisableTiming);

    // Pre-load the first block of boundaries to GPU
    int it0 = p.nt - 1;
    int buf_idx0 = (it0 - 1) % interval;

    int _start = it0 - buf_idx0 - 1;
    int _len   = buf_idx0 + 1;

    boundary_saver.load_cpu_to_gpu(_start, _len, copy_stream);
    cudaEventRecord(copy_done_event, copy_stream);

    cudaStream_t compute_stream = at::cuda::getCurrentCUDAStream();

    int next_start = 0;
    int next_len = 0;

    for (int it = p.nt - 1; it >= 1; --it) {

        buf_idx = (it - 1) % interval;

        // Adjoint modeling
        LAUNCH_3DELASTIC_VELOCITY_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        ); // t-0.5

        LAUNCH_3DELASTIC_STRESS_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        ); // t-1.0

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }
        // Wavefield reconstruction
        // Substract source term from forward wavefield
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 3, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }

        if (buf_idx == interval - 1)
            cudaStreamWaitEvent(compute_stream, copy_done_event, 0);
        
        LAUNCH_3DELASTIC_STRESS_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float *field2[6] = {for_view.sxx, for_view.syy, for_view.szz, for_view.sxy, for_view.sxz, for_view.syz};

        for (int f = 3; f < 9; ++f){
            gpu_idx = f * interval + buf_idx;
            boundary_kernel3d<<<launch_config.grid, launch_config.block>>>(
                field2[f-3],

                boundary_saver.top_gpu.data_ptr<float>()    + gpu_idx * boundary_saver.top_stride,
                boundary_saver.bottom_gpu.data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride,
                boundary_saver.front_gpu.data_ptr<float>()  + gpu_idx * boundary_saver.front_stride,
                boundary_saver.back_gpu.data_ptr<float>()   + gpu_idx * boundary_saver.back_stride,
                boundary_saver.left_gpu.data_ptr<float>()   + gpu_idx * boundary_saver.left_stride,
                boundary_saver.right_gpu.data_ptr<float>()  + gpu_idx * boundary_saver.right_stride,

                0,
                save_width,
                -p.M,
                solver,
                BOUNDARY_RESTORE
            );
        }
        // Gradient calculation
        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            adj_view,

            fvx_prev.data_ptr<float>(),
            fvy_prev.data_ptr<float>(),
            fvz_prev.data_ptr<float>(),

            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),

            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),

            grad_ctx,
            solver
        );

        fvz_prev.copy_(forward.vz_t);
        fvy_prev.copy_(forward.vy_t);
        fvx_prev.copy_(forward.vx_t);

        // Update Velocity components
        LAUNCH_3DELASTIC_VELOCITY_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            solver
        );
        
        float *field1[3] = {for_view.vx, for_view.vy, for_view.vz};

        for (int f = 0; f < 3; ++f){
            gpu_idx = f * interval + buf_idx;

            boundary_kernel3d<<<launch_config.grid, launch_config.block>>>(
                field1[f],
                
                boundary_saver.top_gpu.data_ptr<float>()    + gpu_idx * boundary_saver.top_stride,
                boundary_saver.bottom_gpu.data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride,
                boundary_saver.front_gpu.data_ptr<float>()  + gpu_idx * boundary_saver.front_stride,
                boundary_saver.back_gpu.data_ptr<float>()   + gpu_idx * boundary_saver.back_stride,
                boundary_saver.left_gpu.data_ptr<float>()   + gpu_idx * boundary_saver.left_stride,
                boundary_saver.right_gpu.data_ptr<float>()  + gpu_idx * boundary_saver.right_stride,

                0,
                save_width,
                -p.M,
                solver,
                BOUNDARY_RESTORE
            );
        }

        if (buf_idx == 0 && it > 1) {

            int chunk_id = (it - 1) / interval;

            int chunk_start = chunk_id * interval;

            int next_chunk = chunk_id - 1;

            if (next_chunk >= 0){

                int remain   = (int)p.nt - next_start;
                int next_len = std::min(interval, remain);
                int next_start = next_chunk * interval;                
                boundary_saver.load_cpu_to_gpu(next_start, next_len, copy_stream);
                cudaEventRecord(copy_done_event, copy_stream);
            }

        }
        // if (it == 15) u_all_t[0].copy_(forward.vz_t);
            
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

}
