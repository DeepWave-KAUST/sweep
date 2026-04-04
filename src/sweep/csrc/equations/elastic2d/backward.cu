#include <torch/extension.h>

#include "kernels.cuh"
#include "elastic2d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace elastic2d {

BackwardOutput backward(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, false, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto f_this = torch::zeros_like(vp); // for gradient calculation
    
    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Generate pointer views
    auto adj_view = adjoint.view();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    int t2=0;

    for (int it = p.nt - 1; it >= 1; --it) {

        LAUNCH_ELASTIC_VELOCITY_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_STRESS_ADJOINT(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<source_config.grid, source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }
        
        if (it+1>=p.nt) t2=it; else t2=it+1;

        LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
            order,
            launch_config.grid,
            launch_config.block,

            adj_view,

            p.u_forward.select(0, it).select(0, 0).data_ptr<float>(), // vx
            p.u_forward.select(0, it).select(0, 1).data_ptr<float>(), // vz
            
            p.u_forward.select(0, t2).select(0, 0).data_ptr<float>(), // vx
            p.u_forward.select(0, t2).select(0, 1).data_ptr<float>(), // vz

            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),

            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),

            grad_ctx,
            solver
        );

    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}


BackwardOutput backward_bs(const BackwardInput& in)
{

    const auto& p = in;
    BackwardOutput out;

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto f_this = torch::zeros_like(vp); // for gradient calculation
    
    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2);
    ElasticWavefieldTensor forward;

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Copy last step of forward wavefield from u_last_two
    forward.allocate(vp, 2, false);
    forward.vx_t.copy_(p.u_last_two.select(0,0).select(0,0));
    forward.vz_t.copy_(p.u_last_two.select(0,1).select(0,0));
    forward.sxx_t.copy_(p.u_last_two.select(0,2).select(0,0));
    forward.szz_t.copy_(p.u_last_two.select(0,3).select(0,0));
    forward.sxz_t.copy_(p.u_last_two.select(0,4).select(0,0));

    auto neg_forward_source = -p.forward_source;

    // Generate pointer views
    auto for_view = forward.view();
    auto adj_view = adjoint.view();
    
    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();


    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, false, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, true, 1, {}, p.boundary_gpu, {}, false, p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    // Set boundarys of the last frame to be zeors
    // set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.vx, solver.abcn+solver.M, nx, nz);
    // set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.vz, solver.abcn+solver.M, nx, nz);
    // set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.szz, solver.abcn+solver.M, nx, nz);
    // set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.sxx, solver.abcn+solver.M, nx, nz);
    // set_boundary_zeros<<<launch_config.grid, launch_config.block>>>(for_view.sxz, solver.abcn+solver.M, nx, nz);

    auto fvz_prev = torch::zeros_like(vp);
    auto fvx_prev = torch::zeros_like(vp);
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

    // auto u_all_for = torch::zeros({nt, B, 1, nz, nx}, vp.options());
    // auto u_all_adj = torch::zeros({nt, B, 1, nz, nx}, vp.options());

    for (int it = p.nt - 1; it >= 1; --it) {
        buf_idx = (it - 1) % interval;
        
        LAUNCH_ELASTIC_VELOCITY_ADJOINT(
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

        LAUNCH_ELASTIC_STRESS_ADJOINT(
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
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        // Wavefield reconstruction
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }
        // Update Stress components
        LAUNCH_ELASTIC_STRESS_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float *field2[3] = {for_view.sxx, for_view.szz, for_view.sxz};

        if (staged_boundary && buf_idx == interval - 1)
            async_copy.wait_for_copy();

        for (int f = 2; f < 5; ++f) {
            float* top_ptr = nullptr;
            float* bottom_ptr = nullptr;
            float* left_ptr = nullptr;
            float* right_ptr = nullptr;

            if (staged_boundary) {
                gpu_idx = f * interval + buf_idx;
                top_ptr = boundary_saver.top_gpu.data_ptr<float>() + gpu_idx * boundary_saver.top_stride;
                bottom_ptr = boundary_saver.bottom_gpu.data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride;
                left_ptr = boundary_saver.left_gpu.data_ptr<float>() + gpu_idx * boundary_saver.left_stride;
                right_ptr = boundary_saver.right_gpu.data_ptr<float>() + gpu_idx * boundary_saver.right_stride;
            } else {
                top_ptr = boundary_saver.top_t[f].data_ptr<float>();
                bottom_ptr = boundary_saver.bottom_t[f].data_ptr<float>();
                left_ptr = boundary_saver.left_t[f].data_ptr<float>();
                right_ptr = boundary_saver.right_t[f].data_ptr<float>();
            }

            boundary_kernel2d<<<launch_config.grid, launch_config.block>>>(
                field2[f-2],
                top_ptr,
                bottom_ptr,
                left_ptr,
                right_ptr,
                staged_boundary ? 0 : it-1,
                save_width,
                -p.M,
                solver,
                BOUNDARY_RESTORE
            );
        }

        // Gradient calculation
        LAUNCH_CALCULATE_GRAD_ELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            adj_view,

            fvx_prev.data_ptr<float>(),
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
        fvx_prev.copy_(forward.vx_t);

        // Update Velocity components
        LAUNCH_ELASTIC_VELOCITY_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float *field1[2] = {for_view.vx, for_view.vz};

        for (int f = 0; f < 2; ++f) {
            float* top_ptr = nullptr;
            float* bottom_ptr = nullptr;
            float* left_ptr = nullptr;
            float* right_ptr = nullptr;

            if (staged_boundary) {
                gpu_idx = f * interval + buf_idx;
                top_ptr = boundary_saver.top_gpu.data_ptr<float>() + gpu_idx * boundary_saver.top_stride;
                bottom_ptr = boundary_saver.bottom_gpu.data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride;
                left_ptr = boundary_saver.left_gpu.data_ptr<float>() + gpu_idx * boundary_saver.left_stride;
                right_ptr = boundary_saver.right_gpu.data_ptr<float>() + gpu_idx * boundary_saver.right_stride;
            } else {
                top_ptr = boundary_saver.top_t[f].data_ptr<float>();
                bottom_ptr = boundary_saver.bottom_t[f].data_ptr<float>();
                left_ptr = boundary_saver.left_t[f].data_ptr<float>();
                right_ptr = boundary_saver.right_t[f].data_ptr<float>();
            }

            boundary_kernel2d<<<launch_config.grid, launch_config.block>>>(
                field1[f],
                top_ptr,
                bottom_ptr,
                left_ptr,
                right_ptr,
                staged_boundary ? 0 : it-1,
                save_width,
                -p.M,
                solver,
                BOUNDARY_RESTORE
            );
        }

        if (staged_boundary && buf_idx == 0 && it > 1) {
            int next_chunk = (it - 1) / interval - 1;
            if (next_chunk >= 0) {
                int next_start = next_chunk * interval;
                int remain = (int)p.nt - next_start - 1;
                int next_len = remain < interval ? remain : interval;
                boundary_saver.load_cpu_to_gpu(next_start, next_len, async_copy.copy_stream);
                async_copy.record_copy_ready();
            }
        }

        // u_all_for[it].copy_(forward.vz_t); // for visualization
        // u_all_adj[it].copy_(adjoint.vz_t); // for visualization
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;

}

}
