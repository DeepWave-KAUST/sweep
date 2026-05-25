#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "kernels.cuh"
#include "elastic3d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"

namespace elastic3d {

namespace {

ElasticWavefieldTensor make_velocity_view_3d(
    const torch::Tensor& vx,
    const torch::Tensor& vy,
    const torch::Tensor& vz
)
{
    ElasticWavefieldTensor view;
    view.dim = 3;
    view.use_pml = false;
    view.allocated = true;
    view.vx_t = vx;
    view.vy_t = vy;
    view.vz_t = vz;
    view.sxx_t = vx;
    view.syy_t = vx;
    view.szz_t = vx;
    view.sxy_t = vx;
    view.sxz_t = vx;
    view.syz_t = vx;
    return view;
}

int find_previous_checkpoint_idx_3d(
    const int* checkpoint_steps,
    int num_saved_checkpoints,
    int target_time
)
{
    int checkpoint_idx = -1;
    for (int i = 0; i < num_saved_checkpoints; ++i) {
        if (checkpoint_steps[i] < target_time)
            checkpoint_idx = i;
        else
            break;
    }
    return checkpoint_idx;
}

void replay_forward_to_time_3d(
    const BackwardInput& p,
    ElasticWavefieldTensor& forward,
    torch::Tensor& current_vx,
    torch::Tensor& current_vy,
    torch::Tensor& current_vz,
    torch::Tensor& next_vx,
    torch::Tensor& next_vy,
    torch::Tensor& next_vz,
    int target_index,
    const int* checkpoint_steps,
    int num_saved_checkpoints,
    CheckpointRuntime& checkpoint_runtime,
    int order,
    const fdtd::LaunchConfig& launch_config,
    const fdtd::LaunchConfig& fwd_source_config,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml_view,
    SolverContext solver,
    const torch::Tensor& source_fields,
    const torch::Tensor& lambda,
    const torch::Tensor& mu,
    const torch::Tensor& rho
)
{
    current_vx.zero_();
    current_vy.zero_();
    current_vz.zero_();
    next_vx.zero_();
    next_vy.zero_();
    next_vz.zero_();

    const int checkpoint_idx = find_previous_checkpoint_idx_3d(checkpoint_steps, num_saved_checkpoints, target_index + 1);
    int start_time = 0;
    if (checkpoint_idx >= 0) {
        checkpoint_runtime.load(checkpoint_idx, forward.checkpoint_tensors());
        start_time = checkpoint_steps[checkpoint_idx];
    } else {
        checkpoint_runtime.zero_state(forward.state_tensors());
    }

    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();

    for (int it = start_time; it < p.nt; ++it) {
        auto for_view = forward.view();

        LAUNCH_3DELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_3DELASTIC_STRESS(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            nullptr,
            grad_ctx,
            cpml_view,
            solver
        );

        if (it == target_index) {
            current_vx.copy_(forward.vx_t);
            current_vy.copy_(forward.vy_t);
            current_vz.copy_(forward.vz_t);
        }
        if (it == target_index + 1) {
            next_vx.copy_(forward.vx_t);
            next_vy.copy_(forward.vy_t);
            next_vz.copy_(forward.vz_t);
            break;
        }

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 3, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }

        if (it == target_index && target_index + 1 >= p.nt) {
            break;
        }
    }
}

void apply_adjoint_step_3d(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    const torch::Tensor& lambda,
    const torch::Tensor& mu,
    const torch::Tensor& rho,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    ElasticAdjointWorkspaceTensor& workspace
)
{
    auto adj_view = adjoint.view();

    LAUNCH_3DELASTIC_STRESS_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        lambda.data_ptr<float>(),
        mu.data_ptr<float>(),
        cpml_view,
        solver,
        workspace.qxx_t.data_ptr<float>(),
        workspace.qxy_t.data_ptr<float>(),
        workspace.qxz_t.data_ptr<float>(),
        workspace.qyx_t.data_ptr<float>(),
        workspace.qyy_t.data_ptr<float>(),
        workspace.qyz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>(),
        workspace.qzy_t.data_ptr<float>(),
        workspace.qzz_t.data_ptr<float>()
    );

    LAUNCH_3DELASTIC_STRESS_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        workspace.qxx_t.data_ptr<float>(),
        workspace.qxy_t.data_ptr<float>(),
        workspace.qxz_t.data_ptr<float>(),
        workspace.qyx_t.data_ptr<float>(),
        workspace.qyy_t.data_ptr<float>(),
        workspace.qyz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>(),
        workspace.qzy_t.data_ptr<float>(),
        workspace.qzz_t.data_ptr<float>(),
        grad_ctx,
        solver
    );

    LAUNCH_3DELASTIC_VELOCITY_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        rho.data_ptr<float>(),
        cpml_view,
        solver,
        workspace.pxx_t.data_ptr<float>(),
        workspace.pxy_t.data_ptr<float>(),
        workspace.pxz_t.data_ptr<float>(),
        workspace.pyx_t.data_ptr<float>(),
        workspace.pyy_t.data_ptr<float>(),
        workspace.pyz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(),
        workspace.pzy_t.data_ptr<float>(),
        workspace.pzz_t.data_ptr<float>()
    );

    LAUNCH_3DELASTIC_VELOCITY_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        workspace.pxx_t.data_ptr<float>(),
        workspace.pxy_t.data_ptr<float>(),
        workspace.pxz_t.data_ptr<float>(),
        workspace.pyx_t.data_ptr<float>(),
        workspace.pyy_t.data_ptr<float>(),
        workspace.pyz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(),
        workspace.pzy_t.data_ptr<float>(),
        workspace.pzz_t.data_ptr<float>(),
        grad_ctx,
        solver
    );
}

void backward_segment_3d(
    const BackwardInput& p,
    const torch::Tensor& vp,
    const torch::Tensor& vs,
    const torch::Tensor& rho,
    ElasticWavefieldTensor& start_state,
    ElasticWavefieldTensor& adjoint,
    CheckpointRuntime& checkpoint_runtime,
    int start,
    int end,
    int order,
    const fdtd::LaunchConfig& launch_config,
    const fdtd::LaunchConfig& fwd_source_config,
    const fdtd::LaunchConfig& adj_source_config,
    SGradParam grad_ctx,
    ElasticCPMLPointer cpml_view,
    SolverContext solver,
    const torch::Tensor& source_fields,
    const torch::Tensor& receiver_fields,
    ElasticAdjointWorkspaceTensor& workspace,
    const torch::Tensor& next_segment_vx,
    const torch::Tensor& next_segment_vy,
    const torch::Tensor& next_segment_vz,
    torch::Tensor& grad_vp,
    torch::Tensor& grad_vs,
    torch::Tensor& grad_rho,
    torch::Tensor& prev_segment_next_vx,
    torch::Tensor& prev_segment_next_vy,
    torch::Tensor& prev_segment_next_vz
)
{
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    const int nrec_fields = p.receiver_field_indices.numel();
    const int segment_len = end - start;
    const int B = vp.size(0) * vp.size(1);

    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto mu = rho * vs * vs;

    auto seg_vx = torch::zeros({segment_len + 1, B, 1, vp.size(2), vp.size(3), vp.size(4)}, vp.options());
    auto seg_vy = torch::zeros_like(seg_vx);
    auto seg_vz = torch::zeros_like(seg_vx);

    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(vp, 3, true);

    checkpoint_runtime.copy_state(forward.state_tensors(), start_state.state_tensors());

    seg_vx.select(0, 0).copy_(forward.vx_t);
    seg_vy.select(0, 0).copy_(forward.vy_t);
    seg_vz.select(0, 0).copy_(forward.vz_t);

    for (int it = start; it < end; ++it) {
        auto for_view = forward.view();

        LAUNCH_3DELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_3DELASTIC_STRESS(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            nullptr,
            grad_ctx,
            cpml_view,
            solver
        );

        seg_vx.select(0, it - start + 1).copy_(forward.vx_t);
        seg_vy.select(0, it - start + 1).copy_(forward.vy_t);
        seg_vz.select(0, it - start + 1).copy_(forward.vz_t);

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 3, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }
    }

    for (int it = end - 1; it >= start; --it) {
        auto adj_view = adjoint.view();

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

        const int offset = it - start;
        const int now_offset = offset + 1;
        const int next_offset = now_offset + 1;
        auto current_forward = make_velocity_view_3d(
            seg_vx.select(0, now_offset),
            seg_vy.select(0, now_offset),
            seg_vz.select(0, now_offset)
        );

        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            current_forward.view(),
            adj_view,
            (next_offset <= segment_len) ? seg_vx.select(0, next_offset).data_ptr<float>() : next_segment_vx.data_ptr<float>(),
            (next_offset <= segment_len) ? seg_vy.select(0, next_offset).data_ptr<float>() : next_segment_vy.data_ptr<float>(),
            (next_offset <= segment_len) ? seg_vz.select(0, next_offset).data_ptr<float>() : next_segment_vz.data_ptr<float>(),
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        if (it == 0) {
            continue;
        }

        apply_adjoint_step_3d(
            order,
            launch_config,
            adjoint,
            lambda,
            mu,
            rho,
            cpml_view,
            grad_ctx,
            solver,
            workspace
        );
    }

    prev_segment_next_vx.copy_(seg_vx.select(0, 1));
    prev_segment_next_vy.copy_(seg_vy.select(0, 1));
    prev_segment_next_vz.copy_(seg_vz.select(0, 1));
}

} // namespace

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

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
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    // GeneralBoundarySaverMore boundary_saver;
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    boundary_saver.allocate(
        true, 3, 9, solver, vp, save_width, 1,
        true, !staged_boundary, staged_boundary ? p.transfer_interval : 1,
        staged_boundary ? p.boundary_cpu : std::vector<torch::Tensor>{},
        p.boundary_gpu,
        {}, p.use_pinned_memory
    );
    auto bs = boundary_saver.view();

    auto fvx_prev = torch::zeros_like(vp);
    auto fvy_prev = torch::zeros_like(vp);
    auto fvz_prev = torch::zeros_like(vp);

    // auto u_all_t = torch::zeros({2, B, 1, nz, ny, nx}, vp.options());
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    // Copy the last block boundaries
    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        3,
        true,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    for (int it = p.nt - 1; it >= 1; --it) {

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
            boundary_runtime.restore_backward_3d_field(
                it,
                field2[f-3],
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                -p.M,
                solver,
                f,
                f == 3,
                false
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

        apply_adjoint_step_3d(
            order,
            launch_config,
            adjoint,
            lambda,
            mu,
            rho,
            cpml_view,
            grad_ctx,
            solver,
            workspace
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
            boundary_runtime.restore_backward_3d_field(
                it,
                field1[f],
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                -p.M,
                solver,
                f,
                false,
                f == 2
            );
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
        // if (it == 15) u_all_t[0].copy_(forward.vz_t);
            
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
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
    int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 3);
    zero_wavefield_state(adjoint);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto adj_view = adjoint.view();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    auto zero_velocity = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    for (int it = p.nt - 1; it >= 0; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        auto current_forward = make_velocity_view_3d(
            p.u_forward.select(0, it).select(0, 0),
            p.u_forward.select(0, it).select(0, 1),
            p.u_forward.select(0, it).select(0, 2)
        );

        const float* vx_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vy_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vz_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 2).data_ptr<float>() : zero_velocity.data_ptr<float>();

        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            current_forward.view(),
            adj_view,
            vx_prev,
            vy_prev,
            vz_prev,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        if (it == 0) {
            continue;
        }

        apply_adjoint_step_3d(
            order,
            launch_config,
            adjoint,
            lambda,
            mu,
            rho,
            cpml_view,
            grad_ctx,
            solver,
            workspace
        );
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;

    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 36, "Elastic 3D checkpointing expects 36 checkpoint tensors");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        36,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "elastic3d"
    );

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 3, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto mu = rho * vs * vs;

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    ElasticWavefieldTensor start_state;
    start_state.allocate(vp, 3, true);
    if (!start_state.m_syzx_t.defined()) start_state.m_syzx_t = torch::zeros_like(vp);
    auto next_segment_vx = torch::zeros_like(vp);
    auto next_segment_vy = torch::zeros_like(vp);
    auto next_segment_vz = torch::zeros_like(vp);
    auto prev_segment_next_vx = torch::zeros_like(vp);
    auto prev_segment_next_vy = torch::zeros_like(vp);
    auto prev_segment_next_vz = torch::zeros_like(vp);

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);
        if (chunk_id == 0) {
            checkpoint_runtime.zero_state(start_state.state_tensors());
        } else {
            checkpoint_runtime.load(chunk_id, start_state.checkpoint_tensors());
        }
        backward_segment_3d(
            p,
            vp,
            vs,
            rho,
            start_state,
            adjoint,
            checkpoint_runtime,
            start,
            end,
            order,
            launch_config,
            fwd_source_config,
            adj_source_config,
            grad_ctx,
            cpml_view,
            solver,
            source_fields,
            receiver_fields,
            workspace,
            next_segment_vx,
            next_segment_vy,
            next_segment_vz,
            grad_vp,
            grad_vs,
            grad_rho,
            prev_segment_next_vx,
            prev_segment_next_vy,
            prev_segment_next_vz
        );
        next_segment_vx.copy_(prev_segment_next_vx);
        next_segment_vy.copy_(prev_segment_next_vy);
        next_segment_vz.copy_(prev_segment_next_vz);
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;

    TORCH_CHECK(p.checkpoints.size() == 36, "Elastic 3D recursive checkpointing expects 36 checkpoint tensors");

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        36,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "elastic3d"
    );

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 3, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto mu = rho * vs * vs;

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    TORCH_CHECK(
        p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
        "checkpoint_count does not match checkpoint_steps"
    );
    TORCH_CHECK(
        static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
        "checkpoint buffer is smaller than checkpoint_steps"
    );

    const int* checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();
    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(vp, 3, true);
    if (!forward.m_syzx_t.defined()) forward.m_syzx_t = torch::zeros_like(vp);
    auto current_vx = torch::zeros_like(vp);
    auto current_vy = torch::zeros_like(vp);
    auto current_vz = torch::zeros_like(vp);
    auto next_vx = torch::zeros_like(vp);
    auto next_vy = torch::zeros_like(vp);
    auto next_vz = torch::zeros_like(vp);

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        for (int irec = 0; irec < static_cast<int>(p.receiver_field_indices.numel()); ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                p.adjoint_sources_loc.size(1),
                solver
            );
        }

        replay_forward_to_time_3d(
            p,
            forward,
            current_vx,
            current_vy,
            current_vz,
            next_vx,
            next_vy,
            next_vz,
            it,
            checkpoint_steps,
            num_saved_checkpoints,
            checkpoint_runtime,
            order,
            launch_config,
            fwd_source_config,
            grad_ctx,
            cpml_view,
            solver,
            source_fields,
            lambda,
            mu,
            rho
        );

        auto current_forward = make_velocity_view_3d(current_vx, current_vy, current_vz);

        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            current_forward.view(),
            adj_view,
            next_vx.data_ptr<float>(),
            next_vy.data_ptr<float>(),
            next_vz.data_ptr<float>(),
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        if (it == 0)
            continue;

        apply_adjoint_step_3d(
            order,
            launch_config,
            adjoint,
            lambda,
            mu,
            rho,
            cpml_view,
            grad_ctx,
            solver,
            workspace
        );
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}


// ===========================================================================
// APM (Dong 2023, elastic limit) 3-D backward — full + boundary-saving.
// ===========================================================================
// Expects the 21-tensor APM 3-D model layout assembled by _c.py:
//   models = [vp, vs, rho, lam, mu, lam_2mu,
//             alpha_xx, alpha_yy, alpha_zz,
//             lam_xx_yy, lam_xx_zz, lam_yy_xx, lam_yy_zz,
//             lam_zz_xx, lam_zz_yy,
//             mu_xy, mu_xz, mu_yz,
//             inv_rho_x, inv_rho_y, inv_rho_z]
// Returns 21 grad tensors — positions 0..2 (vp, vs, rho) are
// chain-ruled inside the gradient kernel; positions 3..20 are zero.

static inline void apm3d_apply_adjoint_step(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    const torch::Tensor& alpha_xx,
    const torch::Tensor& alpha_yy,
    const torch::Tensor& alpha_zz,
    const torch::Tensor& lam_xx_yy,
    const torch::Tensor& lam_xx_zz,
    const torch::Tensor& lam_yy_xx,
    const torch::Tensor& lam_yy_zz,
    const torch::Tensor& lam_zz_xx,
    const torch::Tensor& lam_zz_yy,
    const torch::Tensor& mu_xy,
    const torch::Tensor& mu_xz,
    const torch::Tensor& mu_yz,
    const torch::Tensor& inv_rho_x,
    const torch::Tensor& inv_rho_y,
    const torch::Tensor& inv_rho_z,
    const torch::Tensor& category,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    ElasticAdjointWorkspaceTensor& workspace
)
{
    auto adj_view = adjoint.view();

    LAUNCH_3DELASTIC_STRESS_ADJOINT_PREPARE_APM(
        order, launch_config.grid, launch_config.block, adj_view,
        alpha_xx.data_ptr<float>(), alpha_yy.data_ptr<float>(), alpha_zz.data_ptr<float>(),
        lam_xx_yy.data_ptr<float>(), lam_xx_zz.data_ptr<float>(),
        lam_yy_xx.data_ptr<float>(), lam_yy_zz.data_ptr<float>(),
        lam_zz_xx.data_ptr<float>(), lam_zz_yy.data_ptr<float>(),
        mu_xy.data_ptr<float>(), mu_xz.data_ptr<float>(), mu_yz.data_ptr<float>(),
        category.data_ptr<int>(), cpml_view, solver,
        workspace.qxx_t.data_ptr<float>(), workspace.qxy_t.data_ptr<float>(), workspace.qxz_t.data_ptr<float>(),
        workspace.qyx_t.data_ptr<float>(), workspace.qyy_t.data_ptr<float>(), workspace.qyz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>(), workspace.qzy_t.data_ptr<float>(), workspace.qzz_t.data_ptr<float>()
    );

    LAUNCH_3DELASTIC_STRESS_ADJOINT_APPLY(
        order, launch_config.grid, launch_config.block, adj_view,
        workspace.qxx_t.data_ptr<float>(), workspace.qxy_t.data_ptr<float>(), workspace.qxz_t.data_ptr<float>(),
        workspace.qyx_t.data_ptr<float>(), workspace.qyy_t.data_ptr<float>(), workspace.qyz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>(), workspace.qzy_t.data_ptr<float>(), workspace.qzz_t.data_ptr<float>(),
        grad_ctx, solver
    );

    LAUNCH_3DELASTIC_VELOCITY_ADJOINT_PREPARE_APM(
        order, launch_config.grid, launch_config.block, adj_view,
        inv_rho_x.data_ptr<float>(), inv_rho_y.data_ptr<float>(), inv_rho_z.data_ptr<float>(),
        category.data_ptr<int>(), cpml_view, solver,
        workspace.pxx_t.data_ptr<float>(), workspace.pxy_t.data_ptr<float>(), workspace.pxz_t.data_ptr<float>(),
        workspace.pyx_t.data_ptr<float>(), workspace.pyy_t.data_ptr<float>(), workspace.pyz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(), workspace.pzy_t.data_ptr<float>(), workspace.pzz_t.data_ptr<float>()
    );

    LAUNCH_3DELASTIC_VELOCITY_ADJOINT_APPLY(
        order, launch_config.grid, launch_config.block, adj_view,
        workspace.pxx_t.data_ptr<float>(), workspace.pxy_t.data_ptr<float>(), workspace.pxz_t.data_ptr<float>(),
        workspace.pyx_t.data_ptr<float>(), workspace.pyy_t.data_ptr<float>(), workspace.pyz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(), workspace.pzy_t.data_ptr<float>(), workspace.pzz_t.data_ptr<float>(),
        grad_ctx, solver
    );
}


BackwardOutput apm_backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() >= 21,
        "elastic3d::apm_backward expects 21-tensor models list; got ",
        p.models.size());
    TORCH_CHECK(p.use_apm && p.topo_category.defined() && p.topo_category.numel() > 0,
        "apm_backward requires use_apm=true and topo_category tensor");

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    auto vp        = p.models[0];
    auto vs        = p.models[1];
    auto rho       = p.models[2];
    auto lam_raw   = p.models[3];
    auto mu_raw    = p.models[4];
    auto alpha_xx  = p.models[6];
    auto alpha_yy  = p.models[7];
    auto alpha_zz  = p.models[8];
    auto lam_xx_yy = p.models[9];
    auto lam_xx_zz = p.models[10];
    auto lam_yy_xx = p.models[11];
    auto lam_yy_zz = p.models[12];
    auto lam_zz_xx = p.models[13];
    auto lam_zz_yy = p.models[14];
    auto mu_xy     = p.models[15];
    auto mu_xz     = p.models[16];
    auto mu_yz     = p.models[17];
    auto inv_rho_x = p.models[18];
    auto inv_rho_y = p.models[19];
    auto inv_rho_z = p.models[20];

    int N = vp.size(0), C = vp.size(1);
    int nz = vp.size(2), ny = vp.size(3), nx = vp.size(4);
    int B = N * C;
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn,
                         /*free_surface=*/false,
                         p.lap_coes.data_ptr<float>(),
                         p.grad_coes.data_ptr<float>(), dx, dy, dz};
    solver.topo_category = p.topo_category.data_ptr<int>();
    solver.use_apm = true;

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 3);
    zero_wavefield_state(adjoint);
    auto adj_view = adjoint.view();

    auto grad_vp  = torch::zeros_like(vp);
    auto grad_vs  = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    auto zero_velocity = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    for (int it = p.nt - 1; it >= 0; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(), it, adjoint_nsrc, solver
            );
        }

        auto current_forward = make_velocity_view_3d(
            p.u_forward.select(0, it).select(0, 0),
            p.u_forward.select(0, it).select(0, 1),
            p.u_forward.select(0, it).select(0, 2)
        );
        const float* vx_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vy_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vz_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 2).data_ptr<float>() : zero_velocity.data_ptr<float>();

        LAUNCH_CALCULATE_GRAD_3DELASTIC_APM_BS(
            order, launch_config.grid, launch_config.block,
            current_forward.view(), adj_view,
            vx_prev, vy_prev, vz_prev,
            vp.data_ptr<float>(), vs.data_ptr<float>(), rho.data_ptr<float>(),
            lam_raw.data_ptr<float>(), mu_raw.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(), grad_rho.data_ptr<float>(),
            grad_ctx, solver
        );

        if (it == 0) continue;

        apm3d_apply_adjoint_step(
            order, launch_config, adjoint,
            alpha_xx, alpha_yy, alpha_zz,
            lam_xx_yy, lam_xx_zz, lam_yy_xx, lam_yy_zz, lam_zz_xx, lam_zz_yy,
            mu_xy, mu_xz, mu_yz,
            inv_rho_x, inv_rho_y, inv_rho_z,
            p.topo_category,
            cpml_view, grad_ctx, solver, workspace
        );
    }

    auto z = torch::zeros_like(vp);
    // Return 21 grads matching the 21-tensor APM model layout.
    out.grads = {grad_vp, grad_vs, grad_rho,
                 z, z, z,
                 z, z, z, z, z, z, z, z, z,
                 z, z, z,
                 z, z, z};
    return out;
}


BackwardOutput apm_backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() >= 21,
        "elastic3d::apm_backward_bs expects 21-tensor models list; got ",
        p.models.size());
    TORCH_CHECK(p.use_apm && p.topo_category.defined() && p.topo_category.numel() > 0,
        "apm_backward_bs requires use_apm=true and topo_category tensor");

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    auto vp        = p.models[0];
    auto vs        = p.models[1];
    auto rho       = p.models[2];
    auto lam_raw   = p.models[3];
    auto mu_raw    = p.models[4];
    auto alpha_xx  = p.models[6];
    auto alpha_yy  = p.models[7];
    auto alpha_zz  = p.models[8];
    auto lam_xx_yy = p.models[9];
    auto lam_xx_zz = p.models[10];
    auto lam_yy_xx = p.models[11];
    auto lam_yy_zz = p.models[12];
    auto lam_zz_xx = p.models[13];
    auto lam_zz_yy = p.models[14];
    auto mu_xy     = p.models[15];
    auto mu_xz     = p.models[16];
    auto mu_yz     = p.models[17];
    auto inv_rho_x = p.models[18];
    auto inv_rho_y = p.models[19];
    auto inv_rho_z = p.models[20];

    int N = vp.size(0), C = vp.size(1);
    int nz = vp.size(2), ny = vp.size(3), nx = vp.size(4);
    int B = N * C;
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn,
                         /*free_surface=*/false,
                         p.lap_coes.data_ptr<float>(),
                         p.grad_coes.data_ptr<float>(), dx, dy, dz};
    solver.topo_category = p.topo_category.data_ptr<int>();
    solver.use_apm = true;

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
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    auto grad_vp  = torch::zeros_like(vp);
    auto grad_vs  = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    boundary_saver.allocate(
        true, 3, 9, solver, vp, save_width, 1,
        true, !staged_boundary, staged_boundary ? p.transfer_interval : 1,
        staged_boundary ? p.boundary_cpu : std::vector<torch::Tensor>{},
        p.boundary_gpu, {}, p.use_pinned_memory
    );
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto fvx_prev = torch::zeros_like(vp);
    auto fvy_prev = torch::zeros_like(vp);
    auto fvz_prev = torch::zeros_like(vp);
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver, 3, true,
        p.boundary_on_cpu, p.boundary_on_disk, p.boundary_disk_async_read,
        p.transfer_interval, p.boundary_ring_buffers, p.boundary_disk_files,
        async_copy.compute_stream, async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    for (int it = p.nt - 1; it >= 1; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(), it, adjoint_nsrc, solver
            );
        }

        // Reverse forward replay: -source then reverse stress step.
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 3, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field, neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(), it, forward_nsrc, solver
            );
        }

        LAUNCH_3DELASTIC_STRESS_NOPML_APM(
            order, launch_config.grid, launch_config.block,
            for_view,
            alpha_xx.data_ptr<float>(), alpha_yy.data_ptr<float>(), alpha_zz.data_ptr<float>(),
            lam_xx_yy.data_ptr<float>(), lam_xx_zz.data_ptr<float>(),
            lam_yy_xx.data_ptr<float>(), lam_yy_zz.data_ptr<float>(),
            lam_zz_xx.data_ptr<float>(), lam_zz_yy.data_ptr<float>(),
            mu_xy.data_ptr<float>(), mu_xz.data_ptr<float>(), mu_yz.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_ctx, solver
        );

        float* field2[6] = {for_view.sxx, for_view.syy, for_view.szz,
                            for_view.sxy, for_view.sxz, for_view.syz};
        for (int f = 3; f < 9; ++f) {
            boundary_runtime.restore_backward_3d_field(
                it, field2[f-3], launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver, f, f == 3, false
            );
        }

        LAUNCH_CALCULATE_GRAD_3DELASTIC_APM_BS(
            order, launch_config.grid, launch_config.block,
            for_view, adj_view,
            fvx_prev.data_ptr<float>(), fvy_prev.data_ptr<float>(), fvz_prev.data_ptr<float>(),
            vp.data_ptr<float>(), vs.data_ptr<float>(), rho.data_ptr<float>(),
            lam_raw.data_ptr<float>(), mu_raw.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(), grad_rho.data_ptr<float>(),
            grad_ctx, solver
        );

        apm3d_apply_adjoint_step(
            order, launch_config, adjoint,
            alpha_xx, alpha_yy, alpha_zz,
            lam_xx_yy, lam_xx_zz, lam_yy_xx, lam_yy_zz, lam_zz_xx, lam_zz_yy,
            mu_xy, mu_xz, mu_yz,
            inv_rho_x, inv_rho_y, inv_rho_z,
            p.topo_category,
            cpml_view, grad_ctx, solver, workspace
        );

        fvz_prev.copy_(forward.vz_t);
        fvy_prev.copy_(forward.vy_t);
        fvx_prev.copy_(forward.vx_t);

        LAUNCH_3DELASTIC_VELOCITY_NOPML_APM(
            order, launch_config.grid, launch_config.block,
            for_view,
            inv_rho_x.data_ptr<float>(), inv_rho_y.data_ptr<float>(), inv_rho_z.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_ctx, solver
        );

        float* field1[3] = {for_view.vx, for_view.vy, for_view.vz};
        for (int f = 0; f < 3; ++f) {
            boundary_runtime.restore_backward_3d_field(
                it, field1[f], launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver, f, false, f == 2
            );
        }
        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    auto z = torch::zeros_like(vp);
    out.grads = {grad_vp, grad_vs, grad_rho,
                 z, z, z,
                 z, z, z, z, z, z, z, z, z,
                 z, z, z,
                 z, z, z};
    return out;
}

}
