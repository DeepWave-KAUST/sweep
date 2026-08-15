#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernels.cuh"
#include "das_mu2d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace das_mu2d {

namespace {

void apply_adjoint_step_2d(
    int order,
    const fdtd::LaunchConfig& launch_config,
    DasMuWavefieldTensor2D& adjoint,
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
    auto elastic_adj_view = adjoint.elastic_view();

    LAUNCH_DAS_MU2D_STRESS_STRAIN_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        lambda.data_ptr<float>(),
        mu.data_ptr<float>(),
        cpml_view,
        solver,
        workspace.qxx_t.data_ptr<float>(),
        workspace.qzz_t.data_ptr<float>(),
        workspace.qxz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>()
    );

    LAUNCH_ELASTIC_STRESS_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        elastic_adj_view,
        workspace.qxx_t.data_ptr<float>(),
        workspace.qzz_t.data_ptr<float>(),
        workspace.qxz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>(),
        grad_ctx,
        solver
    );

    LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        elastic_adj_view,
        rho.data_ptr<float>(),
        cpml_view,
        solver,
        workspace.pxx_t.data_ptr<float>(),
        workspace.pzz_t.data_ptr<float>(),
        workspace.pxz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>()
    );

    LAUNCH_ELASTIC_VELOCITY_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        elastic_adj_view,
        workspace.pxx_t.data_ptr<float>(),
        workspace.pzz_t.data_ptr<float>(),
        workspace.pxz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(),
        grad_ctx,
        solver
    );
}

int find_previous_checkpoint_idx(
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

void replay_forward_to_time_2d(
    const BackwardInput& p,
    DasMuWavefieldTensor2D& forward,
    torch::Tensor& current_vx,
    torch::Tensor& current_vz,
    torch::Tensor& next_vx,
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
    current_vz.zero_();
    next_vx.zero_();
    next_vz.zero_();

    const int checkpoint_idx = find_previous_checkpoint_idx(checkpoint_steps, num_saved_checkpoints, target_index + 1);
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
        auto elastic_for_view = forward.elastic_view();

        LAUNCH_ELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            elastic_for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_DAS_MU2D_STRESS_STRAIN(
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
            current_vz.copy_(forward.vz_t);
        }
        if (it == target_index + 1) {
            next_vx.copy_(forward.vx_t);
            next_vz.copy_(forward.vz_t);
            break;
        }

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = das_mu2d_field_ptr(for_view, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
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

void backward_segment_2d(
    const BackwardInput& p,
    const torch::Tensor& vp,
    const torch::Tensor& vs,
    const torch::Tensor& rho,
    DasMuWavefieldTensor2D& start_state,
    DasMuWavefieldTensor2D& adjoint,
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
    const torch::Tensor& next_segment_vx,
    const torch::Tensor& next_segment_vz,
    torch::Tensor& grad_vp,
    torch::Tensor& grad_vs,
    torch::Tensor& grad_rho,
    torch::Tensor& prev_segment_next_vx,
    torch::Tensor& prev_segment_next_vz
)
{
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    const int nrec_fields = p.receiver_field_indices.numel();
    const int segment_len = end - start;

    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto mu  = rho * vs * vs;
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    auto seg_vx = torch::zeros({segment_len + 1, vp.size(0) * vp.size(1), 1, vp.size(2), vp.size(3)}, vp.options());
    auto seg_vz = torch::zeros_like(seg_vx);
    DasMuWavefieldTensor2D forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(vp, true);

    checkpoint_runtime.copy_state(forward.state_tensors(), start_state.state_tensors());

    seg_vx.select(0, 0).copy_(forward.vx_t);
    seg_vz.select(0, 0).copy_(forward.vz_t);

    for (int it = start; it < end; ++it) {
        auto for_view = forward.view();
        auto elastic_for_view = forward.elastic_view();

        LAUNCH_ELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            elastic_for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_DAS_MU2D_STRESS_STRAIN(
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
        seg_vz.select(0, it - start + 1).copy_(forward.vz_t);

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = das_mu2d_field_ptr(for_view, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
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
        auto elastic_adj_view = adjoint.elastic_view();

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das_mu2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
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

        // Body-force (velocity) sources: the rho imaging correlates the adjoint
        // velocity with stored v-differences that still contain the raw injected
        // amplitude; the true derivative has no such term (the injection is
        // rho-independent).  Compensate at the source cells (common.cu).
        if (it + 1 < p.nt) {
            for (int isrc_bf = 0; isrc_bf < source_fields.numel(); ++isrc_bf) {
                int sfield_bf = source_fields[isrc_bf].item<int>();
                if (sfield_bf > 1) continue;
                float* adj_field_bf = das_mu2d_field_ptr(adj_view, sfield_bf);
                if (adj_field_bf == nullptr) continue;
                add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
                    grad_rho.data_ptr<float>(),
                    adj_field_bf,
                    rho.data_ptr<float>(),
                    p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it + 1,
                    (int)p.forward_sources_loc.size(1),
                    2,
                    solver
                );
            }
        }

        const int offset = it - start;
        const int now_offset = offset + 1;
        const int next_offset = now_offset + 1;

        LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            elastic_adj_view,
            seg_vx.select(0, now_offset).data_ptr<float>(),
            seg_vz.select(0, now_offset).data_ptr<float>(),
            (next_offset <= segment_len) ? seg_vx.select(0, next_offset).data_ptr<float>() : next_segment_vx.data_ptr<float>(),
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

        apply_adjoint_step_2d(
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
    prev_segment_next_vz.copy_(seg_vz.select(0, 1));

}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
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

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    DasMuWavefieldTensor2D adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, true);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto zero_velocity = torch::zeros_like(vp);

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();
        auto elastic_adj_view = adjoint.elastic_view();

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das_mu2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
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

        // Body-force (velocity) sources: the rho imaging correlates the adjoint
        // velocity with stored v-differences that still contain the raw injected
        // amplitude; the true derivative has no such term (the injection is
        // rho-independent).  Compensate at the source cells (common.cu).
        if (it + 1 < p.nt) {
            for (int isrc_bf = 0; isrc_bf < source_fields.numel(); ++isrc_bf) {
                int sfield_bf = source_fields[isrc_bf].item<int>();
                if (sfield_bf > 1) continue;
                float* adj_field_bf = das_mu2d_field_ptr(adj_view, sfield_bf);
                if (adj_field_bf == nullptr) continue;
                add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
                    grad_rho.data_ptr<float>(),
                    adj_field_bf,
                    rho.data_ptr<float>(),
                    p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it + 1,
                    (int)p.forward_sources_loc.size(1),
                    2,
                    solver
                );
            }
        }

        const float* vx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* vz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* vx_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vz_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero_velocity.data_ptr<float>();

        LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            elastic_adj_view,
            vx_now,
            vz_now,
            vx_next,
            vz_next,
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

        apply_adjoint_step_2d(
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
    TORCH_CHECK(p.checkpoints.size() == 18, "DAS Mu 2D checkpointing expects 18 checkpoint tensors");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        18,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "das_mu2d"
    );

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    DasMuWavefieldTensor2D adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    DasMuWavefieldTensor2D start_state;
    start_state.allocate(vp, true);
    auto next_segment_vx = torch::zeros_like(vp);
    auto next_segment_vz = torch::zeros_like(vp);
    auto prev_segment_next_vx = torch::zeros_like(vp);
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
        backward_segment_2d(
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
            next_segment_vx,
            next_segment_vz,
            grad_vp,
            grad_vs,
            grad_rho,
            prev_segment_next_vx,
            prev_segment_next_vz
        );
        next_segment_vx.copy_(prev_segment_next_vx);
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

    TORCH_CHECK(p.checkpoints.size() == 18, "DAS Mu 2D recursive checkpointing expects 18 checkpoint tensors");

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        18,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "das_mu2d"
    );

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    DasMuWavefieldTensor2D adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto mu  = rho * vs * vs;

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

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
    DasMuWavefieldTensor2D forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(vp, true);
    auto current_vx = torch::zeros_like(vp);
    auto current_vz = torch::zeros_like(vp);
    auto next_vx = torch::zeros_like(vp);
    auto next_vz = torch::zeros_like(vp);

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();
        auto elastic_adj_view = adjoint.elastic_view();

        for (int irec = 0; irec < static_cast<int>(p.receiver_field_indices.numel()); ++irec) {
            float* field = das_mu2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                p.adjoint_sources_loc.size(1),
                solver
            );
        }

        // Body-force (velocity) sources: the rho imaging correlates the adjoint
        // velocity with stored v-differences that still contain the raw injected
        // amplitude; the true derivative has no such term (the injection is
        // rho-independent).  Compensate at the source cells (common.cu).
        if (it + 1 < p.nt) {
            for (int isrc_bf = 0; isrc_bf < source_fields.numel(); ++isrc_bf) {
                int sfield_bf = source_fields[isrc_bf].item<int>();
                if (sfield_bf > 1) continue;
                float* adj_field_bf = das_mu2d_field_ptr(adj_view, sfield_bf);
                if (adj_field_bf == nullptr) continue;
                add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
                    grad_rho.data_ptr<float>(),
                    adj_field_bf,
                    rho.data_ptr<float>(),
                    p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it + 1,
                    (int)p.forward_sources_loc.size(1),
                    2,
                    solver
                );
            }
        }

        replay_forward_to_time_2d(
            p,
            forward,
            current_vx,
            current_vz,
            next_vx,
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

        LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            elastic_adj_view,
            current_vx.data_ptr<float>(),
            current_vz.data_ptr<float>(),
            next_vx.data_ptr<float>(),
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

        apply_adjoint_step_2d(
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


BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

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

    DasMuWavefieldTensor2D adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, true);
    DasMuWavefieldTensor2D forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(vp, true);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Copy last step of forward wavefield from u_last_two
    forward.vx_t.copy_(p.u_last_two.select(0,0).select(0,0));
    forward.vz_t.copy_(p.u_last_two.select(0,1).select(0,0));
    forward.sxx_t.copy_(p.u_last_two.select(0,2).select(0,0));
    forward.szz_t.copy_(p.u_last_two.select(0,3).select(0,0));
    forward.sxz_t.copy_(p.u_last_two.select(0,4).select(0,0));
    if (p.u_last_two.size(0) >= 8) {
        forward.exx_t.copy_(p.u_last_two.select(0,5).select(0,0));
        forward.ezz_t.copy_(p.u_last_two.select(0,6).select(0,0));
        forward.exz_t.copy_(p.u_last_two.select(0,7).select(0,0));
    }

    auto neg_forward_source = -p.forward_source;

    // Generate pointer views
    auto for_view = forward.view();
    auto adj_view = adjoint.view();
    auto elastic_for_view = forward.elastic_view();
    auto elastic_adj_view = adjoint.elastic_view();

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();


    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 8, solver, vp, save_width, 1, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 8, solver, vp, save_width, 1, true, true, 1, {}, p.boundary_gpu, {}, p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

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

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        2,
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

    // auto u_all_for = torch::zeros({nt, B, 1, nz, nx}, vp.options());
    // auto u_all_adj = torch::zeros({nt, B, 1, nz, nx}, vp.options());

    for (int it = p.nt - 1; it >= 1; --it) {

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das_mu2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
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

        // Body-force (velocity) sources: the rho imaging correlates the adjoint
        // velocity with stored v-differences that still contain the raw injected
        // amplitude; the true derivative has no such term (the injection is
        // rho-independent).  Compensate at the source cells (common.cu).
        if (it + 1 < p.nt) {
            for (int isrc_bf = 0; isrc_bf < source_fields.numel(); ++isrc_bf) {
                int sfield_bf = source_fields[isrc_bf].item<int>();
                if (sfield_bf > 1) continue;
                float* adj_field_bf = das_mu2d_field_ptr(adj_view, sfield_bf);
                if (adj_field_bf == nullptr) continue;
                add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
                    grad_rho.data_ptr<float>(),
                    adj_field_bf,
                    rho.data_ptr<float>(),
                    p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it + 1,
                    (int)p.forward_sources_loc.size(1),
                    2,
                    solver
                );
            }
        }

        // Wavefield reconstruction
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = das_mu2d_field_ptr(for_view, source_fields[isrc].item<int>());
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
            elastic_for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float *field2[3] = {for_view.sxx, for_view.szz, for_view.sxz};

        for (int f = 2; f < 5; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                field2[f-2],
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                -p.M,
                solver,
                f,
                f == 2,
                false
            );
        }

        // Gradient calculation
        LAUNCH_CALCULATE_GRAD_ELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            elastic_for_view,
            elastic_adj_view,

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

        apply_adjoint_step_2d(
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
        fvx_prev.copy_(forward.vx_t);

        // Update Velocity components
        LAUNCH_ELASTIC_VELOCITY_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            elastic_for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float *field1[2] = {for_view.vx, for_view.vz};

        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
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
                f == 1
            );
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);

        // u_all_for[it].copy_(forward.vz_t); // for visualization
        // u_all_adj[it].copy_(adjoint.vz_t); // for visualization
    }

    for (int irec = 0; irec < nrec_fields; ++irec) {
        float* field = das_mu2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
        if (field == nullptr) continue;
        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            field,
            p.adjoint_source[irec].data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            0,
            adjoint_nsrc,
            solver
        );
    }

    // Body-force (velocity) sources: the rho imaging correlates the adjoint
    // velocity with stored v-differences that still contain the raw injected
    // amplitude; the true derivative has no such term (the injection is
    // rho-independent).  Compensate at the source cells (common.cu).
    if (0 + 1 < p.nt) {
        for (int isrc_bf = 0; isrc_bf < source_fields.numel(); ++isrc_bf) {
            int sfield_bf = source_fields[isrc_bf].item<int>();
            if (sfield_bf > 1) continue;
            float* adj_field_bf = das_mu2d_field_ptr(adj_view, sfield_bf);
            if (adj_field_bf == nullptr) continue;
            add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
                grad_rho.data_ptr<float>(),
                adj_field_bf,
                rho.data_ptr<float>(),
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                0 + 1,
                (int)p.forward_sources_loc.size(1),
                2,
                solver
            );
        }
    }

    LAUNCH_CALCULATE_GRAD_ELASTIC_BS(
        order,
        launch_config.grid,
        launch_config.block,
        elastic_for_view,
        elastic_adj_view,

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

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;

}

}
