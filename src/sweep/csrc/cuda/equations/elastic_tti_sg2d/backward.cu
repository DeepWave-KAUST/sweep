#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <array>

#include "elastic_tti_sg2d.h"
#include "kernels.cuh"
#include "tensors.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"   // elastic_signed_adjoint_sources
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace elastic_tti_sg2d {

namespace {

// Undo the just-injected receiver residual from this step's rho imaging, at
// every velocity-receiver cell (kernel and rationale in common.cuh).  Velocity
// fields here are 0=vx, 1=vy, 2=vz; stress receivers have no rho term.
void undo_receiver_rho_injection_tti_sg2d(
    const fdtd::LaunchConfig& adj_source_config,
    torch::Tensor& grad_rho,
    const float* fv_now[3],
    const float* fv_next[3],
    const torch::Tensor& rho,
    const BackwardInput& p,
    const torch::Tensor& receiver_fields,
    int it,
    int adjoint_nsrc,
    const SolverContext& solver
)
{
    const int nrec_fields = static_cast<int>(receiver_fields.numel());
    for (int irec = 0; irec < nrec_fields; ++irec) {
        const int field = receiver_fields[irec].item<int>();
        if (field > 2) continue;
        sub_receiver_rho_grad_correction<<<adj_source_config.grid, adj_source_config.block>>>(
            grad_rho.data_ptr<float>(), fv_now[field], fv_next[field],
            rho.data_ptr<float>(),
            p.adjoint_source[irec].data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it, adjoint_nsrc, 2, p.M, solver);
    }
}

// Body-force (velocity) sources: the rho imaging correlates the adjoint velocity
// with the stored difference v(it) - v(it+1), which at a source cell still
// carries the raw injected amplitude; the true derivative has no such term.
//
// Two details, both established by finite-difference arbitration on the elastic
// path (46172fd -> f59e833 / a23c701) and re-confirmed on DASMu (b591fbc):
//   * the amplitude in u_forward[it] - u_forward[it+1] is amp(it), not amp(it+1);
//   * this must run BEFORE the step's receiver residuals are injected, or at a
//     cell that is both source and receiver the two corrections double-count.
// The cherry-picked 52976be introduced this correction with both of those wrong;
// this is the same shape as the elastic/DASMu helpers.
void undo_body_force_source_injection_tti_sg2d(
    const fdtd::LaunchConfig& fwd_source_config,
    torch::Tensor& grad_rho,
    WavefieldPointer& adj_view,
    const torch::Tensor& rho,
    const BackwardInput& p,
    const torch::Tensor& source_fields,
    int it,
    const SolverContext& solver
)
{
    if (it < 0 || it >= static_cast<int>(p.nt)) return;
    for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
        const int sfield = source_fields[isrc].item<int>();
        if (sfield > 2) continue;                       // vx = 0, vy = 1, vz = 2
        float* adj_field = field_ptr(adj_view, sfield);
        if (adj_field == nullptr) continue;
        add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            adj_field,
            rho.data_ptr<float>(),
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            (int)p.forward_sources_loc.size(1),
            2,
            solver
        );
    }
}

void apply_adjoint_step(
    int order,
    const fdtd::LaunchConfig& launch_config,
    WavefieldTensor& adjoint,
    StiffnessPointer model,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    const std::array<torch::Tensor, 6>& workspace
)
{
    auto adj_view = adjoint.view();

    LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        model,
        cpml_view,
        solver,
        workspace[0].data_ptr<float>(),
        workspace[1].data_ptr<float>(),
        workspace[2].data_ptr<float>(),
        workspace[3].data_ptr<float>(),
        workspace[4].data_ptr<float>(),
        workspace[5].data_ptr<float>()
    );

    LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        workspace[0].data_ptr<float>(),
        workspace[1].data_ptr<float>(),
        workspace[2].data_ptr<float>(),
        workspace[3].data_ptr<float>(),
        workspace[4].data_ptr<float>(),
        workspace[5].data_ptr<float>(),
        grad_ctx,
        solver
    );

    LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        model,
        cpml_view,
        solver,
        workspace[0].data_ptr<float>(),
        workspace[1].data_ptr<float>(),
        workspace[2].data_ptr<float>(),
        workspace[3].data_ptr<float>(),
        workspace[4].data_ptr<float>(),
        workspace[5].data_ptr<float>()
    );

    LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        workspace[0].data_ptr<float>(),
        workspace[1].data_ptr<float>(),
        workspace[2].data_ptr<float>(),
        workspace[3].data_ptr<float>(),
        workspace[4].data_ptr<float>(),
        workspace[5].data_ptr<float>(),
        grad_ctx,
        solver
    );
}

void backward_segment_ckpt(
    const BackwardInput& p,
    const torch::Tensor& rho,
    WavefieldTensor& start_state,
    WavefieldTensor& adjoint,
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
    StiffnessPointer model,
    StiffnessGradPointer grad_view,
    torch::Tensor& grad_rho,
    const torch::Tensor& source_fields,
    const torch::Tensor& receiver_fields,
    const std::array<torch::Tensor, 6>& workspace,
    const torch::Tensor& next_segment_vx,
    const torch::Tensor& next_segment_vy,
    const torch::Tensor& next_segment_vz,
    torch::Tensor& prev_segment_next_vx,
    torch::Tensor& prev_segment_next_vy,
    torch::Tensor& prev_segment_next_vz
)
{
    const int segment_len = end - start;
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    const int nrec_fields = p.receiver_field_indices.numel();

    std::vector<int64_t> seg_shape = rho.sizes().vec();
    seg_shape.insert(seg_shape.begin(), static_cast<int64_t>(segment_len + 1));
    auto seg_vx = torch::zeros(seg_shape, rho.options());
    auto seg_vy = torch::zeros(seg_shape, rho.options());
    auto seg_vz = torch::zeros(seg_shape, rho.options());

    WavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields);
    else
        forward.allocate(rho);
    checkpoint_runtime.copy_state(forward.state_tensors(), start_state.state_tensors());

    seg_vx.select(0, 0).copy_(forward.vx_t);
    seg_vy.select(0, 0).copy_(forward.vy_t);
    seg_vz.select(0, 0).copy_(forward.vz_t);

    for (int it = start; it < end; ++it) {
        auto for_view = forward.view();

        LAUNCH_ELASTIC_TTI_SG_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            model,
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_TTI_SG_STRESS(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            model,
            nullptr,
            grad_ctx,
            cpml_view,
            solver
        );

        const int offset = it - start + 1;
        seg_vx.select(0, offset).copy_(forward.vx_t);
        seg_vy.select(0, offset).copy_(forward.vy_t);
        seg_vz.select(0, offset).copy_(forward.vz_t);

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = field_ptr(for_view, source_fields[isrc].item<int>());
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

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = end - 1; it >= start; --it) {
        auto adj_view = adjoint.view();

        undo_body_force_source_injection_tti_sg2d(fwd_source_config, grad_rho, adj_view,
                                                  rho, p, source_fields, it, solver);
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        const int offset = it - start;
        const int now_offset = offset + 1;
        const int next_offset = now_offset + 1;
        const float* vx_next = (next_offset <= segment_len)
            ? seg_vx.select(0, next_offset).data_ptr<float>()
            : next_segment_vx.data_ptr<float>();
        const float* vy_next = (next_offset <= segment_len)
            ? seg_vy.select(0, next_offset).data_ptr<float>()
            : next_segment_vy.data_ptr<float>();
        const float* vz_next = (next_offset <= segment_len)
            ? seg_vz.select(0, next_offset).data_ptr<float>()
            : next_segment_vz.data_ptr<float>();

        LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            model,
            grad_view,
            seg_vx.select(0, now_offset).data_ptr<float>(),
            seg_vy.select(0, now_offset).data_ptr<float>(),
            seg_vz.select(0, now_offset).data_ptr<float>(),
            vx_next,
            vy_next,
            vz_next,
            grad_ctx,
            solver
        );

        {
            const float* fv_now[3]  = {seg_vx.select(0, now_offset).data_ptr<float>(),
                                       seg_vy.select(0, now_offset).data_ptr<float>(),
                                       seg_vz.select(0, now_offset).data_ptr<float>()};
            const float* fv_next[3] = {vx_next, vy_next, vz_next};
            undo_receiver_rho_injection_tti_sg2d(adj_source_config, grad_rho, fv_now, fv_next,
                                                 rho, p, receiver_fields, it, adjoint_nsrc, solver);
        }

        if (it == 0)
            continue;

        apply_adjoint_step(
            order,
            launch_config,
            adjoint,
            model,
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

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 16, "ElasticTTISG backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTISG backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_forward.defined(), "ElasticTTISG full backward expects saved forward wavefields");
    TORCH_CHECK(p.u_forward.dim() == 5 && p.u_forward.size(1) == 8,
                "ElasticTTISG full backward expects u_forward with shape (nt, 8, B, nz, nx)");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    WavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(rho);
    auto adj_view = adjoint.view();

    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto workspace = std::array<torch::Tensor, 6>{
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
    };
    auto zero_velocity = torch::zeros_like(rho);

    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);
    auto source_fields = p.source_field_indices.to(torch::kCPU);

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);

    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        undo_body_force_source_injection_tti_sg2d(fwd_source_config, grads[0], adj_view,
                                                  rho, p, source_fields, it, solver);
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<source_config.grid, source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }


        // Body-force (velocity) sources: the rho imaging correlates the adjoint
        // velocity with v(it) - v(it+1), which at a source cell still contains
        // the raw injected amplitude amp(it+1); the true derivative has no such
        // term (the injection is rho-independent).  Compensate at the source
        // cells (kernel in common.cu).  Fields: vx = 0, vy = 1, vz = 2.

        const float* vx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* vy_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* vz_now = p.u_forward.select(0, it).select(0, 2).data_ptr<float>();
        const float* vx_next = (it + 1 < static_cast<int>(p.nt))
            ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>()
            : zero_velocity.data_ptr<float>();
        const float* vy_next = (it + 1 < static_cast<int>(p.nt))
            ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>()
            : zero_velocity.data_ptr<float>();
        const float* vz_next = (it + 1 < static_cast<int>(p.nt))
            ? p.u_forward.select(0, it + 1).select(0, 2).data_ptr<float>()
            : zero_velocity.data_ptr<float>();

        LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            model,
            grad_view,
            vx_now,
            vy_now,
            vz_now,
            vx_next,
            vy_next,
            vz_next,
            grad_ctx,
            solver
        );

        {
            const float* fv_now[3]  = {vx_now, vy_now, vz_now};
            const float* fv_next[3] = {vx_next, vy_next, vz_next};
            undo_receiver_rho_injection_tti_sg2d(source_config, grads[0], fv_now, fv_next,
                                                 rho, p, receiver_fields, it, adjoint_nsrc, solver);
        }

        if (it == 0)
            continue;

        apply_adjoint_step(
            order,
            launch_config,
            adjoint,
            model,
            cpml_view,
            grad_ctx,
            solver,
            workspace
        );
    }

    out.grads = grads;
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 16, "ElasticTTISG boundary-saving backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTISG boundary-saving backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_last_two.defined(), "ElasticTTISG boundary-saving backward expects last-two wavefield tensor");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    WavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(rho);

    WavefieldTensor forward;
    forward.allocate(rho);
    forward.vx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
    forward.vy_t.copy_(p.u_last_two.select(0, 1).select(0, 0));
    forward.vz_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
    forward.sxx_t.copy_(p.u_last_two.select(0, 3).select(0, 0));
    forward.szz_t.copy_(p.u_last_two.select(0, 4).select(0, 0));
    forward.syz_t.copy_(p.u_last_two.select(0, 5).select(0, 0));
    forward.sxz_t.copy_(p.u_last_two.select(0, 6).select(0, 0));
    forward.sxy_t.copy_(p.u_last_two.select(0, 7).select(0, 0));

    auto adj_view = adjoint.view();
    auto for_view = forward.view();
    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto workspace = std::array<torch::Tensor, 6>{
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
    };
    auto fvx_next = torch::zeros_like(rho);
    auto fvy_next = torch::zeros_like(rho);
    auto fvz_next = torch::zeros_like(rho);

    EffectiveBoundarySaver boundary_saver;
    const int save_width = solver.M + 1;
    const bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            true,
            2,
            8,
            solver,
            rho,
            save_width,
            1,
            true,
            false,
            p.transfer_interval,
            p.boundary_cpu,
            p.boundary_gpu,
            {},
            p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            true,
            2,
            8,
            solver,
            rho,
            save_width,
            1,
            true,
            true,
            1,
            {},
            p.boundary_gpu,
            {},
            p.use_pinned_memory
        );
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, rho);
    }
    auto bs = boundary_saver.view();

    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    const int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);
    auto neg_forward_source = -p.forward_source;

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

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

    for (int it = static_cast<int>(p.nt) - 1; it >= 1; --it) {
        undo_body_force_source_injection_tti_sg2d(fwd_source_config, grads[0], adj_view,
                                                  rho, p, source_fields, it, solver);
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }


        // Body-force (velocity) sources: the rho imaging correlates the adjoint
        // velocity with v(it) - v(it+1), which at a source cell still contains
        // the raw injected amplitude amp(it+1); the true derivative has no such
        // term (the injection is rho-independent).  Compensate at the source
        // cells (kernel in common.cu).  Fields: vx = 0, vy = 1, vz = 2.

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = field_ptr(for_view, source_fields[isrc].item<int>());
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

        LAUNCH_ELASTIC_TTI_SG_STRESS_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            model,
            grad_ctx,
            solver
        );

        float* stress_fields[5] = {
            for_view.sxx,
            for_view.szz,
            for_view.syz,
            for_view.sxz,
            for_view.sxy
        };
        for (int f = 3; f < 8; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                stress_fields[f - 3],
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

        LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            model,
            grad_view,
            for_view.vx,
            for_view.vy,
            for_view.vz,
            fvx_next.data_ptr<float>(),
            fvy_next.data_ptr<float>(),
            fvz_next.data_ptr<float>(),
            grad_ctx,
            solver
        );

        {
            const float* fv_now[3]  = {for_view.vx, for_view.vy, for_view.vz};
            const float* fv_next[3] = {fvx_next.data_ptr<float>(),
                                       fvy_next.data_ptr<float>(),
                                       fvz_next.data_ptr<float>()};
            undo_receiver_rho_injection_tti_sg2d(adj_source_config, grads[0], fv_now, fv_next,
                                                 rho, p, receiver_fields, it, adjoint_nsrc, solver);
        }

        apply_adjoint_step(
            order,
            launch_config,
            adjoint,
            model,
            cpml_view,
            grad_ctx,
            solver,
            workspace
        );

        fvx_next.copy_(forward.vx_t);
        fvy_next.copy_(forward.vy_t);
        fvz_next.copy_(forward.vz_t);

        LAUNCH_ELASTIC_TTI_SG_VELOCITY_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            model,
            grad_ctx,
            solver
        );

        float* velocity_fields[3] = {
            for_view.vx,
            for_view.vy,
            for_view.vz
        };
        for (int f = 0; f < 3; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                velocity_fields[f],
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
    }

    out.grads = grads;
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 16, "ElasticTTISG checkpoint backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTISG checkpoint backward expects cpmls PML profiles");
    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 20, "ElasticTTISG checkpointing expects 20 checkpoint tensors");

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        20,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "elastic_tti_sg2d"
    );

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    WavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(rho);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    WavefieldTensor start_state;
    start_state.allocate(rho);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto workspace = std::array<torch::Tensor, 6>{
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
        torch::zeros_like(rho),
    };
    auto next_segment_vx = torch::zeros_like(rho);
    auto next_segment_vy = torch::zeros_like(rho);
    auto next_segment_vz = torch::zeros_like(rho);
    auto prev_segment_next_vx = torch::zeros_like(rho);
    auto prev_segment_next_vy = torch::zeros_like(rho);
    auto prev_segment_next_vz = torch::zeros_like(rho);

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);

    const int chunk_size = p.checkpoint_interval;
    const int num_chunks = (static_cast<int>(p.nt) + chunk_size - 1) / chunk_size;
    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        const int start = chunk_id * chunk_size;
        const int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        if (chunk_id == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(chunk_id, start_state.checkpoint_tensors());

        backward_segment_ckpt(
            p,
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
            model,
            grad_view,
            grads[0],
            source_fields,
            receiver_fields,
            workspace,
            next_segment_vx,
            next_segment_vy,
            next_segment_vz,
            prev_segment_next_vx,
            prev_segment_next_vy,
            prev_segment_next_vz
        );

        next_segment_vx.copy_(prev_segment_next_vx);
        next_segment_vy.copy_(prev_segment_next_vy);
        next_segment_vz.copy_(prev_segment_next_vz);
    }

    out.grads = grads;
    return out;
}

} // namespace elastic_tti_sg2d
