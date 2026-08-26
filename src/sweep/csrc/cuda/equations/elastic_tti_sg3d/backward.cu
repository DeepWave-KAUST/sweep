#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "elastic_tti_sg3d.h"
#include "kernels.cuh"
#include "tensors.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"

namespace elastic_tti_sg3d {

namespace {

void apply_adjoint_step(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    StiffnessPointer model,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    ElasticAdjointWorkspaceTensor& workspace
)
{
    auto adj_view = adjoint.view();

    LAUNCH_ELASTIC_TTI_SG3D_STRESS_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        model,
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

    LAUNCH_ELASTIC_TTI_SG3D_STRESS_ADJOINT_APPLY(
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

    LAUNCH_ELASTIC_TTI_SG3D_VELOCITY_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        model.rho,
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

    LAUNCH_ELASTIC_TTI_SG3D_VELOCITY_ADJOINT_APPLY(
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

// The rho imaging correlates the adjoint velocity with stored v-differences
// that still contain the raw body-force injection amplitude; the true
// derivative has no such term (the injection is rho-independent).  Compensate
// at the source cells (common.cu), mirroring elastic3d.
void apply_body_force_rho_correction(
    const BackwardInput& p,
    const torch::Tensor& source_fields,
    ElasticWavefieldPointer adj_view,
    torch::Tensor& grad_rho,
    const torch::Tensor& rho,
    const fdtd::LaunchConfig& fwd_source_config,
    SolverContext solver,
    int it
)
{
    // amp(it), not amp(it+1): the injected amplitude that survives in
    // v(it) - v(it+1) is the one added at step it (3ba5663 on tti_sg2d).
    if (it < 0 || it >= static_cast<int>(p.nt)) return;
    for (int isrc_bf = 0; isrc_bf < source_fields.numel(); ++isrc_bf) {
        int sfield_bf = source_fields[isrc_bf].item<int>();
        if (sfield_bf > 2) continue;
        float* adj_field_bf = elastic_field_ptr(adj_view, 3, sfield_bf);
        if (adj_field_bf == nullptr) continue;
        add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            adj_field_bf,
            rho.data_ptr<float>(),
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            (int)p.forward_sources_loc.size(1),
            3,
            solver
        );
    }
}

// Undo the just-injected receiver residual from this reverse step's rho
// imaging, at every VELOCITY-receiver cell; stress receivers carry no rho
// term.  Call it right after the imaging launch, while fv_now / fv_next still
// point at the operands the imaging correlated.  Same helper elastic3d uses
// (undo_receiver_rho_injection_3d) -- ported here because ElasticTTISG3D was
// forked from the pre-PR#62 elastic3d template.
void undo_receiver_rho_injection(
    const fdtd::LaunchConfig& adj_source_config,
    torch::Tensor& grad_rho,
    const float* fv_now[3],
    const float* fv_next[3],
    const torch::Tensor& rho,
    const BackwardInput& p,
    const torch::Tensor& receiver_fields,
    int it,
    int adjoint_nsrc,
    SolverContext solver
)
{
    const int nrec_fields = static_cast<int>(receiver_fields.numel());
    for (int irec = 0; irec < nrec_fields; ++irec) {
        const int field = receiver_fields[irec].item<int>();
        if (field > 2) continue;                      // stress receiver: no rho term
        sub_receiver_rho_grad_correction<<<adj_source_config.grid, adj_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            fv_now[field],
            fv_next[field],
            rho.data_ptr<float>(),
            p.adjoint_source[irec].data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            3,
            p.M,                                      // imaging halo (order/2 == M)
            solver
        );
    }
}

void backward_segment_ckpt(
    const BackwardInput& p,
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
    StiffnessPointer model,
    StiffnessGradPointer grad_view,
    torch::Tensor& grad_rho,
    const torch::Tensor& source_fields,
    const torch::Tensor& receiver_fields,
    ElasticAdjointWorkspaceTensor& workspace,
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

    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(rho, 3);
    checkpoint_runtime.copy_state(forward.state_tensors(), start_state.state_tensors());

    seg_vx.select(0, 0).copy_(forward.vx_t);
    seg_vy.select(0, 0).copy_(forward.vy_t);
    seg_vz.select(0, 0).copy_(forward.vz_t);

    for (int it = start; it < end; ++it) {
        auto for_view = forward.view();

        LAUNCH_ELASTIC_TTI_SG3D_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            model.rho,
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_TTI_SG3D_STRESS(
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

    // Stress receivers inject with the opposite sign (the constitutive update
    // folds the negated-stress convention into scale = -dt); injecting the
    // residual raw negates EVERY model gradient.  Same fix as 3ba5663 on
    // elastic_tti_sg2d, which shares this 3-D-style field layout.
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = end - 1; it >= start; --it) {
        auto adj_view = adjoint.view();

        // BEFORE the receiver injection: at a cell that is both source and
        // receiver, the post-injection adjoint carries the residual too and the
        // correction would over-shoot.
        apply_body_force_rho_correction(
            p, source_fields, adj_view, grad_rho, rho, fwd_source_config, solver, it
        );

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
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

        LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG3D_NOBS(
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
            const float* fv_now[3] = {seg_vx.select(0, now_offset).data_ptr<float>(), seg_vy.select(0, now_offset).data_ptr<float>(), seg_vz.select(0, now_offset).data_ptr<float>()};
            const float* fv_next[3] = {vx_next, vy_next, vz_next};
            undo_receiver_rho_injection(adj_source_config, grad_rho, fv_now, fv_next,
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

    TORCH_CHECK(p.models.size() == 22, "ElasticTTISG3D backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 12, "ElasticTTISG3D backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_forward.defined(), "ElasticTTISG3D full backward expects saved forward wavefields");
    TORCH_CHECK(p.u_forward.dim() == 6 && p.u_forward.size(1) == 3,
                "ElasticTTISG3D full backward expects u_forward with shape (nt, 3, B, nz, ny, nx)");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int ny = rho.size(3);
    const int nx = rho.size(4);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dy = p.spacing[1];
    const float dz = p.spacing[2];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(rho, 3);
    zero_wavefield_state(adjoint);
    auto adj_view = adjoint.view();

    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, rho, 3);
    auto zero_velocity = torch::zeros_like(rho);

    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);

    // Stress receivers inject with the opposite sign (the constitutive update
    // folds the negated-stress convention into scale = -dt); injecting the
    // residual raw negates EVERY model gradient.  Same fix as 3ba5663 on
    // elastic_tti_sg2d, which shares this 3-D-style field layout.
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        // BEFORE the receiver injection: at a cell that is both source and
        // receiver, the post-injection adjoint carries the residual too and the
        // correction would over-shoot.
        apply_body_force_rho_correction(
            p, source_fields, adj_view, grads[0], rho, fwd_source_config, solver, it
        );

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

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

        LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG3D_NOBS(
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
            const float* fv_now[3] = {vx_now, vy_now, vz_now};
            const float* fv_next[3] = {vx_next, vy_next, vz_next};
            undo_receiver_rho_injection(source_config, grads[0], fv_now, fv_next,
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

    TORCH_CHECK(p.models.size() == 22, "ElasticTTISG3D boundary-saving backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 12, "ElasticTTISG3D boundary-saving backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_last_two.defined(), "ElasticTTISG3D boundary-saving backward expects last-two wavefield tensor");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int ny = rho.size(3);
    const int nx = rho.size(4);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dy = p.spacing[1];
    const float dz = p.spacing[2];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(rho, 3);
    zero_wavefield_state(adjoint);

    ElasticWavefieldTensor forward;
    forward.allocate(rho, 3);
    forward.vx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
    forward.vy_t.copy_(p.u_last_two.select(0, 1).select(0, 0));
    forward.vz_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
    forward.sxx_t.copy_(p.u_last_two.select(0, 3).select(0, 0));
    forward.syy_t.copy_(p.u_last_two.select(0, 4).select(0, 0));
    forward.szz_t.copy_(p.u_last_two.select(0, 5).select(0, 0));
    forward.sxy_t.copy_(p.u_last_two.select(0, 6).select(0, 0));
    forward.sxz_t.copy_(p.u_last_two.select(0, 7).select(0, 0));
    forward.syz_t.copy_(p.u_last_two.select(0, 8).select(0, 0));

    auto adj_view = adjoint.view();
    auto for_view = forward.view();
    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, rho, 3);

    auto fvx_next = torch::zeros_like(rho);
    auto fvy_next = torch::zeros_like(rho);
    auto fvz_next = torch::zeros_like(rho);

    EffectiveBoundarySaver boundary_saver;
    const int save_width = solver.M + 1;
    const bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 3, 9, solver, rho, save_width, 1,
            true, false, p.transfer_interval,
            p.boundary_cpu, p.boundary_gpu,
            {}, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            true, 3, 9, solver, rho, save_width, 1,
            true, true, 1,
            {}, p.boundary_gpu,
            {}, p.use_pinned_memory
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
    auto neg_forward_source = -p.forward_source;

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

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

    // Stress receivers inject with the opposite sign (the constitutive update
    // folds the negated-stress convention into scale = -dt); injecting the
    // residual raw negates EVERY model gradient.  Same fix as 3ba5663 on
    // elastic_tti_sg2d, which shares this 3-D-style field layout.
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = static_cast<int>(p.nt) - 1; it >= 1; --it) {
        // BEFORE the receiver injection: at a cell that is both source and
        // receiver, the post-injection adjoint carries the residual too and the
        // correction would over-shoot.
        apply_body_force_rho_correction(
            p, source_fields, adj_view, grads[0], rho, fwd_source_config, solver, it
        );

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        // Wavefield reconstruction: subtract source, reverse the stress
        // update, then restore the saved stress boundary ring.
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

        LAUNCH_ELASTIC_TTI_SG3D_STRESS_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            model,
            grad_ctx,
            solver
        );

        float* stress_fields[6] = {
            for_view.sxx, for_view.syy, for_view.szz,
            for_view.sxy, for_view.sxz, for_view.syz
        };
        for (int f = 3; f < 9; ++f) {
            boundary_runtime.restore_backward_3d_field(
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

        LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG3D_NOBS(
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
            const float* fv_now[3] = {for_view.vx, for_view.vy, for_view.vz};
            const float* fv_next[3] = {fvx_next.data_ptr<float>(), fvy_next.data_ptr<float>(), fvz_next.data_ptr<float>()};
            undo_receiver_rho_injection(adj_source_config, grads[0], fv_now, fv_next,
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

        LAUNCH_ELASTIC_TTI_SG3D_VELOCITY_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            model.rho,
            grad_ctx,
            solver
        );

        float* velocity_fields[3] = {
            for_view.vx, for_view.vy, for_view.vz
        };
        for (int f = 0; f < 3; ++f) {
            boundary_runtime.restore_backward_3d_field(
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

    TORCH_CHECK(p.models.size() == 22, "ElasticTTISG3D checkpoint backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 12, "ElasticTTISG3D checkpoint backward expects cpmls PML profiles");
    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 36, "ElasticTTISG3D checkpointing expects 36 checkpoint tensors");

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        36,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "elastic_tti_sg3d"
    );

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int ny = rho.size(3);
    const int nx = rho.size(4);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dy = p.spacing[1];
    const float dz = p.spacing[2];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    auto model = stiffness_view(p.models);
    auto grads = zero_model_grads(p.models);
    auto grad_view = stiffness_grad_view(grads);

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(rho, 3);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticWavefieldTensor start_state;
    start_state.allocate(rho, 3);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, rho, 3);

    auto next_segment_vx = torch::zeros_like(rho);
    auto next_segment_vy = torch::zeros_like(rho);
    auto next_segment_vz = torch::zeros_like(rho);
    auto prev_segment_next_vx = torch::zeros_like(rho);
    auto prev_segment_next_vy = torch::zeros_like(rho);
    auto prev_segment_next_vz = torch::zeros_like(rho);

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
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

} // namespace elastic_tti_sg3d
