#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

#include "kernels.cuh"
#include "elastic2d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

// Receiver-side counterpart of sub_receiver_rho_grad_correction (common.cu) for
// the APM path: same just-injected residual to undo, but the APM kinetic term
// divides by the per-component EFFECTIVE density and chains through
// d(rho_eff)/d(rho), so the correction has to carry both.  ``is_z_component``
// picks rho_z / drho_z_drho for a vz receiver, rho_x / drho_x_drho for vx.
// ``static`` because this is the only translation unit that launches it and a
// non-template __global__ in a shared header would break the link (ODR).
static __global__ void sub_receiver_rho_grad_correction_apm_2d(
    float* __restrict__ grad_rho,
    const float* __restrict__ fv_now,
    const float* __restrict__ fv_next,
    const float* __restrict__ rho_eff,
    const int* __restrict__ category,
    const float* __restrict__ adjoint_source,
    const int* __restrict__ receivers_loc,
    int it,
    int nrec,
    int halo,
    int is_z_component,
    const SolverContext solver
) {
    int b = blockIdx.x;
    int s = blockIdx.y * blockDim.x + threadIdx.x;

    if (b >= solver.B || s >= nrec) return;
    if (it < 0 || it >= solver.nt) return;

    int base = (b * nrec + s) * 2;
    int ix = receivers_loc[base + 0];
    int iz = receivers_loc[base + 1];

    if (ix < halo || ix >= solver.nx - halo ||
        iz < halo || iz >= solver.nz - halo)
        return;

    int spatial_size = solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    int u_idx = b * spatial_size + idx;
    int src_idx = (b * nrec + s) * solver.nt + it;

    float drho_x_drho, drho_z_drho;
    apm_rho_jacobian(category[idx], &drho_x_drho, &drho_z_drho);
    const float jac = is_z_component ? drho_z_drho : drho_x_drho;

    atomicAdd(&grad_rho[u_idx],
              -adjoint_source[src_idx] * (fv_now[u_idx] - fv_next[u_idx])
                  / rho_eff[u_idx] * jac);
}

namespace elastic2d {

namespace {

// Stress-adjoint half of the adjoint step (prepare + apply): reads the
// adjoint stresses (same-cell), writes the adjoint velocities.
void apply_stress_adjoint_2d(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    const torch::Tensor& lambda,
    const torch::Tensor& mu,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    ElasticAdjointWorkspaceTensor& workspace,
    // Optional fused per-step gradient imaging (full-mode only).  When the
    // grad_*_out accumulators are non-null, the stress-adjoint-prepare kernel
    // folds in the calculate_grad_elastic_nobs correlation for this reverse
    // step (operands are the un-mutated post-source adjoint at kernel entry).
    // Default null => behaviour byte-for-byte identical for bs/ckpt callers.
    const float* grad_fvx       = nullptr,
    const float* grad_fvz       = nullptr,
    const float* grad_fvx_prev  = nullptr,
    const float* grad_fvz_prev  = nullptr,
    const float* grad_vp_model  = nullptr,
    const float* grad_vs_model  = nullptr,
    const float* grad_rho_model = nullptr,
    float* grad_vp_out          = nullptr,
    float* grad_vs_out          = nullptr,
    float* grad_rho_out         = nullptr
)
{
    auto adj_view = adjoint.view();

    LAUNCH_ELASTIC_STRESS_ADJOINT_PREPARE(
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
        workspace.qzx_t.data_ptr<float>(),
        grad_ctx,
        grad_fvx, grad_fvz, grad_fvx_prev, grad_fvz_prev,
        grad_vp_model, grad_vs_model, grad_rho_model,
        grad_vp_out, grad_vs_out, grad_rho_out
    );

    LAUNCH_ELASTIC_STRESS_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        workspace.qxx_t.data_ptr<float>(),
        workspace.qzz_t.data_ptr<float>(),
        workspace.qxz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>(),
        grad_ctx,
        solver
    );

}

// Velocity-adjoint half of the adjoint step (prepare + apply): reads the
// adjoint velocities (same-cell), writes the adjoint stresses.
void apply_velocity_adjoint_2d(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    const torch::Tensor& rho,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    ElasticAdjointWorkspaceTensor& workspace
)
{
    auto adj_view = adjoint.view();

    LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
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
        adj_view,
        workspace.pxx_t.data_ptr<float>(),
        workspace.pzz_t.data_ptr<float>(),
        workspace.pxz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(),
        grad_ctx,
        solver
    );
}

void apply_adjoint_step_2d(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    const torch::Tensor& lambda,
    const torch::Tensor& mu,
    const torch::Tensor& rho,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    ElasticAdjointWorkspaceTensor& workspace,
    // Optional fused per-step gradient imaging (full-mode only).  When the
    // grad_*_out accumulators are non-null, the stress-adjoint-prepare kernel
    // folds in the calculate_grad_elastic_nobs correlation for this reverse
    // step (operands are the un-mutated post-source adjoint at kernel entry).
    // Default null => behaviour byte-for-byte identical for bs/ckpt callers.
    const float* grad_fvx       = nullptr,
    const float* grad_fvz       = nullptr,
    const float* grad_fvx_prev  = nullptr,
    const float* grad_fvz_prev  = nullptr,
    const float* grad_vp_model  = nullptr,
    const float* grad_vs_model  = nullptr,
    const float* grad_rho_model = nullptr,
    float* grad_vp_out          = nullptr,
    float* grad_vs_out          = nullptr,
    float* grad_rho_out         = nullptr
)
{
    auto adj_view = adjoint.view();

    LAUNCH_ELASTIC_STRESS_ADJOINT_PREPARE(
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
        workspace.qzx_t.data_ptr<float>(),
        grad_ctx,
        grad_fvx, grad_fvz, grad_fvx_prev, grad_fvz_prev,
        grad_vp_model, grad_vs_model, grad_rho_model,
        grad_vp_out, grad_vs_out, grad_rho_out
    );

    LAUNCH_ELASTIC_STRESS_ADJOINT_APPLY(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
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
        adj_view,
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
        adj_view,
        workspace.pxx_t.data_ptr<float>(),
        workspace.pzz_t.data_ptr<float>(),
        workspace.pxz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(),
        grad_ctx,
        solver
    );
}

// Validate the stepped-backward segment fields (bw_it_begin/bw_it_end),
// the DD cut mask and the backward phase split for the elastic2d entry
// points.  ``need_recon`` is true for boundary-saving mode, where the
// 7-tensor reconstruction list must be Python-owned to survive segments.
void check_stepped_backward_elastic_2d(const BackwardInput& p, bool need_recon)
{
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    TORCH_CHECK(0 <= it_lo && it_lo < it_hi && it_hi <= static_cast<int>(p.nt),
                "stepped backward: require 0 <= bw_it_end < bw_it_begin <= nt, got [",
                it_lo, ", ", it_hi, ") with nt=", p.nt);
    TORCH_CHECK((p.cut_face_mask & ~0xF) == 0,
                "2D cut_face_mask uses bits 0..3 (x_lo, x_hi, z_lo, z_hi) only, got ",
                p.cut_face_mask);
    TORCH_CHECK(p.step_phase >= 0 && p.step_phase <= 2,
                "elastic backward step_phase must be 0, 1 or 2");
    const bool phased = (p.step_phase != 0);
    if (phased) {
        TORCH_CHECK(need_recon,
                    "phased elastic backward (step_phase) is only supported "
                    "for the boundary-saving path (backward_bs)");
        TORCH_CHECK(it_hi == it_lo + 1,
                    "elastic backward phase-split requires a single-step "
                    "segment (bw_it_begin == bw_it_end + 1)");
    }
    if (need_recon && p.cut_face_mask != 0) {
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "domain-decomposed backward_bs (cut_face_mask) supports "
                    "gpu-direct boundary storage only "
                    "(boundary_on_cpu/boundary_on_disk unsupported in v1)");
    }
    if (!p.bw_stepped() && !phased)
        return;
    TORCH_CHECK(p.adjoint_wavefields.size() == 15,
                "stepped elastic backward requires the 15-tensor adjoint "
                "wavefield list bound from Python");
    TORCH_CHECK(p.grads_out.size() == p.models.size(),
                "stepped elastic backward requires Python-bound grads_out "
                "(one per model: vp, vs, rho — elastic computes no "
                "grad_wavelet)");
    TORCH_CHECK(p.illum_out.empty(),
                "elastic backward computes no illuminations; illum_out must "
                "be empty");
    if (need_recon) {
        TORCH_CHECK(p.forward_wavefields.size() == 7,
                    "stepped elastic backward_bs requires the 7-tensor "
                    "reconstruction list [vx, vz, sxx, szz, sxz, fvx_prev, "
                    "fvz_prev] bound from Python");
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "stepped backward_bs supports gpu-direct boundary storage "
                    "only (boundary_on_cpu/boundary_on_disk unsupported in v1)");
    }
}

// Bind the three model-gradient accumulators from Python when provided
// (stepped), else fall back to internal zero allocation (legacy
// monolithic behaviour).  Bound tensors are accumulated "+=" and NOT
// zeroed here — Python zeroes them once before the first segment.
void bind_elastic_grads(
    const BackwardInput& p,
    torch::Tensor& grad_vp,
    torch::Tensor& grad_vs,
    torch::Tensor& grad_rho)
{
    if (!p.grads_out.empty()) {
        TORCH_CHECK(p.grads_out.size() == 3,
                    "elastic grads_out must hold exactly {grad_vp, grad_vs, "
                    "grad_rho}");
        grad_vp = p.grads_out[0];
        grad_vs = p.grads_out[1];
        grad_rho = p.grads_out[2];
    } else {
        grad_vp = torch::zeros_like(p.models[0]);
        grad_vs = torch::zeros_like(p.models[0]);
        grad_rho = torch::zeros_like(p.models[0]);
    }
}

// Undo the just-injected receiver residual from this reverse step's rho
// imaging, at every velocity-receiver cell (see the kernel comment in
// common.cuh).  Stress receivers have no rho term to correct.  Call it right
// after the step's imaging launch, while ``fv*_now`` / ``fv*_next`` still point
// at the operands the imaging correlated.
void undo_receiver_rho_injection_2d(
    const fdtd::LaunchConfig& adj_source_config,
    torch::Tensor& grad_rho,
    const float* fvx_now,
    const float* fvz_now,
    const float* fvx_next,
    const float* fvz_next,
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
        const float* fv_now  = (field == 0) ? fvx_now  : (field == 1) ? fvz_now  : nullptr;
        const float* fv_next = (field == 0) ? fvx_next : (field == 1) ? fvz_next : nullptr;
        if (fv_now == nullptr) continue;              // stress receiver: no rho term
        sub_receiver_rho_grad_correction<<<adj_source_config.grid, adj_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            fv_now,
            fv_next,
            rho.data_ptr<float>(),
            p.adjoint_source[irec].data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            2,
            p.M,                                      // imaging halo (order/2 == M)
            solver
        );
    }
}

// Body-force (velocity) sources: the rho imaging correlates the adjoint
// velocity with v(it) - v(it+1), which at a source cell still contains the raw
// injected amplitude amp(it+1); the true derivative has no such term (the
// injection is rho-independent).  Compensate at the source cells.
//
// Must run BEFORE this step's receiver residuals are injected.  At a cell that
// is BOTH a source and a receiver the post-injection adjoint velocity carries
// the residual as well, and the correction then over-shoots by
// resid * amp / rho — which is exactly the case (body-force source with a
// velocity receiver on the same cell) where impl='c' used to return ~2x the
// true d(loss)/d(rho) in 2-D and ~3.4x in 3-D.
void undo_body_force_source_injection_2d(
    const fdtd::LaunchConfig& fwd_source_config,
    torch::Tensor& grad_rho,
    ElasticWavefieldPointer& adj_view,
    const torch::Tensor& rho,
    const BackwardInput& p,
    const torch::Tensor& source_fields,
    int it,
    const SolverContext& solver
)
{
    if (it < 0 || it >= p.nt) return;
    for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
        const int sfield = source_fields[isrc].item<int>();
        if (sfield > 1) continue;                       // vx = 0, vz = 1
        float* adj_field = elastic_field_ptr(adj_view, 2, sfield);
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

// APM variant of the above: the APM kinetic rho term uses the per-component
// effective density and its Jacobian, so it needs its own kernel.
void undo_receiver_rho_injection_apm_2d(
    const fdtd::LaunchConfig& adj_source_config,
    torch::Tensor& grad_rho,
    const float* fvx_now,
    const float* fvz_now,
    const float* fvx_next,
    const float* fvz_next,
    const torch::Tensor& rho_x_eff,
    const torch::Tensor& rho_z_eff,
    const torch::Tensor& category,
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
        if (field > 1) continue;                      // stress receiver: no rho term
        const bool is_z = (field == 1);
        sub_receiver_rho_grad_correction_apm_2d<<<adj_source_config.grid, adj_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            is_z ? fvz_now : fvx_now,
            is_z ? fvz_next : fvx_next,
            (is_z ? rho_z_eff : rho_x_eff).data_ptr<float>(),
            category.data_ptr<int>(),
            p.adjoint_source[irec].data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            adjoint_nsrc,
            p.M,
            is_z ? 1 : 0,
            solver
        );
    }
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
    ElasticWavefieldTensor& forward,
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

        LAUNCH_ELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_STRESS(
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
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
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
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 2);

    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto mu  = rho * vs * vs;
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    auto seg_vx = torch::zeros({segment_len + 1, vp.size(0) * vp.size(1), 1, vp.size(2), vp.size(3)}, vp.options());
    auto seg_vz = torch::zeros_like(seg_vx);
    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(vp, 2, true);

    checkpoint_runtime.copy_state(forward.state_tensors(), start_state.state_tensors());

    seg_vx.select(0, 0).copy_(forward.vx_t);
    seg_vz.select(0, 0).copy_(forward.vz_t);

    for (int it = start; it < end; ++it) {
        auto for_view = forward.view();

        LAUNCH_ELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_STRESS(
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
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
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

        undo_body_force_source_injection_2d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
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

        const float* seg_vx_now  = seg_vx.select(0, now_offset).data_ptr<float>();
        const float* seg_vz_now  = seg_vz.select(0, now_offset).data_ptr<float>();
        const float* seg_vx_next = (next_offset <= segment_len)
            ? seg_vx.select(0, next_offset).data_ptr<float>() : next_segment_vx.data_ptr<float>();
        const float* seg_vz_next = (next_offset <= segment_len)
            ? seg_vz.select(0, next_offset).data_ptr<float>() : next_segment_vz.data_ptr<float>();

        LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            seg_vx_now,
            seg_vz_now,
            seg_vx_next,
            seg_vz_next,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        undo_receiver_rho_injection_2d(
            adj_source_config, grad_rho,
            seg_vx_now, seg_vz_now, seg_vx_next, seg_vz_next,
            rho, p, receiver_fields, it, adjoint_nsrc, solver
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
    check_stepped_backward_elastic_2d(p, /*need_recon=*/false);
    // Segment bounds: process [it_lo, it_hi) in descending order.  Defaults
    // (bw_it_begin = -1 => nt, bw_it_end = 0) reproduce the monolithic call.
    // it == 0 keeps its legacy asymmetry (gradient only, no adjoint step)
    // and runs in whichever segment contains it — position-based, so any
    // partition reproduces the monolithic loop bit-for-bit.
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
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
    solver.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    if (p.has_topo) { solver.topo_rows = p.topo_rows.data_ptr<int>(); solver.has_topo = true; }
    // DD: cut-aware PML predicates in the adjoint prepare kernels.
    solver.cut_mask = p.cut_face_mask;

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    auto adj_view = adjoint.view();

    torch::Tensor grad_vp, grad_vs, grad_rho;
    bind_elastic_grads(p, grad_vp, grad_vs, grad_rho);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    // PML coefficients
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto zero_velocity = torch::zeros_like(vp);
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 2);

    // it_hi == nt and it_lo == 0 without DD, so this is dev's full
    // reverse loop verbatim in the single-domain case.
    for (int it = it_hi - 1; it >= it_lo; --it) {
        undo_body_force_source_injection_2d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
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

        const float* vx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* vz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* vx_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vz_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero_velocity.data_ptr<float>();

        // Reverse step 0 has no adjoint apply kernel (the loop `continue`s
        // below), so its gradient imaging cannot be folded — emit it as a
        // standalone calculate_grad pass (mirrors the acoustic trailing call).
        if (it == 0) {
            LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
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
            undo_receiver_rho_injection_2d(
                source_config, grad_rho, vx_now, vz_now, vx_next, vz_next,
                rho, p, receiver_fields, it, adjoint_nsrc, solver
            );
            continue;
        }

        // FULL-mode fusion: fold this reverse step's vp/vs/rho-gradient imaging
        // into the stress-adjoint-prepare kernel (it reads the un-mutated
        // post-source adjoint[it] stress+velocity at entry, exactly what
        // calculate_grad_elastic_nobs(it) would correlate), eliminating the
        // separate full-grid calculate_grad launch for steps it >= 1.
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
            workspace,
            vx_now, vz_now, vx_next, vz_next,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>()
        );

        undo_receiver_rho_injection_2d(
            source_config, grad_rho, vx_now, vz_now, vx_next, vz_next,
            rho, p, receiver_fields, it, adjoint_nsrc, solver
        );

    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "checkpoint backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 15, "Elastic 2D checkpointing expects 15 checkpoint tensors");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        15,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "elastic2d"
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
    solver.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    if (p.has_topo) { solver.topo_rows = p.topo_rows.data_ptr<int>(); solver.has_topo = true; }
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2, true);
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

    ElasticWavefieldTensor start_state;
    start_state.allocate(vp, 2, true);
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
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "checkpoint backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

    TORCH_CHECK(p.checkpoints.size() == 15, "Elastic 2D recursive checkpointing expects 15 checkpoint tensors");

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        15,
        true,
        true,
        p.checkpoint_interval,
        checkpoint_steps_cpu,
        p.checkpoint_on_cpu,
        "backward_recursive",
        "elastic2d"
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
    solver.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    if (p.has_topo) { solver.topo_rows = p.topo_rows.data_ptr<int>(); solver.has_topo = true; }
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2, true);
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
    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, true);
    else
        forward.allocate(vp, 2, true);
    auto current_vx = torch::zeros_like(vp);
    auto current_vz = torch::zeros_like(vp);
    auto next_vx = torch::zeros_like(vp);
    auto next_vz = torch::zeros_like(vp);
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 2);

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        undo_body_force_source_injection_2d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

        for (int irec = 0; irec < static_cast<int>(p.receiver_field_indices.numel()); ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                p.adjoint_sources_loc.size(1),
                solver
            );
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
            adj_view,
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

        undo_receiver_rho_injection_2d(
            adj_source_config, grad_rho,
            current_vx.data_ptr<float>(), current_vz.data_ptr<float>(),
            next_vx.data_ptr<float>(), next_vz.data_ptr<float>(),
            rho, p, receiver_fields, it,
            (int)p.adjoint_sources_loc.size(1), solver
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

    check_stepped_backward_elastic_2d(p, /*need_recon=*/true);
    // Segment bounds: process [it_lo, it_hi) in descending order.  Defaults
    // (bw_it_begin = -1 => nt, bw_it_end = 0) reproduce the monolithic call.
    // The elastic BS loop legacy floor is it == 1 (no it==0 tail), so the
    // segment containing it == 0 simply runs nothing extra.
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    const bool first_segment = (it_hi == static_cast<int>(p.nt));
    // Phase split (DD): 1 = inject + stress recon/restore + gradient +
    // stress adjoint; 2 = velocity adjoint + fv_prev capture + velocity
    // recon/restore.  The driver exchanges the adjoint-velocity and
    // recon-stress halos between phases, and the adjoint-stress and
    // recon-velocity halos after phase 2 (each M wide).
    const bool do_p1 = (p.step_phase != 2);
    const bool do_p2 = (p.step_phase != 1);

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
    solver.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    if (p.has_topo) { solver.topo_rows = p.topo_rows.data_ptr<int>(); solver.has_topo = true; }
    // DD: skip cut faces in the strip restore, collapse the NOPML exclusion
    // bands to the stencil halo on cut sides, and route cut-side cells of
    // the adjoint prepare kernels through the interior branch.
    solver.cut_mask = p.cut_face_mask;
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2);

    // Reconstruction state: 5 physical fields plus the carried fv*_prev
    // velocities (v at time it+1, consumed by the gradient kernel).  When
    // stepping, all 7 must be Python-owned to survive segment boundaries.
    ElasticWavefieldTensor forward;
    torch::Tensor fvx_prev, fvz_prev;
    if (!p.forward_wavefields.empty()) {
        TORCH_CHECK(p.forward_wavefields.size() == 7,
                    "elastic2d backward_bs reconstruction list must hold 7 "
                    "tensors [vx, vz, sxx, szz, sxz, fvx_prev, fvz_prev]; got ",
                    p.forward_wavefields.size());
        forward.bind(std::vector<torch::Tensor>(p.forward_wavefields.begin(),
                                                p.forward_wavefields.begin() + 5),
                     /*use_pml=*/false);
        fvx_prev = p.forward_wavefields[5];
        fvz_prev = p.forward_wavefields[6];
    } else {
        forward.allocate(vp, 2, false);
        fvx_prev = torch::zeros_like(vp);
        fvz_prev = torch::zeros_like(vp);
    }

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Seed the reverse reconstruction from the saved last snapshot — FIRST
    // segment only (and not on a phase-2 re-entry); re-running this
    // mid-stream would clobber the carried reconstruction state.
    if (first_segment && do_p1) {
        forward.vx_t.copy_(p.u_last_two.select(0,0).select(0,0));
        forward.vz_t.copy_(p.u_last_two.select(0,1).select(0,0));
        forward.sxx_t.copy_(p.u_last_two.select(0,2).select(0,0));
        forward.szz_t.copy_(p.u_last_two.select(0,3).select(0,0));
        forward.sxz_t.copy_(p.u_last_two.select(0,4).select(0,0));
    }

    auto neg_forward_source = -p.forward_source;

    // Generate pointer views
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    torch::Tensor grad_vp, grad_vs, grad_rho;
    bind_elastic_grads(p, grad_vp, grad_vs, grad_rho);
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
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, true, 1, {}, p.boundary_gpu, {}, p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

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

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 2);

    for (int it = it_hi - 1; it >= std::max(it_lo, 1); --it) {

        if (do_p1) {

        undo_body_force_source_injection_2d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
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

        // Same operands the imaging just correlated: for_view.v* is v(it),
        // fv*_prev is v(it+1) (overwritten a few lines below).
        undo_receiver_rho_injection_2d(
            adj_source_config, grad_rho,
            for_view.vx, for_view.vz,
            fvx_prev.data_ptr<float>(), fvz_prev.data_ptr<float>(),
            rho, p, receiver_fields, it, adjoint_nsrc, solver
        );

        apply_stress_adjoint_2d(
            order,
            launch_config,
            adjoint,
            lambda,
            mu,
            cpml_view,
            grad_ctx,
            solver,
            workspace
        );

        }  // do_p1

        if (do_p2) {

        apply_velocity_adjoint_2d(
            order,
            launch_config,
            adjoint,
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
            for_view,
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

        }  // do_p2
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;

}

// ---------------------------------------------------------------------------
// APM backward — full & boundary-saving modes.
// ---------------------------------------------------------------------------
// Mirror image-method ``backward()`` / ``backward_bs()`` but consume the
// 11-tensor APM model layout
//   p.models = [vp, vs, rho, lam, mu, lam_2mu, lam_eff, mu_eff, mu_xz,
//               rho_x_eff, rho_z_eff]
// (set by _c.py:1087 in APM mode) plus ``p.topo_category``.
//
// Adjoint kernels (stress/velocity prepare, gradient kernels, nopml replay)
// are APM-specific (per-category Jacobian chain back to raw vp/vs/rho).
// The ``_adjoint_apply`` kernels (q/p -> v/sigma) don't touch moduli and
// are shared with the image path.

namespace {

void apply_adjoint_step_apm_2d(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    const torch::Tensor& lam_eff,
    const torch::Tensor& mu_eff,
    const torch::Tensor& mu_xz_node,
    const torch::Tensor& rho_x_eff,
    const torch::Tensor& rho_z_eff,
    const torch::Tensor& category,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    ElasticAdjointWorkspaceTensor& workspace
)
{
    auto adj_view = adjoint.view();

    LAUNCH_ELASTIC_STRESS_ADJOINT_PREPARE_APM(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        lam_eff.data_ptr<float>(),
        mu_eff.data_ptr<float>(),
        mu_xz_node.data_ptr<float>(),
        category.data_ptr<int>(),
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
        adj_view,
        workspace.qxx_t.data_ptr<float>(),
        workspace.qzz_t.data_ptr<float>(),
        workspace.qxz_t.data_ptr<float>(),
        workspace.qzx_t.data_ptr<float>(),
        grad_ctx,
        solver
    );

    LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE_APM(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        rho_x_eff.data_ptr<float>(),
        rho_z_eff.data_ptr<float>(),
        category.data_ptr<int>(),
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
        adj_view,
        workspace.pxx_t.data_ptr<float>(),
        workspace.pzz_t.data_ptr<float>(),
        workspace.pxz_t.data_ptr<float>(),
        workspace.pzx_t.data_ptr<float>(),
        grad_ctx,
        solver
    );
}

}  // namespace

BackwardOutput apm_backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "APM backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

    TORCH_CHECK(p.models.size() >= 11,
        "elastic2d::apm_backward expects 11-tensor models list "
        "[vp,vs,rho,lam,mu,lam_2mu,lam_eff,mu_eff,mu_xz,rho_x,rho_z]; got ",
        p.models.size());
    TORCH_CHECK(p.use_apm && p.topo_category.defined() && p.topo_category.numel() > 0,
        "apm_backward requires use_apm=true and topo_category tensor");

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    auto vp        = p.models[0];
    auto vs        = p.models[1];
    auto rho       = p.models[2];
    auto lam_raw   = p.models[3];
    auto mu_raw    = p.models[4];
    auto lam_eff   = p.models[6];
    auto mu_eff    = p.models[7];
    auto mu_xz_n   = p.models[8];
    auto rho_x_eff = p.models[9];
    auto rho_z_eff = p.models[10];

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

    // APM forces free_surface=false (image-method z-derivative substitution is
    // off).  Plumb topo_category through ctx so adjoint+grad kernels can read it.
    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, /*free_surface=*/false,
                         p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                         dx, 0.f, dz};
    solver.topo_category = p.topo_category.data_ptr<int>();
    solver.use_apm = true;

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2);
    auto adj_view = adjoint.view();

    auto grad_vp  = torch::zeros_like(vp);
    auto grad_vs  = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto zero_velocity = torch::zeros_like(vp);
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 2);

    for (int it = p.nt - 1; it >= 0; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
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

        const float* vx_now  = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* vz_now  = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* vx_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vz_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero_velocity.data_ptr<float>();

        LAUNCH_CALCULATE_GRAD_ELASTIC_APM_NOBS(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            vx_now, vz_now,
            vx_next, vz_next,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            lam_raw.data_ptr<float>(),
            mu_raw.data_ptr<float>(),
            rho_x_eff.data_ptr<float>(),
            rho_z_eff.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        undo_receiver_rho_injection_apm_2d(
            source_config, grad_rho, vx_now, vz_now, vx_next, vz_next,
            rho_x_eff, rho_z_eff, p.topo_category,
            p, receiver_fields, it, adjoint_nsrc, solver
        );

        if (it == 0) continue;

        apply_adjoint_step_apm_2d(
            order, launch_config, adjoint,
            lam_eff, mu_eff, mu_xz_n, rho_x_eff, rho_z_eff,
            p.topo_category,
            cpml_view, grad_ctx, solver, workspace
        );
    }

    // Return 11 grads matching the 11-tensor model layout.
    // Chain rule from (lam_eff,...,rho_z) -> (lam,mu,rho) -> (vp,vs,rho) is
    // done inside the gradient kernel, so positions 3..10 are zero (autograd
    // adds zero into the upstream leaves vp/vs/rho).
    auto z = torch::zeros_like(vp);
    out.grads = {grad_vp, grad_vs, grad_rho, z, z, z, z, z, z, z, z};
    return out;
}

BackwardOutput apm_backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "APM backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

    TORCH_CHECK(p.models.size() >= 11,
        "elastic2d::apm_backward_bs expects 11-tensor models list");
    TORCH_CHECK(p.use_apm && p.topo_category.defined() && p.topo_category.numel() > 0,
        "apm_backward_bs requires use_apm=true and topo_category tensor");

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    auto vp        = p.models[0];
    auto vs        = p.models[1];
    auto rho       = p.models[2];
    auto lam_raw   = p.models[3];
    auto mu_raw    = p.models[4];
    auto lam_eff   = p.models[6];
    auto mu_eff    = p.models[7];
    auto mu_xz_n   = p.models[8];
    auto rho_x_eff = p.models[9];
    auto rho_z_eff = p.models[10];

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

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, /*free_surface=*/false,
                         p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                         dx, 0.f, dz};
    solver.topo_category = p.topo_category.data_ptr<int>();
    solver.use_apm = true;
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2);
    ElasticWavefieldTensor forward;
    forward.allocate(vp, 2, false);
    forward.vx_t.copy_(p.u_last_two.select(0,0).select(0,0));
    forward.vz_t.copy_(p.u_last_two.select(0,1).select(0,0));
    forward.sxx_t.copy_(p.u_last_two.select(0,2).select(0,0));
    forward.szz_t.copy_(p.u_last_two.select(0,3).select(0,0));
    forward.sxz_t.copy_(p.u_last_two.select(0,4).select(0,0));

    auto neg_forward_source = -p.forward_source;
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    auto grad_vp  = torch::zeros_like(vp);
    auto grad_vs  = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 2);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, true,
                                1, {}, p.boundary_gpu, {}, p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto fvz_prev = torch::zeros_like(vp);
    auto fvx_prev = torch::zeros_like(vp);

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver, 2, true,
        p.boundary_on_cpu, p.boundary_on_disk, p.boundary_disk_async_read,
        p.transfer_interval, p.boundary_ring_buffers, p.boundary_disk_files,
        async_copy.compute_stream, async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 2);

    for (int it = p.nt - 1; it >= 1; --it) {
        // Adjoint source injection
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it, adjoint_nsrc, solver
            );
        }

        // Reverse forward replay: inject -source then reverse stress step
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it, forward_nsrc, solver
            );
        }

        LAUNCH_ELASTIC_STRESS_NOPML_APM(
            order, launch_config.grid, launch_config.block,
            for_view,
            lam_eff.data_ptr<float>(),
            mu_eff.data_ptr<float>(),
            mu_xz_n.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_ctx, solver
        );

        float* field2[3] = {for_view.sxx, for_view.szz, for_view.sxz};
        for (int f = 2; f < 5; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, field2[f-2], launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver, f, f == 2, false
            );
        }

        LAUNCH_CALCULATE_GRAD_ELASTIC_APM_BS(
            order, launch_config.grid, launch_config.block,
            for_view, adj_view,
            fvx_prev.data_ptr<float>(),
            fvz_prev.data_ptr<float>(),
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            lam_raw.data_ptr<float>(),
            mu_raw.data_ptr<float>(),
            rho_x_eff.data_ptr<float>(),
            rho_z_eff.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx, solver
        );

        // Same operands the imaging just correlated: for_view.v* is v(it),
        // fv*_prev is v(it+1) (overwritten a few lines below).
        undo_receiver_rho_injection_apm_2d(
            adj_source_config, grad_rho,
            for_view.vx, for_view.vz,
            fvx_prev.data_ptr<float>(), fvz_prev.data_ptr<float>(),
            rho_x_eff, rho_z_eff, p.topo_category,
            p, receiver_fields, it, adjoint_nsrc, solver
        );

        apply_adjoint_step_apm_2d(
            order, launch_config, adjoint,
            lam_eff, mu_eff, mu_xz_n, rho_x_eff, rho_z_eff,
            p.topo_category,
            cpml_view, grad_ctx, solver, workspace
        );

        fvz_prev.copy_(forward.vz_t);
        fvx_prev.copy_(forward.vx_t);

        LAUNCH_ELASTIC_VELOCITY_NOPML_APM(
            order, launch_config.grid, launch_config.block,
            for_view,
            rho_x_eff.data_ptr<float>(),
            rho_z_eff.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_ctx, solver
        );

        float* field1[2] = {for_view.vx, for_view.vz};
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, field1[f], launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver, f, false, f == 1
            );
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    auto z = torch::zeros_like(vp);
    out.grads = {grad_vp, grad_vs, grad_rho, z, z, z, z, z, z, z, z};
    return out;
}

}
