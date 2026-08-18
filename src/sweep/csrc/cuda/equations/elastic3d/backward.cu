#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>

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

// Undo the just-injected receiver residual from this reverse step's rho
// imaging, at every velocity-receiver cell (see the kernel comment in
// common.cuh).  Stress receivers have no rho term to correct.  Call it right
// after the step's imaging launch, while ``fv*_now`` / ``fv*_next`` still point
// at the operands the imaging correlated.  The 3-D APM rho term shares the
// image-method form (plain rho, no per-component effective density), so this
// covers the APM paths too.
void undo_receiver_rho_injection_3d(
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

// Body-force (velocity) sources: the rho imaging correlates the adjoint
// velocity with the stored v-difference, which at a source cell still contains
// the raw injected amplitude; the true derivative has no such term.  Compensate
// at the source cells.
//
// Must run BEFORE this step's receiver residuals are injected — at a cell that
// is BOTH a source and a receiver the post-injection adjoint velocity carries
// the residual too and the correction over-shoots by resid * amp / rho (that is
// the case where impl='c' used to return ~3.4x the true d(loss)/d(rho) in 3-D).
void undo_body_force_source_injection_3d(
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
        if (sfield > 2) continue;                       // vx = 0, vy = 1, vz = 2
        float* adj_field = elastic_field_ptr(adj_view, 3, sfield);
        if (adj_field == nullptr) continue;
        add_body_force_rho_grad_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            adj_field,
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

// Stress-adjoint half of the adjoint step (prepare + apply): reads the
// adjoint stresses (same-cell), writes the adjoint velocities.
void apply_stress_adjoint_3d(
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
    // folds in the calculate_grad_elastic3d_bs correlation for this reverse
    // step (operands are the un-mutated post-source adjoint at kernel entry).
    // Default null => behaviour byte-for-byte identical for bs/ckpt callers.
    const float* grad_fvx       = nullptr,
    const float* grad_fvy       = nullptr,
    const float* grad_fvz       = nullptr,
    const float* grad_fvx_prev  = nullptr,
    const float* grad_fvy_prev  = nullptr,
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
        workspace.qzz_t.data_ptr<float>(),
        grad_ctx,
        grad_fvx, grad_fvy, grad_fvz,
        grad_fvx_prev, grad_fvy_prev, grad_fvz_prev,
        grad_vp_model, grad_vs_model, grad_rho_model,
        grad_vp_out, grad_vs_out, grad_rho_out
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

}

// Velocity-adjoint half of the adjoint step (prepare + apply): reads the
// adjoint velocities (same-cell), writes the adjoint stresses.
void apply_velocity_adjoint_3d(
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
    ElasticAdjointWorkspaceTensor& workspace,
    // Optional fused per-step gradient imaging (full-mode only).  When the
    // grad_*_out accumulators are non-null, the stress-adjoint-prepare kernel
    // folds in the calculate_grad_elastic3d_bs correlation for this reverse
    // step (operands are the un-mutated post-source adjoint at kernel entry).
    // Default null => behaviour byte-for-byte identical for bs/ckpt callers.
    const float* grad_fvx       = nullptr,
    const float* grad_fvy       = nullptr,
    const float* grad_fvz       = nullptr,
    const float* grad_fvx_prev  = nullptr,
    const float* grad_fvy_prev  = nullptr,
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
        workspace.qzz_t.data_ptr<float>(),
        grad_ctx,
        grad_fvx, grad_fvy, grad_fvz,
        grad_fvx_prev, grad_fvy_prev, grad_fvz_prev,
        grad_vp_model, grad_vs_model, grad_rho_model,
        grad_vp_out, grad_vs_out, grad_rho_out
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

// Validate the stepped-backward segment fields (bw_it_begin/bw_it_end),
// the DD cut mask and the backward phase split for the elastic3d entry
// points.  ``need_recon`` is true for boundary-saving mode, where the
// 12-tensor reconstruction list must be Python-owned to survive segments.
void check_stepped_backward_elastic_3d(const BackwardInput& p, bool need_recon)
{
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    TORCH_CHECK(0 <= it_lo && it_lo < it_hi && it_hi <= static_cast<int>(p.nt),
                "stepped backward: require 0 <= bw_it_end < bw_it_begin <= nt, got [",
                it_lo, ", ", it_hi, ") with nt=", p.nt);
    TORCH_CHECK((p.cut_face_mask & ~0x33) == 0,
                "elastic3d backward cut_face_mask supports x/y bits only "
                "(bit0=x_lo, bit1=x_hi, bit4=y_lo, bit5=y_hi), got ",
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
    TORCH_CHECK(p.adjoint_wavefields.size() == 36,
                "stepped elastic backward requires the 36-tensor adjoint "
                "wavefield list bound from Python");
    TORCH_CHECK(p.grads_out.size() == p.models.size(),
                "stepped elastic backward requires Python-bound grads_out "
                "(one per model: vp, vs, rho — elastic computes no "
                "grad_wavelet)");
    TORCH_CHECK(p.illum_out.empty(),
                "elastic backward computes no illuminations; illum_out must "
                "be empty");
    if (need_recon) {
        TORCH_CHECK(p.forward_wavefields.size() == 12,
                    "stepped elastic backward_bs requires the 12-tensor "
                    "reconstruction list [vx, vy, vz, sxx, syy, szz, sxy, "
                    "sxz, syz, fvx_prev, fvy_prev, fvz_prev] bound from "
                    "Python");
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "stepped backward_bs supports gpu-direct boundary storage "
                    "only (boundary_on_cpu/boundary_on_disk unsupported in v1)");
    }
}

// Bind the three model-gradient accumulators from Python when provided
// (stepped), else fall back to internal zero allocation (legacy
// monolithic behaviour).  Bound tensors are accumulated "+=" and NOT
// zeroed here — Python zeroes them once before the first segment.
void bind_elastic_grads_3d(
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

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = end - 1; it >= start; --it) {
        auto adj_view = adjoint.view();

        undo_body_force_source_injection_3d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

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
        auto current_forward = make_velocity_view_3d(
            seg_vx.select(0, now_offset),
            seg_vy.select(0, now_offset),
            seg_vz.select(0, now_offset)
        );

        const float* seg_now[3] = {
            seg_vx.select(0, now_offset).data_ptr<float>(),
            seg_vy.select(0, now_offset).data_ptr<float>(),
            seg_vz.select(0, now_offset).data_ptr<float>(),
        };
        const float* seg_next[3] = {
            (next_offset <= segment_len) ? seg_vx.select(0, next_offset).data_ptr<float>() : next_segment_vx.data_ptr<float>(),
            (next_offset <= segment_len) ? seg_vy.select(0, next_offset).data_ptr<float>() : next_segment_vy.data_ptr<float>(),
            (next_offset <= segment_len) ? seg_vz.select(0, next_offset).data_ptr<float>() : next_segment_vz.data_ptr<float>(),
        };

        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            order,
            launch_config.grid,
            launch_config.block,
            current_forward.view(),
            adj_view,
            seg_next[0],
            seg_next[1],
            seg_next[2],
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        undo_receiver_rho_injection_3d(
            adj_source_config, grad_rho, seg_now, seg_next,
            rho, p, receiver_fields, it, adjoint_nsrc, solver
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

    const auto& p = in;
    BackwardOutput out;

    check_stepped_backward_elastic_3d(p, /*need_recon=*/true);
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
    // DD: skip cut faces in the strip restore, collapse the NOPML exclusion
    // bands to the stencil halo on cut sides, and route cut-side cells of
    // the adjoint prepare kernels through the interior branch.
    solver.cut_mask = p.cut_face_mask;

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 3);

    // Reconstruction state: 9 physical fields plus the carried fv*_prev
    // velocities (v at time it+1, consumed by the gradient kernel).  When
    // stepping, all 12 must be Python-owned to survive segment boundaries.
    ElasticWavefieldTensor forward;
    torch::Tensor fvx_prev, fvy_prev, fvz_prev;
    if (!p.forward_wavefields.empty()) {
        if (p.forward_wavefields.size() == 12) {
            forward.bind(std::vector<torch::Tensor>(p.forward_wavefields.begin(),
                                                    p.forward_wavefields.begin() + 9),
                         /*use_pml=*/false);
            fvx_prev = p.forward_wavefields[9];
            fvy_prev = p.forward_wavefields[10];
            fvz_prev = p.forward_wavefields[11];
        } else {
            forward.bind(p.forward_wavefields, false);
            fvx_prev = torch::zeros_like(vp);
            fvy_prev = torch::zeros_like(vp);
            fvz_prev = torch::zeros_like(vp);
        }
    } else {
        forward.allocate(vp, 3, false);
        fvx_prev = torch::zeros_like(vp);
        fvy_prev = torch::zeros_like(vp);
        fvz_prev = torch::zeros_like(vp);
    }

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    // Seed the reverse reconstruction from the saved last snapshot — FIRST
    // segment only (and not on a phase-2 re-entry); re-running this
    // mid-stream would clobber the carried reconstruction state.
    if (first_segment && do_p1) {
        forward.vx_t.copy_(p.u_last_two.select(0,0).select(0,0));
        forward.vy_t.copy_(p.u_last_two.select(0,1).select(0,0));
        forward.vz_t.copy_(p.u_last_two.select(0,2).select(0,0));
        forward.sxx_t.copy_(p.u_last_two.select(0,3).select(0,0));
        forward.syy_t.copy_(p.u_last_two.select(0,4).select(0,0));
        forward.szz_t.copy_(p.u_last_two.select(0,5).select(0,0));
        forward.sxy_t.copy_(p.u_last_two.select(0,6).select(0,0));
        forward.sxz_t.copy_(p.u_last_two.select(0,7).select(0,0));
        forward.syz_t.copy_(p.u_last_two.select(0,8).select(0,0));
    }

    auto neg_forward_source = -p.forward_source;

    // Generate pointer views
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    torch::Tensor grad_vp, grad_vs, grad_rho;
    bind_elastic_grads_3d(p, grad_vp, grad_vs, grad_rho);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    // PML coefficients
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
        p.boundary_gpu,
        {}, p.use_pinned_memory
    );
    auto bs = boundary_saver.view();

    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

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

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = it_hi - 1; it >= std::max(it_lo, 1); --it) {

        if (do_p1) {

        undo_body_force_source_injection_3d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

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

        // Same operands the imaging just correlated: for_view.v* is v(it),
        // fv*_prev is v(it+1) (overwritten a few lines below).
        {
            const float* fv_now[3]  = {for_view.vx, for_view.vy, for_view.vz};
            const float* fv_next[3] = {fvx_prev.data_ptr<float>(),
                                       fvy_prev.data_ptr<float>(),
                                       fvz_prev.data_ptr<float>()};
            undo_receiver_rho_injection_3d(
                adj_source_config, grad_rho, fv_now, fv_next,
                rho, p, receiver_fields, it, adjoint_nsrc, solver
            );
        }

        apply_stress_adjoint_3d(
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

        apply_velocity_adjoint_3d(
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

        }  // do_p2
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    check_stepped_backward_elastic_3d(p, /*need_recon=*/false);
    // Segment bounds: process [it_lo, it_hi) in descending order.  Defaults
    // (bw_it_begin = -1 => nt, bw_it_end = 0) reproduce the monolithic call.
    // it == 0 keeps its legacy asymmetry (gradient only, no adjoint step)
    // and runs in whichever segment contains it.
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    const bool first_segment = (it_hi == static_cast<int>(p.nt));
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
    // DD: cut-aware PML predicates in the adjoint prepare kernels.
    solver.cut_mask = p.cut_face_mask;

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 3);
    // FIRST segment only: a continuation call must keep the carried
    // adjoint state (legacy monolithic calls always have bw_begin == nt).
    if (first_segment)
        zero_wavefield_state(adjoint);

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);
    auto adj_view = adjoint.view();

    torch::Tensor grad_vp, grad_vs, grad_rho;
    bind_elastic_grads_3d(p, grad_vp, grad_vs, grad_rho);
    auto zero_velocity = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = it_hi - 1; it >= it_lo; --it) {
        undo_body_force_source_injection_3d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

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

        const float* vx_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vy_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>() : zero_velocity.data_ptr<float>();
        const float* vz_prev = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 2).data_ptr<float>() : zero_velocity.data_ptr<float>();

        // Reverse step 0 has no adjoint apply kernel (the loop `continue`s
        // below), so its gradient imaging cannot be folded — emit it as a
        // standalone calculate_grad pass (mirrors the acoustic trailing call).
        if (it == 0) {
            auto current_forward = make_velocity_view_3d(
                p.u_forward.select(0, it).select(0, 0),
                p.u_forward.select(0, it).select(0, 1),
                p.u_forward.select(0, it).select(0, 2)
            );
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
            {
                const float* fv_now[3]  = {vx_now, vy_now, vz_now};
                const float* fv_next[3] = {vx_prev, vy_prev, vz_prev};
                undo_receiver_rho_injection_3d(
                    source_config, grad_rho, fv_now, fv_next,
                    rho, p, receiver_fields, it, adjoint_nsrc, solver
                );
            }
            continue;
        }

        // FULL-mode fusion: fold this reverse step's vp/vs/rho-gradient imaging
        // into the stress-adjoint-prepare kernel (it reads the un-mutated
        // post-source adjoint[it] stress+velocity at entry, exactly what
        // calculate_grad_elastic3d_bs(it) would correlate), eliminating the
        // separate full-grid calculate_grad launch for steps it >= 1.
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
            workspace,
            vx_now, vy_now, vz_now,
            vx_prev, vy_prev, vz_prev,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>()
        );

        {
            const float* fv_now[3]  = {vx_now, vy_now, vz_now};
            const float* fv_next[3] = {vx_prev, vy_prev, vz_prev};
            undo_receiver_rho_injection_3d(
                source_config, grad_rho, fv_now, fv_next,
                rho, p, receiver_fields, it, adjoint_nsrc, solver
            );
        }
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
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "checkpoint backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

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
    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        undo_body_force_source_injection_3d(fwd_source_config, grad_rho, adj_view,
                                            rho, p, source_fields, it, solver);

        for (int irec = 0; irec < static_cast<int>(p.receiver_field_indices.numel()); ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
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

        {
            const float* fv_now[3]  = {current_vx.data_ptr<float>(),
                                       current_vy.data_ptr<float>(),
                                       current_vz.data_ptr<float>()};
            const float* fv_next[3] = {next_vx.data_ptr<float>(),
                                       next_vy.data_ptr<float>(),
                                       next_vz.data_ptr<float>()};
            undo_receiver_rho_injection_3d(
                adj_source_config, grad_rho, fv_now, fv_next,
                rho, p, receiver_fields, it,
                (int)p.adjoint_sources_loc.size(1), solver
            );
        }

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
// APM (Cao & Chen 2018, 3-D) backward — full + boundary-saving.
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
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "APM backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

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

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = p.nt - 1; it >= 0; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field, adj_source_signed[irec].data_ptr<float>(),
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

        {
            const float* fv_now[3]  = {p.u_forward.select(0, it).select(0, 0).data_ptr<float>(),
                                       p.u_forward.select(0, it).select(0, 1).data_ptr<float>(),
                                       p.u_forward.select(0, it).select(0, 2).data_ptr<float>()};
            const float* fv_next[3] = {vx_prev, vy_prev, vz_prev};
            undo_receiver_rho_injection_3d(
                source_config, grad_rho, fv_now, fv_next,
                rho, p, receiver_fields, it, adjoint_nsrc, solver
            );
        }

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
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "APM backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

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

    const auto adj_source_signed =
        elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);

    for (int it = p.nt - 1; it >= 1; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<adj_source_config.grid, adj_source_config.block>>>(
                field, adj_source_signed[irec].data_ptr<float>(),
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

        {
            const float* fv_now[3]  = {for_view.vx, for_view.vy, for_view.vz};
            const float* fv_next[3] = {fvx_prev.data_ptr<float>(),
                                       fvy_prev.data_ptr<float>(),
                                       fvz_prev.data_ptr<float>()};
            undo_receiver_rho_injection_3d(
                adj_source_config, grad_rho, fv_now, fv_next,
                rho, p, receiver_fields, it, adjoint_nsrc, solver
            );
        }

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
