#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernels.cuh"
#include "elastic_vr2d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace elastic_vr2d {

using namespace elastic_vr2d_kernels;

namespace {

void apply_evr_adjoint_step(
    int order,
    const fdtd::LaunchConfig& launch_config,
    ElasticWavefieldTensor& adjoint,
    const torch::Tensor& vp,
    const torch::Tensor& vs,
    const torch::Tensor& Rp_x,
    const torch::Tensor& Rp_z,
    const torch::Tensor& Rs_x,
    const torch::Tensor& Rs_z,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    torch::Tensor& ws_qxx, torch::Tensor& ws_qzz,
    torch::Tensor& ws_qxz, torch::Tensor& ws_qzx,
    torch::Tensor& ws_pxx, torch::Tensor& ws_pzz,
    torch::Tensor& ws_pxz, torch::Tensor& ws_pzx,
    torch::Tensor& ws_pt_px, torch::Tensor& ws_pt_pz
)
{
    auto adj_view = adjoint.view();

    // ---- Adjoint of forward stress step (2 kernels: prepare + apply) ----
    LAUNCH_EVR_STRESS_ADJOINT_PREPARE(
        order, launch_config.grid, launch_config.block,
        adj_view,
        vp.data_ptr<float>(),
        vs.data_ptr<float>(),
        Rp_x.data_ptr<float>(),
        Rp_z.data_ptr<float>(),
        Rs_x.data_ptr<float>(),
        Rs_z.data_ptr<float>(),
        grad_ctx,
        cpml_view,
        solver,
        ws_qxx.data_ptr<float>(),
        ws_qzz.data_ptr<float>(),
        ws_qxz.data_ptr<float>(),
        ws_qzx.data_ptr<float>(),
        ws_pt_px.data_ptr<float>(),
        ws_pt_pz.data_ptr<float>()
    );

    LAUNCH_EVR_STRESS_ADJOINT_APPLY(
        order, launch_config.grid, launch_config.block,
        adj_view,
        ws_qxx.data_ptr<float>(),
        ws_qzz.data_ptr<float>(),
        ws_qxz.data_ptr<float>(),
        ws_qzx.data_ptr<float>(),
        ws_pt_px.data_ptr<float>(),
        ws_pt_pz.data_ptr<float>(),
        grad_ctx, solver
    );

    // ---- Adjoint of forward momentum step (2 kernels: prepare + apply) ----
    LAUNCH_EVR_MOMENTUM_ADJOINT_PREPARE(
        order, launch_config.grid, launch_config.block,
        adj_view, cpml_view, solver,
        ws_pxx.data_ptr<float>(),
        ws_pzz.data_ptr<float>(),
        ws_pxz.data_ptr<float>(),
        ws_pzx.data_ptr<float>()
    );

    LAUNCH_EVR_MOMENTUM_ADJOINT_APPLY(
        order, launch_config.grid, launch_config.block,
        adj_view,
        ws_pxx.data_ptr<float>(),
        ws_pzz.data_ptr<float>(),
        ws_pxz.data_ptr<float>(),
        ws_pzx.data_ptr<float>(),
        grad_ctx, solver
    );
}

torch::Tensor maybe_workspace_or_zeros(
    const std::vector<torch::Tensor>& pool, int idx,
    const torch::Tensor& like
)
{
    if (static_cast<int>(pool.size()) > idx && pool[idx].defined() && pool[idx].numel() > 0)
        return pool[idx];
    return torch::zeros_like(like);
}

int find_previous_checkpoint_idx(const int* steps, int n, int target)
{
    int idx = -1;
    for (int i = 0; i < n; ++i) {
        if (steps[i] < target) idx = i;
        else break;
    }
    return idx;
}

// Replay the forward from the nearest earlier checkpoint up to `target_index`,
// leaving the momentum p^{target_index} in current_px/current_pz. Used by the
// recursive-checkpoint backward (mirrors elastic2d::replay_forward_to_time_2d;
// EVR needs only the current momentum, no next-step velocity).
void replay_forward_to_time_evr(
    const BackwardInput& p,
    ElasticWavefieldTensor& forward,
    torch::Tensor& current_px, torch::Tensor& current_pz,
    int target_index, const int* checkpoint_steps, int num_saved_checkpoints,
    CheckpointRuntime& checkpoint_runtime, int order,
    const fdtd::LaunchConfig& launch_config,
    const fdtd::LaunchConfig& fwd_source_config,
    SGradParam grad_ctx, ElasticCPMLPointer cpml_view, SolverContext solver,
    const torch::Tensor& source_fields,
    const torch::Tensor& vp, const torch::Tensor& vs,
    const torch::Tensor& Rp_x, const torch::Tensor& Rp_z,
    const torch::Tensor& Rs_x, const torch::Tensor& Rs_z
)
{
    const int checkpoint_idx =
        find_previous_checkpoint_idx(checkpoint_steps, num_saved_checkpoints, target_index + 1);
    int start_time = 0;
    if (checkpoint_idx >= 0) {
        checkpoint_runtime.load(checkpoint_idx, forward.checkpoint_tensors());
        start_time = checkpoint_steps[checkpoint_idx];
    } else {
        checkpoint_runtime.zero_state(forward.state_tensors());
    }

    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();

    for (int it = start_time; it < (int)p.nt; ++it) {
        auto for_view = forward.view();
        LAUNCH_EVR_MOMENTUM(order, launch_config.grid, launch_config.block,
                            for_view, grad_ctx, cpml_view, solver);
        LAUNCH_EVR_STRESS(order, launch_config.grid, launch_config.block,
                          for_view, vp.data_ptr<float>(), vs.data_ptr<float>(),
                          Rp_x.data_ptr<float>(), Rp_z.data_ptr<float>(),
                          Rs_x.data_ptr<float>(), Rs_z.data_ptr<float>(),
                          (float*)nullptr, grad_ctx, cpml_view, solver);
        if (it == target_index) {
            current_px.copy_(forward.vx_t);
            current_pz.copy_(forward.vz_t);
            break;
        }
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field, p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(), it, forward_nsrc, solver);
        }
    }
}

// One backward chunk for the checkpointing path: replay the forward over
// [start, end) (storing per-step momentum), then reverse-sweep the adjoint +
// gradient over the same window. Mirrors elastic2d::backward_segment_2d with
// EVR kernels; no rho term so no next-segment velocity bookkeeping.
void backward_segment_evr(
    const BackwardInput& p,
    const torch::Tensor& vp, const torch::Tensor& vs,
    const torch::Tensor& Rp_x, const torch::Tensor& Rp_z,
    const torch::Tensor& Rs_x, const torch::Tensor& Rs_z,
    ElasticWavefieldTensor& start_state,
    ElasticWavefieldTensor& adjoint,
    CheckpointRuntime& checkpoint_runtime,
    int start, int end, int order,
    const fdtd::LaunchConfig& launch_config,
    const fdtd::LaunchConfig& fwd_source_config,
    const fdtd::LaunchConfig& adj_source_config,
    SGradParam grad_ctx, ElasticCPMLPointer cpml_view, SolverContext solver,
    const torch::Tensor& source_fields, const torch::Tensor& receiver_fields,
    torch::Tensor& grad_vp, torch::Tensor& grad_vs,
    torch::Tensor& grad_Rp_x, torch::Tensor& grad_Rp_z,
    torch::Tensor& grad_Rs_x, torch::Tensor& grad_Rs_z,
    torch::Tensor ws[14], const torch::Tensor& zero_buf
)
{
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    const int nrec_fields = p.receiver_field_indices.numel();
    const int segment_len = end - start;

    auto seg_px = torch::zeros({segment_len + 1, vp.size(0) * vp.size(1), 1,
                                vp.size(2), vp.size(3)}, vp.options());
    auto seg_pz = torch::zeros_like(seg_px);

    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty()) forward.bind(p.forward_wavefields, true);
    else forward.allocate(vp, 2, true);
    checkpoint_runtime.copy_state(forward.state_tensors(), start_state.state_tensors());

    seg_px.select(0, 0).copy_(forward.vx_t);
    seg_pz.select(0, 0).copy_(forward.vz_t);

    // ---- forward replay over the chunk, storing post-step momentum ----
    for (int it = start; it < end; ++it) {
        auto for_view = forward.view();
        LAUNCH_EVR_MOMENTUM(order, launch_config.grid, launch_config.block,
                            for_view, grad_ctx, cpml_view, solver);
        LAUNCH_EVR_STRESS(order, launch_config.grid, launch_config.block,
                          for_view, vp.data_ptr<float>(), vs.data_ptr<float>(),
                          Rp_x.data_ptr<float>(), Rp_z.data_ptr<float>(),
                          Rs_x.data_ptr<float>(), Rs_z.data_ptr<float>(),
                          (float*)nullptr, grad_ctx, cpml_view, solver);
        seg_px.select(0, it - start + 1).copy_(forward.vx_t);
        seg_pz.select(0, it - start + 1).copy_(forward.vz_t);
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field, p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(), it, forward_nsrc, solver);
        }
    }

    // ---- reverse adjoint + gradient sweep ----
    for (int it = end - 1; it >= start; --it) {
        auto adj_view = adjoint.view();
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(), it, adjoint_nsrc, solver);
        }

        const int now_offset = (it - start) + 1;   // seg index of p^{it}
        LAUNCH_CALCULATE_GRAD_EVR_NOBS(
            order, launch_config.grid, launch_config.block,
            adj_view, seg_px.select(0, now_offset).data_ptr<float>(),
            seg_pz.select(0, now_offset).data_ptr<float>(),
            zero_buf.data_ptr<float>(), zero_buf.data_ptr<float>(),
            vp.data_ptr<float>(), vs.data_ptr<float>(),
            Rp_x.data_ptr<float>(), Rp_z.data_ptr<float>(),
            Rs_x.data_ptr<float>(), Rs_z.data_ptr<float>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(),
            grad_Rp_x.data_ptr<float>(), grad_Rp_z.data_ptr<float>(),
            grad_Rs_x.data_ptr<float>(), grad_Rs_z.data_ptr<float>(),
            ws[10].data_ptr<float>(), ws[11].data_ptr<float>(),
            ws[12].data_ptr<float>(), ws[13].data_ptr<float>(),
            grad_ctx, solver);
        LAUNCH_EVR_GRAD_CHAIN_APPLY(
            order, launch_config.grid, launch_config.block,
            ws[10].data_ptr<float>(), ws[11].data_ptr<float>(),
            ws[12].data_ptr<float>(), ws[13].data_ptr<float>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(),
            grad_ctx, solver);

        if (it == 0) continue;

        apply_evr_adjoint_step(
            order, launch_config, adjoint, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z,
            cpml_view, grad_ctx, solver,
            ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7], ws[8], ws[9]);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Full backward -- uses save_all_wavefields. Per timestep (reverse loop
// t = nt-1 down to 0):
//   1. Inject adjoint sources at receivers into the appropriate adjoint
//      wavefield component.
//   2. Compute model gradients via gradient kernel (writes pointwise terms
//      + chain-rule source tensors).
//   3. Apply chain-rule pass (transpose central FD on L_dV* into grad_vp /
//      grad_vs).
//   4. Apply the 4-kernel adjoint step (stress_adjoint then momentum_adjoint).
//      Skip on it == 0 since there are no more timesteps to feed.
// ---------------------------------------------------------------------------

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    TORCH_CHECK(p.models.size() >= 6,
                "elastic_vr2d::backward expects 6 model tensors "
                "(vp, vs, Rp_x, Rp_z, Rs_x, Rs_z)");

    auto vp   = p.models[0];
    auto vs   = p.models[1];
    auto Rp_x = p.models[2];
    auto Rp_z = p.models[3];
    auto Rs_x = p.models[4];
    auto Rs_z = p.models[5];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn,
                         p.free_surface,
                         p.lap_coes.data_ptr<float>(),
                         p.grad_coes.data_ptr<float>(),
                         dx, 0.f, dz};
    if (p.has_topo) {
        solver.topo_rows = p.topo_rows.data_ptr<int>();
        solver.has_topo = true;
    }

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields, true);
    else
        adjoint.allocate(vp, 2);
    auto adj_view = adjoint.view();

    auto grad_vp   = torch::zeros_like(vp);
    auto grad_vs   = torch::zeros_like(vp);
    auto grad_Rp_x = torch::zeros_like(vp);
    auto grad_Rp_z = torch::zeros_like(vp);
    auto grad_Rs_x = torch::zeros_like(vp);
    auto grad_Rs_z = torch::zeros_like(vp);

    // 14 workspace tensors. Pull from pool when pre-allocated by propagator,
    // else allocate fresh.
    auto ws_qxx   = maybe_workspace_or_zeros(p.adjoint_workspace, 0,  vp);
    auto ws_qzz   = maybe_workspace_or_zeros(p.adjoint_workspace, 1,  vp);
    auto ws_qxz   = maybe_workspace_or_zeros(p.adjoint_workspace, 2,  vp);
    auto ws_qzx   = maybe_workspace_or_zeros(p.adjoint_workspace, 3,  vp);
    auto ws_pxx   = maybe_workspace_or_zeros(p.adjoint_workspace, 4,  vp);
    auto ws_pzz   = maybe_workspace_or_zeros(p.adjoint_workspace, 5,  vp);
    auto ws_pxz   = maybe_workspace_or_zeros(p.adjoint_workspace, 6,  vp);
    auto ws_pzx   = maybe_workspace_or_zeros(p.adjoint_workspace, 7,  vp);
    auto ws_pt_px = maybe_workspace_or_zeros(p.adjoint_workspace, 8,  vp);
    auto ws_pt_pz = maybe_workspace_or_zeros(p.adjoint_workspace, 9,  vp);
    auto ws_lvpx  = maybe_workspace_or_zeros(p.adjoint_workspace, 10, vp);
    auto ws_lvpz  = maybe_workspace_or_zeros(p.adjoint_workspace, 11, vp);
    auto ws_lvsx  = maybe_workspace_or_zeros(p.adjoint_workspace, 12, vp);
    auto ws_lvsz  = maybe_workspace_or_zeros(p.adjoint_workspace, 13, vp);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto zero_buf = torch::zeros_like(vp);

    for (int it = p.nt - 1; it >= 0; --it) {
        // 1. Inject adjoint sources
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<source_config.grid, source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it, adjoint_nsrc, solver
            );
        }

        // 2. Saved forward momentum at time t (px in channel 0, pz in channel 1)
        const float* fpx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* fpz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* fpx_next = (it + 1 < (int)p.nt)
            ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>()
            : zero_buf.data_ptr<float>();
        const float* fpz_next = (it + 1 < (int)p.nt)
            ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>()
            : zero_buf.data_ptr<float>();

        // 3. Gradient kernel (pointwise terms + chain-rule sources)
        LAUNCH_CALCULATE_GRAD_EVR_NOBS(
            order, launch_config.grid, launch_config.block,
            adj_view,
            fpx_now, fpz_now,
            fpx_next, fpz_next,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            Rp_x.data_ptr<float>(),
            Rp_z.data_ptr<float>(),
            Rs_x.data_ptr<float>(),
            Rs_z.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_Rp_x.data_ptr<float>(),
            grad_Rp_z.data_ptr<float>(),
            grad_Rs_x.data_ptr<float>(),
            grad_Rs_z.data_ptr<float>(),
            ws_lvpx.data_ptr<float>(),
            ws_lvpz.data_ptr<float>(),
            ws_lvsx.data_ptr<float>(),
            ws_lvsz.data_ptr<float>(),
            grad_ctx, solver
        );

        // 4. Chain-rule pass (transpose central FD into grad_vp / grad_vs)
        LAUNCH_EVR_GRAD_CHAIN_APPLY(
            order, launch_config.grid, launch_config.block,
            ws_lvpx.data_ptr<float>(),
            ws_lvpz.data_ptr<float>(),
            ws_lvsx.data_ptr<float>(),
            ws_lvsz.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_ctx, solver
        );

        if (it == 0) continue;

        // 5. Apply EVR adjoint step (4 kernels total: stress + momentum)
        apply_evr_adjoint_step(
            order, launch_config, adjoint,
            vp, vs, Rp_x, Rp_z, Rs_x, Rs_z,
            cpml_view, grad_ctx, solver,
            ws_qxx, ws_qzz, ws_qxz, ws_qzx,
            ws_pxx, ws_pzz, ws_pxz, ws_pzx,
            ws_pt_px, ws_pt_pz
        );
    }

    out.grads = {grad_vp, grad_vs, grad_Rp_x, grad_Rp_z, grad_Rs_x, grad_Rs_z};
    return out;
}

// ---------------------------------------------------------------------------
// Phase 4 stubs -- boundary saving + checkpointing variants.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Boundary-saving backward. Reconstructs the forward wavefield in reverse time
// from the final state (u_last_two) + saved boundary strips, reusing the
// REVERSE (-=) NOPML kernels. The per-step adjoint + gradient kernels are the
// same verified ones used by full-mode backward. Mirror elastic2d::backward_bs
// with EVR kernel substitutions (no rho; momentum-first forward => reconstruct
// stress then momentum).
// ---------------------------------------------------------------------------
BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    TORCH_CHECK(p.models.size() >= 6,
                "elastic_vr2d::backward_bs expects 6 model tensors");

    auto vp   = p.models[0];
    auto vs   = p.models[1];
    auto Rp_x = p.models[2];
    auto Rp_z = p.models[3];
    auto Rs_x = p.models[4];
    auto Rs_z = p.models[5];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn,
                         p.free_surface,
                         p.lap_coes.data_ptr<float>(),
                         p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    if (p.has_topo) { solver.topo_rows = p.topo_rows.data_ptr<int>(); solver.has_topo = true; }
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty()) adjoint.bind(p.adjoint_wavefields, true);
    else adjoint.allocate(vp, 2);

    // Reconstructed forward, seeded from the final saved full state.
    ElasticWavefieldTensor forward;
    forward.allocate(vp, 2, false);
    forward.vx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));   // px
    forward.vz_t.copy_(p.u_last_two.select(0, 1).select(0, 0));   // pz
    forward.sxx_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
    forward.szz_t.copy_(p.u_last_two.select(0, 3).select(0, 0));
    forward.sxz_t.copy_(p.u_last_two.select(0, 4).select(0, 0));

    auto neg_forward_source = -p.forward_source;
    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    auto grad_vp   = torch::zeros_like(vp);
    auto grad_vs   = torch::zeros_like(vp);
    auto grad_Rp_x = torch::zeros_like(vp);
    auto grad_Rp_z = torch::zeros_like(vp);
    auto grad_Rs_x = torch::zeros_like(vp);
    auto grad_Rs_z = torch::zeros_like(vp);

    auto ws_qxx   = maybe_workspace_or_zeros(p.adjoint_workspace, 0,  vp);
    auto ws_qzz   = maybe_workspace_or_zeros(p.adjoint_workspace, 1,  vp);
    auto ws_qxz   = maybe_workspace_or_zeros(p.adjoint_workspace, 2,  vp);
    auto ws_qzx   = maybe_workspace_or_zeros(p.adjoint_workspace, 3,  vp);
    auto ws_pxx   = maybe_workspace_or_zeros(p.adjoint_workspace, 4,  vp);
    auto ws_pzz   = maybe_workspace_or_zeros(p.adjoint_workspace, 5,  vp);
    auto ws_pxz   = maybe_workspace_or_zeros(p.adjoint_workspace, 6,  vp);
    auto ws_pzx   = maybe_workspace_or_zeros(p.adjoint_workspace, 7,  vp);
    auto ws_pt_px = maybe_workspace_or_zeros(p.adjoint_workspace, 8,  vp);
    auto ws_pt_pz = maybe_workspace_or_zeros(p.adjoint_workspace, 9,  vp);
    auto ws_lvpx  = maybe_workspace_or_zeros(p.adjoint_workspace, 10, vp);
    auto ws_lvpz  = maybe_workspace_or_zeros(p.adjoint_workspace, 11, vp);
    auto ws_lvsx  = maybe_workspace_or_zeros(p.adjoint_workspace, 12, vp);
    auto ws_lvsz  = maybe_workspace_or_zeros(p.adjoint_workspace, 13, vp);
    auto zero_buf = torch::zeros_like(vp);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, {},
                                p.use_pinned_memory);
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

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver, 2, true, p.boundary_on_cpu, p.boundary_on_disk,
        p.boundary_disk_async_read, p.transfer_interval, p.boundary_ring_buffers,
        p.boundary_disk_files, async_copy.compute_stream, async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    for (int it = p.nt - 1; it >= 1; --it) {
        // 1. Inject adjoint sources at receivers (into adjoint momentum).
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(), it, adjoint_nsrc, solver
            );
        }

        // 2. Un-inject the forward source (negated) from the reconstructed field.
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field, neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(), it, forward_nsrc, solver
            );
        }

        // 3. Reverse the stress update (sigma^{it} -> sigma^{it-1}) in the
        //    interior; for_view momentum still holds p^{it}.
        LAUNCH_EVR_STRESS_NOPML(
            order, launch_config.grid, launch_config.block,
            for_view, vp.data_ptr<float>(), vs.data_ptr<float>(),
            Rp_x.data_ptr<float>(), Rp_z.data_ptr<float>(),
            Rs_x.data_ptr<float>(), Rs_z.data_ptr<float>(),
            (float*)nullptr, grad_ctx, solver
        );
        // restore sigma boundary strips (fields 2,3,4 = sxx,szz,sxz)
        float* sfields[3] = {for_view.sxx, for_view.szz, for_view.sxz};
        for (int f = 2; f < 5; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, sfields[f - 2], launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver, f, f == 2, false
            );
        }

        // 4. Gradient (uses reconstructed p^{it} = for_view.vx/vz + adjoint).
        LAUNCH_CALCULATE_GRAD_EVR_NOBS(
            order, launch_config.grid, launch_config.block,
            adj_view, for_view.vx, for_view.vz,
            zero_buf.data_ptr<float>(), zero_buf.data_ptr<float>(),
            vp.data_ptr<float>(), vs.data_ptr<float>(),
            Rp_x.data_ptr<float>(), Rp_z.data_ptr<float>(),
            Rs_x.data_ptr<float>(), Rs_z.data_ptr<float>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(),
            grad_Rp_x.data_ptr<float>(), grad_Rp_z.data_ptr<float>(),
            grad_Rs_x.data_ptr<float>(), grad_Rs_z.data_ptr<float>(),
            ws_lvpx.data_ptr<float>(), ws_lvpz.data_ptr<float>(),
            ws_lvsx.data_ptr<float>(), ws_lvsz.data_ptr<float>(),
            grad_ctx, solver
        );
        LAUNCH_EVR_GRAD_CHAIN_APPLY(
            order, launch_config.grid, launch_config.block,
            ws_lvpx.data_ptr<float>(), ws_lvpz.data_ptr<float>(),
            ws_lvsx.data_ptr<float>(), ws_lvsz.data_ptr<float>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(),
            grad_ctx, solver
        );

        // 5. Adjoint wavefield step (stress-adjoint then momentum-adjoint).
        apply_evr_adjoint_step(
            order, launch_config, adjoint,
            vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, cpml_view, grad_ctx, solver,
            ws_qxx, ws_qzz, ws_qxz, ws_qzx, ws_pxx, ws_pzz, ws_pxz, ws_pzx,
            ws_pt_px, ws_pt_pz
        );

        // 6. Reverse the momentum update (p^{it} -> p^{it-1}) in the interior.
        LAUNCH_EVR_MOMENTUM_NOPML(
            order, launch_config.grid, launch_config.block,
            for_view, grad_ctx, solver
        );
        float* mfields[2] = {for_view.vx, for_view.vz};
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, mfields[f], launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver, f, false, f == 1
            );
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    out.grads = {grad_vp, grad_vs, grad_Rp_x, grad_Rp_z, grad_Rs_x, grad_Rs_z};
    return out;
}

// ---------------------------------------------------------------------------
// Chunked checkpointing backward. Forward saved a checkpoint every
// checkpoint_interval steps; here we walk chunks in reverse, replaying each
// chunk's forward from its checkpoint and running the adjoint+grad sweep.
// ---------------------------------------------------------------------------
BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;

    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 15,
                "elastic_vr2d checkpointing expects 15 checkpoint tensors");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints, 15, true, false, p.checkpoint_interval, p.checkpoint_steps,
        p.checkpoint_on_cpu, "backward_chunk", "elastic_vr2d");

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    auto vp = p.models[0], vs = p.models[1];
    auto Rp_x = p.models[2], Rp_z = p.models[3], Rs_x = p.models[4], Rs_z = p.models[5];

    int N = vp.size(0), C = vp.size(1), nz = vp.size(2), nx = vp.size(3);
    int B = N * C;
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                         p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    if (p.has_topo) { solver.topo_rows = p.topo_rows.data_ptr<int>(); solver.has_topo = true; }
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty()) adjoint.bind(p.adjoint_wavefields, true);
    else adjoint.allocate(vp, 2, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);

    auto grad_vp   = torch::zeros_like(vp);
    auto grad_vs   = torch::zeros_like(vp);
    auto grad_Rp_x = torch::zeros_like(vp);
    auto grad_Rp_z = torch::zeros_like(vp);
    auto grad_Rs_x = torch::zeros_like(vp);
    auto grad_Rs_z = torch::zeros_like(vp);
    auto zero_buf  = torch::zeros_like(vp);
    torch::Tensor ws[14];
    for (int i = 0; i < 14; ++i) ws[i] = maybe_workspace_or_zeros(p.adjoint_workspace, i, vp);

    ElasticWavefieldTensor start_state;
    start_state.allocate(vp, 2, true);

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);
        if (chunk_id == 0) checkpoint_runtime.zero_state(start_state.state_tensors());
        else               checkpoint_runtime.load(chunk_id, start_state.checkpoint_tensors());
        backward_segment_evr(
            p, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z, start_state, adjoint, checkpoint_runtime,
            start, end, order, launch_config, fwd_source_config, adj_source_config,
            grad_ctx, cpml_view, solver, source_fields, receiver_fields,
            grad_vp, grad_vs, grad_Rp_x, grad_Rp_z, grad_Rs_x, grad_Rs_z, ws, zero_buf);
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_Rp_x, grad_Rp_z, grad_Rs_x, grad_Rs_z};
    return out;
}

// ---------------------------------------------------------------------------
// Recursive checkpointing backward. Same chunk machinery but the chunk
// boundaries come from the (irregular) checkpoint_steps schedule rather than a
// fixed interval. Each chunk's start state is replayed from the nearest saved
// checkpoint via the segment helper (which copies start_state then replays).
// ---------------------------------------------------------------------------
BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;

    TORCH_CHECK(p.checkpoint_steps.defined() && p.checkpoint_steps.dim() == 1,
                "recursive checkpointing expects 1-D checkpoint_steps");
    TORCH_CHECK(p.checkpoints.size() == 15,
                "elastic_vr2d checkpointing expects 15 checkpoint tensors");

    auto steps_cpu = p.checkpoint_steps.to(torch::kCPU).contiguous();
    const int* steps = steps_cpu.data_ptr<int>();
    int num_ckpt = static_cast<int>(steps_cpu.numel());

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints, 15, true, true, p.checkpoint_interval, p.checkpoint_steps,
        p.checkpoint_on_cpu, "backward_recursive", "elastic_vr2d");

    float dx = p.spacing[0];
    float dz = p.spacing[1];
    auto vp = p.models[0], vs = p.models[1];
    auto Rp_x = p.models[2], Rp_z = p.models[3], Rs_x = p.models[4], Rs_z = p.models[5];

    int N = vp.size(0), C = vp.size(1), nz = vp.size(2), nx = vp.size(3);
    int B = N * C;
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                         p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    if (p.has_topo) { solver.topo_rows = p.topo_rows.data_ptr<int>(); solver.has_topo = true; }
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    ElasticWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty()) adjoint.bind(p.adjoint_wavefields, true);
    else adjoint.allocate(vp, 2, true);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int nrec_fields = p.receiver_field_indices.numel();
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);

    auto grad_vp   = torch::zeros_like(vp);
    auto grad_vs   = torch::zeros_like(vp);
    auto grad_Rp_x = torch::zeros_like(vp);
    auto grad_Rp_z = torch::zeros_like(vp);
    auto grad_Rs_x = torch::zeros_like(vp);
    auto grad_Rs_z = torch::zeros_like(vp);
    auto zero_buf  = torch::zeros_like(vp);
    torch::Tensor ws[14];
    for (int i = 0; i < 14; ++i) ws[i] = maybe_workspace_or_zeros(p.adjoint_workspace, i, vp);

    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty()) forward.bind(p.forward_wavefields, true);
    else forward.allocate(vp, 2, true);
    auto current_px = torch::zeros_like(vp);
    auto current_pz = torch::zeros_like(vp);

    // Per-step reverse sweep: reconstruct p^{it} by replaying the forward from
    // the nearest earlier checkpoint, then grad + adjoint step. Mirrors
    // elastic2d::backward_recursive_ckpt.
    for (int it = p.nt - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(), it, adjoint_nsrc, solver);
        }

        replay_forward_to_time_evr(
            p, forward, current_px, current_pz, it, steps, num_ckpt,
            checkpoint_runtime, order, launch_config, fwd_source_config,
            grad_ctx, cpml_view, solver, source_fields,
            vp, vs, Rp_x, Rp_z, Rs_x, Rs_z);

        LAUNCH_CALCULATE_GRAD_EVR_NOBS(
            order, launch_config.grid, launch_config.block,
            adj_view, current_px.data_ptr<float>(), current_pz.data_ptr<float>(),
            zero_buf.data_ptr<float>(), zero_buf.data_ptr<float>(),
            vp.data_ptr<float>(), vs.data_ptr<float>(),
            Rp_x.data_ptr<float>(), Rp_z.data_ptr<float>(),
            Rs_x.data_ptr<float>(), Rs_z.data_ptr<float>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(),
            grad_Rp_x.data_ptr<float>(), grad_Rp_z.data_ptr<float>(),
            grad_Rs_x.data_ptr<float>(), grad_Rs_z.data_ptr<float>(),
            ws[10].data_ptr<float>(), ws[11].data_ptr<float>(),
            ws[12].data_ptr<float>(), ws[13].data_ptr<float>(),
            grad_ctx, solver);
        LAUNCH_EVR_GRAD_CHAIN_APPLY(
            order, launch_config.grid, launch_config.block,
            ws[10].data_ptr<float>(), ws[11].data_ptr<float>(),
            ws[12].data_ptr<float>(), ws[13].data_ptr<float>(),
            grad_vp.data_ptr<float>(), grad_vs.data_ptr<float>(), grad_ctx, solver);

        if (it == 0) continue;
        apply_evr_adjoint_step(
            order, launch_config, adjoint, vp, vs, Rp_x, Rp_z, Rs_x, Rs_z,
            cpml_view, grad_ctx, solver,
            ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7], ws[8], ws[9]);
    }

    BackwardOutput out;
    out.grads = {grad_vp, grad_vs, grad_Rp_x, grad_Rp_z, grad_Rs_x, grad_Rs_z};
    return out;
}

}  // namespace elastic_vr2d
