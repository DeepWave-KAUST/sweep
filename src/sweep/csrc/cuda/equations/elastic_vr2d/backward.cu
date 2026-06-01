#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "kernels.cuh"
#include "elastic_vr2d.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
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

BackwardOutput backward_bs(const BackwardInput& in)
{
    TORCH_CHECK(false,
                "elastic_vr2d::backward_bs is not yet implemented (Phase 4). "
                "Use full-mode backward (impl='c', use_ckpt=False) for now.");
    return BackwardOutput{};
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    TORCH_CHECK(false,
                "elastic_vr2d::backward_ckpt is not yet implemented (Phase 4). "
                "Use full-mode backward (impl='c', use_ckpt=False) for now.");
    return BackwardOutput{};
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    TORCH_CHECK(false,
                "elastic_vr2d::backward_recursive_ckpt is not yet implemented (Phase 4). "
                "Use full-mode backward (impl='c', use_ckpt=False) for now.");
    return BackwardOutput{};
}

}  // namespace elastic_vr2d
