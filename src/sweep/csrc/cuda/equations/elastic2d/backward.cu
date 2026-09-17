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
#include "driver_traits.cuh"

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

    long long spatial_size = (long long)solver.nx * solver.nz;
    int idx = iz * solver.nx + ix;
    long long u_idx = (long long)b * spatial_size + idx;
    long long src_idx = ((long long)b * nrec + s) * solver.nt + it;

    float drho_x_drho, drho_z_drho;
    apm_rho_jacobian(category[idx], &drho_x_drho, &drho_z_drho);
    const float jac = is_z_component ? drho_z_drho : drho_x_drho;

    atomicAdd(&grad_rho[u_idx],
              -adjoint_source[src_idx] * (fv_now[u_idx] - fv_next[u_idx])
                  / rho_eff[u_idx] * jac);
}

namespace elastic2d {

namespace {

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

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward<Driver>(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_ckpt<Driver>(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_recursive_ckpt<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_bs<Driver>(in);
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
    elastic_init_aux_slabs(solver, adjoint);
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
    elastic_init_aux_slabs(solver, adjoint);
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

    // last_two is bound but never read here: this backward seeds its
    // reconstruction from p.u_last_two directly, and an unbound saver would
    // allocate an nvar-wavefield copy on every call (in host memory on the
    // staged path).
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.u_last_two, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, 2, 5, solver, vp, save_width, 1, true, true,
                                1, {}, p.boundary_gpu, p.u_last_two, p.use_pinned_memory);
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

BackwardRunnerPtr backward_bs_runner(const BackwardInput& in)
{
    return std::make_shared<eqdrv::SgBackwardBsRunner<Driver>>(in);
}

}
