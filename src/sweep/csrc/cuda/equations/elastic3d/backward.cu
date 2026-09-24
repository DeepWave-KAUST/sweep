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
#include "driver_traits.cuh"

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

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward<Driver>(in);
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_bs<Driver>(in);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_ckpt<Driver>(in);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    return eqdrv::sg_generic_backward_recursive_ckpt<Driver>(in);
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
    elastic_init_aux_slabs(solver, adjoint);
    zero_wavefield_state(adjoint);
    auto adj_view = adjoint.view();

    // grads_out = {grad_vp, grad_vs, grad_rho, <one shared zero under every
    // derived APM model>} from the propagator, or fresh tensors when unbound.
    const auto& gs = p.grads_out;
    TORCH_CHECK(gs.empty() || gs.size() == p.models.size(),
                "elastic3d APM backward: grads_out must be empty or hold one tensor per model (",
                p.models.size(), "), got ", gs.size());
    auto grad_vp  = pool_or_zeros(gs, 0, vp, "grads_out");
    auto grad_vs  = pool_or_zeros(gs, 1, vp, "grads_out");
    auto grad_rho = pool_or_zeros(gs, 2, vp, "grads_out");
    auto zero_velocity = torch::zeros_like(vp);
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);
    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    const auto adj_source_signs =
        elastic_adjoint_source_signs(p.adjoint_source, receiver_fields, 3);

    for (int it = p.nt - 1; it >= 0; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d_signed<<<source_config.grid, source_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(), it, adjoint_nsrc,
                adj_source_signs[irec], solver
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

    // The derived models' placeholder: the propagator's shared zero when bound.
    auto z = gs.size() > 3 ? gs[3] : torch::zeros_like(vp);
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
    elastic_init_aux_slabs(solver, adjoint);

    ElasticWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields, false);
    else
        forward.allocate(vp, 3, false);
    copy_tensor_cuda_async(forward.vx_t, p.u_last_two.select(0,0).select(0,0));
    copy_tensor_cuda_async(forward.vy_t, p.u_last_two.select(0,1).select(0,0));
    copy_tensor_cuda_async(forward.vz_t, p.u_last_two.select(0,2).select(0,0));
    copy_tensor_cuda_async(forward.sxx_t, p.u_last_two.select(0,3).select(0,0));
    copy_tensor_cuda_async(forward.syy_t, p.u_last_two.select(0,4).select(0,0));
    copy_tensor_cuda_async(forward.szz_t, p.u_last_two.select(0,5).select(0,0));
    copy_tensor_cuda_async(forward.sxy_t, p.u_last_two.select(0,6).select(0,0));
    copy_tensor_cuda_async(forward.sxz_t, p.u_last_two.select(0,7).select(0,0));
    copy_tensor_cuda_async(forward.syz_t, p.u_last_two.select(0,8).select(0,0));

    auto for_view = forward.view();
    auto adj_view = adjoint.view();

    // grads_out = {grad_vp, grad_vs, grad_rho, <one shared zero under every
    // derived APM model>} from the propagator, or fresh tensors when unbound.
    const auto& gs = p.grads_out;
    TORCH_CHECK(gs.empty() || gs.size() == p.models.size(),
                "elastic3d APM backward: grads_out must be empty or hold one tensor per model (",
                p.models.size(), "), got ", gs.size());
    auto grad_vp  = pool_or_zeros(gs, 0, vp, "grads_out");
    auto grad_vs  = pool_or_zeros(gs, 1, vp, "grads_out");
    auto grad_rho = pool_or_zeros(gs, 2, vp, "grads_out");
    ElasticAdjointWorkspaceTensor workspace;
    init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    // last_two is bound but never read here: this backward seeds its
    // reconstruction from p.u_last_two directly, and an unbound saver would
    // allocate an nvar-wavefield copy on every call (in host memory on the
    // staged path).
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    boundary_saver.allocate(
        true, 3, 9, solver, vp, save_width, 1,
        true, !staged_boundary, staged_boundary ? p.transfer_interval : 1,
        staged_boundary ? p.boundary_cpu : std::vector<torch::Tensor>{},
        p.boundary_gpu, p.u_last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
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

    const auto adj_source_signs =
        elastic_adjoint_source_signs(p.adjoint_source, receiver_fields, 3);

    for (int it = p.nt - 1; it >= 1; --it) {
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d_signed<<<adj_source_config.grid, adj_source_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(), it, adjoint_nsrc,
                adj_source_signs[irec], solver
            );
        }

        // Reverse forward replay: -source then reverse stress step.
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(for_view, 3, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d_signed<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field, p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(), it, forward_nsrc, -1.0f, solver
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

        copy_tensor_cuda_async(fvz_prev, forward.vz_t);
        copy_tensor_cuda_async(fvy_prev, forward.vy_t);
        copy_tensor_cuda_async(fvx_prev, forward.vx_t);

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

    // The derived models' placeholder: the propagator's shared zero when bound.
    auto z = gs.size() > 3 ? gs[3] : torch::zeros_like(vp);
    out.grads = {grad_vp, grad_vs, grad_rho,
                 z, z, z,
                 z, z, z, z, z, z, z, z, z,
                 z, z, z,
                 z, z, z};
    return out;
}

BackwardRunnerPtr backward_bs_runner(const BackwardInput& in)
{
    return std::make_shared<eqdrv::SgBackwardBsRunner<Driver>>(in);
}

}
