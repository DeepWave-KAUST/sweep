#include <torch/extension.h>
#include <cuda_runtime.h>


#include "elastic3d.h"
#include "kernels.cuh"

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

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::sg_generic_forward<Driver>(in);
}


// ===========================================================================
// APM 3-D forward — Cao & Chen 2018 (3-D).
// ===========================================================================
// Expects the 21-tensor model layout assembled by _c.py's APM 3-D dispatch:
//   models = [vp, vs, rho, lam, mu, lam_2mu,
//             alpha_xx, alpha_yy, alpha_zz,
//             lam_xx_yy, lam_xx_zz, lam_yy_xx, lam_yy_zz,
//             lam_zz_xx, lam_zz_yy,
//             mu_xy, mu_xz, mu_yz,
//             inv_rho_x, inv_rho_y, inv_rho_z]
// `vp,vs,rho,lam,mu,lam_2mu` are carried for the backward pass; only
// indices 6..20 (the 15 effective arrays) are consumed by the APM
// kernels here.  ``solver.free_surface=false`` is forced and
// ``solver.topo_category`` + ``solver.use_apm=true`` are plumbed for
// the kernel-internal AIR / traction-BC branches.
ForwardOutput apm_forward(const ForwardInput& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    ForwardOutput out;

    SWEEP_CHECK(p.it_begin == 0 &&
                (p.it_end < 0 || p.it_end == static_cast<int>(p.nt)) &&
                p.step_phase == 0,
                "stepped forward not supported for the elastic3d APM path");
    SWEEP_CHECK(p.models.size() >= 21,
        "elastic3d::apm_forward expects 21-tensor models list "
        "[vp,vs,rho,lam,mu,lam_2mu,alpha_xx,alpha_yy,alpha_zz,"
        "lam_xx_yy,lam_xx_zz,lam_yy_xx,lam_yy_zz,lam_zz_xx,lam_zz_yy,"
        "mu_xy,mu_xz,mu_yz,inv_rho_x,inv_rho_y,inv_rho_z]; got ",
        p.models.size());
    SWEEP_CHECK(p.use_apm && p.topo_category.defined() && p.topo_category.numel() > 0,
        "apm_forward requires use_apm=true and topo_category tensor");

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    auto vp     = p.models[0];
    // Effective moduli (indices 6..20).
    auto alpha_xx = p.models[6];
    auto alpha_yy = p.models[7];
    auto alpha_zz = p.models[8];
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

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B  = N * C;

    ElasticWavefieldTensor wavefield;
    SWEEP_CHECK(!p.wavefields.empty(),
                "elastic3d/apm_forward requires the propagator-bound wavefields "
                "(36-slot layout); nothing allocates them here");
    wavefield.bind(p.wavefields, true);
    SWEEP_CHECK(wavefield.m_syzx_t.defined(),
                "elastic3d/apm_forward: the bound wavefield list must carry m_syzx");
    auto wf = wavefield.view();

    ElasticCPMLTensor cpml;
    cpml.bind(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto record = bound_required(p.record_out, {nrec_fields, B, nrec, p.nt}, vp.options(), "record_out");

    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = bound_required(p.u_allt_out, {p.nt, 3, B, nz, ny, nx}, vp.options(), "u_allt_out");

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    // APM forces ``free_surface=false`` (no image mirror) and exposes
    // topo_category via ctx so the kernel-internal AIR / traction-BC
    // branches fire.
    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn,
                         /*free_surface=*/false,
                         p.lap_coes.data_ptr<float>(),
                         p.grad_coes.data_ptr<float>(),
                         dx, dy, dz};
    solver.topo_category = p.topo_category.data_ptr<int>();
    solver.use_apm = true;
    elastic_init_aux_slabs(solver, wavefield);

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    boundary_saver.allocate(
        p.use_boundary_saving, 3, 9, solver, vp, save_width, 1,
        true, !staged_boundary, staged_boundary ? p.transfer_interval : 1,
        staged_boundary ? p.boundary_cpu : std::vector<torch::Tensor>{},
        p.boundary_gpu,
        p.last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
    );
    auto bs = boundary_saver.view();

    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;
    float* u_this_t = nullptr;

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    BoundaryRuntime boundary_runtime(
        boundary_saver, 3, p.use_boundary_saving,
        p.boundary_on_cpu, p.boundary_on_disk, p.boundary_disk_async_read,
        p.transfer_interval, p.boundary_ring_buffers, p.boundary_disk_files,
        async_copy.compute_stream, async_copy.copy_stream
    );
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints, 36, p.use_checkpoint, p.use_recursive_checkpoint,
        p.checkpoint_interval, p.checkpoint_steps, p.checkpoint_on_cpu,
        "forward", "elastic3d"
    );

    for (unsigned int it = 0; it < p.nt; ++it) {
        u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        LAUNCH_3DELASTIC_VELOCITY_APM(
            order, launch_config.grid, launch_config.block,
            wf,
            inv_rho_x.data_ptr<float>(),
            inv_rho_y.data_ptr<float>(),
            inv_rho_z.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            grad_ctx, cpml_view, solver
        );

        LAUNCH_3DELASTIC_STRESS_APM(
            order, launch_config.grid, launch_config.block,
            wf,
            alpha_xx.data_ptr<float>(),
            alpha_yy.data_ptr<float>(),
            alpha_zz.data_ptr<float>(),
            lam_xx_yy.data_ptr<float>(),
            lam_xx_zz.data_ptr<float>(),
            lam_yy_xx.data_ptr<float>(),
            lam_yy_zz.data_ptr<float>(),
            lam_zz_xx.data_ptr<float>(),
            lam_zz_yy.data_ptr<float>(),
            mu_xy.data_ptr<float>(),
            mu_xz.data_ptr<float>(),
            mu_yz.data_ptr<float>(),
            p.topo_category.data_ptr<int>(),
            u_this_t,
            grad_ctx, cpml_view, solver
        );

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(wf, 3, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field, p.source.data_ptr<float>(),
                p.sources_loc.data_ptr<int>(), it, nsrc, solver
            );
        }

        checkpoint_runtime.save_forward(static_cast<int>(it),
                                        static_cast<int>(p.nt),
                                        wavefield.checkpoint_tensors());

        if (p.use_boundary_saving) {
            float* fields[9] = {
                wf.vx, wf.vy, wf.vz,
                wf.sxx, wf.syy, wf.szz,
                wf.sxy, wf.sxz, wf.syz
            };
            for (int f = 0; f < 9; ++f) {
                boundary_runtime.save_forward_3d_field(
                    it, p.nt, fields[f],
                    launch_config.grid, launch_config.block,
                    bs, save_width, -p.M, solver, f, f == 8
                );
            }
        }

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(wf, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            record_kernel_3d<<<record_config.grid, record_config.block>>>(
                field, record[irec].data_ptr<float>(),
                p.receivers_loc.data_ptr<int>(), it, nrec, solver
            );
        }
    }

    if (p.use_boundary_saving) {
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,0).select(0,0), wavefield.vx_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,1).select(0,0), wavefield.vy_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,2).select(0,0), wavefield.vz_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,3).select(0,0), wavefield.sxx_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,4).select(0,0), wavefield.syy_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,5).select(0,0), wavefield.szz_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,6).select(0,0), wavefield.sxy_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,7).select(0,0), wavefield.sxz_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,8).select(0,0), wavefield.syz_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = p.use_boundary_saving ? p.last_two : torch::Tensor();   // the tensor Python bound
    out.record = record;

    return out;
}

ForwardRunnerPtr forward_runner(const ForwardInput& in)
{
    return std::make_shared<eqdrv::SgForwardRunner<Driver>>(in);
}

}
