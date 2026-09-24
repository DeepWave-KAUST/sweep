#include <torch/extension.h>
#include <cuda_runtime.h>


#include "elastic2d.h"
#include "kernels.cuh"

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

namespace elastic2d {

ForwardOutput forward(const ForwardInput& in)
{
    return eqdrv::sg_generic_forward<Driver>(in);
}


// ---------------------------------------------------------------------------
// APM (Cao & Chen 2018) forward
// ---------------------------------------------------------------------------
// Mirrors ``forward`` above but launches the APM kernel variants which read
// effective parameters (lam_eff, mu_eff, mu_xz_node, rho_x_eff, rho_z_eff)
// and the per-cell category tensor.  The propagator is expected to have set
// ``in.free_surface=false`` and ``in.has_topo=false`` for this path; the
// APM moduli themselves carry the free-surface BC.
ForwardOutput apm_forward(const ForwardInput& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));

    const auto& p = in;
    ForwardOutput out;

    SWEEP_CHECK(p.it_begin == 0 &&
                (p.it_end < 0 || p.it_end == static_cast<int>(p.nt)) &&
                p.step_phase == 0,
                "stepped forward not supported for the elastic2d APM path");
    SWEEP_CHECK(p.models.size() >= 11,
                "elastic2d::apm_forward expects 11 model tensors: "
                "[vp, vs, rho, lame_lambda, lame_mu, lame_lambda_2mu, "
                "lam_eff, mu_eff, mu_xz_node, rho_x_eff, rho_z_eff]");
    SWEEP_CHECK(p.topo_category.defined() && p.topo_category.numel() > 0,
                "elastic2d::apm_forward requires topo_category int32 tensor");

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    auto vp        = p.models[0];   // unused but required for shape probe
    auto lam_eff   = p.models[6];
    auto mu_eff    = p.models[7];
    auto mu_xz     = p.models[8];
    auto rho_x_eff = p.models[9];
    auto rho_z_eff = p.models[10];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    ElasticWavefieldTensor wavefield;
    SWEEP_CHECK(!p.wavefields.empty(),
                "elastic2d/apm_forward requires the propagator-bound wavefields "
                "(15-slot layout); nothing allocates them here");
    wavefield.bind(p.wavefields, true);
    auto wf = wavefield.view();

    ElasticCPMLTensor cpml;
    cpml.bind(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto record = bound_required(p.record_out, {nrec_fields, B, nrec, p.nt}, vp.options(), "record_out");

    torch::Tensor u_allt;
    if (p.save_all_wavefields) u_allt = bound_required(p.u_allt_out, {p.nt, 2, B, nz, nx}, vp.options(), "u_allt_out");

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn,
                         /* free_surface */ false,
                         p.lap_coes.data_ptr<float>(),
                         p.grad_coes.data_ptr<float>(),
                         dx, 0.f, dz};
    solver.topo_category = p.topo_category.data_ptr<int>();
    solver.use_apm = true;
    solver.set_cut_mask(p.cut_face_mask);   // cut-aware phys bounds (0 = single domain)
    elastic_init_aux_slabs(solver, wavefield);

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary)
        boundary_saver.allocate(p.use_boundary_saving, 2, 5, solver, vp, save_width, 1, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
    else
        boundary_saver.allocate(p.use_boundary_saving, 2, 5, solver, vp, save_width, 1, true, true, 1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    float* u_this_t = nullptr;

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    BoundaryRuntime boundary_runtime(
        boundary_saver, 2, p.use_boundary_saving, p.boundary_on_cpu,
        p.boundary_on_disk, p.boundary_disk_async_read, p.transfer_interval,
        p.boundary_ring_buffers, p.boundary_disk_files,
        async_copy.compute_stream, async_copy.copy_stream
    );

    for (unsigned int it = 0; it < p.nt; ++it) {
        u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        LAUNCH_ELASTIC_VELOCITY_APM(
            order, launch_config.grid, launch_config.block,
            wf,
            rho_x_eff.data_ptr<float>(),
            rho_z_eff.data_ptr<float>(),
            solver.topo_category,
            grad_ctx, cpml_view, solver
        );

        LAUNCH_ELASTIC_STRESS_APM(
            order, launch_config.grid, launch_config.block,
            wf,
            lam_eff.data_ptr<float>(),
            mu_eff.data_ptr<float>(),
            mu_xz.data_ptr<float>(),
            solver.topo_category,
            u_this_t,
            grad_ctx, cpml_view, solver
        );

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(wf, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<source_config.grid, source_config.block>>>(
                field, p.source.data_ptr<float>(), p.sources_loc.data_ptr<int>(),
                it, nsrc, solver
            );
        }

        if (p.use_boundary_saving) {
            float* fields[5] = {wf.vx, wf.vz, wf.sxx, wf.szz, wf.sxz};
            for (int f = 0; f < 5; ++f) {
                boundary_runtime.save_forward_2d_field(
                    it, p.nt, fields[f], launch_config.grid, launch_config.block,
                    bs, save_width, -p.M, solver, f, f == 4
                );
            }
        }

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(wf, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            record_kernel<<<record_config.grid, record_config.block>>>(
                field, record[irec].data_ptr<float>(),
                p.receivers_loc.data_ptr<int>(), it, nrec, solver
            );
        }
    }

    if (p.use_boundary_saving) {
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,0).select(0,0), wavefield.vx_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,1).select(0,0), wavefield.vz_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,2).select(0,0), wavefield.sxx_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,3).select(0,0), wavefield.szz_t);
        copy_tensor_cuda_async(boundary_saver.last_two.select(0,4).select(0,0), wavefield.sxz_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_th_;
    out.record = record;
    return out;
}

ForwardRunnerPtr forward_runner(const ForwardInput& in)
{
    return std::make_shared<eqdrv::SgForwardRunner<Driver>>(in);
}

}
