#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "elastic_tti_2nd2d.h"
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

namespace elastic_tti_2nd2d {

ForwardOutput forward(const ForwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 7, "ElasticTTI2nd forward expects prepared models: rho plus 6 stiffness tensors");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTI2nd forward expects cpmls PML profiles");
    TORCH_CHECK(!p.free_surface, "ElasticTTI2nd has no free-surface support (anisotropic media reject the image method)");
    if (p.use_checkpoint) {
        TORCH_CHECK(!p.use_recursive_checkpoint, "ElasticTTI2nd recursive checkpointing is not implemented yet");
        TORCH_CHECK(p.checkpoints.size() == 14, "ElasticTTI2nd checkpointing expects 14 checkpoint tensors");
        TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    }

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];

    WavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields);
    else
        wavefield.allocate(rho);
    auto model = stiffness_view(p.models);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    const int nsrc = p.sources_loc.size(1);
    const int nrec = p.receivers_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    const int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto record = torch::zeros({nrec_fields, B, nrec, p.nt}, rho.options());

    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, 2, B, nz, nx}, rho.options());

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto sxx_ws = torch::zeros_like(rho);
    auto szz_ws = torch::zeros_like(rho);
    auto sxz_ws = torch::zeros_like(rho);

    EffectiveBoundarySaver boundary_saver;
    const int save_width = solver.M + 1;
    const bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            p.use_boundary_saving, 2, 2, solver, rho, save_width, 2,
            true, false, p.transfer_interval,
            p.boundary_cpu, p.boundary_gpu,
            p.last_two, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            p.use_boundary_saving, 2, 2, solver, rho, save_width, 2,
            true, true, 1,
            {}, p.boundary_gpu,
            p.last_two, p.use_pinned_memory
        );
    }
    auto bs = boundary_saver.view();
    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        2,
        p.use_boundary_saving,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        14,
        p.use_checkpoint,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "elastic_tti_2nd2d"
    );

    for (unsigned int it = 0; it < p.nt; ++it) {
        auto wf = wavefield.view();

        TTI2ND_LAUNCH(tti2nd_stress_kernel, order,
            launch_config.grid, launch_config.block,
            wf, model, cpml_view, grad_ctx, solver,
            sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>());

        TTI2ND_LAUNCH(tti2nd_displacement_kernel, order,
            launch_config.grid, launch_config.block,
            wf, model,
            sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>(),
            nullptr, cpml_view, grad_ctx, solver);

        if (p.use_boundary_saving) {
            // Boundary ring of the pre-step state W_it (mirrors acoustic2d:
            // save u_now BEFORE the source of this step lands in u_next).
            float* fields[2] = { wf.ux, wf.uz };
            for (int f = 0; f < 2; ++f) {
                boundary_runtime.save_forward_2d_field(
                    it, p.nt, fields[f],
                    launch_config.grid, launch_config.block,
                    bs, save_width, -p.M, solver, f, f == 1
                );
            }
        }

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = field_ptr(wf, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<source_config.grid, source_config.block>>>(
                field,
                p.source.data_ptr<float>(),
                p.sources_loc.data_ptr<int>(),
                it,
                nsrc,
                solver
            );
        }

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = field_ptr(wf, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            record_kernel<<<record_config.grid, record_config.block>>>(
                field,
                record[irec].data_ptr<float>(),
                p.receivers_loc.data_ptr<int>(),
                it,
                nrec,
                solver
            );
        }

        if (u_allt.defined()) {
            u_allt[it].select(0, 0).copy_(wavefield.ux_nxt_t.view({B, nz, nx}));
            u_allt[it].select(0, 1).copy_(wavefield.uz_nxt_t.view({B, nz, nx}));
        }

        wavefield.swap_u();

        checkpoint_runtime.save_forward(static_cast<int>(it), static_cast<int>(p.nt), wavefield.checkpoint_tensors());
    }

    if (p.use_boundary_saving) {
        // (storage, level): level 0 = W_{nt-1} (u_pre after final swap),
        // level 1 = W_nt (u_now after final swap).
        boundary_saver.last_two_t.select(0, 0).select(0, 0).copy_(wavefield.ux_pre_t);
        boundary_saver.last_two_t.select(0, 0).select(0, 1).copy_(wavefield.ux_t);
        boundary_saver.last_two_t.select(0, 1).select(0, 0).copy_(wavefield.uz_pre_t);
        boundary_saver.last_two_t.select(0, 1).select(0, 1).copy_(wavefield.uz_t);
    }
    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;
    return out;
}

}
