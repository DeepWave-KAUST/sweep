#include <torch/extension.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include "elastic_tti_sg3d.h"
#include "kernels.cuh"
#include "tensors.h"

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

namespace elastic_tti_sg3d {

ForwardOutput forward(const ForwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 22, "ElasticTTISG3D forward expects prepared models: rho plus 21 stiffness tensors");
    TORCH_CHECK(p.pml_vals.size() == 12, "ElasticTTISG3D forward expects cpmls PML profiles (12)");
    TORCH_CHECK(!p.free_surface, "ElasticTTISG3D has no free-surface support (anisotropic media reject the image method)");
    if (p.use_checkpoint) {
        TORCH_CHECK(!p.use_recursive_checkpoint, "ElasticTTISG3D recursive checkpointing is not implemented yet");
        TORCH_CHECK(p.checkpoints.size() == 36, "ElasticTTISG3D checkpointing expects 36 checkpoint tensors");
        TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    }

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

    ElasticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, true);
    else
        wavefield.allocate(rho, 3);
    auto wf = wavefield.view();
    auto model = stiffness_view(p.models);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
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
        u_allt = torch::zeros({p.nt, 3, B, nz, ny, nx}, rho.options());

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    EffectiveBoundarySaver boundary_saver;
    const int save_width = solver.M + 1;
    const bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    boundary_saver.allocate(
        p.use_boundary_saving, 3, 9, solver, rho, save_width, 1,
        true, !staged_boundary, staged_boundary ? p.transfer_interval : 1,
        staged_boundary ? p.boundary_cpu : std::vector<torch::Tensor>{},
        p.boundary_gpu,
        p.last_two, p.use_pinned_memory
    );
    auto bs = boundary_saver.view();

    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    float* u_this_t = nullptr;

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        3,
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
        36,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "elastic_tti_sg3d"
    );

    for (unsigned int it = 0; it < p.nt; ++it) {

        u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        LAUNCH_ELASTIC_TTI_SG3D_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            model.rho,
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_ELASTIC_TTI_SG3D_STRESS(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            model,
            u_this_t,
            grad_ctx,
            cpml_view,
            solver
        );

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = elastic_field_ptr(wf, 3, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field,
                p.source.data_ptr<float>(),
                p.sources_loc.data_ptr<int>(),
                it,
                nsrc,
                solver
            );
        }

        checkpoint_runtime.save_forward(static_cast<int>(it), static_cast<int>(p.nt), wavefield.checkpoint_tensors());

        if (p.use_boundary_saving) {

            float* fields[9] = {
                wf.vx, wf.vy, wf.vz,
                wf.sxx, wf.syy, wf.szz,
                wf.sxy, wf.sxz, wf.syz
            };

            for (int f = 0; f < 9; ++f) {
                boundary_runtime.save_forward_3d_field(
                    it,
                    p.nt,
                    fields[f],
                    launch_config.grid,
                    launch_config.block,
                    bs,
                    save_width,
                    -p.M,
                    solver,
                    f,
                    f == 8
                );
            }

        }

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(wf, 3, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            record_kernel_3d<<<record_config.grid, record_config.block>>>(
                field,
                record[irec].data_ptr<float>(),
                p.receivers_loc.data_ptr<int>(),
                it,
                nrec,
                solver
            );
        }

    }

    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(0,0).select(0,0).copy_(wavefield.vx_t);
        boundary_saver.last_two_t.select(0,1).select(0,0).copy_(wavefield.vy_t);
        boundary_saver.last_two_t.select(0,2).select(0,0).copy_(wavefield.vz_t);
        boundary_saver.last_two_t.select(0,3).select(0,0).copy_(wavefield.sxx_t);
        boundary_saver.last_two_t.select(0,4).select(0,0).copy_(wavefield.syy_t);
        boundary_saver.last_two_t.select(0,5).select(0,0).copy_(wavefield.szz_t);
        boundary_saver.last_two_t.select(0,6).select(0,0).copy_(wavefield.sxy_t);
        boundary_saver.last_two_t.select(0,7).select(0,0).copy_(wavefield.sxz_t);
        boundary_saver.last_two_t.select(0,8).select(0,0).copy_(wavefield.syz_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

}
