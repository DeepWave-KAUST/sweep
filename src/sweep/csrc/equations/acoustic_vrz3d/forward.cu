#include <torch/extension.h>
#include <cuda_runtime.h>

#include "acoustic_vrz3d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

namespace acoustic_vrz3d {

ForwardOutput forward(const ForwardInput& in)
{
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ3D CUDA forward expects models [vp, z]");

    auto vp = p.models[0];
    auto z = p.models[1];
    auto inv_z = torch::reciprocal(z);

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;
    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, dy, dz};

    AcousticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, 3, true);
    else
        wavefield.allocate(vp, 3, true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto record = torch::zeros({N, nrec, p.nt}, vp.options());

    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, 7, B, nz, ny, nx}, vp.options());

    if (p.use_checkpoint)
        TORCH_CHECK(p.checkpoints.size() == 8, "AcousticVRZ3D checkpointing expects 8 checkpoint tensors");
    if (p.use_recursive_checkpoint) {
        TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps");
        TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D");
    }

    int save_width = p.M + 1;
    int boundary_offset = -p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    LaplaceParam lap_ctx{nx, ny, p.M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

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
    int next_ckpt_idx = 0;
    int num_checkpoint_steps = p.use_recursive_checkpoint ? static_cast<int>(p.checkpoint_steps.numel()) : 0;
    const int* checkpoint_steps = p.use_recursive_checkpoint ? p.checkpoint_steps.data_ptr<int>() : nullptr;

    for (int it = 0; it < p.nt; ++it) {
        auto view = wavefield.view();

        ACOUSTIC_VRZ3D(
            order,
            launch_config.grid,
            launch_config.block,
            view,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            inv_z.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        if (p.use_boundary_saving) {
            boundary_runtime.save_forward_3d(
                it,
                p.nt,
                view.u_now,
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                boundary_offset,
                ctx
            );
        }

        add_source_3d<<<source_config.grid, source_config.block>>>(
            view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        record_kernel_3d<<<record_config.grid, record_config.block>>>(
            view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        wavefield.swap();

        if (u_allt.defined()) {
            u_allt.select(0, it).select(0, 0).copy_(wavefield.u_now_t.squeeze(1));
            u_allt.select(0, it).select(0, 1).copy_(wavefield.psix_t.squeeze(1));
            u_allt.select(0, it).select(0, 2).copy_(wavefield.psiy_t.squeeze(1));
            u_allt.select(0, it).select(0, 3).copy_(wavefield.psiz_t.squeeze(1));
            u_allt.select(0, it).select(0, 4).copy_(wavefield.zetax_t.squeeze(1));
            u_allt.select(0, it).select(0, 5).copy_(wavefield.zetay_t.squeeze(1));
            u_allt.select(0, it).select(0, 6).copy_(wavefield.zetaz_t.squeeze(1));
        }

        if (p.use_checkpoint) {
            int ckpt_idx = -1;
            if (p.use_recursive_checkpoint) {
                if (next_ckpt_idx < num_checkpoint_steps && checkpoint_steps[next_ckpt_idx] == it + 1)
                    ckpt_idx = next_ckpt_idx++;
            } else if (((it + 1) % p.checkpoint_interval == 0) && (it + 1 < p.nt)) {
                ckpt_idx = (it + 1) / p.checkpoint_interval;
            }

            if (ckpt_idx >= 0) {
                p.checkpoints[0].select(0, ckpt_idx).copy_(wavefield.u_prev_t);
                p.checkpoints[1].select(0, ckpt_idx).copy_(wavefield.u_now_t);
                p.checkpoints[2].select(0, ckpt_idx).copy_(wavefield.psix_t);
                p.checkpoints[3].select(0, ckpt_idx).copy_(wavefield.psiy_t);
                p.checkpoints[4].select(0, ckpt_idx).copy_(wavefield.psiz_t);
                p.checkpoints[5].select(0, ckpt_idx).copy_(wavefield.zetax_t);
                p.checkpoints[6].select(0, ckpt_idx).copy_(wavefield.zetay_t);
                p.checkpoints[7].select(0, ckpt_idx).copy_(wavefield.zetaz_t);
            }
        }
    }

    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(1, 0).copy_(wavefield.u_prev_t);
        boundary_saver.last_two_t.select(1, 1).copy_(wavefield.u_now_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

} // namespace acoustic_vrz3d
