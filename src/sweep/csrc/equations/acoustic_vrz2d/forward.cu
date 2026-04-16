#include <torch/extension.h>
#include <cuda_runtime.h>

#include "acoustic_vrz2d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

namespace acoustic_vrz2d {

ForwardOutput forward(const ForwardInput& in) {

    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "AcousticVRZ CUDA forward expects models [vp, z]");

    auto vp = p.models[0];
    auto z = p.models[1];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int B = N * C;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
                      dx, 0.f, dz};

    AcousticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, 2, true);
    else
        wavefield.allocate(vp, 2, true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto record = torch::zeros({N, p.receivers_loc.size(1), p.nt}, vp.options());

    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, 5, B, 1, nz, nx}, vp.options());

    if (p.use_checkpoint)
        TORCH_CHECK(p.checkpoints.size() == 6, "AcousticVRZ checkpointing expects 6 checkpoint tensors");
    if (p.use_recursive_checkpoint) {
        TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps");
        TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D");
    }

    int save_width = 2 * p.M + 1;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary)
        boundary_saver.allocate(p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two,
                                false, p.use_pinned_memory);
    else
        boundary_saver.allocate(p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, true,
                                1, {}, p.boundary_gpu, p.last_two, false, p.use_pinned_memory);
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    LaplaceParam lap_ctx{nx, 1, p.M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};
    int interval = p.transfer_interval;
    int buf_idx = 0;
    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    int next_ckpt_idx = 0;
    int num_checkpoint_steps = p.use_recursive_checkpoint ? static_cast<int>(p.checkpoint_steps.numel()) : 0;
    const int* checkpoint_steps = p.use_recursive_checkpoint ? p.checkpoint_steps.data_ptr<int>() : nullptr;

    for (int it = 0; it < p.nt; ++it) {
        buf_idx = it % interval;

        auto view = wavefield.view();
        if (u_allt.defined()) {
            u_allt.select(0, it).select(0, 0).copy_(wavefield.u_now_t);
            u_allt.select(0, it).select(0, 1).copy_(wavefield.psix_t);
            u_allt.select(0, it).select(0, 2).copy_(wavefield.psiz_t);
            u_allt.select(0, it).select(0, 3).copy_(wavefield.zetax_t);
            u_allt.select(0, it).select(0, 4).copy_(wavefield.zetaz_t);
        }

        ACOUSTIC_VRZ2D(
            order,
            launch_config.grid,
            launch_config.block,
            view,
            false,
            nullptr,
            vp.data_ptr<float>(),
            z.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_z,
            cpml,
            ctx
        );

        if (p.use_boundary_saving) {
            float* top_ptr = staged_boundary ? boundary_saver.top_gpu.data_ptr<float>() + buf_idx * boundary_saver.top_stride
                                             : bs.top;
            float* bottom_ptr = staged_boundary ? boundary_saver.bottom_gpu.data_ptr<float>() + buf_idx * boundary_saver.bottom_stride
                                                : bs.bottom;
            float* left_ptr = staged_boundary ? boundary_saver.left_gpu.data_ptr<float>() + buf_idx * boundary_saver.left_stride
                                              : bs.left;
            float* right_ptr = staged_boundary ? boundary_saver.right_gpu.data_ptr<float>() + buf_idx * boundary_saver.right_stride
                                               : bs.right;

            boundary_kernel2d<<<launch_config.grid, launch_config.block>>>(
                view.u_now,
                top_ptr,
                bottom_ptr,
                left_ptr,
                right_ptr,
                staged_boundary ? 0 : it,
                save_width,
                -2 * p.M,
                ctx,
                BOUNDARY_SAVE
            );

            if (staged_boundary && (buf_idx == interval - 1 || it == p.nt - 1)) {
                int start = it - buf_idx;
                int len = buf_idx + 1;
                async_copy.record_compute_ready();
                async_copy.wait_for_compute();
                boundary_saver.flush_gpu_to_cpu(start, len, async_copy.copy_stream);
            }
        }

        add_source<<<source_config.grid, source_config.block>>>(
            view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        record_kernel<<<record_config.grid, record_config.block>>>(
            view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        wavefield.swap();

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
                p.checkpoints[3].select(0, ckpt_idx).copy_(wavefield.psiz_t);
                p.checkpoints[4].select(0, ckpt_idx).copy_(wavefield.zetax_t);
                p.checkpoints[5].select(0, ckpt_idx).copy_(wavefield.zetaz_t);
            }
        }
    }

    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(1, 0).copy_(wavefield.u_prev_t);
        boundary_saver.last_two_t.select(1, 1).copy_(wavefield.u_now_t);
    }

    async_copy.synchronize_copy();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

} // namespace acoustic_vrz2d
