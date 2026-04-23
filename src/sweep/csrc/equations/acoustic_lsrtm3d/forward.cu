#include <cuda_runtime.h>
#include <torch/extension.h>

#include "acoustic_lsrtm3d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace acoustic_lsrtm3d {

namespace {

std::vector<torch::Tensor> slice_wavefields(
    const std::vector<torch::Tensor>& tensors,
    size_t start,
    size_t count
) {
    TORCH_CHECK(
        tensors.size() >= start + count,
        "Acoustic LSRTM 3D wavefield buffer does not contain enough tensors."
    );
    return std::vector<torch::Tensor>(
        tensors.begin() + static_cast<long>(start),
        tensors.begin() + static_cast<long>(start + count)
    );
}

} // namespace

ForwardOutput forward(const ForwardInput& in) {
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 3D expects two models: vp and mp.");

    auto vp = p.models[0];
    auto mp = p.models[1];

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
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    AcousticWavefieldTensor bg;
    AcousticWavefieldTensor sc;
    if (!p.wavefields.empty()) {
        TORCH_CHECK(p.wavefields.size() == 18, "Acoustic LSRTM 3D expects 18 wavefield tensors.");
        bg.bind(slice_wavefields(p.wavefields, 0, 9), 3, true);
        sc.bind(slice_wavefields(p.wavefields, 9, 9), 3, true);
    } else {
        bg.allocate(vp, 3, true);
        sc.allocate(vp, 3, true);
    }

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    auto record = torch::zeros({N, nrec, p.nt}, vp.options());
    torch::Tensor bg_utt_all;
    if (p.save_all_wavefields) {
        bg_utt_all = torch::zeros({p.nt, B, nz, ny, nx}, vp.options());
    }

    if (p.use_checkpoint) {
        TORCH_CHECK(p.checkpoints.size() == 8, "Acoustic LSRTM 3D checkpointing expects 8 checkpoint tensors.");
    }
    if (p.use_recursive_checkpoint) {
        TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps.");
        TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D.");
    }

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary) {
        boundary_saver.allocate(
            p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two,
            false, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, p.last_two, false, p.use_pinned_memory
        );
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

    int interval = p.transfer_interval;
    int buf_idx = 0;
    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    int next_ckpt_idx = 0;
    int num_checkpoint_steps = p.use_recursive_checkpoint ? static_cast<int>(p.checkpoint_steps.numel()) : 0;
    const int* checkpoint_steps = p.use_recursive_checkpoint ? p.checkpoint_steps.data_ptr<int>() : nullptr;

    for (int it = 0; it < p.nt; ++it) {
        buf_idx = it % interval;

        auto bg_view = bg.view();
        auto sc_view = sc.view();
        float* bg_utt_ptr = bg_utt_all.defined() ? bg_utt_all[it].data_ptr<float>() : nullptr;

        ACOUSTIC_LSRTM3D_COUPLED(
            order,
            launch_config.grid,
            launch_config.block,
            bg_view,
            sc_view,
            p.save_all_wavefields,
            bg_utt_ptr,
            vp.data_ptr<float>(),
            mp.data_ptr<float>(),
            lap_ctx,
            grad_ctx,
            grad_ctx_x,
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        if (p.use_boundary_saving) {
            float* top_ptr = nullptr;
            float* bottom_ptr = nullptr;
            float* front_ptr = nullptr;
            float* back_ptr = nullptr;
            float* left_ptr = nullptr;
            float* right_ptr = nullptr;

            if (staged_boundary) {
                top_ptr = boundary_saver.top_gpu.data_ptr<float>() + buf_idx * boundary_saver.top_stride;
                bottom_ptr = boundary_saver.bottom_gpu.data_ptr<float>() + buf_idx * boundary_saver.bottom_stride;
                front_ptr = boundary_saver.front_gpu.data_ptr<float>() + buf_idx * boundary_saver.front_stride;
                back_ptr = boundary_saver.back_gpu.data_ptr<float>() + buf_idx * boundary_saver.back_stride;
                left_ptr = boundary_saver.left_gpu.data_ptr<float>() + buf_idx * boundary_saver.left_stride;
                right_ptr = boundary_saver.right_gpu.data_ptr<float>() + buf_idx * boundary_saver.right_stride;
            } else {
                top_ptr = bs.top;
                bottom_ptr = bs.bottom;
                front_ptr = bs.front;
                back_ptr = bs.back;
                left_ptr = bs.left;
                right_ptr = bs.right;
            }

            boundary_kernel3d<<<launch_config.grid, launch_config.block>>>(
                bg_view.u_now,
                top_ptr,
                bottom_ptr,
                front_ptr,
                back_ptr,
                left_ptr,
                right_ptr,
                staged_boundary ? 0 : it,
                save_width,
                0,
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

        add_source_3d<<<source_config.grid, source_config.block>>>(
            bg_view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        record_kernel_3d<<<record_config.grid, record_config.block>>>(
            sc_view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        bg.swap();
        sc.swap();

        if (p.use_checkpoint) {
            int ckpt_idx = -1;
            if (p.use_recursive_checkpoint) {
                if (next_ckpt_idx < num_checkpoint_steps && checkpoint_steps[next_ckpt_idx] == it + 1) {
                    ckpt_idx = next_ckpt_idx++;
                }
            } else if (((it + 1) % p.checkpoint_interval == 0) && (it + 1 < p.nt)) {
                ckpt_idx = (it + 1) / p.checkpoint_interval;
            }

            if (ckpt_idx >= 0) {
                p.checkpoints[0].select(0, ckpt_idx).copy_(bg.u_prev_t);
                p.checkpoints[1].select(0, ckpt_idx).copy_(bg.u_now_t);
                p.checkpoints[2].select(0, ckpt_idx).copy_(bg.psix_t);
                p.checkpoints[3].select(0, ckpt_idx).copy_(bg.psiy_t);
                p.checkpoints[4].select(0, ckpt_idx).copy_(bg.psiz_t);
                p.checkpoints[5].select(0, ckpt_idx).copy_(bg.zetax_t);
                p.checkpoints[6].select(0, ckpt_idx).copy_(bg.zetay_t);
                p.checkpoints[7].select(0, ckpt_idx).copy_(bg.zetaz_t);
            }
        }
    }

    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(1, 0).copy_(bg.u_prev_t);
        boundary_saver.last_two_t.select(1, 1).copy_(bg.u_now_t);
    }

    out.wavefield = bg_utt_all.defined() ? bg_utt_all : torch::Tensor();
    out.last_two = boundary_saver.last_two_t;
    out.record = record;
    return out;
}

} // namespace acoustic_lsrtm3d
