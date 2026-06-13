#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "acoustic_lsrtm2d.h"
#include "kernels.cuh"
#include "../../common/acoustic.h"
#include "../../common/boundary_runtime.cuh"
#include "../../common/boundarysaver.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

namespace acoustic_lsrtm2d {

namespace {

std::vector<torch::Tensor> slice_wavefields(
    const std::vector<torch::Tensor>& tensors,
    size_t start,
    size_t count
) {
    TORCH_CHECK(
        tensors.size() >= start + count,
        "Acoustic LSRTM 2D wavefield buffer does not contain enough tensors."
    );
    return std::vector<torch::Tensor>(tensors.begin() + static_cast<long>(start),
                                      tensors.begin() + static_cast<long>(start + count));
}

} // namespace

ForwardOutput forward(const ForwardInput& in) {
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(p.models.size() == 2, "Acoustic LSRTM 2D expects two models: vp and mp.");

    auto vp = p.models[0];
    auto mp = p.models[1];

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
                      p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor bg;
    AcousticWavefieldTensor sc;
    if (!p.wavefields.empty()) {
        TORCH_CHECK(p.wavefields.size() == 18, "Acoustic LSRTM 2D expects 18 wavefield tensors (bg+sc, each 9 with psi double-buffer).");
        bg.bind(slice_wavefields(p.wavefields, 0, 9), 2, true);
        sc.bind(slice_wavefields(p.wavefields, 9, 9), 2, true);
    } else {
        bg.allocate(vp, 2, true, /*double_buffer_psi=*/true);
        sc.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    }

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto record = torch::zeros({N, nrec, p.nt}, vp.options());

    torch::Tensor bg_utt_all;
    if (p.save_all_wavefields)
        bg_utt_all = torch::zeros({p.nt, B, nz, nx}, vp.options());

    if (p.use_checkpoint)
        TORCH_CHECK(p.checkpoints.size() == 6, "Acoustic LSRTM 2D checkpointing expects 6 checkpoint tensors.");
    if (p.use_recursive_checkpoint) {
        TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps.");
        TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D.");
    }

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, false,
            p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, true,
            1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory
        );
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    LaplaceParam lap_ctx{nx, 1, p.M, p.lap_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    GradParam grad_ctx_x{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, p.M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

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
        6,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "acoustic_lsrtm2d"
    );

    for (int it = 0; it < p.nt; ++it) {
        auto bg_view = bg.view();
        auto sc_view = sc.view();
        float* bg_utt_ptr = bg_utt_all.defined() ? bg_utt_all[it].data_ptr<float>() : nullptr;

        ACOUSTIC_LSRTM2D_COUPLED(
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
            grad_ctx_z,
            cpml,
            ctx
        );

        if (p.use_boundary_saving) {
            boundary_runtime.save_forward_2d(
                it,
                p.nt,
                bg_view.u_now,
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                0,
                ctx
            );
        }

        add_source<<<source_config.grid, source_config.block>>>(
            bg_view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );

        record_kernel<<<record_config.grid, record_config.block>>>(
            sc_view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );

        bg.swap_pml();   // rotate u AND psi<->psin: race-free psi double-buffer
        sc.swap_pml();

        checkpoint_runtime.save_forward(it, static_cast<int>(p.nt), bg.checkpoint_tensors());
    }

    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(1, 0).copy_(bg.u_prev_t);
        boundary_saver.last_two_t.select(1, 1).copy_(bg.u_now_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = bg_utt_all;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;
    return out;
}

} // namespace acoustic_lsrtm2d
