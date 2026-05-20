#include <torch/extension.h>
#include <cuda_runtime.h>


#include <c10/cuda/CUDAGuard.h>
#include "acoustic2d.h"
#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/laplace.cuh"
#include "../../operators/gradient.cuh"

namespace acoustic2d {

ForwardOutput forward(const ForwardInput& in) {
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

    const auto& p = in;
    ForwardOutput out;

    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int B = N * C;

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext ctx{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AcousticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, 2, true);
    else
        wavefield.allocate(vp, 2, true);

    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 2);
    auto cpml = cpml_tensor.view();

    auto record = torch::zeros(
        {N, p.receivers_loc.size(1), p.nt},
        vp.options()
    );

    // Wavefields for all timestep
    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, B, nz, nx}, vp.options());

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        6,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "forward",
        "acoustic2d"
    );

    int save_width = p.abcn > 0 ? p.M + 1 : p.M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary)
        boundary_saver.allocate(p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory);
    else
        boundary_saver.allocate(p.use_boundary_saving, 2, 1, ctx, vp, save_width, 2, true, true, 1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory);
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    float* u_thist = nullptr;

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
    for (int it = 0; it < p.nt; ++it) {

        auto view = wavefield.view();

        u_thist = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        ACOUSTIC2D(
            order,
            launch_config.grid,
            launch_config.block,
            view,
            p.save_all_wavefields,
            u_thist,
            vp.data_ptr<float>(),
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
                view.u_now,
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                0,
                ctx
            );
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

        checkpoint_runtime.save_forward(it, static_cast<int>(p.nt), wavefield.checkpoint_tensors());

    }

    // Save the last two time steps for backward
    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(1,0).copy_(wavefield.u_prev_t);
        boundary_saver.last_two_t.select(1,1).copy_(wavefield.u_now_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

} // namespace acoustic2d
