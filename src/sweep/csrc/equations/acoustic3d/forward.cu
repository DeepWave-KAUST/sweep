#include <torch/extension.h>
#include <cuda_runtime.h>

#include "acoustic3d.h"
#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/cudautils.h"
#include "../../common/wavetypes.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../launch/config.h"
#include "../../operators/gradient.cuh"
#include "../../operators/laplace.cuh"

namespace acoustic3d {

ForwardOutput forward(const ForwardInput& in)
{

    const auto& p = in;
    ForwardOutput out;

    auto vp = p.models[0];

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);

    int B     = N * C;
    int nsrc  = p.sources_loc.size(1);
    int nrec  = p.receivers_loc.size(1);

    unsigned int nt = in.nt;
    float dt = in.dt;
    int M = in.M;
    int abcn = in.abcn;

    const int order =
        (M <= 4) ? static_cast<int>(2 * M) : -1;

    SolverContext ctx{3, nx, ny, nz, B, dt, nt, M, abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};

    AcousticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, 3, true);
    else
        wavefield.allocate(vp, 3, true);

    // ----------------------------
    // PML parameters
    // ----------------------------
    AcousticCPMLTensor cpml_tensor;
    cpml_tensor.allocate(p.pml_vals, 3);
    auto cpml = cpml_tensor.view();

    // ----------------------------
    // record
    // ----------------------------
    auto record = torch::zeros(
        {N, nrec, nt},
        vp.options()
    );

    // ----------------------------
    // save all wavefields
    // ----------------------------
    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({nt, B, nz, ny, nx}, vp.options());

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        8,
        p.use_checkpoint,
        p.use_recursive_checkpoint,
        p.checkpoint_interval,
        p.checkpoint_steps,
        "forward",
        "acoustic3d"
    );

    // ----------------------------
    // boundary saving (3D)
    // ----------------------------
    int save_width = abcn > 0 ? M + 1 : M;
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2,
            true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            p.use_boundary_saving, 3, 1, ctx, vp, save_width, 2,
            true, true, 1, {}, p.boundary_gpu, p.last_two, p.use_pinned_memory
        );
    }
    auto bs = boundary_saver.view();
    
    // ----------------------------
    // CUDA launch config
    // ----------------------------
    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    float* u_thist = nullptr;

    // Laplace Gradient Contexts
    LaplaceParam lap_ctx{nx, ny, M, p.lap_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx{1, nx, nx*ny, M, p.grad_coes.data_ptr<float>(), dx, dy, dz};
    GradParam grad_ctx_x{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dx, 0.f, 0.f};
    GradParam grad_ctx_y{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dy, 0.f, 0.f};
    GradParam grad_ctx_z{1, 0, 0, M, p.grad_coes.data_ptr<float>(), dz, 0.f, 0.f};

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
    // ============================================================
    // time stepping
    // ============================================================
    for (int it = 0; it < nt; ++it)
    {
        auto view = wavefield.view();

        u_thist = u_allt.defined()
            ? u_allt[it].data_ptr<float>()
            : nullptr;

        ACOUSTIC3D(
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
            grad_ctx_y,
            grad_ctx_z,
            cpml,
            ctx
        );

        if (p.use_boundary_saving) {
            boundary_runtime.save_forward_3d(
                it,
                nt,
                view.u_now,
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                0,
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

        checkpoint_runtime.save_forward(it, static_cast<int>(nt), wavefield.checkpoint_tensors());
    }

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


} // namespace acoustic3d
