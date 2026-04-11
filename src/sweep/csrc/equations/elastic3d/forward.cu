#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

#include "elastic3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"

namespace elastic3d {

namespace {

void save_checkpoint_state_3d(
    const ForwardInput& p,
    const ElasticWavefieldTensor& wavefield,
    int checkpoint_idx
)
{
    p.checkpoints[0].select(0, checkpoint_idx).copy_(wavefield.vx_t);
    p.checkpoints[1].select(0, checkpoint_idx).copy_(wavefield.vy_t);
    p.checkpoints[2].select(0, checkpoint_idx).copy_(wavefield.vz_t);
    p.checkpoints[3].select(0, checkpoint_idx).copy_(wavefield.sxx_t);
    p.checkpoints[4].select(0, checkpoint_idx).copy_(wavefield.syy_t);
    p.checkpoints[5].select(0, checkpoint_idx).copy_(wavefield.szz_t);
    p.checkpoints[6].select(0, checkpoint_idx).copy_(wavefield.sxy_t);
    p.checkpoints[7].select(0, checkpoint_idx).copy_(wavefield.sxz_t);
    p.checkpoints[8].select(0, checkpoint_idx).copy_(wavefield.syz_t);
    p.checkpoints[9].select(0, checkpoint_idx).copy_(wavefield.m_vxx_t);
    p.checkpoints[10].select(0, checkpoint_idx).copy_(wavefield.m_vxy_t);
    p.checkpoints[11].select(0, checkpoint_idx).copy_(wavefield.m_vxz_t);
    p.checkpoints[12].select(0, checkpoint_idx).copy_(wavefield.m_vyx_t);
    p.checkpoints[13].select(0, checkpoint_idx).copy_(wavefield.m_vyy_t);
    p.checkpoints[14].select(0, checkpoint_idx).copy_(wavefield.m_vyz_t);
    p.checkpoints[15].select(0, checkpoint_idx).copy_(wavefield.m_vzx_t);
    p.checkpoints[16].select(0, checkpoint_idx).copy_(wavefield.m_vzy_t);
    p.checkpoints[17].select(0, checkpoint_idx).copy_(wavefield.m_vzz_t);
    p.checkpoints[18].select(0, checkpoint_idx).copy_(wavefield.m_sxxx_t);
    p.checkpoints[19].select(0, checkpoint_idx).copy_(wavefield.m_sxxy_t);
    p.checkpoints[20].select(0, checkpoint_idx).copy_(wavefield.m_sxxz_t);
    p.checkpoints[21].select(0, checkpoint_idx).copy_(wavefield.m_syyx_t);
    p.checkpoints[22].select(0, checkpoint_idx).copy_(wavefield.m_syyy_t);
    p.checkpoints[23].select(0, checkpoint_idx).copy_(wavefield.m_syyz_t);
    p.checkpoints[24].select(0, checkpoint_idx).copy_(wavefield.m_szzx_t);
    p.checkpoints[25].select(0, checkpoint_idx).copy_(wavefield.m_szzy_t);
    p.checkpoints[26].select(0, checkpoint_idx).copy_(wavefield.m_szzz_t);
    p.checkpoints[27].select(0, checkpoint_idx).copy_(wavefield.m_sxyx_t);
    p.checkpoints[28].select(0, checkpoint_idx).copy_(wavefield.m_sxyy_t);
    p.checkpoints[29].select(0, checkpoint_idx).copy_(wavefield.m_sxyz_t);
    p.checkpoints[30].select(0, checkpoint_idx).copy_(wavefield.m_sxzx_t);
    p.checkpoints[31].select(0, checkpoint_idx).copy_(wavefield.m_sxzy_t);
    p.checkpoints[32].select(0, checkpoint_idx).copy_(wavefield.m_sxzz_t);
    p.checkpoints[33].select(0, checkpoint_idx).copy_(wavefield.m_syzx_t);
    p.checkpoints[34].select(0, checkpoint_idx).copy_(wavefield.m_syzy_t);
    p.checkpoints[35].select(0, checkpoint_idx).copy_(wavefield.m_syzz_t);
}

} // namespace

ForwardOutput forward(const ForwardInput& in)
{
    const auto& p = in;
    ForwardOutput out;

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    // parse model parameters
    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N  = vp.size(0);
    int C  = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);

    int B = N * C;

    ElasticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, true);
    else
        wavefield.allocate(vp, 3);
    if (!wavefield.m_syzx_t.defined())
        wavefield.m_syzx_t = torch::zeros_like(vp);
    auto wf = wavefield.view();

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto record = torch::zeros({nrec_fields, B, nrec, p.nt}, vp.options());

    if (p.use_checkpoint) {
        TORCH_CHECK(p.checkpoints.size() == 36, "Elastic 3D checkpointing expects 36 checkpoint tensors");
        if (p.use_recursive_checkpoint) {
            TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps");
            TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D");
        } else {
            TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
        }
    }

    torch::Tensor u_allt;
    if (p.save_all_wavefields)
        u_allt = torch::zeros({p.nt, 3, B, nz, ny, nx}, vp.options());

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    SolverContext solver{3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, dy, dz};
    
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    boundary_saver.allocate(
        p.use_boundary_saving, 3, 9, solver, vp, save_width, 1,
        true, !p.boundary_on_cpu, p.transfer_interval,
        p.boundary_on_cpu ? p.boundary_cpu : std::vector<torch::Tensor>{},
        p.boundary_gpu,
        p.last_two,
        false, p.use_pinned_memory
    );
    // auto bs = boundary_saver.view();

    SGradParam grad_ctx{1, nx, nx*ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    float* u_this_t = nullptr;

    int interval = p.transfer_interval;
    int buf_idx = 0;

    int gpu_idx = 0;

    // For copying data
    AsyncCopyContext async_copy(p.boundary_on_cpu && p.use_boundary_saving);
    int next_ckpt_idx = 0;
    int num_checkpoint_steps = p.use_recursive_checkpoint ? static_cast<int>(p.checkpoint_steps.numel()) : 0;
    const int* checkpoint_steps = p.use_recursive_checkpoint ? p.checkpoint_steps.data_ptr<int>() : nullptr;

    for (unsigned int it = 0; it < p.nt; ++it) {

        buf_idx = it % interval;
        u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        LAUNCH_3DELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        ); // t+0.5

        LAUNCH_3DELASTIC_STRESS(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            u_this_t,
            grad_ctx,
            cpml_view,
            solver
        ); // t+1.0

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

        if (p.use_checkpoint) {
            int ckpt_idx = -1;
            if (p.use_recursive_checkpoint) {
                if (next_ckpt_idx < num_checkpoint_steps && checkpoint_steps[next_ckpt_idx] == static_cast<int>(it + 1)) {
                    ckpt_idx = next_ckpt_idx;
                    ++next_ckpt_idx;
                }
            } else if (((it + 1) % p.checkpoint_interval == 0) && (it + 1 < p.nt)) {
                ckpt_idx = static_cast<int>((it + 1) / p.checkpoint_interval);
            }
            if (ckpt_idx >= 0) {
                save_checkpoint_state_3d(p, wavefield, ckpt_idx);
            }
        }

        if (p.use_boundary_saving) {

            float* fields[9] = {
                wf.vx, wf.vy, wf.vz,
                wf.sxx, wf.syy, wf.szz,
                wf.sxy, wf.sxz, wf.syz
            };

            for (int f = 0; f < 9; ++f) {
                gpu_idx = p.boundary_on_cpu ? f * interval + buf_idx : f * p.nt;
                boundary_kernel3d<<<launch_config.grid, launch_config.block>>>(
                    fields[f],
                    (p.boundary_on_cpu ? boundary_saver.top_gpu : boundary_saver.top_t).data_ptr<float>()    + gpu_idx * boundary_saver.top_stride,
                    (p.boundary_on_cpu ? boundary_saver.bottom_gpu : boundary_saver.bottom_t).data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride,
                    (p.boundary_on_cpu ? boundary_saver.front_gpu : boundary_saver.front_t).data_ptr<float>()  + gpu_idx * boundary_saver.front_stride,
                    (p.boundary_on_cpu ? boundary_saver.back_gpu : boundary_saver.back_t).data_ptr<float>()   + gpu_idx * boundary_saver.back_stride,
                    (p.boundary_on_cpu ? boundary_saver.left_gpu : boundary_saver.left_t).data_ptr<float>()   + gpu_idx * boundary_saver.left_stride,
                    (p.boundary_on_cpu ? boundary_saver.right_gpu : boundary_saver.right_t).data_ptr<float>()  + gpu_idx * boundary_saver.right_stride,

                    p.boundary_on_cpu ? 0 : it,
                    save_width,
                    -p.M, // offset
                    solver,
                    BOUNDARY_SAVE
                );
            }

            if (p.boundary_on_cpu && (buf_idx == interval - 1 || it == p.nt - 1)) {
                int start = it - buf_idx;
                int len = buf_idx + 1;

                async_copy.record_compute_ready();
                async_copy.wait_for_compute();
                boundary_saver.flush_gpu_to_cpu(start, len, async_copy.copy_stream);

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

    async_copy.synchronize_copy();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;

}

}
