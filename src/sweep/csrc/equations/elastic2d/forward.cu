#include <torch/extension.h>
#include <cuda_runtime.h>

#include "elastic2d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace elastic2d {

namespace {

void save_checkpoint_state_2d(
    const ForwardInput& p,
    const ElasticWavefieldTensor& wavefield,
    int checkpoint_idx
)
{
    p.checkpoints[0].select(0, checkpoint_idx).copy_(wavefield.vx_t);
    p.checkpoints[1].select(0, checkpoint_idx).copy_(wavefield.vz_t);
    p.checkpoints[2].select(0, checkpoint_idx).copy_(wavefield.sxx_t);
    p.checkpoints[3].select(0, checkpoint_idx).copy_(wavefield.szz_t);
    p.checkpoints[4].select(0, checkpoint_idx).copy_(wavefield.sxz_t);
    p.checkpoints[5].select(0, checkpoint_idx).copy_(wavefield.m_vxx_t);
    p.checkpoints[6].select(0, checkpoint_idx).copy_(wavefield.m_vxz_t);
    p.checkpoints[7].select(0, checkpoint_idx).copy_(wavefield.m_vzx_t);
    p.checkpoints[8].select(0, checkpoint_idx).copy_(wavefield.m_vzz_t);
    p.checkpoints[9].select(0, checkpoint_idx).copy_(wavefield.m_sxxx_t);
    p.checkpoints[10].select(0, checkpoint_idx).copy_(wavefield.m_sxxz_t);
    p.checkpoints[11].select(0, checkpoint_idx).copy_(wavefield.m_szzx_t);
    p.checkpoints[12].select(0, checkpoint_idx).copy_(wavefield.m_szzz_t);
    p.checkpoints[13].select(0, checkpoint_idx).copy_(wavefield.m_sxzx_t);
    p.checkpoints[14].select(0, checkpoint_idx).copy_(wavefield.m_sxzz_t);
}

} // namespace

ForwardOutput forward(const ForwardInput& in)
{

    const auto& p = in;
    ForwardOutput out;

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    // parse model parameters
    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    ElasticWavefieldTensor wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields, true);
    else
        wavefield.allocate(vp, 2);
    auto wf = wavefield.view();

    auto mu  = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto record = torch::zeros({nrec_fields, B, nrec, p.nt}, vp.options());

    if (p.use_checkpoint) {
        TORCH_CHECK(p.checkpoints.size() == 15, "Elastic 2D checkpointing expects 15 checkpoint tensors");
        if (p.use_recursive_checkpoint) {
            TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps");
            TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D");
        } else {
            TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
        }
    }

    torch::Tensor u_allt;
    if (p.save_all_wavefields) u_allt = torch::zeros({p.nt, 2, B, nz, nx}, vp.options()); // Only save Vx and Vz.

    SolverContext solver{2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface, p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(), dx, 0.f, dz};
    
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu;
    if (staged_boundary)
        boundary_saver.allocate(p.use_boundary_saving, 2, 5, solver, vp, save_width, 1, true, false, p.transfer_interval, p.boundary_cpu, p.boundary_gpu, p.last_two, false, p.use_pinned_memory);
    else
        boundary_saver.allocate(p.use_boundary_saving, 2, 5, solver, vp, save_width, 1, true, true, 1, {}, p.boundary_gpu, p.last_two, false, p.use_pinned_memory);

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);

    const int order =
        (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    float* u_this_t = nullptr;
    int interval = p.transfer_interval;
    int buf_idx = 0;
    int gpu_idx = 0;

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    int next_ckpt_idx = 0;
    int num_checkpoint_steps = p.use_recursive_checkpoint ? static_cast<int>(p.checkpoint_steps.numel()) : 0;
    const int* checkpoint_steps = p.use_recursive_checkpoint ? p.checkpoint_steps.data_ptr<int>() : nullptr;

    for (unsigned int it = 0; it < p.nt; ++it) {

        buf_idx = it % interval;
        u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        LAUNCH_ELASTIC_VELOCITY(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            rho.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        ); // t+0.5
        
        LAUNCH_ELASTIC_STRESS(
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
            float* field = elastic_field_ptr(wf, 2, source_fields[isrc].item<int>());
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
                save_checkpoint_state_2d(p, wavefield, ckpt_idx);
            }
        }

        if (p.use_boundary_saving) {

            float* fields[5] = {
                wf.vx,
                wf.vz,
                wf.sxx,
                wf.szz,
                wf.sxz
            };

            for (int f = 0; f < 5; ++f) {
                float* top_ptr = nullptr;
                float* bottom_ptr = nullptr;
                float* left_ptr = nullptr;
                float* right_ptr = nullptr;

                if (staged_boundary) {
                    gpu_idx = f * interval + buf_idx;
                    top_ptr = boundary_saver.top_gpu.data_ptr<float>() + gpu_idx * boundary_saver.top_stride;
                    bottom_ptr = boundary_saver.bottom_gpu.data_ptr<float>() + gpu_idx * boundary_saver.bottom_stride;
                    left_ptr = boundary_saver.left_gpu.data_ptr<float>() + gpu_idx * boundary_saver.left_stride;
                    right_ptr = boundary_saver.right_gpu.data_ptr<float>() + gpu_idx * boundary_saver.right_stride;
                } else {
                    top_ptr = boundary_saver.top_t[f].data_ptr<float>();
                    bottom_ptr = boundary_saver.bottom_t[f].data_ptr<float>();
                    left_ptr = boundary_saver.left_t[f].data_ptr<float>();
                    right_ptr = boundary_saver.right_t[f].data_ptr<float>();
                }

                boundary_kernel2d<<<launch_config.grid, launch_config.block>>>(
                    fields[f],
                    top_ptr,
                    bottom_ptr,
                    left_ptr,
                    right_ptr,
                    staged_boundary ? 0 : it,
                    save_width,
                    -p.M, // offset
                    solver,
                    BOUNDARY_SAVE
                );
            }

            if (staged_boundary && (buf_idx == interval - 1 || it == p.nt - 1)) {
                int start = it - buf_idx;
                int len = buf_idx + 1;

                async_copy.record_compute_ready();
                async_copy.wait_for_compute();
                boundary_saver.flush_gpu_to_cpu(start, len, async_copy.copy_stream);
            }
        }
        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = elastic_field_ptr(wf, 2, receiver_fields[irec].item<int>());
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
    }

    if (p.use_boundary_saving) {
        boundary_saver.last_two_t.select(0,0).select(0,0).copy_(wavefield.vx_t);
        boundary_saver.last_two_t.select(0,1).select(0,0).copy_(wavefield.vz_t);
        boundary_saver.last_two_t.select(0,2).select(0,0).copy_(wavefield.sxx_t);
        boundary_saver.last_two_t.select(0,3).select(0,0).copy_(wavefield.szz_t);
        boundary_saver.last_two_t.select(0,4).select(0,0).copy_(wavefield.sxz_t);
    }

    async_copy.synchronize_copy();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;

}

}
