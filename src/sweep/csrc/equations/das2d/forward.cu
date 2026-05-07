#include <torch/extension.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "das2d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace das2d {

ForwardOutput forward(const ForwardInput& in)
{
    const auto& p = in;
    ForwardOutput out;

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    auto mu = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);
    c10::cuda::CUDAGuard device_guard(vp.device());

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    DasWavefieldTensor2D wavefield;
    if (!p.wavefields.empty())
        wavefield.bind(p.wavefields);
    else
        wavefield.allocate(vp);
    auto wf = wavefield.view();

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

    auto tmp_sxx_x = torch::zeros_like(vp);
    auto tmp_szz_z = torch::zeros_like(vp);
    auto tmp_txx_z = torch::zeros_like(vp);
    auto tmp_tzz_x = torch::zeros_like(vp);
    torch::Tensor u_allt;
    if (p.save_all_wavefields) {
        u_allt = torch::zeros({p.nt, 2, B, nz, nx}, vp.options());
    }

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            p.use_boundary_saving,
            2,
            9,
            solver,
            vp,
            save_width,
            1,
            true,
            false,
            p.transfer_interval,
            p.boundary_cpu,
            p.boundary_gpu,
            p.last_two,
            p.use_pinned_memory
        );
    } else {
        boundary_saver.allocate(
            p.use_boundary_saving,
            2,
            9,
            solver,
            vp,
            save_width,
            1,
            true,
            true,
            1,
            {},
            p.boundary_gpu,
            p.last_two,
            p.use_pinned_memory
        );
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

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

    for (unsigned int it = 0; it < p.nt; ++it) {
        tmp_sxx_x.zero_();
        tmp_szz_z.zero_();
        tmp_txx_z.zero_();
        tmp_tzz_x.zero_();

        LAUNCH_DAS2D_FIRST(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            tmp_sxx_x.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_DAS2D_SECOND(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            tmp_sxx_x.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            rho.data_ptr<float>(),
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = das2d_field_ptr(wf, source_fields[isrc].item<int>());
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

        if (p.use_boundary_saving) {
            float* fields[6] = {
                wf.exx,
                wf.ezz,
                wf.sxx,
                wf.szz,
                wf.txx,
                wf.tzz,
            };

            for (int f = 0; f < 6; ++f) {
                boundary_runtime.save_forward_2d_field(
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
                    f == 5
                );
            }
        }

        if (u_allt.defined()) {
            auto history_t = u_allt.select(0, it);
            history_t.select(0, 0).copy_(wavefield.exx_t.view({B, nz, nx}));
            history_t.select(0, 1).copy_(wavefield.ezz_t.view({B, nz, nx}));
        }

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das2d_field_ptr(wf, receiver_fields[irec].item<int>());
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
        boundary_saver.last_two_t.select(0, 0).select(0, 0).copy_(wavefield.exx_t);
        boundary_saver.last_two_t.select(0, 1).select(0, 0).copy_(wavefield.ezz_t);
        boundary_saver.last_two_t.select(0, 2).select(0, 0).copy_(wavefield.sxx_t);
        boundary_saver.last_two_t.select(0, 3).select(0, 0).copy_(wavefield.szz_t);
        boundary_saver.last_two_t.select(0, 4).select(0, 0).copy_(wavefield.txx_t);
        boundary_saver.last_two_t.select(0, 5).select(0, 0).copy_(wavefield.tzz_t);
        boundary_saver.last_two_t.select(0, 6).select(0, 0).copy_(wavefield.das35_t);
        boundary_saver.last_two_t.select(0, 7).select(0, 0).copy_(wavefield.das54x_t);
        boundary_saver.last_two_t.select(0, 8).select(0, 0).copy_(wavefield.das54z_t);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;
    return out;
}

}
