#include <torch/extension.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "das2d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace das2d {

ForwardOutput forward(const ForwardInput& in)
{
    const auto& p = in;
    ForwardOutput out;

    TORCH_CHECK(
        !p.save_all_wavefields && !p.use_boundary_saving && !p.use_checkpoint,
        "DAS 2D CUDA currently supports forward inference only; eager mode remains the reference for gradients."
    );

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

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(nsrc, B);
    auto record_config = fdtd::Geom::make(nrec, B);
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

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

    out.wavefield = torch::Tensor();
    out.last_two = torch::empty({0}, vp.options());
    out.record = record;
    return out;
}

}
