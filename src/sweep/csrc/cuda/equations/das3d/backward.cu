#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

#include "das3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace das3d {

namespace {

torch::Tensor recompute_strain_history(const BackwardInput& p)
{
    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    auto mu = rho * vs * vs;
    auto lambda = rho * (vp * vp - 2 * vs * vs);
    c10::cuda::CUDAGuard device_guard(vp.device());

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;

    int forward_nsrc = p.forward_sources_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);

    DasWavefieldTensor3D wavefield;
    wavefield.allocate(vp);
    auto wf = wavefield.view();

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto tmp_sxx_x = torch::zeros_like(vp);
    auto tmp_syy_y = torch::zeros_like(vp);
    auto tmp_szz_z = torch::zeros_like(vp);
    auto tmp_txx_y = torch::zeros_like(vp);
    auto tmp_txx_z = torch::zeros_like(vp);
    auto tmp_tyy_x = torch::zeros_like(vp);
    auto tmp_tyy_z = torch::zeros_like(vp);
    auto tmp_tzz_x = torch::zeros_like(vp);
    auto tmp_tzz_y = torch::zeros_like(vp);
    auto history = torch::zeros({p.nt, 3, B, nz, ny, nx}, vp.options());

    SolverContext solver{
        3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, dy, dz
    };
    SGradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(forward_nsrc, B);
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    for (unsigned int it = 0; it < p.nt; ++it) {
        tmp_sxx_x.zero_();
        tmp_syy_y.zero_();
        tmp_szz_z.zero_();
        tmp_txx_y.zero_();
        tmp_txx_z.zero_();
        tmp_tyy_x.zero_();
        tmp_tyy_z.zero_();
        tmp_tzz_x.zero_();
        tmp_tzz_y.zero_();

        LAUNCH_DAS3D_FIRST(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            tmp_sxx_x.data_ptr<float>(),
            tmp_syy_y.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_y.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tyy_x.data_ptr<float>(),
            tmp_tyy_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            tmp_tzz_y.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_DAS3D_SECOND(
            order,
            launch_config.grid,
            launch_config.block,
            wf,
            tmp_sxx_x.data_ptr<float>(),
            tmp_syy_y.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_y.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tyy_x.data_ptr<float>(),
            tmp_tyy_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            tmp_tzz_y.data_ptr<float>(),
            rho.data_ptr<float>(),
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            grad_ctx,
            cpml_view,
            solver
        );

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = das3d_field_ptr(wf, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }

        auto history_t = history.select(0, it);
        history_t.select(0, 0).copy_(wavefield.exx_t.view({B, nz, ny, nx}));
        history_t.select(0, 1).copy_(wavefield.eyy_t.view({B, nz, ny, nx}));
        history_t.select(0, 2).copy_(wavefield.ezz_t.view({B, nz, ny, nx}));
    }

    return history;
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.u_forward.defined(), "DAS 3D full backward requires saved exx/eyy/ezz wavefields.");
    TORCH_CHECK(p.u_forward.dim() == 6, "DAS 3D saved wavefields must have shape (nt, 3, B, nz, ny, nx).");
    TORCH_CHECK(p.u_forward.size(0) == p.nt, "DAS 3D saved wavefield time dimension does not match nt.");
    TORCH_CHECK(p.u_forward.size(1) == 3, "DAS 3D full backward saves only exx/eyy/ezz histories.");

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    c10::cuda::CUDAGuard device_guard(vp.device());

    float dx = p.spacing[0];
    float dy = p.spacing[1];
    float dz = p.spacing[2];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int ny = vp.size(3);
    int nx = vp.size(4);
    int B = N * C;

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, dy, dz
    };
    SGradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    DasWavefieldTensor3D adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(vp);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    auto grad_vp = torch::zeros_like(vp);
    auto grad_vs = torch::zeros_like(vp);
    auto grad_rho = torch::zeros_like(vp);

    auto zero_exx = torch::zeros_like(vp);
    auto zero_eyy = torch::zeros_like(vp);
    auto zero_ezz = torch::zeros_like(vp);

    auto q_dxx_sxx = torch::zeros_like(vp);
    auto q_dyy_syy = torch::zeros_like(vp);
    auto q_dzz_szz = torch::zeros_like(vp);
    auto q_dyy_txx = torch::zeros_like(vp);
    auto q_dzz_txx = torch::zeros_like(vp);
    auto q_dxx_tyy = torch::zeros_like(vp);
    auto q_dzz_tyy = torch::zeros_like(vp);
    auto q_dxx_tzz = torch::zeros_like(vp);
    auto q_dyy_tzz = torch::zeros_like(vp);

    auto bar_sxx_x = torch::zeros_like(vp);
    auto bar_syy_y = torch::zeros_like(vp);
    auto bar_szz_z = torch::zeros_like(vp);
    auto bar_txx_y = torch::zeros_like(vp);
    auto bar_txx_z = torch::zeros_like(vp);
    auto bar_tyy_x = torch::zeros_like(vp);
    auto bar_tyy_z = torch::zeros_like(vp);
    auto bar_tzz_x = torch::zeros_like(vp);
    auto bar_tzz_y = torch::zeros_like(vp);

    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das3d_field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        q_dxx_sxx.zero_();
        q_dyy_syy.zero_();
        q_dzz_szz.zero_();
        q_dyy_txx.zero_();
        q_dzz_txx.zero_();
        q_dxx_tyy.zero_();
        q_dzz_tyy.zero_();
        q_dxx_tzz.zero_();
        q_dyy_tzz.zero_();
        bar_sxx_x.zero_();
        bar_syy_y.zero_();
        bar_szz_z.zero_();
        bar_txx_y.zero_();
        bar_txx_z.zero_();
        bar_tyy_x.zero_();
        bar_tyy_z.zero_();
        bar_tzz_x.zero_();
        bar_tzz_y.zero_();

        const float* exx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* eyy_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* ezz_now = p.u_forward.select(0, it).select(0, 2).data_ptr<float>();
        const float* exx_prev = (it > 0)
            ? p.u_forward.select(0, it - 1).select(0, 0).data_ptr<float>()
            : zero_exx.data_ptr<float>();
        const float* eyy_prev = (it > 0)
            ? p.u_forward.select(0, it - 1).select(0, 1).data_ptr<float>()
            : zero_eyy.data_ptr<float>();
        const float* ezz_prev = (it > 0)
            ? p.u_forward.select(0, it - 1).select(0, 2).data_ptr<float>()
            : zero_ezz.data_ptr<float>();

        LAUNCH_DAS3D_PROJECT_MODEL_GRAD(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            exx_now,
            eyy_now,
            ezz_now,
            exx_prev,
            eyy_prev,
            ezz_prev,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            q_dxx_sxx.data_ptr<float>(),
            q_dyy_syy.data_ptr<float>(),
            q_dzz_szz.data_ptr<float>(),
            q_dyy_txx.data_ptr<float>(),
            q_dzz_txx.data_ptr<float>(),
            q_dxx_tyy.data_ptr<float>(),
            q_dzz_tyy.data_ptr<float>(),
            q_dxx_tzz.data_ptr<float>(),
            q_dyy_tzz.data_ptr<float>(),
            solver
        );

        if (it == 0) {
            continue;
        }

#define DAS3D_SECOND_ADJOINT(direction, q, bar, memory)                       \
        LAUNCH_DAS3D_SECOND_ADJOINT(                                          \
            order,                                                            \
            direction,                                                        \
            launch_config.grid,                                               \
            launch_config.block,                                              \
            adj_view,                                                         \
            q.data_ptr<float>(),                                              \
            bar.data_ptr<float>(),                                            \
            memory,                                                           \
            grad_ctx,                                                         \
            cpml_view,                                                        \
            solver                                                            \
        )

        DAS3D_SECOND_ADJOINT(X, q_dxx_sxx, bar_sxx_x, adj_view.m_sxx_xb);
        DAS3D_SECOND_ADJOINT(Y, q_dyy_syy, bar_syy_y, adj_view.m_syy_yb);
        DAS3D_SECOND_ADJOINT(Z, q_dzz_szz, bar_szz_z, adj_view.m_szz_zb);
        DAS3D_SECOND_ADJOINT(Y, q_dyy_txx, bar_txx_y, adj_view.m_txx_yb);
        DAS3D_SECOND_ADJOINT(Z, q_dzz_txx, bar_txx_z, adj_view.m_txx_zb);
        DAS3D_SECOND_ADJOINT(X, q_dxx_tyy, bar_tyy_x, adj_view.m_tyy_xb);
        DAS3D_SECOND_ADJOINT(Z, q_dzz_tyy, bar_tyy_z, adj_view.m_tyy_zb);
        DAS3D_SECOND_ADJOINT(X, q_dxx_tzz, bar_tzz_x, adj_view.m_tzz_xb);
        DAS3D_SECOND_ADJOINT(Y, q_dyy_tzz, bar_tzz_y, adj_view.m_tzz_yb);

#undef DAS3D_SECOND_ADJOINT

#define DAS3D_FIRST_ADJOINT(direction, bar, memory, field)                    \
        LAUNCH_DAS3D_FIRST_ADJOINT(                                           \
            order,                                                            \
            direction,                                                        \
            launch_config.grid,                                               \
            launch_config.block,                                              \
            adj_view,                                                         \
            bar.data_ptr<float>(),                                            \
            memory,                                                           \
            field,                                                            \
            grad_ctx,                                                         \
            cpml_view,                                                        \
            solver                                                            \
        )

        DAS3D_FIRST_ADJOINT(X, bar_sxx_x, adj_view.m_sxx_xf, adj_view.sxx);
        DAS3D_FIRST_ADJOINT(Y, bar_syy_y, adj_view.m_syy_yf, adj_view.syy);
        DAS3D_FIRST_ADJOINT(Z, bar_szz_z, adj_view.m_szz_zf, adj_view.szz);
        DAS3D_FIRST_ADJOINT(Y, bar_txx_y, adj_view.m_txx_yf, adj_view.txx);
        DAS3D_FIRST_ADJOINT(Z, bar_txx_z, adj_view.m_txx_zf, adj_view.txx);
        DAS3D_FIRST_ADJOINT(X, bar_tyy_x, adj_view.m_tyy_xf, adj_view.tyy);
        DAS3D_FIRST_ADJOINT(Z, bar_tyy_z, adj_view.m_tyy_zf, adj_view.tyy);
        DAS3D_FIRST_ADJOINT(X, bar_tzz_x, adj_view.m_tzz_xf, adj_view.tzz);
        DAS3D_FIRST_ADJOINT(Y, bar_tzz_y, adj_view.m_tzz_yf, adj_view.tzz);

#undef DAS3D_FIRST_ADJOINT
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    BackwardInput replay = in;
    replay.u_forward = recompute_strain_history(in);
    return backward(replay);
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    BackwardInput replay = in;
    replay.u_forward = recompute_strain_history(in);
    return backward(replay);
}

BackwardOutput backward_recursive_ckpt(const BackwardInput& in)
{
    BackwardInput replay = in;
    replay.u_forward = recompute_strain_history(in);
    return backward(replay);
}

}
