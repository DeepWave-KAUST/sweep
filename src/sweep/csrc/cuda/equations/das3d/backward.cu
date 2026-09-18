#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

#include "das3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace das3d {

// p.grads_out as the propagator binds it: {grad_vp, grad_vs, grad_rho}, in
// BackwardOutput.grads order (zeroed per backward on the Python side and
// accumulated here), or empty for an unbound caller, which then gets fresh
// zeros per slot.
static const std::vector<torch::Tensor>& grad_slots(const BackwardInput& p)
{
    TORCH_CHECK(p.grads_out.empty() || p.grads_out.size() == 3,
                "DAS3D backward: grads_out must be empty or hold 3 tensors ({grad_vp, grad_vs, grad_rho}), got ", p.grads_out.size());
    return p.grads_out;
}

namespace {

// Layout of p.adjoint_workspace, declared on the Python side as
// DASZhao3D.cuda_layout.backward_workspace_nvar (one padded grid per shot each).
// ZERO is read only: the zero strain that stands in for the missing neighbour
// step, kept zero by the propagator's per-forward zeroing of the pool. The
// checkpoint replay (recompute_strain_history) finishes before the backward
// touches Q_*, so its nine derivative temporaries alias those slots.
enum WorkspaceSlot : int {
    ZERO = 0,
    Q_DXX_SXX, Q_DYY_SYY, Q_DZZ_SZZ, Q_DYY_TXX, Q_DZZ_TXX, Q_DXX_TYY, Q_DZZ_TYY, Q_DXX_TZZ, Q_DYY_TZZ,
    BAR_SXX_X, BAR_SYY_Y, BAR_SZZ_Z, BAR_TXX_Y, BAR_TXX_Z, BAR_TYY_X, BAR_TYY_Z, BAR_TZZ_X, BAR_TZZ_Y,
    N_SLOTS,
    REPLAY_TMP_SXX_X = Q_DXX_SXX, REPLAY_TMP_SYY_Y = Q_DYY_SYY, REPLAY_TMP_SZZ_Z = Q_DZZ_SZZ, REPLAY_TMP_TXX_Y = Q_DYY_TXX, REPLAY_TMP_TXX_Z = Q_DZZ_TXX, REPLAY_TMP_TYY_X = Q_DXX_TYY, REPLAY_TMP_TYY_Z = Q_DZZ_TYY, REPLAY_TMP_TZZ_X = Q_DXX_TZZ, REPLAY_TMP_TZZ_Y = Q_DYY_TZZ,
};

// Either unbound (every slot then falls back to a fresh zero tensor) or exactly
// N_SLOTS: a pool of any other size means the Python declaration drifted.
const std::vector<torch::Tensor>& workspace_slots(const BackwardInput& p)
{
    TORCH_CHECK(p.adjoint_workspace.empty() || p.adjoint_workspace.size() == N_SLOTS,
                "DAS3D backward: adjoint_workspace must be empty or hold ",
                static_cast<int>(N_SLOTS), " tensors, got ", p.adjoint_workspace.size());
    return p.adjoint_workspace;
}


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

    const auto& ws = workspace_slots(p);
    auto tmp_sxx_x = pool_or_zeros(ws, REPLAY_TMP_SXX_X, vp);
    auto tmp_syy_y = pool_or_zeros(ws, REPLAY_TMP_SYY_Y, vp);
    auto tmp_szz_z = pool_or_zeros(ws, REPLAY_TMP_SZZ_Z, vp);
    auto tmp_txx_y = pool_or_zeros(ws, REPLAY_TMP_TXX_Y, vp);
    auto tmp_txx_z = pool_or_zeros(ws, REPLAY_TMP_TXX_Z, vp);
    auto tmp_tyy_x = pool_or_zeros(ws, REPLAY_TMP_TYY_X, vp);
    auto tmp_tyy_z = pool_or_zeros(ws, REPLAY_TMP_TYY_Z, vp);
    auto tmp_tzz_x = pool_or_zeros(ws, REPLAY_TMP_TZZ_X, vp);
    auto tmp_tzz_y = pool_or_zeros(ws, REPLAY_TMP_TZZ_Y, vp);
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

    const auto& gs = grad_slots(p);
    auto grad_vp = pool_or_zeros(gs, 0, vp, "grads_out");
    auto grad_vs = pool_or_zeros(gs, 1, vp, "grads_out");
    auto grad_rho = pool_or_zeros(gs, 2, vp, "grads_out");

    const auto& ws = workspace_slots(p);
    auto zero_strain = pool_or_zeros(ws, ZERO, vp);   // read only

    auto q_dxx_sxx = pool_or_zeros(ws, Q_DXX_SXX, vp);
    auto q_dyy_syy = pool_or_zeros(ws, Q_DYY_SYY, vp);
    auto q_dzz_szz = pool_or_zeros(ws, Q_DZZ_SZZ, vp);
    auto q_dyy_txx = pool_or_zeros(ws, Q_DYY_TXX, vp);
    auto q_dzz_txx = pool_or_zeros(ws, Q_DZZ_TXX, vp);
    auto q_dxx_tyy = pool_or_zeros(ws, Q_DXX_TYY, vp);
    auto q_dzz_tyy = pool_or_zeros(ws, Q_DZZ_TYY, vp);
    auto q_dxx_tzz = pool_or_zeros(ws, Q_DXX_TZZ, vp);
    auto q_dyy_tzz = pool_or_zeros(ws, Q_DYY_TZZ, vp);

    auto bar_sxx_x = pool_or_zeros(ws, BAR_SXX_X, vp);
    auto bar_syy_y = pool_or_zeros(ws, BAR_SYY_Y, vp);
    auto bar_szz_z = pool_or_zeros(ws, BAR_SZZ_Z, vp);
    auto bar_txx_y = pool_or_zeros(ws, BAR_TXX_Y, vp);
    auto bar_txx_z = pool_or_zeros(ws, BAR_TXX_Z, vp);
    auto bar_tyy_x = pool_or_zeros(ws, BAR_TYY_X, vp);
    auto bar_tyy_z = pool_or_zeros(ws, BAR_TYY_Z, vp);
    auto bar_tzz_x = pool_or_zeros(ws, BAR_TZZ_X, vp);
    auto bar_tzz_y = pool_or_zeros(ws, BAR_TZZ_Y, vp);

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
            : zero_strain.data_ptr<float>();
        const float* eyy_prev = (it > 0)
            ? p.u_forward.select(0, it - 1).select(0, 1).data_ptr<float>()
            : zero_strain.data_ptr<float>();
        const float* ezz_prev = (it > 0)
            ? p.u_forward.select(0, it - 1).select(0, 2).data_ptr<float>()
            : zero_strain.data_ptr<float>();

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
