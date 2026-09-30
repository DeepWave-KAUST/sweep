#include <cuda_runtime.h>


#include "das3d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../launch/config.h"

namespace das3d {

// p.grads_out as the propagator binds it: {grad_vp, grad_vs, grad_rho}, in
// BackwardOutputCore.grads order, zeroed per backward on the Python side and
// accumulated here.  Mandatory: propagator/_c.py Wrapper.backward always sets
// `params.grads_out = _gradient_buffers(...)`, one slot per model
// (cuda_layout.grads_out_has_wavelet is False here, so no wavelet slot).
static BufList grad_slots(const BackwardInputCore& p)
{
    SWEEP_CHECK(p.grads_out.size() == 3,
                "das3d/backward requires the propagator-bound grads_out "
                "(3 tensors {grad_vp, grad_vs, grad_rho}), got ", p.grads_out.size());
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

// Exactly N_SLOTS: a pool of any other size means the Python declaration
// drifted.  Mandatory: DASZhao3D.cuda_layout.backward_workspace_nvar = 19 and
// propagator/_c.py Wrapper.backward always sets
// `params.adjoint_workspace = list(cp.adjoint_workspace)`.
BufList workspace_slots(const BackwardInputCore& p)
{
    SWEEP_CHECK(static_cast<int>(p.adjoint_workspace.size()) == N_SLOTS,
                "das3d/backward requires the propagator-bound adjoint_workspace (",
                static_cast<int>(N_SLOTS), " tensors, "
                "cuda_layout.backward_workspace_nvar), got ", p.adjoint_workspace.size());
    return p.adjoint_workspace;
}

// The checkpoint replay state the propagator binds as p.forward_wavefields
// (set 0, zeroed per backward call): one full DasWavefieldTensor3D::bind()
// list -- 9 physical + 18 CPML memory + 4 DAS projections, cuda_layout
// checkpoint_state_nvar = base_nvar + pml_nvar.  The recompute steps it from
// the quiescent zero state at it = 0.  Mandatory: the only Python-reachable
// callers are backward_ckpt / backward_recursive_ckpt, and Wrapper.backward
// binds `params.forward_wavefields = _forward_state_buffers(cp.forward_state_shapes, ...)`
// on both.  backward_bs also routes here, but DASZhao3D declares
// supports_boundary_saving_c = False (equations/das.py), so the propagator
// resolves boundary saving away (or raises on an explicit request) and that
// entry point is unreachable.
constexpr int CKPT_STATE_COUNT = 31;


Buf recompute_strain_history(const BackwardInputCore& p)
{
    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    const auto lame = derived::lame(p, vp, vs, rho, "das3d::recompute_strain_history");
    auto mu = lame.mu;
    auto lambda = lame.lambda;
    sweep::DeviceGuard device_guard(device_index_of(vp));

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
    int nsrc_fields = p.source_field_indices.size();
    const IntSpan source_fields = p.source_field_indices;

    DasWavefieldTensor3D wavefield;
    {
        const char* what = "das3d ckpt replay state";
        SWEEP_CHECK(!p.forward_wavefields.empty(),
                    "das3d/ckpt requires the propagator-bound forward_wavefields "
                    "replay state (cuda_layout base_nvar + pml_nvar slots)");
        auto state = wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT, what);
        for (int i = 0; i < CKPT_STATE_COUNT; ++i)
            pool_slot_checked(state, i, vp, what);   // every slot is model-shaped
        wavefield.bind(state);
    }
    auto wf = wavefield.view();

    ElasticCPMLTensor cpml;
    cpml.bind(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    const auto& ws = workspace_slots(p);
    auto tmp_sxx_x = pool_required(ws, REPLAY_TMP_SXX_X, vp, "adjoint_workspace");
    auto tmp_syy_y = pool_required(ws, REPLAY_TMP_SYY_Y, vp, "adjoint_workspace");
    auto tmp_szz_z = pool_required(ws, REPLAY_TMP_SZZ_Z, vp, "adjoint_workspace");
    auto tmp_txx_y = pool_required(ws, REPLAY_TMP_TXX_Y, vp, "adjoint_workspace");
    auto tmp_txx_z = pool_required(ws, REPLAY_TMP_TXX_Z, vp, "adjoint_workspace");
    auto tmp_tyy_x = pool_required(ws, REPLAY_TMP_TYY_X, vp, "adjoint_workspace");
    auto tmp_tyy_z = pool_required(ws, REPLAY_TMP_TYY_Z, vp, "adjoint_workspace");
    auto tmp_tzz_x = pool_required(ws, REPLAY_TMP_TZZ_X, vp, "adjoint_workspace");
    auto tmp_tzz_y = pool_required(ws, REPLAY_TMP_TZZ_Y, vp, "adjoint_workspace");
    // Python-allocated with the checkpoint snapshots (cuda_layout.checkpoint_replay_shapes
    // is declared, so _ensure_checkpoint_buffers always fills self.checkpoint_replay);
    // every step is written before the backward reads it.
    auto history = pool_required(p.checkpoint_replay, 0, {p.nt, 3, B, nz, ny, nx}, "checkpoint_replay");

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
        zero_tensor_device_async(tmp_sxx_x);
        zero_tensor_device_async(tmp_syy_y);
        zero_tensor_device_async(tmp_szz_z);
        zero_tensor_device_async(tmp_txx_y);
        zero_tensor_device_async(tmp_txx_z);
        zero_tensor_device_async(tmp_tyy_x);
        zero_tensor_device_async(tmp_tyy_z);
        zero_tensor_device_async(tmp_tzz_x);
        zero_tensor_device_async(tmp_tzz_y);

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
            float* field = das3d_field_ptr(wf, source_fields[isrc]);
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
        copy_tensor_cuda_async(history_t.select(0, 0), wavefield.exx_t.view({B, nz, ny, nx}));
        copy_tensor_cuda_async(history_t.select(0, 1), wavefield.eyy_t.view({B, nz, ny, nx}));
        copy_tensor_cuda_async(history_t.select(0, 2), wavefield.ezz_t.view({B, nz, ny, nx}));
    }

    return history;
}

} // namespace

BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(p.u_forward.defined(), "DAS 3D full backward requires saved exx/eyy/ezz wavefields.");
    SWEEP_CHECK(p.u_forward.dim() == 6, "DAS 3D saved wavefields must have shape (nt, 3, B, nz, ny, nx).");
    SWEEP_CHECK(p.u_forward.size(0) == p.nt, "DAS 3D saved wavefield time dimension does not match nt.");
    SWEEP_CHECK(p.u_forward.size(1) == 3, "DAS 3D full backward saves only exx/eyy/ezz histories.");

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    sweep::DeviceGuard device_guard(device_index_of(vp));

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
    int nrec_fields = p.receiver_field_indices.size();
    const IntSpan receiver_fields = p.receiver_field_indices;

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        3, nx, ny, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, dy, dz
    };
    SGradParam grad_ctx{1, nx, nx * ny, p.M, p.grad_coes.data_ptr<float>(), dx, dy, dz};

    // Mandatory: propagator/_c.py Wrapper.backward always binds
    // `params.adjoint_wavefields = [a.zero_() for a in cp.adjoint_wavefields]`.
    DasWavefieldTensor3D adjoint;
    SWEEP_CHECK(!p.adjoint_wavefields.empty(),
                "das3d/backward requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);

    ElasticCPMLTensor cpml;
    cpml.bind(p.pml_vals, 3);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave3D::make(nx, ny, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    const auto& gs = grad_slots(p);
    auto grad_vp = pool_required(gs, 0, vp, "grads_out");
    auto grad_vs = pool_required(gs, 1, vp, "grads_out");
    auto grad_rho = pool_required(gs, 2, vp, "grads_out");

    const auto& ws = workspace_slots(p);
    auto zero_strain = pool_required(ws, ZERO, vp, "adjoint_workspace");   // read only

    auto q_dxx_sxx = pool_required(ws, Q_DXX_SXX, vp, "adjoint_workspace");
    auto q_dyy_syy = pool_required(ws, Q_DYY_SYY, vp, "adjoint_workspace");
    auto q_dzz_szz = pool_required(ws, Q_DZZ_SZZ, vp, "adjoint_workspace");
    auto q_dyy_txx = pool_required(ws, Q_DYY_TXX, vp, "adjoint_workspace");
    auto q_dzz_txx = pool_required(ws, Q_DZZ_TXX, vp, "adjoint_workspace");
    auto q_dxx_tyy = pool_required(ws, Q_DXX_TYY, vp, "adjoint_workspace");
    auto q_dzz_tyy = pool_required(ws, Q_DZZ_TYY, vp, "adjoint_workspace");
    auto q_dxx_tzz = pool_required(ws, Q_DXX_TZZ, vp, "adjoint_workspace");
    auto q_dyy_tzz = pool_required(ws, Q_DYY_TZZ, vp, "adjoint_workspace");

    auto bar_sxx_x = pool_required(ws, BAR_SXX_X, vp, "adjoint_workspace");
    auto bar_syy_y = pool_required(ws, BAR_SYY_Y, vp, "adjoint_workspace");
    auto bar_szz_z = pool_required(ws, BAR_SZZ_Z, vp, "adjoint_workspace");
    auto bar_txx_y = pool_required(ws, BAR_TXX_Y, vp, "adjoint_workspace");
    auto bar_txx_z = pool_required(ws, BAR_TXX_Z, vp, "adjoint_workspace");
    auto bar_tyy_x = pool_required(ws, BAR_TYY_X, vp, "adjoint_workspace");
    auto bar_tyy_z = pool_required(ws, BAR_TYY_Z, vp, "adjoint_workspace");
    auto bar_tzz_x = pool_required(ws, BAR_TZZ_X, vp, "adjoint_workspace");
    auto bar_tzz_y = pool_required(ws, BAR_TZZ_Y, vp, "adjoint_workspace");

    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das3d_field_ptr(adj_view, receiver_fields[irec]);
            if (field == nullptr) continue;
            add_source_3d<<<source_config.grid, source_config.block>>>(
                field,
                p.adjoint_source.select(0, irec).data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        zero_tensor_device_async(q_dxx_sxx);
        zero_tensor_device_async(q_dyy_syy);
        zero_tensor_device_async(q_dzz_szz);
        zero_tensor_device_async(q_dyy_txx);
        zero_tensor_device_async(q_dzz_txx);
        zero_tensor_device_async(q_dxx_tyy);
        zero_tensor_device_async(q_dzz_tyy);
        zero_tensor_device_async(q_dxx_tzz);
        zero_tensor_device_async(q_dyy_tzz);
        zero_tensor_device_async(bar_sxx_x);
        zero_tensor_device_async(bar_syy_y);
        zero_tensor_device_async(bar_szz_z);
        zero_tensor_device_async(bar_txx_y);
        zero_tensor_device_async(bar_txx_z);
        zero_tensor_device_async(bar_tyy_x);
        zero_tensor_device_async(bar_tyy_z);
        zero_tensor_device_async(bar_tzz_x);
        zero_tensor_device_async(bar_tzz_y);

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

BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    BackwardInputCore replay = in;
    replay.u_forward = recompute_strain_history(in);
    return backward_core(replay);
}

BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    BackwardInputCore replay = in;
    replay.u_forward = recompute_strain_history(in);
    return backward_core(replay);
}

BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& in)
{
    BackwardInputCore replay = in;
    replay.u_forward = recompute_strain_history(in);
    return backward_core(replay);
}






}
