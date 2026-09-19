#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAGuard.h>

#include "das2d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/das.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace das2d {

// p.grads_out as the propagator binds it: {grad_vp, grad_vs, grad_rho}, in
// BackwardOutput.grads order, zeroed per backward on the Python side and
// accumulated here.  Mandatory: propagator/_c.py Wrapper.backward always sets
// `params.grads_out = _gradient_buffers(...)`, one slot per model
// (cuda_layout.grads_out_has_wavelet is False here, so no wavelet slot).
static const std::vector<torch::Tensor>& grad_slots(const BackwardInput& p)
{
    TORCH_CHECK(p.grads_out.size() == 3,
                "das2d/backward requires the propagator-bound grads_out "
                "(3 tensors {grad_vp, grad_vs, grad_rho}), got ", p.grads_out.size());
    return p.grads_out;
}

namespace {

// Layout of p.adjoint_workspace, declared on the Python side by
// DASZhao.cuda_layout.backward_workspace_shapes (one padded grid per shot each).
// ZERO is read only: the zero strain that stands in for the missing neighbour
// step, kept zero by the propagator's per-forward zeroing of the pool. Full and
// checkpoint modes use the first N_SLOTS_FULL; boundary saving also keeps the
// two current strains and four derivative temporaries alive through its
// adjoint loop and gets N_SLOTS_BS. The checkpoint replay
// (recompute_strain_history) finishes before the full backward touches BAR_*,
// so its four derivative temporaries alias those slots.
enum WorkspaceSlot : int {
    ZERO = 0,
    BAR_DXX_SXX, BAR_DZZ_SZZ, BAR_DZZ_TXX, BAR_DXX_TZZ,
    BAR_SXX_X, BAR_SZZ_Z, BAR_TXX_Z, BAR_TZZ_X,
    N_SLOTS_FULL,
    CURRENT_EXX = N_SLOTS_FULL, CURRENT_EZZ,
    TMP_SXX_X, TMP_SZZ_Z, TMP_TXX_Z, TMP_TZZ_X,
    N_SLOTS_BS,
    REPLAY_TMP_SXX_X = BAR_DXX_SXX, REPLAY_TMP_SZZ_Z = BAR_DZZ_SZZ,
    REPLAY_TMP_TXX_Z = BAR_DZZ_TXX, REPLAY_TMP_TZZ_X = BAR_DXX_TZZ,
};

// Exactly the count this mode declares: any other size means the Python
// declaration drifted.  Mandatory: DASZhao.cuda_layout.backward_workspace_shapes
// (_das2d_adjoint_workspace) declares 9 slots for full/ckpt/recursive and 15
// for bs, and propagator/_c.py Wrapper.backward always sets
// `params.adjoint_workspace = list(cp.adjoint_workspace)`.
const std::vector<torch::Tensor>& workspace_slots(const BackwardInput& p, int n_slots)
{
    TORCH_CHECK(static_cast<int>(p.adjoint_workspace.size()) == n_slots,
                "das2d/backward requires the propagator-bound adjoint_workspace (",
                n_slots, " tensors for this mode, "
                "cuda_layout.backward_workspace_shapes), got ", p.adjoint_workspace.size());
    return p.adjoint_workspace;
}

// The checkpoint replay state the propagator binds as p.forward_wavefields
// (set 0, zeroed per backward call): one full DasWavefieldTensor2D::bind()
// list -- 6 physical + 8 CPML memory + 3 DAS projections, cuda_layout
// checkpoint_state_nvar = base_nvar + pml_nvar.  The recompute steps it from
// the quiescent zero state at it = 0.  Mandatory: this function runs only from
// backward_ckpt / backward_recursive_ckpt, and propagator/_c.py Wrapper.backward
// binds `params.forward_wavefields = _forward_state_buffers(cp.forward_state_shapes, ...)`
// on both checkpoint paths.
constexpr int CKPT_STATE_COUNT = 17;


torch::Tensor recompute_strain_history(const BackwardInput& p)
{
    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    const auto lame = derived::lame(p, vp, vs, rho, "das2d::recompute_strain_history");
    auto mu = lame.mu;
    auto lambda = lame.lambda;
    c10::cuda::CUDAGuard device_guard(vp.device());

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    int forward_nsrc = p.forward_sources_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);

    DasWavefieldTensor2D wavefield;
    {
        const char* what = "das2d ckpt replay state";
        TORCH_CHECK(!p.forward_wavefields.empty(),
                    "das2d/ckpt requires the propagator-bound forward_wavefields "
                    "replay state (cuda_layout base_nvar + pml_nvar slots)");
        auto state = wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT, what);
        for (int i = 0; i < CKPT_STATE_COUNT; ++i)
            pool_slot_checked(state, i, vp, what);   // every slot is model-shaped
        wavefield.bind(state);
    }
    auto wf = wavefield.view();

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    const auto& ws = workspace_slots(p, N_SLOTS_FULL);
    auto tmp_sxx_x = pool_required(ws, REPLAY_TMP_SXX_X, vp, "adjoint_workspace");
    auto tmp_szz_z = pool_required(ws, REPLAY_TMP_SZZ_Z, vp, "adjoint_workspace");
    auto tmp_txx_z = pool_required(ws, REPLAY_TMP_TXX_Z, vp, "adjoint_workspace");
    auto tmp_tzz_x = pool_required(ws, REPLAY_TMP_TZZ_X, vp, "adjoint_workspace");
    // Python-allocated with the checkpoint snapshots (cuda_layout.checkpoint_replay_shapes
    // is declared, so _ensure_checkpoint_buffers always fills self.checkpoint_replay);
    // every step is written before the backward reads it.
    auto history = pool_required(p.checkpoint_replay, 0, {p.nt, 2, B, nz, nx}, vp.options(), "checkpoint_replay");

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(forward_nsrc, B);
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
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }

        auto history_t = history.select(0, it);
        history_t.select(0, 0).copy_(wavefield.exx_t.view({B, nz, nx}));
        history_t.select(0, 1).copy_(wavefield.ezz_t.view({B, nz, nx}));
    }

    return history;
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.u_forward.defined(), "DAS 2D full backward requires saved exx/ezz wavefields.");
    TORCH_CHECK(p.u_forward.dim() == 5, "DAS 2D saved wavefields must have shape (nt, 2, B, nz, nx).");
    TORCH_CHECK(p.u_forward.size(0) == p.nt, "DAS 2D saved wavefield time dimension does not match nt.");
    TORCH_CHECK(p.u_forward.size(1) == 2, "DAS 2D full backward saves only exx/ezz histories.");

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    c10::cuda::CUDAGuard device_guard(vp.device());

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    // Mandatory: propagator/_c.py Wrapper.backward always binds
    // `params.adjoint_wavefields = [a.zero_() for a in cp.adjoint_wavefields]`
    // (_ensure_wavefield_buffers allocates the adjoint pool for every
    // gradient-bearing call).
    DasWavefieldTensor2D adjoint;
    TORCH_CHECK(!p.adjoint_wavefields.empty(),
                "das2d/full requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto source_config = fdtd::Geom::make(adjoint_nsrc, B);

    const auto& gs = grad_slots(p);
    auto grad_vp = pool_required(gs, 0, vp, "grads_out");
    auto grad_vs = pool_required(gs, 1, vp, "grads_out");
    auto grad_rho = pool_required(gs, 2, vp, "grads_out");

    const auto& ws = workspace_slots(p, N_SLOTS_FULL);
    auto zero_strain = pool_required(ws, ZERO, vp, "adjoint_workspace");   // read only

    auto bar_dxx_sxx = pool_required(ws, BAR_DXX_SXX, vp, "adjoint_workspace");
    auto bar_dzz_szz = pool_required(ws, BAR_DZZ_SZZ, vp, "adjoint_workspace");
    auto bar_dzz_txx = pool_required(ws, BAR_DZZ_TXX, vp, "adjoint_workspace");
    auto bar_dxx_tzz = pool_required(ws, BAR_DXX_TZZ, vp, "adjoint_workspace");
    auto bar_sxx_x = pool_required(ws, BAR_SXX_X, vp, "adjoint_workspace");
    auto bar_szz_z = pool_required(ws, BAR_SZZ_Z, vp, "adjoint_workspace");
    auto bar_txx_z = pool_required(ws, BAR_TXX_Z, vp, "adjoint_workspace");
    auto bar_tzz_x = pool_required(ws, BAR_TZZ_X, vp, "adjoint_workspace");

    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        auto adj_view = adjoint.view();

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<source_config.grid, source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        bar_dxx_sxx.zero_();
        bar_dzz_szz.zero_();
        bar_dzz_txx.zero_();
        bar_dxx_tzz.zero_();
        bar_sxx_x.zero_();
        bar_szz_z.zero_();
        bar_txx_z.zero_();
        bar_tzz_x.zero_();

        const float* exx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* ezz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* exx_prev = (it > 0)
            ? p.u_forward.select(0, it - 1).select(0, 0).data_ptr<float>()
            : zero_strain.data_ptr<float>();
        const float* ezz_prev = (it > 0)
            ? p.u_forward.select(0, it - 1).select(0, 1).data_ptr<float>()
            : zero_strain.data_ptr<float>();

        LAUNCH_DAS2D_PROJECT_MODEL_GRAD(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            exx_now,
            ezz_now,
            exx_prev,
            ezz_prev,
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            bar_dxx_sxx.data_ptr<float>(),
            bar_dzz_szz.data_ptr<float>(),
            bar_dzz_txx.data_ptr<float>(),
            bar_dxx_tzz.data_ptr<float>(),
            solver
        );

        if (it == 0) {
            continue;
        }

        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dxx_sxx.data_ptr<float>(),
            bar_sxx_x.data_ptr<float>(),
            adj_view.m_sxx_xb,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dzz_szz.data_ptr<float>(),
            bar_szz_z.data_ptr<float>(),
            adj_view.m_szz_zb,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dzz_txx.data_ptr<float>(),
            bar_txx_z.data_ptr<float>(),
            adj_view.m_txx_zb,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dxx_tzz.data_ptr<float>(),
            bar_tzz_x.data_ptr<float>(),
            adj_view.m_tzz_xb,
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_sxx_x.data_ptr<float>(),
            adj_view.m_sxx_xf,
            adj_view.sxx,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_szz_z.data_ptr<float>(),
            adj_view.m_szz_zf,
            adj_view.szz,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_txx_z.data_ptr<float>(),
            adj_view.m_txx_zf,
            adj_view.txx,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_tzz_x.data_ptr<float>(),
            adj_view.m_tzz_xf,
            adj_view.tzz,
            grad_ctx,
            cpml_view,
            solver
        );
    }

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    auto vp = p.models[0];
    auto vs = p.models[1];
    auto rho = p.models[2];
    c10::cuda::CUDAGuard device_guard(vp.device());

    float dx = p.spacing[0];
    float dz = p.spacing[1];

    int N = vp.size(0);
    int C = vp.size(1);
    int nz = vp.size(2);
    int nx = vp.size(3);
    int B = N * C;

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    const auto lame = derived::lame(p, vp, vs, rho, "das2d::backward");
    auto mu = lame.mu;
    auto lambda = lame.lambda;

    DasWavefieldTensor2D adjoint;
    TORCH_CHECK(!p.adjoint_wavefields.empty(),
                "das2d/bs requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);

    // Reconstruction state: the propagator binds the RECON_NVAR = 9 zeroed
    // grids RECON_LIST_DESC (no CPML memory -- the NOPML reverse kernels below
    // never read or write it, so those 8 members stay undefined and view()
    // nulls them).  Mandatory: cuda_layout.bs_reconstruction_nvar = 9, and
    // propagator/_c.py Wrapper.backward binds
    // `params.forward_wavefields = _forward_state_buffers(cp.forward_state_shapes, ...)`
    // on the boundary-saving path.
    DasWavefieldTensor2D forward;
    wavefields_required(p.forward_wavefields, DasWavefieldTensor2D::RECON_NVAR, vp,
                        "das2d/bs reconstruction (cuda_layout.bs_reconstruction_nvar)");
    forward.bind_recon(p.forward_wavefields);
    TORCH_CHECK(p.u_last_two.defined(), "DAS 2D boundary-saving backward requires the final forward state.");
    TORCH_CHECK(p.u_last_two.size(0) >= 6, "DAS 2D boundary-saving last_two must contain at least 6 fields.");
    forward.exx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
    forward.ezz_t.copy_(p.u_last_two.select(0, 1).select(0, 0));
    forward.sxx_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
    forward.szz_t.copy_(p.u_last_two.select(0, 3).select(0, 0));
    forward.txx_t.copy_(p.u_last_two.select(0, 4).select(0, 0));
    forward.tzz_t.copy_(p.u_last_two.select(0, 5).select(0, 0));

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    // last_two is bound but never read here: this backward seeds its
    // reconstruction from p.u_last_two directly, and an unbound saver would
    // allocate an nvar-wavefield copy on every call (in host memory on the
    // staged path).
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            true,
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
            p.u_last_two,
            p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
    } else {
        boundary_saver.allocate(
            true,
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
            p.u_last_two,
            p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, B);

    const auto& gs = grad_slots(p);
    auto grad_vp = pool_required(gs, 0, vp, "grads_out");
    auto grad_vs = pool_required(gs, 1, vp, "grads_out");
    auto grad_rho = pool_required(gs, 2, vp, "grads_out");

    const auto& ws = workspace_slots(p, N_SLOTS_BS);
    auto zero_strain = pool_required(ws, ZERO, vp, "adjoint_workspace");   // read only
    auto current_exx = pool_required(ws, CURRENT_EXX, vp, "adjoint_workspace");
    auto current_ezz = pool_required(ws, CURRENT_EZZ, vp, "adjoint_workspace");

    auto tmp_sxx_x = pool_required(ws, TMP_SXX_X, vp, "adjoint_workspace");
    auto tmp_szz_z = pool_required(ws, TMP_SZZ_Z, vp, "adjoint_workspace");
    auto tmp_txx_z = pool_required(ws, TMP_TXX_Z, vp, "adjoint_workspace");
    auto tmp_tzz_x = pool_required(ws, TMP_TZZ_X, vp, "adjoint_workspace");

    auto bar_dxx_sxx = pool_required(ws, BAR_DXX_SXX, vp, "adjoint_workspace");
    auto bar_dzz_szz = pool_required(ws, BAR_DZZ_SZZ, vp, "adjoint_workspace");
    auto bar_dzz_txx = pool_required(ws, BAR_DZZ_TXX, vp, "adjoint_workspace");
    auto bar_dxx_tzz = pool_required(ws, BAR_DXX_TZZ, vp, "adjoint_workspace");
    auto bar_sxx_x = pool_required(ws, BAR_SXX_X, vp, "adjoint_workspace");
    auto bar_szz_z = pool_required(ws, BAR_SZZ_Z, vp, "adjoint_workspace");
    auto bar_txx_z = pool_required(ws, BAR_TXX_Z, vp, "adjoint_workspace");
    auto bar_tzz_x = pool_required(ws, BAR_TZZ_X, vp, "adjoint_workspace");

    TORCH_CHECK(nsrc_fields == 0 || p.forward_source.defined(),
                "das2d::backward_bs: forward_source must be defined when source "
                "fields are un-injected.");

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        2,
        true,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    for (int it = static_cast<int>(p.nt) - 1; it >= 1; --it) {
        auto adj_view = adjoint.view();
        auto for_view = forward.view();

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = das2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<adj_source_config.grid, adj_source_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        current_exx.copy_(forward.exx_t);
        current_ezz.copy_(forward.ezz_t);

        // Un-inject the forward source (sign -1) before time-reversing the
        // step: exact sign flip in-kernel, no negated copy of the source.
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = das2d_field_ptr(for_view, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_signed<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                -1.0f,
                solver
            );
        }

        LAUNCH_DAS2D_REVERSE_STRESS_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            lambda.data_ptr<float>(),
            mu.data_ptr<float>(),
            solver
        );

        float* stress_fields[4] = {
            for_view.sxx,
            for_view.szz,
            for_view.txx,
            for_view.tzz,
        };
        for (int f = 2; f < 6; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                stress_fields[f - 2],
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                -p.M,
                solver,
                f,
                f == 2,
                false
            );
        }

        tmp_sxx_x.zero_();
        tmp_szz_z.zero_();
        tmp_txx_z.zero_();
        tmp_tzz_x.zero_();

        LAUNCH_DAS2D_FIRST_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            tmp_sxx_x.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            grad_ctx,
            solver
        );

        LAUNCH_DAS2D_REVERSE_STRAIN_NOPML(
            order,
            launch_config.grid,
            launch_config.block,
            for_view,
            tmp_sxx_x.data_ptr<float>(),
            tmp_szz_z.data_ptr<float>(),
            tmp_txx_z.data_ptr<float>(),
            tmp_tzz_x.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        float* strain_fields[2] = {
            for_view.exx,
            for_view.ezz,
        };
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                strain_fields[f],
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                -p.M,
                solver,
                f,
                false,
                f == 1
            );
        }

        bar_dxx_sxx.zero_();
        bar_dzz_szz.zero_();
        bar_dzz_txx.zero_();
        bar_dxx_tzz.zero_();
        bar_sxx_x.zero_();
        bar_szz_z.zero_();
        bar_txx_z.zero_();
        bar_tzz_x.zero_();

        LAUNCH_DAS2D_PROJECT_MODEL_GRAD(
            order,
            launch_config.grid,
            launch_config.block,
            adj_view,
            current_exx.data_ptr<float>(),
            current_ezz.data_ptr<float>(),
            forward.exx_t.data_ptr<float>(),
            forward.ezz_t.data_ptr<float>(),
            vp.data_ptr<float>(),
            vs.data_ptr<float>(),
            rho.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_vs.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            bar_dxx_sxx.data_ptr<float>(),
            bar_dzz_szz.data_ptr<float>(),
            bar_dzz_txx.data_ptr<float>(),
            bar_dxx_tzz.data_ptr<float>(),
            solver
        );

        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dxx_sxx.data_ptr<float>(),
            bar_sxx_x.data_ptr<float>(),
            adj_view.m_sxx_xb,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dzz_szz.data_ptr<float>(),
            bar_szz_z.data_ptr<float>(),
            adj_view.m_szz_zb,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dzz_txx.data_ptr<float>(),
            bar_txx_z.data_ptr<float>(),
            adj_view.m_txx_zb,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_SECOND_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_dxx_tzz.data_ptr<float>(),
            bar_tzz_x.data_ptr<float>(),
            adj_view.m_tzz_xb,
            grad_ctx,
            cpml_view,
            solver
        );

        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_sxx_x.data_ptr<float>(),
            adj_view.m_sxx_xf,
            adj_view.sxx,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_szz_z.data_ptr<float>(),
            adj_view.m_szz_zf,
            adj_view.szz,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            Z,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_txx_z.data_ptr<float>(),
            adj_view.m_txx_zf,
            adj_view.txx,
            grad_ctx,
            cpml_view,
            solver
        );
        LAUNCH_DAS2D_FIRST_ADJOINT(
            order,
            X,
            launch_config.grid,
            launch_config.block,
            adj_view,
            bar_tzz_x.data_ptr<float>(),
            adj_view.m_tzz_xf,
            adj_view.tzz,
            grad_ctx,
            cpml_view,
            solver
        );

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    auto adj_view = adjoint.view();
    for (int irec = 0; irec < nrec_fields; ++irec) {
        float* field = das2d_field_ptr(adj_view, receiver_fields[irec].item<int>());
        if (field == nullptr) continue;
        add_source<<<adj_source_config.grid, adj_source_config.block>>>(
            field,
            p.adjoint_source[irec].data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            0,
            adjoint_nsrc,
            solver
        );
    }

    bar_dxx_sxx.zero_();
    bar_dzz_szz.zero_();
    bar_dzz_txx.zero_();
    bar_dxx_tzz.zero_();

    LAUNCH_DAS2D_PROJECT_MODEL_GRAD(
        order,
        launch_config.grid,
        launch_config.block,
        adj_view,
        forward.exx_t.data_ptr<float>(),
        forward.ezz_t.data_ptr<float>(),
        zero_strain.data_ptr<float>(),
        zero_strain.data_ptr<float>(),
        vp.data_ptr<float>(),
        vs.data_ptr<float>(),
        rho.data_ptr<float>(),
        grad_vp.data_ptr<float>(),
        grad_vs.data_ptr<float>(),
        grad_rho.data_ptr<float>(),
        bar_dxx_sxx.data_ptr<float>(),
        bar_dzz_szz.data_ptr<float>(),
        bar_dzz_txx.data_ptr<float>(),
        bar_dxx_tzz.data_ptr<float>(),
        solver
    );

    out.grads = {grad_vp, grad_vs, grad_rho};
    return out;
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

} // namespace das2d
