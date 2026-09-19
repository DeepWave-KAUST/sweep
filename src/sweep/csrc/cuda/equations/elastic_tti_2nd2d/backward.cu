#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <array>

#include "elastic_tti_2nd2d.h"
#include "kernels.cuh"
#include "tensors.h"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../launch/config.h"

namespace elastic_tti_2nd2d {

// p.grads_out as the propagator binds it: one per prepared model, in
// BackwardOutput.grads order, zeroed per backward on the Python side.
// Mandatory: propagator/_c.py Wrapper.backward always sets
// `params.grads_out = _gradient_buffers(...)`, one slot per model
// (cuda_layout.grads_out_has_wavelet is False here, so no wavelet slot).
static std::vector<torch::Tensor> model_grads(const BackwardInput& p)
{
    TORCH_CHECK(p.grads_out.size() == p.models.size(),
                "elastic_tti_2nd2d/backward requires the propagator-bound grads_out "
                "(one tensor per model, ",
                p.models.size(), "), got ", p.grads_out.size());
    for (size_t i = 0; i < p.models.size(); ++i)
        TORCH_CHECK(p.grads_out[i].sizes() == p.models[i].sizes(),
                    "grads_out[", i, "] has shape ", p.grads_out[i].sizes(),
                    " but model ", i, " is ", p.models[i].sizes());
    return p.grads_out;
}

namespace {

// Layout of p.adjoint_workspace, declared on the Python side by
// ElasticTTI2nd.cuda_layout.backward_workspace_shapes (one padded grid per shot
// each). The first N_POOL are the adjoint stress/velocity workspace bound by
// AdjointWorkspace::init. What follows depends on the mode -- the modes never
// share a pool: full mode keeps one read-only zero field (the missing history
// step), boundary saving and checkpointing keep the three stress workspaces of
// the replayed forward step.
enum WorkspaceSlot : int {
    N_POOL = 8,
    ZERO_FIELD = N_POOL,                          // full mode only, never written
    SXX_WS = N_POOL, SZZ_WS, SXZ_WS,              // bs / ckpt modes
    N_SLOTS_FULL = N_POOL + 1,
    N_SLOTS_BS = N_POOL + 3,
};

// Exactly the count this mode declares: any other size means the Python
// declaration drifted.  Mandatory:
// ElasticTTI2nd.cuda_layout.backward_workspace_shapes
// (_adjoint_workspace_shapes) declares 9 slots for full and 11 for bs/ckpt,
// and propagator/_c.py Wrapper.backward always sets
// `params.adjoint_workspace = list(cp.adjoint_workspace)`.
const std::vector<torch::Tensor>& workspace_slots(const BackwardInput& p, int n_slots)
{
    TORCH_CHECK(static_cast<int>(p.adjoint_workspace.size()) == n_slots,
                "elastic_tti_2nd2d/backward requires the propagator-bound adjoint_workspace (",
                n_slots, " tensors for this mode, "
                "cuda_layout.backward_workspace_shapes), got ", p.adjoint_workspace.size());
    return p.adjoint_workspace;
}

// The checkpoint replay state the propagator binds as p.forward_wavefields
// (set 0, zeroed per backward call): one full WavefieldTensor::bind() list --
// the (ux, uz) x (now, pre, next) displacement triple plus the 8 CPML memory
// fields, cuda_layout checkpoint_state_nvar = base_nvar + pml_nvar.  The
// checkpoint snapshots hold that same list (checkpoint_tensors() is
// state_tensors()), so it is also the CheckpointRuntime tensor count.
// Mandatory: propagator/_c.py Wrapper.backward binds
// `params.forward_wavefields = _forward_state_buffers(cp.forward_state_shapes, ...)`
// on the checkpoint path, and _forward_state_shapes("ckpt") is the forward slot
// list (base_nvar + pml_nvar = CKPT_STATE_COUNT).
constexpr int CKPT_STATE_COUNT = 14;

struct AdjointWorkspace {
    std::array<torch::Tensor, 8> t;

    // The pool is mandatory: every mode's backward_workspace_shapes declares
    // at least N_POOL slots, and workspace_slots() has already checked the
    // exact count for this mode.
    void init(const std::vector<torch::Tensor>& external)
    {
        TORCH_CHECK(external.size() >= N_POOL,
                    "elastic_tti_2nd2d/backward requires the propagator-bound "
                    "adjoint_workspace (at least ", static_cast<int>(N_POOL),
                    " tensors, cuda_layout.backward_workspace_shapes), got ",
                    external.size());
        for (int i = 0; i < 8; ++i) {
            t[i] = external[i];
            t[i].zero_();
        }
    }

    float* q(int i) { return t[i].data_ptr<float>(); }
    float* pw(int i) { return t[4 + i].data_ptr<float>(); }
};

// One reverse sweep element: K1 (bar of the CPML-corrected divergence terms
// from lam_{t+1}) must run BEFORE the adjoint advance so the q workspace and
// the adjoint memory transposition see the source-completed lam_{t+1}.
void adjoint_k1(
    int order,
    const fdtd::LaunchConfig& launch_config,
    WavefieldTensor& adjoint,
    StiffnessPointer model,
    ElasticCPMLPointer cpml_view,
    SolverContext solver,
    AdjointWorkspace& ws
)
{
    TTI2ND_LAUNCH(tti2nd_adjoint_div_prepare, order,
        launch_config.grid, launch_config.block,
        adjoint.view(), model, cpml_view, solver,
        ws.q(0), ws.q(1), ws.q(2), ws.q(3));
}

void adjoint_advance(
    int order,
    const fdtd::LaunchConfig& launch_config,
    WavefieldTensor& adjoint,
    StiffnessPointer model,
    ElasticCPMLPointer cpml_view,
    SGradParam grad_ctx,
    SolverContext solver,
    AdjointWorkspace& ws
)
{
    TTI2ND_LAUNCH(tti2nd_adjoint_strain_prepare, order,
        launch_config.grid, launch_config.block,
        adjoint.view(), model, cpml_view, grad_ctx, solver,
        ws.q(0), ws.q(1), ws.q(2), ws.q(3),
        ws.pw(0), ws.pw(1), ws.pw(2), ws.pw(3));

    TTI2ND_LAUNCH(tti2nd_adjoint_displacement_apply, order,
        launch_config.grid, launch_config.block,
        adjoint.view(),
        ws.pw(0), ws.pw(1), ws.pw(2), ws.pw(3),
        grad_ctx, solver);

    adjoint.swap_u();
}

void inject_adjoint_source(
    const BackwardInput& p,
    WavefieldTensor& adjoint,
    const torch::Tensor& receiver_fields,
    const fdtd::LaunchConfig& adj_source_config,
    SolverContext solver,
    int it
)
{
    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int nrec_fields = p.receiver_field_indices.numel();
    for (int irec = 0; irec < nrec_fields; ++irec) {
        const int fld = receiver_fields[irec].item<int>();
        float* field = nullptr;
        if (fld == 0) field = adjoint.ux_t.data_ptr<float>();
        else if (fld == 1) field = adjoint.uz_t.data_ptr<float>();
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
}

void rho_source_correction(
    const BackwardInput& p,
    WavefieldTensor& adjoint,
    const torch::Tensor& source_fields,
    torch::Tensor& grad_rho,
    const torch::Tensor& rho,
    const fdtd::LaunchConfig& fwd_source_config,
    SolverContext solver,
    int it
)
{
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();
    for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
        const int fld = source_fields[isrc].item<int>();
        float* adj_field = nullptr;
        if (fld == 0) adj_field = adjoint.ux_t.data_ptr<float>();
        else if (fld == 1) adj_field = adjoint.uz_t.data_ptr<float>();
        if (adj_field == nullptr) continue;
        tti2nd_rho_grad_source_correction<<<fwd_source_config.grid, fwd_source_config.block>>>(
            grad_rho.data_ptr<float>(),
            adj_field,
            rho.data_ptr<float>(),
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            forward_nsrc,
            solver
        );
    }
}

} // namespace

BackwardOutput backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 7, "ElasticTTI2nd backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTI2nd backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_forward.defined(), "ElasticTTI2nd full backward expects saved forward wavefields");
    TORCH_CHECK(p.u_forward.dim() == 5 && p.u_forward.size(1) == 2,
                "ElasticTTI2nd full backward expects u_forward with shape (nt, 2, B, nz, nx)");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    // Mandatory: propagator/_c.py Wrapper.backward always binds
    // `params.adjoint_wavefields = [a.zero_() for a in cp.adjoint_wavefields]`.
    WavefieldTensor adjoint;
    TORCH_CHECK(!p.adjoint_wavefields.empty(),
                "elastic_tti_2nd2d/full requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);
    for (auto& tsr : adjoint.state_tensors())
        tsr.zero_();

    auto model = stiffness_view(p.models);
    auto grads = model_grads(p);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    AdjointWorkspace ws;
    const auto& slots = workspace_slots(p, N_SLOTS_FULL);
    ws.init(slots);
    auto zero_field = pool_required(slots, ZERO_FIELD, rho, "adjoint_workspace");   // read only

    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);

    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        inject_adjoint_source(p, adjoint, receiver_fields, adj_source_config, solver, it);

        adjoint_k1(order, launch_config, adjoint, model, cpml_view, solver, ws);

        const float* ux_t = (it >= 1) ? p.u_forward.select(0, it - 1).select(0, 0).data_ptr<float>() : zero_field.data_ptr<float>();
        const float* uz_t = (it >= 1) ? p.u_forward.select(0, it - 1).select(0, 1).data_ptr<float>() : zero_field.data_ptr<float>();
        const float* ux_next = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* uz_next = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* ux_prev = (it >= 2) ? p.u_forward.select(0, it - 2).select(0, 0).data_ptr<float>() : zero_field.data_ptr<float>();
        const float* uz_prev = (it >= 2) ? p.u_forward.select(0, it - 2).select(0, 1).data_ptr<float>() : zero_field.data_ptr<float>();

        TTI2ND_LAUNCH(tti2nd_calculate_grad, order,
            launch_config.grid, launch_config.block,
            adjoint.view(), model, grad_view,
            ux_t, uz_t, ux_next, uz_next, ux_prev, uz_prev,
            ws.q(0), ws.q(1), ws.q(2), ws.q(3),
            grad_ctx, solver);

        rho_source_correction(p, adjoint, source_fields, grads[0], rho, fwd_source_config, solver, it);

        if (it == 0)
            continue;

        adjoint_advance(order, launch_config, adjoint, model, cpml_view, grad_ctx, solver, ws);
    }

    out.grads = grads;
    return out;
}

BackwardOutput backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 7, "ElasticTTI2nd boundary-saving backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTI2nd boundary-saving backward expects cpmls PML profiles");
    TORCH_CHECK(p.u_last_two.defined(), "ElasticTTI2nd boundary-saving backward expects last-two wavefield tensor");

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    WavefieldTensor adjoint;
    TORCH_CHECK(!p.adjoint_wavefields.empty(),
                "elastic_tti_2nd2d/bs requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);
    for (auto& tsr : adjoint.state_tensors())
        tsr.zero_();

    // Reconstruction state: the RECON_WF_COUNT displacement grids bound from
    // BackwardInput.forward_wavefields (Python-zeroed, no CPML memory -- the
    // nopml reverse kernels below never read it).  Mandatory:
    // cuda_layout.bs_reconstruction_nvar = 6, and propagator/_c.py
    // Wrapper.backward binds
    // `params.forward_wavefields = _forward_state_buffers(cp.forward_state_shapes, ...)`
    // on the boundary-saving path.
    WavefieldTensor forward;
    wavefields_required(p.forward_wavefields, WavefieldTensor::RECON_WF_COUNT, rho,
                        "elastic_tti_2nd2d/bs reconstruction "
                        "(cuda_layout.bs_reconstruction_nvar)");
    forward.bind_recon(p.forward_wavefields);
    // (storage, level): level 1 = W_nt goes to the pre slot (later time),
    // level 0 = W_{nt-1} becomes the current state — acoustic2d convention.
    forward.ux_pre_t.copy_(p.u_last_two.select(0, 0).select(0, 1));
    forward.ux_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
    forward.uz_pre_t.copy_(p.u_last_two.select(0, 1).select(0, 1));
    forward.uz_t.copy_(p.u_last_two.select(0, 1).select(0, 0));

    auto model = stiffness_view(p.models);
    auto grads = model_grads(p);
    auto grad_view = stiffness_grad_view(grads);

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    AdjointWorkspace ws;
    const auto& slots = workspace_slots(p, N_SLOTS_BS);
    ws.init(slots);

    auto sxx_ws = pool_required(slots, SXX_WS, rho, "adjoint_workspace");
    auto szz_ws = pool_required(slots, SZZ_WS, rho, "adjoint_workspace");
    auto sxz_ws = pool_required(slots, SXZ_WS, rho, "adjoint_workspace");

    // last_two is bound but never read here: this backward seeds its
    // reconstruction from p.u_last_two directly, and an unbound saver would
    // allocate an nvar-wavefield copy on every call (in host memory on the
    // staged path).
    EffectiveBoundarySaver boundary_saver;
    const int save_width = solver.M + 1;
    const bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            true, 2, 2, solver, rho, save_width, 2,
            true, false, p.transfer_interval,
            p.boundary_cpu, p.boundary_gpu,
            p.u_last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
    } else {
        boundary_saver.allocate(
            true, 2, 2, solver, rho, save_width, 2,
            true, true, 1,
            {}, p.boundary_gpu,
            p.u_last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging
        );
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, rho);
    }
    auto bs = boundary_saver.view();

    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();

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
        auto for_view = forward.view();

        inject_adjoint_source(p, adjoint, receiver_fields, adj_source_config, solver, it);

        adjoint_k1(order, launch_config, adjoint, model, cpml_view, solver, ws);

        // Time-reversed reconstruction: with (u_now = W_it, u_pre = W_{it+1})
        // the reversed leapfrog writes W_{it-1} - S_it into u_next; the ring
        // restore fixes the PML-adjacent band, then the source re-add
        // completes W_{it-1}.
        TTI2ND_LAUNCH(tti2nd_stress_kernel_nopml, order,
            launch_config.grid, launch_config.block,
            for_view, model, grad_ctx, solver,
            sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>());

        TTI2ND_LAUNCH(tti2nd_displacement_kernel_nopml_rev, order,
            launch_config.grid, launch_config.block,
            for_view, model,
            sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>(),
            grad_ctx, solver);

        float* rec_fields[2] = { for_view.ux_nxt, for_view.uz_nxt };
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                rec_fields[f],
                launch_config.grid,
                launch_config.block,
                bs,
                save_width,
                -p.M,
                solver,
                f,
                f == 0,
                f == 1
            );
        }

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = field_ptr(for_view, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }

        // W_t = u_now, W_{t+1} = u_pre (later time), W_{t-1} = u_next.
        TTI2ND_LAUNCH(tti2nd_calculate_grad, order,
            launch_config.grid, launch_config.block,
            adjoint.view(), model, grad_view,
            for_view.ux, for_view.uz,
            for_view.ux_pre, for_view.uz_pre,
            for_view.ux_nxt, for_view.uz_nxt,
            ws.q(0), ws.q(1), ws.q(2), ws.q(3),
            grad_ctx, solver);

        rho_source_correction(p, adjoint, source_fields, grads[0], rho, fwd_source_config, solver, it);

        adjoint_advance(order, launch_config, adjoint, model, cpml_view, grad_ctx, solver, ws);

        forward.swap_u();

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    out.grads = grads;
    return out;
}

BackwardOutput backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(p.models.size() == 7, "ElasticTTI2nd checkpoint backward expects prepared models");
    TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTI2nd checkpoint backward expects cpmls PML profiles");
    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK(static_cast<int>(p.checkpoints.size()) == CKPT_STATE_COUNT,
                "ElasticTTI2nd checkpointing expects ", CKPT_STATE_COUNT, " checkpoint tensors, got ",
                p.checkpoints.size());

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        CKPT_STATE_COUNT,
        true,
        false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "elastic_tti_2nd2d"
    );

    const auto& rho = p.models[0];
    const int N = rho.size(0);
    const int C = rho.size(1);
    const int nz = rho.size(2);
    const int nx = rho.size(3);
    const int B = N * C;

    const float dx = p.spacing[0];
    const float dz = p.spacing[1];
    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(), dx, 0.f, dz};

    auto model = stiffness_view(p.models);
    auto grads = model_grads(p);
    auto grad_view = stiffness_grad_view(grads);

    WavefieldTensor adjoint;
    TORCH_CHECK(!p.adjoint_wavefields.empty(),
                "elastic_tti_2nd2d/ckpt requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    AdjointWorkspace ws;
    const auto& slots = workspace_slots(p, N_SLOTS_BS);
    ws.init(slots);

    auto sxx_ws = pool_required(slots, SXX_WS, rho, "adjoint_workspace");
    auto szz_ws = pool_required(slots, SZZ_WS, rho, "adjoint_workspace");
    auto sxz_ws = pool_required(slots, SXZ_WS, rho, "adjoint_workspace");

    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), B);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), B);
    const int forward_nsrc = p.forward_sources_loc.size(1);
    const int nsrc_fields = p.source_field_indices.numel();

    const int chunk_size = p.checkpoint_interval;
    const int num_chunks = (static_cast<int>(p.nt) + chunk_size - 1) / chunk_size;

    // Replay state: the CKPT_STATE_COUNT model-shaped slots the propagator
    // hands over as p.forward_wavefields (set 0, zeroed per backward call).
    // Every chunk seeds all of it (zero_state / checkpoint load) before
    // stepping.
    WavefieldTensor replay;
    {
        const char* what = "elastic_tti_2nd2d ckpt replay state";
        TORCH_CHECK(!p.forward_wavefields.empty(),
                    "elastic_tti_2nd2d/ckpt requires the propagator-bound "
                    "forward_wavefields replay state (cuda_layout base_nvar + pml_nvar slots)");
        auto state = wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT, what);
        for (int i = 0; i < CKPT_STATE_COUNT; ++i)
            pool_slot_checked(state, i, rho, what);   // every slot is model-shaped
        replay.bind(state);
    }

    // seg[k] = W_{start-1+k}: two history levels + one entry per replayed
    // step, so the reverse pass below has all three time slices in-chunk.
    // Python-allocated with the checkpoint snapshots at chunk_size + 2 rows
    // (cuda_layout.checkpoint_replay_shapes is declared, so
    // _ensure_checkpoint_buffers always fills self.checkpoint_replay with the
    // two slots); each chunk views the prefix it uses.  Rows 0..seg_len+1 are
    // all written before the reverse pass reads them, so the buffer is never
    // re-zeroed between chunks.
    std::vector<int64_t> seg_shape = rho.sizes().vec();
    seg_shape.insert(seg_shape.begin(), static_cast<int64_t>(chunk_size + 2));
    auto seg_ux_full = pool_required(p.checkpoint_replay, 0, seg_shape, rho.options(), "checkpoint_replay");
    auto seg_uz_full = pool_required(p.checkpoint_replay, 1, seg_shape, rho.options(), "checkpoint_replay");

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        const int start = chunk_id * chunk_size;
        const int end = std::min(static_cast<int>(p.nt), start + chunk_size);
        const int seg_len = end - start;

        // Seed the chunk-start state straight into the replay struct: the
        // snapshot holds the full state list in role order, and
        // state_tensors() lists replay's members by role whatever swap_u()
        // rotated into them -- the same bytes the old load-into-a-scratch-
        // state-then-copy_state produced, one state copy fewer.
        if (chunk_id == 0)
            checkpoint_runtime.zero_state(replay.state_tensors());
        else
            checkpoint_runtime.load(chunk_id, replay.checkpoint_tensors());

        auto seg_ux = seg_ux_full.narrow(0, 0, seg_len + 2);
        auto seg_uz = seg_uz_full.narrow(0, 0, seg_len + 2);
        seg_ux.select(0, 0).copy_(replay.ux_pre_t);
        seg_uz.select(0, 0).copy_(replay.uz_pre_t);
        seg_ux.select(0, 1).copy_(replay.ux_t);
        seg_uz.select(0, 1).copy_(replay.uz_t);

        for (int it = start; it < end; ++it) {
            auto rep_view = replay.view();

            TTI2ND_LAUNCH(tti2nd_stress_kernel, order,
                launch_config.grid, launch_config.block,
                rep_view, model, cpml_view, grad_ctx, solver,
                sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>());

            TTI2ND_LAUNCH(tti2nd_displacement_kernel, order,
                launch_config.grid, launch_config.block,
                rep_view, model,
                sxx_ws.data_ptr<float>(), szz_ws.data_ptr<float>(), sxz_ws.data_ptr<float>(),
                nullptr, cpml_view, grad_ctx, solver);

            for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
                float* field = field_ptr(rep_view, source_fields[isrc].item<int>());
                if (field == nullptr) continue;
                add_source<<<fwd_source_config.grid, fwd_source_config.block>>>(
                    field,
                    p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it,
                    forward_nsrc,
                    solver
                );
            }

            seg_ux.select(0, it - start + 2).copy_(replay.ux_nxt_t);
            seg_uz.select(0, it - start + 2).copy_(replay.uz_nxt_t);

            replay.swap_u();
        }

        for (int it = end - 1; it >= start; --it) {
            inject_adjoint_source(p, adjoint, receiver_fields, adj_source_config, solver, it);

            adjoint_k1(order, launch_config, adjoint, model, cpml_view, solver, ws);

            const int k = it - start;
            TTI2ND_LAUNCH(tti2nd_calculate_grad, order,
                launch_config.grid, launch_config.block,
                adjoint.view(), model, grad_view,
                seg_ux.select(0, k + 1).data_ptr<float>(),
                seg_uz.select(0, k + 1).data_ptr<float>(),
                seg_ux.select(0, k + 2).data_ptr<float>(),
                seg_uz.select(0, k + 2).data_ptr<float>(),
                seg_ux.select(0, k).data_ptr<float>(),
                seg_uz.select(0, k).data_ptr<float>(),
                ws.q(0), ws.q(1), ws.q(2), ws.q(3),
                grad_ctx, solver);

            rho_source_correction(p, adjoint, source_fields, grads[0], rho, fwd_source_config, solver, it);

            if (it == 0)
                continue;

            adjoint_advance(order, launch_config, adjoint, model, cpml_view, grad_ctx, solver, ws);
        }
    }

    out.grads = grads;
    return out;
}

} // namespace elastic_tti_2nd2d
