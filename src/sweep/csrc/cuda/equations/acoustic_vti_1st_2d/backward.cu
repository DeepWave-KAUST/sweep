// ---------------------------------------------------------------------------
// AcousticVTI1st (Duveneck 2008) — 2D first-order CUDA backward.
//
// Phase 1 scope:
//   - Implements `backward()` (full mode) using the saved forward wavefield
//     (u_forward shape: (nt, 4, B, nz, nx) containing vx, vz, sH, sV).
//   - Adjoint PML memory variables are NOT tracked → interior gradients are
//     correct; gradients inside the PML band are approximate.  This matches
//     standard FWI/RTM usage where the PML region is masked off.
//   - backward_bs reconstructs the forward state from u_last_two with the
//     NOPML kernels; its 4-tensor reconstruction list [vx, vz, sH, sV]
//     (cuda_layout.bs_reconstruction_nvar) is bound from
//     BackwardInputCore.forward_wavefields, or allocated here when unbound.
//   - backward_ckpt is chunked replay: its CKPT_STATE_COUNT-slot replay
//     state (the forward slot list: 4 physical + 4 CPML memory) is bound
//     from BackwardInputCore.forward_wavefields set 0 (cuda_layout
//     checkpoint_state_nvar), or allocated here when unbound.
//     backward_recursive_ckpt is a stub.
//
// Gradient ordering follows AcousticVTI1st.MODEL_SPECS: [vp, ε, δ, ρ].
//
// Adjoint derivation lives in kernels.cuh; see the long comment block above
// `adjoint_step_kernel` and `calculate_grad_kernel`.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>


#include "acoustic_vti_1st_2d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../launch/config.h"

namespace acoustic_vti_1st_2d {

namespace {

enum GradSlot : int { GRAD_VP = 0, GRAD_EPS, GRAD_DELTA, GRAD_RHO, N_GRADS };

// p.grads_out as the propagator binds it: {grad_vp, grad_eps, grad_delta, grad_rho}, in
// BackwardOutputCore.grads order, zeroed per backward on the Python side and
// accumulated here.  Mandatory: propagator/_c.py Wrapper.backward always sets
// `params.grads_out = _gradient_buffers(...)`, one slot per model
// (cuda_layout.grads_out_has_wavelet is False here, so no wavelet slot).
static BufList grad_slots(const BackwardInputCore& p)
{
    SWEEP_CHECK(static_cast<int>(p.grads_out.size()) == N_GRADS,
                "acoustic_vti_1st_2d/backward requires the propagator-bound grads_out (",
                static_cast<int>(N_GRADS), " tensors {grad_vp, grad_eps, grad_delta, grad_rho}), got ",
                p.grads_out.size());
    return p.grads_out;
}

// Layout of p.adjoint_workspace, declared on the Python side as
// AcousticVTI1st.cuda_layout.backward_workspace_nvar (one padded grid per shot
// each). ZERO_PREV is read only: it stands in for the "previous stress" at the
// first reverse step and relies on the propagator zeroing the pool before every
// gradient-bearing forward.
enum WorkspaceSlot : int { SCRATCH_X = 0, SCRATCH_Z, ZERO_PREV, SEED_SH, SEED_SV, N_SLOTS };

// Exactly N_SLOTS: a pool of any other size means the Python declaration
// drifted.  Mandatory: AcousticVTI1st.cuda_layout.backward_workspace_nvar = 5
// in every mode, and propagator/_c.py Wrapper.backward always sets
// `params.adjoint_workspace = list(cp.adjoint_workspace)`.
BufList workspace_slots(const BackwardInputCore& p)
{
    SWEEP_CHECK(static_cast<int>(p.adjoint_workspace.size()) == N_SLOTS,
                "acoustic_vti_1st_2d/backward requires the propagator-bound adjoint_workspace (",
                static_cast<int>(N_SLOTS), " tensors, cuda_layout.backward_workspace_nvar), got ",
                p.adjoint_workspace.size());
    return p.adjoint_workspace;
}

// Same wavefield-tensor helper as forward.cu, lives in an anonymous namespace
// so the linker sees it once per TU only.
//
// Two bindings: bind() takes the full 8-slot list (4 physical + 4 CPML
// memory, AcousticVTI1st.FIELD_SPECS order -- the adjoint state, and the
// CKPT_STATE_COUNT-slot checkpoint replay state); bind_physical() takes
// the RECON_WF_COUNT-slot boundary-saving reconstruction list
// (cuda_layout.bs_reconstruction_nvar) and leaves the m_* members undefined
// -- the bs reverse loop steps the reconstruction with the NOPML kernels
// only, which never read or write CPML memory, so view() hands those slots
// out as nullptr (ptr_or_null).
struct AdjWavefieldTensor {
    static constexpr int RECON_WF_COUNT = 4;
    static constexpr const char* RECON_LIST_DESC = "[vx, vz, sH, sV]";
    // The checkpoint replay state the propagator binds: one full bind() list
    // (cuda_layout checkpoint_state_nvar = base_nvar + pml_nvar).
    static constexpr int CKPT_STATE_COUNT = 8;

    Buf vx_t, vz_t, sH_t, sV_t;
    Buf m_sHx_t, m_sVz_t, m_vxx_t, m_vzz_t;

    // torch spelling: the drivers still hand over the input struct's tensor
    // lists; the descriptors are what the struct keeps.

    void bind(const std::vector<Buf>& tensors)
    {
        SWEEP_CHECK(tensors.size() == 8,
                    "AcousticVTI1st2D backward: expects 8 adjoint wavefield tensors, got ",
                    tensors.size());
        vx_t    = tensors[0];
        vz_t    = tensors[1];
        sH_t    = tensors[2];
        sV_t    = tensors[3];
        m_sHx_t = tensors[4];
        m_sVz_t = tensors[5];
        m_vxx_t = tensors[6];
        m_vzz_t = tensors[7];
    }

    // Boundary-saving reconstruction: the physical prefix only, in
    // RECON_LIST_DESC order; m_sHx_t / m_sVz_t / m_vxx_t / m_vzz_t stay
    // undefined.
    // torch spelling: the drivers still hand over the input struct's tensor
    // lists; the descriptors are what the struct keeps.

    void bind_physical(const std::vector<Buf>& tensors)
    {
        SWEEP_CHECK(static_cast<int>(tensors.size()) == RECON_WF_COUNT,
                    "AcousticVTI1st2D backward_bs: reconstruction list must hold ",
                    RECON_WF_COUNT, " tensors ", RECON_LIST_DESC, ", got ",
                    tensors.size());
        vx_t = tensors[0];
        vz_t = tensors[1];
        sH_t = tensors[2];
        sV_t = tensors[3];
    }

    // No allocate(): every state this struct carries -- the adjoint, the bs
    // reconstruction and the checkpoint replay -- is bound by the propagator.

    VTIWavefieldPointer view() const
    {
        VTIWavefieldPointer p{};
        p.vx    = vx_t.data_ptr<float>();
        p.vz    = vz_t.data_ptr<float>();
        p.sH    = sH_t.data_ptr<float>();
        p.sV    = sV_t.data_ptr<float>();
        // nullptr after bind_physical(): only the NOPML kernels see such a
        // view and they never touch CPML memory.
        p.m_sHx = ptr_or_null(m_sHx_t);
        p.m_sVz = ptr_or_null(m_sVz_t);
        p.m_vxx = ptr_or_null(m_vxx_t);
        p.m_vzz = ptr_or_null(m_vzz_t);
        return p;
    }

    // For CheckpointRuntime save/load: the 8 tensors saved as one "state".
    std::vector<Buf> state_tensors() const
    {
        return {vx_t, vz_t, sH_t, sV_t,
                m_sHx_t, m_sVz_t, m_vxx_t, m_vzz_t};
    }

    void zero_state() const
    {
        zero_tensor_device_async(vx_t);   zero_tensor_device_async(vz_t);
        zero_tensor_device_async(sH_t);   zero_tensor_device_async(sV_t);
        zero_tensor_device_async(m_sHx_t); zero_tensor_device_async(m_sVz_t);
        zero_tensor_device_async(m_vxx_t); zero_tensor_device_async(m_vzz_t);
    }
};

// Adjoint half-step A (stress -> velocity), with the stiffness multiplied in at
// the stress location before the derivative.  `sx` / `sz` are scratch buffers of
// one wavefield component each; they are reused by half-step B.
void vti_adjoint_A_2d(
    int order,
    const fdtd::LaunchConfig& lc,
    VTIWavefieldPointer adj_view,
    const Buf& c11, const Buf& c33, const Buf& c13,
    Buf& sx, Buf& sz,
    const SGradParam& grad_ctx, const SolverContext& solver)
{
    adjoint_premultiply_stress_kernel<0><<<lc.grid, lc.block>>>(
        adj_view, c11.data_ptr<float>(), c33.data_ptr<float>(),
        c13.data_ptr<float>(), sx.data_ptr<float>(), sz.data_ptr<float>(), solver);
    LAUNCH_VTI_ADJOINT_STRESS_TO_VEL(
        order, lc.grid, lc.block,
        adj_view, sx.data_ptr<float>(), sz.data_ptr<float>(), grad_ctx, solver);
}

// Adjoint half-step B (velocity -> stress), with inv_rho multiplied in at the
// velocity location before the derivative.
void vti_adjoint_B_2d(
    int order,
    const fdtd::LaunchConfig& lc,
    VTIWavefieldPointer adj_view,
    const Buf& inv_rho,
    Buf& sx, Buf& sz,
    const SGradParam& grad_ctx, const SolverContext& solver)
{
    adjoint_premultiply_vel_kernel<0><<<lc.grid, lc.block>>>(
        adj_view, inv_rho.data_ptr<float>(),
        sx.data_ptr<float>(), sz.data_ptr<float>(), solver);
    LAUNCH_VTI_ADJOINT_VEL_TO_STRESS(
        order, lc.grid, lc.block,
        adj_view, sx.data_ptr<float>(), sz.data_ptr<float>(), grad_ctx, solver);
}

}  // anonymous namespace


BackwardOutputCore backward_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(!p.free_surface,
                "AcousticVTI1st2D CUDA backward: free_surface=True not supported "
                "(Robertsson 1996 / Mittet 2002 anisotropic FS — follow-up).");

    SWEEP_CHECK(p.spacing.size() >= 2,
                "AcousticVTI1st2D backward: spacing must have length >= 2");
    float dz = p.spacing[0];
    float dx = p.spacing[1];

    SWEEP_CHECK(p.models.size() == 4,
                "AcousticVTI1st2D backward expects 4 model tensors "
                "[vp, epsilon, delta, rho]; got ", p.models.size());
    auto vp_t      = p.models[0];
    auto epsilon_t = p.models[1];
    auto delta_t   = p.models[2];
    auto rho_t     = p.models[3];

    int N  = vp_t.size(0);
    int C  = vp_t.size(1);
    int nz = vp_t.size(2);
    int nx = vp_t.size(3);
    int B  = N * C;

    // Cached stiffness (matches forward.cu)
    const auto stiff = derived::vti_stiffness(p, vp_t, epsilon_t, delta_t, rho_t,
                                              "acoustic_vti_1st_2d::backward");
    auto c11_t   = stiff.c11;
    auto c33_t   = stiff.c33;
    auto c13_t   = stiff.c13;
    auto inv_rho_t = stiff.inv_rho;

    SWEEP_CHECK(p.u_forward.defined(),
                "AcousticVTI1st2D backward (full mode) requires the forward to "
                "be run with save_all_wavefields=True so u_forward is populated.");
    SWEEP_CHECK(p.u_forward.dim() == 5,
                "u_forward must be (nt, n_comp, B, nz, nx); got dim ",
                p.u_forward.dim());
    SWEEP_CHECK(p.u_forward.size(1) == 4,
                "u_forward second dim must be 4 (vx, vz, sH, sV); got ",
                p.u_forward.size(1));

    // Bind the adjoint wavefield (zero-initialised on the Python side).
    // Mandatory: propagator/_c.py Wrapper.backward always binds
    // `params.adjoint_wavefields = [a.zero_() for a in cp.adjoint_wavefields]`.
    AdjWavefieldTensor adjoint;
    SWEEP_CHECK(!p.adjoint_wavefields.empty(),
                "acoustic_vti_1st_2d/full requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);
    auto adj_view = adjoint.view();

    // Output gradient tensors (one per model)
    const auto& gs  = grad_slots(p);
    auto grad_vp    = pool_required(gs, GRAD_VP, vp_t, "grads_out");
    auto grad_eps   = pool_required(gs, GRAD_EPS, epsilon_t, "grads_out");
    auto grad_delta = pool_required(gs, GRAD_DELTA, delta_t, "grads_out");
    auto grad_rho   = pool_required(gs, GRAD_RHO, rho_t, "grads_out");

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(),
                        dx, 0.f, dz};

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);

    // Scratch for the adjoint pre-multiplications (see kernels.cuh).  One
    // wavefield component each; half-steps A and B reuse the same pair.
    const auto& ws = workspace_slots(p);
    auto scratch_x = pool_required(ws, SCRATCH_X, vp_t, "adjoint_workspace");
    auto scratch_z = pool_required(ws, SCRATCH_Z, vp_t, "adjoint_workspace");

    // Zero buffer the size of one wavefield component, used as the "previous"
    // stress at iter `it = 0` (no prior step exists; initial state is zero).
    // ZERO_PREV is never written, and the pool is zeroed before every
    // gradient-bearing forward, so it is still zero here.
    auto zero_state = pool_required(ws, ZERO_PREV, vp_t, "adjoint_workspace");

    int adjoint_nsrc = p.adjoint_sources_loc.defined()
                       ? p.adjoint_sources_loc.size(1) : 0;
    int nrec_fields = p.receiver_field_indices.size();
    const IntSpan receiver_fields = p.receiver_field_indices;
    auto adjoint_src_config = (adjoint_nsrc > 0)
        ? fdtd::Geom::make(adjoint_nsrc, B) : fdtd::Geom::make(1, B);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    // -------- backward time loop --------
    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        // 1. Inject adjoint source at receivers (residual back-projected)
        for (int irec = 0; irec < nrec_fields; ++irec) {
            int field_index = receiver_fields[irec];
            float* field = vti_field_ptr(adj_view, field_index);
            if (field == nullptr) continue;
            add_source<<<adjoint_src_config.grid, adjoint_src_config.block>>>(
                field,
                p.adjoint_source.select(0, irec).data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        // 2. Accumulate gradients using forward state at the right times:
        //    - vx_now, vz_now from u_forward.select(0, it) for the stiffness gradients
        //      (vx_new in step it's stress sub-step == u_forward.select(0, it).vx)
        //    - sH_prev, sV_prev from u_forward.select(0, it-1) for the inv_rho gradient
        //      (sH_old in step it's velocity sub-step == u_forward.select(0, it-1).sH)
        //    For it == 0, u_forward.select(0, -1) is "the initial state" = zeros.
        const float* fvx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        const float* fvz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        const float* fsH_prev;
        const float* fsV_prev;
        if (it > 0) {
            fsH_prev = p.u_forward.select(0, it - 1).select(0, 2).data_ptr<float>();
            fsV_prev = p.u_forward.select(0, it - 1).select(0, 3).data_ptr<float>();
        } else {
            fsH_prev = zero_state.data_ptr<float>();
            fsV_prev = zero_state.data_ptr<float>();
        }

        // EXPERIMENT: image between the two adjoint half-steps.
        //
        // The forward does velocity-then-stress, so the adjoint must do
        // stress-to-vel (kernel A) then vel-to-stress (kernel B).  The inv_rho
        // gradient at step `it` needs the COMPLETE lambda_v(it), which only
        // exists after kernel A applies this step's ds(it)/dv(it) coupling --
        // imaging before A leaves that term out.  Kernel A writes lambda_vx /
        // lambda_vz and only READS lambda_sH / lambda_sV, so moving the imaging
        // between A and B hands the inv_rho term its correct operand while
        // leaving the stiffness term's operand bit-identical.
        if (it > 0) {
            vti_adjoint_A_2d(order, launch_config, adj_view,
                             c11_t, c33_t, c13_t, scratch_x, scratch_z,
                             grad_ctx, solver);
        }

        LAUNCH_VTI_CALC_GRAD(
            order,
            launch_config.grid,
            launch_config.block,
            fvx_now, fvz_now, fsH_prev, fsV_prev,
            adj_view,
            vp_t.data_ptr<float>(),
            epsilon_t.data_ptr<float>(),
            delta_t.data_ptr<float>(),
            rho_t.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_eps.data_ptr<float>(),
            grad_delta.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx,
            solver
        );

        if (it > 0) {
            vti_adjoint_B_2d(order, launch_config, adj_view, inv_rho_t,
                             scratch_x, scratch_z, grad_ctx, solver);
        }
    }

    out.grads = {grad_vp, grad_eps, grad_delta, grad_rho};
    return out;
}


// ---------------------------------------------------------------------------
// Boundary-saving backward (Phase 2)
//
// FWI-grade memory mode: the forward saved boundary cells (PML + halo band)
// at every time step.  Here we:
//   1. Initialise the forward state from `last_two` (the saved final state).
//   2. For each backward iter from nt-1 down to 1:
//      a. Inject adjoint source from receiver residual.
//      b. Time-reverse the forward STRESS step (subtract dt) so that
//         the stress fields go from state_{it+1} back to state_{it}.
//      c. Restore stress boundary cells from the saved boundary.
//      d. Compute gradient using the forward state at this step.
//      e. Propagate the adjoint state one step back in time (same as Phase 1).
//      f. Time-reverse the forward VELOCITY step.
//      g. Restore velocity boundary cells.
// The wavelet source is also subtracted (add_source_signed, sign -1) on the
// forward side so that the source-injection contribution is removed when
// stepping back; no negated copy of the source is built.
// ---------------------------------------------------------------------------
BackwardOutputCore backward_bs_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(!p.free_surface,
                "AcousticVTI1st2D backward_bs: free_surface=True not supported "
                "(Robertsson 1996 / Mittet 2002 anisotropic FS — follow-up).");

    SWEEP_CHECK(p.spacing.size() >= 2,
                "AcousticVTI1st2D backward_bs: spacing length >= 2 required.");
    float dz = p.spacing[0];
    float dx = p.spacing[1];

    SWEEP_CHECK(p.models.size() == 4,
                "AcousticVTI1st2D backward_bs expects 4 models "
                "[vp, eps, delta, rho]; got ", p.models.size());
    auto vp_t      = p.models[0];
    auto epsilon_t = p.models[1];
    auto delta_t   = p.models[2];
    auto rho_t     = p.models[3];

    int N  = vp_t.size(0);
    int C  = vp_t.size(1);
    int nz = vp_t.size(2);
    int nx = vp_t.size(3);
    int B  = N * C;

    const auto stiff = derived::vti_stiffness(p, vp_t, epsilon_t, delta_t, rho_t,
                                              "acoustic_vti_1st_2d::backward_bs");
    auto c11_t   = stiff.c11;
    auto c33_t   = stiff.c33;
    auto c13_t   = stiff.c13;
    auto inv_rho_t = stiff.inv_rho;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(),
                        dx, 0.f, dz};

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    // Adjoint wavefield (Phase 1-style: 4 physical + 4 CPML memory).
    AdjWavefieldTensor adjoint;
    SWEEP_CHECK(!p.adjoint_wavefields.empty(),
                "acoustic_vti_1st_2d/bs requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);
    auto adj_view = adjoint.view();

    // Reconstructed forward state: the RECON_WF_COUNT physical fields the
    // propagator hands over as p.forward_wavefields (zeroed, model-shaped).
    // Only the NOPML kernels step this state, so it carries no CPML memory.
    // Mandatory: cuda_layout.bs_reconstruction_nvar = 4, and
    // propagator/_c.py Wrapper.backward binds
    // `params.forward_wavefields = _forward_state_buffers(cp.forward_state_shapes, ...)`
    // on the boundary-saving path.
    AdjWavefieldTensor forward;
    wavefields_required(p.forward_wavefields, AdjWavefieldTensor::RECON_WF_COUNT, vp_t,
                        "acoustic_vti_1st_2d/bs reconstruction "
                        "(cuda_layout.bs_reconstruction_nvar)");
    forward.bind_physical(p.forward_wavefields);
    auto for_view = forward.view();

    // Initialise forward state from u_last_two (final state at t=nt-1, the
    // post-step / post-source-inject snapshot of step nt-1).
    SWEEP_CHECK(p.u_last_two.defined(),
                "AcousticVTI1st2D backward_bs requires p.u_last_two from the "
                "boundary-saving forward.");
    copy_tensor_cuda_async(forward.vx_t, p.u_last_two.select(0, 0).select(0, 0));
    copy_tensor_cuda_async(forward.vz_t, p.u_last_two.select(0, 1).select(0, 0));
    copy_tensor_cuda_async(forward.sH_t, p.u_last_two.select(0, 2).select(0, 0));
    copy_tensor_cuda_async(forward.sV_t, p.u_last_two.select(0, 3).select(0, 0));

    // Gradient outputs.
    const auto& gs  = grad_slots(p);
    auto grad_vp    = pool_required(gs, GRAD_VP, vp_t, "grads_out");
    auto grad_eps   = pool_required(gs, GRAD_EPS, epsilon_t, "grads_out");
    auto grad_delta = pool_required(gs, GRAD_DELTA, delta_t, "grads_out");
    auto grad_rho   = pool_required(gs, GRAD_RHO, rho_t, "grads_out");

    // CPML coefficients (cpmls 8-tuple) for the adjoint propagation step.
    ElasticCPMLTensor cpml;
    cpml.bind(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    // Boundary saver / runtime — mirror forward.cu's allocation but in
    // backward mode (data flows from saver → forward state, not the other
    // way around).
    // last_two is bound but never read here: this backward seeds its
    // reconstruction from p.u_last_two directly, and an unbound saver would
    // allocate an nvar-wavefield copy on every call (in host memory on the
    // staged path).
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            /*use_bs=*/true, /*dim=*/2, /*nvar=*/4, solver, vp_t,
            save_width, /*last_two_nvar=*/1, /*override_storage=*/true,
            /*store_on_gpu_override=*/false, p.transfer_interval,
            p.boundary_cpu, p.boundary_gpu, /*last_two=*/p.u_last_two,
            p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
    } else {
        boundary_saver.allocate(
            /*use_bs=*/true, /*dim=*/2, /*nvar=*/4, solver, vp_t,
            save_width, /*last_two_nvar=*/1, /*override_storage=*/true,
            /*store_on_gpu_override=*/true, /*transfer_interval=*/1,
            /*boundary_cpu=*/{}, p.boundary_gpu, /*last_two=*/p.u_last_two,
            p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
        // If the propagator handed us an in-memory boundary tensor list
        // (legacy path), load it now.
        if (p.boundary_gpu.empty() && !p.u_boundary.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp_t);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);

    // Scratch for the adjoint pre-multiplications (see kernels.cuh).  One
    // wavefield component each; half-steps A and B reuse the same pair.
    const auto& ws = workspace_slots(p);
    auto scratch_x = pool_required(ws, SCRATCH_X, vp_t, "adjoint_workspace");
    auto scratch_z = pool_required(ws, SCRATCH_Z, vp_t, "adjoint_workspace");
    int adjoint_nsrc = p.adjoint_sources_loc.defined()
                       ? p.adjoint_sources_loc.size(1) : 0;
    int forward_nsrc = p.forward_sources_loc.defined()
                       ? p.forward_sources_loc.size(1) : 0;
    int nsrc_fields = p.source_field_indices.size();
    int nrec_fields = p.receiver_field_indices.size();
    const IntSpan source_fields = p.source_field_indices;
    const IntSpan receiver_fields = p.receiver_field_indices;
    auto adj_src_config = (adjoint_nsrc > 0)
        ? fdtd::Geom::make(adjoint_nsrc, B) : fdtd::Geom::make(1, B);
    auto fwd_src_config = (forward_nsrc > 0)
        ? fdtd::Geom::make(forward_nsrc, B) : fdtd::Geom::make(1, B);
    SWEEP_CHECK(nsrc_fields == 0 || p.forward_source.defined(),
                "AcousticVTI1st2D backward_bs: forward_source must be defined "
                "when source fields are un-injected.");

    AsyncCopyContext async_copy(staged_boundary);
    const std::vector<std::string> disk_files = p.boundary_disk_files.vec();   // the runtime keeps a pointer to it
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        /*dim=*/2,
        /*use_bs=*/true,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
    boundary_runtime.prefetch_initial_backward_chunk(p.nt);

    for (int it = static_cast<int>(p.nt) - 1; it >= 1; --it) {
        // (a) adjoint source injection (receiver-side residual)
        for (int irec = 0; irec < nrec_fields; ++irec) {
            int field_index = receiver_fields[irec];
            float* field = vti_field_ptr(adj_view, field_index);
            if (field == nullptr) continue;
            add_source<<<adj_src_config.grid, adj_src_config.block>>>(
                field, p.adjoint_source.select(0, irec).data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it, adjoint_nsrc, solver);
        }

        // (b-i) forward source subtraction (add_source_signed, sign -1) —
        //       removes the source contribution that was added during the
        //       forward step we are about to time-reverse. The in-kernel
        //       sign flip is exact; no negated copy of the source is built.
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            int field_index = source_fields[isrc];
            float* field = vti_field_ptr(for_view, field_index);
            if (field == nullptr) continue;
            add_source_signed<<<fwd_src_config.grid, fwd_src_config.block>>>(
                field, p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it, forward_nsrc, -1.0f, solver);
        }

        // (b-ii) time-reverse the STRESS update (uses NEW velocity which
        //        still lives in for_view from the previous iter's end)
        LAUNCH_VTI_STRESS_NOPML(
            order, launch_config.grid, launch_config.block,
            for_view,
            c11_t.data_ptr<float>(),
            c33_t.data_ptr<float>(),
            c13_t.data_ptr<float>(),
            grad_ctx, solver);

        // (c) restore stress boundaries (sH, sV are field indices 2 and 3)
        float* stress_fields[2] = { for_view.sH, for_view.sV };
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, stress_fields[f],
                launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver,
                /*field_idx=*/2 + f,
                /*is_first=*/(f == 0), /*is_last=*/false);
        }

        // (d) gradient accumulation — same logic as Phase 1 full mode, but
        //     fsH_prev / fsV_prev are now obtained from THIS iter's
        //     reconstructed forward state (which represents state_{it-1}
        //     after the time-reverse stress step) plus the velocity that
        //     hasn't been reversed yet (= u_forward.select(0, it).vx for the
        //     stiffness path).
        //
        //     Concretely:
        //       - fvx_now / fvz_now : for_view.vx / for_view.vz (state at
        //         END of step it, before we time-reverse the velocity step)
        //       - fsH_prev / fsV_prev : approximated as for_view.sH / sV
        //         AFTER stress reverse but BEFORE velocity reverse.  This
        //         matches u_forward.select(0, it-1).sH used in the full-mode kernel.
        // (e) half-step A first: the inv_rho gradient needs the COMPLETE
        //     lambda_v(it), which only exists once this step's ds/dv coupling
        //     has been applied.  A writes lambda_v and only reads lambda_s, so
        //     imaging between A and B leaves the stiffness operand untouched.
        vti_adjoint_A_2d(order, launch_config, adj_view,
                         c11_t, c33_t, c13_t, scratch_x, scratch_z,
                         grad_ctx, solver);

        LAUNCH_VTI_CALC_GRAD(
            order, launch_config.grid, launch_config.block,
            for_view.vx, for_view.vz, for_view.sH, for_view.sV,
            adj_view,
            vp_t.data_ptr<float>(),
            epsilon_t.data_ptr<float>(),
            delta_t.data_ptr<float>(),
            rho_t.data_ptr<float>(),
            grad_vp.data_ptr<float>(),
            grad_eps.data_ptr<float>(),
            grad_delta.data_ptr<float>(),
            grad_rho.data_ptr<float>(),
            grad_ctx, solver);

        vti_adjoint_B_2d(order, launch_config, adj_view, inv_rho_t,
                         scratch_x, scratch_z, grad_ctx, solver);

        // (f) time-reverse the VELOCITY update — turns vx_new into vx_old
        LAUNCH_VTI_VELOCITY_NOPML(
            order, launch_config.grid, launch_config.block,
            for_view,
            inv_rho_t.data_ptr<float>(),
            grad_ctx, solver);

        // (g) restore velocity boundaries (vx, vz are field indices 0 and 1)
        float* vel_fields[2] = { for_view.vx, for_view.vz };
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, vel_fields[f],
                launch_config.grid, launch_config.block,
                bs, save_width, -p.M, solver,
                /*field_idx=*/f,
                /*is_first=*/false, /*is_last=*/(f == 1));
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, p.nt);
    }

    boundary_runtime.synchronize();

    out.grads = {grad_vp, grad_eps, grad_delta, grad_rho};
    return out;
}
// ---------------------------------------------------------------------------
// Chunked-checkpoint backward (Phase 2)
//
// Forward saved a checkpoint every `checkpoint_interval` steps.  Backward
// walks chunks from the last back to the first; within each chunk we
// 1. Load the chunk-start state from the saved checkpoint (or zero for chunk 0).
// 2. Re-run the forward PML kernels chunk_size steps, capturing the full
//    wavefield in a per-chunk buffer (`u_chunk`).
// 3. Run an inner "full-mode" backward pass over that chunk using `u_chunk`
//    plus the adjoint state carried over from the previous chunk.
//
// Memory cost: O(num_chunks) checkpoints + O(chunk_size) inner replay buffer
// — significantly smaller than save_all_wavefields when nt is large.
// ---------------------------------------------------------------------------
BackwardOutputCore backward_ckpt_core(const BackwardInputCore& in)
{
    sweep::DeviceGuard device_guard(device_index_of(in.models[0]));
    const auto& p = in;
    BackwardOutputCore out;

    SWEEP_CHECK(!p.free_surface,
                "AcousticVTI1st2D backward_ckpt: free_surface=True not supported.");
    SWEEP_CHECK(p.checkpoint_interval >= 1,
                "checkpoint_interval must be >= 1");
    SWEEP_CHECK(p.checkpoints.size() == 8,
                "AcousticVTI1st2D backward_ckpt expects 8 checkpoint tensors "
                "(4 physical + 4 PML memory); got ", p.checkpoints.size());
    SWEEP_CHECK(p.spacing.size() >= 2,
                "spacing length >= 2 required.");

    float dz = p.spacing[0];
    float dx = p.spacing[1];

    auto vp_t      = p.models[0];
    auto epsilon_t = p.models[1];
    auto delta_t   = p.models[2];
    auto rho_t     = p.models[3];

    int N  = vp_t.size(0);
    int C  = vp_t.size(1);
    int nz = vp_t.size(2);
    int nx = vp_t.size(3);
    int B  = N * C;

    const auto stiff = derived::vti_stiffness(p, vp_t, epsilon_t, delta_t, rho_t,
                                              "acoustic_vti_1st_2d::backward_ckpt");
    auto c11_t   = stiff.c11;
    auto c33_t   = stiff.c33;
    auto c13_t   = stiff.c13;
    auto inv_rho_t = stiff.inv_rho;

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };
    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(),
                        dx, 0.f, dz};

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    // Adjoint wavefield (carried across chunks; cleared once at the start).
    AdjWavefieldTensor adjoint;
    SWEEP_CHECK(!p.adjoint_wavefields.empty(),
                "acoustic_vti_1st_2d/ckpt requires the propagator-bound adjoint_wavefields "
                "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
    adjoint.bind(p.adjoint_wavefields);
    auto adj_view = adjoint.view();
    adjoint.zero_state();

    // Forward-state buffer used during each chunk's replay: the
    // CKPT_STATE_COUNT model-shaped slots the propagator hands over as
    // p.forward_wavefields (set 0 of the replay state, zeroed per backward
    // call).  Every chunk re-seeds all of it (zero_state / checkpoint load)
    // before stepping, so nothing depends on the initial zeros.  Mandatory:
    // propagator/_c.py Wrapper.backward binds
    // `params.forward_wavefields = _forward_state_buffers(cp.forward_state_shapes, ...)`
    // on the checkpoint path, and _forward_state_shapes("ckpt") is the forward
    // slot list (base_nvar + pml_nvar = CKPT_STATE_COUNT).
    AdjWavefieldTensor fwd_state;
    {
        const char* what = "acoustic_vti_1st_2d ckpt replay state";
        SWEEP_CHECK(!p.forward_wavefields.empty(),
                    "acoustic_vti_1st_2d/ckpt requires the propagator-bound "
                    "forward_wavefields replay state (cuda_layout base_nvar + pml_nvar slots)");
        auto state = wavefield_set(p.forward_wavefields, 0, AdjWavefieldTensor::CKPT_STATE_COUNT, what);
        for (int i = 0; i < AdjWavefieldTensor::CKPT_STATE_COUNT; ++i)
            pool_slot_checked(state, i, vp_t, what);   // every slot is model-shaped
        fwd_state.bind(state);
    }
    auto fwd_view = fwd_state.view();

    // CheckpointRuntime — load checkpoints saved by the forward.
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints,
        /*expected_tensors=*/8,
        /*enabled=*/true,
        /*recursive=*/false,
        p.checkpoint_interval,
        p.checkpoint_steps,
        p.checkpoint_on_cpu,
        "backward_chunk",
        "acoustic_vti_1st_2d"
    );

    ElasticCPMLTensor cpml;
    cpml.bind(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    const auto& gs  = grad_slots(p);
    auto grad_vp    = pool_required(gs, GRAD_VP, vp_t, "grads_out");
    auto grad_eps   = pool_required(gs, GRAD_EPS, epsilon_t, "grads_out");
    auto grad_delta = pool_required(gs, GRAD_DELTA, delta_t, "grads_out");
    auto grad_rho   = pool_required(gs, GRAD_RHO, rho_t, "grads_out");

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);

    // Scratch for the adjoint pre-multiplications (see kernels.cuh).  One
    // wavefield component each; half-steps A and B reuse the same pair.
    const auto& ws = workspace_slots(p);
    auto scratch_x = pool_required(ws, SCRATCH_X, vp_t, "adjoint_workspace");
    auto scratch_z = pool_required(ws, SCRATCH_Z, vp_t, "adjoint_workspace");

    int adjoint_nsrc = p.adjoint_sources_loc.defined()
                       ? p.adjoint_sources_loc.size(1) : 0;
    int forward_nsrc = p.forward_sources_loc.defined()
                       ? p.forward_sources_loc.size(1) : 0;
    int nsrc_fields = p.source_field_indices.size();
    int nrec_fields = p.receiver_field_indices.size();
    const IntSpan source_fields = p.source_field_indices;
    const IntSpan receiver_fields = p.receiver_field_indices;
    auto adj_src_config = (adjoint_nsrc > 0)
        ? fdtd::Geom::make(adjoint_nsrc, B) : fdtd::Geom::make(1, B);
    auto fwd_src_config = (forward_nsrc > 0)
        ? fdtd::Geom::make(forward_nsrc, B) : fdtd::Geom::make(1, B);

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (static_cast<int>(p.nt) + chunk_size - 1) / chunk_size;

    // Per-chunk replay buffer: shape (chunk_size, 4, B, nz, nx) — only the
    // 4 physical fields are needed for the gradient kernel.
    // Python-allocated with the checkpoint snapshots (cuda_layout.checkpoint_replay_shapes
    // is declared, so _ensure_checkpoint_buffers always fills self.checkpoint_replay);
    // every row is written by the replay before the reverse pass reads it.
    auto u_chunk = pool_required(p.checkpoint_replay, 0, {chunk_size, 4, B, nz, nx}, "checkpoint_replay");

    // zero_prev is read-only (ZERO_PREV stays zero, see backward()); the two seed
    // buffers are overwritten at every chunk boundary before they are read.
    auto zero_prev = pool_required(ws, ZERO_PREV, vp_t, "adjoint_workspace");
    auto seed_sH   = pool_required(ws, SEED_SH, vp_t, "adjoint_workspace");
    auto seed_sV   = pool_required(ws, SEED_SV, vp_t, "adjoint_workspace");

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);
        int local_len = end - start;

        // (1) Initialise chunk-start forward state.
        if (chunk_id == 0) {
            fwd_state.zero_state();
        } else {
            checkpoint_runtime.load(chunk_id, fwd_state.state_tensors());
        }

        // Snapshot the seed state's stress: this IS the forward state at step
        // start-1, which the inv_rho gradient needs at the chunk's first
        // backward step.  The old code substituted zero there and called it an
        // O(chunk_size) bias; it showed up as a rho-only error in ckpt_chunk
        // once the operator adjoint was fixed.
        {
            const size_t nbytes = (size_t)B * nz * nx * sizeof(float);
            cudaMemcpyAsync(seed_sH.data_ptr<float>(), fwd_view.sH, nbytes,
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(seed_sV.data_ptr<float>(), fwd_view.sV, nbytes,
                            cudaMemcpyDeviceToDevice);
        }

        // (2) Replay forward across the chunk, capturing the 4 physical
        //     fields after each step (post-source-inject snapshot, matching
        //     the main forward.cu save convention).
        for (int local_i = 0; local_i < local_len; ++local_i) {
            int it = start + local_i;

            LAUNCH_VTI_VELOCITY(
                order, launch_config.grid, launch_config.block,
                fwd_view, inv_rho_t.data_ptr<float>(),
                grad_ctx, cpml_view, solver);

            LAUNCH_VTI_STRESS(
                order, launch_config.grid, launch_config.block,
                fwd_view,
                c11_t.data_ptr<float>(),
                c33_t.data_ptr<float>(),
                c13_t.data_ptr<float>(),
                grad_ctx, cpml_view, solver);

            for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
                int field_index = source_fields[isrc];
                float* field = vti_field_ptr(fwd_view, field_index);
                if (field == nullptr) continue;
                add_source<<<fwd_src_config.grid, fwd_src_config.block>>>(
                    field, p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it, forward_nsrc, solver);
            }

            // Snapshot to chunk buffer (vx, vz, sH, sV).
            int spatial_size = solver.nx * solver.nz;
            float* slot = u_chunk.select(0, local_i).data_ptr<float>();
            const int comp_stride = B * spatial_size;
            cudaMemcpyAsync(slot + 0 * comp_stride, fwd_view.vx,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(slot + 1 * comp_stride, fwd_view.vz,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(slot + 2 * comp_stride, fwd_view.sH,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(slot + 3 * comp_stride, fwd_view.sV,
                            B * spatial_size * sizeof(float),
                            cudaMemcpyDeviceToDevice);
        }

        // (3) Inner backward over [end-1 ... start], using u_chunk as the
        //     local saved forward wavefield.  Same logic as the full-mode
        //     `backward()` driver, restricted to this chunk's time range.
        for (int local_i = local_len - 1; local_i >= 0; --local_i) {
            int it = start + local_i;

            // adjoint source from record.select(0, it)
            for (int irec = 0; irec < nrec_fields; ++irec) {
                int field_index = receiver_fields[irec];
                float* field = vti_field_ptr(adj_view, field_index);
                if (field == nullptr) continue;
                add_source<<<adj_src_config.grid, adj_src_config.block>>>(
                    field, p.adjoint_source.select(0, irec).data_ptr<float>(),
                    p.adjoint_sources_loc.data_ptr<int>(),
                    it, adjoint_nsrc, solver);
            }

            // gradient computation — fvx_now / fvz_now / fsH / fsV come from
            // u_chunk.select(0, local_i); fsH_prev / fsV_prev come from u_chunk.select(0, local_i-1)
            // (or zero when local_i == 0 AND chunk_id == 0).
            const float* fvx_now = u_chunk.select(0, local_i).select(0, 0).data_ptr<float>();
            const float* fvz_now = u_chunk.select(0, local_i).select(0, 1).data_ptr<float>();
            const float* fsH_prev;
            const float* fsV_prev;
            if (it > 0) {
                if (local_i > 0) {
                    fsH_prev = u_chunk.select(0, local_i - 1).select(0, 2).data_ptr<float>();
                    fsV_prev = u_chunk.select(0, local_i - 1).select(0, 3).data_ptr<float>();
                } else {
                    // Crossing a chunk boundary: u_chunk.select(0, -1) does not exist,
                    // but the state at step start-1 is exactly the checkpoint
                    // this chunk was replayed from, snapshotted above.
                    fsH_prev = seed_sH.data_ptr<float>();
                    fsV_prev = seed_sV.data_ptr<float>();
                }
            } else {
                fsH_prev = zero_prev.data_ptr<float>();
                fsV_prev = zero_prev.data_ptr<float>();
            }

            if (it > 0) {
                vti_adjoint_A_2d(order, launch_config, adj_view,
                                 c11_t, c33_t, c13_t, scratch_x, scratch_z,
                                 grad_ctx, solver);
            }

            LAUNCH_VTI_CALC_GRAD(
                order, launch_config.grid, launch_config.block,
                fvx_now, fvz_now, fsH_prev, fsV_prev,
                adj_view,
                vp_t.data_ptr<float>(),
                epsilon_t.data_ptr<float>(),
                delta_t.data_ptr<float>(),
                rho_t.data_ptr<float>(),
                grad_vp.data_ptr<float>(),
                grad_eps.data_ptr<float>(),
                grad_delta.data_ptr<float>(),
                grad_rho.data_ptr<float>(),
                grad_ctx, solver);

            if (it > 0) {
                vti_adjoint_B_2d(order, launch_config, adj_view, inv_rho_t,
                                 scratch_x, scratch_z, grad_ctx, solver);
            }
        }
    }

    out.grads = {grad_vp, grad_eps, grad_delta, grad_rho};
    return out;
}
BackwardOutputCore backward_recursive_ckpt_core(const BackwardInputCore& /*in*/)
{
    SWEEP_CHECK(false,
        "AcousticVTI1st2D CUDA backward_recursive_ckpt is not yet implemented.");
    return {};
}






}  // namespace acoustic_vti_1st_2d
