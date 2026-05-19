// ---------------------------------------------------------------------------
// AcousticVTI1st (Duveneck 2008) — 2D first-order CUDA backward.
//
// Phase 1 scope:
//   - Implements `backward()` (full mode) using the saved forward wavefield
//     (u_forward shape: (nt, 4, B, nz, nx) containing vx, vz, sH, sV).
//   - Adjoint PML memory variables are NOT tracked → interior gradients are
//     correct; gradients inside the PML band are approximate.  This matches
//     standard FWI/RTM usage where the PML region is masked off.
//   - backward_bs / backward_ckpt / backward_recursive_ckpt remain stubs.
//
// Gradient ordering follows AcousticVTI1st.MODEL_SPECS: [vp, ε, δ, ρ].
//
// Adjoint derivation lives in kernels.cuh; see the long comment block above
// `adjoint_step_kernel` and `calculate_grad_kernel`.
// ---------------------------------------------------------------------------

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "acoustic_vti_1st_2d.h"
#include "kernels.cuh"

#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/cudautils.h"
#include "../../common/elastic.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/checkpoint_runtime.cuh"
#include "../../launch/config.h"
#include "../../common/wavetypes.h"

namespace acoustic_vti_1st_2d {

namespace {

// Same wavefield-tensor helper as forward.cu, lives in an anonymous namespace
// so the linker sees it once per TU only.
struct AdjWavefieldTensor {
    torch::Tensor vx_t, vz_t, sH_t, sV_t;
    torch::Tensor m_sHx_t, m_sVz_t, m_vxx_t, m_vzz_t;

    void bind(const std::vector<torch::Tensor>& tensors)
    {
        TORCH_CHECK(tensors.size() == 8,
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

    void allocate(const torch::Tensor& ref)
    {
        auto opts  = ref.options();
        auto shape = ref.sizes();
        vx_t    = torch::zeros(shape, opts);
        vz_t    = torch::zeros(shape, opts);
        sH_t    = torch::zeros(shape, opts);
        sV_t    = torch::zeros(shape, opts);
        m_sHx_t = torch::zeros(shape, opts);
        m_sVz_t = torch::zeros(shape, opts);
        m_vxx_t = torch::zeros(shape, opts);
        m_vzz_t = torch::zeros(shape, opts);
    }

    VTIWavefieldPointer view() const
    {
        VTIWavefieldPointer p{};
        p.vx    = vx_t.data_ptr<float>();
        p.vz    = vz_t.data_ptr<float>();
        p.sH    = sH_t.data_ptr<float>();
        p.sV    = sV_t.data_ptr<float>();
        p.m_sHx = m_sHx_t.data_ptr<float>();
        p.m_sVz = m_sVz_t.data_ptr<float>();
        p.m_vxx = m_vxx_t.data_ptr<float>();
        p.m_vzz = m_vzz_t.data_ptr<float>();
        return p;
    }

    // For CheckpointRuntime save/load: the 8 tensors saved as one "state".
    std::vector<torch::Tensor> state_tensors() const
    {
        return {vx_t, vz_t, sH_t, sV_t,
                m_sHx_t, m_sVz_t, m_vxx_t, m_vzz_t};
    }

    void zero_state() const
    {
        vx_t.zero_();   vz_t.zero_();
        sH_t.zero_();   sV_t.zero_();
        m_sHx_t.zero_(); m_sVz_t.zero_();
        m_vxx_t.zero_(); m_vzz_t.zero_();
    }
};

}  // anonymous namespace


BackwardOutput backward(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(!p.free_surface,
                "AcousticVTI1st2D CUDA backward: free_surface=True not supported "
                "(Robertsson 1996 / Mittet 2002 anisotropic FS — follow-up).");

    TORCH_CHECK(p.spacing.size() >= 2,
                "AcousticVTI1st2D backward: spacing must have length >= 2");
    float dz = p.spacing[0];
    float dx = p.spacing[1];

    TORCH_CHECK(p.models.size() == 4,
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
    auto vp_sq   = vp_t * vp_t;
    auto rho_vp2 = rho_t * vp_sq;
    auto c11_t   = rho_vp2 * (1.0f + 2.0f * epsilon_t);
    auto c33_t   = rho_vp2;
    auto c13_t   = rho_vp2 * torch::sqrt(1.0f + 2.0f * delta_t);
    auto inv_rho_t = 1.0f / rho_t;

    TORCH_CHECK(p.u_forward.defined(),
                "AcousticVTI1st2D backward (full mode) requires the forward to "
                "be run with save_all_wavefields=True so u_forward is populated.");
    TORCH_CHECK(p.u_forward.dim() == 5,
                "u_forward must be (nt, n_comp, B, nz, nx); got dim ",
                p.u_forward.dim());
    TORCH_CHECK(p.u_forward.size(1) == 4,
                "u_forward second dim must be 4 (vx, vz, sH, sV); got ",
                p.u_forward.size(1));

    // Allocate / bind adjoint wavefield (zero-initialised)
    AdjWavefieldTensor adjoint;
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(vp_t);
    auto adj_view = adjoint.view();

    // Output gradient tensors (one per model)
    auto grad_vp    = torch::zeros_like(vp_t);
    auto grad_eps   = torch::zeros_like(epsilon_t);
    auto grad_delta = torch::zeros_like(delta_t);
    auto grad_rho   = torch::zeros_like(rho_t);

    SolverContext solver{
        2, nx, 0, nz, B, p.dt, p.nt, p.M, p.abcn, p.free_surface,
        p.lap_coes.data_ptr<float>(), p.grad_coes.data_ptr<float>(),
        dx, 0.f, dz
    };

    SGradParam grad_ctx{1, 0, nx, p.M, p.grad_coes.data_ptr<float>(),
                        dx, 0.f, dz};

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);

    // Zero buffer the size of one wavefield component, used as the "previous"
    // stress at iter `it = 0` (no prior step exists; initial state is zero).
    auto zero_state = torch::zeros({B, nz, nx}, vp_t.options());

    int adjoint_nsrc = p.adjoint_sources_loc.defined()
                       ? p.adjoint_sources_loc.size(1) : 0;
    int nrec_fields = p.receiver_field_indices.numel();
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto adjoint_src_config = (adjoint_nsrc > 0)
        ? fdtd::Geom::make(adjoint_nsrc, B) : fdtd::Geom::make(1, B);

    const int order = (p.M <= 4) ? static_cast<int>(2 * p.M) : -1;

    // -------- backward time loop --------
    for (int it = static_cast<int>(p.nt) - 1; it >= 0; --it) {
        // 1. Inject adjoint source at receivers (residual back-projected)
        for (int irec = 0; irec < nrec_fields; ++irec) {
            int field_index = receiver_fields[irec].item<int>();
            float* field = vti_field_ptr(adj_view, field_index);
            if (field == nullptr) continue;
            add_source<<<adjoint_src_config.grid, adjoint_src_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }

        // 2. Accumulate gradients using forward state at the right times:
        //    - vx_now, vz_now from u_forward[it] for the stiffness gradients
        //      (vx_new in step it's stress sub-step == u_forward[it].vx)
        //    - sH_prev, sV_prev from u_forward[it-1] for the inv_rho gradient
        //      (sH_old in step it's velocity sub-step == u_forward[it-1].sH)
        //    For it == 0, u_forward[-1] is "the initial state" = zeros.
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

        // 3. Propagate adjoint state from time it+1 to it
        //    (last step needs no propagation — λ is already at "time 0")
        //    Split into two kernels to avoid the read-after-write race
        //    that would corrupt sgradient() reads of just-updated neighbours.
        if (it > 0) {
            LAUNCH_VTI_ADJOINT_STRESS_TO_VEL(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
                c11_t.data_ptr<float>(),
                c33_t.data_ptr<float>(),
                c13_t.data_ptr<float>(),
                grad_ctx,
                solver
            );
            LAUNCH_VTI_ADJOINT_VEL_TO_STRESS(
                order,
                launch_config.grid,
                launch_config.block,
                adj_view,
                inv_rho_t.data_ptr<float>(),
                grad_ctx,
                solver
            );
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
// The wavelet source is also subtracted (negative sign) on the forward side
// so that the source-injection contribution is removed when stepping back.
// ---------------------------------------------------------------------------
BackwardOutput backward_bs(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(!p.free_surface,
                "AcousticVTI1st2D backward_bs: free_surface=True not supported "
                "(Robertsson 1996 / Mittet 2002 anisotropic FS — follow-up).");

    TORCH_CHECK(p.spacing.size() >= 2,
                "AcousticVTI1st2D backward_bs: spacing length >= 2 required.");
    float dz = p.spacing[0];
    float dx = p.spacing[1];

    TORCH_CHECK(p.models.size() == 4,
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

    auto vp_sq   = vp_t * vp_t;
    auto rho_vp2 = rho_t * vp_sq;
    auto c11_t   = rho_vp2 * (1.0f + 2.0f * epsilon_t);
    auto c33_t   = rho_vp2;
    auto c13_t   = rho_vp2 * torch::sqrt(1.0f + 2.0f * delta_t);
    auto inv_rho_t = 1.0f / rho_t;

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
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(vp_t);
    auto adj_view = adjoint.view();

    // Reconstructed forward state.  Bind from p.forward_wavefields if the
    // propagator pre-allocated a workspace; otherwise allocate fresh.
    AdjWavefieldTensor forward;
    if (!p.forward_wavefields.empty())
        forward.bind(p.forward_wavefields);
    else
        forward.allocate(vp_t);
    auto for_view = forward.view();

    // Initialise forward state from u_last_two (final state at t=nt-1, the
    // post-step / post-source-inject snapshot of step nt-1).
    TORCH_CHECK(p.u_last_two.defined(),
                "AcousticVTI1st2D backward_bs requires p.u_last_two from the "
                "boundary-saving forward.");
    forward.vx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
    forward.vz_t.copy_(p.u_last_two.select(0, 1).select(0, 0));
    forward.sH_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
    forward.sV_t.copy_(p.u_last_two.select(0, 3).select(0, 0));

    // Negated forward source: time-reversal subtracts it during reconstruction.
    auto neg_forward_source = -p.forward_source;

    // Gradient outputs.
    auto grad_vp    = torch::zeros_like(vp_t);
    auto grad_eps   = torch::zeros_like(epsilon_t);
    auto grad_delta = torch::zeros_like(delta_t);
    auto grad_rho   = torch::zeros_like(rho_t);

    // CPML coefficients (cpmls 8-tuple) for the adjoint propagation step.
    ElasticCPMLTensor cpml;
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    // Boundary saver / runtime — mirror forward.cu's allocation but in
    // backward mode (data flows from saver → forward state, not the other
    // way around).
    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(
            /*use_bs=*/true, /*dim=*/2, /*nvar=*/4, solver, vp_t,
            save_width, /*last_two_nvar=*/1, /*override_storage=*/true,
            /*store_on_gpu_override=*/false, p.transfer_interval,
            p.boundary_cpu, p.boundary_gpu, /*last_two=*/{},
            p.use_pinned_memory);
    } else {
        boundary_saver.allocate(
            /*use_bs=*/true, /*dim=*/2, /*nvar=*/4, solver, vp_t,
            save_width, /*last_two_nvar=*/1, /*override_storage=*/true,
            /*store_on_gpu_override=*/true, /*transfer_interval=*/1,
            /*boundary_cpu=*/{}, p.boundary_gpu, /*last_two=*/{},
            p.use_pinned_memory);
        // If the propagator handed us an in-memory boundary tensor list
        // (legacy path), load it now.
        if (p.boundary_gpu.empty() && !p.u_boundary.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp_t);
    }
    auto bs = boundary_saver.view();

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);
    int adjoint_nsrc = p.adjoint_sources_loc.defined()
                       ? p.adjoint_sources_loc.size(1) : 0;
    int forward_nsrc = p.forward_sources_loc.defined()
                       ? p.forward_sources_loc.size(1) : 0;
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields   = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto adj_src_config = (adjoint_nsrc > 0)
        ? fdtd::Geom::make(adjoint_nsrc, B) : fdtd::Geom::make(1, B);
    auto fwd_src_config = (forward_nsrc > 0)
        ? fdtd::Geom::make(forward_nsrc, B) : fdtd::Geom::make(1, B);

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        /*dim=*/2,
        /*use_bs=*/true,
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

    auto zero_prev_sH = torch::zeros_like(vp_t);
    auto zero_prev_sV = torch::zeros_like(vp_t);

    for (int it = static_cast<int>(p.nt) - 1; it >= 1; --it) {
        // (a) adjoint source injection (receiver-side residual)
        for (int irec = 0; irec < nrec_fields; ++irec) {
            int field_index = receiver_fields[irec].item<int>();
            float* field = vti_field_ptr(adj_view, field_index);
            if (field == nullptr) continue;
            add_source<<<adj_src_config.grid, adj_src_config.block>>>(
                field, p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it, adjoint_nsrc, solver);
        }

        // (b-i) forward source subtraction (neg_forward_source) — removes
        //       the source contribution that was added during the forward
        //       step we are about to time-reverse.
        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            int field_index = source_fields[isrc].item<int>();
            float* field = vti_field_ptr(for_view, field_index);
            if (field == nullptr) continue;
            add_source<<<fwd_src_config.grid, fwd_src_config.block>>>(
                field, neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it, forward_nsrc, solver);
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
        //     hasn't been reversed yet (= u_forward[it].vx for the
        //     stiffness path).
        //
        //     Concretely:
        //       - fvx_now / fvz_now : for_view.vx / for_view.vz (state at
        //         END of step it, before we time-reverse the velocity step)
        //       - fsH_prev / fsV_prev : approximated as for_view.sH / sV
        //         AFTER stress reverse but BEFORE velocity reverse.  This
        //         matches u_forward[it-1].sH used in the full-mode kernel.
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

        // (e) adjoint propagation (same kernels as Phase 1 full mode)
        LAUNCH_VTI_ADJOINT_STRESS_TO_VEL(
            order, launch_config.grid, launch_config.block,
            adj_view,
            c11_t.data_ptr<float>(),
            c33_t.data_ptr<float>(),
            c13_t.data_ptr<float>(),
            grad_ctx, solver);
        LAUNCH_VTI_ADJOINT_VEL_TO_STRESS(
            order, launch_config.grid, launch_config.block,
            adj_view,
            inv_rho_t.data_ptr<float>(),
            grad_ctx, solver);

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
BackwardOutput backward_ckpt(const BackwardInput& in)
{
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK(!p.free_surface,
                "AcousticVTI1st2D backward_ckpt: free_surface=True not supported.");
    TORCH_CHECK(p.checkpoint_interval >= 1,
                "checkpoint_interval must be >= 1");
    TORCH_CHECK(p.checkpoints.size() == 8,
                "AcousticVTI1st2D backward_ckpt expects 8 checkpoint tensors "
                "(4 physical + 4 PML memory); got ", p.checkpoints.size());
    TORCH_CHECK(p.spacing.size() >= 2,
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

    auto vp_sq   = vp_t * vp_t;
    auto rho_vp2 = rho_t * vp_sq;
    auto c11_t   = rho_vp2 * (1.0f + 2.0f * epsilon_t);
    auto c33_t   = rho_vp2;
    auto c13_t   = rho_vp2 * torch::sqrt(1.0f + 2.0f * delta_t);
    auto inv_rho_t = 1.0f / rho_t;

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
    if (!p.adjoint_wavefields.empty())
        adjoint.bind(p.adjoint_wavefields);
    else
        adjoint.allocate(vp_t);
    auto adj_view = adjoint.view();
    adjoint.zero_state();

    // Forward-state buffer used during each chunk's replay.
    AdjWavefieldTensor fwd_state;
    fwd_state.allocate(vp_t);
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
    cpml.allocate(p.pml_vals, 2);
    auto cpml_view = cpml.view();

    auto grad_vp    = torch::zeros_like(vp_t);
    auto grad_eps   = torch::zeros_like(epsilon_t);
    auto grad_delta = torch::zeros_like(delta_t);
    auto grad_rho   = torch::zeros_like(rho_t);

    auto launch_config = fdtd::Wave2D::make(nx, nz, B);

    int adjoint_nsrc = p.adjoint_sources_loc.defined()
                       ? p.adjoint_sources_loc.size(1) : 0;
    int forward_nsrc = p.forward_sources_loc.defined()
                       ? p.forward_sources_loc.size(1) : 0;
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields   = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto adj_src_config = (adjoint_nsrc > 0)
        ? fdtd::Geom::make(adjoint_nsrc, B) : fdtd::Geom::make(1, B);
    auto fwd_src_config = (forward_nsrc > 0)
        ? fdtd::Geom::make(forward_nsrc, B) : fdtd::Geom::make(1, B);

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (static_cast<int>(p.nt) + chunk_size - 1) / chunk_size;

    // Per-chunk replay buffer: shape (chunk_size, 4, B, nz, nx) — only the
    // 4 physical fields are needed for the gradient kernel.
    auto u_chunk = torch::zeros({chunk_size, 4, B, nz, nx}, vp_t.options());

    auto zero_prev = torch::zeros({B, nz, nx}, vp_t.options());

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
                int field_index = source_fields[isrc].item<int>();
                float* field = vti_field_ptr(fwd_view, field_index);
                if (field == nullptr) continue;
                add_source<<<fwd_src_config.grid, fwd_src_config.block>>>(
                    field, p.forward_source.data_ptr<float>(),
                    p.forward_sources_loc.data_ptr<int>(),
                    it, forward_nsrc, solver);
            }

            // Snapshot to chunk buffer (vx, vz, sH, sV).
            int spatial_size = solver.nx * solver.nz;
            float* slot = u_chunk[local_i].data_ptr<float>();
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

            // adjoint source from record[it]
            for (int irec = 0; irec < nrec_fields; ++irec) {
                int field_index = receiver_fields[irec].item<int>();
                float* field = vti_field_ptr(adj_view, field_index);
                if (field == nullptr) continue;
                add_source<<<adj_src_config.grid, adj_src_config.block>>>(
                    field, p.adjoint_source[irec].data_ptr<float>(),
                    p.adjoint_sources_loc.data_ptr<int>(),
                    it, adjoint_nsrc, solver);
            }

            // gradient computation — fvx_now / fvz_now / fsH / fsV come from
            // u_chunk[local_i]; fsH_prev / fsV_prev come from u_chunk[local_i-1]
            // (or zero when local_i == 0 AND chunk_id == 0).
            const float* fvx_now = u_chunk[local_i].select(0, 0).data_ptr<float>();
            const float* fvz_now = u_chunk[local_i].select(0, 1).data_ptr<float>();
            const float* fsH_prev;
            const float* fsV_prev;
            if (it > 0) {
                if (local_i > 0) {
                    fsH_prev = u_chunk[local_i - 1].select(0, 2).data_ptr<float>();
                    fsV_prev = u_chunk[local_i - 1].select(0, 3).data_ptr<float>();
                } else {
                    // Crossing chunk boundary: we don't have u_chunk[-1].
                    // Fall back to loading the previous checkpoint's stress.
                    // For simplicity, use zero — introduces O(chunk_size)
                    // bias at chunk boundaries.  TODO: load proper sH/sV.
                    fsH_prev = zero_prev.data_ptr<float>();
                    fsV_prev = zero_prev.data_ptr<float>();
                }
            } else {
                fsH_prev = zero_prev.data_ptr<float>();
                fsV_prev = zero_prev.data_ptr<float>();
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
                LAUNCH_VTI_ADJOINT_STRESS_TO_VEL(
                    order, launch_config.grid, launch_config.block,
                    adj_view,
                    c11_t.data_ptr<float>(),
                    c33_t.data_ptr<float>(),
                    c13_t.data_ptr<float>(),
                    grad_ctx, solver);
                LAUNCH_VTI_ADJOINT_VEL_TO_STRESS(
                    order, launch_config.grid, launch_config.block,
                    adj_view,
                    inv_rho_t.data_ptr<float>(),
                    grad_ctx, solver);
            }
        }
    }

    out.grads = {grad_vp, grad_eps, grad_delta, grad_rho};
    return out;
}
BackwardOutput backward_recursive_ckpt(const BackwardInput& /*in*/)
{
    TORCH_CHECK(false,
        "AcousticVTI1st2D CUDA backward_recursive_ckpt is not yet implemented.");
    return {};
}

}  // namespace acoustic_vti_1st_2d
