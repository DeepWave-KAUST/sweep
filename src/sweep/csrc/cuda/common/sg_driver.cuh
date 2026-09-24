// Shared driver skeleton for the STAGGERED (elastic-family) equations.
//
// Sibling of eq_driver.cuh (the second-order acoustic-family skeleton), same
// philosophy: the control flow every hand-written copy shared lives here once,
// per-equation physics stays in composite traits hooks, and cross-cutting
// abilities (the stepped it_begin/it_end range, the physics phase-split, the
// segmented backward) become properties of the skeleton instead of of whichever
// copies happened to implement them.
//
// The staggered shape differs from the acoustic one in ways that are
// bit-load-bearing, which is why it is a second template rather than more
// hooks on the first: fields update in place (no buffer-role rotation), each
// step is a velocity substep then a stress substep (the phase-split is a
// PHYSICS split, not a spatial strip split), sources/receivers are per-field
// index loops, boundary saving stores a field LIST per step, last_two is a
// final-state field snapshot, and the backward computes no grad_wavelet and no
// illumination.  Reference transcription: elastic2d (gated by bitgate tiers
// A/B/C/T and ddgate).  das2d/das3d (derivative-buffer shape) and
// elastic_tti_2nd2d (second-order displacement — acoustic-shaped) are NOT this
// family.
//
// ---------------------------------------------------------------------------
// HOOK TIMING MAP — read this before any equation's driver_traits.cuh.
// Per entry point, the traits hooks fire in exactly this order; everything
// not named here is shared runtime (checkpoint / boundary machinery).
// Prologue of every entry (in call order): validate_backward (backward only),
// parse_models, setup_ctx, bind_or_alloc_* wavefields, init_aux_slabs,
// alloc_cpml, bind_grads, make_workspace, make_state,
// adjoint_source_signs.
//
// sg_generic_forward — per it in [it_begin, it_end):
//   velocity_substep           v: t -> t+1/2           (DD step_phase 1)
//   stress_substep             s: t -> t+1, u_allt[it] (DD step_phase 2 from here)
//   inject_source              per source field
//   <checkpoint save>          shared runtime, not a hook
//   save_boundary_fields       BS strips (when use_boundary_saving)
//   record_field               per receiver field
//   after the loop: save_last_state (final 5-field snapshot for backward_bs)
//
// sg_generic_backward (full storage) — prologue: the read-only zero grid the
//   imaging reads as v(nt) at it = nt-1, adjoint_workspace slot
//   SgCarrierSlots::FULL_ZERO (IMAGING_USES_NEXT_V equations only).  Per
//   reverse it:
//   fix_rho_grad_at_sources    body-force rho correction (pre-residual)
//   inject_residuals           signed residuals into the adjoint fields
//   vel_ptrs_from_u_forward    v(it) / v(it+1) pointers from u_forward
//   it == 0: image_standalone + fix_rho_grad_at_receivers, loop ends
//   it  > 0: full_mode_step    imaging + receiver-rho + adjoint step, in
//                              the equation's exact fused order
//
// sg_generic_backward_bs — per reverse it, floor max(it_lo, 1):
//   fix_rho_grad_at_sources / inject_residuals / uninject_forward_source  [inject_step]
//   bs_stress_half             stress recon (NOPML) + strip restore +
//                              imaging + receiver-rho + stress-adjoint half
//   bs_velocity_half           velocity-adjoint half + carrier capture +
//                              velocity recon (NOPML) + strip restore + prefetch
//   before the loop (first segment): seed_recon from u_last_two.
//   (DD runs step_phase 3 = injections, then 1, then 2 — same op order.)
//
// sg_generic_backward_ckpt — prologue: bind_or_alloc_recon_ckpt (the replay
//   state = forward_wavefields set 0, CKPT_STATE_COUNT tensors, one for the
//   whole call), seg_buffers (the N_VEL velocity histories, checkpoint_replay
//   slots sized for the longest chunk), the cross-chunk carriers from
//   adjoint_workspace (SgCarrierSlots, after the equation's WS_CARRIERS
//   scratch slots).  Per chunk (sg_backward_segment; the chunk's snapshot is
//   loaded straight into the replay state, chunk 0 zeroes it):
//   replay:  velocity_substep / stress_substep / save_seg_velocities /
//            inject_forward_sources
//   reverse: fix_rho_grad_at_sources / inject_residuals / vel_ptrs_from_seg /
//            image_standalone / fix_rho_grad_at_receivers /
//            (it > 0) plain_adjoint_step
//   after each chunk: export_seg_next_v hands v(start+1) to the older chunk.
//
// sg_generic_backward_recursive_ckpt — prologue: bind_or_alloc_recon_ckpt
//   (replay state = forward_wavefields set 0), current_v / next_v from
//   adjoint_workspace (SgCarrierSlots).  Per reverse it:
//   fix_rho_grad_at_sources / inject_residuals
//   sg_replay_forward_to_time: velocity/stress substeps + capture_velocities
//                              (IMAGING_USES_NEXT_V eqs also capture v at it+1)
//   vel_ptrs_from_carriers / image_standalone / fix_rho_grad_at_receivers /
//   (it > 0) plain_adjoint_step
// ---------------------------------------------------------------------------
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "common.cuh"
#include "context.h"
#include "checkpoint_runtime.cuh"
#include "cudautils.h"
#include "boundarysaver.cuh"
#include "boundary_runtime.cuh"
#include "boundary/session.cuh"
#include "wavetypes.h"
#include "eq_driver.cuh"     // Dims / read_dims / make_ctx / stencil_order
#include "../launch/config.h"

namespace eqdrv {

// ------------------------------------------------------------------------- //
// sg_generic_forward
// ------------------------------------------------------------------------- //

// Persistent forward runner: the whole prologue (validation, binding,
// runtime construction) runs ONCE in the constructor; run() is only the
// time loop.  The monolithic sg_generic_forward below is construct + one
// run, so this is the exact code path the bit gates exercise.  Reuse
// (a second run() call) requires gpu-direct boundary storage and no
// checkpointing -- the only modes whose cross-call state lives entirely in
// Python-bound buffers.
template <class Eq>
class SgForwardRunner final : public IForwardRunner {
public:
    explicit SgForwardRunner(const ForwardInput& in)
        : p(in)
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());

        models = Eq::parse_models(p);
        const auto& vp = p.models[0];
        d = read_dims<Eq::NDIM>(vp);

        // ---- Stepped forward range [it_begin, it_end) ----
        const int it0 = p.it_begin;
        const int it1 = (p.it_end < 0) ? static_cast<int>(p.nt) : p.it_end;
        check_run_args(it0, it1, p.step_phase, p.nt);
        const bool stepped = (it0 != 0) || (it1 != static_cast<int>(p.nt));
        // The staggered phase-split is a PHYSICS split (unlike acoustic's spatial
        // strip split): phase 1 = full-grid velocity update only, phase 2 =
        // full-grid stress update + source/record/BS/checkpoint tail. A DD
        // driver exchanges v halos between the phases and s halos after phase
        // 2, so the cut-adjacent stress columns read exchanged (not locally
        // recomputed) velocities — that keeps the transverse CPML memory
        // divergence in the halo columns from ever reaching owned cells.

        // On a continuation call the internal allocate() would silently zero the
        // propagation state — the caller must keep binding the same tensors.
        // (No buffer-role rotation: every field updates in place, so the caller
        // binds the SAME tensor list for every segment.)
        TORCH_CHECK(it0 == 0 || !p.wavefields.empty(),
                    "stepped continuation (it_begin>0) requires Python-bound wavefields");
        Eq::bind_or_alloc_forward(wavefield, p, vp);
        wf = Eq::view(wavefield);

        Eq::alloc_cpml(cpml_tensor, p);
        cpml = cpml_tensor.view();

        nsrc = p.sources_loc.size(1);
        nrec = p.receivers_loc.size(1);
        nsrc_fields = p.source_field_indices.numel();
        nrec_fields = p.receiver_field_indices.numel();
        source_fields = p.source_field_indices.to(torch::kCPU);
        receiver_fields = p.receiver_field_indices.to(torch::kCPU);
        // An empty tensor counts as unbound: nothing could be recorded into it.
        TORCH_CHECK(!stepped || (p.record_out.defined() && p.record_out.numel() > 0),
                    "stepped forward requires record_out bound from Python");
        // MANDATORY, every mode: every staggered equation declares
        // cuda_layout.record_shape (record_multi), so _c.py Wrapper.forward
        // always allocates the record and binds it as ForwardInput.record_out.
        TORCH_CHECK(p.record_out.defined() && p.record_out.numel() > 0,
                    Eq::NAME, "/forward requires the propagator-bound record_out "
                    "(cuda_layout.record_shape)");
        record = bound_required(p.record_out, {nrec_fields, d.B, nrec, p.nt}, vp.options(),
                                "record_out");

        Eq::validate_forward(p);

        if (p.use_checkpoint) {
            TORCH_CHECK((int)p.checkpoints.size() == Eq::CKPT_NVAR,
                        Eq::CKPT_COUNT_MSG);
            if (p.use_recursive_checkpoint) {
                TORCH_CHECK(p.checkpoint_steps.defined(), "Recursive checkpointing expects checkpoint_steps");
                TORCH_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D");
            } else {
                TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
            }
        }

        if (p.save_all_wavefields) {
            TORCH_CHECK(!stepped || p.u_allt_out.defined(),
                        "stepped + save_all_wavefields requires u_allt_out bound from Python");
            // MANDATORY whenever save_all_wavefields is on: every staggered
            // equation declares cuda_layout.save_all_shape, and _c.py
            // Wrapper.forward binds u_allt_out in exactly that case.
            TORCH_CHECK(p.u_allt_out.defined() && p.u_allt_out.numel() > 0,
                        Eq::NAME, "/full requires the propagator-bound u_allt_out "
                        "(cuda_layout.save_all_shape)");
            u_allt = bound_required(p.u_allt_out, Eq::allt_shape(d, p.nt), vp.options(),
                                    "u_allt_out");
        }

        solver_.emplace(make_ctx<Eq>(p, d));
        SolverContext& solver = *solver_;
        Eq::setup_ctx(solver, p);
        solver.set_cut_mask(p.cut_face_mask);   // cut-aware phys bounds (0 = single domain)
        Eq::init_aux_slabs(solver, wavefield);

        save_width = solver.M + 1;
        // The internal full-storage fallback ring is per-call; segments after the
        // first would lose everything saved before them.
        if (stepped && p.use_boundary_saving)
            TORCH_CHECK(!p.boundary_gpu.empty(),
                        "stepped forward with boundary saving requires Python-bound boundary_gpu");
        staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
        if (staged_boundary)
            boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, solver, vp,
                                    save_width, 1, true, false, p.transfer_interval,
                                    p.boundary_cpu, p.boundary_gpu, p.last_two,
                                    p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
        else
            boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, solver, vp,
                                    save_width, 1, true, true, 1, {}, p.boundary_gpu,
                                    p.last_two, p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
        bs = boundary_saver.view();

        launch_config = wave_config<Eq::NDIM>(d);
        source_config = fdtd::Geom::make(nsrc, d.B);
        record_config = fdtd::Geom::make(nrec, d.B);

        state.emplace(Eq::make_state(p, d, models, launch_config,
                                     source_config, record_config));

        // A Python-owned BoundarySession, when bound, keeps the copy stream and
        // the ring events alive ACROSS calls.  Under DD every time step is a
        // separate entry into the extension, so a per-call copy stream can never
        // hold a transfer in flight and transfer_interval/ring_buffers overlap
        // nothing -- it also destroys and recreates the stream and its events
        // every step.  Without a session BoundaryScope builds both locally and
        // the behaviour is exactly the old per-call path.  The acoustic
        // skeleton has had this since dev 848a100; the staggered one had not,
        // which is half of why staged storage was refused here.
        boundary_scope.emplace(
            p.boundary_session ? p.boundary_session->impl() : nullptr,
            BoundarySessionImpl::Phase::Forward,
            boundary_saver,
            Eq::NDIM,
            p.use_boundary_saving,
            p.boundary_on_cpu,
            p.boundary_on_disk,
            p.boundary_disk_async_read,
            p.transfer_interval,
            p.boundary_ring_buffers,
            p.boundary_disk_files
        );
        boundary_runtime = &boundary_scope->runtime();
        checkpoint_runtime.emplace(
            p.checkpoints,
            Eq::CKPT_NVAR,
            p.use_checkpoint,
            p.use_recursive_checkpoint,
            p.checkpoint_interval,
            p.checkpoint_steps,
            p.checkpoint_on_cpu,
            "forward",
            Eq::NAME,
            p.it_begin
        );
    }

    ForwardOutput run(int run_it_begin, int run_it_end, int run_step_phase) override
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());
        SolverContext& solver = *solver_;

        const int it0 = run_it_begin;
        const int it1 = (run_it_end < 0) ? static_cast<int>(p.nt) : run_it_end;
        check_run_args(it0, it1, run_step_phase, p.nt);
        if (run_calls++ > 0)
            TORCH_CHECK(!p.use_checkpoint && !staged_boundary,
                        "persistent stepped runner reuse requires gpu-direct "
                        "boundary storage and no checkpointing");

        const bool do_v = (run_step_phase == 0 || run_step_phase == 1);
        const bool do_s = (run_step_phase == 0 || run_step_phase == 2);
        TORCH_CHECK(run_step_phase == 0 || it1 == it0 + 1,
                    "elastic phase-split requires a single step (it_end == it_begin + 1)");

        float* u_this_t = nullptr;

        for (int it = it0; it < it1; ++it) {

            u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

            if (do_v)
                Eq::velocity_substep(*state, wf, cpml, solver);   // t+0.5

            if (!do_s)
                continue;

            Eq::stress_substep(*state, wf, cpml, solver, u_this_t);   // t+1.0

            for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
                float* field = Eq::field_ptr(wf, source_fields[isrc].item<int>());
                if (field == nullptr) continue;
                Eq::inject_source(*state, solver, field, p.source, p.sources_loc,
                                  it, nsrc);
            }

            // Deferred u_allt snapshot, AFTER the injection.  Every equation on
            // this skeleton today writes u_allt from inside its stress kernel
            // (through u_this_t) and leaves this a no-op -- the acoustic
            // skeleton has the same hook for the same reason (eq_driver.cuh).
            // It exists because the pseudo-acoustic VTI family cannot: it needs
            // the POST-inject state, since u_forward[it-1].sH is the sH INPUT to
            // step it's velocity substep and the rho gradient reads it as
            // fsH_prev.  Capturing before the injection changes grad[rho]
            // whenever the source lands on sH/sV, which is that family's default.
            Eq::capture_allt(u_allt, wf, solver, it);

            checkpoint_runtime->save_forward(static_cast<int>(it), static_cast<int>(p.nt),
                                             wavefield.checkpoint_tensors());

            if (p.use_boundary_saving)
                Eq::save_boundary_fields(*boundary_runtime, *state, solver, wf,
                                         it, (int)p.nt, bs, save_width);

            for (int irec = 0; irec < nrec_fields; ++irec) {
                float* field = Eq::field_ptr(wf, receiver_fields[irec].item<int>());
                if (field == nullptr) continue;
                Eq::record_field(*state, solver, field, record, irec,
                                 p.receivers_loc, it, nrec);
            }
        }

        // Save the final state for backward (only once the final segment has
        // run; mid-run segments leave last_two untouched).
        if (p.use_boundary_saving && it1 == static_cast<int>(p.nt))
            Eq::save_last_state(boundary_saver, wavefield);

        // With a persistent session the trailing sync belongs to the phase, not
        // to this one call -- Python drives it via session.finish().
        if (boundary_scope->owns())
            boundary_runtime->synchronize();

        ForwardOutput out;
        out.wavefield = u_allt;
        out.last_two = boundary_saver.last_two_t;
        out.record = record;

        return out;
    }

private:
    static void check_run_args(int it0, int it1, int step_phase, int64_t nt)
    {
        TORCH_CHECK(0 <= it0 && it0 <= it1 && it1 <= static_cast<int>(nt),
                    "stepped forward: require 0 <= it_begin <= it_end <= nt, got [",
                    it0, ", ", it1, ") with nt=", nt);
        TORCH_CHECK(step_phase == 0 || step_phase == 1 || step_phase == 2,
                    "elastic step_phase must be 0, 1 or 2");
    }

    // Declaration order == construction order; destruction runs in reverse,
    // matching the hand-written function's stack unwind.
    ForwardInput p;
    typename Eq::Models models;
    Dims d;
    typename Eq::Wavefield wavefield;
    typename Eq::WfView wf;
    typename Eq::CPML cpml_tensor;
    decltype(std::declval<typename Eq::CPML>().view()) cpml;
    int nsrc = 0, nrec = 0, nsrc_fields = 0, nrec_fields = 0;
    torch::Tensor source_fields, receiver_fields;
    torch::Tensor record;
    torch::Tensor u_allt;
    std::optional<SolverContext> solver_;
    int save_width = 0;
    bool staged_boundary = false;
    EffectiveBoundarySaver boundary_saver;
    GeneralBoundaryPointer bs{};
    fdtd::LaunchConfig launch_config{}, source_config{}, record_config{};
    std::optional<typename Eq::State> state;
    std::optional<BoundaryScope> boundary_scope;
    BoundaryRuntime* boundary_runtime = nullptr;
    std::optional<CheckpointRuntime> checkpoint_runtime;
    int run_calls = 0;
};

template <class Eq>
ForwardOutput sg_generic_forward(const ForwardInput& in)
{
    SgForwardRunner<Eq> runner(in);
    return runner.run(in.it_begin, in.it_end, in.step_phase);
}


// Validate the stepped-backward segment fields, the DD cut mask and the
// backward phase split for staggered-family entry points.  ``need_recon`` is
// true for boundary-saving mode.
// ``bw_it_begin``/``bw_it_end``/``step_phase`` are explicit so a persistent
// runner can re-validate each run() with that call's range; the monolithic
// entries pass the input-struct fields, reproducing the legacy behaviour.
template <class Eq>
void sg_check_stepped_backward(const BackwardInput& p, bool need_recon,
                               int bw_it_begin, int bw_it_end, int step_phase)
{
    const int it_hi = (bw_it_begin < 0) ? static_cast<int>(p.nt) : bw_it_begin;
    const int it_lo = bw_it_end;
    TORCH_CHECK(0 <= it_lo && it_lo < it_hi && it_hi <= static_cast<int>(p.nt),
                "stepped backward: require 0 <= bw_it_end < bw_it_begin <= nt, got [",
                it_lo, ", ", it_hi, ") with nt=", p.nt);
    TORCH_CHECK((p.cut_face_mask & ~Eq::CUT_MASK_BITS) == 0,
                Eq::NDIM, "D cut_face_mask uses ", Eq::CUT_MASK_DESC,
                " only, got ", p.cut_face_mask);
    TORCH_CHECK(step_phase >= 0 && step_phase <= 3,
                "elastic backward step_phase must be 0, 1, 2 or 3 "
                "(3 = residual/source injections only)");
    const bool phased = (step_phase != 0);
    if (phased) {
        TORCH_CHECK(need_recon,
                    "phased elastic backward (step_phase) is only supported "
                    "for the boundary-saving path (backward_bs)");
        TORCH_CHECK(it_hi == it_lo + 1,
                    "elastic backward phase-split requires a single-step "
                    "segment (bw_it_begin == bw_it_end + 1)");
    }
    if (need_recon && p.cut_face_mask != 0) {
        TORCH_CHECK(!p.boundary_on_disk,
                    "domain-decomposed backward_bs (cut_face_mask) supports "
                    "gpu-direct or cpu boundary storage only "
                    "(boundary_on_disk unsupported in v1)");
    }
    const bool stepped = (it_hi < static_cast<int>(p.nt)) || (it_lo > 0);   // == bw_stepped()
    if (!stepped && !phased)
        return;
    TORCH_CHECK((int)p.adjoint_wavefields.size() == Eq::ADJ_WF_COUNT,
                "stepped elastic backward requires the ", Eq::ADJ_WF_COUNT,
                "-tensor adjoint wavefield list bound from Python");
    TORCH_CHECK(p.grads_out.size() == p.models.size(),
                "stepped elastic backward requires Python-bound grads_out "
                "(one per model: vp, vs, rho — elastic computes no "
                "grad_wavelet)");
    TORCH_CHECK(p.illum_out.empty(),
                "elastic backward computes no illuminations; illum_out must "
                "be empty");
    if (need_recon) {
        TORCH_CHECK((int)p.forward_wavefields.size() == Eq::RECON_WF_COUNT,
                    "stepped elastic backward_bs requires the ", Eq::RECON_WF_COUNT,
                    "-tensor reconstruction list ", Eq::RECON_LIST_DESC,
                    " bound from Python");
        TORCH_CHECK(!p.boundary_on_disk,
                    "stepped backward_bs supports gpu-direct or cpu boundary "
                    "storage only (boundary_on_disk unsupported in v1)");
    }
}

// Skeleton-owned slots of the adjoint_workspace pool.  The pool
// (cuda_layout.backward_workspace_shapes(mode)) holds the equation's own
// adjoint scratch first -- Eq::WS_CARRIERS slots, what Eq::make_workspace
// binds -- and the skeleton's model-shaped slots after it:
//   full:  FULL_ZERO, one read-only zero grid -- the v(nt) the imaging at
//          it = nt-1 reads (IMAGING_USES_NEXT_V equations only; the others
//          take no slot and hand their imaging null next-pointers);
//   ckpt / recursive, N_VEL velocity carriers each:
//   NOW  = next_segment_v (ckpt: the younger chunk's v(end)) /
//          current_v      (recursive: the captured v(it));
//   NEXT = prev_segment_next_v (ckpt: this chunk's v(start+1), for the older
//          chunk) / next_v (recursive: the captured v(it+1)) -- only for
//          equations whose imaging reads v(t+1) (IMAGING_USES_NEXT_V).
// The propagator zeroes the pool once per gradient-bearing forward, the state
// a fresh zeros_like started from.  The full entry only ever reads FULL_ZERO
// (every consumer takes it as a const float*), so re-entry (stepped / DD)
// is safe.  The ckpt entry relies on the zero state for next_segment_v at
// the tail chunk (v(nt) does not exist) and is entered at most once per
// forward: it refuses stepped/DD calls, and a second backward over one graph
// is refused in Python (_c.py Wrapper.backward).
template <class Eq>
struct SgCarrierSlots {
    static constexpr int FULL_ZERO = Eq::WS_CARRIERS;
    static constexpr int FULL_COUNT = Eq::IMAGING_USES_NEXT_V ? 1 : 0;
    static constexpr int NOW = Eq::WS_CARRIERS;
    static constexpr int NEXT = Eq::WS_CARRIERS + Eq::N_VEL;
    static constexpr int CKPT_COUNT = Eq::IMAGING_USES_NEXT_V ? 2 * Eq::N_VEL : 0;
    static constexpr int RECURSIVE_COUNT = Eq::IMAGING_USES_NEXT_V ? 2 * Eq::N_VEL : Eq::N_VEL;
};

// Exactly the equation's scratch plus this mode's skeleton slots (the full
// mode's zero grid, the checkpoint modes' velocity carriers): a pool of any
// other size means the Python declaration drifted.  MANDATORY -- the pool is
// cuda_layout.backward_workspace_shapes(mode), which every staggered equation
// declares as a callable, so _c.py sizes and binds it for every backward
// (Prop.forward calls _ensure_adjoint_workspace_buffers(batch, workspace_mode)
// whenever a gradient is asked for, and Wrapper.backward hands the pool over
// as BackwardInput.adjoint_workspace).
template <class Eq>
void sg_check_ckpt_workspace(const BackwardInput& p, int n_carriers, const char* mode)
{
    const int expected = Eq::WS_CARRIERS + n_carriers;
    TORCH_CHECK(static_cast<int>(p.adjoint_workspace.size()) == expected,
                Eq::NAME, " backward_", mode, ": requires the propagator-bound "
                "adjoint_workspace (cuda_layout.backward_workspace_shapes) holding ",
                expected, " tensors ([0-", Eq::WS_CARRIERS - 1, "] = the adjoint scratch, then ",
                n_carriers, " skeleton slots), got ", p.adjoint_workspace.size());
}

// ---- sg_generic_backward (full storage) ----
template <class Eq>
BackwardOutput sg_generic_backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    // Per-equation entry validation (model/PML counts, mode-specific input
    // tensors) with the equation's own per-mode message texts.  Runs FIRST,
    // like the hand-written entry points it replaces.
    Eq::validate_backward(p, "full");
    sg_check_stepped_backward<Eq>(p, /*need_recon=*/false,
                                  p.bw_it_begin, p.bw_it_end, p.step_phase);
    // it == 0 keeps its legacy asymmetry (gradient only, no adjoint step)
    // and runs in whichever segment contains it — position-based, so any
    // partition reproduces the monolithic loop bit-for-bit.
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    const bool first_segment = (it_hi == static_cast<int>(p.nt));
    BackwardOutput out;

    typename Eq::Models models = Eq::parse_models(p);
    const auto& vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto source_fields = p.source_field_indices.to(torch::kCPU);

    SolverContext solver = make_ctx<Eq>(p, d);
    Eq::setup_ctx(solver, p);
    // DD: cut-aware PML predicates in the adjoint prepare kernels.
    solver.set_cut_mask(p.cut_face_mask);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);
    Eq::init_aux_slabs(solver, adjoint);
    // A continuation segment must keep the carried adjoint state; the 3-D
    // twin zeroes it on the FIRST segment only (2-D relies on Python-zeroed
    // buffers and no-ops here).
    Eq::zero_adjoint_if_first_segment(adjoint, first_segment);

    auto adj_view = Eq::view(adjoint);

    std::vector<torch::Tensor> grads;
    Eq::bind_grads(p, grads);
    sg_check_ckpt_workspace<Eq>(p, SgCarrierSlots<Eq>::FULL_COUNT, "full");
    typename Eq::Workspace workspace = Eq::make_workspace(p, vp);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml_view = cpml_tensor.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);
    typename Eq::State state = Eq::make_state(p, d, models, launch_config,
                                              fwd_source_config, adj_source_config);

    // v(nt) for the imaging at it = nt-1: the read-only zero grid at
    // adjoint_workspace slot FULL_ZERO (SgCarrierSlots), zeroed by the
    // propagator once per gradient-bearing forward and only ever read here
    // (vel_ptrs_from_u_forward hands it out as a const float*).  Equations
    // without a v(t+1) imaging term take no slot and never index it.
    // MANDATORY: sg_check_ckpt_workspace above already required the pool at
    // WS_CARRIERS + FULL_COUNT, which is exactly what the "full" branch of
    // cuda_layout.backward_workspace_shapes declares for every
    // IMAGING_USES_NEXT_V equation of this family.
    const torch::Tensor zero_velocity = Eq::IMAGING_USES_NEXT_V
        ? pool_required(p.adjoint_workspace, SgCarrierSlots<Eq>::FULL_ZERO, vp,
                        "adjoint_workspace")
        : torch::Tensor();
    const auto adj_source_signs = Eq::adjoint_source_signs(p, receiver_fields);

    // it_hi == nt and it_lo == 0 without DD, so this is dev's full
    // reverse loop verbatim in the single-domain case.
    for (int it = it_hi - 1; it >= it_lo; --it) {
        Eq::fix_rho_grad_at_sources(state, solver, adj_view, p, source_fields, it, grads);
        Eq::inject_residuals(state, solver, adj_view, p, receiver_fields,
                             adj_source_signs, it, adjoint_nsrc);

        typename Eq::VelPtrs vptrs =
            Eq::vel_ptrs_from_u_forward(p, it, zero_velocity);

        // Reverse step 0 has no adjoint apply kernel, so its gradient imaging
        // cannot be folded — emit it as a standalone calculate_grad pass.
        if (it == 0) {
            Eq::image_standalone(state, solver, adj_view, vptrs, grads);
            Eq::fix_rho_grad_at_receivers(state, solver, grads, vptrs, p,
                                  receiver_fields, it, adjoint_nsrc);
            continue;
        }

        // FULL-mode step, in this equation's exact order: elastic folds the
        // imaging into the stress-adjoint-prepare kernel and corrects rho
        // after; das_mu images standalone, corrects rho, THEN steps the
        // adjoint.  The receiver-rho correction position is bit-load-bearing,
        // so the whole compound lives in the hook.
        Eq::full_mode_step(state, solver, adjoint, workspace, cpml_view,
                            vptrs, grads, p, receiver_fields, it, adjoint_nsrc);
    }

    out.grads = grads;
    return out;
}

// ---- sg_generic_backward_bs ----
// Persistent boundary-saving backward runner: prologue once in the
// constructor, run(bw_it_begin, bw_it_end, step_phase) is the reverse loop.
// The first-segment hooks (zero_adjoint_if_first_segment_bs, seed_recon) stay in run() gated
// on that run's range, exactly like the hand-written per-call gating.
// zero_adjoint_if_first_segment_bs runs at the head of run() instead of right after the
// adjoint bind: every construction op in between touches other tensors
// only, so the value sequence is identical.  Reuse (a second run()) requires
// gpu-direct boundary storage.
template <class Eq>
class SgBackwardBsRunner final : public IBackwardRunner {
public:
    explicit SgBackwardBsRunner(const BackwardInput& in)
        : p(in)
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());

        Eq::validate_backward(p, "bs");
        sg_check_stepped_backward<Eq>(p, /*need_recon=*/true,
                                      p.bw_it_begin, p.bw_it_end, p.step_phase);

        models = Eq::parse_models(p);
        const auto& vp = p.models[0];
        d = read_dims<Eq::NDIM>(vp);

        adjoint_nsrc = p.adjoint_sources_loc.size(1);
        forward_nsrc = p.forward_sources_loc.size(1);
        source_fields = p.source_field_indices.to(torch::kCPU);
        receiver_fields = p.receiver_field_indices.to(torch::kCPU);

        solver_.emplace(make_ctx<Eq>(p, d));
        SolverContext& solver = *solver_;
        Eq::setup_ctx(solver, p);
        // DD: skip cut faces in the strip restore, collapse the NOPML exclusion
        // bands to the stencil halo on cut sides, and route cut-side cells of
        // the adjoint prepare kernels through the interior branch.
        solver.set_cut_mask(p.cut_face_mask);

        Eq::bind_or_alloc_adjoint(adjoint, p, vp);
        Eq::init_aux_slabs(solver, adjoint);

        // Reconstruction state: the physical fields plus the carried velocity
        // tensors (v at time it+1, consumed by the gradient kernel).  When
        // stepping, all must be Python-owned to survive segment boundaries.
        carriers = Eq::bind_or_alloc_recon(forward, p, vp);

        for_view = Eq::view(forward);
        adj_view = Eq::view(adjoint);

        Eq::bind_grads(p, grads);
        workspace.emplace(Eq::make_workspace(p, vp));

        Eq::alloc_cpml(cpml_tensor, p);
        cpml_view = cpml_tensor.view();

        save_width = solver.M + 1;
        staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
        // ``bs.last_two`` is dead in the BACKWARD: ``save_last_state`` writes it
        // in the forward only, and ``seed_recon`` reads ``p.u_last_two``
        // directly.  Passing {} sent allocate_last_two down its self-allocating
        // branch and built a {BS_NVAR, 1, B, 1, nz[, ny], nx} FP32 buffer that
        // nothing ever touches -- 9 padded 3-D grids for elastic3d and
        // elastic_tti_sg3d, 15 for das_mu3d -- on the GPU, or in PINNED HOST
        // memory on the staged path.  Binding the tensor Python already owns
        // costs nothing and returns all of it.  Same defect and same fix as the
        // acoustic skeleton (eq_driver.cuh, dev 4290248); allocate_last_two
        // still self-allocates if the tensor is undefined.
        const torch::Tensor& last_two_bound = p.u_last_two;
        if (staged_boundary) {
            boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, solver, vp, save_width,
                                    1, true, false, p.transfer_interval,
                                    p.boundary_cpu, p.boundary_gpu, last_two_bound,
                                    p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
        } else {
            boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, solver, vp, save_width,
                                    1, true, true, 1, {}, p.boundary_gpu, last_two_bound,
                                    p.use_pinned_memory, /*tangent_pad=*/0, p.boundary_staging);
            if (p.boundary_gpu.empty())
                boundary_saver.load_from_vector(p.u_boundary, vp);
        }
        bs = boundary_saver.view();

        launch_config = wave_config<Eq::NDIM>(d);
        fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
        adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);
        state.emplace(Eq::make_state(p, d, models, launch_config,
                                     fwd_source_config, adj_source_config));

        // Same persistent-session handling as the forward; see there.
        boundary_scope.emplace(
            p.boundary_session ? p.boundary_session->impl() : nullptr,
            BoundarySessionImpl::Phase::Backward,
            boundary_saver,
            Eq::NDIM,
            true,
            p.boundary_on_cpu,
            p.boundary_on_disk,
            p.boundary_disk_async_read,
            p.transfer_interval,
            p.boundary_ring_buffers,
            p.boundary_disk_files
        );
        boundary_runtime = &boundary_scope->runtime();

        adj_source_signs = Eq::adjoint_source_signs(p, receiver_fields);
    }

    BackwardOutput run(int bw_it_begin, int bw_it_end, int run_step_phase) override
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());
        SolverContext& solver = *solver_;
        BackwardOutput out;

        sg_check_stepped_backward<Eq>(p, /*need_recon=*/true,
                                      bw_it_begin, bw_it_end, run_step_phase);
        // The staggered BS loop legacy floor is it == 1 (no it==0 tail), so the
        // segment containing it == 0 simply runs nothing extra.
        const int it_hi = (bw_it_begin < 0) ? static_cast<int>(p.nt) : bw_it_begin;
        const int it_lo = bw_it_end;
        const bool first_segment = (it_hi == static_cast<int>(p.nt));
        // Phase split (DD): 1 = inject + stress recon/restore + gradient +
        // stress adjoint; 2 = velocity adjoint + fv_prev capture + velocity
        // recon/restore.  step_phase 3 = injections only (driver prologue for the
        // first reverse step).  Monolithic step_phase 0 keeps the original
        // head-of-loop position; the executed op sequence is identical either way.
        const bool inject_only = (run_step_phase == 3);
        const bool phased = (run_step_phase != 0);
        const bool do_p1 = !inject_only && (run_step_phase != 2);
        const bool do_p2 = !inject_only && (run_step_phase != 1);

        if (run_calls++ > 0)
            TORCH_CHECK(!staged_boundary,
                        "persistent stepped runner reuse requires gpu-direct "
                        "boundary storage");

        // Equations whose hand-written backward_bs zeroed the adjoint state after
        // binding hook it here (FIRST segment only — a continuation segment must
        // keep the carried adjoint state); the rest no-op.
        Eq::zero_adjoint_if_first_segment_bs(adjoint, first_segment);

        // Seed the reverse reconstruction from the saved last snapshot — FIRST
        // segment only (and not on a phase-2 re-entry).  In phased mode the seed
        // belongs to the INJECTION sub-phase (3): the driver runs 3 before 1, and
        // the injections write recon fields the seed would otherwise overwrite.
        if (first_segment && (inject_only || (!phased && do_p1)))
            Eq::seed_recon(forward, p);

        // Without it_hi the default primes the chunk holding step nt-1 -- the
        // TAIL chunk -- on every segment, so a stepped or domain-decomposed
        // reverse loop fetched the wrong slabs on all but its first call.
        boundary_runtime->prefetch_initial_backward_chunk((int)p.nt, it_hi);

        // Residual / source injections for reverse index jt, one unit (the rho
        // correction must read the adjoint velocity BEFORE jt's residuals land):
        // body-force source-cell rho correction, signed receiver residuals,
        // reconstruction un-injection of the forward source (add_source_signed
        // with sign -1: the sample's sign bit flipped at the atomicAdd,
        // bit-identical to the negated copy this used to build per call).
        auto inject_step = [&](int jt) {
            Eq::fix_rho_grad_at_sources(*state, solver, adj_view, p, source_fields, jt, grads);
            Eq::inject_residuals(*state, solver, adj_view, p, receiver_fields,
                                 adj_source_signs, jt, adjoint_nsrc);
            Eq::uninject_forward_source(*state, solver, for_view, p, source_fields,
                                        jt, forward_nsrc);
        };

        for (int it = it_hi - 1; it >= std::max(it_lo, 1); --it) {

            // Monolithic runs the injections at the head of the loop; the DD
            // phased path runs them as their own sub-phase (step_phase 3) at the
            // SAME position, so the op sequence is identical.
            if (!phased || inject_only)
                inject_step(it);
            if (inject_only)
                continue;

            if (do_p1)
                Eq::bs_stress_half(*state, solver, for_view, adj_view, adjoint, *workspace,
                              cpml_view, *boundary_runtime, bs, save_width,
                              grads, carriers, p, receiver_fields, it, adjoint_nsrc);

            if (do_p2)
                Eq::bs_velocity_half(*state, solver, for_view, adjoint, *workspace, cpml_view,
                              *boundary_runtime, bs, save_width, carriers, forward,
                              it, (int)p.nt);
        }

        out.grads = grads;
        return out;
    }

private:
    // Declaration order == construction order; destruction runs in reverse,
    // matching the hand-written function's stack unwind.
    BackwardInput p;
    typename Eq::Models models;
    Dims d;
    int adjoint_nsrc = 0, forward_nsrc = 0;
    torch::Tensor source_fields, receiver_fields;
    std::optional<SolverContext> solver_;
    typename Eq::Wavefield adjoint;
    typename Eq::Wavefield forward;
    typename Eq::ReconCarriers carriers;
    typename Eq::WfView for_view;
    typename Eq::WfView adj_view;
    std::vector<torch::Tensor> grads;
    std::optional<typename Eq::Workspace> workspace;
    typename Eq::CPML cpml_tensor;
    decltype(std::declval<typename Eq::CPML>().view()) cpml_view;
    int save_width = 0;
    bool staged_boundary = false;
    EffectiveBoundarySaver boundary_saver;
    GeneralBoundaryPointer bs{};
    fdtd::LaunchConfig launch_config{}, fwd_source_config{}, adj_source_config{};
    std::optional<typename Eq::State> state;
    std::optional<BoundaryScope> boundary_scope;
    BoundaryRuntime* boundary_runtime = nullptr;
    std::vector<float> adj_source_signs;
    int run_calls = 0;
};

template <class Eq>
BackwardOutput sg_generic_backward_bs(const BackwardInput& in)
{
    SgBackwardBsRunner<Eq> runner(in);
    return runner.run(in.bw_it_begin, in.bw_it_end, in.step_phase);
}

// ---- sg_generic_backward_ckpt (chunked segments with velocity carriers) ----

// N_VEL model-shaped carriers from adjoint_workspace slots [first, first + N_VEL).
// MANDATORY: the checkpoint modes' branch of
// cuda_layout.backward_workspace_shapes declares WS_CARRIERS + CKPT_COUNT /
// + RECURSIVE_COUNT slots, and sg_check_ckpt_workspace has already required
// the pool at exactly that size before this runs.
template <class Eq>
std::vector<torch::Tensor> sg_carriers(const BackwardInput& p, int first,
                                       const torch::Tensor& vp)
{
    std::vector<torch::Tensor> v;
    v.reserve(Eq::N_VEL);
    for (int c = 0; c < Eq::N_VEL; ++c)
        v.push_back(pool_required(p.adjoint_workspace, first + c, vp, "adjoint_workspace"));
    return v;
}

// One chunk: ``forward`` holds the chunk's start snapshot (loaded by the
// caller), ``seg_full`` the per-call velocity histories (Eq::seg_buffers,
// N_VEL x (max_rows, B, 1, grid)).
template <class Eq>
void sg_backward_segment(
    const BackwardInput& p,
    typename Eq::Models& models,
    typename Eq::State& state,
    typename Eq::Wavefield& forward,
    typename Eq::Wavefield& adjoint,
    typename Eq::Workspace& workspace,
    const std::vector<torch::Tensor>& seg_full,
    int start, int end,
    decltype(std::declval<typename Eq::CPML>().view()) cpml_view,
    SolverContext& solver,
    const torch::Tensor& source_fields,
    const torch::Tensor& receiver_fields,
    const std::vector<torch::Tensor>& next_segment_v,
    std::vector<torch::Tensor>& grads,
    std::vector<torch::Tensor>& prev_segment_next_v)
{
    const int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    const int segment_len = end - start;
    const auto adj_source_signs = Eq::adjoint_source_signs(p, receiver_fields);

    // This chunk's rows of the histories: row 0 = v(start), row k = v(start + k)
    // after the k-th replayed step.  Every row the reverse sweep reads
    // (1..segment_len, and export_seg_next_v's row 1) is written by this
    // chunk's replay first, so nothing an earlier chunk left behind reaches a
    // kernel -- reusing the per-call buffers is bit-identical to the
    // per-chunk zero tensors they replace.
    std::vector<torch::Tensor> seg;
    seg.reserve(seg_full.size());
    for (const auto& history : seg_full)
        seg.push_back(history.narrow(0, 0, segment_len + 1));

    Eq::save_seg_velocities(seg, forward, 0);
    auto for_view = Eq::view(forward);

    for (int it = start; it < end; ++it) {
        Eq::velocity_substep(state, for_view, cpml_view, solver);
        Eq::stress_substep(state, for_view, cpml_view, solver, nullptr);
        Eq::save_seg_velocities(seg, forward, it - start + 1);
        Eq::inject_forward_sources(state, solver, for_view, p, source_fields, it);
    }

    auto adj_view = Eq::view(adjoint);
    for (int it = end - 1; it >= start; --it) {
        Eq::fix_rho_grad_at_sources(state, solver, adj_view, p, source_fields, it, grads);
        Eq::inject_residuals(state, solver, adj_view, p, receiver_fields,
                             adj_source_signs, it, adjoint_nsrc);

        const int now_offset = it - start + 1;
        const int next_offset = now_offset + 1;
        typename Eq::VelPtrs vptrs = Eq::vel_ptrs_from_seg(
            seg, now_offset,
            next_offset <= segment_len ? next_offset : -1,
            next_segment_v);

        Eq::image_standalone(state, solver, adj_view, vptrs, grads);
        Eq::fix_rho_grad_at_receivers(state, solver, grads, vptrs, p,
                              receiver_fields, it, adjoint_nsrc);

        if (it == 0)
            continue;

        Eq::plain_adjoint_step(state, solver, adjoint, workspace, cpml_view);
    }

    Eq::export_seg_next_v(prev_segment_next_v, seg);
}

template <class Eq>
BackwardOutput sg_generic_backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    Eq::validate_backward(p, "ckpt");
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "checkpoint backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

    TORCH_CHECK(p.checkpoint_interval >= 1, "checkpoint_interval must be >= 1");
    TORCH_CHECK((int)p.checkpoints.size() == Eq::CKPT_NVAR, Eq::CKPT_COUNT_MSG);
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints, Eq::CKPT_NVAR, true, false,
        p.checkpoint_interval, p.checkpoint_steps, p.checkpoint_on_cpu,
        "backward_chunk", Eq::NAME);

    typename Eq::Models models = Eq::parse_models(p);
    const auto& vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);

    SolverContext solver = make_ctx<Eq>(p, d);
    Eq::setup_ctx(solver, p);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);
    Eq::init_aux_slabs(solver, adjoint);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml_view = cpml_tensor.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), d.B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), d.B);
    typename Eq::State state = Eq::make_state(p, d, models, launch_config,
                                              fwd_source_config, adj_source_config);

    // bind_grads: _c.py Wrapper.backward allocates grads_out for EVERY
    // backward (_gradient_buffers, one per model), so the hook takes the
    // bound accumulators here too.  It is handed the full BackwardInput
    // because equations whose gradient set is not {vp, vs, rho} need all of
    // p.models to check it.
    std::vector<torch::Tensor> grads;
    Eq::bind_grads(p, grads);
    sg_check_ckpt_workspace<Eq>(p, SgCarrierSlots<Eq>::CKPT_COUNT, "ckpt");
    typename Eq::Workspace workspace = Eq::make_workspace(p, vp);

    // The replay state, ONE for the whole call: forward_wavefields set 0 when
    // Python bound it (zeroed per backward call), else today's allocation
    // from the snapshot shapes.  Each chunk loads its snapshot straight into
    // it (chunk 0 zeroes it).  The hand-written driver kept a start_state
    // only to copy it into a fresh per-chunk forward -- the same bytes reach
    // the same kernels, minus one CKPT_NVAR-grid allocation and copy per chunk.
    typename Eq::Wavefield forward;
    Eq::bind_or_alloc_recon_ckpt(forward, p, vp);
    Eq::check_ckpt_aux_layout(forward, adjoint);
    // The velocity histories of one chunk (checkpoint_replay), bound at the
    // longest chunk and narrowed per chunk inside sg_backward_segment.
    const int max_rows = p.checkpoint_interval + 1;   // v(start) + one row per replayed step
    const auto seg = Eq::seg_buffers(p, vp, max_rows);
    // Cross-chunk velocity carriers (adjoint_workspace, SgCarrierSlots): the
    // younger chunk's v(end), read by the older chunk's imaging at its last
    // step.  Equations whose imaging has no velocity(t+1) term
    // (IMAGING_USES_NEXT_V == false) skip them entirely; their vel_ptrs_from_seg
    // hands the imaging null next-pointers instead.  next_segment_v must be
    // zero for the tail chunk (v(nt) does not exist): the pool is zero at
    // entry, see SgCarrierSlots.
    std::vector<torch::Tensor> next_segment_v, prev_segment_next_v;
    if (Eq::IMAGING_USES_NEXT_V) {
        next_segment_v = sg_carriers<Eq>(p, SgCarrierSlots<Eq>::NOW, vp);
        prev_segment_next_v = sg_carriers<Eq>(p, SgCarrierSlots<Eq>::NEXT, vp);
    }

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);
        if (chunk_id == 0) {
            checkpoint_runtime.zero_state(forward.state_tensors());
        } else {
            checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors());
        }
        sg_backward_segment<Eq>(p, models, state, forward, adjoint, workspace,
                                seg, start, end, cpml_view, solver,
                                source_fields, receiver_fields,
                                next_segment_v, grads, prev_segment_next_v);
        for (int c = 0; Eq::IMAGING_USES_NEXT_V && c < Eq::N_VEL; ++c)
            copy_tensor_cuda_async(next_segment_v[c], prev_segment_next_v[c]);
    }

    BackwardOutput out;
    out.grads = grads;
    return out;
}

// ---- sg_generic_backward_recursive_ckpt (per-step replay) ----

inline int sg_find_previous_checkpoint_idx(
    const int* checkpoint_steps, int num_saved_checkpoints, int target_time)
{
    int checkpoint_idx = -1;
    for (int i = 0; i < num_saved_checkpoints; ++i) {
        if (checkpoint_steps[i] < target_time)
            checkpoint_idx = i;
        else
            break;
    }
    return checkpoint_idx;
}

template <class Eq>
void sg_replay_forward_to_time(
    const BackwardInput& p,
    typename Eq::State& state,
    typename Eq::Wavefield& forward,
    std::vector<torch::Tensor>& current_v,
    std::vector<torch::Tensor>& next_v,
    int target_index,
    const int* checkpoint_steps,
    int num_saved_checkpoints,
    CheckpointRuntime& checkpoint_runtime,
    decltype(std::declval<typename Eq::CPML>().view()) cpml_view,
    SolverContext& solver,
    const torch::Tensor& source_fields)
{
    // next_v must be zero when target_index + 1 == nt (never captured below).
    // IMAGING_USES_NEXT_V == false equations skip both zeroings like their hand-written
    // replay did: next_v is empty and current_v is always overwritten at the
    // capture.
    if (Eq::IMAGING_USES_NEXT_V) {
        for (auto& t : current_v) zero_tensor_device_async(t);
        for (auto& t : next_v) zero_tensor_device_async(t);
    }

    const int checkpoint_idx = sg_find_previous_checkpoint_idx(
        checkpoint_steps, num_saved_checkpoints, target_index + 1);
    int start_time = 0;
    if (checkpoint_idx >= 0) {
        checkpoint_runtime.load(checkpoint_idx, forward.checkpoint_tensors());
        start_time = checkpoint_steps[checkpoint_idx];
    } else {
        checkpoint_runtime.zero_state(forward.state_tensors());
    }

    auto for_view = Eq::view(forward);
    for (int it = start_time; it < p.nt; ++it) {
        Eq::velocity_substep(state, for_view, cpml_view, solver);
        Eq::stress_substep(state, for_view, cpml_view, solver, nullptr);

        if (it == target_index) {
            Eq::capture_velocities(current_v, forward);
            // No velocity(t+1) imaging term: stop before this step's source
            // injection, exactly like the hand-written replay.
            if (!Eq::IMAGING_USES_NEXT_V)
                break;
        }
        if (it == target_index + 1) {
            Eq::capture_velocities(next_v, forward);
            break;
        }

        Eq::inject_forward_sources(state, solver, for_view, p, source_fields, it);

        if (it == target_index && target_index + 1 >= p.nt)
            break;
    }
}

template <class Eq>
BackwardOutput sg_generic_backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    Eq::validate_backward(p, "ckpt_recursive");
    TORCH_CHECK(!in.bw_stepped() && in.step_phase == 0 && in.cut_face_mask == 0,
                "checkpoint backward does not support bw_it_begin/bw_it_end, "
                "step_phase or cut_face_mask in v1");

    TORCH_CHECK((int)p.checkpoints.size() == Eq::CKPT_NVAR,
                Eq::CKPT_RECURSIVE_COUNT_MSG);

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints, Eq::CKPT_NVAR, true, true,
        p.checkpoint_interval, checkpoint_steps_cpu, p.checkpoint_on_cpu,
        "backward_recursive", Eq::NAME);

    typename Eq::Models models = Eq::parse_models(p);
    const auto& vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);

    SolverContext solver = make_ctx<Eq>(p, d);
    Eq::setup_ctx(solver, p);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);
    Eq::init_aux_slabs(solver, adjoint);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml_view = cpml_tensor.view();

    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), d.B);
    auto adj_source_config = fdtd::Geom::make(p.adjoint_sources_loc.size(1), d.B);
    typename Eq::State state = Eq::make_state(p, d, models, launch_config,
                                              fwd_source_config, adj_source_config);

    std::vector<torch::Tensor> grads;
    Eq::bind_grads(p, grads);
    sg_check_ckpt_workspace<Eq>(p, SgCarrierSlots<Eq>::RECURSIVE_COUNT, "ckpt_recursive");
    typename Eq::Workspace workspace = Eq::make_workspace(p, vp);

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    TORCH_CHECK(p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
                "checkpoint_count does not match checkpoint_steps");
    TORCH_CHECK(static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
                "checkpoint buffer is smaller than checkpoint_steps");

    const int* checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();
    // The replay state: forward_wavefields set 0 when Python bound it (zeroed
    // per backward call), else today's allocation from the snapshot shapes.
    typename Eq::Wavefield forward;
    Eq::bind_or_alloc_recon_ckpt(forward, p, vp);
    // The captured v(it) / v(it+1) the imaging reads (adjoint_workspace,
    // SgCarrierSlots); sg_replay_forward_to_time zeroes or overwrites them in
    // full before every imaging step, so their entry state is never read.
    std::vector<torch::Tensor> current_v = sg_carriers<Eq>(p, SgCarrierSlots<Eq>::NOW, vp);
    std::vector<torch::Tensor> next_v;
    if (Eq::IMAGING_USES_NEXT_V)
        next_v = sg_carriers<Eq>(p, SgCarrierSlots<Eq>::NEXT, vp);
    const auto adj_source_signs = Eq::adjoint_source_signs(p, receiver_fields);

    auto adj_view = Eq::view(adjoint);
    for (int it = p.nt - 1; it >= 0; --it) {
        Eq::fix_rho_grad_at_sources(state, solver, adj_view, p, source_fields, it, grads);
        Eq::inject_residuals(state, solver, adj_view, p, receiver_fields,
                             adj_source_signs, it, (int)p.adjoint_sources_loc.size(1));

        sg_replay_forward_to_time<Eq>(p, state, forward, current_v, next_v, it,
                                      checkpoint_steps, num_saved_checkpoints,
                                      checkpoint_runtime, cpml_view, solver,
                                      source_fields);

        typename Eq::VelPtrs vptrs = Eq::vel_ptrs_from_carriers(current_v, next_v);
        Eq::image_standalone(state, solver, adj_view, vptrs, grads);
        Eq::fix_rho_grad_at_receivers(state, solver, grads, vptrs, p, receiver_fields,
                              it, (int)p.adjoint_sources_loc.size(1));

        if (it == 0)
            continue;

        Eq::plain_adjoint_step(state, solver, adjoint, workspace, cpml_view);
    }

    BackwardOutput out;
    out.grads = grads;
    return out;
}
} // namespace eqdrv
