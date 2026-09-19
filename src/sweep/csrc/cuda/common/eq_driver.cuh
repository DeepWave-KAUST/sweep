// Shared per-equation driver skeleton.
//
// Every equation directory used to hand-copy a ~300-line forward driver and a
// ~1000-line four-mode backward driver; they were 60-80% line-identical, and
// cross-cutting abilities (the stepped it_begin/it_end range that domain
// decomposition needs, phase-split launches, boundary-tail truncation) existed
// only in the copies that happened to have them.  This header owns that
// skeleton ONCE, as ``template <class Eq>`` drivers; an equation supplies a
// traits struct (constants + composite launch hooks) and 1-line entry points.
//
// Hook granularity is deliberately COARSE — one hook per in-step compound
// operation, not per kernel.  The three families disagree on the in-step
// ORDER (e.g. 2-D acoustic images the boundary-saving gradient after the
// forward source injection and swap, 3-D acoustic before the injection, VRZ
// injects before the restore), and that order is bit-load-bearing.  The
// template owns what is genuinely identical: input validation, stepped/phase
// bookkeeping, buffer binding with legacy fallback allocation, Boundary- and
// Checkpoint-runtime orchestration, the time loops, and output packing.
//
// Bit-exactness contract: this skeleton is a line-faithful transcription of
// acoustic2d's drivers (the reference, gated by bitgate tiers A/B/C/T and
// ddgate).  Physics kernels are not touched by the migration.  Where another
// equation's copy disagreed with acoustic2d in loop structure, the difference
// lives in that equation's hooks, never in a per-equation branch here.
//
// Buffer ownership: every buffer these drivers read or write is allocated by
// the propagator (src/sweep/propagator/_c.py, the only place that builds a
// ForwardInput / BackwardInput) and bound on the input struct.  A binding the
// propagator makes unconditionally for every equation and mode reaching a site
// is REQUIRED here -- a missing one is a TORCH_CHECK naming the cuda_layout
// field that declares it, never a quiet driver-side allocation.  The two
// conditional bindings keep their fallback and say why at the site:
// ``illum_out`` (bound only when the caller asked for illumination) and the
// ADCIG cube (allocated by init_rtm_output for an ADCIG-only backward).
//
// ---------------------------------------------------------------------------
// HOOK TIMING MAP — read this before any equation's driver_traits.cuh.
// Per entry point, the traits hooks fire in exactly this order; everything
// not named here is shared runtime (checkpoint / boundary machinery).
// Prologue of every entry (in call order): validate_forward / (backward:
// check_stepped + validate_backward + bind_backward_outputs +
// rtm gate), bind_or_alloc_* wavefields, alloc_cpml, setup_ctx,
// init_aux_slabs, make_state, make_bwd_workspace.
//
// generic_forward — per it in [it_begin, it_end):
//   launch_step_range          the whole per-range stencil step (air-clear
//                              prepass included); DD phase 1 = the cut-side
//                              M-wide strips, phase 2 = strict complement,
//                              unphased = (0, nx)
//   save_boundary_fwd          BS strips (when use_boundary_saving)
//   inject_source_fwd          source injection
//   record                     receiver sampling
//   rotate_buffers             u_pre/u_now buffer-role rotation
//   capture_allt               deferred u_allt snapshot (only 3-D uses it)
//   <checkpoint save>          shared runtime, not a hook
//   after the loop: save_last_state (final u pair for backward_bs)
//
// generic_backward (full storage) — per reverse it:
//   adjoint_step               adjoint stencil; with HAS_FUSED_FULL_IMG the
//                              imaging of u_forward[it+1] fuses into it
//   inject_adjoint_source      residual injection
//   rotate_adjoint_buffers     adjoint buffer-role rotation
//   accumulate_source_grad     grad_wavelet sampling
//   image_step                 standalone imaging / RTM+illumination taps
//                              (skipped when fused, except for RTM)
//   after the loop (fused only): one trailing image_step at it == 0.
//
// generic_backward_bs — per reverse it, floor max(max(it_lo, 1), bs_stop):
//   adjoint_step / inject_adjoint_source / rotate_adjoint_buffers /
//   accumulate_source_grad     same four as full mode
//   bs_recon_step              reconstruction (un-inject, NOPML reverse,
//                              strip restore) + gradient imaging, in the
//                              equation's exact order
//   bs_rtm_tap                 RTM / illumination tap
//   before the loop (first segment): seed_reconstruction from u_last_two;
//   after the loop (BS_HAS_IT0_ADJOINT_TAIL): the four adjoint hooks once at it == 0.
//
// generic_backward_ckpt — per chunk: replay then reverse:
//   replay:  replay_step / inject_source_fwd / rotate_recon_buffers
//   reverse: adjoint_step / inject_adjoint_source / rotate_adjoint_buffers /
//            accumulate_source_grad / image_step
//
// generic_backward_recursive_ckpt — bisection over each ckpt segment; a
//   leaf runs one replay triple, then the reverse-five of ckpt mode with
//   the imaging fed from the leaf's scratch u.
//
// Propagator-owned buffers of the two checkpoint skeletons (see the
// AcousticCkptReplaySlot / AcousticRecursiveWorkspaceSlot enums): the replay STATE rides
// p.forward_wavefields as K sets of Eq::CKPT_STATE_COUNT tensors (set 0 via
// bind_or_alloc_recon_ckpt; recursive mode adds one scratch set per bisection
// level via bind_or_alloc_recursive_scratch), the chunk history rides
// p.checkpoint_replay, the leaf scratch p.adjoint_workspace.
// ---------------------------------------------------------------------------
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <optional>

#include "common.cuh"
#include "context.h"
#include "checkpoint_runtime.cuh"
#include "cudautils.h"
#include "boundarysaver.cuh"
#include "boundary_runtime.cuh"
#include "boundary/session.cuh"
#include "wavetypes.h"
#include "../launch/config.h"

namespace eqdrv {

// ------------------------------------------------------------------------- //
// Small shared helpers
// ------------------------------------------------------------------------- //

struct Dims {
    int N, C, nz, ny, nx, B;
};

template <int NDIM>
inline Dims read_dims(const torch::Tensor& model)
{
    Dims d;
    d.N = model.size(0);
    d.C = model.size(1);
    if constexpr (NDIM == 2) {
        d.nz = model.size(2);
        d.ny = 0;
        d.nx = model.size(3);
    } else {
        d.nz = model.size(2);
        d.ny = model.size(3);
        d.nx = model.size(4);
    }
    d.B = d.N * d.C;
    return d;
}

template <int NDIM>
inline fdtd::LaunchConfig wave_config(const Dims& d)
{
    if constexpr (NDIM == 2)
        return fdtd::Wave2D::make(d.nx, d.nz, d.B);
    else
        return fdtd::Wave3D::make(d.nx, d.ny, d.nz, d.B);
}

// SolverContext construction is shared; the per-family extras (topography
// rows, per-edge free-surface faces, APM flags) go through Eq::setup_ctx.
template <class Eq, class P>
inline SolverContext make_ctx(const P& p, const Dims& d)
{
    if constexpr (Eq::NDIM == 2) {
        return SolverContext{2, d.nx, 0, d.nz, d.B, p.dt, p.nt, p.M, p.abcn,
                             p.free_surface,
                             p.lap_coes.template data_ptr<float>(),
                             p.grad_coes.template data_ptr<float>(),
                             p.spacing[0], 0.f, p.spacing[1]};
    } else {
        return SolverContext{3, d.nx, d.ny, d.nz, d.B, p.dt, p.nt, p.M, p.abcn,
                             p.free_surface,
                             p.lap_coes.template data_ptr<float>(),
                             p.grad_coes.template data_ptr<float>(),
                             p.spacing[0], p.spacing[1], p.spacing[2]};
    }
}

inline int stencil_order(int M)
{
    return (M <= 4) ? 2 * M : -1;
}

// ------------------------------------------------------------------------- //
// generic_forward
// ------------------------------------------------------------------------- //

// Persistent forward runner: the whole prologue (validation, binding,
// runtime construction) runs ONCE in the constructor; run() is only the
// time loop.  The monolithic generic_forward below is construct + one run,
// so this is the exact code path the bit gates exercise.  Reuse (a second
// run() call) requires gpu-direct boundary storage and no checkpointing --
// the only modes whose cross-call state lives entirely in Python-bound
// buffers.
template <class Eq>
class GenericForwardRunner final : public IForwardRunner {
public:
    explicit GenericForwardRunner(const ForwardInput& in)
        : p(in)
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());

        auto vp = p.models[0];
        d = read_dims<Eq::NDIM>(vp);

        nsrc = p.sources_loc.size(1);
        nrec = p.receivers_loc.size(1);

        ctx_.emplace(make_ctx<Eq>(p, d));
        SolverContext& ctx = *ctx_;
        Eq::setup_ctx(ctx, p);
        // Cut-aware physical bounds (0 = single domain → legacy per-edge pad + M).
        ctx.set_cut_mask(p.cut_face_mask);

        const int it0 = p.it_begin;
        const int it1 = (p.it_end < 0) ? static_cast<int>(p.nt) : p.it_end;
        check_run_args(it0, it1, p.step_phase);
        const bool stepped = (it0 != 0) || (it1 != static_cast<int>(p.nt));

        cut_x_lo = (p.cut_face_mask & 1) != 0;
        cut_x_hi = (p.cut_face_mask & 2) != 0;

        // On a continuation call the internal allocate() would silently zero the
        // propagation state — the caller must keep binding the same tensors.
        TORCH_CHECK(it0 == 0 || !p.wavefields.empty(),
                    "stepped continuation (it_begin>0) requires Python-bound wavefields");
        Eq::bind_or_alloc_forward(wavefield, p, vp);
        Eq::init_aux_slabs(ctx, wavefield);

        Eq::alloc_cpml(cpml_tensor, p);
        cpml = cpml_tensor.view();

        // An empty tensor counts as unbound: nothing could be recorded into it.
        TORCH_CHECK(!stepped || (p.record_out.defined() && p.record_out.numel() > 0),
                    "stepped forward requires record_out bound from Python");
        // Shape comes from the equation, like allt_shape two statements below.
        // The three equations on this skeleton all return {N, nrec, nt}; the
        // literal lived here because they were the only ones. A multi-field
        // receiver record -- elastic_tti_2nd2d writes {nfield, B, nrec, nt},
        // and sg_driver.cuh:151 already allocates that shape -- has no place to
        // say so while the skeleton decides. Everything downstream is already
        // shape-blind: the per-step Eq::record hook, out.record, the record_out
        // check (contiguity and trailing nt only), and
        // _c.py::_cuda_record_to_canonical, which dispatches on syn.ndim.
        // MANDATORY: every equation on this skeleton declares
        // cuda_layout.record_shape (acoustic2d / acoustic3d / acoustic_vrz2d),
        // so the propagator allocates the record and binds it as record_out on
        // EVERY call (_c.py Wrapper.forward: ``if cp.record_shape is not None:
        // params.record_out = _record_buffer(...)``) -- no supported path
        // arrives here unbound, and the driver keeps no allocation for one.
        TORCH_CHECK(p.record_out.defined(),
                    Eq::NAME, "/forward requires the propagator-bound record_out "
                    "(cuda_layout.record_shape)");
        record = bound_required(p.record_out, Eq::record_shape(d, p), vp.options(), "record_out");

        // Wavefields for all timestep
        if (p.save_all_wavefields) {
            TORCH_CHECK(!stepped || p.u_allt_out.defined(),
                        "stepped + save_all_wavefields requires u_allt_out bound from Python");
            // MANDATORY under save_all_wavefields: the propagator binds the
            // history on exactly the same condition (_c.py Wrapper.forward:
            // ``if save_all_wavefields and cp.u_allt_shape is not None``), and
            // cuda_layout.save_all_shape is declared by every equation on this
            // skeleton, so the flag implies the binding.
            TORCH_CHECK(p.u_allt_out.defined(),
                        Eq::NAME, "/forward with save_all_wavefields requires the "
                        "propagator-bound u_allt_out (cuda_layout.save_all_shape)");
            u_allt = bound_required(p.u_allt_out, Eq::allt_shape(d, p.nt), vp.options(), "u_allt_out");
        }

        Eq::validate_forward(p);

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
            it0
        );

        save_width = Eq::save_width(p.abcn, p.M);
        // The internal full-storage fallback ring is per-call; segments after the
        // first would lose everything saved before them.
        if (stepped && p.use_boundary_saving)
            TORCH_CHECK(!p.boundary_gpu.empty(),
                        "stepped forward with boundary saving requires Python-bound boundary_gpu");
        staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
        if (staged_boundary)
            boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, ctx, vp,
                                    save_width, Eq::BS_LAST_TWO_NVAR, true, false,
                                    p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
                                    p.last_two, p.use_pinned_memory, Eq::TANGENT_PAD * p.M, p.boundary_staging);
        else
            boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, ctx, vp,
                                    save_width, Eq::BS_LAST_TWO_NVAR, true, true,
                                    1, {}, p.boundary_gpu,
                                    p.last_two, p.use_pinned_memory, Eq::TANGENT_PAD * p.M, p.boundary_staging);
        bs = boundary_saver.view();

        launch_config = wave_config<Eq::NDIM>(d);
        source_config = fdtd::Geom::make(nsrc, d.B);
        record_config = fdtd::Geom::make(nrec, d.B);

        state.emplace(Eq::make_state(p, d, ctx, launch_config,
                                     source_config, record_config));

        // Boundary tail truncation: with boundary_tail_steps = K > 0 only the
        // last K steps' boundary strips are saved; the runtime and the Python
        // buffers work in shifted "saved-step" coordinates [0, K).  bs_it0 = 0
        // when disabled, making every shift below a no-op (bit-exact legacy).
        // Stepped/DD segments compose transparently: ``it`` is the GLOBAL step
        // index, so the save guard and shift never look at the segment bounds;
        // the Python-bound boundary_gpu ring (mandatory under stepped) is
        // allocated tail-shrunk by _ensure_boundary_buffers(nt_saved=...).
        bs_it0 = (p.use_boundary_saving && p.boundary_tail_steps > 0)
            ? std::max(0, (int)p.nt - p.boundary_tail_steps) : 0;
        // A Python-owned BoundarySession, when bound, keeps the copy stream and
        // the ring events alive ACROSS calls.  Under DD every time step is a
        // separate entry into the extension, so a per-call copy stream can never
        // hold a transfer in flight and transfer_interval/ring_buffers overlap
        // nothing.  Without a session BoundaryScope builds both locally and the
        // behaviour is exactly the old per-call path.  (dev 848a100.)
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
    }

    ForwardOutput run(int run_it_begin, int run_it_end, int run_step_phase) override
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());
        SolverContext& ctx = *ctx_;
        ForwardOutput out;

        const int it0 = run_it_begin;
        const int it1 = (run_it_end < 0) ? static_cast<int>(p.nt) : run_it_end;
        check_run_args(it0, it1, run_step_phase);
        const int phase = run_step_phase;
        if (run_calls++ > 0)
            TORCH_CHECK(!p.use_checkpoint && !staged_boundary,
                        "persistent stepped runner reuse requires gpu-direct "
                        "boundary storage and no checkpointing");

        float* u_thist = nullptr;

        for (int it = it0; it < it1; ++it) {

            auto view = wavefield.view();

            u_thist = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

            // Ranged stencil launch over x in [xb, xe); (0, nx) reproduces the
            // legacy full launch bit-identically (same grid dims, x_base = 0).
            // The hook owns the equation's whole per-range step (air-clear
            // prepass included where the equation has one).
            if (phase == 1) {
                // Boundary phase: ONLY the cut-adjacent M-wide physical edge
                // strips — exactly what the halo exchange sends.
                if (cut_x_lo)
                    Eq::launch_step_range(*state, ctx, ctx.phys_x0(), ctx.phys_x0() + p.M,
                                          view, p.save_all_wavefields, u_thist, cpml);
                if (cut_x_hi)
                    Eq::launch_step_range(*state, ctx, ctx.phys_x1() - p.M, ctx.phys_x1(),
                                          view, p.save_all_wavefields, u_thist, cpml);
            } else if (phase == 2) {
                // Interior phase: the strict complement of the phase-1 strips
                // (no overlap — re-running a strip cell would double-advance
                // its CPML psi double-buffer write).
                Eq::launch_step_range(*state, ctx,
                                      cut_x_lo ? ctx.phys_x0() + p.M : 0,
                                      cut_x_hi ? ctx.phys_x1() - p.M : d.nx,
                                      view, p.save_all_wavefields, u_thist, cpml);
            } else {
                Eq::launch_step_range(*state, ctx, 0, d.nx,
                                      view, p.save_all_wavefields, u_thist, cpml);
            }

            if (phase == 1)
                continue;   // no boundary saving / source / record / swap / ckpt

            if (p.use_boundary_saving && it >= bs_it0) {
                Eq::save_boundary_fwd(*boundary_runtime, *state, ctx, view,
                                      it - bs_it0, (int)p.nt - bs_it0,
                                      bs, save_width);
            }

            Eq::inject_source_fwd(*state, ctx, view, p, it, nsrc);

            Eq::record(*state, ctx, view, record, p, it, nrec);

            Eq::rotate_buffers(wavefield);

            Eq::capture_allt(u_allt, wavefield, it);

            checkpoint_runtime->save_forward(it, static_cast<int>(p.nt),
                                             wavefield.checkpoint_tensors());

        }

        // Save the last state for backward (only once the final segment has run;
        // mid-run segments leave it untouched).  Phase 1 has not swapped yet —
        // roles would be wrong; phase 2 of the same step does the copy.
        if (p.use_boundary_saving && it1 == static_cast<int>(p.nt) && phase != 1) {
            Eq::save_last_state(boundary_saver, wavefield);
        }

        // With a persistent session the trailing sync belongs to the phase, not
        // to this one call -- Python drives it via session.finish(). (dev 848a100.)
        if (boundary_scope->owns())
            boundary_runtime->synchronize();

        out.wavefield = u_allt;
        out.last_two = boundary_saver.last_two_t;
        out.record = record;

        return out;
    }

private:
    void check_run_args(int it0, int it1, int phase) const
    {
        const SolverContext& ctx = *ctx_;
        TORCH_CHECK(0 <= it0 && it0 <= it1 && it1 <= static_cast<int>(p.nt),
                    "stepped forward: require 0 <= it_begin <= it_end <= nt, got [",
                    it0, ", ", it1, ") with nt=", p.nt);
        // ---- DD phase-split step (comm/compute overlap) ----
        if (phase != 0) {
            TORCH_CHECK(phase == 1 || phase == 2,
                        "step_phase must be 0 (legacy), 1 (boundary strips) or 2 (interior)");
            TORCH_CHECK(it1 == it0 + 1,
                        "phased forward (step_phase != 0) drives a single step: "
                        "require it_end == it_begin + 1, got [", it0, ", ", it1, ")");
            TORCH_CHECK(p.cut_face_mask != 0,
                        "phased forward requires cut_face_mask != 0");
            TORCH_CHECK((p.cut_face_mask & ~0x3) == 0,
                        "phased forward v1 supports x-face cuts only (bits 0/1), got ",
                        p.cut_face_mask);
            TORCH_CHECK(ctx.phys_x1() - ctx.phys_x0() >= 2 * p.M,
                        "tile too narrow for phase-split strips: nx_phys=",
                        ctx.phys_x1() - ctx.phys_x0(), " < 2M=", 2 * p.M);
        }
    }

    // Declaration order == construction order; destruction runs in reverse,
    // matching the hand-written function's stack unwind.
    ForwardInput p;
    Dims d;
    int nsrc = 0, nrec = 0;
    std::optional<SolverContext> ctx_;
    bool cut_x_lo = false, cut_x_hi = false;
    typename Eq::Wavefield wavefield;
    typename Eq::CPML cpml_tensor;
    decltype(std::declval<typename Eq::CPML>().view()) cpml;
    torch::Tensor record;
    torch::Tensor u_allt;
    std::optional<CheckpointRuntime> checkpoint_runtime;
    int save_width = 0;
    bool staged_boundary = false;
    EffectiveBoundarySaver boundary_saver;
    GeneralBoundaryPointer bs{};
    fdtd::LaunchConfig launch_config{}, source_config{}, record_config{};
    std::optional<typename Eq::State> state;
    std::optional<BoundaryScope> boundary_scope;
    int bs_it0 = 0;
    BoundaryRuntime* boundary_runtime = nullptr;
    int run_calls = 0;
};

template <class Eq>
ForwardOutput generic_forward(const ForwardInput& in)
{
    GenericForwardRunner<Eq> runner(in);
    return runner.run(in.it_begin, in.it_end, in.step_phase);
}


// Validate the stepped-backward segment fields (bw_it_begin/bw_it_end).
// ``need_recon`` is true for boundary-saving mode, where the reconstruction
// wavefield list must be Python-owned to survive segments.
// ``bw_it_begin``/``bw_it_end``/``step_phase`` are explicit so a persistent
// runner can re-validate each run() with that call's range; the monolithic
// entries pass the input-struct fields, reproducing the legacy behaviour.
template <class Eq>
void check_stepped_backward(const BackwardInput& p, bool need_recon,
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
    // No second-order backward in this family implements a phase split; a
    // phased schedule (e.g. the VRZ coupling exchange) reaching an equation
    // without one must fail loudly, not run un-phased.
    TORCH_CHECK(step_phase == 0,
                Eq::NAME, " backward does not implement step_phase (got ",
                step_phase, ")");
    TORCH_CHECK(need_recon || p.cut_face_mask == 0,
                "domain-decomposed backward (cut_face_mask) is boundary-saving "
                "only; the full-storage path does not support DD (use backward_bs)");
    if (need_recon && p.cut_face_mask != 0) {
        TORCH_CHECK(!p.boundary_on_disk,
                    "domain-decomposed backward_bs (cut_face_mask) supports "
                    "gpu-direct or cpu boundary storage only "
                    "(boundary_on_disk unsupported in v1)");
    }
    const bool stepped = (it_hi < static_cast<int>(p.nt)) || (it_lo > 0);   // == bw_stepped()
    if (!stepped)
        return;
    TORCH_CHECK((int)p.adjoint_wavefields.size() == Eq::ADJ_WF_COUNT,
                "stepped backward requires the ", Eq::ADJ_WF_COUNT,
                "-tensor adjoint wavefield list bound from Python");
    TORCH_CHECK(p.grads_out.size() == p.models.size() + 1,
                "stepped backward requires Python-bound grads_out "
                "(slot 0 = grad_wavelet, then one per model)");
    TORCH_CHECK(p.illum_out.size() == 2,
                "stepped backward requires Python-bound illum_out "
                "{source_illumination, receiver_illumination}");
    if (need_recon) {
        TORCH_CHECK((int)p.forward_wavefields.size() == Eq::RECON_WF_COUNT,
                    "stepped backward_bs requires the ", Eq::RECON_WF_COUNT,
                    "-tensor reconstruction wavefield list bound from Python");
        TORCH_CHECK(!p.boundary_on_disk,
                    "stepped backward_bs supports gpu-direct or cpu boundary "
                    "storage only (boundary_on_disk unsupported in v1)");
        TORCH_CHECK(p.cut_face_mask != 0 || !p.boundary_on_cpu,
                    "stepped backward_bs cpu boundary staging requires a DD cut "
                    "mask (cut_face_mask != 0); single-tile cpu staging is "
                    "unsupported here (use gpu-direct or a monolithic backward)");
    }
}

inline void init_rtm_output(RTMOutput& out, const torch::Tensor& vp,
                            bool want_adcig, int nlag)
{
    out.source_illumination = torch::zeros_like(vp);
    out.receiver_illumination = torch::zeros_like(vp);
    if (want_adcig) {
        std::vector<int64_t> shape{(long)nlag};
        for (auto s : vp.sizes()) shape.push_back(s);
        out.adcig = torch::zeros(shape, vp.options());
    }
}

// Acoustic-flavoured output binding: grads = {grad_wavelet, grad_model} with
// Python-owned accumulators on the stepped path, plus the RTM/illumination
// buffers.  Traits whose equation has no wavelet gradient or illumination
// (VRZ) implement their own bind hook instead of calling this helper.
template <class Eq>
void acoustic_bind_backward_outputs(const BackwardInput& p,
                                    std::vector<torch::Tensor>& grads,
                                    RTMOutput& illumination,
                                    bool want_adcig)
{
    // MANDATORY: the propagator builds grads_out on EVERY backward, in every
    // memory mode (_c.py Wrapper.backward: ``params.grads_out =
    // _gradient_buffers(cp.grads_out_has_wavelet, ...)``, unconditional), and
    // the equations on this hook declare cuda_layout.grads_out_has_wavelet, so
    // the list is models.size()+1 long with grad_wavelet first.
    TORCH_CHECK(!p.grads_out.empty(),
                Eq::NAME, " backward requires the propagator-bound grads_out "
                "(cuda_layout.grads_out_has_wavelet: slot 0 = grad_wavelet, "
                "then one per model)");
    TORCH_CHECK(p.grads_out.size() == p.models.size() + 1,
                "grads_out must hold models.size()+1 tensors "
                "(slot 0 = grad_wavelet)");
    torch::Tensor grad_wavelet = p.grads_out[0];
    torch::Tensor grad = p.grads_out[1];
    // OPTIONAL, deliberately: the propagator binds illum_out only when the
    // caller asked for illumination (_c.py Wrapper.backward: ``if
    // params.compute_illumination and cp.illum_nvar > 0``), and it never binds
    // the ADCIG cube at all -- so an ADCIG-only backward (compute_adcig with
    // compute_illumination off) legitimately arrives with an empty list and the
    // allocation below is the only way it gets its buffers.
    if (!p.illum_out.empty()) {
        TORCH_CHECK(p.illum_out.size() == 2,
                    "illum_out must be {source_illumination, receiver_illumination}");
        illumination.source_illumination = p.illum_out[0];
        illumination.receiver_illumination = p.illum_out[1];
        TORCH_CHECK(!p.compute_adcig,
                    "compute_adcig is not supported on the segmented "
                    "(stepped / domain-decomposed) backward: the ADCIG cube "
                    "has no cross-segment accumulator. Run ADCIG on the "
                    "single-segment backward instead.");
    } else if (p.compute_illumination || p.compute_adcig) {
        // Only when something will read them.  ``rtm_out_full`` / ``rtm_out_bs``
        // already return nullptr on exactly this predicate, so the three
        // ``zeros_like(vp)`` fields were allocated and memset on EVERY backward
        // -- a plain FWI gradient included -- for kernels that never ran and a
        // Python side that drops them (``compute_illumination`` is itself
        // derived from whether Python allocated a real buffer).  Left
        // undefined, ``pack_outputs`` hands Python None and its
        // ``isinstance(..., torch.Tensor)`` guards skip the copy.
        init_rtm_output(illumination, p.models[0],
                        want_adcig && p.compute_adcig, 2 * p.adcig_max_lag + 1);
    }
    grads = {grad_wavelet, grad};
}

// ---- generic_backward (full storage) ----
template <class Eq>
BackwardOutput generic_backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    check_stepped_backward<Eq>(in, /*need_recon=*/false,
                               in.bw_it_begin, in.bw_it_end, in.step_phase);
    Eq::validate_backward(in, /*need_recon=*/false);
    BackwardOutput out;
    std::vector<torch::Tensor> grads;
    RTMOutput illumination;
    Eq::bind_backward_outputs(in, grads, illumination,
                              /*want_adcig=*/Eq::ADCIG_IN_FULL_MODES);
    RTMOutput* rtm_out = Eq::rtm_out_full(in, illumination);

    const auto& p = in;
    auto vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);
    auto forward_source_config = fdtd::Geom::make(forward_nsrc, d.B);

    SolverContext ctx = make_ctx<Eq>(p, d);
    Eq::setup_ctx(ctx, p);
    ctx.set_cut_mask(0);  // full-storage path is DD-free (DD is backward_bs only)
    Eq::init_aux_slabs(ctx, adjoint);

    typename Eq::State state = Eq::make_state(p, d, ctx, launch_config,
                                              forward_source_config,
                                              adj_source_config);
    typename Eq::BwdWorkspace ws = Eq::make_bwd_workspace(p, state, ctx, adjoint);

    const bool fused_img = Eq::HAS_FUSED_FULL_IMG
        && !p.bw_stepped() && p.cut_face_mask == 0;
    float* fuse_grad = fused_img ? Eq::fused_grad_ptr(grads) : nullptr;

    // ``it`` goes to image_step as well as the pointer. One time level and one
    // pointer is enough for an equation whose gradient is a pointwise product at
    // step it, which is every equation on this skeleton today -- they all ignore
    // the argument. It is not enough for a second-order ANISOTROPIC gradient,
    // which reads three consecutive levels and has to substitute a zero field
    // for the ones that do not exist yet at it < 2; deriving those by pointer
    // arithmetic off u_forward is possible, but knowing WHEN to clamp is not.
    for (int it = p.bw_begin() - 1; it >= p.bw_it_end; --it) {
        auto adj_view = adjoint.view();
        const float* img_fwd = (fused_img && fuse_grad != nullptr && it + 1 < p.nt)
                             ? Eq::u_forward_ptr(p, it + 1) : nullptr;
        Eq::adjoint_step(state, ctx, adj_view, cpml, ws,
                         img_fwd, img_fwd ? fuse_grad : nullptr);
        Eq::inject_adjoint_source(state, ctx, adj_view, p, it, adjoint_nsrc, ws);
        Eq::rotate_adjoint_buffers(adjoint);
        Eq::accumulate_source_grad(state, ctx, adjoint, p, grads,
                                   it, forward_nsrc);
        if (!fused_img || rtm_out != nullptr) {
            Eq::image_step(state, ctx, Eq::u_forward_ptr(p, it), it, adjoint,
                           fused_img ? nullptr : &grads, rtm_out, ws);
        }
    }
    if (fused_img && fuse_grad != nullptr) {
        Eq::image_step(state, ctx, Eq::u_forward_ptr(p, 0), 0, adjoint,
                       &grads, nullptr, ws);
    }

    Eq::pack_outputs(out, grads, illumination);
    return out;
}

// ---- generic_backward_bs ----
// Persistent boundary-saving backward runner: prologue once in the
// constructor, run(bw_it_begin, bw_it_end, step_phase) is the reverse loop.
// seed_reconstruction stays in run() gated on that run's range (first
// segment only), exactly like the hand-written per-call gating; moving it
// past the scratch/runtime construction crosses ops that never touch the
// reconstruction wavefield, so the value sequence is identical.  Reuse (a
// second run()) requires gpu-direct boundary storage.
template <class Eq>
class GenericBackwardBsRunner final : public IBackwardRunner {
public:
    explicit GenericBackwardBsRunner(const BackwardInput& in)
        : p(in)
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());

        check_stepped_backward<Eq>(p, /*need_recon=*/true,
                                   p.bw_it_begin, p.bw_it_end, p.step_phase);
        Eq::validate_backward(p, /*need_recon=*/true);

        auto vp = p.models[0];
        d = read_dims<Eq::NDIM>(vp);
        adjoint_nsrc = p.adjoint_sources_loc.size(1);
        forward_nsrc = p.forward_sources_loc.size(1);

        ctx_.emplace(make_ctx<Eq>(p, d));
        SolverContext& ctx = *ctx_;
        Eq::setup_ctx(ctx, p);
        // DD: skip cut faces in the strip restore, the seed rim-zeroing, the
        // NOPML exclusion band and the fused-adjoint pure_interior test.
        ctx.set_cut_mask(p.cut_face_mask);

        Eq::bind_or_alloc_adjoint(adjoint, p, vp);
        Eq::bind_or_alloc_recon(forward, p, vp);
        Eq::init_aux_slabs(ctx, adjoint);

        Eq::bind_backward_outputs(p, grads, illumination, /*want_adcig=*/true);
        bs_rtm = Eq::rtm_out_bs(p, illumination);

        Eq::alloc_cpml(cpml_tensor, p);
        cpml = cpml_tensor.view();

        save_width = Eq::save_width(p.abcn, p.M);
        staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
        // ``bs.last_two`` is never read in the backward -- the reverse seeds come
        // straight from ``p.u_last_two``.  Passing {} made allocate_last_two take
        // its self-allocating branch and build a full two-wavefield FP32 buffer on
        // EVERY call, in HOST memory on the staged path.  Harmless for a monolithic
        // backward, ruinous under DD/stepped (one call per time step): a production
        // 3-D run went 1760 -> 166 s/iteration once the backward bound the tensor
        // instead.  Bind it here for every equation on this skeleton.
        // (dev 4290248, which fixed the pre-template acoustic3d/backward.cu.)
        const torch::Tensor& last_two_bound = p.u_last_two;
        if (staged_boundary) {
            boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, ctx, vp, save_width,
                                    Eq::BS_LAST_TWO_NVAR, true, false,
                                    p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
                                    last_two_bound, p.use_pinned_memory, Eq::TANGENT_PAD * p.M, p.boundary_staging);
        } else {
            boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, ctx, vp, save_width,
                                    Eq::BS_LAST_TWO_NVAR, true, true, 1, {}, p.boundary_gpu,
                                    last_two_bound, p.use_pinned_memory, Eq::TANGENT_PAD * p.M, p.boundary_staging);
            if (p.boundary_gpu.empty())
                boundary_saver.load_from_vector(p.u_boundary, vp);
        }
        bs = boundary_saver.view();

        launch_config = wave_config<Eq::NDIM>(d);
        fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
        adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);

        state.emplace(Eq::make_state(p, d, ctx, launch_config,
                                     fwd_source_config,
                                     adj_source_config));
        ws.emplace(Eq::make_bwd_workspace(p, *state, ctx, adjoint));

        bs_scratch.emplace(Eq::make_bs_scratch(p, vp));

        // Same persistent-session handling as the forward; see there. (dev 848a100.)
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
        TORCH_CHECK(p.boundary_tail_steps >= 0, "boundary_tail_steps must be >= 0");
        bs_it0 = (p.boundary_tail_steps > 0)
            ? std::max(0, (int)p.nt - p.boundary_tail_steps) : 0;
        bs_stop = bs_it0 > 0 ? bs_it0 + 1 : 0;
    }

    BackwardOutput run(int bw_it_begin, int bw_it_end, int run_step_phase) override
    {
        c10::cuda::CUDAGuard device_guard(p.models[0].device());
        SolverContext& ctx = *ctx_;
        BackwardOutput out;

        check_stepped_backward<Eq>(p, /*need_recon=*/true,
                                   bw_it_begin, bw_it_end, run_step_phase);
        const int it_hi = (bw_it_begin < 0) ? static_cast<int>(p.nt) : bw_it_begin;
        const int it_lo = bw_it_end;
        const bool first_segment = (it_hi == static_cast<int>(p.nt));

        if (run_calls++ > 0)
            TORCH_CHECK(!staged_boundary,
                        "persistent stepped runner reuse requires gpu-direct "
                        "boundary storage");

        // FIRST segment only: re-running the seeding (last-state copy + any
        // rim-zeroing) mid-stream would clobber the carried reconstruction state.
        if (first_segment)
            Eq::seed_reconstruction(*state, ctx, forward, p);

        boundary_runtime->prefetch_initial_backward_chunk((int)p.nt - bs_it0,
                                                          it_hi - bs_it0);

        for (int it = it_hi - 1; it >= std::max(std::max(it_lo, 1), bs_stop); --it) {
            auto adj_view = adjoint.view();

            Eq::adjoint_step(*state, ctx, adj_view, cpml, *ws, nullptr, nullptr);
            Eq::inject_adjoint_source(*state, ctx, adj_view, p, it, adjoint_nsrc, *ws);
            Eq::rotate_adjoint_buffers(adjoint);
            Eq::accumulate_source_grad(*state, ctx, adjoint, p, grads,
                                       it, forward_nsrc);

            // Reconstruction + gradient imaging, in this equation's exact order.
            Eq::bs_recon_step(*state, ctx, forward, adjoint, *boundary_runtime,
                                bs, save_width, cpml, p, grads, bs_rtm, *ws,
                                *bs_scratch, it, bs_it0);

            boundary_runtime->prefetch_next_backward_chunk_if_needed(
                it - bs_it0, (int)p.nt - bs_it0);

            Eq::bs_rtm_tap(*state, ctx, forward, adjoint, illumination,
                              p.compute_illumination);
        }

        if (Eq::BS_HAS_IT0_ADJOINT_TAIL && it_lo == 0 && p.nt > 0 && bs_it0 == 0) {
            auto adj_view = adjoint.view();
            Eq::adjoint_step(*state, ctx, adj_view, cpml, *ws, nullptr, nullptr);
            Eq::inject_adjoint_source(*state, ctx, adj_view, p, 0, adjoint_nsrc, *ws);
            Eq::rotate_adjoint_buffers(adjoint);
            Eq::accumulate_source_grad(*state, ctx, adjoint, p, grads,
                                       0, forward_nsrc);
            // The store-based loop accumulates illumination for it = nt-1 .. 0,
            // right here in its own iteration order; this loop floors at it == 1,
            // so it was one lambda(0)^2 short -- and lambda is largest at it == 0,
            // straight after the last residual injection. Measured 1.7e-2 (2-D)
            // / 1.8e-3 (3-D) against the full store. The forward field is NOT
            // reconstructed at it == 0, so only the receiver term can be closed;
            // the source term's it == 0 contribution is u_tt(0)^2, which the same
            // comparison bounds below 2.7e-8.
            if (bs_rtm != nullptr && p.compute_illumination)
                Eq::bs_illum_tail(*state, ctx, adjoint, *bs_rtm);
        }

        Eq::pack_outputs(out, grads, illumination);
        return out;
    }

private:
    // Declaration order == construction order; destruction runs in reverse,
    // matching the hand-written function's stack unwind.
    BackwardInput p;
    Dims d;
    int adjoint_nsrc = 0, forward_nsrc = 0;
    std::optional<SolverContext> ctx_;
    typename Eq::Wavefield adjoint;
    typename Eq::Wavefield forward;
    std::vector<torch::Tensor> grads;
    RTMOutput illumination;
    RTMOutput* bs_rtm = nullptr;
    typename Eq::CPML cpml_tensor;
    decltype(std::declval<typename Eq::CPML>().view()) cpml;
    int save_width = 0;
    bool staged_boundary = false;
    EffectiveBoundarySaver boundary_saver;
    GeneralBoundaryPointer bs{};
    fdtd::LaunchConfig launch_config{}, fwd_source_config{}, adj_source_config{};
    std::optional<typename Eq::State> state;
    std::optional<typename Eq::BwdWorkspace> ws;
    std::optional<typename Eq::BsScratch> bs_scratch;
    std::optional<BoundaryScope> boundary_scope;
    BoundaryRuntime* boundary_runtime = nullptr;
    int bs_it0 = 0, bs_stop = 0;
    int run_calls = 0;
};

template <class Eq>
BackwardOutput generic_backward_bs(const BackwardInput& in)
{
    GenericBackwardBsRunner<Eq> runner(in);
    return runner.run(in.bw_it_begin, in.bw_it_end, in.step_phase);
}

// Pool slots of the two checkpoint skeletons, declared per equation in
// cuda_layout (checkpoint_replay_shapes / backward_workspace_shapes, by memory
// mode) and bound by the propagator.  Both declarations are unconditional for
// the equations that instantiate these two entry points (acoustic2d /
// acoustic3d: ``checkpoint_replay_shapes`` returns the chunk history in "ckpt"
// mode and ``backward_workspace_shapes`` the leaf scratch in "recursive"
// mode), so each slot is REQUIRED in the mode that reads it -- there is no
// fallback allocation left.
//
// checkpoint_replay, ckpt mode: the recomputed chunk, Eq::allt_shape(d,
// chunk_size) rows.  One buffer per call serves every chunk (the last, shorter
// chunk uses a prefix of its rows): each chunk replays rows [0, end - start)
// before its reverse loop reads them, and the rim the stencil never writes
// keeps its allocation-time zero -- exactly what the per-call torch::zeros it
// replaces held -- so the pool is never re-zeroed.  Recursive mode keeps no
// history: the leaf images from its u_this scratch.
enum AcousticCkptReplaySlot : int {
    ACOUSTIC_CKPT_CHUNK_FORWARD = 0, N_ACOUSTIC_CKPT_REPLAY
};
// adjoint_workspace, recursive mode: the leaf's model-shaped u_this, zeroed by
// the leaf before the replay kernel writes it (so an uninitialised slot is
// fine); the other modes of this skeleton take no workspace.
enum AcousticRecursiveWorkspaceSlot : int {
    ACOUSTIC_RECURSIVE_U_THIS = 0, N_ACOUSTIC_RECURSIVE_WORKSPACE
};

// ---- generic_backward_ckpt (uniform chunks; acoustic-family shape) ----
template <class Eq>
BackwardOutput generic_backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    TORCH_CHECK(!in.bw_stepped(),
                "checkpoint backward does not support bw_it_begin/bw_it_end in v1");
    const auto& p = in;
    BackwardOutput out;
    TORCH_CHECK(static_cast<int>(p.checkpoint_replay.size()) == N_ACOUSTIC_CKPT_REPLAY,
                Eq::NAME, "/ckpt backward requires the propagator-bound "
                "checkpoint_replay (cuda_layout.checkpoint_replay_shapes): ",
                N_ACOUSTIC_CKPT_REPLAY, " slot (the chunk history), got ",
                p.checkpoint_replay.size());
    TORCH_CHECK(static_cast<int>(p.forward_wavefields.size()) == Eq::CKPT_STATE_COUNT,
                Eq::NAME, "/ckpt backward requires the propagator-bound replay "
                "state (cuda_layout.checkpoint_state_nvar / the forward slot "
                "table): one set of ", Eq::CKPT_STATE_COUNT,
                " forward_wavefields, got ", p.forward_wavefields.size());

    CheckpointRuntime checkpoint_runtime(
        p.checkpoints, Eq::CKPT_NVAR, true, false,
        p.checkpoint_interval, p.checkpoint_steps, p.checkpoint_on_cpu,
        "backward_chunk", Eq::NAME);

    auto vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);

    SolverContext ctx = make_ctx<Eq>(p, d);
    Eq::setup_ctx(ctx, p);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);
    // The chunk replay state: replay state set 0 of p.forward_wavefields.
    typename Eq::Wavefield forward;
    Eq::bind_or_alloc_recon_ckpt(forward, p, vp);
    // Slab geometry follows the FORWARD-state aux layout (the recompute runs
    // the forward kernel); the adjoint aux stays full-domain.
    Eq::init_aux_slabs(ctx, forward);

    // Same as full mode: gradients bound from grads_out (or allocated when
    // unbound), illumination only when something will read it.
    std::vector<torch::Tensor> grads;
    RTMOutput illumination;
    Eq::bind_backward_outputs(in, grads, illumination,
                              /*want_adcig=*/Eq::ADCIG_IN_FULL_MODES);
    RTMOutput* rtm_out = Eq::rtm_out_full(in, illumination);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);

    typename Eq::State state = Eq::make_state(p, d, ctx, launch_config,
                                              fwd_source_config,
                                              adj_source_config);
    typename Eq::BwdWorkspace ws = Eq::make_bwd_workspace(p, state, ctx, adjoint);

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    // The recomputed chunk (ACOUSTIC_CKPT_CHUNK_FORWARD), once per call for every chunk.
    auto chunk_forward = pool_required(p.checkpoint_replay, ACOUSTIC_CKPT_CHUNK_FORWARD,
                                       Eq::allt_shape(d, chunk_size), vp.options(),
                                       "checkpoint_replay");

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors());

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* u_this = chunk_forward[it - start].template data_ptr<float>();
            Eq::replay_step(state, ctx, for_view, cpml, true, u_this);
            Eq::inject_source_fwd(state, ctx, for_view, p, it, forward_nsrc);
            Eq::rotate_recon_buffers(forward);
        }

        for (int it = end - 1; it >= start; --it) {
            auto adj_view = adjoint.view();
            Eq::adjoint_step(state, ctx, adj_view, cpml, ws, nullptr, nullptr);
            Eq::inject_adjoint_source(state, ctx, adj_view, p, it, adjoint_nsrc, ws);
            Eq::rotate_adjoint_buffers(adjoint);
            Eq::accumulate_source_grad(state, ctx, adjoint, p, grads,
                                       it, forward_nsrc);
            Eq::image_step(state, ctx,
                           chunk_forward[it - start].template data_ptr<float>(),
                           it, adjoint, &grads, rtm_out, ws);
        }
    }

    Eq::pack_outputs(out, grads, illumination);
    return out;
}

// ---- generic_backward_recursive_ckpt (acoustic-family bisection) ----
inline int recursive_checkpoint_scratch_depth(int interval_length)
{
    int depth = 0;
    while (interval_length > 1) {
        interval_length = (interval_length + 1) / 2;
        ++depth;
    }
    return depth;
}

template <class Eq>
void advance_forward_interval(typename Eq::Wavefield& forward, int start, int end,
                              typename Eq::State& state, const SolverContext& ctx,
                              const BackwardInput& p,
                              decltype(std::declval<typename Eq::CPML>().view()) cpml,
                              int forward_nsrc)
{
    for (int it = start; it < end; ++it) {
        auto view = forward.view();
        Eq::replay_step(state, ctx, view, cpml, false, nullptr);
        Eq::inject_source_fwd(state, ctx, view, p, it, forward_nsrc);
        Eq::rotate_recon_buffers(forward);
    }
}

template <class Eq>
void process_recursive_interval(int start, int end,
                                typename Eq::Wavefield& start_state,
                                typename Eq::Wavefield& adjoint,
                                const BackwardInput& p,
                                std::vector<torch::Tensor>& grads,
                                RTMOutput* rtm_out,
                                typename Eq::State& state, const SolverContext& ctx,
                                typename Eq::BwdWorkspace& ws,
                                decltype(std::declval<typename Eq::CPML>().view()) cpml,
                                int forward_nsrc, int adjoint_nsrc,
                                CheckpointRuntime& checkpoint_runtime,
                                std::vector<typename Eq::Wavefield>& scratch_states,
                                int scratch_depth,
                                torch::Tensor& u_this_scratch)
{
    if (start >= end)
        return;

    if (end - start == 1) {
        zero_tensor_device_async(u_this_scratch);
        float* u_this = u_this_scratch.data_ptr<float>();
        auto fwd_view = start_state.view();
        Eq::replay_step(state, ctx, fwd_view, cpml, true, u_this);
        Eq::inject_source_fwd(state, ctx, fwd_view, p, start, forward_nsrc);
        Eq::rotate_recon_buffers(start_state);

        auto adj_view = adjoint.view();
        Eq::adjoint_step(state, ctx, adj_view, cpml, ws, nullptr, nullptr);
        Eq::inject_adjoint_source(state, ctx, adj_view, p, start, adjoint_nsrc, ws);
        Eq::rotate_adjoint_buffers(adjoint);
        Eq::accumulate_source_grad(state, ctx, adjoint, p, grads,
                                   start, forward_nsrc);
        Eq::image_step(state, ctx, u_this, start, adjoint, &grads, rtm_out, ws);
        return;
    }

    int mid = start + (end - start) / 2;
    TORCH_CHECK(scratch_depth < static_cast<int>(scratch_states.size()),
                "Recursive checkpoint scratch depth exhausted.");
    typename Eq::Wavefield& mid_state = scratch_states[scratch_depth];
    checkpoint_runtime.copy_state(mid_state.state_tensors(), start_state.state_tensors());
    advance_forward_interval<Eq>(mid_state, start, mid, state, ctx, p, cpml, forward_nsrc);
    process_recursive_interval<Eq>(mid, end, mid_state, adjoint, p, grads,
                                   rtm_out, state, ctx, ws, cpml,
                                   forward_nsrc, adjoint_nsrc, checkpoint_runtime,
                                   scratch_states, scratch_depth + 1, u_this_scratch);
    process_recursive_interval<Eq>(start, mid, start_state, adjoint, p, grads,
                                   rtm_out, state, ctx, ws, cpml,
                                   forward_nsrc, adjoint_nsrc, checkpoint_runtime,
                                   scratch_states, scratch_depth + 1, u_this_scratch);
}

template <class Eq>
BackwardOutput generic_backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    TORCH_CHECK(!in.bw_stepped(),
                "checkpoint backward does not support bw_it_begin/bw_it_end in v1");
    const auto& p = in;
    BackwardOutput out;

    TORCH_CHECK((int)p.checkpoints.size() == Eq::CKPT_NVAR,
                Eq::NAME, " recursive checkpointing expects ", Eq::CKPT_NVAR,
                " checkpoint tensors");
    TORCH_CHECK(static_cast<int>(p.adjoint_workspace.size())
                    == N_ACOUSTIC_RECURSIVE_WORKSPACE,
                Eq::NAME, "/recursive backward requires the propagator-bound "
                "adjoint_workspace (cuda_layout.backward_workspace_shapes in "
                "\"recursive\" mode): ", N_ACOUSTIC_RECURSIVE_WORKSPACE,
                " slot (the leaf's u_this scratch), got ",
                p.adjoint_workspace.size());
    TORCH_CHECK(p.checkpoint_replay.empty(),
                Eq::NAME, " recursive checkpoint backward keeps no segment history "
                "(the leaf images from its u_this scratch): checkpoint_replay must be "
                "empty, got ", p.checkpoint_replay.size());

    auto checkpoint_steps_cpu = p.checkpoint_steps.to(torch::kCPU).to(torch::kInt32).contiguous();
    TORCH_CHECK(checkpoint_steps_cpu.dim() == 1, "checkpoint_steps must be 1-D");
    CheckpointRuntime checkpoint_runtime(
        p.checkpoints, Eq::CKPT_NVAR, true, true,
        p.checkpoint_interval, checkpoint_steps_cpu, p.checkpoint_on_cpu,
        "backward_recursive", Eq::NAME);

    auto vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);

    SolverContext ctx = make_ctx<Eq>(p, d);
    Eq::setup_ctx(ctx, p);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);
    checkpoint_runtime.zero_state(adjoint.state_tensors());

    // Same as full mode: gradients bound from grads_out (or allocated when
    // unbound), illumination only when something will read it.
    std::vector<torch::Tensor> grads;
    RTMOutput illumination;
    Eq::bind_backward_outputs(in, grads, illumination,
                              /*want_adcig=*/Eq::ADCIG_IN_FULL_MODES);
    RTMOutput* rtm_out = Eq::rtm_out_full(in, illumination);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);

    typename Eq::State state = Eq::make_state(p, d, ctx, launch_config,
                                              fwd_source_config,
                                              adj_source_config);
    typename Eq::BwdWorkspace ws = Eq::make_bwd_workspace(p, state, ctx, adjoint);

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    TORCH_CHECK(p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
                "checkpoint_count does not match checkpoint_steps");
    TORCH_CHECK(static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
                "checkpoint buffer is smaller than checkpoint_steps");

    const int* checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();

    int max_segment_length = 0;
    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];
        max_segment_length = std::max(max_segment_length, end - start);
    }

    // Replay state sets of p.forward_wavefields: set 0 is the segment start
    // state (zeroed or checkpoint-loaded per segment before any read), sets
    // 1..depth the bisection's scratch states (copy_state-filled from their
    // parent before any read).  The propagator hands 1 + depth sets, its depth
    // (_c.py _recursive_scratch_depth) mirroring recursive_checkpoint_scratch_depth
    // on the same longest segment.
    const int scratch_depth = recursive_checkpoint_scratch_depth(max_segment_length);
    TORCH_CHECK(static_cast<int>(p.forward_wavefields.size())
                    == (1 + scratch_depth) * Eq::CKPT_STATE_COUNT,
                Eq::NAME, "/recursive backward requires the propagator-bound "
                "replay state sets (cuda_layout.recursive_state_depth): ",
                1 + scratch_depth, " sets of ", Eq::CKPT_STATE_COUNT,
                " forward_wavefields, got ", p.forward_wavefields.size());

    typename Eq::Wavefield start_state;
    Eq::bind_or_alloc_recon_ckpt(start_state, p, vp);
    Eq::init_aux_slabs(ctx, start_state);

    std::vector<typename Eq::Wavefield> scratch_states(scratch_depth);
    for (int level = 0; level < scratch_depth; ++level)
        Eq::bind_or_alloc_recursive_scratch(scratch_states[level], p, vp,
                                            /*set=*/level + 1, start_state);
    // The leaf's u_this (ACOUSTIC_RECURSIVE_U_THIS): zeroed per leaf before the replay
    // kernel writes it, so it needs no initial contents.
    auto u_this_scratch = pool_required(p.adjoint_workspace, ACOUSTIC_RECURSIVE_U_THIS, vp,
                                        "adjoint_workspace");

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(),
                                    start_state.next_tensors());

        process_recursive_interval<Eq>(start, end, start_state, adjoint, p,
                                       grads, rtm_out, state, ctx, ws, cpml,
                                       forward_nsrc, adjoint_nsrc,
                                       checkpoint_runtime, scratch_states, 0,
                                       u_this_scratch);
    }

    Eq::pack_outputs(out, grads, illumination);
    return out;
}

} // namespace eqdrv
