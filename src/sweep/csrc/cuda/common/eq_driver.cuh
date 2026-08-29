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
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>

#include "common.cuh"
#include "context.h"
#include "checkpoint_runtime.cuh"
#include "cudautils.h"
#include "boundarysaver.cuh"
#include "boundary_runtime.cuh"
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

template <class Eq>
ForwardOutput generic_forward(const ForwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

    const auto& p = in;
    ForwardOutput out;

    auto vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);

    const int order = stencil_order(p.M);

    SolverContext ctx = make_ctx<Eq>(p, d);
    Eq::setup_ctx(ctx, p);
    // Cut-aware physical bounds (0 = single domain → legacy per-edge pad + M).
    ctx.set_cut_mask(p.cut_face_mask);

    const int it0 = p.it_begin;
    const int it1 = (p.it_end < 0) ? static_cast<int>(p.nt) : p.it_end;
    TORCH_CHECK(0 <= it0 && it0 <= it1 && it1 <= static_cast<int>(p.nt),
                "stepped forward: require 0 <= it_begin <= it_end <= nt, got [",
                it0, ", ", it1, ") with nt=", p.nt);
    const bool stepped = (it0 != 0) || (it1 != static_cast<int>(p.nt));

    // ---- DD phase-split step (comm/compute overlap) ----
    const int phase = p.step_phase;
    const bool cut_x_lo = (p.cut_face_mask & 1) != 0;
    const bool cut_x_hi = (p.cut_face_mask & 2) != 0;
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

    typename Eq::Wavefield wavefield;
    // On a continuation call the internal allocate() would silently zero the
    // propagation state — the caller must keep binding the same tensors.
    TORCH_CHECK(it0 == 0 || !p.wavefields.empty(),
                "stepped continuation (it_begin>0) requires Python-bound wavefields");
    Eq::bind_or_alloc_forward(wavefield, p, vp);
    Eq::init_aux_slabs(ctx, wavefield);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    TORCH_CHECK(!stepped || p.record_out.defined(),
                "stepped forward requires record_out bound from Python");
    auto record = p.record_out.defined()
        ? p.record_out
        : torch::zeros({d.N, p.receivers_loc.size(1), p.nt}, vp.options());
    if (p.record_out.defined())
        TORCH_CHECK(record.is_contiguous() &&
                    record.size(-1) == static_cast<long>(p.nt),
                    "record_out must be contiguous with trailing dim nt");

    // Wavefields for all timestep
    torch::Tensor u_allt;
    if (p.save_all_wavefields) {
        TORCH_CHECK(!stepped || p.u_allt_out.defined(),
                    "stepped + save_all_wavefields requires u_allt_out bound from Python");
        u_allt = p.u_allt_out.defined()
            ? p.u_allt_out
            : torch::zeros(Eq::allt_shape(d, p.nt), vp.options());
    }

    CheckpointRuntime checkpoint_runtime(
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

    int save_width = Eq::save_width(p.abcn, p.M);
    // The internal full-storage fallback ring is per-call; segments after the
    // first would lose everything saved before them.
    if (stepped && p.use_boundary_saving)
        TORCH_CHECK(!p.boundary_gpu.empty(),
                    "stepped forward with boundary saving requires Python-bound boundary_gpu");
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary)
        boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, ctx, vp,
                                save_width, Eq::BS_LAST_TWO_NVAR, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
                                p.last_two, p.use_pinned_memory, Eq::TANGENT_PAD * p.M);
    else
        boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, ctx, vp,
                                save_width, Eq::BS_LAST_TWO_NVAR, true, true,
                                1, {}, p.boundary_gpu,
                                p.last_two, p.use_pinned_memory, Eq::TANGENT_PAD * p.M);
    auto bs = boundary_saver.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto source_config = fdtd::Geom::make(nsrc, d.B);
    auto record_config = fdtd::Geom::make(nrec, d.B);

    float* u_thist = nullptr;

    typename Eq::State state = Eq::make_state(p, d, ctx, launch_config,
                                              source_config, record_config);

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
    // Boundary tail truncation: with boundary_tail_steps = K > 0 only the
    // last K steps' boundary strips are saved; the runtime and the Python
    // buffers work in shifted "saved-step" coordinates [0, K).  bs_it0 = 0
    // when disabled, making every shift below a no-op (bit-exact legacy).
    // Stepped/DD segments compose transparently: ``it`` is the GLOBAL step
    // index, so the save guard and shift never look at the segment bounds;
    // the Python-bound boundary_gpu ring (mandatory under stepped) is
    // allocated tail-shrunk by _ensure_boundary_buffers(nt_saved=...).
    const int bs_it0 = (p.use_boundary_saving && p.boundary_tail_steps > 0)
        ? std::max(0, (int)p.nt - p.boundary_tail_steps) : 0;
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        Eq::NDIM,
        p.use_boundary_saving,
        p.boundary_on_cpu,
        p.boundary_on_disk,
        p.boundary_disk_async_read,
        p.transfer_interval,
        p.boundary_ring_buffers,
        p.boundary_disk_files,
        async_copy.compute_stream,
        async_copy.copy_stream
    );
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
                Eq::launch_step_range(state, ctx, ctx.phys_x0(), ctx.phys_x0() + p.M,
                                      view, p.save_all_wavefields, u_thist, cpml);
            if (cut_x_hi)
                Eq::launch_step_range(state, ctx, ctx.phys_x1() - p.M, ctx.phys_x1(),
                                      view, p.save_all_wavefields, u_thist, cpml);
        } else if (phase == 2) {
            // Interior phase: the strict complement of the phase-1 strips
            // (no overlap — re-running a strip cell would double-advance
            // its CPML psi double-buffer write).
            Eq::launch_step_range(state, ctx,
                                  cut_x_lo ? ctx.phys_x0() + p.M : 0,
                                  cut_x_hi ? ctx.phys_x1() - p.M : d.nx,
                                  view, p.save_all_wavefields, u_thist, cpml);
        } else {
            Eq::launch_step_range(state, ctx, 0, d.nx,
                                  view, p.save_all_wavefields, u_thist, cpml);
        }

        if (phase == 1)
            continue;   // no boundary saving / source / record / swap / ckpt

        if (p.use_boundary_saving && it >= bs_it0) {
            Eq::save_boundary_fwd(boundary_runtime, state, ctx, view,
                                  it - bs_it0, (int)p.nt - bs_it0,
                                  bs, save_width);
        }

        Eq::inject_source_fwd(state, ctx, view, p, it, nsrc);

        Eq::record(state, ctx, view, record, p, it, nrec);

        Eq::end_of_step(wavefield);

        checkpoint_runtime.save_forward(it, static_cast<int>(p.nt),
                                        wavefield.checkpoint_tensors());

    }

    // Save the last state for backward (only once the final segment has run;
    // mid-run segments leave it untouched).  Phase 1 has not swapped yet —
    // roles would be wrong; phase 2 of the same step does the copy.
    if (p.use_boundary_saving && it1 == static_cast<int>(p.nt) && phase != 1) {
        Eq::save_last_state(boundary_saver, wavefield);
    }

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

} // namespace eqdrv
