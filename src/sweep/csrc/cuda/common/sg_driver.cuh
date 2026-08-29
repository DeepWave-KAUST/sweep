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
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <vector>

#include "common.cuh"
#include "context.h"
#include "checkpoint_runtime.cuh"
#include "cudautils.h"
#include "boundarysaver.cuh"
#include "boundary_runtime.cuh"
#include "wavetypes.h"
#include "eq_driver.cuh"     // Dims / read_dims / make_ctx / stencil_order
#include "../launch/config.h"

namespace eqdrv {

// ------------------------------------------------------------------------- //
// sg_generic_forward
// ------------------------------------------------------------------------- //

template <class Eq>
ForwardOutput sg_generic_forward(const ForwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());

    const auto& p = in;
    ForwardOutput out;

    typename Eq::Models models = Eq::parse_models(p);
    const auto& vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);

    // ---- Stepped forward range [it_begin, it_end) ----
    const int it0 = p.it_begin;
    const int it1 = (p.it_end < 0) ? static_cast<int>(p.nt) : p.it_end;
    TORCH_CHECK(0 <= it0 && it0 <= it1 && it1 <= static_cast<int>(p.nt),
                "stepped forward: require 0 <= it_begin <= it_end <= nt, got [",
                it0, ", ", it1, ") with nt=", p.nt);
    const bool stepped = (it0 != 0) || (it1 != static_cast<int>(p.nt));
    // The staggered phase-split is a PHYSICS split (unlike acoustic's spatial
    // strip split): phase 1 = full-grid velocity update only, phase 2 =
    // full-grid stress update + source/record/BS/checkpoint tail. A DD
    // driver exchanges v halos between the phases and s halos after phase
    // 2, so the cut-adjacent stress columns read exchanged (not locally
    // recomputed) velocities — that keeps the transverse CPML memory
    // divergence in the halo columns from ever reaching owned cells.
    TORCH_CHECK(p.step_phase == 0 || p.step_phase == 1 || p.step_phase == 2,
                "elastic step_phase must be 0, 1 or 2");

    typename Eq::Wavefield wavefield;
    // On a continuation call the internal allocate() would silently zero the
    // propagation state — the caller must keep binding the same tensors.
    // (No buffer-role rotation: every field updates in place, so the caller
    // binds the SAME tensor list for every segment.)
    TORCH_CHECK(it0 == 0 || !p.wavefields.empty(),
                "stepped continuation (it_begin>0) requires Python-bound wavefields");
    Eq::bind_or_alloc_forward(wavefield, p, vp);
    auto wf = Eq::view(wavefield);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    int nsrc = p.sources_loc.size(1);
    int nrec = p.receivers_loc.size(1);
    int nsrc_fields = p.source_field_indices.numel();
    int nrec_fields = p.receiver_field_indices.numel();
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);
    TORCH_CHECK(!stepped || p.record_out.defined(),
                "stepped forward requires record_out bound from Python");
    auto record = p.record_out.defined()
        ? p.record_out
        : torch::zeros({nrec_fields, d.B, nrec, p.nt}, vp.options());
    if (p.record_out.defined())
        TORCH_CHECK(record.is_contiguous() &&
                    record.size(-1) == static_cast<long>(p.nt),
                    "record_out must be contiguous with trailing dim nt");

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

    torch::Tensor u_allt;
    if (p.save_all_wavefields) {
        TORCH_CHECK(!stepped || p.u_allt_out.defined(),
                    "stepped + save_all_wavefields requires u_allt_out bound from Python");
        u_allt = p.u_allt_out.defined()
            ? p.u_allt_out
            : torch::zeros(Eq::allt_shape(d, p.nt), vp.options());
    }

    SolverContext solver = make_ctx<Eq>(p, d);
    Eq::setup_ctx(solver, p);
    solver.set_cut_mask(p.cut_face_mask);   // cut-aware phys bounds (0 = single domain)
    Eq::init_aux_slabs(solver, wavefield);

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    // The internal full-storage fallback ring is per-call; segments after the
    // first would lose everything saved before them.
    if (stepped && p.use_boundary_saving)
        TORCH_CHECK(!p.boundary_gpu.empty(),
                    "stepped forward with boundary saving requires Python-bound boundary_gpu");
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary)
        boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, solver, vp,
                                save_width, 1, true, false, p.transfer_interval,
                                p.boundary_cpu, p.boundary_gpu, p.last_two,
                                p.use_pinned_memory);
    else
        boundary_saver.allocate(p.use_boundary_saving, Eq::NDIM, Eq::BS_NVAR, solver, vp,
                                save_width, 1, true, true, 1, {}, p.boundary_gpu,
                                p.last_two, p.use_pinned_memory);
    auto bs = boundary_saver.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto source_config = fdtd::Geom::make(nsrc, d.B);
    auto record_config = fdtd::Geom::make(nrec, d.B);

    float* u_this_t = nullptr;

    typename Eq::State state = Eq::make_state(p, d, models, launch_config,
                                              source_config, record_config);

    AsyncCopyContext async_copy(staged_boundary && p.use_boundary_saving);
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

    const bool do_v = (p.step_phase == 0 || p.step_phase == 1);
    const bool do_s = (p.step_phase == 0 || p.step_phase == 2);
    TORCH_CHECK(p.step_phase == 0 || it1 == it0 + 1,
                "elastic phase-split requires a single step (it_end == it_begin + 1)");

    for (int it = it0; it < it1; ++it) {

        u_this_t = u_allt.defined() ? u_allt[it].data_ptr<float>() : nullptr;

        if (do_v)
            Eq::velocity_substep(state, wf, cpml, solver);   // t+0.5

        if (!do_s)
            continue;

        Eq::stress_substep(state, wf, cpml, solver, u_this_t);   // t+1.0

        for (int isrc = 0; isrc < nsrc_fields; ++isrc) {
            float* field = Eq::field_ptr(wf, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            Eq::inject_source(state, solver, field, p.source, p.sources_loc,
                              it, nsrc);
        }

        checkpoint_runtime.save_forward(static_cast<int>(it), static_cast<int>(p.nt),
                                        wavefield.checkpoint_tensors());

        if (p.use_boundary_saving)
            Eq::save_boundary_fields(boundary_runtime, state, solver, wf,
                                     it, (int)p.nt, bs, save_width);

        for (int irec = 0; irec < nrec_fields; ++irec) {
            float* field = Eq::field_ptr(wf, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            Eq::record_field(state, solver, field, record, irec,
                             p.receivers_loc, it, nrec);
        }
    }

    // Save the final state for backward (only once the final segment has
    // run; mid-run segments leave last_two untouched).
    if (p.use_boundary_saving && it1 == static_cast<int>(p.nt))
        Eq::save_last_state(boundary_saver, wavefield);

    boundary_runtime.synchronize();

    out.wavefield = u_allt;
    out.last_two = boundary_saver.last_two_t;
    out.record = record;

    return out;
}

} // namespace eqdrv
