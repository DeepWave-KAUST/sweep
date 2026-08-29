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


// Validate the stepped-backward segment fields, the DD cut mask and the
// backward phase split for staggered-family entry points.  ``need_recon`` is
// true for boundary-saving mode.
template <class Eq>
void sg_check_stepped_backward(const BackwardInput& p, bool need_recon)
{
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    TORCH_CHECK(0 <= it_lo && it_lo < it_hi && it_hi <= static_cast<int>(p.nt),
                "stepped backward: require 0 <= bw_it_end < bw_it_begin <= nt, got [",
                it_lo, ", ", it_hi, ") with nt=", p.nt);
    TORCH_CHECK((p.cut_face_mask & ~Eq::CUT_MASK_BITS) == 0,
                Eq::NDIM, "D cut_face_mask uses ", Eq::CUT_MASK_DESC,
                " only, got ", p.cut_face_mask);
    TORCH_CHECK(p.step_phase >= 0 && p.step_phase <= 3,
                "elastic backward step_phase must be 0, 1, 2 or 3 "
                "(3 = residual/source injections only)");
    const bool phased = (p.step_phase != 0);
    if (phased) {
        TORCH_CHECK(need_recon,
                    "phased elastic backward (step_phase) is only supported "
                    "for the boundary-saving path (backward_bs)");
        TORCH_CHECK(it_hi == it_lo + 1,
                    "elastic backward phase-split requires a single-step "
                    "segment (bw_it_begin == bw_it_end + 1)");
    }
    if (need_recon && p.cut_face_mask != 0) {
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "domain-decomposed backward_bs (cut_face_mask) supports "
                    "gpu-direct boundary storage only "
                    "(boundary_on_cpu/boundary_on_disk unsupported in v1)");
    }
    if (!p.bw_stepped() && !phased)
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
        TORCH_CHECK(!p.boundary_on_cpu && !p.boundary_on_disk,
                    "stepped backward_bs supports gpu-direct boundary storage "
                    "only (boundary_on_cpu/boundary_on_disk unsupported in v1)");
    }
}

// ---- sg_generic_backward (full storage) ----
template <class Eq>
BackwardOutput sg_generic_backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    sg_check_stepped_backward<Eq>(p, /*need_recon=*/false);
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
    Eq::prep_adjoint(adjoint, first_segment);

    auto adj_view = Eq::view(adjoint);

    std::vector<torch::Tensor> grads;
    Eq::bind_grads(p, grads);
    typename Eq::Workspace workspace = Eq::make_workspace(p, vp);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml_view = cpml_tensor.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(p.forward_sources_loc.size(1), d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);
    typename Eq::State state = Eq::make_state(p, d, models, launch_config,
                                              fwd_source_config, adj_source_config);

    auto zero_velocity = torch::zeros_like(vp);
    const auto adj_source_signed = Eq::signed_adjoint_sources(p, receiver_fields);

    // it_hi == nt and it_lo == 0 without DD, so this is dev's full
    // reverse loop verbatim in the single-domain case.
    for (int it = it_hi - 1; it >= it_lo; --it) {
        Eq::undo_body_force(state, solver, adj_view, p, source_fields, it, grads);
        Eq::inject_residuals(state, solver, adj_view, p, receiver_fields,
                             adj_source_signed, it, adjoint_nsrc);

        typename Eq::VelPtrs vptrs =
            Eq::select_forward_velocities(p, it, zero_velocity);

        // Reverse step 0 has no adjoint apply kernel, so its gradient imaging
        // cannot be folded — emit it as a standalone calculate_grad pass.
        if (it == 0) {
            Eq::image_standalone(state, solver, adj_view, vptrs, grads);
            Eq::undo_receiver_rho(state, solver, grads, vptrs, p,
                                  receiver_fields, it, adjoint_nsrc);
            continue;
        }

        // FULL-mode step: equations with grad fusion fold the imaging into the
        // stress-adjoint-prepare kernel; the others image standalone first.
        Eq::full_fused_step(state, solver, adjoint, workspace, cpml_view,
                            vptrs, grads);

        Eq::undo_receiver_rho(state, solver, grads, vptrs, p,
                              receiver_fields, it, adjoint_nsrc);
    }

    out.grads = grads;
    return out;
}

// ---- sg_generic_backward_bs ----
template <class Eq>
BackwardOutput sg_generic_backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    sg_check_stepped_backward<Eq>(p, /*need_recon=*/true);
    // The staggered BS loop legacy floor is it == 1 (no it==0 tail), so the
    // segment containing it == 0 simply runs nothing extra.
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    const bool first_segment = (it_hi == static_cast<int>(p.nt));
    // Phase split (DD): 1 = inject + stress recon/restore + gradient +
    // stress adjoint; 2 = velocity adjoint + fv_prev capture + velocity
    // recon/restore.  step_phase 3 = injections only (driver prologue for the
    // first reverse step).  Monolithic step_phase 0 keeps the original
    // head-of-loop position; the executed op sequence is identical either way.
    const bool inject_only = (p.step_phase == 3);
    const bool phased = (p.step_phase != 0);
    const bool do_p1 = !inject_only && (p.step_phase != 2);
    const bool do_p2 = !inject_only && (p.step_phase != 1);

    typename Eq::Models models = Eq::parse_models(p);
    const auto& vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);

    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);
    auto source_fields = p.source_field_indices.to(torch::kCPU);
    auto receiver_fields = p.receiver_field_indices.to(torch::kCPU);

    SolverContext solver = make_ctx<Eq>(p, d);
    Eq::setup_ctx(solver, p);
    // DD: skip cut faces in the strip restore, collapse the NOPML exclusion
    // bands to the stencil halo on cut sides, and route cut-side cells of
    // the adjoint prepare kernels through the interior branch.
    solver.set_cut_mask(p.cut_face_mask);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);
    Eq::init_aux_slabs(solver, adjoint);

    // Reconstruction state: the physical fields plus the carried velocity
    // tensors (v at time it+1, consumed by the gradient kernel).  When
    // stepping, all must be Python-owned to survive segment boundaries.
    typename Eq::Wavefield forward;
    typename Eq::ReconCarriers carriers =
        Eq::bind_or_alloc_recon(forward, p, vp);

    // Seed the reverse reconstruction from the saved last snapshot — FIRST
    // segment only (and not on a phase-2 re-entry).  In phased mode the seed
    // belongs to the INJECTION sub-phase (3): the driver runs 3 before 1, and
    // the injections write recon fields the seed would otherwise overwrite.
    if (first_segment && (inject_only || (!phased && do_p1)))
        Eq::seed_recon(forward, p);

    auto neg_forward_source = -p.forward_source;

    auto for_view = Eq::view(forward);
    auto adj_view = Eq::view(adjoint);

    std::vector<torch::Tensor> grads;
    Eq::bind_grads(p, grads);
    typename Eq::Workspace workspace = Eq::make_workspace(p, vp);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml_view = cpml_tensor.view();

    EffectiveBoundarySaver boundary_saver;
    int save_width = solver.M + 1;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, solver, vp, save_width,
                                1, true, false, p.transfer_interval,
                                p.boundary_cpu, p.boundary_gpu, {}, p.use_pinned_memory);
    } else {
        boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, solver, vp, save_width,
                                1, true, true, 1, {}, p.boundary_gpu, {},
                                p.use_pinned_memory);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);
    typename Eq::State state = Eq::make_state(p, d, models, launch_config,
                                              fwd_source_config, adj_source_config);

    AsyncCopyContext async_copy(staged_boundary);
    BoundaryRuntime boundary_runtime(
        boundary_saver,
        Eq::NDIM,
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

    const auto adj_source_signed = Eq::signed_adjoint_sources(p, receiver_fields);

    // Residual / source injections for reverse index jt, one unit (the rho
    // correction must read the adjoint velocity BEFORE jt's residuals land):
    // body-force source-cell rho correction, signed receiver residuals,
    // reconstruction un-injection of the forward source.
    auto inject_step = [&](int jt) {
        Eq::undo_body_force(state, solver, adj_view, p, source_fields, jt, grads);
        Eq::inject_residuals(state, solver, adj_view, p, receiver_fields,
                             adj_source_signed, jt, adjoint_nsrc);
        Eq::uninject_forward_source(state, solver, for_view, p, source_fields,
                                    neg_forward_source, jt, forward_nsrc);
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
            Eq::bs_phase1(state, solver, for_view, adj_view, adjoint, workspace,
                          cpml_view, boundary_runtime, bs, save_width,
                          grads, carriers, p, receiver_fields, it, adjoint_nsrc);

        if (do_p2)
            Eq::bs_phase2(state, solver, for_view, adjoint, workspace, cpml_view,
                          boundary_runtime, bs, save_width, carriers, forward,
                          it, (int)p.nt);
    }

    out.grads = grads;
    return out;
}

// ---- sg_generic_backward_ckpt (chunked segments with velocity carriers) ----

template <class Eq>
void sg_backward_segment(
    const BackwardInput& p,
    typename Eq::Models& models,
    typename Eq::State& state,
    typename Eq::Wavefield& start_state,
    typename Eq::Wavefield& adjoint,
    typename Eq::Workspace& workspace,
    CheckpointRuntime& checkpoint_runtime,
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
    const auto adj_source_signed = Eq::signed_adjoint_sources(p, receiver_fields);

    auto seg = Eq::alloc_seg_buffers(p.models[0], segment_len);
    typename Eq::Wavefield forward;
    Eq::bind_or_alloc_recon_ckpt(forward, p, p.models[0]);

    checkpoint_runtime.copy_state(forward.state_tensors(), start_state.state_tensors());

    Eq::capture_seg(seg, forward, 0);
    auto for_view = Eq::view(forward);

    for (int it = start; it < end; ++it) {
        Eq::velocity_substep(state, for_view, cpml_view, solver);
        Eq::stress_substep(state, for_view, cpml_view, solver, nullptr);
        Eq::capture_seg(seg, forward, it - start + 1);
        Eq::inject_sources_fwd_bw(state, solver, for_view, p, source_fields, it);
    }

    auto adj_view = Eq::view(adjoint);
    for (int it = end - 1; it >= start; --it) {
        Eq::undo_body_force(state, solver, adj_view, p, source_fields, it, grads);
        Eq::inject_residuals(state, solver, adj_view, p, receiver_fields,
                             adj_source_signed, it, adjoint_nsrc);

        const int now_offset = it - start + 1;
        const int next_offset = now_offset + 1;
        typename Eq::VelPtrs vptrs = Eq::seg_vel_ptrs(
            seg, now_offset,
            next_offset <= segment_len ? next_offset : -1,
            next_segment_v);

        Eq::image_standalone(state, solver, adj_view, vptrs, grads);
        Eq::undo_receiver_rho(state, solver, grads, vptrs, p,
                              receiver_fields, it, adjoint_nsrc);

        if (it == 0)
            continue;

        Eq::plain_adjoint_step(state, solver, adjoint, workspace, cpml_view);
    }

    Eq::store_prev_segment(prev_segment_next_v, seg);
}

template <class Eq>
BackwardOutput sg_generic_backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
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

    std::vector<torch::Tensor> grads;
    Eq::alloc_grads(vp, grads);
    typename Eq::Workspace workspace = Eq::make_workspace(p, vp);

    typename Eq::Wavefield start_state;
    Eq::alloc_recursive_start_state(start_state, p, vp);
    Eq::check_ckpt_aux_layout(start_state, adjoint);
    std::vector<torch::Tensor> next_segment_v, prev_segment_next_v;
    for (int c = 0; c < Eq::N_VEL; ++c) {
        next_segment_v.push_back(torch::zeros_like(vp));
        prev_segment_next_v.push_back(torch::zeros_like(vp));
    }

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);
        if (chunk_id == 0) {
            checkpoint_runtime.zero_state(start_state.state_tensors());
        } else {
            checkpoint_runtime.load(chunk_id, start_state.checkpoint_tensors());
        }
        sg_backward_segment<Eq>(p, models, state, start_state, adjoint, workspace,
                                checkpoint_runtime, start, end, cpml_view, solver,
                                source_fields, receiver_fields,
                                next_segment_v, grads, prev_segment_next_v);
        for (int c = 0; c < Eq::N_VEL; ++c)
            next_segment_v[c].copy_(prev_segment_next_v[c]);
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
    for (auto& t : current_v) t.zero_();
    for (auto& t : next_v) t.zero_();

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

        if (it == target_index)
            Eq::capture_velocities(current_v, forward);
        if (it == target_index + 1) {
            Eq::capture_velocities(next_v, forward);
            break;
        }

        Eq::inject_sources_fwd_bw(state, solver, for_view, p, source_fields, it);

        if (it == target_index && target_index + 1 >= p.nt)
            break;
    }
}

template <class Eq>
BackwardOutput sg_generic_backward_recursive_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
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
    Eq::alloc_grads(vp, grads);
    typename Eq::Workspace workspace = Eq::make_workspace(p, vp);

    const int num_saved_checkpoints = static_cast<int>(checkpoint_steps_cpu.numel());
    TORCH_CHECK(p.checkpoint_count == num_saved_checkpoints || p.checkpoint_count == 0,
                "checkpoint_count does not match checkpoint_steps");
    TORCH_CHECK(static_cast<int>(p.checkpoints[0].size(0)) >= num_saved_checkpoints,
                "checkpoint buffer is smaller than checkpoint_steps");

    const int* checkpoint_steps = checkpoint_steps_cpu.data_ptr<int>();
    typename Eq::Wavefield forward;
    Eq::bind_or_alloc_recon_ckpt(forward, p, vp);
    std::vector<torch::Tensor> current_v, next_v;
    for (int c = 0; c < Eq::N_VEL; ++c) {
        current_v.push_back(torch::zeros_like(vp));
        next_v.push_back(torch::zeros_like(vp));
    }
    const auto adj_source_signed = Eq::signed_adjoint_sources(p, receiver_fields);

    auto adj_view = Eq::view(adjoint);
    for (int it = p.nt - 1; it >= 0; --it) {
        Eq::undo_body_force(state, solver, adj_view, p, source_fields, it, grads);
        Eq::inject_residuals(state, solver, adj_view, p, receiver_fields,
                             adj_source_signed, it, (int)p.adjoint_sources_loc.size(1));

        sg_replay_forward_to_time<Eq>(p, state, forward, current_v, next_v, it,
                                      checkpoint_steps, num_saved_checkpoints,
                                      checkpoint_runtime, cpml_view, solver,
                                      source_fields);

        typename Eq::VelPtrs vptrs = Eq::carrier_vel_ptrs(current_v, next_v);
        Eq::image_standalone(state, solver, adj_view, vptrs, grads);
        Eq::undo_receiver_rho(state, solver, grads, vptrs, p, receiver_fields,
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
