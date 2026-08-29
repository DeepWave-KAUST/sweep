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


// Validate the stepped-backward segment fields (bw_it_begin/bw_it_end).
// ``need_recon`` is true for boundary-saving mode, where the reconstruction
// wavefield list must be Python-owned to survive segments.
template <class Eq>
void check_stepped_backward(const BackwardInput& p, bool need_recon)
{
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    TORCH_CHECK(0 <= it_lo && it_lo < it_hi && it_hi <= static_cast<int>(p.nt),
                "stepped backward: require 0 <= bw_it_end < bw_it_begin <= nt, got [",
                it_lo, ", ", it_hi, ") with nt=", p.nt);
    TORCH_CHECK((p.cut_face_mask & ~Eq::CUT_MASK_BITS) == 0,
                Eq::NDIM, "D cut_face_mask uses ", Eq::CUT_MASK_DESC,
                " only, got ", p.cut_face_mask);
    TORCH_CHECK(need_recon || p.cut_face_mask == 0,
                "domain-decomposed backward (cut_face_mask) is boundary-saving "
                "only; the full-storage path does not support DD (use backward_bs)");
    if (need_recon && p.cut_face_mask != 0) {
        TORCH_CHECK(!p.boundary_on_disk,
                    "domain-decomposed backward_bs (cut_face_mask) supports "
                    "gpu-direct or cpu boundary storage only "
                    "(boundary_on_disk unsupported in v1)");
    }
    if (!p.bw_stepped())
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
    out.image = torch::zeros_like(vp);
    out.source_illumination = torch::zeros_like(vp);
    out.receiver_illumination = torch::zeros_like(vp);
    if (want_adcig) {
        std::vector<int64_t> shape{(long)nlag};
        for (auto s : vp.sizes()) shape.push_back(s);
        out.adcig = torch::zeros(shape, vp.options());
    }
}

template <class Eq>
void bind_backward_outputs(const BackwardInput& p,
                           torch::Tensor& grad_wavelet,
                           torch::Tensor& grad,
                           RTMOutput& illumination)
{
    if (!p.grads_out.empty()) {
        TORCH_CHECK(p.grads_out.size() == p.models.size() + 1,
                    "grads_out must hold models.size()+1 tensors "
                    "(slot 0 = grad_wavelet)");
        grad_wavelet = p.grads_out[0];
        grad = p.grads_out[1];
    } else {
        grad_wavelet = torch::zeros_like(p.forward_source);
        grad = torch::zeros_like(p.models[0]);
    }
    if (!p.illum_out.empty()) {
        TORCH_CHECK(p.illum_out.size() == 2,
                    "illum_out must be {source_illumination, receiver_illumination}");
        illumination.image = torch::zeros_like(p.models[0]);
        illumination.source_illumination = p.illum_out[0];
        illumination.receiver_illumination = p.illum_out[1];
        TORCH_CHECK(!p.compute_adcig,
                    "compute_adcig is not supported on the segmented "
                    "(stepped / domain-decomposed) backward: the ADCIG cube "
                    "has no cross-segment accumulator. Run ADCIG on the "
                    "single-segment backward instead.");
    } else {
        init_rtm_output(illumination, p.models[0],
                        p.compute_adcig, 2 * p.adcig_max_lag + 1);
    }
}

// ---- generic_backward (full storage) ----
template <class Eq>
BackwardOutput generic_backward(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    check_stepped_backward<Eq>(in, /*need_recon=*/false);
    BackwardOutput out;
    torch::Tensor grad, grad_wavelet;
    RTMOutput illumination;
    bind_backward_outputs<Eq>(in, grad_wavelet, grad, illumination);
    RTMOutput* rtm_out = (in.compute_illumination || in.compute_adcig)
        ? &illumination : nullptr;

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
    // NOTE role reuse: in backward states, source_config = FORWARD sources,
    // record_config slot = ADJOINT sources.

    float* grad_ptr = grad.defined() ? grad.data_ptr<float>() : nullptr;
    const bool fused_img = !p.bw_stepped() && p.cut_face_mask == 0;

    for (int it = p.bw_begin() - 1; it >= p.bw_it_end; --it) {
        auto adj_view = adjoint.view();
        const float* img_fwd = (fused_img && grad_ptr != nullptr && it + 1 < p.nt)
                             ? p.u_forward[it + 1].data_ptr<float>() : nullptr;
        Eq::adjoint_step(state, ctx, adj_view, cpml,
                         img_fwd, img_fwd ? grad_ptr : nullptr);
        Eq::inject_adjoint_source(state, ctx, adj_view, p, it, adjoint_nsrc);
        Eq::post_adjoint(adjoint);
        Eq::accumulate_source_grad(state, ctx, adjoint, p, &grad_wavelet,
                                   it, forward_nsrc);
        torch::Tensor* step_grad = fused_img ? nullptr : &grad;
        if (step_grad != nullptr || rtm_out != nullptr) {
            Eq::image_step(state, ctx,
                           p.u_forward[it].data_ptr<float>(),
                           adjoint.u_now_t.template data_ptr<float>(),
                           vp, step_grad, rtm_out);
        }
    }
    if (fused_img && grad_ptr != nullptr) {
        Eq::image_step(state, ctx,
                       p.u_forward[0].data_ptr<float>(),
                       adjoint.u_now_t.template data_ptr<float>(),
                       vp, &grad, nullptr);
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

// ---- generic_backward_bs ----
template <class Eq>
BackwardOutput generic_backward_bs(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    const auto& p = in;
    BackwardOutput out;

    check_stepped_backward<Eq>(p, /*need_recon=*/true);
    const int it_hi = p.bw_begin();
    const int it_lo = p.bw_it_end;
    const bool first_segment = (it_hi == static_cast<int>(p.nt));

    auto vp = p.models[0];
    const Dims d = read_dims<Eq::NDIM>(vp);
    int adjoint_nsrc = p.adjoint_sources_loc.size(1);
    int forward_nsrc = p.forward_sources_loc.size(1);

    SolverContext ctx = make_ctx<Eq>(p, d);
    Eq::setup_ctx(ctx, p);
    // DD: skip cut faces in the strip restore, the seed rim-zeroing, the
    // NOPML exclusion band and the fused-adjoint pure_interior test.
    ctx.set_cut_mask(p.cut_face_mask);

    typename Eq::Wavefield adjoint;
    Eq::bind_or_alloc_adjoint(adjoint, p, vp);
    typename Eq::Wavefield forward;
    Eq::bind_or_alloc_recon(forward, p, vp);
    Eq::init_aux_slabs(ctx, adjoint);

    torch::Tensor grad, grad_wavelet;
    RTMOutput illumination;
    bind_backward_outputs<Eq>(p, grad_wavelet, grad, illumination);

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    int save_width = Eq::save_width(p.abcn, p.M);
    EffectiveBoundarySaver boundary_saver;
    bool staged_boundary = p.boundary_on_cpu || p.boundary_on_disk;
    if (staged_boundary) {
        boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, ctx, vp, save_width,
                                Eq::BS_LAST_TWO_NVAR, true, false,
                                p.transfer_interval, p.boundary_cpu, p.boundary_gpu,
                                {}, p.use_pinned_memory, Eq::TANGENT_PAD * p.M);
    } else {
        boundary_saver.allocate(true, Eq::NDIM, Eq::BS_NVAR, ctx, vp, save_width,
                                Eq::BS_LAST_TWO_NVAR, true, true, 1, {}, p.boundary_gpu,
                                {}, p.use_pinned_memory, Eq::TANGENT_PAD * p.M);
        if (p.boundary_gpu.empty())
            boundary_saver.load_from_vector(p.u_boundary, vp);
    }
    auto bs = boundary_saver.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);

    typename Eq::State state = Eq::make_state(p, d, ctx, launch_config,
                                              fwd_source_config,
                                              adj_source_config);

    // FIRST segment only: re-running the seeding (last-state copy + any
    // rim-zeroing) mid-stream would clobber the carried reconstruction state.
    if (first_segment)
        Eq::seed_reconstruction(state, ctx, forward, p);

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
    TORCH_CHECK(p.boundary_tail_steps >= 0, "boundary_tail_steps must be >= 0");
    const int bs_it0 = (p.boundary_tail_steps > 0)
        ? std::max(0, (int)p.nt - p.boundary_tail_steps) : 0;
    const int bs_stop = bs_it0 > 0 ? bs_it0 + 1 : 0;
    boundary_runtime.prefetch_initial_backward_chunk((int)p.nt - bs_it0,
                                                     it_hi - bs_it0);

    for (int it = it_hi - 1; it >= std::max(std::max(it_lo, 1), bs_stop); --it) {
        auto adj_view = adjoint.view();

        Eq::adjoint_step(state, ctx, adj_view, cpml, nullptr, nullptr);
        Eq::inject_adjoint_source(state, ctx, adj_view, p, it, adjoint_nsrc);
        Eq::post_adjoint(adjoint);
        Eq::accumulate_source_grad(state, ctx, adjoint, p, &grad_wavelet,
                                   it, forward_nsrc);

        // Reconstruction + gradient imaging, in this equation's exact order.
        Eq::bs_reverse_step(state, ctx, forward, adjoint, boundary_runtime,
                            bs, save_width, cpml, p, grad, it, bs_it0);

        boundary_runtime.prefetch_next_backward_chunk_if_needed(
            it - bs_it0, (int)p.nt - bs_it0);

        Eq::bs_image_step(state, ctx, forward, adjoint, illumination,
                          in.compute_illumination);
    }

    if (it_lo == 0 && p.nt > 0 && bs_it0 == 0) {
        auto adj_view = adjoint.view();
        Eq::adjoint_step(state, ctx, adj_view, cpml, nullptr, nullptr);
        Eq::inject_adjoint_source(state, ctx, adj_view, p, 0, adjoint_nsrc);
        Eq::post_adjoint(adjoint);
        Eq::accumulate_source_grad(state, ctx, adjoint, p, &grad_wavelet,
                                   0, forward_nsrc);
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

// ---- generic_backward_ckpt (uniform chunks) ----
template <class Eq>
BackwardOutput generic_backward_ckpt(const BackwardInput& in)
{
    c10::cuda::CUDAGuard device_guard(in.models[0].device());
    TORCH_CHECK(!in.bw_stepped(),
                "checkpoint backward does not support bw_it_begin/bw_it_end in v1");
    const auto& p = in;
    BackwardOutput out;

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
    typename Eq::Wavefield forward;
    Eq::bind_or_alloc_recon_ckpt(forward, p, vp);
    // Slab geometry follows the FORWARD-state aux layout (the recompute runs
    // the forward kernel); the adjoint aux stays full-domain.
    Eq::init_aux_slabs(ctx, forward);

    auto grad = torch::zeros_like(vp);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output(illumination, vp, in.compute_adcig, 2 * in.adcig_max_lag + 1);
    RTMOutput* rtm_out = (in.compute_illumination || in.compute_adcig)
        ? &illumination : nullptr;

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);

    typename Eq::State state = Eq::make_state(p, d, ctx, launch_config,
                                              fwd_source_config,
                                              adj_source_config);

    int chunk_size = p.checkpoint_interval;
    int num_chunks = (p.nt + chunk_size - 1) / chunk_size;
    auto chunk_forward = torch::zeros(Eq::allt_shape(d, chunk_size), vp.options());

    for (int chunk_id = num_chunks - 1; chunk_id >= 0; --chunk_id) {
        int start = chunk_id * chunk_size;
        int end = std::min(static_cast<int>(p.nt), start + chunk_size);

        checkpoint_runtime.load(chunk_id, forward.checkpoint_tensors());

        for (int it = start; it < end; ++it) {
            auto for_view = forward.view();
            float* u_this = chunk_forward[it - start].template data_ptr<float>();
            Eq::replay_step(state, ctx, for_view, cpml, true, u_this);
            Eq::inject_source_fwd(state, ctx, for_view, p, it, forward_nsrc);
            Eq::swap_recon(forward);
        }

        for (int it = end - 1; it >= start; --it) {
            auto adj_view = adjoint.view();
            Eq::adjoint_step(state, ctx, adj_view, cpml, nullptr, nullptr);
            Eq::inject_adjoint_source(state, ctx, adj_view, p, it, adjoint_nsrc);
            Eq::post_adjoint(adjoint);
            Eq::accumulate_source_grad(state, ctx, adjoint, p, &grad_wavelet,
                                       it, forward_nsrc);
            Eq::image_step(state, ctx,
                           chunk_forward[it - start].template data_ptr<float>(),
                           adjoint.u_now_t.template data_ptr<float>(),
                           vp, &grad, rtm_out);
        }
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

// ---- generic_backward_recursive_ckpt ----
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
        Eq::swap_recon(forward);
    }
}

template <class Eq>
void process_recursive_interval(int start, int end,
                                typename Eq::Wavefield& start_state,
                                typename Eq::Wavefield& adjoint,
                                const BackwardInput& p,
                                const torch::Tensor& vp,
                                torch::Tensor* grad, torch::Tensor* grad_wavelet,
                                RTMOutput* rtm_out,
                                typename Eq::State& state, const SolverContext& ctx,
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
        Eq::swap_recon(start_state);

        auto adj_view = adjoint.view();
        Eq::adjoint_step(state, ctx, adj_view, cpml, nullptr, nullptr);
        Eq::inject_adjoint_source(state, ctx, adj_view, p, start, adjoint_nsrc);
        Eq::post_adjoint(adjoint);
        Eq::accumulate_source_grad(state, ctx, adjoint, p, grad_wavelet,
                                   start, forward_nsrc);
        Eq::image_step(state, ctx, u_this,
                       adjoint.u_now_t.template data_ptr<float>(),
                       vp, grad, rtm_out);
        return;
    }

    int mid = start + (end - start) / 2;
    TORCH_CHECK(scratch_depth < static_cast<int>(scratch_states.size()),
                "Recursive checkpoint scratch depth exhausted.");
    typename Eq::Wavefield& mid_state = scratch_states[scratch_depth];
    checkpoint_runtime.copy_state(mid_state.state_tensors(), start_state.state_tensors());
    advance_forward_interval<Eq>(mid_state, start, mid, state, ctx, p, cpml, forward_nsrc);
    process_recursive_interval<Eq>(mid, end, mid_state, adjoint, p, vp, grad,
                                   grad_wavelet, rtm_out, state, ctx, cpml,
                                   forward_nsrc, adjoint_nsrc, checkpoint_runtime,
                                   scratch_states, scratch_depth + 1, u_this_scratch);
    process_recursive_interval<Eq>(start, mid, start_state, adjoint, p, vp, grad,
                                   grad_wavelet, rtm_out, state, ctx, cpml,
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

    auto grad = torch::zeros_like(vp);
    auto grad_wavelet = torch::zeros_like(p.forward_source);
    RTMOutput illumination;
    init_rtm_output(illumination, vp, in.compute_adcig, 2 * in.adcig_max_lag + 1);
    RTMOutput* rtm_out = (in.compute_illumination || in.compute_adcig)
        ? &illumination : nullptr;

    typename Eq::CPML cpml_tensor;
    Eq::alloc_cpml(cpml_tensor, p);
    auto cpml = cpml_tensor.view();

    auto launch_config = wave_config<Eq::NDIM>(d);
    auto fwd_source_config = fdtd::Geom::make(forward_nsrc, d.B);
    auto adj_source_config = fdtd::Geom::make(adjoint_nsrc, d.B);

    typename Eq::State state = Eq::make_state(p, d, ctx, launch_config,
                                              fwd_source_config,
                                              adj_source_config);

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

    typename Eq::Wavefield start_state;
    Eq::alloc_recursive_start_state(start_state, p, vp);
    Eq::init_aux_slabs(ctx, start_state);

    std::vector<typename Eq::Wavefield> scratch_states(
        recursive_checkpoint_scratch_depth(max_segment_length));
    for (auto& scratch_state : scratch_states)
        scratch_state.allocate_like(vp, start_state);
    auto u_this_scratch = torch::empty_like(vp);

    for (int segment_idx = num_saved_checkpoints; segment_idx >= 0; --segment_idx) {
        int start = (segment_idx == 0) ? 0 : checkpoint_steps[segment_idx - 1];
        int end = (segment_idx == num_saved_checkpoints) ? static_cast<int>(p.nt) : checkpoint_steps[segment_idx];

        if (segment_idx == 0)
            checkpoint_runtime.zero_state(start_state.state_tensors());
        else
            checkpoint_runtime.load(segment_idx - 1, start_state.checkpoint_tensors(),
                                    start_state.next_tensors());

        process_recursive_interval<Eq>(start, end, start_state, adjoint, p, vp,
                                       &grad, &grad_wavelet, rtm_out,
                                       state, ctx, cpml,
                                       forward_nsrc, adjoint_nsrc,
                                       checkpoint_runtime, scratch_states, 0,
                                       u_this_scratch);
    }

    out.grads = {grad_wavelet, grad};
    out.source_illumination = illumination.source_illumination;
    out.receiver_illumination = illumination.receiver_illumination;
    out.adcig = illumination.adcig;
    return out;
}

} // namespace eqdrv
