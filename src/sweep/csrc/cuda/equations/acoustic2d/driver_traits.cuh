// Driver traits for the 2-D acoustic equation: the per-equation half of the
// shared skeleton in ``common/eq_driver.cuh``.  Everything here is a
// line-faithful transcription of the launches the hand-written drivers made;
// the physics kernels are untouched.
//
// REFERENCE EQUATION of the acoustic family.  The other acoustic-family
// members (acoustic3d, acoustic_vrz2d) describe their deltas against this
// baseline; the properties below are what "same as acoustic2d" means:
//   * constants: NDIM = 2, CKPT_NVAR = 6, BS_NVAR = 1 (the saver stores u only), BS_LAST_TWO_NVAR = 2 (u_prev, u_now);
//   * TANGENT_PAD = 0 (no tangential pad on the boundary strips); CUT_MASK_BITS = 0xF (x_lo, x_hi, z_lo, z_hi);
//   * ADJ_WF_COUNT = 11 (u triple + psi/zeta double-buffer), RECON_WF_COUNT = 3;
//   * HAS_FUSED_FULL_IMG = true: full mode folds the lagged vp-gradient imaging into the adjoint kernel (fused_grad_ptr = grads[1]; one trailing image_step at it == 0);
//   * ADCIG_IN_FULL_MODES = true: the flag opens rtm_out_full on compute_adcig as well as on compute_illumination;
//   * BS_HAS_IT0_ADJOINT_TAIL = true: the bs reverse loop runs an it == 0 adjoint-only tail (grad_wavelet);
//   * one model, vp = p.models[0], held as a raw pointer in State beside the Laplace/gradient parameter blocks, the launch configs, order/M, nx/nz/B and has_topo;
//   * BwdWorkspace and BsScratch are empty: the fused adjoint keeps its scratch in the psi/zeta buffers and the NOPML kernel writes no per-step scratch field;
//   * validate_forward / validate_backward are no-ops;
//   * setup_ctx installs topo_rows / has_topo (topo_category = nullptr, use_apm = false) and the per-edge free surface via ctx.set_per_edge(fs_faces, pad_lo, pad_hi);
//   * init_aux_slabs = acoustic_init_aux_slabs (CPML aux slabs); alloc_cpml = cpml.allocate(pml_vals, 2);
//   * allt_shape = (nt, B, nz, nx): u only, written in-kernel through u_thist (capture_allt is a no-op);
//   * save_width = abcn > 0 ? M + 1 : M; boundary save/restore offset 0;
//   * bind_or_alloc_forward binds p.wavefields or allocates with double_buffer_psi = true; bind_or_alloc_adjoint binds p.adjoint_wavefields or allocate(vp, 2, true); bind_or_alloc_recon binds/allocates WITHOUT CPML (use_pml = false);
//   * launch_step_range: an air-clear prepass (has_topo only, x range widened by M) then the ACOUSTIC2D kernel ranged over x in [xb, xe) via ctx.x_base / x_limit (phase-split capable);
//   * forward loop: inject_source_fwd and record act on u_next; rotate_buffers = swap_pml (u triple AND psi<->psin); save_last_state stores (u_prev, u_now) in last_two;
//   * backward outputs: alloc_grads = {grad_wavelet, grad_vp}; bind_backward_outputs = eqdrv::acoustic_bind_backward_outputs; pack_outputs returns grads + source/receiver illumination + adcig;
//   * u_forward_ptr = u_forward[it] (vp^2 * Lap(u), the vp-gradient operand, NOT raw pressure);
//   * adjoint_step = ACOUSTIC2D_ADJOINT_FUSED: single-kernel exact adjoint, g_* recomputed at each tap, next-step psi/zeta written to the double-buffer out-tensors, grad_forward_img / grad_out fused in;
//   * inject_adjoint_source adds p.adjoint_source into u_next with the record_config slot (the adjoint source config in backward states); rotate_adjoint_buffers = swap_aux (u + psi + zeta rotation);
//   * accumulate_source_grad = accumulate_source_grad_2d from the adjoint u_now into grads[0];
//   * image_step = calculate_grad (vp gradient) + accumulate_rtm_image_2d; no ADCIG launch here (the full store is not raw pressure);
//   * rtm_out_bs opens on compute_illumination || compute_adcig (consumed in-step by the 3-D twin; 2-D images after the prefetch in bs_rtm_tap);
//   * seed_reconstruction: u_prev <- u_last_two[:, 1], u_now <- u_last_two[:, 0], then set_boundary_zeros on both over the abcn + M rim with the cut faces excluded (ctx.cut_mask()); make_bs_scratch returns {};
//   * bs_recon_step order: ACOUSTIC2D_NOPML (+ fused vp imaging on the cells the restore will not overwrite) -> restore_backward_2d -> calculate_grad_utt_band on the restore strips -> forward-source add_source into u_next -> forward.swap();
//   * bs_rtm_tap, after the prefetch: accumulate_rtm_image_2d gated by compute_illumination, then accumulate_adcig_2d when illumination.adcig is non-empty, both on the co-resident (forward.u_now, adjoint.u_now) pair;
//   * ckpt: bind_or_alloc_recon_ckpt binds p.forward_wavefields or allocate_from_snapshots(vp, checkpoints, 2); alloc_recursive_start_state = allocate_from_snapshots;
//   * ckpt replay: replay_step = full-domain ACOUSTIC2D (no air-clear prepass, no x range) -> inject_source_fwd (BackwardInput overload) -> rotate_recon_buffers = swap().
//
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/eq_driver.cuh.
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../common/eq_driver.cuh"
#include "../../launch/config.h"
#include "../../operators/laplace.cuh"
#include "../../operators/gradient.cuh"

namespace acoustic2d {

struct Driver {
    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // Constants, type aliases, State/workspace types, validation, and the
    // prologue hooks.  Skeleton prologue, in call order (every entry point):
    //   validate_forward / (backward: check_stepped + validate_backward +
    //   bind_backward_outputs / alloc_grads + rtm gate), bind_or_alloc_*
    //   wavefields, alloc_cpml, setup_ctx, init_aux_slabs, make_state,
    //   make_bwd_workspace.
    // ===================================================================== //

    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "acoustic2d";
    static constexpr int CKPT_NVAR = 6;
    static constexpr int BS_NVAR = 1;           // the saver stores u only
    static constexpr int BS_LAST_TWO_NVAR = 2;  // u_prev, u_now
    static constexpr int TANGENT_PAD = 0;       // (x TANGENT_PAD*M; VRZ uses 1)
    static constexpr int CUT_MASK_BITS = 0xF;
    static constexpr const char* CUT_MASK_DESC = "bits 0..3 (x_lo, x_hi, z_lo, z_hi)";
    static constexpr int ADJ_WF_COUNT = 11;     // u triple + psi/zeta double-buffer
    static constexpr int RECON_WF_COUNT = 3;
    // Full mode folds the lagged vp-gradient imaging into the adjoint kernel.
    static constexpr bool HAS_FUSED_FULL_IMG = true;
    // 2-D serves ADCIG from full/ckpt modes too (raw-pressure imaging).
    static constexpr bool ADCIG_IN_FULL_MODES = true;
    // The bs reverse loop runs an it==0 adjoint-only tail (grad_wavelet).
    static constexpr bool BS_HAS_IT0_ADJOINT_TAIL = true;

    using Wavefield = AcousticWavefieldTensor;
    using CPML = AcousticCPMLTensor;

    // Per-call bundle built once before the time loop: model pointer, operator
    // parameter blocks, launch configs, and the few scalars the hooks need.
    struct State {
        const float* vp;
        LaplaceParam lap_ctx;
        GradParam grad_ctx;
        GradParam grad_ctx_x;
        GradParam grad_ctx_z;
        fdtd::LaunchConfig launch_config;
        fdtd::LaunchConfig source_config;
        fdtd::LaunchConfig record_config;
        int order;
        int M;
        int nx, nz, B;
        bool has_topo;
    };

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d,
                            const SolverContext& /*ctx*/,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        float dx = p.spacing[0];
        float dz = p.spacing[1];
        State s;
        s.vp = p.models[0].template data_ptr<float>();
        s.lap_ctx = LaplaceParam{d.nx, 1, p.M, p.lap_coes.template data_ptr<float>(), dx, 0.f, dz};
        s.grad_ctx = GradParam{1, 0, d.nx, p.M, p.grad_coes.template data_ptr<float>(), dx, 0.f, dz};
        s.grad_ctx_x = GradParam{1, 0, 0, p.M, p.grad_coes.template data_ptr<float>(), dx, 0.f, 0.f};
        s.grad_ctx_z = GradParam{1, 0, 0, p.M, p.grad_coes.template data_ptr<float>(), dz, 0.f, 0.f};
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        s.M = p.M;
        s.nx = d.nx;
        s.nz = d.nz;
        s.B = d.B;
        s.has_topo = p.has_topo;
        return s;
    }

    struct BwdWorkspace {};   // fused adjoint keeps scratch in psi/zeta buffers
    static BwdWorkspace make_bwd_workspace(const BackwardInput&, const State&,
                                           const SolverContext&, Wavefield&)
    { return {}; }

    struct BsScratch {};   // 2-D NOPML writes no per-step scratch field
    // (its factory make_bs_scratch lives in section [4], at its firing point)

    static void validate_forward(const ForwardInput&) {}
    static void validate_backward(const BackwardInput&, bool) {}

    template <class P>
    static void setup_ctx(SolverContext& ctx, const P& p)
    {
        ctx.topo_rows    = p.has_topo ? p.topo_rows.template data_ptr<int>() : nullptr;
        ctx.has_topo     = p.has_topo;
        ctx.topo_category = nullptr;
        ctx.use_apm      = false;
        ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    }

    static void init_aux_slabs(SolverContext& ctx, Wavefield& wf)
    {
        acoustic_init_aux_slabs(ctx, wf);
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.allocate(p.pml_vals, 2);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, d.B, d.nz, d.nx};
    }

    static int save_width(int abcn, int M) { return abcn > 0 ? M + 1 : M; }

    // ===================================================================== //
    // [2] FORWARD — generic_forward, per it in [it_begin, it_end):
    //   launch_step_range -> save_boundary_fwd -> inject_source_fwd ->
    //   record -> rotate_buffers -> capture_allt -> <checkpoint save (shared)>;
    //   after the loop: save_last_state (final u pair for backward_bs).
    // ===================================================================== //

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.wavefields.empty())
            wf.bind(p.wavefields, 2, true);
        else
            wf.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    }

    // One forward step over x in [xb, xe).
    // Pre-pass: clear air cells in a separate kernel launch so the
    // main acoustic2nd kernel only reads (never writes) air cells.
    // Eliminates intra-launch RAW race on PML aux fields that was
    // showing up as ~30% non-deterministic forward output across
    // processes (sweep VTI history pattern).  The air-clear range is
    // widened by the stencil halo M so a phase-split stencil launch
    // still only reads air cells cleared earlier THIS step (re-clearing
    // across phases writes the same zeros — idempotent).
    static void launch_step_range(const State& s, const SolverContext& ctx,
                                  int xb, int xe,
                                  AcousticWavefieldPointer view,
                                  bool save_all, float* u_thist,
                                  AcousticCPMLPointer cpml)
    {
        if (xe <= xb) return;
        if (s.has_topo) {
            int axb = std::max(0, xb - s.M);
            int axe = std::min(s.nx, xe + s.M);
            SolverContext actx = ctx;
            actx.x_base = axb;
            actx.x_limit = axe;
            auto alc = fdtd::Wave2D::make(axe - axb, s.nz, s.B);
            acoustic2d_air_clear_kernel<<<alc.grid, alc.block>>>(
                view, save_all, u_thist, actx
            );
        }
        SolverContext sctx = ctx;
        sctx.x_base = xb;
        sctx.x_limit = xe;
        auto lc = fdtd::Wave2D::make(xe - xb, s.nz, s.B);
        ACOUSTIC2D(
            s.order,
            lc.grid,
            lc.block,
            view,
            save_all,
            u_thist,
            s.vp,
            s.lap_ctx,
            s.grad_ctx,
            s.grad_ctx_x,
            s.grad_ctx_z,
            cpml,
            sctx
        );
    }

    static void save_boundary_fwd(BoundaryRuntime& rt, const State& s,
                                  const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  int it_shifted, int nt_shifted,
                                  const GeneralBoundaryPointer& bs, int save_width)
    {
        rt.save_forward_2d(
            it_shifted,
            nt_shifted,
            view.u_now,
            s.launch_config.grid,
            s.launch_config.block,
            bs,
            save_width,
            0,
            ctx
        );
    }

    // (also fired by the ckpt/recursive replay loops; the BackwardInput
    // overload for those lives in section [5])
    static void inject_source_fwd(const State& s, const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  const ForwardInput& p, int it, int nsrc)
    {
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            view.u_next,
            p.source.data_ptr<float>(),
            p.sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
    }

    static void record(const State& s, const SolverContext& ctx,
                       const AcousticWavefieldPointer& view,
                       torch::Tensor& record, const ForwardInput& p,
                       int it, int nrec)
    {
        record_kernel<<<s.record_config.grid, s.record_config.block>>>(
            view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );
    }

    static void rotate_buffers(Wavefield& wf)
    {
        wf.swap_pml();   // rotate u AND psi<->psin: race-free psi double-buffer
    }

    static void capture_allt(torch::Tensor&, Wavefield&, int) {}

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        saver.last_two_t.select(1, 0).copy_(wf.u_prev_t);
        saver.last_two_t.select(1, 1).copy_(wf.u_now_t);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — generic_backward, per reverse it:
    //   adjoint_step -> inject_adjoint_source -> rotate_adjoint_buffers ->
    //   accumulate_source_grad -> image_step;
    //   after the loop (fused only): one trailing image_step at it == 0.
    //   (adjoint_step .. accumulate_source_grad also fire per reverse it in
    //   the bs / ckpt / recursive modes of sections [4] and [5].)
    // ===================================================================== //

    static void bind_backward_outputs(const BackwardInput& p,
                                      std::vector<torch::Tensor>& grads,
                                      RTMOutput& illumination, bool want_adcig)
    {
        eqdrv::acoustic_bind_backward_outputs<Driver>(p, grads, illumination,
                                                      want_adcig);
    }

    static void alloc_grads(const BackwardInput& p,
                            std::vector<torch::Tensor>& grads)
    {
        grads = {torch::zeros_like(p.forward_source),
                 torch::zeros_like(p.models[0])};
    }

    static void pack_outputs(BackwardOutput& out,
                             std::vector<torch::Tensor>& grads,
                             RTMOutput& illumination)
    {
        out.grads = {grads[0], grads[1]};
        out.source_illumination = illumination.source_illumination;
        out.receiver_illumination = illumination.receiver_illumination;
        out.adcig = illumination.adcig;
    }

    static RTMOutput* rtm_out_full(const BackwardInput& p, RTMOutput& illumination)
    {
        return (p.compute_illumination ||
                (ADCIG_IN_FULL_MODES && p.compute_adcig))
            ? &illumination : nullptr;
    }

    static float* fused_grad_ptr(std::vector<torch::Tensor>& grads)
    {
        return grads[1].data_ptr<float>();
    }

    static const float* u_forward_ptr(const BackwardInput& p, int it)
    {
        return p.u_forward[it].data_ptr<float>();
    }

    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.adjoint_wavefields.empty())
            wf.bind(p.adjoint_wavefields, 2, true);
        else
            wf.allocate(vp, 2, true);
    }

    // FUSED single-kernel exact adjoint: recompute the per-cell g_* inline at
    // each stencil tap and write next-step psi/zeta into the wavefield's
    // double-buffer out-tensors (psixn/psizn/zetaxn/zetazn).  Read-old/write-new
    // => race-free even though aux is read at neighbours; no 6-field scratch
    // round-trip => ~forward bandwidth.  Caller must swap_aux() each step.
    static void adjoint_step(const State& s, const SolverContext& ctx,
                             AcousticWavefieldPointer adj_view,
                             AcousticCPMLPointer cpml, BwdWorkspace&,
                             const float* grad_forward_img, float* grad_out)
    {
        TORCH_CHECK(adj_view.zetaxn != nullptr && adj_view.psixn != nullptr,
            "fused adjoint needs the adjoint wavefield bound with psi+zeta "
            "double-buffer (11 tensors in 2D); set cuda_layout.adjoint_extra_nvar=2.");
        ACOUSTIC2D_ADJOINT_FUSED(s.order, s.launch_config.grid, s.launch_config.block,
            adj_view, s.vp, s.lap_ctx, s.grad_ctx_x, s.grad_ctx_z, cpml, ctx,
            adj_view.psixn, adj_view.psizn, adj_view.zetaxn, adj_view.zetazn,
            grad_forward_img, grad_out);
    }

    static void inject_adjoint_source(const State& s, const SolverContext& ctx,
                                      const AcousticWavefieldPointer& adj_view,
                                      const BackwardInput& p, int it, int nsrc,
                                      BwdWorkspace&)
    {
        // record_config slot carries the ADJOINT source config in backward states.
        add_source<<<s.record_config.grid, s.record_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
    }

    static void rotate_adjoint_buffers(Wavefield& wf)
    {
        wf.swap_aux();   // fused adjoint: rotate u + psi + zeta double-buffer
    }

    static void accumulate_source_grad(const State& s, const SolverContext& ctx,
                                       Wavefield& adjoint, const BackwardInput& p,
                                       std::vector<torch::Tensor>& grads,
                                       int it, int nsrc)
    {
        accumulate_source_grad_2d<<<s.source_config.grid, s.source_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(),
            grads[0].data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
    }

    static void image_step(const State& s, const SolverContext& ctx,
                           const float* forward_ptr, Wavefield& adjoint,
                           std::vector<torch::Tensor>* grads,
                           RTMOutput* rtm_out, BwdWorkspace&)
    {
        const float* adjoint_ptr = adjoint.u_now_t.data_ptr<float>();
        if (grads != nullptr) {
            calculate_grad<<<s.launch_config.grid, s.launch_config.block>>>(
                forward_ptr, adjoint_ptr,
                s.vp,
                (*grads)[1].data_ptr<float>(),
                s.nx, s.nz, ctx.dt,
                ctx.phys_x0(), ctx.phys_x1(), ctx.phys_z0(), ctx.phys_z1()
            );
        }
        if (rtm_out != nullptr) {
            accumulate_rtm_image_2d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward_ptr, adjoint_ptr,
                rtm_out->image.data_ptr<float>(),
                rtm_out->source_illumination.data_ptr<float>(),
                rtm_out->receiver_illumination.data_ptr<float>(),
                s.nx, s.nz
            );
        }
        // NOTE: ADCIG is NOT accumulated here.  This hook's ``forward_ptr`` is
        // the full/checkpoint forward store, which for acoustic is vp^2*Lap(u)
        // (kept for the vp gradient), NOT the raw pressure the space-lag imaging
        // condition needs.  ADCIG is launched only from bs_rtm_tap, where the
        // reconstructed raw pressure is co-resident.
    }

    // ===================================================================== //
    // [4] BACKWARD_BS — generic_backward_bs, per reverse it (floor
    // max(max(it_lo, 1), bs_stop)):
    //   adjoint_step / inject_adjoint_source / rotate_adjoint_buffers /
    //   accumulate_source_grad (section [3]) -> bs_recon_step ->
    //   bs_rtm_tap;
    //   before the loop (first segment): seed_reconstruction from u_last_two;
    //   after the loop (BS_HAS_IT0_ADJOINT_TAIL): the four adjoint hooks once at it == 0.
    // ===================================================================== //

    static RTMOutput* rtm_out_bs(const BackwardInput& p, RTMOutput& illumination)
    {
        // 2-D images after the prefetch (bs_rtm_tap); the gated pointer is
        // consumed by the 3-D twin's in-step imaging and ignored here.
        return (p.compute_illumination || p.compute_adcig)
            ? &illumination : nullptr;
    }

    static void bind_or_alloc_recon(Wavefield& wf, const BackwardInput& p,
                                    const torch::Tensor& vp)
    {
        if (!p.forward_wavefields.empty())
            wf.bind(p.forward_wavefields, 2, false);
        else
            wf.allocate(vp, 2, false);
    }

    // Seed the reverse reconstruction from the saved last two snapshots, then
    // zero the absorbing rim (cut faces excluded) — FIRST segment only.
    static void seed_reconstruction(const State& s, const SolverContext& ctx,
                                    Wavefield& forward, const BackwardInput& p)
    {
        forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
        forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
        auto for_view = forward.view();
        set_boundary_zeros<<<s.launch_config.grid, s.launch_config.block>>>(
            for_view.u_prev, ctx.abcn + ctx.M, s.nx, s.nz,
            ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2), ctx.cut_mask());
        set_boundary_zeros<<<s.launch_config.grid, s.launch_config.block>>>(
            for_view.u_now, ctx.abcn + ctx.M, s.nx, s.nz,
            ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2), ctx.cut_mask());
    }

    static BsScratch make_bs_scratch(const BackwardInput&, const torch::Tensor&)
    { return {}; }

    // One boundary-saving reverse-reconstruction step, in THIS equation's
    // bit-load-bearing order: NOPML step, strip restore, u_tt gradient
    // imaging, forward source injection, swap.  (The 3-D twin images before
    // the injection; VRZ injects before the restore — order lives here.)
    static void bs_recon_step(const State& s, const SolverContext& ctx,
                                Wavefield& forward, Wavefield& adjoint,
                                BoundaryRuntime& boundary_runtime,
                                const GeneralBoundaryPointer& bs, int save_width,
                                AcousticCPMLPointer /*cpml*/,
                                const BackwardInput& p,
                                std::vector<torch::Tensor>& grads,
                                RTMOutput* /*rtm_out: 2-D images after the
                                             prefetch, in bs_rtm_tap*/,
                                BwdWorkspace& /*ws*/,
                                BsScratch& /*scratch*/,
                                int it, int bs_it0)
    {
        auto for_view = forward.view();
        // Reconstruction with the vp-gradient imaging fused in (every computed
        // cell the restore below will not overwrite); the restore strips are
        // imaged by the band kernel after the restore.  Same per-cell
        // expression and operands as the former standalone calculate_grad_utt.
        ACOUSTIC2D_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            s.vp,
            s.lap_ctx,
            ctx,
            adjoint.u_now_t.data_ptr<float>(),
            grads[1].data_ptr<float>(),
            save_width
        );
        boundary_runtime.restore_backward_2d(
            it - bs_it0,
            for_view.u_next,
            s.launch_config.grid,
            s.launch_config.block,
            bs,
            save_width,
            0,
            ctx
        );
        {
            const int wxl = ctx.cut_x_lo() ? 0 : save_width;
            const int wxh = ctx.cut_x_hi() ? 0 : save_width;
            const int wzl = ctx.cut_z_lo() ? 0 : save_width;
            const int wzh = ctx.cut_z_hi() ? 0 : save_width;
            const int bw = ctx.phys_x1() - ctx.phys_x0();
            const int bh = ctx.phys_z1() - ctx.phys_z0() - wzl - wzh;
            const int n_strip = (wzl + wzh) * bw + (wxl + wxh) * (bh > 0 ? bh : 0);
            if (n_strip > 0) {
                dim3 band_grid((n_strip + 255) / 256, s.B);
                calculate_grad_utt_band<<<band_grid, 256>>>(
                    forward.u_prev_t.data_ptr<float>(),
                    for_view.u_next,
                    forward.u_now_t.data_ptr<float>(),
                    adjoint.u_now_t.data_ptr<float>(),
                    s.vp,
                    grads[1].data_ptr<float>(),
                    s.nx, s.nz, ctx.dt,
                    ctx.phys_x0(), ctx.phys_x1(), ctx.phys_z0(), ctx.phys_z1(),
                    wxl, wxh, wzl, wzh
                );
            }
        }
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            /*nsrc=*/(int)p.forward_sources_loc.size(1),
            ctx
        );
        forward.swap();
    }

    static void bs_rtm_tap(const State& s, const SolverContext& /*ctx*/,
                              Wavefield& forward, Wavefield& adjoint,
                              RTMOutput& illumination, bool compute_illumination)
    {
        // Gate illumination on compute_illumination (mirror FULL path). When
        // off, skip the per-step RTM pass entirely; the FWI vp-gradient is
        // produced by calculate_grad_utt and is unaffected.
        if (compute_illumination) {
            accumulate_rtm_image_2d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward.u_now_t.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                illumination.image.data_ptr<float>(),
                illumination.source_illumination.data_ptr<float>(),
                illumination.receiver_illumination.data_ptr<float>(),
                s.nx, s.nz
            );
        }
        // Space-lag ADCIG rides the same co-resident (u_s(t), u_r(t)) pair.
        if (illumination.adcig.defined() && illumination.adcig.numel() > 0) {
            int nlag = illumination.adcig.size(0);
            int Bloc = illumination.adcig.size(1) * illumination.adcig.size(2);
            int max_lag = (nlag - 1) / 2;
            accumulate_adcig_2d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward.u_now_t.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                illumination.adcig.data_ptr<float>(),
                nlag, max_lag, Bloc, s.nx, s.nz
            );
        }
    }

    // ===================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING — generic_backward_ckpt, per chunk:
    //   replay:  replay_step -> inject_source_fwd -> rotate_recon_buffers
    //   reverse: adjoint_step / inject_adjoint_source / rotate_adjoint_buffers /
    //            accumulate_source_grad / image_step (section [3]).
    //   generic_backward_recursive_ckpt bisects each ckpt segment; a leaf
    //   runs one replay triple, then the reverse-five of ckpt mode with the
    //   imaging fed from the leaf's scratch u.
    // ===================================================================== //

    static void bind_or_alloc_recon_ckpt(Wavefield& wf, const BackwardInput& p,
                                         const torch::Tensor& vp)
    {
        if (!p.forward_wavefields.empty())
            wf.bind(p.forward_wavefields, 2, true);
        else
            // Aux shapes must follow the Python-allocated checkpoint slots
            // (possibly per-axis slabs); a plain allocate() would build
            // full-domain aux and break the snapshot copies.
            wf.allocate_from_snapshots(vp, p.checkpoints, 2);
    }

    static void alloc_recursive_start_state(Wavefield& wf, const BackwardInput& p,
                                            const torch::Tensor& vp)
    {
        wf.allocate_from_snapshots(vp, p.checkpoints, 2);
    }

    static void replay_step(const State& s, const SolverContext& ctx,
                            AcousticWavefieldPointer view,
                            AcousticCPMLPointer cpml,
                            bool save_all, float* u_this)
    {
        ACOUSTIC2D(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            view,
            save_all,
            u_this,
            s.vp,
            s.lap_ctx,
            s.grad_ctx,
            s.grad_ctx_x,
            s.grad_ctx_z,
            cpml,
            ctx
        );
    }

    // Overload for ckpt replay: BackwardInput spells the forward source fields
    // differently from ForwardInput.
    static void inject_source_fwd(const State& s, const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  const BackwardInput& p, int it, int nsrc)
    {
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
    }

    static void rotate_recon_buffers(Wavefield& wf) { wf.swap(); }
};

} // namespace acoustic2d
