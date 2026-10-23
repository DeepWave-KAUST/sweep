// Driver traits for the 2-D pure-qP VTI / TTI equations (kernels.cuh): the
// per-equation half of the shared skeleton in ``common/eq_driver.cuh``, for
// the forward, the full-storage backward and the boundary-saving backward
// (no checkpointing, no domain decomposition, no illumination / ADCIG).
//
// What differs from acoustic2d, and why: the qP operator is NONLINEAR in p
// (S depends on grad p), so its adjoint step k is the transpose of the step
// linearised about the forward state of step k -- p^k and the four CPML
// memories psix^k, psiz^k, zetax^k, zetaz^k.  Every backward mode therefore
// hands the adjoint the whole forward state of step it + 1 at reverse step it
// (the step whose lambda^{it+2} the adjoint is transposing), and the model
// gradient of step it + 1 is imaged in the same pass (fused).
//
//   * forward: the memories are full grids; psi is double-buffered (read at
//     neighbours), zeta is updated in place (own cell).  After the step,
//     (psixn, psizn, zetax, zetaz) hold the memories of step it + 1.
//   * full mode: capture_allt stores the state of step it + 1 -- p^{it+1}
//     (u_next after the source) and its memories -- as history row it,
//     (nt, 5, B, nz, nx).  Reverse step it reads row it.
//   * boundary saving: forward step it saves, on the whole CPML frame (the
//     absorbing pad plus M + 1 cells of the physical box, all four faces,
//     corners included), p^it and the memories of step it + 1.  Reverse step
//     it rebuilds p^{it-1} in the deep interior, adds the step-it source,
//     restores the frame saved at step it - 1 (p^{it-1} and the memories of
//     step it) and leaves (p^it, memories^it) for reverse step it - 1.
//
// The adjoint step is two launches: qp_adjoint_prep (state-dependent
// coefficients, CPML adjoints of the own cell, the fused model gradient) and
// qp_adjoint_apply (the transposed stencils).
//
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/eq_driver.cuh.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../../core/runner.h"
#include "../../common/eq_driver.cuh"
#include "../../launch/config.h"

// Defined with the acoustic2d kernels (acoustic2d/kernels.cu): lambda at the
// source cells into grad_wavelet, the same imaging of the wavelet gradient.
__global__ void accumulate_source_grad_2d(
    const float* __restrict__ u_backward,   // (B, nz, nx)
    float* __restrict__ grad_source,        // (B, nsrc, nt)
    const int* __restrict__ sources_loc,    // (B, nsrc, 2)
    int it,
    int nsrc,
    SolverContext solver
);

namespace qp2d {

template <bool TILT>
struct Driver {
    static constexpr int NDIM = 2;
    static constexpr const char* NAME = TILT ? "qp_tti2d" : "qp_vti2d";
    static constexpr int NMODELS = TILT ? 4 : 3;     // vp, epsilon, delta[, theta]
    static constexpr int CKPT_NVAR = 6;              // never used: no checkpoint mode
    static constexpr int BS_NVAR = 5;                // p + psix, psiz, zetax, zetaz on the frame
    static constexpr int BS_LAST_TWO_NVAR = 2;       // u_prev, u_now
    static constexpr int TANGENT_PAD = 0;            // superseded by bs_tangent_pad()
    static constexpr int CUT_MASK_BITS = 0;          // no domain decomposition
    static constexpr const char* CUT_MASK_DESC = "none (qp has no domain decomposition)";
    static constexpr int ADJ_WF_COUNT = 11;          // u triple + psi/zeta double-buffer
    static constexpr int RECON_WF_COUNT = 7;         // u triple + the restored memories
    static constexpr int CKPT_STATE_COUNT = 7;
    static constexpr bool HAS_FUSED_FULL_IMG = true; // the model gradient rides qp_adjoint_prep
    static constexpr bool ADCIG_IN_FULL_MODES = false;
    static constexpr bool BS_HAS_IT0_ADJOINT_TAIL = true;
    static constexpr int HISTORY_FIELDS = 5;

    using Wavefield = AcousticWavefieldTensor;
    using CPML = AcousticCPMLTensor;

    struct State {
        Models models;
        fdtd::LaunchConfig launch_config;
        fdtd::LaunchConfig source_config;
        fdtd::LaunchConfig record_config;
        int order;
        int M;
        int nx, nz, B;
        int abcn;
        int exact_s;    // the s_gradient mode: 0 frozen, 1 exact, 2 adjoint
    };

    // eq_aux = {one float32 CPU flag}: the equation's s_gradient (qP_vti.s_gradient_aux).
    template <class P>
    static int read_exact_s(const P& p)
    {
        SWEEP_CHECK(p.eq_aux.size() == 1 && p.eq_aux[0].numel() == 1 && !p.eq_aux[0].is_cuda(),
                    NAME, " expects eq_aux = {one float32 CPU flag} (s_gradient), got ",
                    p.eq_aux.size(), " tensors");
        const int mode = static_cast<int>(p.eq_aux[0].template data_ptr<float>()[0]);
        SWEEP_CHECK(mode >= 0 && mode <= 2, NAME, ": s_gradient flag ", mode, " is not 0/1/2");
        return mode;
    }

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d,
                            const SolverContext& /*ctx*/,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        State s;
        s.models.vp = p.models[0].template data_ptr<float>();
        s.models.eps = p.models[1].template data_ptr<float>();
        s.models.del = p.models[2].template data_ptr<float>();
        s.models.th = TILT ? p.models[3].template data_ptr<float>() : nullptr;
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        s.M = p.M;
        s.nx = d.nx;
        s.nz = d.nz;
        s.B = d.B;
        s.abcn = p.abcn;
        s.exact_s = read_exact_s(p);
        return s;
    }

    // Per backward call: the adjoint scratch, the model-gradient accumulators
    // and, in boundary-saving mode, the forward state the next reverse step
    // reads (set by bs_recon_step; none before the first reverse step).
    struct BwdWorkspace {
        Scratch scr{};
        Grads grads{};
        qp2d::State bs_state{};
        bool have_bs_state = false;
    };

    static BwdWorkspace make_bwd_workspace(const BackwardInputCore& p, const State&,
                                           const SolverContext&, Wavefield&)
    {
        SWEEP_CHECK(p.adjoint_workspace.size() == 4,
                    NAME, " backward requires the propagator-bound adjoint_workspace "
                    "(cuda_layout.backward_workspace_nvar = 4), got ", p.adjoint_workspace.size());
        BwdWorkspace ws;
        ws.scr.tx = p.adjoint_workspace[0].template data_ptr<float>();
        ws.scr.tz = p.adjoint_workspace[1].template data_ptr<float>();
        ws.scr.dux = p.adjoint_workspace[2].template data_ptr<float>();
        ws.scr.duz = p.adjoint_workspace[3].template data_ptr<float>();
        ws.grads.vp = p.grads_out[1].template data_ptr<float>();
        ws.grads.eps = p.grads_out[2].template data_ptr<float>();
        ws.grads.del = p.grads_out[3].template data_ptr<float>();
        ws.grads.th = TILT ? p.grads_out[4].template data_ptr<float>() : nullptr;
        return ws;
    }

    struct BsScratch {};

    template <class P>
    static void validate_common(const P& p)
    {
        SWEEP_CHECK(static_cast<int>(p.models.size()) == NMODELS,
                    NAME, " expects ", NMODELS, " models (vp, epsilon, delta",
                    TILT ? ", theta" : "", "), got ", p.models.size());
        SWEEP_CHECK(p.cut_face_mask == 0, NAME, " has no domain-decomposed path");
        SWEEP_CHECK(!p.has_topo && !p.use_apm, NAME, " has no topography");
        SWEEP_CHECK(!p.free_surface && (p.fs_faces <= 0),
                    NAME, ": the isotropic image free surface is wrong for an anisotropic medium");
    }

    static void validate_forward(const ForwardInputCore& p)
    {
        validate_common(p);
        SWEEP_CHECK(!p.use_checkpoint,
                    NAME, " has no checkpoint backward (the adjoint of reverse step it needs "
                    "the forward state of step it + 1); use memory=Full() or BoundarySaving()");
        SWEEP_CHECK(!p.use_boundary_saving || p.boundary_tail_steps == 0,
                    NAME, " does not implement the truncated boundary backward (tail_steps)");
        SWEEP_CHECK(!p.use_boundary_saving || p.abcn > 0,
                    NAME, " boundary saving needs an absorbing pad (abcn > 0)");
    }

    static void validate_backward(const BackwardInputCore& p, bool need_recon)
    {
        validate_common(p);
        SWEEP_CHECK(!p.compute_illumination && !p.compute_adcig,
                    NAME, " computes no illumination / ADCIG");
        SWEEP_CHECK(!need_recon || p.boundary_tail_steps == 0,
                    NAME, " does not implement the truncated boundary backward (tail_steps)");
        SWEEP_CHECK(!need_recon || p.abcn > 0,
                    NAME, " boundary saving needs an absorbing pad (abcn > 0)");
    }

    template <class P>
    static void setup_ctx(SolverContext& ctx, const P& p)
    {
        ctx.topo_rows = nullptr;
        ctx.has_topo = false;
        ctx.topo_category = nullptr;
        ctx.use_apm = false;
        ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    }

    static void init_aux_slabs(SolverContext& ctx, Wavefield& wf)
    {
        acoustic_init_aux_slabs(ctx, wf);    // full-grid memories: identity layout
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.bind(p.pml_vals, 2);
    }

    static std::vector<int64_t> record_shape(const eqdrv::Dims& d, const ForwardInputCore& p)
    {
        return {d.N, p.receivers_loc.size(1), static_cast<int64_t>(p.nt)};
    }

    // Full-mode history: the forward state of step it + 1 in row it.
    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, HISTORY_FIELDS, d.B, d.nz, d.nx};
    }

    // The CPML frame: offset -abcn from the physical box, abcn + M + 1 wide,
    // abcn past the box tangentially (cuda_layout.boundary_frame).
    static int save_width(int abcn, int M) { return abcn + M + 1; }
    static int bs_tangent_pad(int abcn, int /*M*/) { return abcn; }

    // ===================================================================== //
    // FORWARD
    // ===================================================================== //

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p, const Buf&)
    {
        SWEEP_CHECK(p.wavefields.size() == 9,
                    NAME, "/forward requires the 9 propagator-bound wavefields "
                    "(u triple, psix, psiz, zetax, zetaz, psixn, psizn), got ", p.wavefields.size());
        wf.bind(p.wavefields, 2, true);
    }

    static void launch_step_range(const State& s, const SolverContext& ctx, int xb, int xe,
                                  AcousticWavefieldPointer view, bool /*save_all*/,
                                  float* /*u_thist*/, AcousticCPMLPointer cpml)
    {
        if (xe <= xb) return;
        SolverContext sctx = ctx;
        sctx.x_base = xb;
        sctx.x_limit = xe;
        auto L = fdtd::Wave2D::make(xe - xb, s.nz, s.B);
        Tables<TILT>::forward()[sweep_order_slot(s.order)]<<<L.grid, L.block>>>(
            view, s.models, cpml, sctx);
    }

    static void save_boundary_fwd(BoundaryRuntime& rt, const State& s,
                                  const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  int it_shifted, int nt_shifted,
                                  const GeneralBoundaryPointer& bs, int save_width)
    {
        // p^it and the memories of step it + 1 (written by the step just run).
        float* fields[BS_NVAR] = {view.u_now, view.psixn, view.psizn, view.zetax, view.zetaz};
        for (int f = 0; f < BS_NVAR; ++f) {
            rt.save_forward_2d_field(it_shifted, nt_shifted, fields[f],
                                     s.launch_config.grid, s.launch_config.block,
                                     bs, save_width, -s.abcn, ctx, f,
                                     /*flush_chunk=*/f == BS_NVAR - 1,
                                     /*tangent_pad=*/s.abcn);
        }
    }

    static void inject_source_fwd(const State& s, const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  const ForwardInputCore& p, int it, int nsrc)
    {
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            view.u_next, p.source.data_ptr<float>(), p.sources_loc.data_ptr<int>(), it, nsrc, ctx);
    }

    static void record(const State& s, const SolverContext& ctx,
                       const AcousticWavefieldPointer& view,
                       Buf& record, const ForwardInputCore& p, int it, int nrec)
    {
        record_kernel<<<s.record_config.grid, s.record_config.block>>>(
            view.u_next, record.data_ptr<float>(), p.receivers_loc.data_ptr<int>(), it, nrec, ctx);
    }

    static void rotate_buffers(Wavefield& wf)
    {
        wf.swap_pml();   // u triple + psi <-> psin; zeta is in place
    }

    // Full mode: the state of step it + 1 (p after the source, the memories
    // the step just wrote) as history row it.  Before rotate_buffers.
    static void capture_allt(Buf& u_allt, Wavefield& wf, int it)
    {
        if (!u_allt.defined() || u_allt.numel() == 0) return;
        Buf row = u_allt.select(0, it);
        const Buf* src[HISTORY_FIELDS] = {&wf.u_next_t, &wf.psixn_t, &wf.psizn_t,
                                          &wf.zetax_t, &wf.zetaz_t};
        for (int f = 0; f < HISTORY_FIELDS; ++f)
            copy_tensor_cuda_async(row.select(0, f), *src[f]);
    }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        copy_tensor_cuda_async(saver.last_two.select(1, 0), wf.u_prev_t);
        copy_tensor_cuda_async(saver.last_two.select(1, 1), wf.u_now_t);
    }

    // ===================================================================== //
    // BACKWARD (shared + full mode)
    // ===================================================================== //

    static void bind_backward_outputs(const BackwardInputCore& p,
                                      std::vector<Buf>& grads,
                                      RTMOutputCore& /*illumination*/, bool /*want_adcig*/)
    {
        SWEEP_CHECK(p.grads_out.size() == p.models.size() + 1,
                    NAME, " backward requires the propagator-bound grads_out "
                    "(slot 0 = grad_wavelet, then one per model)");
        grads.assign(p.grads_out.begin(), p.grads_out.end());
    }

    static void pack_outputs(BackwardOutputCore& out, std::vector<Buf>& grads,
                             RTMOutputCore& /*illumination*/)
    {
        out.grads = grads;
    }

    static RTMOutputCore* rtm_out_full(const BackwardInputCore&, RTMOutputCore&) { return nullptr; }
    static RTMOutputCore* rtm_out_bs(const BackwardInputCore&, RTMOutputCore&) { return nullptr; }

    // Non-null = "image in the adjoint step"; the gradients themselves are
    // the workspace's four accumulators.
    static float* fused_grad_ptr(std::vector<Buf>& grads) { return grads[1].data_ptr<float>(); }

    // The forward state of step k is history row k - 1 (see capture_allt);
    // there is no row for step 0, whose state is zero.
    static const float* u_forward_ptr(const BackwardInputCore& p, int k)
    {
        return k >= 1 ? p.u_forward.select(0, k - 1).data_ptr<float>() : nullptr;
    }

    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInputCore& p, const Buf&)
    {
        SWEEP_CHECK(p.adjoint_wavefields.size() == ADJ_WF_COUNT,
                    NAME, "/backward requires the ", ADJ_WF_COUNT, " propagator-bound "
                    "adjoint_wavefields (cuda_layout.adjoint_extra_nvar = 2), got ",
                    p.adjoint_wavefields.size());
        wf.bind(p.adjoint_wavefields, 2, true);
    }

    // Reverse step it: the transpose of forward step it + 1.  ``state_row``
    // is history row it (full mode, null on the first reverse step); in
    // boundary saving it is null and the state is the one bs_recon_step left.
    static void adjoint_step(const State& s, const SolverContext& ctx,
                             AcousticWavefieldPointer adj_view,
                             AcousticCPMLPointer cpml, BwdWorkspace& ws,
                             const float* state_row, float* grad_out)
    {
        qp2d::State st{nullptr, nullptr, nullptr, nullptr, nullptr};
        Grads gr{nullptr, nullptr, nullptr, nullptr};
        if (state_row != nullptr) {
            const long n = (long)s.B * s.nx * s.nz;
            st = qp2d::State{state_row, state_row + n, state_row + 2 * n,
                             state_row + 3 * n, state_row + 4 * n};
            if (grad_out != nullptr) gr = ws.grads;
        } else if (ws.have_bs_state) {
            st = ws.bs_state;
            gr = ws.grads;
        }
        const int slot = sweep_order_slot(s.order);
        Tables<TILT>::prep()[slot]<<<s.launch_config.grid, s.launch_config.block>>>(
            adj_view, st, s.models, cpml, ctx, ws.scr, gr, s.exact_s);
        qp_adjoint_apply_by_order[slot]<<<s.launch_config.grid, s.launch_config.block>>>(
            adj_view, ws.scr, cpml, ctx);
    }

    static void inject_adjoint_source(const State& s, const SolverContext& ctx,
                                      const AcousticWavefieldPointer& adj_view,
                                      const BackwardInputCore& p, int it, int nsrc,
                                      BwdWorkspace&)
    {
        add_source<<<s.record_config.grid, s.record_config.block>>>(
            adj_view.u_next, p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(), it, nsrc, ctx);
    }

    static void rotate_adjoint_buffers(Wavefield& wf)
    {
        wf.swap_aux();
    }

    static void accumulate_source_grad(const State& s, const SolverContext& ctx,
                                       Wavefield& adjoint, const BackwardInputCore& p,
                                       std::vector<Buf>& grads, int it, int nsrc)
    {
        accumulate_source_grad_2d<<<s.source_config.grid, s.source_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(), grads[0].data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(), it, nsrc, ctx);
    }

    // The gradient is fused into adjoint_step; the trailing step-0 image is
    // zero (the state of step 0 is zero).
    static void image_step(const State&, const SolverContext&, const float*, int,
                           Wavefield&, std::vector<Buf>*, RTMOutputCore*, BwdWorkspace&) {}

    // ===================================================================== //
    // BACKWARD_BS
    // ===================================================================== //

    static void bind_or_alloc_recon(Wavefield& wf, const BackwardInputCore& p, const Buf&)
    {
        SWEEP_CHECK(p.forward_wavefields.size() == RECON_WF_COUNT,
                    NAME, "/backward_bs requires the ", RECON_WF_COUNT, " propagator-bound "
                    "reconstruction grids (cuda_layout.bs_reconstruction_nvar), got ",
                    p.forward_wavefields.size());
        wf.bind(p.forward_wavefields, 2, true);
    }

    // u_prev <- p^nt, u_now <- p^{nt-1}: full fields, nothing to zero (the
    // reverse step only writes the deep interior, the frame is restored).
    static void seed_reconstruction(const State&, const SolverContext&,
                                    Wavefield& forward, const BackwardInputCore& p)
    {
        copy_tensor_cuda_async(forward.u_prev_t, p.u_last_two.select(1, 1).squeeze(0));
        copy_tensor_cuda_async(forward.u_now_t, p.u_last_two.select(1, 0).squeeze(0));
    }

    static BsScratch make_bs_scratch(const BackwardInputCore&, const Buf&) { return {}; }

    static void bs_illum_tail(const State&, const SolverContext&, Wavefield&, RTMOutputCore&) {}

    static void bs_recon_step(const State& s, const SolverContext& ctx,
                              Wavefield& forward, Wavefield& /*adjoint*/,
                              BoundaryRuntime& boundary_runtime,
                              const GeneralBoundaryPointer& bs, int save_width,
                              AcousticCPMLPointer cpml,
                              const BackwardInputCore& p,
                              std::vector<Buf>& /*grads*/,
                              RTMOutputCore* /*rtm_out*/,
                              BwdWorkspace& ws,
                              BsScratch& /*scratch*/,
                              int it, int bs_it0)
    {
        auto for_view = forward.view();
        // p^{it-1} on the deep interior, without the step-it source ...
        Tables<TILT>::reverse()[sweep_order_slot(s.order)]<<<s.launch_config.grid, s.launch_config.block>>>(
            for_view, s.models, cpml, ctx, save_width);
        // ... plus it; a source cell inside the frame is overwritten below.
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            for_view.u_next, p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(), it,
            (int)p.forward_sources_loc.size(1), ctx);
        // The frame saved at forward step it - 1: p^{it-1} and the memories of
        // step it, which reverse step it - 1 reads.
        float* fields[BS_NVAR] = {for_view.u_next, for_view.psix, for_view.psiz,
                                  for_view.zetax, for_view.zetaz};
        for (int f = 0; f < BS_NVAR; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it - bs_it0, fields[f], s.launch_config.grid, s.launch_config.block,
                bs, save_width, -s.abcn, ctx, f,
                /*wait_chunk=*/f == 0, /*record_done=*/f == BS_NVAR - 1,
                /*tangent_pad=*/s.abcn);
        }
        forward.swap();   // u_prev <- p^it, u_now <- p^{it-1}
        ws.bs_state = qp2d::State{forward.u_prev_t.data_ptr<float>(),
                                  forward.psix_t.data_ptr<float>(),
                                  forward.psiz_t.data_ptr<float>(),
                                  forward.zetax_t.data_ptr<float>(),
                                  forward.zetaz_t.data_ptr<float>()};
        ws.have_bs_state = true;
    }

    static void bs_rtm_tap(const State&, const SolverContext&, Wavefield&, Wavefield&,
                           RTMOutputCore&, bool) {}
};

} // namespace qp2d
