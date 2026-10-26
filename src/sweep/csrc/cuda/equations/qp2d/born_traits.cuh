// Driver traits of the Born (LSRTM) pair of the pure-qP equations,
// AcousticVTIBorn / AcousticTTIBorn (qP_born.py): a background field and a
// scattered field, each with its own CPML memories and its own S_n, the
// scattered field driven by mp times the background's update (kernels.cuh,
// qp_born_forward).  The backward returns the reflectivity gradient: the
// scattered field's adjoint (the qp adjoint kernels about the scattered state,
// ``s_gradient`` as for the forward pair) imaged against the background update
//
//   grad_mp = sum_k lambda_s^{k+1} r0^k,   r0^k = v^2 dt^2 [..](background^k).
//
// The background models' and the wavelet's gradients need the background
// adjoint and are not computed: the Python side refuses them for impl='c'.
//
// Storage, as driver_traits.cuh with the two fields:
//   * full: history row it = the scattered state of step it + 1 (5 fields)
//     and r0 of step it (the forward kernel writes it), (nt, 6, B, nz, nx);
//     reverse step it reads the state from row it and r0^{it+1} from row it + 1;
//   * boundary saving: both fields' frames (10 fields), both rebuilt in the
//     deep interior; r0 recomputed from the background state.
#pragma once

#include "driver_traits.cuh"

namespace qp2d {

struct BornWavefield {
    AcousticWavefieldTensor bg, sc;
    bool has_bg = false;
    struct View {
        AcousticWavefieldPointer bg{}, sc{};
    };
    View view()
    {
        View v;
        if (has_bg) v.bg = bg.view();
        v.sc = sc.view();
        return v;
    }
    std::vector<Buf> checkpoint_tensors() const { return {}; }   // no checkpointing
};

template <bool TILT>
struct BornDriver {
    static constexpr int NDIM = 2;
    static constexpr const char* NAME = TILT ? "qp_tti_born2d" : "qp_vti_born2d";
    static constexpr int NMODELS = TILT ? 5 : 4;     // vp, epsilon, delta[, theta], mp
    static constexpr int CKPT_NVAR = 6;
    static constexpr int BS_NVAR = 10;               // both frames: p + 4 memories each
    static constexpr int BS_LAST_TWO_NVAR = 2;
    static constexpr int TANGENT_PAD = 0;
    static constexpr int CUT_MASK_BITS = 0;
    static constexpr const char* CUT_MASK_DESC = "none (qp has no domain decomposition)";
    static constexpr int ADJ_WF_COUNT = 18;          // the propagator's adjoint pool; 11 are used
    static constexpr int RECON_WF_COUNT = 14;        // two 7-tensor reconstruction states
    static constexpr int CKPT_STATE_COUNT = 7;
    static constexpr bool HAS_FUSED_FULL_IMG = true;
    static constexpr bool ADCIG_IN_FULL_MODES = false;
    static constexpr bool BS_HAS_IT0_ADJOINT_TAIL = true;
    static constexpr int HISTORY_FIELDS = 6;

    using Wavefield = BornWavefield;
    using CPML = AcousticCPMLTensor;
    using View = BornWavefield::View;
    using Base = Driver<TILT>;

    struct State {
        Models models;
        const float* mp;
        fdtd::LaunchConfig launch_config, source_config, record_config;
        int order, M, nx, nz, B, abcn, exact_s;
    };

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d, const SolverContext&,
                            fdtd::LaunchConfig launch_config, fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        State s;
        s.models.vp = p.models[0].template data_ptr<float>();
        s.models.eps = p.models[1].template data_ptr<float>();
        s.models.del = p.models[2].template data_ptr<float>();
        s.models.th = TILT ? p.models[3].template data_ptr<float>() : nullptr;
        s.mp = p.models[NMODELS - 1].template data_ptr<float>();
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        s.M = p.M;
        s.nx = d.nx;
        s.nz = d.nz;
        s.B = d.B;
        s.abcn = p.abcn;
        s.exact_s = Base::read_exact_s(p);
        return s;
    }

    struct BwdWorkspace {
        Scratch scr{};
        float* grad_mp = nullptr;
        qp2d::State bs_bg{}, bs_sc{};
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
        ws.grad_mp = p.grads_out[NMODELS].template data_ptr<float>();
        return ws;
    }

    struct BsScratch {};

    template <class P>
    static void validate_common(const P& p)
    {
        SWEEP_CHECK(static_cast<int>(p.models.size()) == NMODELS,
                    NAME, " expects ", NMODELS, " models (vp, epsilon, delta",
                    TILT ? ", theta" : "", ", mp), got ", p.models.size());
        SWEEP_CHECK(p.cut_face_mask == 0, NAME, " has no domain-decomposed path");
        SWEEP_CHECK(!p.has_topo && !p.use_apm, NAME, " has no topography");
        SWEEP_CHECK(!p.free_surface && (p.fs_faces <= 0),
                    NAME, ": the isotropic image free surface is wrong for an anisotropic medium");
    }

    static void validate_forward(const ForwardInputCore& p)
    {
        validate_common(p);
        SWEEP_CHECK(!p.use_checkpoint, NAME, " has no checkpoint backward; use memory=Full() or BoundarySaving()");
        SWEEP_CHECK(!p.use_boundary_saving || p.boundary_tail_steps == 0,
                    NAME, " does not implement the truncated boundary backward (tail_steps)");
        SWEEP_CHECK(!p.use_boundary_saving || p.abcn > 0,
                    NAME, " boundary saving needs an absorbing pad (abcn > 0)");
    }

    static void validate_backward(const BackwardInputCore& p, bool need_recon)
    {
        validate_common(p);
        SWEEP_CHECK(!p.compute_illumination && !p.compute_adcig, NAME, " computes no illumination / ADCIG");
        SWEEP_CHECK(!need_recon || p.boundary_tail_steps == 0,
                    NAME, " does not implement the truncated boundary backward (tail_steps)");
        SWEEP_CHECK(!need_recon || p.abcn > 0, NAME, " boundary saving needs an absorbing pad (abcn > 0)");
    }

    template <class P>
    static void setup_ctx(SolverContext& ctx, const P& p) { Base::setup_ctx(ctx, p); }

    static void init_aux_slabs(SolverContext& ctx, Wavefield& wf)
    {
        acoustic_init_aux_slabs(ctx, wf.sc);    // full-grid memories: identity layout
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p) { cpml.bind(p.pml_vals, 2); }

    static std::vector<int64_t> record_shape(const eqdrv::Dims& d, const ForwardInputCore& p)
    {
        return {d.N, p.receivers_loc.size(1), static_cast<int64_t>(p.nt)};
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, HISTORY_FIELDS, d.B, d.nz, d.nx};
    }

    static int save_width(int abcn, int M) { return abcn + M + 1; }
    static int bs_tangent_pad(int abcn, int /*M*/) { return abcn; }

    static std::vector<Buf> slice(const BufList& l, int64_t from, int64_t n)
    {
        return std::vector<Buf>(l.begin() + from, l.begin() + from + n);
    }

    // ===================================================================== //
    // FORWARD
    // ===================================================================== //

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p, const Buf&)
    {
        SWEEP_CHECK(p.wavefields.size() == 18,
                    NAME, "/forward requires the 18 propagator-bound wavefields (background then "
                    "scattered: u triple, psix, psiz, zetax, zetaz, psixn, psizn), got ", p.wavefields.size());
        wf.bg.bind(slice(p.wavefields, 0, 9), 2, true);
        wf.sc.bind(slice(p.wavefields, 9, 9), 2, true);
        wf.has_bg = true;
    }

    static void launch_step_range(const State& s, const SolverContext& ctx, int xb, int xe,
                                  View view, bool /*save_all*/, float* u_thist, AcousticCPMLPointer cpml)
    {
        if (xe <= xb) return;
        SolverContext sctx = ctx;
        sctx.x_base = xb;
        sctx.x_limit = xe;
        auto L = fdtd::Wave2D::make(xe - xb, s.nz, s.B);
        float* rhs0 = u_thist ? u_thist + 5L * s.B * s.nx * s.nz : nullptr;   // history row it, slot 5
        Tables<TILT>::born_forward()[sweep_order_slot(s.order)]<<<L.grid, L.block>>>(
            view.bg, view.sc, s.models, s.mp, cpml, sctx, rhs0);
    }

    static void save_boundary_fwd(BoundaryRuntime& rt, const State& s, const SolverContext& ctx,
                                  const View& view, int it_shifted, int nt_shifted,
                                  const GeneralBoundaryPointer& bs, int save_width)
    {
        float* fields[BS_NVAR] = {view.bg.u_now, view.bg.psixn, view.bg.psizn, view.bg.zetax, view.bg.zetaz,
                                  view.sc.u_now, view.sc.psixn, view.sc.psizn, view.sc.zetax, view.sc.zetaz};
        for (int f = 0; f < BS_NVAR; ++f)
            rt.save_forward_2d_field(it_shifted, nt_shifted, fields[f],
                                     s.launch_config.grid, s.launch_config.block,
                                     bs, save_width, -s.abcn, ctx, f,
                                     /*flush_chunk=*/f == BS_NVAR - 1, /*tangent_pad=*/s.abcn);
    }

    static void inject_source_fwd(const State& s, const SolverContext& ctx, const View& view,
                                  const ForwardInputCore& p, int it, int nsrc)
    {
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            view.bg.u_next, p.source.data_ptr<float>(), p.sources_loc.data_ptr<int>(), it, nsrc, ctx);
    }

    static void record(const State& s, const SolverContext& ctx, const View& view,
                       Buf& record, const ForwardInputCore& p, int it, int nrec)
    {
        record_kernel<<<s.record_config.grid, s.record_config.block>>>(
            view.sc.u_next, record.data_ptr<float>(), p.receivers_loc.data_ptr<int>(), it, nrec, ctx);
    }

    static void rotate_buffers(Wavefield& wf)
    {
        wf.bg.swap_pml();
        wf.sc.swap_pml();
    }

    // Row it, slots 0..4: the scattered state of step it + 1 (slot 5 holds
    // r0^it, written by the forward kernel).
    static void capture_allt(Buf& u_allt, Wavefield& wf, int it)
    {
        if (!u_allt.defined() || u_allt.numel() == 0) return;
        Buf row = u_allt.select(0, it);
        const Buf* src[5] = {&wf.sc.u_next_t, &wf.sc.psixn_t, &wf.sc.psizn_t, &wf.sc.zetax_t, &wf.sc.zetaz_t};
        for (int f = 0; f < 5; ++f)
            copy_tensor_cuda_async(row.select(0, f), *src[f]);
    }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        copy_tensor_cuda_async(saver.last_two.select(0, 0).select(0, 0), wf.bg.u_prev_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 0).select(0, 1), wf.bg.u_now_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 1).select(0, 0), wf.sc.u_prev_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 1).select(0, 1), wf.sc.u_now_t);
    }

    // ===================================================================== //
    // BACKWARD
    // ===================================================================== //

    static void bind_backward_outputs(const BackwardInputCore& p, std::vector<Buf>& grads,
                                      RTMOutputCore&, bool)
    {
        SWEEP_CHECK(p.grads_out.size() == p.models.size() + 1,
                    NAME, " backward requires the propagator-bound grads_out "
                    "(slot 0 = grad_wavelet, then one per model)");
        grads.assign(p.grads_out.begin(), p.grads_out.end());
    }

    static void pack_outputs(BackwardOutputCore& out, std::vector<Buf>& grads, RTMOutputCore&)
    {
        out.grads = grads;
    }

    static RTMOutputCore* rtm_out_full(const BackwardInputCore&, RTMOutputCore&) { return nullptr; }
    static RTMOutputCore* rtm_out_bs(const BackwardInputCore&, RTMOutputCore&) { return nullptr; }
    static float* fused_grad_ptr(std::vector<Buf>& grads) { return grads[NMODELS].data_ptr<float>(); }

    static const float* u_forward_ptr(const BackwardInputCore& p, int k)
    {
        return k >= 1 ? p.u_forward.select(0, k - 1).data_ptr<float>() : nullptr;
    }

    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInputCore& p, const Buf&)
    {
        SWEEP_CHECK(p.adjoint_wavefields.size() >= 11,
                    NAME, "/backward requires at least 11 propagator-bound adjoint_wavefields, got ",
                    p.adjoint_wavefields.size());
        wf.sc.bind(slice(p.adjoint_wavefields, 0, 11), 2, true);
        wf.has_bg = false;
    }

    // Reverse step it (forward step k = it + 1): the reflectivity image of
    // step k, then the scattered adjoint linearised about the scattered state.
    static void adjoint_step(const State& s, const SolverContext& ctx, View adj_view,
                             AcousticCPMLPointer cpml, BwdWorkspace& ws,
                             const float* state_row, float* grad_out)
    {
        qp2d::State sc{nullptr, nullptr, nullptr, nullptr, nullptr};
        const int slot = sweep_order_slot(s.order);
        const long n = (long)s.B * s.nx * s.nz;
        if (state_row != nullptr) {
            sc = qp2d::State{state_row, state_row + n, state_row + 2 * n, state_row + 3 * n, state_row + 4 * n};
            if (grad_out != nullptr) {
                const float* rhs0 = state_row + HISTORY_FIELDS * n + 5 * n;   // row k, slot 5
                Tables<TILT>::born_image()[slot]<<<s.launch_config.grid, s.launch_config.block>>>(
                    adj_view.sc, qp2d::State{nullptr, nullptr, nullptr, nullptr, nullptr}, rhs0,
                    s.models, cpml, ctx, ws.grad_mp);
            }
        } else if (ws.have_bs_state) {
            sc = ws.bs_sc;
            Tables<TILT>::born_image()[slot]<<<s.launch_config.grid, s.launch_config.block>>>(
                adj_view.sc, ws.bs_bg, nullptr, s.models, cpml, ctx, ws.grad_mp);
        }
        Tables<TILT>::prep()[slot]<<<s.launch_config.grid, s.launch_config.block>>>(
            adj_view.sc, sc, s.models, cpml, ctx, ws.scr, Grads{nullptr, nullptr, nullptr, nullptr},
            s.exact_s);
        qp_adjoint_apply_by_order[slot]<<<s.launch_config.grid, s.launch_config.block>>>(
            adj_view.sc, ws.scr, cpml, ctx);
    }

    static void inject_adjoint_source(const State& s, const SolverContext& ctx, const View& adj_view,
                                      const BackwardInputCore& p, int it, int nsrc, BwdWorkspace&)
    {
        add_source<<<s.record_config.grid, s.record_config.block>>>(
            adj_view.sc.u_next, p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(), it, nsrc, ctx);
    }

    static void rotate_adjoint_buffers(Wavefield& wf) { wf.sc.swap_aux(); }

    // The wavelet enters the background field; its gradient needs the
    // background adjoint, which this backward does not run (refused in Python).
    static void accumulate_source_grad(const State&, const SolverContext&, Wavefield&,
                                       const BackwardInputCore&, std::vector<Buf>&, int, int) {}

    static void image_step(const State&, const SolverContext&, const float*, int,
                           Wavefield&, std::vector<Buf>*, RTMOutputCore*, BwdWorkspace&) {}

    static void bind_or_alloc_recon(Wavefield& wf, const BackwardInputCore& p, const Buf&)
    {
        SWEEP_CHECK(p.forward_wavefields.size() == RECON_WF_COUNT,
                    NAME, "/backward_bs requires the ", RECON_WF_COUNT, " propagator-bound "
                    "reconstruction grids (cuda_layout.bs_reconstruction_nvar), got ",
                    p.forward_wavefields.size());
        wf.bg.bind(slice(p.forward_wavefields, 0, 7), 2, true);
        wf.sc.bind(slice(p.forward_wavefields, 7, 7), 2, true);
        wf.has_bg = true;
    }

    // u_last_two: [field (bg, sc), time (prev, now), B, 1, nz, nx].
    static void seed_reconstruction(const State&, const SolverContext&, Wavefield& forward,
                                    const BackwardInputCore& p)
    {
        copy_tensor_cuda_async(forward.bg.u_prev_t, p.u_last_two.select(0, 0).select(0, 1));
        copy_tensor_cuda_async(forward.bg.u_now_t, p.u_last_two.select(0, 0).select(0, 0));
        copy_tensor_cuda_async(forward.sc.u_prev_t, p.u_last_two.select(0, 1).select(0, 1));
        copy_tensor_cuda_async(forward.sc.u_now_t, p.u_last_two.select(0, 1).select(0, 0));
    }

    static BsScratch make_bs_scratch(const BackwardInputCore&, const Buf&) { return {}; }
    static void bs_illum_tail(const State&, const SolverContext&, Wavefield&, RTMOutputCore&) {}

    static void bs_recon_step(const State& s, const SolverContext& ctx, Wavefield& forward,
                              Wavefield& /*adjoint*/, BoundaryRuntime& boundary_runtime,
                              const GeneralBoundaryPointer& bs, int save_width,
                              AcousticCPMLPointer cpml, const BackwardInputCore& p,
                              std::vector<Buf>&, RTMOutputCore*, BwdWorkspace& ws, BsScratch&,
                              int it, int bs_it0)
    {
        auto v = forward.view();
        Tables<TILT>::born_reverse()[sweep_order_slot(s.order)]<<<s.launch_config.grid, s.launch_config.block>>>(
            v.bg, v.sc, s.models, s.mp, cpml, ctx, save_width);
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            v.bg.u_next, p.forward_source.data_ptr<float>(), p.forward_sources_loc.data_ptr<int>(), it,
            (int)p.forward_sources_loc.size(1), ctx);
        float* fields[BS_NVAR] = {v.bg.u_next, v.bg.psix, v.bg.psiz, v.bg.zetax, v.bg.zetaz,
                                  v.sc.u_next, v.sc.psix, v.sc.psiz, v.sc.zetax, v.sc.zetaz};
        for (int f = 0; f < BS_NVAR; ++f)
            boundary_runtime.restore_backward_2d_field(
                it - bs_it0, fields[f], s.launch_config.grid, s.launch_config.block,
                bs, save_width, -s.abcn, ctx, f,
                /*wait_chunk=*/f == 0, /*record_done=*/f == BS_NVAR - 1, /*tangent_pad=*/s.abcn);
        forward.bg.swap();
        forward.sc.swap();
        auto state_of = [](AcousticWavefieldTensor& w) {
            return qp2d::State{w.u_prev_t.data_ptr<float>(), w.psix_t.data_ptr<float>(),
                               w.psiz_t.data_ptr<float>(), w.zetax_t.data_ptr<float>(),
                               w.zetaz_t.data_ptr<float>()};
        };
        ws.bs_bg = state_of(forward.bg);
        ws.bs_sc = state_of(forward.sc);
        ws.have_bs_state = true;
    }

    static void bs_rtm_tap(const State&, const SolverContext&, Wavefield&, Wavefield&, RTMOutputCore&, bool) {}
};

} // namespace qp2d
