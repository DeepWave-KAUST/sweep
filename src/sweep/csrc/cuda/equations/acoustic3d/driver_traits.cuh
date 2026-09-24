// Driver traits for the 3-D acoustic equation (see acoustic2d/driver_traits.cuh
// for the pattern).  Line-faithful transcription of the hand-written drivers;
// physics kernels untouched.  Deltas vs 2-D worth naming:
//   * no ctx.set_per_edge anywhere — per-edge free surface is 2-D only;
//   * the fused adjoint carries a psi AND zeta triple double-buffer
//     (15 adjoint tensors, adjoint_extra_nvar=3);
//   * CKPT_STATE_COUNT = 9 (u triple + the 6 CPML aux slabs) per checkpoint
//     replay state set of p.forward_wavefields;
//   * the boundary-saving reverse step images BEFORE the forward source
//     injection (2-D images after injection + swap), and its NOPML kernel
//     writes a per-step scratch field (BsScratch.f_this);
//   * ADCIG is served only by backward_bs (full/ckpt imaging correlates
//     vp^2*Lap(u), not raw pressure), and there is no seed rim-zeroing;
//   * the old hand-written backward_bs built its SolverContext with nullptr
//     lap/grad coefficient pointers; the shared skeleton passes the real
//     pointers everywhere.  Nothing on the bs path dereferences them (it
//     could not have run before otherwise), so this is bit-inert.
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

namespace acoustic3d {

struct Driver {
    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // Constants, nested types, and the prologue hooks every entry point runs
    // before its time loop (per the timing map, in call order):
    //   validate_forward / (backward: validate_backward +
    //   bind_backward_outputs + rtm gate), bind_or_alloc_*
    //   wavefields, alloc_cpml, setup_ctx, init_aux_slabs, make_state,
    //   make_bwd_workspace.
    // ===================================================================== //

    static constexpr int NDIM = 3;
    static constexpr const char* NAME = "acoustic3d";
    static constexpr int CKPT_NVAR = 8;
    static constexpr int BS_NVAR = 1;
    static constexpr int BS_LAST_TWO_NVAR = 2;
    static constexpr int TANGENT_PAD = 0;
    static constexpr int CUT_MASK_BITS = 0x3F;
    static constexpr const char* CUT_MASK_DESC =
        "bits 0..5 (x_lo, x_hi, z_lo, z_hi, y_lo, y_hi)";
    static constexpr int ADJ_WF_COUNT = 15;     // u triple + psi/zeta triple double-buffer
    static constexpr int RECON_WF_COUNT = 3;
    // One checkpoint replay state set (ckpt / recursive backward), in bind
    // order: u_prev, u_now, u_next + the 6 CPML aux slabs = CKPT_NVAR + u_next.
    static constexpr int CKPT_STATE_COUNT = CKPT_NVAR + 1;
    static constexpr bool HAS_FUSED_FULL_IMG = true;
    static constexpr bool ADCIG_IN_FULL_MODES = false;
    static constexpr bool BS_HAS_IT0_ADJOINT_TAIL = true;
    using Wavefield = AcousticWavefieldTensor;
    using CPML = AcousticCPMLTensor;

    struct State {
        const float* vp;
        LaplaceParam lap_ctx;
        GradParam grad_ctx;
        GradParam grad_ctx_x;
        GradParam grad_ctx_y;
        GradParam grad_ctx_z;
        fdtd::LaunchConfig launch_config;
        fdtd::LaunchConfig source_config;
        fdtd::LaunchConfig record_config;
        int order;
        int M;
        int nx, ny, nz, B;
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
        float dy = p.spacing[1];
        float dz = p.spacing[2];
        State s;
        s.vp = p.models[0].template data_ptr<float>();
        s.lap_ctx = LaplaceParam{d.nx, d.ny, p.M, p.lap_coes.template data_ptr<float>(), dx, dy, dz};
        s.grad_ctx = GradParam{1, d.nx, d.nx * d.ny, p.M, p.grad_coes.template data_ptr<float>(), dx, dy, dz};
        s.grad_ctx_x = GradParam{1, 0, 0, p.M, p.grad_coes.template data_ptr<float>(), dx, 0.f, 0.f};
        s.grad_ctx_y = GradParam{1, 0, 0, p.M, p.grad_coes.template data_ptr<float>(), dy, 0.f, 0.f};
        s.grad_ctx_z = GradParam{1, 0, 0, p.M, p.grad_coes.template data_ptr<float>(), dz, 0.f, 0.f};
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        s.M = p.M;
        s.nx = d.nx;
        s.ny = d.ny;
        s.nz = d.nz;
        s.B = d.B;
        s.has_topo = p.has_topo;
        return s;
    }

    struct BwdWorkspace {};   // fused adjoint keeps scratch in psi/zeta buffers
    static BwdWorkspace make_bwd_workspace(const BackwardInput&, const State&,
                                           const SolverContext&, Wavefield&)
    { return {}; }

    // (factory make_bs_scratch lives in section [4] BACKWARD_BS)
    struct BsScratch {
        torch::Tensor f_this;   // per-step NOPML output field
    };

    static void validate_forward(const ForwardInput&) {}
    static void validate_backward(const BackwardInput&, bool) {}

    template <class P>
    static void setup_ctx(SolverContext& ctx, const P& p)
    {
        // No set_per_edge: per-edge free surface is not wired for 3-D.
        if (p.has_topo) {
            ctx.topo_rows = p.topo_rows.template data_ptr<int>();
            ctx.has_topo  = true;
        }
    }

    static void init_aux_slabs(SolverContext& ctx, Wavefield& wf)
    {
        acoustic_init_aux_slabs(ctx, wf);
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.allocate(p.pml_vals, 3);
    }

    // The record this equation writes: one field, {N, nrec, nt}.  Moved here
    // verbatim from the skeleton -- same expression, same operands.
    static std::vector<int64_t> record_shape(const eqdrv::Dims& d,
                                             const ForwardInput& p)
    {
        return {d.N, p.receivers_loc.size(1), static_cast<int64_t>(p.nt)};
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, d.B, d.nz, d.ny, d.nx};
    }

    static int save_width(int abcn, int M) { return abcn > 0 ? M + 1 : M; }

    // ===================================================================== //
    // [2] FORWARD — generic_forward, per it in [it_begin, it_end):
    //   launch_step_range -> save_boundary_fwd -> inject_source_fwd ->
    //   record -> rotate_buffers -> capture_allt -> <checkpoint save>;
    //   after the loop: save_last_state.
    // ===================================================================== //

    // MANDATORY: the propagator hands the forward state on every call --
    // persistent buffers in save_all mode, a per-call transient set otherwise
    // (_c.py PropBase.forward: ``forward_wavefields = self._slice_wavefield_buffers(...)``
    // then ``if not forward_wavefields: ... _transient_forward_wavefields(...)``,
    // both sized by cuda_layout.base_nvar + pml_nvar = 3 + 9).  bind() checks the
    // count (3 / 9 / 12 / 15) and installs the psi double-buffer for the 12-slot
    // layout, which is why the driver must not build its own.
    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& /*vp*/)
    {
        TORCH_CHECK(!p.wavefields.empty(),
                    "acoustic3d/forward requires the propagator-bound wavefields "
                    "(cuda_layout.base_nvar + cuda_layout.pml_nvar)");
        wf.bind(p.wavefields, 3, true);
    }

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
            auto alc = fdtd::Wave3D::make(axe - axb, s.ny, s.nz, s.B);
            acoustic3d_air_clear_kernel<<<alc.grid, alc.block>>>(
                view, save_all, u_thist, actx
            );
        }
        SolverContext sctx = ctx;
        sctx.x_base = xb;
        sctx.x_limit = xe;
        auto lc = fdtd::Wave3D::make(xe - xb, s.ny, s.nz, s.B);
        ACOUSTIC3D(
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
            s.grad_ctx_y,
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
        rt.save_forward_3d(
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

    // (the ckpt/recursive replay uses the BackwardInput overload in section [5])
    static void inject_source_fwd(const State& s, const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  const ForwardInput& p, int it, int nsrc)
    {
        add_source_3d<<<s.source_config.grid, s.source_config.block>>>(
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
        record_kernel_3d<<<s.record_config.grid, s.record_config.block>>>(
            view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );
    }

    static void rotate_buffers(Wavefield& wf) { wf.swap_pml(); }

    static void capture_allt(torch::Tensor&, Wavefield&, int) {}

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        copy_tensor_cuda_async(saver.last_two.select(1, 0), wf.u_prev_t);
        copy_tensor_cuda_async(saver.last_two.select(1, 1), wf.u_now_t);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — generic_backward, per reverse it:
    //   adjoint_step -> inject_adjoint_source -> rotate_adjoint_buffers ->
    //   accumulate_source_grad -> image_step (fused here, so image_step is
    //   skipped in-loop except for RTM; one trailing image_step at it == 0
    //   after the loop).
    // ===================================================================== //

    static void bind_backward_outputs(const BackwardInput& p,
                                      std::vector<torch::Tensor>& grads,
                                      RTMOutput& illumination, bool want_adcig)
    {
        eqdrv::acoustic_bind_backward_outputs<Driver>(p, grads, illumination,
                                                      want_adcig);
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


    // MANDATORY: a backward only runs when something required a gradient, which
    // is the same predicate that allocates the adjoint pool (_c.py
    // PropBase.forward: ``_ensure_wavefield_buffers(..., need_adjoint=requires_backward)``),
    // and Wrapper.backward binds it zeroed on every call (``params.adjoint_wavefields =
    // [a.zero_() for a in cp.adjoint_wavefields]``), 15 tensors for this equation
    // (base_nvar + pml_nvar + cuda_layout.adjoint_extra_nvar = 3 + 9 + 3).
    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInput& p,
                                      const torch::Tensor& /*vp*/)
    {
        TORCH_CHECK(!p.adjoint_wavefields.empty(),
                    "acoustic3d/backward requires the propagator-bound "
                    "adjoint_wavefields (cuda_layout.adjoint_extra_nvar on top of "
                    "base_nvar + pml_nvar)");
        wf.bind(p.adjoint_wavefields, 3, true);
    }

    // FUSED single-kernel exact 3D adjoint (see acoustic2d twin).
    static void adjoint_step(const State& s, const SolverContext& ctx,
                             AcousticWavefieldPointer adj_view,
                             AcousticCPMLPointer cpml, BwdWorkspace&,
                             const float* grad_forward_img, float* grad_out)
    {
        TORCH_CHECK(adj_view.zetaxn != nullptr && adj_view.psixn != nullptr,
            "fused 3D adjoint needs the adjoint wavefield bound with psi+zeta "
            "double-buffer (15 tensors); set cuda_layout.adjoint_extra_nvar=3.");
        ACOUSTIC3D_ADJOINT_FUSED(s.order, s.launch_config.grid, s.launch_config.block,
            adj_view, s.vp, s.lap_ctx,
            s.grad_ctx_x, s.grad_ctx_y, s.grad_ctx_z, cpml, ctx,
            adj_view.psixn, adj_view.psiyn, adj_view.psizn,
            adj_view.zetaxn, adj_view.zetayn, adj_view.zetazn,
            grad_forward_img, grad_out);
    }

    static void inject_adjoint_source(const State& s, const SolverContext& ctx,
                                      const AcousticWavefieldPointer& adj_view,
                                      const BackwardInput& p, int it, int nsrc,
                                      BwdWorkspace&)
    {
        add_source_3d<<<s.record_config.grid, s.record_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
    }

    static void rotate_adjoint_buffers(Wavefield& wf) { wf.swap_aux(); }

    static void accumulate_source_grad(const State& s, const SolverContext& ctx,
                                       Wavefield& adjoint, const BackwardInput& p,
                                       std::vector<torch::Tensor>& grads,
                                       int it, int nsrc)
    {
        accumulate_source_grad_3d<<<s.source_config.grid, s.source_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(),
            grads[0].data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
    }

    static void image_step(const State& s, const SolverContext& ctx,
                           const float* forward_ptr, int /*it*/, Wavefield& adjoint,
                           std::vector<torch::Tensor>* grads,
                           RTMOutput* rtm_out, BwdWorkspace&)
    {
        const float* adjoint_ptr = adjoint.u_now_t.data_ptr<float>();
        if (grads != nullptr) {
            calculate_grad_3d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward_ptr, adjoint_ptr,
                s.vp,
                (*grads)[1].data_ptr<float>(),
                s.B, s.nx, s.ny, s.nz, ctx.dt,
                ctx.phys_x0(), ctx.phys_x1(), ctx.phys_y0(), ctx.phys_y1(),
                ctx.phys_z0(), ctx.phys_z1()
            );
        }
        if (rtm_out != nullptr) {
            accumulate_illumination_3d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward_ptr, nullptr, nullptr,   // the store IS u_tt
                adjoint_ptr,
                rtm_out->source_illumination.data_ptr<float>(),
                rtm_out->receiver_illumination.data_ptr<float>(),
                s.B, s.nx, s.ny, s.nz, ctx.dt,
                ctx.phys_x0(), ctx.phys_x1(), ctx.phys_y0(), ctx.phys_y1(),
                ctx.phys_z0(), ctx.phys_z1()
            );
        }
        // No ADCIG here: this forward store is vp^2*Lap(u), not raw pressure.
    }

    // ===================================================================== //
    // [4] BACKWARD_BS — generic_backward_bs, per reverse it (floor
    // max(max(it_lo, 1), bs_stop)):
    //   adjoint_step / inject_adjoint_source / rotate_adjoint_buffers /
    //   accumulate_source_grad (the four from section [3]) ->
    //   bs_recon_step -> bs_rtm_tap;
    //   before the loop (first segment): seed_reconstruction; after the loop
    //   (BS_HAS_IT0_ADJOINT_TAIL): the four adjoint hooks once at it == 0.
    // ===================================================================== //

    static RTMOutput* rtm_out_bs(const BackwardInput& p, RTMOutput& illumination)
    {
        return (p.compute_illumination || p.compute_adcig)
            ? &illumination : nullptr;
    }

    // MANDATORY: the boundary-saving backward is handed its reconstruction state
    // per call (_c.py Wrapper.backward bs branch: ``params.forward_wavefields =
    // _forward_state_buffers(cp.forward_state_shapes, ...)`` with
    // ``_forward_state_shapes(..., "bs")`` = cuda_layout.reconstruction_nvar = 3
    // grids from slot_table.ACOUSTIC3D.recon).  bind() checks the count (3, the
    // no-PML layout).
    static void bind_or_alloc_recon(Wavefield& wf, const BackwardInput& p,
                                    const torch::Tensor& /*vp*/)
    {
        TORCH_CHECK(!p.forward_wavefields.empty(),
                    "acoustic3d/backward_bs requires the propagator-bound "
                    "reconstruction state forward_wavefields "
                    "(cuda_layout.bs_reconstruction_nvar / slots.recon)");
        wf.bind(p.forward_wavefields, 3, false);
    }

    // Seed from the saved last two snapshots only — 3-D has no rim-zeroing.
    static void seed_reconstruction(const State& /*s*/, const SolverContext& /*ctx*/,
                                    Wavefield& forward, const BackwardInput& p)
    {
        copy_tensor_cuda_async(forward.u_prev_t, p.u_last_two.select(1, 1).squeeze(0));
        copy_tensor_cuda_async(forward.u_now_t, p.u_last_two.select(1, 0).squeeze(0));
    }

    static BsScratch make_bs_scratch(const BackwardInput& /*p*/,
                                     const torch::Tensor& vp)
    {
        return {torch::Tensor()};   // f_this retired: nothing consumed the field
    }

    // 3-D bs reverse step: NOPML(f_this) -> strip restore -> u_tt gradient +
    // rtm + ADCIG imaging (BEFORE the forward source injection — the 2-D twin
    // images after injection + swap) -> inject -> swap.
    // Receiver-only illumination for the it == 0 tail; see the call site.
    static void bs_illum_tail(const State& s, const SolverContext& ctx,
                              Wavefield& adjoint, RTMOutput& illumination)
    {
        accumulate_illumination_3d<<<s.launch_config.grid, s.launch_config.block>>>(
            nullptr, nullptr, nullptr,
            adjoint.u_now_t.data_ptr<float>(),
            /*source_illumination=*/nullptr,
            illumination.receiver_illumination.data_ptr<float>(),
            s.B, s.nx, s.ny, s.nz, ctx.dt,
            ctx.phys_x0(), ctx.phys_x1(), ctx.phys_y0(), ctx.phys_y1(),
            ctx.phys_z0(), ctx.phys_z1()
        );
    }

    static void bs_recon_step(const State& s, const SolverContext& ctx,
                                Wavefield& forward, Wavefield& adjoint,
                                BoundaryRuntime& boundary_runtime,
                                const GeneralBoundaryPointer& bs, int save_width,
                                AcousticCPMLPointer /*cpml*/,
                                const BackwardInput& p,
                                std::vector<torch::Tensor>& grads,
                                RTMOutput* rtm_out,
                                BwdWorkspace& /*ws*/,
                                BsScratch& scratch,
                                int it, int bs_it0)
    {
        auto for_view = forward.view();
        ACOUSTIC3D_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            (float*)nullptr,   // f_this was a dead per-step store (nothing reads it)
            s.vp,
            s.lap_ctx,
            ctx,
            adjoint.u_now_t.data_ptr<float>(),   // fused vp-gradient imaging
            grads[1].data_ptr<float>(),
            save_width
        );
        boundary_runtime.restore_backward_3d(
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
            // Restore strips: the only box cells the fused imaging skipped.
            const int wxl = ctx.cut_x_lo() ? 0 : save_width;
            const int wxh = ctx.cut_x_hi() ? 0 : save_width;
            const int wyl = ctx.cut_y_lo() ? 0 : save_width;
            const int wyh = ctx.cut_y_hi() ? 0 : save_width;
            const int wzl = ctx.cut_z_lo() ? 0 : save_width;
            const int wzh = ctx.cut_z_hi() ? 0 : save_width;
            const int bx = ctx.phys_x1() - ctx.phys_x0();
            const int by = ctx.phys_y1() - ctx.phys_y0();
            const int bzi = ctx.phys_z1() - ctx.phys_z0() - wzl - wzh;
            const int byi = by - wyl - wyh;
            const int n_strip = (wzl + wzh) * bx * by
                              + (wyl + wyh) * bx * (bzi > 0 ? bzi : 0)
                              + (wxl + wxh) * (byi > 0 ? byi : 0) * (bzi > 0 ? bzi : 0);
            if (n_strip > 0) {
                dim3 band_grid((n_strip + 255) / 256, s.B);
                calculate_grad_utt_3d_band<<<band_grid, 256>>>(
                    forward.u_prev_t.data_ptr<float>(),
                    for_view.u_next,
                    forward.u_now_t.data_ptr<float>(),
                    adjoint.u_now_t.data_ptr<float>(),
                    s.vp,
                    grads[1].data_ptr<float>(),
                    s.nx, s.ny, s.nz, ctx.dt,
                    ctx.phys_x0(), ctx.phys_x1(), ctx.phys_y0(), ctx.phys_y1(),
                    ctx.phys_z0(), ctx.phys_z1(),
                    wxl, wxh, wyl, wyh, wzl, wzh
                );
            }
        }
        if (rtm_out != nullptr) {
            // The three time levels calculate_grad_utt_3d_band just used, so
            // boundary saving reports the same pseudo-Hessian the full store
            // does. Its own kernel: no gradient arithmetic is touched.
            accumulate_illumination_3d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward.u_prev_t.data_ptr<float>(),
                for_view.u_next,
                forward.u_now_t.data_ptr<float>(),
                adjoint.u_now_t.data_ptr<float>(),
                rtm_out->source_illumination.data_ptr<float>(),
                rtm_out->receiver_illumination.data_ptr<float>(),
                s.B, s.nx, s.ny, s.nz, ctx.dt,
                ctx.phys_x0(), ctx.phys_x1(), ctx.phys_y0(), ctx.phys_y1(),
                ctx.phys_z0(), ctx.phys_z1()
            );
            if (rtm_out->adcig.defined() && rtm_out->adcig.numel() > 0) {
                int nlag = rtm_out->adcig.size(0);
                int Bloc = rtm_out->adcig.size(1) * rtm_out->adcig.size(2);
                int max_lag = (nlag - 1) / 2;
                accumulate_adcig_3d<<<s.launch_config.grid, s.launch_config.block>>>(
                    for_view.u_next,
                    adjoint.u_now_t.data_ptr<float>(),
                    rtm_out->adcig.data_ptr<float>(),
                    nlag, max_lag, Bloc, s.nx, s.ny, s.nz
                );
            }
        }
        add_source_3d<<<s.source_config.grid, s.source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            (int)p.forward_sources_loc.size(1),
            ctx
        );
        forward.swap();
    }

    static void bs_rtm_tap(const State& /*s*/, const SolverContext& /*ctx*/,
                              Wavefield& /*forward*/, Wavefield& /*adjoint*/,
                              RTMOutput& /*illumination*/, bool /*compute_illumination*/)
    {
        // no-op: 3-D images inside bs_recon_step, before the injection.
    }

    // ===================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING — generic_backward_ckpt, per chunk:
    //   replay: replay_step -> inject_source_fwd -> rotate_recon_buffers;
    //   reverse: the five full-mode hooks of section [3].
    // generic_backward_recursive_ckpt bisects each ckpt segment; a leaf runs
    // one replay triple, then the reverse-five with imaging fed from the
    // leaf's scratch u.
    // ===================================================================== //

    // Replay state set ``set`` of p.forward_wavefields, which holds K sets of
    // CKPT_STATE_COUNT tensors back to back (ckpt: K = 1; recursive: 1 + the
    // bisection depth, cuda_layout.recursive_state_depth).  The 9-tensor bind
    // is the in-place psi layout the checkpoint snapshots carry (use_pml, no
    // psi double-buffer: the replay pairs with the u-only swap()).
    static void bind_replay_state(Wavefield& wf, const BackwardInput& p,
                                  const torch::Tensor& vp, int set)
    {
        wf.bind_replay_state(wavefield_set(p.forward_wavefields, set, CKPT_STATE_COUNT,
                                           "acoustic3d ckpt replay state"),
                             vp, p.checkpoints, 3);
    }

    // Set 0: the chunk replay state (ckpt) / the segment start state
    // (recursive).  The propagator zeroes the set per backward call, which is
    // the state the replay and the bisection both start from.
    // MANDATORY: both checkpoint modes are handed their replay state sets per
    // call (_c.py Wrapper.backward ckpt branch: ``params.forward_wavefields =
    // _forward_state_buffers(cp.forward_state_shapes, ...)``, with
    // ``_forward_state_shapes(..., "ckpt"/"recursive")`` = the forward slot list
    // minus the psi shadows = CKPT_STATE_COUNT slots, aux slabs included).  The
    // count is checked by the skeleton before this hook fires and again by
    // bind_replay_state, which also checks the aux geometry against the
    // checkpoint slots -- the reason the deleted fallback had to derive its
    // shapes from the snapshots rather than from vp.
    static void bind_or_alloc_recon_ckpt(Wavefield& wf, const BackwardInput& p,
                                         const torch::Tensor& vp)
    {
        TORCH_CHECK(!p.forward_wavefields.empty(),
                    "acoustic3d/ckpt backward requires the propagator-bound replay "
                    "state forward_wavefields (cuda_layout.checkpoint_state_nvar / "
                    "the forward slot table)");
        bind_replay_state(wf, p, vp, 0);
    }

    // Sets 1..depth: the bisection's scratch states, each copy_state-filled
    // from its parent interval before any read.
    // MANDATORY for the same reason as set 0: cuda_layout.recursive_state_depth
    // makes the propagator hand 1 + depth(longest segment) sets, and the skeleton
    // checks that total against its own bisection depth before binding.
    static void bind_or_alloc_recursive_scratch(Wavefield& wf, const BackwardInput& p,
                                                const torch::Tensor& vp, int set,
                                                const Wavefield& /*start_state*/)
    {
        TORCH_CHECK(!p.forward_wavefields.empty(),
                    "acoustic3d/recursive backward requires the propagator-bound "
                    "replay state sets forward_wavefields "
                    "(cuda_layout.recursive_state_depth)");
        bind_replay_state(wf, p, vp, set);
    }

    static void replay_step(const State& s, const SolverContext& ctx,
                            AcousticWavefieldPointer view,
                            AcousticCPMLPointer cpml,
                            bool save_all, float* u_this)
    {
        ACOUSTIC3D(
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
            s.grad_ctx_y,
            s.grad_ctx_z,
            cpml,
            ctx
        );
    }

    // Replay twin of the forward-loop inject_source_fwd in section [2].
    static void inject_source_fwd(const State& s, const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  const BackwardInput& p, int it, int nsrc)
    {
        add_source_3d<<<s.source_config.grid, s.source_config.block>>>(
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

} // namespace acoustic3d
