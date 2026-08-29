// Driver traits for the 3-D acoustic equation (see acoustic2d/driver_traits.cuh
// for the pattern).  Line-faithful transcription of the hand-written drivers;
// physics kernels untouched.  Deltas vs 2-D worth naming:
//   * no ctx.set_per_edge anywhere — per-edge free surface is 2-D only;
//   * the fused adjoint carries a psi AND zeta triple double-buffer
//     (15 adjoint tensors, adjoint_extra_nvar=3);
//   * the boundary-saving reverse step images BEFORE the forward source
//     injection (2-D images after injection + swap), and its NOPML kernel
//     writes a per-step scratch field (BsScratch.f_this);
//   * ADCIG is served only by backward_bs (full/ckpt imaging correlates
//     vp^2*Lap(u), not raw pressure), and there is no seed rim-zeroing;
//   * the old hand-written backward_bs built its SolverContext with nullptr
//     lap/grad coefficient pointers; the shared skeleton passes the real
//     pointers everywhere.  Nothing on the bs path dereferences them (it
//     could not have run before otherwise), so this is bit-inert.
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
    static constexpr bool ADCIG_IN_FULL_MODES = false;

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

    template <class P>
    static void setup_ctx(SolverContext& ctx, const P& p)
    {
        // No set_per_edge: per-edge free surface is not wired for 3-D.
        if (p.has_topo) {
            ctx.topo_rows = p.topo_rows.template data_ptr<int>();
            ctx.has_topo  = true;
        }
    }

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.wavefields.empty())
            wf.bind(p.wavefields, 3, true);
        else
            wf.allocate(vp, 3, true, /*double_buffer_psi=*/true);
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

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, d.B, d.nz, d.ny, d.nx};
    }

    static int save_width(int abcn, int M) { return abcn > 0 ? M + 1 : M; }

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

    static void end_of_step(Wavefield& wf) { wf.swap_pml(); }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        saver.last_two_t.select(1, 0).copy_(wf.u_prev_t);
        saver.last_two_t.select(1, 1).copy_(wf.u_now_t);
    }

    // ---- backward hooks -------------------------------------------------- //

    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.adjoint_wavefields.empty())
            wf.bind(p.adjoint_wavefields, 3, true);
        else
            wf.allocate(vp, 3, true);
    }

    static void bind_or_alloc_recon(Wavefield& wf, const BackwardInput& p,
                                    const torch::Tensor& vp)
    {
        if (!p.forward_wavefields.empty())
            wf.bind(p.forward_wavefields, 3, false);
        else
            wf.allocate(vp, 3, false);
    }

    static void bind_or_alloc_recon_ckpt(Wavefield& wf, const BackwardInput& p,
                                         const torch::Tensor& vp)
    {
        if (!p.forward_wavefields.empty())
            wf.bind(p.forward_wavefields, 3, true);
        else
            wf.allocate_from_snapshots(vp, p.checkpoints, 3);
    }

    static void alloc_recursive_start_state(Wavefield& wf, const BackwardInput& p,
                                            const torch::Tensor& vp)
    {
        wf.allocate_from_snapshots(vp, p.checkpoints, 3);
    }

    // FUSED single-kernel exact 3D adjoint (see acoustic2d twin).
    static void adjoint_step(const State& s, const SolverContext& ctx,
                             AcousticWavefieldPointer adj_view,
                             AcousticCPMLPointer cpml,
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
                                      const BackwardInput& p, int it, int nsrc)
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

    static void post_adjoint(Wavefield& wf) { wf.swap_aux(); }

    static void swap_recon(Wavefield& wf) { wf.swap(); }

    static void accumulate_source_grad(const State& s, const SolverContext& ctx,
                                       Wavefield& adjoint, const BackwardInput& p,
                                       torch::Tensor* grad_wavelet,
                                       int it, int nsrc)
    {
        if (grad_wavelet == nullptr)
            return;
        accumulate_source_grad_3d<<<s.source_config.grid, s.source_config.block>>>(
            adjoint.u_now_t.data_ptr<float>(),
            grad_wavelet->data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            ctx
        );
    }

    static void image_step(const State& s, const SolverContext& ctx,
                           const float* forward_ptr, const float* adjoint_ptr,
                           const torch::Tensor& vp,
                           torch::Tensor* grad, RTMOutput* rtm_out)
    {
        if (grad != nullptr) {
            calculate_grad_3d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward_ptr, adjoint_ptr,
                vp.data_ptr<float>(),
                grad->data_ptr<float>(),
                s.B, s.nx, s.ny, s.nz, ctx.dt
            );
        }
        if (rtm_out != nullptr) {
            accumulate_rtm_image_3d<<<s.launch_config.grid, s.launch_config.block>>>(
                forward_ptr, adjoint_ptr,
                rtm_out->image.data_ptr<float>(),
                rtm_out->source_illumination.data_ptr<float>(),
                rtm_out->receiver_illumination.data_ptr<float>(),
                s.B, s.nx, s.ny, s.nz
            );
        }
        // No ADCIG here: this forward store is vp^2*Lap(u), not raw pressure.
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

    // Seed from the saved last two snapshots only — 3-D has no rim-zeroing.
    static void seed_reconstruction(const State& /*s*/, const SolverContext& /*ctx*/,
                                    Wavefield& forward, const BackwardInput& p)
    {
        forward.u_prev_t.copy_(p.u_last_two.select(1, 1).squeeze(0));
        forward.u_now_t.copy_(p.u_last_two.select(1, 0).squeeze(0));
    }

    struct BsScratch {
        torch::Tensor f_this;   // per-step NOPML output field
    };
    static BsScratch make_bs_scratch(const BackwardInput& /*p*/,
                                     const torch::Tensor& vp)
    {
        return {torch::zeros_like(vp)};
    }

    // 3-D bs reverse step: NOPML(f_this) -> strip restore -> u_tt gradient +
    // rtm + ADCIG imaging (BEFORE the forward source injection — the 2-D twin
    // images after injection + swap) -> inject -> swap.
    static void bs_reverse_step(const State& s, const SolverContext& ctx,
                                Wavefield& forward, Wavefield& adjoint,
                                BoundaryRuntime& boundary_runtime,
                                const GeneralBoundaryPointer& bs, int save_width,
                                AcousticCPMLPointer /*cpml*/,
                                const BackwardInput& p,
                                torch::Tensor& grad,
                                RTMOutput* rtm_out,
                                BsScratch& scratch,
                                int it, int bs_it0)
    {
        auto for_view = forward.view();
        ACOUSTIC3D_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            scratch.f_this.data_ptr<float>(),
            s.vp,
            s.lap_ctx,
            ctx
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
        calculate_grad_utt_3d<<<s.launch_config.grid, s.launch_config.block>>>(
            forward.u_prev_t.data_ptr<float>(),
            for_view.u_next,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            s.vp,
            grad.data_ptr<float>(),
            s.B, s.nx, s.ny, s.nz, ctx.dt
        );
        if (rtm_out != nullptr) {
            accumulate_rtm_image_3d<<<s.launch_config.grid, s.launch_config.block>>>(
                for_view.u_next,
                adjoint.u_now_t.data_ptr<float>(),
                rtm_out->image.data_ptr<float>(),
                rtm_out->source_illumination.data_ptr<float>(),
                rtm_out->receiver_illumination.data_ptr<float>(),
                s.B, s.nx, s.ny, s.nz
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

    static void bs_image_step(const State& /*s*/, const SolverContext& /*ctx*/,
                              Wavefield& /*forward*/, Wavefield& /*adjoint*/,
                              RTMOutput& /*illumination*/, bool /*compute_illumination*/)
    {
        // no-op: 3-D images inside bs_reverse_step, before the injection.
    }
};

} // namespace acoustic3d
