// Driver traits for the 2-D velocity-reflectivity (EVR) elastic equation: the
// per-equation half of the staggered-family skeleton in
// ``common/sg_driver.cuh``.  Line-faithful transcription of the hand-written
// drivers; physics kernels untouched.  Deltas from the elastic2d reference:
//   * six raw models {vp, vs, Rp_x, Rp_z, Rs_x, Rs_z}, six gradients, and no
//     rho — every rho hook (fix_rho_grad_at_sources / fix_rho_grad_at_receivers) is a no-op
//     and the imaging has no velocity(t+1) term (IMAGING_USES_NEXT_V = false: the
//     recursive replay breaks at the target step and no cross-segment
//     velocity carriers exist);
//   * the wavefield reuses ElasticWavefieldTensor (vx/vz slots hold the
//     momentum px/pz), so the 15-tensor bind/checkpoint layout and the
//     5-field boundary-saving list match elastic2d;
//   * every backward mode zeroes the adjoint stress surface row right after
//     the residual injection (adjoint of the forward free-surface BC), so
//     the kernel lives at the tail of inject_residuals;
//   * the gradient kernel is followed by a chain-rule pass
//     (LAUNCH_EVR_GRAD_CHAIN_APPLY) in every mode — both live in
//     image_standalone;
//   * the 14-slot adjoint workspace pool splits into the adjoint-step half
//     (slots 0-9, the Workspace) and the imaging half (slots 10-13 plus the
//     zero next-momentum buffer, carried in the State so image_standalone
//     can reach them);
//   * bs: bind_or_alloc_recon binds the 5-tensor p.forward_wavefields list
//     [px, pz, sxx, szz, sxz] with use_pml = false (no carriers: the imaging
//     has no velocity(t+1) term) or allocate(vp, 2, false) when the list is
//     empty; the ckpt/recursive states use the plain full-shape allocate
//     (no snapshot-driven aux layout).
//
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/sg_driver.cuh.
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../common/eq_driver.cuh"
#include "../../common/sg_driver.cuh"
#include "../../launch/config.h"

namespace elastic_vr2d {

using namespace elastic_vr2d_kernels;

struct Driver {
    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // Prologue of every entry (timing map, in call order): validate_backward
    // (backward only), parse_models, setup_ctx, bind_or_alloc_* wavefields,
    // init_aux_slabs, alloc_cpml, bind_grads / alloc_grads, make_workspace,
    // make_state, signed_adjoint_sources.
    // ===================================================================== //

    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "elastic_vr2d";
    static constexpr int CKPT_NVAR = 15;
    static constexpr const char* CKPT_COUNT_MSG =
        "elastic_vr2d checkpointing expects 15 checkpoint tensors";
    static constexpr const char* CKPT_RECURSIVE_COUNT_MSG =
        "elastic_vr2d checkpointing expects 15 checkpoint tensors";
    static constexpr int BS_NVAR = 5;   // px, pz, sxx, szz, sxz
    static constexpr int CUT_MASK_BITS = 0xF;
    static constexpr const char* CUT_MASK_DESC = "bits 0..3 (x_lo, x_hi, z_lo, z_hi)";
    static constexpr int ADJ_WF_COUNT = 15;
    static constexpr int RECON_WF_COUNT = 5;   // == cuda_layout.bs_reconstruction_nvar
    static constexpr const char* RECON_LIST_DESC =
        "[px, pz, sxx, szz, sxz]";
    static_assert(RECON_WF_COUNT == BS_NVAR,
                  "bs reconstruction list = the physical fields only (no carriers)");
    static constexpr int N_VEL = 2;
    static constexpr bool IMAGING_USES_NEXT_V = false;   // no velocity(t+1) imaging term

    using Wavefield = ElasticWavefieldTensor;
    using WfView = ElasticWavefieldPointer;
    using CPML = ElasticCPMLTensor;

    struct Models {
        torch::Tensor vp, vs, Rp_x, Rp_z, Rs_x, Rs_z;
    };

    template <class P>
    static Models models_from(const P& p)
    {
        Models m;
        m.vp   = p.models[0];
        m.vs   = p.models[1];
        m.Rp_x = p.models[2];
        m.Rp_z = p.models[3];
        m.Rs_x = p.models[4];
        m.Rs_z = p.models[5];
        return m;
    }

    static Models parse_models(const ForwardInput& p)
    {
        TORCH_CHECK(p.models.size() >= 6,
                    "elastic_vr2d::forward expects 6 model tensors "
                    "(vp, vs, Rp_x, Rp_z, Rs_x, Rs_z)");
        return models_from(p);
    }

    static Models parse_models(const BackwardInput& p)
    {
        // full/bs counts are checked by validate_backward (which the drivers
        // run before this); the hand-written ckpt/recursive entries accessed
        // the models unchecked.
        return models_from(p);
    }

    struct State {
        Models models;
        SGradParam grad_ctx;
        fdtd::LaunchConfig launch_config;
        fdtd::LaunchConfig source_config;
        fdtd::LaunchConfig record_config;
        int order;
        // Imaging half of the 14-slot adjoint workspace pool (chain-rule
        // sources L_dV*), plus the zero next-momentum buffer for the
        // bs/ckpt/recursive imaging calls (full mode gets its zero fallback
        // from the driver's zero_velocity via vel_ptrs_from_u_forward).
        torch::Tensor ws_lvpx, ws_lvpz, ws_lvsx, ws_lvsz;
        torch::Tensor zero_buf;
    };

    template <class P>
    static State make_state_common(const P& p, const eqdrv::Dims& d,
                                   const Models& models,
                                   fdtd::LaunchConfig launch_config,
                                   fdtd::LaunchConfig source_config,
                                   fdtd::LaunchConfig record_config)
    {
        float dx = p.spacing[0];
        float dz = p.spacing[1];
        State s;
        s.models = models;
        s.grad_ctx = SGradParam{1, 0, d.nx, p.M, p.grad_coes.template data_ptr<float>(), dx, 0.f, dz};
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        return s;
    }

    static State make_state(const ForwardInput& p, const eqdrv::Dims& d,
                            const Models& models,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        return make_state_common(p, d, models, launch_config, source_config,
                                 record_config);
    }

    static State make_state(const BackwardInput& p, const eqdrv::Dims& d,
                            const Models& models,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        State s = make_state_common(p, d, models, launch_config, source_config,
                                    record_config);
        const auto& vp = p.models[0];
        s.ws_lvpx = pool_or_zeros(p.adjoint_workspace, 10, vp);
        s.ws_lvpz = pool_or_zeros(p.adjoint_workspace, 11, vp);
        s.ws_lvsx = pool_or_zeros(p.adjoint_workspace, 12, vp);
        s.ws_lvsz = pool_or_zeros(p.adjoint_workspace, 13, vp);
        // u_forward is bound in full mode only, where the imaging always has
        // real next-momentum pointers — allocate the zero buffer for the
        // other three modes, keeping the per-mode allocation count identical
        // to the hand-written drivers.
        if (!p.u_forward.defined())
            s.zero_buf = torch::zeros_like(vp);
        return s;
    }

    // Adjoint-step half of the 14-slot workspace pool (slots 0-9).
    struct Workspace {
        torch::Tensor qxx, qzz, qxz, qzx;
        torch::Tensor pxx, pzz, pxz, pzx;
        torch::Tensor pt_px, pt_pz;
    };

    static Workspace make_workspace(const BackwardInput& p, const torch::Tensor& vp)
    {
        Workspace w;
        w.qxx   = pool_or_zeros(p.adjoint_workspace, 0, vp);
        w.qzz   = pool_or_zeros(p.adjoint_workspace, 1, vp);
        w.qxz   = pool_or_zeros(p.adjoint_workspace, 2, vp);
        w.qzx   = pool_or_zeros(p.adjoint_workspace, 3, vp);
        w.pxx   = pool_or_zeros(p.adjoint_workspace, 4, vp);
        w.pzz   = pool_or_zeros(p.adjoint_workspace, 5, vp);
        w.pxz   = pool_or_zeros(p.adjoint_workspace, 6, vp);
        w.pzx   = pool_or_zeros(p.adjoint_workspace, 7, vp);
        w.pt_px = pool_or_zeros(p.adjoint_workspace, 8, vp);
        w.pt_pz = pool_or_zeros(p.adjoint_workspace, 9, vp);
        return w;
    }

    static void validate_forward(const ForwardInput&) {}

    static void validate_backward(const BackwardInput& p, const char* mode)
    {
        if (std::strcmp(mode, "full") == 0) {
            TORCH_CHECK(p.models.size() >= 6,
                        "elastic_vr2d::backward expects 6 model tensors "
                        "(vp, vs, Rp_x, Rp_z, Rs_x, Rs_z)");
        } else if (std::strcmp(mode, "bs") == 0) {
            TORCH_CHECK(p.models.size() >= 6,
                        "elastic_vr2d::backward_bs expects 6 model tensors");
        } else if (std::strcmp(mode, "ckpt_recursive") == 0) {
            TORCH_CHECK(p.checkpoint_steps.defined() && p.checkpoint_steps.dim() == 1,
                        "recursive checkpointing expects 1-D checkpoint_steps");
        }
        // "ckpt": the interval/count checks run in the shared driver, in the
        // hand-written order.
    }

    template <class P>
    static void setup_ctx(SolverContext& solver, const P& p)
    {
        if (p.has_topo) {
            solver.topo_rows = p.topo_rows.template data_ptr<int>();
            solver.has_topo = true;
        }
    }

    // The hand-written drivers never ranged the aux slabs (no DD path).
    static void init_aux_slabs(SolverContext&, Wavefield&) {}

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.allocate(p.pml_vals, 2);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, 2, d.B, d.nz, d.nx};   // only px and pz
    }

    static float* field_ptr(WfView& wf, int field_idx)
    {
        return elastic_field_ptr(wf, 2, field_idx);
    }

    static WfView view(Wavefield& wf) { return wf.view(); }

    // ===================================================================== //
    // [2] FORWARD — sg_generic_forward, per it in [it_begin, it_end):
    //   velocity_substep, stress_substep, inject_source, <checkpoint save>,
    //   save_boundary_fields, record_field; after the loop: save_last_state.
    // ===================================================================== //

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& vp)
    {
        // Reuse ElasticWavefieldTensor (same 15-field layout):
        //   vx/vz -> px/pz, sxx/szz/sxz unchanged, 10 CPML memvars at the
        //   same slots as elastic2d.
        if (!p.wavefields.empty())
            wf.bind(p.wavefields, true);
        else
            wf.allocate(vp, 2);
    }

    // No-op: this equation writes u_allt from inside its stress kernel.
    // See the call site in sg_driver.cuh for why the hook exists.
    static void capture_allt(torch::Tensor&, WfView&, const SolverContext&, int) {}

    static void velocity_substep(const State& s, WfView& wf,
                                 ElasticCPMLPointer cpml_view,
                                 const SolverContext& solver)
    {
        LAUNCH_EVR_MOMENTUM(
            s.order,
            s.launch_config.grid, s.launch_config.block,
            wf, s.grad_ctx, cpml_view, solver
        );
    }

    static void stress_substep(const State& s, WfView& wf,
                               ElasticCPMLPointer cpml_view,
                               const SolverContext& solver, float* u_this_t)
    {
        LAUNCH_EVR_STRESS(
            s.order,
            s.launch_config.grid, s.launch_config.block,
            wf,
            s.models.vp.data_ptr<float>(),
            s.models.vs.data_ptr<float>(),
            s.models.Rp_x.data_ptr<float>(),
            s.models.Rp_z.data_ptr<float>(),
            s.models.Rs_x.data_ptr<float>(),
            s.models.Rs_z.data_ptr<float>(),
            u_this_t,
            s.grad_ctx, cpml_view, solver
        );
    }

    static void inject_source(const State& s, const SolverContext& solver,
                              float* field, const torch::Tensor& source,
                              const torch::Tensor& sources_loc, int it, int nsrc)
    {
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            field, source.data_ptr<float>(), sources_loc.data_ptr<int>(),
            it, nsrc, solver
        );
    }

    static void save_boundary_fields(BoundaryRuntime& rt, const State& s,
                                     const SolverContext& solver, WfView& wf,
                                     int it, int nt,
                                     const GeneralBoundaryPointer& bs,
                                     int save_width)
    {
        float* fields[5] = {wf.vx, wf.vz, wf.sxx, wf.szz, wf.sxz};
        for (int f = 0; f < 5; ++f) {
            rt.save_forward_2d_field(
                it, nt, fields[f],
                s.launch_config.grid, s.launch_config.block,
                bs, save_width, -solver.M, solver, f, f == 4
            );
        }
    }

    static void record_field(const State& s, const SolverContext& solver,
                             float* field, torch::Tensor& record, int irec,
                             const torch::Tensor& receivers_loc, int it, int nrec)
    {
        record_kernel<<<s.record_config.grid, s.record_config.block>>>(
            field, record[irec].data_ptr<float>(),
            receivers_loc.data_ptr<int>(), it, nrec, solver
        );
    }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        saver.last_two_t.select(0, 0).select(0, 0).copy_(wf.vx_t);
        saver.last_two_t.select(0, 1).select(0, 0).copy_(wf.vz_t);
        saver.last_two_t.select(0, 2).select(0, 0).copy_(wf.sxx_t);
        saver.last_two_t.select(0, 3).select(0, 0).copy_(wf.szz_t);
        saver.last_two_t.select(0, 4).select(0, 0).copy_(wf.sxz_t);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — sg_generic_backward, per reverse it:
    //   fix_rho_grad_at_sources, inject_residuals, vel_ptrs_from_u_forward;
    //   it == 0: image_standalone + fix_rho_grad_at_receivers, loop ends;
    //   it  > 0: full_mode_step (imaging + adjoint step, fused order).
    // ===================================================================== //

private:
    // Shared adjoint-step halves (stress: prepare + apply; momentum: prepare
    // + apply).  Fired by plain_adjoint_step (full / ckpt / recursive modes)
    // and by the backward_bs hooks bs_stress_half (stress) / bs_velocity_half (momentum).
    static void stress_adjoint_half(const State& s, const SolverContext& solver,
                                    WfView& adj_view, Workspace& w,
                                    ElasticCPMLPointer cpml_view)
    {
        LAUNCH_EVR_STRESS_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.models.vp.data_ptr<float>(),
            s.models.vs.data_ptr<float>(),
            s.models.Rp_x.data_ptr<float>(),
            s.models.Rp_z.data_ptr<float>(),
            s.models.Rs_x.data_ptr<float>(),
            s.models.Rs_z.data_ptr<float>(),
            s.grad_ctx,
            cpml_view,
            solver,
            w.qxx.data_ptr<float>(),
            w.qzz.data_ptr<float>(),
            w.qxz.data_ptr<float>(),
            w.qzx.data_ptr<float>(),
            w.pt_px.data_ptr<float>(),
            w.pt_pz.data_ptr<float>()
        );
        LAUNCH_EVR_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            w.qxx.data_ptr<float>(),
            w.qzz.data_ptr<float>(),
            w.qxz.data_ptr<float>(),
            w.qzx.data_ptr<float>(),
            w.pt_px.data_ptr<float>(),
            w.pt_pz.data_ptr<float>(),
            s.grad_ctx, solver
        );
    }

    static void momentum_adjoint_half(const State& s, const SolverContext& solver,
                                      WfView& adj_view, Workspace& w,
                                      ElasticCPMLPointer cpml_view)
    {
        LAUNCH_EVR_MOMENTUM_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view, cpml_view, solver,
            w.pxx.data_ptr<float>(),
            w.pzz.data_ptr<float>(),
            w.pxz.data_ptr<float>(),
            w.pzx.data_ptr<float>()
        );
        LAUNCH_EVR_MOMENTUM_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            w.pxx.data_ptr<float>(),
            w.pzz.data_ptr<float>(),
            w.pxz.data_ptr<float>(),
            w.pzx.data_ptr<float>(),
            s.grad_ctx, solver
        );
    }

public:
    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.adjoint_wavefields.empty())
            wf.bind(p.adjoint_wavefields, true);
        else
            wf.allocate(vp, 2);
    }

    static void zero_adjoint_if_first_segment(Wavefield&, bool) {}

    // The hand-written drivers never bound grads_out (elastic_vr2d has no
    // stepped/DD path); always allocate the six accumulators.
    // grads_out from the propagator (one per model: vp, vs, Rp_x, Rp_z,
    // Rs_x, Rs_z; zeroed per backward on the Python side, accumulated here)
    // or fresh zeros when unbound.
    static void bind_grads(const BackwardInput& p, std::vector<torch::Tensor>& grads)
    {
        if (p.grads_out.empty()) {
            alloc_grads(p.models[0], grads);
            return;
        }
        TORCH_CHECK(p.grads_out.size() == 6,
                    "elastic_vr2d backward: grads_out must hold 6 tensors "
                    "(vp, vs, Rp_x, Rp_z, Rs_x, Rs_z), got ", p.grads_out.size());
        grads = p.grads_out;
    }

    static void alloc_grads(const torch::Tensor& vp, std::vector<torch::Tensor>& grads)
    {
        grads = {torch::zeros_like(vp), torch::zeros_like(vp),
                 torch::zeros_like(vp), torch::zeros_like(vp),
                 torch::zeros_like(vp), torch::zeros_like(vp)};
    }

    // The EVR adjoint injects the raw residuals for every receiver field —
    // the hand-written drivers apply no stress-receiver sign flip.
    static std::vector<torch::Tensor> signed_adjoint_sources(
        const BackwardInput& p, const torch::Tensor& receiver_fields)
    {
        std::vector<torch::Tensor> out;
        const int64_t nfield = receiver_fields.numel();
        out.reserve(static_cast<size_t>(nfield));
        for (int64_t i = 0; i < nfield; ++i)
            out.push_back(p.adjoint_source[i]);
        return out;
    }

    struct VelPtrs {
        const float* px_now;
        const float* pz_now;
        const float* px_next;
        const float* pz_next;
    };

    static VelPtrs vel_ptrs_from_u_forward(const BackwardInput& p, int it,
                                             const torch::Tensor& zero_velocity)
    {
        // Saved forward momentum at time t (px in channel 0, pz in channel 1).
        VelPtrs v;
        v.px_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        v.pz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        v.px_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>()
                                    : zero_velocity.data_ptr<float>();
        v.pz_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>()
                                    : zero_velocity.data_ptr<float>();
        return v;
    }

    // No rho model: no body-force or receiver rho corrections.
    static void fix_rho_grad_at_sources(const State&, const SolverContext&, WfView&,
                                const BackwardInput&, const torch::Tensor&, int,
                                std::vector<torch::Tensor>&) {}

    static void inject_residuals(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const BackwardInput& p,
                                 const torch::Tensor& receiver_fields,
                                 const std::vector<torch::Tensor>& adj_source_signed,
                                 int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.numel(); ++irec) {
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<s.record_config.grid, s.record_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it, adjoint_nsrc, solver
            );
        }
        // Adjoint of the forward free-surface BC (szz = sxz = 0 at the
        // surface): zero the adjoint stresses at the surface row before any
        // grad/adjoint kernel reads them.  No-op when free_surface = false.
        // Every hand-written mode ran this immediately after the residual
        // injection, so it lives here.
        elastic_vr2d_kernels::evr_adjoint_zero_top_fs<0>
            <<<s.launch_config.grid, s.launch_config.block>>>(adj_view, solver);
    }

    // Gradient kernel (pointwise terms + chain-rule sources), then the
    // chain-rule pass (transpose central FD on L_dV* into grad_vp/grad_vs).
    // Null next pointers (bs/ckpt/recursive) read the zero buffer, exactly
    // the zero_buf argument of the hand-written calls.
    // (also fired by backward_bs phase 1 and the ckpt/recursive imaging)
    static void image_standalone(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const VelPtrs& v,
                                 std::vector<torch::Tensor>& grads)
    {
        const float* px_next = v.px_next ? v.px_next : s.zero_buf.data_ptr<float>();
        const float* pz_next = v.pz_next ? v.pz_next : s.zero_buf.data_ptr<float>();
        LAUNCH_CALCULATE_GRAD_EVR_NOBS(
            s.order,
            s.launch_config.grid, s.launch_config.block,
            adj_view,
            v.px_now, v.pz_now,
            px_next, pz_next,
            s.models.vp.data_ptr<float>(),
            s.models.vs.data_ptr<float>(),
            s.models.Rp_x.data_ptr<float>(),
            s.models.Rp_z.data_ptr<float>(),
            s.models.Rs_x.data_ptr<float>(),
            s.models.Rs_z.data_ptr<float>(),
            grads[0].data_ptr<float>(),
            grads[1].data_ptr<float>(),
            grads[2].data_ptr<float>(),
            grads[3].data_ptr<float>(),
            grads[4].data_ptr<float>(),
            grads[5].data_ptr<float>(),
            s.ws_lvpx.data_ptr<float>(),
            s.ws_lvpz.data_ptr<float>(),
            s.ws_lvsx.data_ptr<float>(),
            s.ws_lvsz.data_ptr<float>(),
            s.grad_ctx, solver
        );
        LAUNCH_EVR_GRAD_CHAIN_APPLY(
            s.order,
            s.launch_config.grid, s.launch_config.block,
            s.ws_lvpx.data_ptr<float>(),
            s.ws_lvpz.data_ptr<float>(),
            s.ws_lvsx.data_ptr<float>(),
            s.ws_lvsz.data_ptr<float>(),
            grads[0].data_ptr<float>(),
            grads[1].data_ptr<float>(),
            s.grad_ctx, solver
        );
    }

    // (no rho model — no-op, see fix_rho_grad_at_sources above)
    static void fix_rho_grad_at_receivers(const State&, const SolverContext&,
                                  std::vector<torch::Tensor>&, const VelPtrs&,
                                  const BackwardInput&, const torch::Tensor&,
                                  int, int) {}

    // EVR has no fused imaging kernel: image standalone, then run the plain
    // 4-kernel adjoint step (stress-adjoint then momentum-adjoint), exactly
    // the hand-written order.  No receiver-rho correction (no rho model).
    static void full_mode_step(const State& s, const SolverContext& solver,
                                Wavefield& adjoint, Workspace& workspace,
                                ElasticCPMLPointer cpml_view,
                                const VelPtrs& v,
                                std::vector<torch::Tensor>& grads,
                                const BackwardInput&,
                                const torch::Tensor&,
                                int, int)
    {
        auto adj_view = adjoint.view();
        image_standalone(s, solver, adj_view, v, grads);
        plain_adjoint_step(s, solver, adjoint, workspace, cpml_view);
    }

    // ---- Adjoint of forward stress step (2 kernels: prepare + apply),
    //      then adjoint of forward momentum step (2 kernels). ----
    // (also fired by the ckpt/recursive reverse loops)
    static void plain_adjoint_step(const State& s, const SolverContext& solver,
                                   Wavefield& adjoint, Workspace& w,
                                   ElasticCPMLPointer cpml_view)
    {
        auto adj_view = adjoint.view();
        stress_adjoint_half(s, solver, adj_view, w, cpml_view);
        momentum_adjoint_half(s, solver, adj_view, w, cpml_view);
    }

    // ===================================================================== //
    // [4] BACKWARD_BS — sg_generic_backward_bs, per reverse it, floor
    //   max(it_lo, 1): fix_rho_grad_at_sources / inject_residuals /
    //   uninject_forward_source [inject_step], then bs_stress_half (stress recon
    //   + strip restore + imaging + stress-adjoint half), then bs_velocity_half
    //   (velocity-adjoint half + velocity recon + strip restore + prefetch).
    //   Before the loop (first segment): seed_recon from u_last_two.
    // ===================================================================== //

    static void zero_adjoint_if_first_segment_bs(Wavefield&, bool) {}

    // No cross-step carriers: the imaging has no velocity(t+1) term.
    struct ReconCarriers {};

    static ReconCarriers bind_or_alloc_recon(Wavefield& forward,
                                             const BackwardInput& p,
                                             const torch::Tensor& vp)
    {
        // Reconstruction state = the 5 physical fields [px (vx slot), pz (vz
        // slot), sxx, szz, sxz], RECON_LIST_DESC order, bound WITHOUT CPML
        // memvars (the reverse reconstruction uses the NOPML kernels only;
        // view() hands them nullptr for m_*).  Python owns all RECON_WF_COUNT
        // of them when it binds p.forward_wavefields (bs_reconstruction_nvar
        // zeroed grids shaped like vp); the empty-list fallback allocates
        // exactly as before.
        if (wavefields_bound(p.forward_wavefields, RECON_WF_COUNT, vp,
                             "elastic_vr2d backward_bs forward_wavefields"))
            forward.bind(p.forward_wavefields, /*use_pml=*/false);
        else
            forward.allocate(vp, 2, false);
        return {};
    }

    static void seed_recon(Wavefield& forward, const BackwardInput& p)
    {
        forward.vx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));   // px
        forward.vz_t.copy_(p.u_last_two.select(0, 1).select(0, 0));   // pz
        forward.sxx_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
        forward.szz_t.copy_(p.u_last_two.select(0, 3).select(0, 0));
        forward.sxz_t.copy_(p.u_last_two.select(0, 4).select(0, 0));
    }

    static void uninject_forward_source(const State& s, const SolverContext& solver,
                                        WfView& for_view, const BackwardInput& p,
                                        const torch::Tensor& source_fields,
                                        const torch::Tensor& neg_forward_source,
                                        int it, int forward_nsrc)
    {
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<s.source_config.grid, s.source_config.block>>>(
                field,
                neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it, forward_nsrc, solver
            );
        }
    }

    // bs phase 1: stress reconstruction + restore + gradient imaging + the
    // stress-adjoint half.
    static void bs_stress_half(const State& s, const SolverContext& solver,
                          WfView& for_view, WfView& adj_view,
                          Wavefield& adjoint, Workspace& w,
                          ElasticCPMLPointer cpml_view,
                          BoundaryRuntime& boundary_runtime,
                          const GeneralBoundaryPointer& bs, int save_width,
                          std::vector<torch::Tensor>& grads,
                          ReconCarriers&, const BackwardInput& p,
                          const torch::Tensor& receiver_fields,
                          int it, int adjoint_nsrc)
    {
        // Reverse the stress update (sigma^{it} -> sigma^{it-1}) in the
        // interior; for_view momentum still holds p^{it}.
        LAUNCH_EVR_STRESS_NOPML(
            s.order, s.launch_config.grid, s.launch_config.block,
            for_view,
            s.models.vp.data_ptr<float>(),
            s.models.vs.data_ptr<float>(),
            s.models.Rp_x.data_ptr<float>(),
            s.models.Rp_z.data_ptr<float>(),
            s.models.Rs_x.data_ptr<float>(),
            s.models.Rs_z.data_ptr<float>(),
            (float*)nullptr, s.grad_ctx, solver
        );
        // restore sigma boundary strips (fields 2,3,4 = sxx,szz,sxz)
        float* sfields[3] = {for_view.sxx, for_view.szz, for_view.sxz};
        for (int f = 2; f < 5; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, sfields[f - 2], s.launch_config.grid, s.launch_config.block,
                bs, save_width, -solver.M, solver, f, f == 2, false
            );
        }

        // Gradient (uses reconstructed p^{it} = for_view.vx/vz + adjoint);
        // null next pointers read the zero buffer.
        VelPtrs v{for_view.vx, for_view.vz, nullptr, nullptr};
        image_standalone(s, solver, adj_view, v, grads);

        stress_adjoint_half(s, solver, adj_view, w, cpml_view);
    }

    // bs phase 2: momentum-adjoint half, momentum reconstruction + restore,
    // prefetch.  (No carrier capture: no velocity(t+1) imaging term.)
    static void bs_velocity_half(const State& s, const SolverContext& solver,
                          WfView& for_view, Wavefield& adjoint,
                          Workspace& w, ElasticCPMLPointer cpml_view,
                          BoundaryRuntime& boundary_runtime,
                          const GeneralBoundaryPointer& bs, int save_width,
                          ReconCarriers&, Wavefield&, int it, int nt)
    {
        auto adj_view = adjoint.view();
        momentum_adjoint_half(s, solver, adj_view, w, cpml_view);

        // Reverse the momentum update (p^{it} -> p^{it-1}) in the interior.
        LAUNCH_EVR_MOMENTUM_NOPML(
            s.order, s.launch_config.grid, s.launch_config.block,
            for_view, s.grad_ctx, solver
        );
        float* mfields[2] = {for_view.vx, for_view.vz};
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it, mfields[f], s.launch_config.grid, s.launch_config.block,
                bs, save_width, -solver.M, solver, f, false, f == 1
            );
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, nt);
    }

    // ===================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING
    // sg_generic_backward_ckpt, per chunk: replay = velocity_substep /
    //   stress_substep / save_seg_velocities / inject_forward_sources; reverse =
    //   fix_rho_grad_at_sources / inject_residuals / vel_ptrs_from_seg / image_standalone /
    //   fix_rho_grad_at_receivers / (it > 0) plain_adjoint_step; after each chunk:
    //   export_seg_next_v.
    // sg_generic_backward_recursive_ckpt, per reverse it: fix_rho_grad_at_sources /
    //   inject_residuals / replay (substeps + capture_velocities) /
    //   vel_ptrs_from_carriers / image_standalone / fix_rho_grad_at_receivers /
    //   (it > 0) plain_adjoint_step.
    // ===================================================================== //

    static void bind_or_alloc_recon_ckpt(Wavefield& forward,
                                         const BackwardInput& p,
                                         const torch::Tensor& vp)
    {
        if (!p.forward_wavefields.empty())
            forward.bind(p.forward_wavefields, true);
        else
            forward.allocate(vp, 2, true);
    }

    static void alloc_recursive_start_state(Wavefield& wf, const BackwardInput&,
                                            const torch::Tensor& vp)
    {
        wf.allocate(vp, 2, true);
    }

    // Plain full-shape allocations on both sides — nothing to cross-check.
    static void check_ckpt_aux_layout(const Wavefield&, const Wavefield&) {}

    // ---- seg / carrier plumbing (ckpt + recursive modes) ----

    static std::vector<torch::Tensor> alloc_seg_buffers(const torch::Tensor& vp,
                                                        int segment_len)
    {
        auto seg_px = torch::zeros({segment_len + 1, vp.size(0) * vp.size(1), 1,
                                    vp.size(2), vp.size(3)}, vp.options());
        auto seg_pz = torch::zeros_like(seg_px);
        return {seg_px, seg_pz};
    }

    static void save_seg_velocities(std::vector<torch::Tensor>& seg, Wavefield& forward,
                            int slot)
    {
        seg[0].select(0, slot).copy_(forward.vx_t);
        seg[1].select(0, slot).copy_(forward.vz_t);
    }

    static void inject_forward_sources(const State& s, const SolverContext& solver,
                                      WfView& for_view, const BackwardInput& p,
                                      const torch::Tensor& source_fields, int it)
    {
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<s.source_config.grid, s.source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it, (int)p.forward_sources_loc.size(1), solver
            );
        }
    }

    static VelPtrs vel_ptrs_from_seg(const std::vector<torch::Tensor>& seg,
                                int now_offset, int /*next_offset*/,
                                const std::vector<torch::Tensor>& /*next_segment_v*/)
    {
        // No velocity(t+1) term: next stays null (image_standalone reads the
        // zero buffer), matching the hand-written segment's zero_buf args.
        VelPtrs v;
        v.px_now = seg[0].select(0, now_offset).data_ptr<float>();
        v.pz_now = seg[1].select(0, now_offset).data_ptr<float>();
        v.px_next = nullptr;
        v.pz_next = nullptr;
        return v;
    }

    static void export_seg_next_v(std::vector<torch::Tensor>&,
                                   const std::vector<torch::Tensor>&) {}

    static void capture_velocities(std::vector<torch::Tensor>& v, Wavefield& forward)
    {
        v[0].copy_(forward.vx_t);
        v[1].copy_(forward.vz_t);
    }

    static VelPtrs vel_ptrs_from_carriers(const std::vector<torch::Tensor>& current_v,
                                    const std::vector<torch::Tensor>& /*next_v*/)
    {
        VelPtrs v;
        v.px_now = current_v[0].data_ptr<float>();
        v.pz_now = current_v[1].data_ptr<float>();
        v.px_next = nullptr;
        v.pz_next = nullptr;
        return v;
    }
};

} // namespace elastic_vr2d
