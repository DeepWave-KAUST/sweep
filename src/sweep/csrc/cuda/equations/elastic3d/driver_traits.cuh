// Driver traits for the 3-D elastic equation: the per-equation half of the
// staggered-family skeleton in ``common/sg_driver.cuh``.  Line-faithful
// transcription of the hand-written drivers; physics kernels untouched.
// Deltas vs the 2-D reference worth naming:
//   * 9 physical fields / 36 wavefield tensors (CKPT_STATE_COUNT = 36) /
//     18-tensor adjoint workspace (WS_CARRIERS = 18; the checkpoint modes'
//     velocity carriers follow it in adjoint_workspace), three velocity
//     carriers (N_VEL = 3);
//   * DD cut faces are x/y only (mask 0x33) -- the forward validates it;
//   * the ``m_syzx`` memory field is backfilled when a bound list lacks it
//     (historical layout quirk; a no-op on the 36-slot bind the propagator
//     always supplies);
//   * the full backward zeroes the adjoint state on the FIRST segment only
//     (2-D relies on Python-zeroed buffers);
//   * the reconstruction bind requires the 12-tensor list (9 fields + 3
//     carriers).
// The APM entry points stay hand-written in forward.cu/backward.cu.
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/sg_driver.cuh.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../core/input_core.h"
#include "../../core/outputs.h"
#include "../../common/eq_driver.cuh"
#include "../../common/sg_driver.cuh"
#include "../../launch/config.h"
#include "../../operators/staggered.cuh"

namespace elastic3d {

struct Driver {
    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // Constants, type aliases, Models/State/Workspace, and the helpers the
    // shared prologue of every entry point calls (timing map prologue, in
    // call order: validate_backward (backward only), parse_models,
    // setup_ctx, bind_or_alloc_* wavefields, init_aux_slabs, alloc_cpml,
    // bind_grads, make_workspace, make_state,
    // adjoint_source_signs).
    // ===================================================================== //

    static constexpr int NDIM = 3;
    static constexpr const char* NAME = "elastic3d";
    static constexpr int CKPT_NVAR = 36;
    static constexpr const char* CKPT_COUNT_MSG =
        "Elastic 3D checkpointing expects 36 checkpoint tensors";
    static constexpr const char* CKPT_RECURSIVE_COUNT_MSG =
        "Elastic 3D recursive checkpointing expects 36 checkpoint tensors";
    static constexpr int BS_NVAR = 9;   // vx, vy, vz, sxx, syy, szz, sxy, sxz, syz
    static constexpr int CUT_MASK_BITS = 0x33;
    static constexpr const char* CUT_MASK_DESC =
        "x/y bits (bit0=x_lo, bit1=x_hi, bit4=y_lo, bit5=y_hi)";
    static constexpr int ADJ_WF_COUNT = 36;
    static constexpr int RECON_WF_COUNT = 12;
    static constexpr const char* RECON_LIST_DESC =
        "[vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, fvx_prev, fvy_prev, fvz_prev]";
    static constexpr int N_VEL = 3;
    static constexpr bool IMAGING_USES_NEXT_V = true;   // imaging consumes v(t+1) carriers
    // Checkpoint modes: the replay state is one full base+pml struct
    // (forward_wavefields set 0, cuda_layout.checkpoint_state_nvar); the
    // adjoint_workspace pool holds the 18 q*/p* scratch slots first and the
    // skeleton's velocity carriers after them (sg_driver.cuh SgCarrierSlots).
    static constexpr int CKPT_STATE_COUNT = CKPT_NVAR;   // 36
    static constexpr int WS_CARRIERS = 18;               // the nine q* then the nine p*

    using Wavefield = ElasticWavefieldTensor;
    using WfView = ElasticWavefieldPointer;
    using CPML = ElasticCPMLTensor;

    struct Models {
        Buf vp, vs, rho, mu, lambda;
    };

    template <class P>
    static Models parse_models(const P& p)
    {
        Models m;
        m.vp = p.models[0];
        m.vs = p.models[1];
        m.rho = p.models[2];
        const auto lame = derived::lame(p, m.vp, m.vs, m.rho, "elastic3d::parse_models");
        m.mu = lame.mu;
        m.lambda = lame.lambda;
        return m;
    }

    struct State {
        Models models;
        SGradParam grad_ctx;
        fdtd::LaunchConfig launch_config;
        fdtd::LaunchConfig source_config;
        fdtd::LaunchConfig record_config;
        int order;
        int nx, ny, nz, B;
    };

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d, const Models& models,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        float dx = p.spacing[0];
        float dy = p.spacing[1];
        float dz = p.spacing[2];
        State s;
        s.models = models;
        s.grad_ctx = SGradParam{1, d.nx, d.nx * d.ny, p.M,
                                p.grad_coes.template data_ptr<float>(), dx, dy, dz};
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        s.nx = d.nx;
        s.ny = d.ny;
        s.nz = d.nz;
        s.B = d.B;
        return s;
    }

    using Workspace = ElasticAdjointWorkspaceTensor;

    static Workspace make_workspace(const BackwardInputCore& p, const Buf& vp)
    {
        Workspace workspace;
        bind_adjoint_workspace_required(workspace, p.adjoint_workspace, 3,
                                        "elastic3d backward");
        return workspace;
    }

    static void validate_forward(const ForwardInputCore& p)
    {
        // DD cut faces: kernels switch the cut-side PML/interior band to the
        // interior branch; only x/y cuts are wired.
        SWEEP_CHECK((p.cut_face_mask & ~0x33) == 0,
                    "elastic3d forward cut_face_mask supports x/y bits only "
                    "(bit0=x_lo, bit1=x_hi, bit4=y_lo, bit5=y_hi), got ",
                    p.cut_face_mask);
    }

    static void validate_backward(const BackwardInputCore&, const char*) {}

    template <class P>
    static void setup_ctx(SolverContext&, const P&) {}   // no per-edge / topo in 3-D

    static void init_aux_slabs(SolverContext& solver, Wavefield& wf)
    {
        elastic_init_aux_slabs(solver, wf);
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.bind(p.pml_vals, 3);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, 3, d.B, d.nz, d.ny, d.nx};   // only the velocities
    }

    static float* field_ptr(WfView& wf, int field_idx)
    {
        return elastic_field_ptr(wf, 3, field_idx);
    }

    static WfView view(Wavefield& wf) { return wf.view(); }

    // ===================================================================== //
    // [2] FORWARD — sg_generic_forward, per it in [it_begin, it_end):
    //   velocity_substep -> stress_substep -> inject_source ->
    //   <checkpoint save: shared runtime> -> save_boundary_fields ->
    //   record_field; after the loop: save_last_state.
    // ===================================================================== //

    // (also used by the ckpt/recursive binds in section [5])
    static void backfill_syzx(Wavefield& wf, const Buf& /*like*/)
    {
        SWEEP_CHECK(wf.m_syzx_t.defined(),
                    "elastic3d: the bound wavefield list must carry m_syzx "
                    "(36-slot layout); nothing allocates it here");
    }

    // MANDATORY: _c.py Prop.forward always hands the compiled forward its
    // propagation state (persistent _slice_wavefield_buffers in full mode,
    // per-call _transient_forward_wavefields otherwise), sized by
    // cuda_layout.base_nvar + pml_nvar = CKPT_NVAR slots.
    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p,
                                      const Buf& vp)
    {
        SWEEP_CHECK((int)p.wavefields.size() == CKPT_NVAR,
                    "elastic3d/forward requires the propagator-bound wavefields "
                    "(cuda_layout.base_nvar + cuda_layout.pml_nvar = ", CKPT_NVAR,
                    " tensors), got ", p.wavefields.size());
        wf.bind(p.wavefields, true);
        backfill_syzx(wf, vp);
    }

    // No-op: this equation writes u_allt from inside its stress kernel.
    // See the call site in sg_driver.cuh for why the hook exists.
    static void capture_allt(Buf&, WfView&, const SolverContext&, int) {}

    static void velocity_substep(const State& s, WfView& wf,
                                 ElasticCPMLPointer cpml_view,
                                 const SolverContext& solver)
    {
        LAUNCH_3DELASTIC_VELOCITY(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            wf,
            s.models.rho.data_ptr<float>(),
            s.grad_ctx,
            cpml_view,
            solver
        );
    }

    static void stress_substep(const State& s, WfView& wf,
                               ElasticCPMLPointer cpml_view,
                               const SolverContext& solver, float* u_this_t)
    {
        LAUNCH_3DELASTIC_STRESS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            wf,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            u_this_t,
            s.grad_ctx,
            cpml_view,
            solver
        );
    }

    static void inject_source(const State& s, const SolverContext& solver,
                              float* field, const Buf& source,
                              const Buf& sources_loc, int it, int nsrc)
    {
        add_source_3d<<<s.source_config.grid, s.source_config.block>>>(
            field,
            source.data_ptr<float>(),
            sources_loc.data_ptr<int>(),
            it,
            nsrc,
            solver
        );
    }

    static void save_boundary_fields(BoundaryRuntime& rt, const State& s,
                                     const SolverContext& solver, WfView& wf,
                                     int it, int nt,
                                     const GeneralBoundaryPointer& bs,
                                     int save_width)
    {
        float* fields[9] = {
            wf.vx, wf.vy, wf.vz,
            wf.sxx, wf.syy, wf.szz,
            wf.sxy, wf.sxz, wf.syz
        };
        for (int f = 0; f < 9; ++f) {
            rt.save_forward_3d_field(
                it,
                nt,
                fields[f],
                s.launch_config.grid,
                s.launch_config.block,
                bs,
                save_width,
                -solver.M, // offset
                solver,
                f,
                f == 8
            );
        }
    }

    static void record_field(const State& s, const SolverContext& solver,
                             float* field, Buf& record, int irec,
                             const Buf& receivers_loc, int it, int nrec)
    {
        record_kernel_3d<<<s.record_config.grid, s.record_config.block>>>(
            field,
            record.select(0, irec).data_ptr<float>(),
            receivers_loc.data_ptr<int>(),
            it,
            nrec,
            solver
        );
    }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        copy_tensor_cuda_async(saver.last_two.select(0, 0).select(0, 0), wf.vx_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 1).select(0, 0), wf.vy_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 2).select(0, 0), wf.vz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 3).select(0, 0), wf.sxx_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 4).select(0, 0), wf.syy_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 5).select(0, 0), wf.szz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 6).select(0, 0), wf.sxy_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 7).select(0, 0), wf.sxz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 8).select(0, 0), wf.syz_t);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — sg_generic_backward, per reverse it:
    //   fix_rho_grad_at_sources -> inject_residuals -> vel_ptrs_from_u_forward;
    //   it == 0: image_standalone + fix_rho_grad_at_receivers, loop ends;
    //   it  > 0: full_mode_step (imaging + receiver-rho + adjoint step, in
    //            the equation's exact fused order).
    // ===================================================================== //

    // ---- shared adjoint launches ----------------------------------------- //
    // The four adjoint launches shared by full_mode_step, plain_adjoint_step
    // (ckpt/recursive, section [5]) and bs_stress_half / bs_velocity_half (section [4]);
    // the bodies are those hooks' former inline launch statements, verbatim.
private:
    static void stress_adjoint_prepare(const State& s, const SolverContext& solver,
                                       WfView& adj_view, Workspace& workspace,
                                       ElasticCPMLPointer cpml_view,
                                       const float* vx_now, const float* vy_now,
                                       const float* vz_now,
                                       const float* vx_next, const float* vy_next,
                                       const float* vz_next,
                                       const float* vp, const float* vs,
                                       const float* rho,
                                       float* g_vp, float* g_vs, float* g_rho)
    {
        LAUNCH_3DELASTIC_STRESS_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            cpml_view,
            solver,
            workspace.qxx_t.data_ptr<float>(),
            workspace.qxy_t.data_ptr<float>(),
            workspace.qxz_t.data_ptr<float>(),
            workspace.qyx_t.data_ptr<float>(),
            workspace.qyy_t.data_ptr<float>(),
            workspace.qyz_t.data_ptr<float>(),
            workspace.qzx_t.data_ptr<float>(),
            workspace.qzy_t.data_ptr<float>(),
            workspace.qzz_t.data_ptr<float>(),
            s.grad_ctx,
            vx_now, vy_now, vz_now,
            vx_next, vy_next, vz_next,
            vp,
            vs,
            rho,
            g_vp,
            g_vs,
            g_rho
        );
    }

    static void stress_adjoint_apply(const State& s, const SolverContext& solver,
                                     WfView& adj_view, Workspace& workspace)
    {
        LAUNCH_3DELASTIC_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace.qxx_t.data_ptr<float>(),
            workspace.qxy_t.data_ptr<float>(),
            workspace.qxz_t.data_ptr<float>(),
            workspace.qyx_t.data_ptr<float>(),
            workspace.qyy_t.data_ptr<float>(),
            workspace.qyz_t.data_ptr<float>(),
            workspace.qzx_t.data_ptr<float>(),
            workspace.qzy_t.data_ptr<float>(),
            workspace.qzz_t.data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    static void velocity_adjoint_prepare(const State& s, const SolverContext& solver,
                                         WfView& adj_view, Workspace& workspace,
                                         ElasticCPMLPointer cpml_view)
    {
        LAUNCH_3DELASTIC_VELOCITY_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.models.rho.data_ptr<float>(),
            cpml_view,
            solver,
            workspace.pxx_t.data_ptr<float>(),
            workspace.pxy_t.data_ptr<float>(),
            workspace.pxz_t.data_ptr<float>(),
            workspace.pyx_t.data_ptr<float>(),
            workspace.pyy_t.data_ptr<float>(),
            workspace.pyz_t.data_ptr<float>(),
            workspace.pzx_t.data_ptr<float>(),
            workspace.pzy_t.data_ptr<float>(),
            workspace.pzz_t.data_ptr<float>()
        );
    }

    static void velocity_adjoint_apply(const State& s, const SolverContext& solver,
                                       WfView& adj_view, Workspace& workspace)
    {
        LAUNCH_3DELASTIC_VELOCITY_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace.pxx_t.data_ptr<float>(),
            workspace.pxy_t.data_ptr<float>(),
            workspace.pxz_t.data_ptr<float>(),
            workspace.pyx_t.data_ptr<float>(),
            workspace.pyy_t.data_ptr<float>(),
            workspace.pyz_t.data_ptr<float>(),
            workspace.pzx_t.data_ptr<float>(),
            workspace.pzy_t.data_ptr<float>(),
            workspace.pzz_t.data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    // Velocity-adjoint half (prepare then apply): full_mode_step,
    // plain_adjoint_step and bs_velocity_half run it back to back.
    static void velocity_adjoint_half(const State& s, const SolverContext& solver,
                                      WfView& adj_view, Workspace& workspace,
                                      ElasticCPMLPointer cpml_view)
    {
        velocity_adjoint_prepare(s, solver, adj_view, workspace, cpml_view);
        velocity_adjoint_apply(s, solver, adj_view, workspace);
    }
public:

    // ---- backward hooks -------------------------------------------------- //

    // MANDATORY, every backward mode: _ensure_wavefield_buffers allocates the
    // adjoint set whenever a gradient is asked for and Wrapper.backward binds
    // it (zeroed) as adjoint_wavefields.
    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInputCore& p,
                                      const Buf& vp)
    {
        SWEEP_CHECK((int)p.adjoint_wavefields.size() == ADJ_WF_COUNT,
                    "elastic3d/backward requires the propagator-bound "
                    "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar + "
                    "adjoint_extra_nvar = ", ADJ_WF_COUNT, " tensors), got ",
                    p.adjoint_wavefields.size());
        wf.bind(p.adjoint_wavefields, true);
    }

    static void zero_adjoint_if_first_segment(Wavefield& adjoint, bool first_segment)
    {
        // FIRST segment only: a continuation call must keep the carried
        // adjoint state (legacy monolithic calls always have bw_begin == nt).
        if (first_segment)
            zero_wavefield_state(adjoint);
    }

    // The model-gradient accumulators, MANDATORY in every mode: _c.py
    // Wrapper.backward allocates them per backward call (_gradient_buffers,
    // one per model; cuda_layout.grads_out_has_wavelet is false here) and
    // binds them as grads_out.
    static void bind_grads(const BackwardInputCore& p, std::vector<Buf>& grads)
    {
        SWEEP_CHECK(p.grads_out.size() == 3,
                    "elastic3d/backward requires the propagator-bound grads_out "
                    "holding exactly {grad_vp, grad_vs, grad_rho} "
                    "(cuda_layout.grads_out_has_wavelet == false), got ",
                    p.grads_out.size());
        grads = {p.grads_out[0], p.grads_out[1], p.grads_out[2]};
    }

    static std::vector<float> adjoint_source_signs(
        const BackwardInputCore& p, IntSpan receiver_fields)
    {
        return elastic_adjoint_source_signs(p.adjoint_source, receiver_fields, 3);
    }

    struct VelPtrs {
        Buf now_vx, now_vy, now_vz;   // Tensors: the imaging view
        const float* now[3];
        const float* next[3];
    };

    // (also used by vel_ptrs_from_seg / vel_ptrs_from_carriers in section [5])
    static VelPtrs _vel_ptrs(const Buf& nvx, const Buf& nvy,
                             const Buf& nvz,
                             const float* nx0, const float* nx1, const float* nx2)
    {
        VelPtrs v;
        v.now_vx = nvx; v.now_vy = nvy; v.now_vz = nvz;
        v.now[0] = v.now_vx.data_ptr<float>();
        v.now[1] = v.now_vy.data_ptr<float>();
        v.now[2] = v.now_vz.data_ptr<float>();
        v.next[0] = nx0; v.next[1] = nx1; v.next[2] = nx2;
        return v;
    }

    static VelPtrs vel_ptrs_from_u_forward(const BackwardInputCore& p, int it,
                                             const Buf& zero_velocity)
    {
        const bool has_next = (it + 1 < p.nt);
        return _vel_ptrs(
            p.u_forward.select(0, it).select(0, 0),
            p.u_forward.select(0, it).select(0, 1),
            p.u_forward.select(0, it).select(0, 2),
            has_next ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>()
                     : zero_velocity.data_ptr<float>(),
            has_next ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>()
                     : zero_velocity.data_ptr<float>(),
            has_next ? p.u_forward.select(0, it + 1).select(0, 2).data_ptr<float>()
                     : zero_velocity.data_ptr<float>());
    }

    static void fix_rho_grad_at_sources(const State& s, const SolverContext& solver,
                                WfView& adj_view, const BackwardInputCore& p,
                                IntSpan source_fields, int it,
                                std::vector<Buf>& grads)
    {
        if (it < 0 || it >= p.nt) return;
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            const int sfield = source_fields[isrc];
            if (sfield > 2) continue;                       // vx = 0, vy = 1, vz = 2
            float* adj_field = elastic_field_ptr(adj_view, 3, sfield);
            if (adj_field == nullptr) continue;
            add_body_force_rho_grad_correction<<<s.source_config.grid, s.source_config.block>>>(
                grads[2].data_ptr<float>(),
                adj_field,
                s.models.rho.data_ptr<float>(),
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                (int)p.forward_sources_loc.size(1),
                3,
                solver
            );
        }
    }

    static void inject_residuals(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const BackwardInputCore& p,
                                 IntSpan receiver_fields,
                                 const std::vector<float>& adj_source_signs,
                                 int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.size(); ++irec) {
            float* field = elastic_field_ptr(adj_view, 3, receiver_fields[irec]);
            if (field == nullptr) continue;
            add_source_3d_signed<<<s.record_config.grid, s.record_config.block>>>(
                field,
                p.adjoint_source.select(0, irec).data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                adj_source_signs[irec],
                solver
            );
        }
    }

    // A velocity-only wavefield view for the gradient kernel: the stress
    // slots alias vx (never read by the imaging).
    static Wavefield make_velocity_view(const Buf& vx,
                                        const Buf& vy,
                                        const Buf& vz)
    {
        Wavefield view;
        view.dim = 3;
        view.use_pml = false;
        view.allocated = true;
        view.vx_t = vx;
        view.vy_t = vy;
        view.vz_t = vz;
        view.sxx_t = vx;
        view.syy_t = vx;
        view.szz_t = vx;
        view.sxy_t = vx;
        view.sxz_t = vx;
        view.syz_t = vx;
        return view;
    }

    static void image_standalone(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const VelPtrs& v,
                                 std::vector<Buf>& grads)
    {
        auto current_forward = make_velocity_view(v.now_vx, v.now_vy, v.now_vz);
        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            current_forward.view(),
            adj_view,
            v.next[0],
            v.next[1],
            v.next[2],
            s.models.vp.data_ptr<float>(),
            s.models.vs.data_ptr<float>(),
            s.models.rho.data_ptr<float>(),
            grads[0].data_ptr<float>(),
            grads[1].data_ptr<float>(),
            grads[2].data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    static void fix_rho_grad_at_receivers(const State& s, const SolverContext& solver,
                                  std::vector<Buf>& grads,
                                  const VelPtrs& v, const BackwardInputCore& p,
                                  IntSpan receiver_fields,
                                  int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.size(); ++irec) {
            const int field = receiver_fields[irec];
            if (field > 2) continue;                      // stress receiver: no rho term
            sub_receiver_rho_grad_correction<<<s.record_config.grid, s.record_config.block>>>(
                grads[2].data_ptr<float>(),
                v.now[field],
                v.next[field],
                s.models.rho.data_ptr<float>(),
                p.adjoint_source.select(0, irec).data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                3,
                solver.M,                                 // imaging halo (order/2 == M)
                solver
            );
        }
    }

    static void full_mode_step(const State& s, const SolverContext& solver,
                                Wavefield& adjoint, Workspace& workspace,
                                ElasticCPMLPointer cpml_view,
                                const VelPtrs& v,
                                std::vector<Buf>& grads,
                                const BackwardInputCore& p,
                                IntSpan receiver_fields,
                                int it, int adjoint_nsrc)
    {
        auto adj_view = adjoint.view();
        stress_adjoint_prepare(s, solver, adj_view, workspace, cpml_view,
                               v.now[0], v.now[1], v.now[2],
                               v.next[0], v.next[1], v.next[2],
                               s.models.vp.data_ptr<float>(),
                               s.models.vs.data_ptr<float>(),
                               s.models.rho.data_ptr<float>(),
                               grads[0].data_ptr<float>(),
                               grads[1].data_ptr<float>(),
                               grads[2].data_ptr<float>());
        stress_adjoint_apply(s, solver, adj_view, workspace);
        velocity_adjoint_half(s, solver, adj_view, workspace, cpml_view);

        fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it,
                          adjoint_nsrc);
    }

    // (fires only in the ckpt/recursive modes' reverse sweep — section [5])
    static void plain_adjoint_step(const State& s, const SolverContext& solver,
                                   Wavefield& adjoint, Workspace& workspace,
                                   ElasticCPMLPointer cpml_view)
    {
        auto adj_view = adjoint.view();
        stress_adjoint_prepare(s, solver, adj_view, workspace, cpml_view,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr);
        stress_adjoint_apply(s, solver, adj_view, workspace);
        velocity_adjoint_half(s, solver, adj_view, workspace, cpml_view);
    }

    // ===================================================================== //
    // [4] BACKWARD_BS — sg_generic_backward_bs, per reverse it (floor
    //   max(it_lo, 1)):
    //   fix_rho_grad_at_sources / inject_residuals / uninject_forward_source ->
    //   bs_stress_half (stress recon NOPML + strip restore + imaging +
    //   receiver-rho + stress-adjoint half) -> bs_velocity_half (velocity-adjoint
    //   half + carrier capture + velocity recon NOPML + strip restore +
    //   prefetch); before the loop (first segment): seed_recon from
    //   u_last_two.
    // ===================================================================== //

    static void zero_adjoint_if_first_segment_bs(Wavefield&, bool) {}

    struct ReconCarriers {
        Buf fvx_prev, fvy_prev, fvz_prev;
    };

    static ReconCarriers bind_or_alloc_recon(Wavefield& forward,
                                             const BackwardInputCore& p,
                                             const Buf& vp)
    {
        // MANDATORY: the bs backward's reconstruction state is
        // cuda_layout.reconstruction_nvar (slot_table ELASTIC3D.recon = 12)
        // grids -- the 9 physical fields plus the three fv*_prev carriers --
        // allocated per backward call by _forward_state_buffers and bound as
        // forward_wavefields.  The old lenient "any other full list, carriers
        // allocated here" branch had no Python caller: _c.py, the stepped
        // runner and the DD driver all bind exactly RECON_WF_COUNT.
        ReconCarriers c;
        wavefields_required(p.forward_wavefields, RECON_WF_COUNT, vp,
                            "elastic3d/bs reconstruction list "
                            "[9 fields, fvx_prev, fvy_prev, fvz_prev] "
                            "(cuda_layout.reconstruction_nvar)");
        forward.bind(std::vector<Buf>(p.forward_wavefields.begin(),
                                                p.forward_wavefields.begin() + 9),
                     /*use_pml=*/false);
        c.fvx_prev = p.forward_wavefields[9];
        c.fvy_prev = p.forward_wavefields[10];
        c.fvz_prev = p.forward_wavefields[11];
        return c;
    }

    static void seed_recon(Wavefield& forward, const BackwardInputCore& p)
    {
        copy_tensor_cuda_async(forward.vx_t, p.u_last_two.select(0, 0).select(0, 0));
        copy_tensor_cuda_async(forward.vy_t, p.u_last_two.select(0, 1).select(0, 0));
        copy_tensor_cuda_async(forward.vz_t, p.u_last_two.select(0, 2).select(0, 0));
        copy_tensor_cuda_async(forward.sxx_t, p.u_last_two.select(0, 3).select(0, 0));
        copy_tensor_cuda_async(forward.syy_t, p.u_last_two.select(0, 4).select(0, 0));
        copy_tensor_cuda_async(forward.szz_t, p.u_last_two.select(0, 5).select(0, 0));
        copy_tensor_cuda_async(forward.sxy_t, p.u_last_two.select(0, 6).select(0, 0));
        copy_tensor_cuda_async(forward.sxz_t, p.u_last_two.select(0, 7).select(0, 0));
        copy_tensor_cuda_async(forward.syz_t, p.u_last_two.select(0, 8).select(0, 0));
    }

    static void uninject_forward_source(const State& s, const SolverContext& solver,
                                        WfView& for_view, const BackwardInputCore& p,
                                        IntSpan source_fields,
                                        int it, int forward_nsrc)
    {
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            float* field = elastic_field_ptr(for_view, 3, source_fields[isrc]);
            if (field == nullptr) continue;
            add_source_3d_signed<<<s.source_config.grid, s.source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                -1.0f,
                solver
            );
        }
    }

    static void bs_stress_half(const State& s, const SolverContext& solver,
                          WfView& for_view, WfView& adj_view,
                          Wavefield& adjoint, Workspace& workspace,
                          ElasticCPMLPointer cpml_view,
                          BoundaryRuntime& boundary_runtime,
                          const GeneralBoundaryPointer& bs, int save_width,
                          std::vector<Buf>& grads,
                          ReconCarriers& carriers, const BackwardInputCore& p,
                          IntSpan receiver_fields,
                          int it, int adjoint_nsrc)
    {
        LAUNCH_3DELASTIC_STRESS_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            s.grad_ctx,
            solver
        );

        float* field2[6] = {for_view.sxx, for_view.syy, for_view.szz,
                            for_view.sxy, for_view.sxz, for_view.syz};
        for (int f = 3; f < 9; ++f) {
            boundary_runtime.restore_backward_3d_field(
                it,
                field2[f - 3],
                s.launch_config.grid,
                s.launch_config.block,
                bs,
                save_width,
                -solver.M,
                solver,
                f,
                f == 3,
                false
            );
        }

        // Imaging folded into the stress-adjoint-prepare kernel (as in FULL
        // mode): it carries the calculate_grad_elastic3d_bs block verbatim and
        // already streams the adjoint stresses, so the standalone imaging
        // pass is gone.  for_view.v* = v(it) and the carriers = v(it+1) are
        // final here, and prepare touches nothing the imaging reads.

        stress_adjoint_prepare(s, solver, adj_view, workspace, cpml_view,
                               for_view.vx, for_view.vy, for_view.vz,
                               carriers.fvx_prev.data_ptr<float>(),
                               carriers.fvy_prev.data_ptr<float>(),
                               carriers.fvz_prev.data_ptr<float>(),
                               s.models.vp.data_ptr<float>(),
                               s.models.vs.data_ptr<float>(),
                               s.models.rho.data_ptr<float>(),
                               grads[0].data_ptr<float>(),
                               grads[1].data_ptr<float>(),
                               grads[2].data_ptr<float>());

        {
            // Receiver-cell rho correction AFTER the imaging '+=' (same per-cell
            // accumulation order as the standalone pass); operands unchanged.
            // Sits BETWEEN the stress prepare and apply launches on purpose.
            VelPtrs v;
            v.now[0] = for_view.vx;
            v.now[1] = for_view.vy;
            v.now[2] = for_view.vz;
            v.next[0] = carriers.fvx_prev.data_ptr<float>();
            v.next[1] = carriers.fvy_prev.data_ptr<float>();
            v.next[2] = carriers.fvz_prev.data_ptr<float>();
            fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it,
                              adjoint_nsrc);
        }
        stress_adjoint_apply(s, solver, adj_view, workspace);
    }

    static void bs_velocity_half(const State& s, const SolverContext& solver,
                          WfView& for_view, Wavefield& adjoint,
                          Workspace& workspace, ElasticCPMLPointer cpml_view,
                          BoundaryRuntime& boundary_runtime,
                          const GeneralBoundaryPointer& bs, int save_width,
                          ReconCarriers& carriers, Wavefield& forward,
                          int it, int nt)
    {
        auto adj_view = adjoint.view();
        velocity_adjoint_half(s, solver, adj_view, workspace, cpml_view);

        // Carrier capture v(it+1); see the 2-D twin.
        {
            const int wxl = solver.cut_x_lo() ? 0 : save_width;
            const int wxh = solver.cut_x_hi() ? 0 : save_width;
            const int wyl = solver.cut_y_lo() ? 0 : save_width;
            const int wyh = solver.cut_y_hi() ? 0 : save_width;
            const int wzl = solver.cut_z_lo() ? 0 : save_width;
            const int wzh = solver.cut_z_hi() ? 0 : save_width;
            const int bx = solver.phys_x1() - solver.phys_x0();
            const int by = solver.phys_y1() - solver.phys_y0();
            const int bzi = solver.phys_z1() - solver.phys_z0() - wzl - wzh;
            const int byi = by - wyl - wyh;
            const int n_strip = (wzl + wzh) * bx * by
                              + (wyl + wyh) * bx * (bzi > 0 ? bzi : 0)
                              + (wxl + wxh) * (byi > 0 ? byi : 0) * (bzi > 0 ? bzi : 0);
            if (n_strip > 0) {
                dim3 strip_grid((n_strip + 255) / 256, solver.B);
                elastic_capture_strips_3d<<<strip_grid, 256>>>(
                    for_view.vx, for_view.vy, for_view.vz,
                    carriers.fvx_prev.data_ptr<float>(),
                    carriers.fvy_prev.data_ptr<float>(),
                    carriers.fvz_prev.data_ptr<float>(),
                    solver.nx, solver.ny, solver.nz,
                    solver.phys_x0(), solver.phys_x1(), solver.phys_y0(), solver.phys_y1(),
                    solver.phys_z0(), solver.phys_z1(),
                    wxl, wxh, wyl, wyh, wzl, wzh
                );
            }
        }

        LAUNCH_3DELASTIC_VELOCITY_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            s.models.rho.data_ptr<float>(),
            s.grad_ctx,
            solver,
            carriers.fvx_prev.data_ptr<float>(),
            carriers.fvy_prev.data_ptr<float>(),
            carriers.fvz_prev.data_ptr<float>()
        );

        float* field1[3] = {for_view.vx, for_view.vy, for_view.vz};
        for (int f = 0; f < 3; ++f) {
            boundary_runtime.restore_backward_3d_field(
                it,
                field1[f],
                s.launch_config.grid,
                s.launch_config.block,
                bs,
                save_width,
                -solver.M,
                solver,
                f,
                false,
                f == 2
            );
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, nt);
    }

    // ===================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING
    // sg_generic_backward_ckpt, per chunk: replay velocity/stress substeps +
    //   save_seg_velocities / inject_forward_sources; reverse: fix_rho_grad_at_sources /
    //   inject_residuals / vel_ptrs_from_seg / image_standalone /
    //   fix_rho_grad_at_receivers / (it > 0) plain_adjoint_step; after each chunk:
    //   export_seg_next_v.
    // sg_generic_backward_recursive_ckpt, per reverse it: fix_rho_grad_at_sources /
    //   inject_residuals; sg_replay_forward_to_time: velocity/stress
    //   substeps + capture_velocities; vel_ptrs_from_carriers / image_standalone /
    //   fix_rho_grad_at_receivers / (it > 0) plain_adjoint_step.
    // ===================================================================== //

    static void bind_or_alloc_recon_ckpt(Wavefield& forward,
                                         const BackwardInputCore& p,
                                         const Buf& vp)
    {
        // Set 0 of the Python-bound replay state (one set: no bisection
        // scratch here), the full bind -- the CPML aux slots are slab-shaped
        // exactly like the forward's.  MANDATORY: _forward_state_shapes("ckpt"
        // / "recursive") derives CKPT_STATE_COUNT slots from the forward slot
        // table and Wrapper.backward binds them as forward_wavefields on both
        // checkpoint entries.
        SWEEP_CHECK((int)p.forward_wavefields.size() >= CKPT_STATE_COUNT,
                    "elastic3d/ckpt requires the propagator-bound replay state "
                    "(cuda_layout.slots, the forward slot list: ", CKPT_STATE_COUNT,
                    " tensors per set), got ", p.forward_wavefields.size());
        forward.bind(wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT,
                                   "elastic3d ckpt replay state"), true);
        backfill_syzx(forward, vp);
    }

    static void check_ckpt_aux_layout(const Wavefield& forward,
                                      const Wavefield& adjoint)
    {
        SWEEP_CHECK(!adjoint.m_vxx_t.defined() ||
                    forward.m_vxx_t.sizes() == adjoint.m_vxx_t.sizes(),
                    "checkpoint aux layout differs from adjoint aux layout");
    }

    // ---- seg / carrier plumbing (ckpt + recursive modes) ----

    // The chunked backward's velocity histories, checkpoint_replay slots
    // [0, N_VEL) = vx, vy, vz: (max_rows, B, 1, nz, ny, nx) each, bound at the
    // longest chunk (cuda_layout.checkpoint_replay_shapes) and narrowed per
    // chunk by the skeleton; row 0 = v(start), row k = v(start + k).  Never
    // re-zeroed: every row a chunk reads it wrote first.  MANDATORY in the
    // chunked mode: cuda_layout.checkpoint_replay_shapes declares N_VEL
    // histories for mode "ckpt" and _ensure_checkpoint_buffers allocates them
    // next to the snapshots.
    static std::vector<Buf> seg_buffers(const BackwardInputCore& p,
                                                  const Buf& vp, int max_rows)
    {
        SWEEP_CHECK((int)p.checkpoint_replay.size() >= N_VEL,
                    "elastic3d/ckpt requires the propagator-bound checkpoint_replay "
                    "(cuda_layout.checkpoint_replay_shapes, ", N_VEL,
                    " velocity histories), got ", p.checkpoint_replay.size());
        std::vector<Buf> seg;
        seg.reserve(N_VEL);
        for (int c = 0; c < N_VEL; ++c)
            seg.push_back(pool_required(p.checkpoint_replay, c,
                                        {max_rows, vp.size(0) * vp.size(1), 1,
                                         vp.size(2), vp.size(3), vp.size(4)}, "checkpoint_replay"));
        return seg;
    }

    static void save_seg_velocities(std::vector<Buf>& seg, Wavefield& forward,
                            int slot)
    {
        copy_tensor_cuda_async(seg[0].select(0, slot), forward.vx_t);
        copy_tensor_cuda_async(seg[1].select(0, slot), forward.vy_t);
        copy_tensor_cuda_async(seg[2].select(0, slot), forward.vz_t);
    }

    static void inject_forward_sources(const State& s, const SolverContext& solver,
                                      WfView& for_view, const BackwardInputCore& p,
                                      IntSpan source_fields, int it)
    {
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            float* field = elastic_field_ptr(for_view, 3, source_fields[isrc]);
            if (field == nullptr) continue;
            add_source_3d<<<s.source_config.grid, s.source_config.block>>>(
                field,
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                (int)p.forward_sources_loc.size(1),
                solver
            );
        }
    }

    static VelPtrs vel_ptrs_from_seg(const std::vector<Buf>& seg,
                                int now_offset, int next_offset,
                                const std::vector<Buf>& next_segment_v)
    {
        return _vel_ptrs(
            seg[0].select(0, now_offset),
            seg[1].select(0, now_offset),
            seg[2].select(0, now_offset),
            (next_offset >= 0) ? seg[0].select(0, next_offset).data_ptr<float>()
                               : next_segment_v[0].data_ptr<float>(),
            (next_offset >= 0) ? seg[1].select(0, next_offset).data_ptr<float>()
                               : next_segment_v[1].data_ptr<float>(),
            (next_offset >= 0) ? seg[2].select(0, next_offset).data_ptr<float>()
                               : next_segment_v[2].data_ptr<float>());
    }

    static void export_seg_next_v(std::vector<Buf>& prev,
                                   const std::vector<Buf>& seg)
    {
        copy_tensor_cuda_async(prev[0], seg[0].select(0, 1));
        copy_tensor_cuda_async(prev[1], seg[1].select(0, 1));
        copy_tensor_cuda_async(prev[2], seg[2].select(0, 1));
    }

    static void capture_velocities(std::vector<Buf>& v, Wavefield& forward)
    {
        copy_tensor_cuda_async(v[0], forward.vx_t);
        copy_tensor_cuda_async(v[1], forward.vy_t);
        copy_tensor_cuda_async(v[2], forward.vz_t);
    }

    static VelPtrs vel_ptrs_from_carriers(const std::vector<Buf>& current_v,
                                    const std::vector<Buf>& next_v)
    {
        return _vel_ptrs(current_v[0], current_v[1], current_v[2],
                         next_v[0].data_ptr<float>(),
                         next_v[1].data_ptr<float>(),
                         next_v[2].data_ptr<float>());
    }
};

} // namespace elastic3d
