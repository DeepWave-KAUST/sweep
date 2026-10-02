// Driver traits for the 2-D DAS-Mu equation: the per-equation half of the
// staggered-family skeleton in ``common/sg_driver.cuh``.  Line-faithful
// transcription of the hand-written drivers; physics kernels untouched.
// Deltas vs the elastic 2-D reference worth naming:
//   * the velocity substep IS elastic2d's kernel, reached through the
//     wavefield's elastic_view() adapter; the stress substep is the custom
//     stress+strain kernel (strain integration lives inside it), so the
//     per-step view is a PAIR (das view + elastic view);
//   * the CPML memory variables stay full-domain: identity aux slabs MUST be
//     installed before any kernel launch (the aux-slab race incident);
//   * 8-field boundary lists (5 elastic + 3 strains; only the elastic five
//     are restored -- strains are record-only), 18 wavefield tensors,
//     8-field last_two with a lenient 5-field legacy read;
//   * the bs reconstruction is the 7-tensor Python-bound list [vx, vz, sxx,
//     szz, sxz + fvx_prev, fvz_prev carriers] -- elastic2d's list.  The
//     reverse loop steps the elastic fields with the elastic NOPML kernels
//     and images with the elastic bs kernel, so the DAS strains (and the
//     CPML memory) are never touched there and are not carried
//     (bind_elastic).  The list is MANDATORY -- there is no self-allocating
//     fallback left;
//   * the full backward has NO grad fusion: standalone imaging, then the
//     receiver-rho correction, THEN the adjoint step;
//   * checkpoints snapshot the full-domain state (full grids, not per-axis
//     aux slabs), and there is no aux-layout check; the
//     ckpt/recursive replay state is set 0 of the Python-bound
//     forward_wavefields (CKPT_STATE_COUNT = 18, the full bind with CPML
//     memory), the per-segment vx/vz histories are the two
//     checkpoint_replay slots (seg_buffers) and the velocity carriers sit in
//     the adjoint workspace pool behind the 8 struct slots (WS_CARRIERS);
//   * no DD cut support (CUT_MASK_BITS = 0): the borrowed kernels are not
//     cut-aware.
// What the skeleton ADDS (dormant on legacy calls): the stepped range,
// Python-bound record/wavefield/gradient buffers, the physics phase split,
// and loud stepped/phase validation this file never had.  The dead
// ``f_this`` scratch allocation of the old backward_bs is dropped.
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/sg_driver.cuh.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/das_mu.h"
#include "../../common/elastic.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../common/eq_driver.cuh"
#include "../../common/sg_driver.cuh"
#include "../../launch/config.h"

namespace das_mu2d {

struct Driver {
    // =================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // Prologue of every entry point (sg_driver.cuh HOOK TIMING MAP, in call
    // order): validate_backward (backward only), parse_models, setup_ctx,
    // bind_or_alloc_* wavefields, init_aux_slabs, alloc_cpml, bind_grads /
    // make_workspace, make_state, adjoint_source_signs.
    // Constants, types and shared constructors live here; the per-mode
    // bind_or_alloc_* / grads / signed-sources hooks sit at the head of
    // their sections below.
    // =================================================================== //

    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "das_mu2d";
    static constexpr int CKPT_NVAR = 18;
    static constexpr const char* CKPT_COUNT_MSG =
        "DAS Mu 2D checkpointing expects 18 checkpoint tensors";
    static constexpr const char* CKPT_RECURSIVE_COUNT_MSG =
        "DAS Mu 2D recursive checkpointing expects 18 checkpoint tensors";
    // Checkpoint replay state (ckpt + recursive): one set of the Python-bound
    // forward_wavefields, in the forward's full bind order -- 8 fields + 10
    // CPML memory tensors, all full-grid (no aux slabs in this equation).
    // One set only: the recursive backward keeps no per-level scratch here.
    static constexpr int CKPT_STATE_COUNT = CKPT_NVAR;
    static constexpr int BS_NVAR = 8;   // vx, vz, sxx, szz, sxz, exx, ezz, exz
    // The bs reconstruction carries the elastic five only (elastic2d's list):
    // the reverse loop steps them with the elastic NOPML kernels and images
    // with the elastic bs kernel, so the DAS strains are dead there.
    static constexpr int BS_ELASTIC_NVAR = 5;   // vx, vz, sxx, szz, sxz
    static constexpr int CUT_MASK_BITS = 0x0;
    static constexpr const char* CUT_MASK_DESC =
        "no bits (the borrowed elastic kernels are not cut-aware)";
    static constexpr int ADJ_WF_COUNT = 18;
    static constexpr int RECON_WF_COUNT = 7;
    static constexpr const char* RECON_LIST_DESC =
        "[vx, vz, sxx, szz, sxz, fvx_prev, fvz_prev]";
    // Slots of the reconstruction list: [0, BS_ELASTIC_NVAR) are the elastic
    // fields (bound without strains or CPML memory), then the v(it+1)
    // carriers.
    static constexpr int RECON_SLOT_FVX_PREV = BS_ELASTIC_NVAR;       // 5
    static constexpr int RECON_SLOT_FVZ_PREV = BS_ELASTIC_NVAR + 1;   // 6
    static_assert(RECON_SLOT_FVZ_PREV + 1 == RECON_WF_COUNT,
                  "das_mu2d reconstruction list = elastic fields + 2 carriers");
    static constexpr int N_VEL = 2;
    static constexpr bool IMAGING_USES_NEXT_V = true;   // imaging consumes v(t+1) carriers

    using Wavefield = DasMuWavefieldTensor2D;
    using CPML = ElasticCPMLTensor;

    // Per-step view pair: the custom stress/strain kernel and the field maps
    // read the DAS view; the borrowed elastic kernels read the adapter.
    struct WfView {
        DasMuWavefieldPointer2D das;
        ElasticWavefieldPointer el;
    };

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
        const auto lame = derived::lame(p, m.vp, m.vs, m.rho, "das_mu2d::parse_models");
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
        int nx, nz, B;
    };

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d, const Models& models,
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
        s.nx = d.nx;
        s.nz = d.nz;
        s.B = d.B;
        return s;
    }

    // (adjoint scratch; consumed by the backward hooks in sections [3]-[4])
    using Workspace = ElasticAdjointWorkspaceTensor;

    // Adjoint workspace pool (cuda_layout.backward_workspace_shapes): the 8
    // elastic adjoint scratch grids the struct binds [qxx, qzz, qxz, qzx,
    // pxx, pzz, pxz, pzx], then -- ckpt / recursive modes only -- the
    // velocity carriers the skeleton takes at WS_CARRIERS + c (v(t):
    // current_v / next_segment_v) and WS_CARRIERS + N_VEL + c (v(t+1):
    // next_v / prev_segment_next_v), sg_driver.cuh SgCarrierSlots.  The
    // struct's bind takes the head of the longer pool (elastic.h).
    static constexpr int WS_CARRIERS = 8;

    static Workspace make_workspace(const BackwardInputCore& p, const Buf& vp)
    {
        Workspace workspace;
        bind_adjoint_workspace_required(workspace, p.adjoint_workspace, 2,
                                        "das_mu2d backward");
        return workspace;
    }

    static void validate_forward(const ForwardInputCore&) {}
    static void validate_backward(const BackwardInputCore&, const char*) {}

    template <class P>
    static void setup_ctx(SolverContext&, const P&) {}   // no per-edge / topo

    // DAS Mu reuses the elastic kernels, which address the CPML memory
    // variables through the solver's aux slabs.  DAS keeps those tensors
    // full-domain -- its CUDALayoutSpec sets no pml_slot_axes -- so install
    // identity slabs here.  Left default constructed they are lo=hi=n=0,
    // tot() is 0, the aux row stride collapses and every (iz) row aliases
    // the first: a data race whose output changes from run to run.
    static void init_aux_slabs(SolverContext& solver, Wavefield&)
    {
        SWEEP_CHECK(solver.init_aux_slabs(solver.nz, -1, solver.nx),
                    "DAS Mu 2D: full-grid CPML memory variables rejected by "
                    "init_aux_slabs");
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.bind(p.pml_vals, 2);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, 2, d.B, d.nz, d.nx};   // only Vx and Vz
    }

    static float* field_ptr(WfView& wf, int field_idx)
    {
        return das_mu2d_field_ptr(wf.das, field_idx);
    }

    static WfView view(Wavefield& wf)
    {
        return {wf.view(), wf.elastic_view()};
    }

    // =================================================================== //
    // [2] FORWARD — sg_generic_forward, per it in [it_begin, it_end):
    //   velocity_substep / stress_substep / inject_source /
    //   <checkpoint save> / save_boundary_fields / record_field;
    //   after the loop: save_last_state.
    // =================================================================== //

    // MANDATORY: _c.py Prop.forward always hands the compiled forward its
    // propagation state (persistent _slice_wavefield_buffers in full mode,
    // per-call _transient_forward_wavefields otherwise), sized by
    // cuda_layout.base_nvar + pml_nvar = CKPT_NVAR slots.
    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p,
                                      const Buf& vp)
    {
        SWEEP_CHECK((int)p.wavefields.size() == CKPT_NVAR,
                    "das_mu2d/forward requires the propagator-bound wavefields "
                    "(cuda_layout.base_nvar + cuda_layout.pml_nvar = ", CKPT_NVAR,
                    " tensors), got ", p.wavefields.size());
        wf.bind(p.wavefields, true);
    }

    // (velocity/stress substeps are also replayed by the ckpt/recursive
    // backward, section [5])
    // No-op: this equation writes u_allt from inside its stress kernel.
    // See the call site in sg_driver.cuh for why the hook exists.
    static void capture_allt(Buf&, WfView&, const SolverContext&, int) {}

    static void velocity_substep(const State& s, WfView& wf,
                                 ElasticCPMLPointer cpml_view,
                                 const SolverContext& solver)
    {
        LAUNCH_ELASTIC_VELOCITY(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            wf.el,
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
        LAUNCH_DAS_MU2D_STRESS_STRAIN(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            wf.das,
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
        add_source<<<s.source_config.grid, s.source_config.block>>>(
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
        float* fields[8] = {
            wf.das.vx, wf.das.vz,
            wf.das.sxx, wf.das.szz, wf.das.sxz,
            wf.das.exx, wf.das.ezz, wf.das.exz
        };
        for (int f = 0; f < 8; ++f) {
            rt.save_forward_2d_field(
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
                f == 7
            );
        }
    }

    static void record_field(const State& s, const SolverContext& solver,
                             float* field, Buf& record, int irec,
                             const Buf& receivers_loc, int it, int nrec)
    {
        record_kernel<<<s.record_config.grid, s.record_config.block>>>(
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
        copy_tensor_cuda_async(saver.last_two.select(0, 1).select(0, 0), wf.vz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 2).select(0, 0), wf.sxx_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 3).select(0, 0), wf.szz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 4).select(0, 0), wf.sxz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 5).select(0, 0), wf.exx_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 6).select(0, 0), wf.ezz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 7).select(0, 0), wf.exz_t);
    }

    // =================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — sg_generic_backward, per reverse it:
    //   fix_rho_grad_at_sources / inject_residuals / vel_ptrs_from_u_forward /
    //   it == 0: image_standalone + fix_rho_grad_at_receivers, loop ends;
    //   it  > 0: full_mode_step (imaging + receiver-rho + adjoint step).
    // =================================================================== //

    // Adjoint-step halves shared by plain_adjoint_step (full / ckpt /
    // recursive modes run both back to back) and the BS split
    // (bs_stress_half -> stress_adjoint_half, bs_velocity_half -> velocity_adjoint_half).
private:
    static void stress_adjoint_half(const State& s, const SolverContext& solver,
                                    Wavefield& adjoint, Workspace& workspace,
                                    ElasticCPMLPointer cpml_view)
    {
        auto adj_view = adjoint.view();
        auto elastic_adj_view = adjoint.elastic_view();
        LAUNCH_DAS_MU2D_STRESS_STRAIN_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            cpml_view,
            solver,
            workspace.qxx_t.data_ptr<float>(),
            workspace.qzz_t.data_ptr<float>(),
            workspace.qxz_t.data_ptr<float>(),
            workspace.qzx_t.data_ptr<float>()
        );
        LAUNCH_ELASTIC_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            elastic_adj_view,
            workspace.qxx_t.data_ptr<float>(),
            workspace.qzz_t.data_ptr<float>(),
            workspace.qxz_t.data_ptr<float>(),
            workspace.qzx_t.data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    static void velocity_adjoint_half(const State& s, const SolverContext& solver,
                                      Wavefield& adjoint, Workspace& workspace,
                                      ElasticCPMLPointer cpml_view)
    {
        auto elastic_adj_view = adjoint.elastic_view();
        LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            elastic_adj_view,
            s.models.rho.data_ptr<float>(),
            cpml_view,
            solver,
            workspace.pxx_t.data_ptr<float>(),
            workspace.pzz_t.data_ptr<float>(),
            workspace.pxz_t.data_ptr<float>(),
            workspace.pzx_t.data_ptr<float>()
        );
        LAUNCH_ELASTIC_VELOCITY_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            elastic_adj_view,
            workspace.pxx_t.data_ptr<float>(),
            workspace.pzz_t.data_ptr<float>(),
            workspace.pxz_t.data_ptr<float>(),
            workspace.pzx_t.data_ptr<float>(),
            s.grad_ctx,
            solver
        );
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
                    "das_mu2d/backward requires the propagator-bound "
                    "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar + "
                    "adjoint_extra_nvar = ", ADJ_WF_COUNT, " tensors), got ",
                    p.adjoint_wavefields.size());
        wf.bind(p.adjoint_wavefields, true);
    }

    static void zero_adjoint_if_first_segment(Wavefield&, bool) {}

    // The model-gradient accumulators, MANDATORY in every mode: _c.py
    // Wrapper.backward allocates them per backward call (_gradient_buffers,
    // one per model; cuda_layout.grads_out_has_wavelet is false here) and
    // binds them as grads_out.
    static void bind_grads(const BackwardInputCore& p, std::vector<Buf>& grads)
    {
        SWEEP_CHECK(p.grads_out.size() == 3,
                    "das_mu2d/backward requires the propagator-bound grads_out "
                    "holding exactly {grad_vp, grad_vs, grad_rho} "
                    "(cuda_layout.grads_out_has_wavelet == false), got ",
                    p.grads_out.size());
        grads = {p.grads_out[0], p.grads_out[1], p.grads_out[2]};
    }

    static std::vector<float> adjoint_source_signs(
        const BackwardInputCore& p, IntSpan receiver_fields)
    {
        return elastic_adjoint_source_signs(p.adjoint_source, receiver_fields, 2);
    }

    struct VelPtrs {
        const float* vx_now;
        const float* vz_now;
        const float* vx_next;
        const float* vz_next;
    };

    static VelPtrs vel_ptrs_from_u_forward(const BackwardInputCore& p, int it,
                                             const Buf& zero_velocity)
    {
        VelPtrs v;
        v.vx_now = p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
        v.vz_now = p.u_forward.select(0, it).select(0, 1).data_ptr<float>();
        v.vx_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 0).data_ptr<float>()
                                    : zero_velocity.data_ptr<float>();
        v.vz_next = (it + 1 < p.nt) ? p.u_forward.select(0, it + 1).select(0, 1).data_ptr<float>()
                                    : zero_velocity.data_ptr<float>();
        return v;
    }

    static void fix_rho_grad_at_sources(const State& s, const SolverContext& solver,
                                WfView& adj_view, const BackwardInputCore& p,
                                IntSpan source_fields, int it,
                                std::vector<Buf>& grads)
    {
        if (it < 0 || it >= p.nt) return;
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            const int sfield = source_fields[isrc];
            if (sfield > 1) continue;                       // vx = 0, vz = 1
            float* adj_field = das_mu2d_field_ptr(adj_view.das, sfield);
            if (adj_field == nullptr) continue;
            add_body_force_rho_grad_correction<<<s.source_config.grid, s.source_config.block>>>(
                grads[2].data_ptr<float>(),
                adj_field,
                s.models.rho.data_ptr<float>(),
                p.forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                (int)p.forward_sources_loc.size(1),
                2,
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
            float* field = das_mu2d_field_ptr(adj_view.das, receiver_fields[irec]);
            if (field == nullptr) continue;
            add_source_signed<<<s.record_config.grid, s.record_config.block>>>(
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

    // (image_standalone / fix_rho_grad_at_receivers / plain_adjoint_step are also
    // fired by the ckpt/recursive backward, section [5])
    static void image_standalone(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const VelPtrs& v,
                                 std::vector<Buf>& grads)
    {
        LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            adj_view.el,
            v.vx_now,
            v.vz_now,
            v.vx_next,
            v.vz_next,
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
            const float* fv_now  = (field == 0) ? v.vx_now  : (field == 1) ? v.vz_now  : nullptr;
            const float* fv_next = (field == 0) ? v.vx_next : (field == 1) ? v.vz_next : nullptr;
            if (fv_now == nullptr) continue;              // stress/strain receiver: no rho term
            sub_receiver_rho_grad_correction<<<s.record_config.grid, s.record_config.block>>>(
                grads[2].data_ptr<float>(),
                fv_now,
                fv_next,
                s.models.rho.data_ptr<float>(),
                p.adjoint_source.select(0, irec).data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                2,
                solver.M,                                 // imaging halo (order/2 == M)
                solver
            );
        }
    }

    // No grad fusion: standalone imaging, then the receiver-rho correction,
    // THEN the adjoint step -- the hand-written order.
    static void full_mode_step(const State& s, const SolverContext& solver,
                                Wavefield& adjoint, Workspace& workspace,
                                ElasticCPMLPointer cpml_view,
                                const VelPtrs& v,
                                std::vector<Buf>& grads,
                                const BackwardInputCore& p,
                                IntSpan receiver_fields,
                                int it, int adjoint_nsrc)
    {
        auto adj = view(adjoint);
        image_standalone(s, solver, adj, v, grads);
        fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it,
                          adjoint_nsrc);
        plain_adjoint_step(s, solver, adjoint, workspace, cpml_view);
    }

    static void plain_adjoint_step(const State& s, const SolverContext& solver,
                                   Wavefield& adjoint, Workspace& workspace,
                                   ElasticCPMLPointer cpml_view)
    {
        stress_adjoint_half(s, solver, adjoint, workspace, cpml_view);
        velocity_adjoint_half(s, solver, adjoint, workspace, cpml_view);
    }

    // =================================================================== //
    // [4] BACKWARD_BS — sg_generic_backward_bs, per reverse it:
    //   fix_rho_grad_at_sources / inject_residuals / uninject_forward_source /
    //   bs_stress_half (stress recon NOPML + strip restore + imaging +
    //   receiver-rho + stress-adjoint half) / bs_velocity_half (velocity-adjoint
    //   half + carrier capture + velocity recon NOPML + strip restore +
    //   prefetch); before the loop (first segment): seed_recon.
    // =================================================================== //

    static void zero_adjoint_if_first_segment_bs(Wavefield&, bool) {}

    struct ReconCarriers {
        Buf fvx_prev, fvz_prev;
    };

    // Reconstruction state: Python hands the RECON_WF_COUNT list
    // RECON_LIST_DESC (zeroed, model-shaped) -- elastic2d's list.  The
    // elastic fields bind through bind_elastic(): no strains (the reverse
    // loop runs the elastic NOPML kernels and the elastic bs imaging, which
    // never touch them) and no CPML memory (only the NOPML kernels step the
    // reconstruction); view()/elastic_view() hand those kernels nullptr for
    // every unbound slot.  The carriers are the tail of the same list.
    // MANDATORY: cuda_layout.bs_reconstruction_nvar = RECON_WF_COUNT grids,
    // allocated per backward call by _forward_state_buffers and bound as
    // forward_wavefields.
    static ReconCarriers bind_or_alloc_recon(Wavefield& forward,
                                             const BackwardInputCore& p,
                                             const Buf& vp)
    {
        ReconCarriers c;
        wavefields_required(p.forward_wavefields, RECON_WF_COUNT, vp,
                            "das_mu2d/bs reconstruction list "
                            "(cuda_layout.bs_reconstruction_nvar)");
        forward.bind_elastic(std::vector<Buf>(
            p.forward_wavefields.begin(),
            p.forward_wavefields.begin() + BS_ELASTIC_NVAR));
        c.fvx_prev = p.forward_wavefields[RECON_SLOT_FVX_PREV];
        c.fvz_prev = p.forward_wavefields[RECON_SLOT_FVZ_PREV];
        return c;
    }

    // Seed from the last snapshot: the elastic five always.  The strain rows
    // (8-field saves only -- legacy 5-field saves lack them) are copied only
    // when the reconstruction carries strain members; the Python-bound
    // reconstruction (bind_elastic, the only one left) has none, so that
    // branch no longer fires -- the reverse loop never reads them anyway.
    static void seed_recon(Wavefield& forward, const BackwardInputCore& p)
    {
        copy_tensor_cuda_async(forward.vx_t, p.u_last_two.select(0, 0).select(0, 0));
        copy_tensor_cuda_async(forward.vz_t, p.u_last_two.select(0, 1).select(0, 0));
        copy_tensor_cuda_async(forward.sxx_t, p.u_last_two.select(0, 2).select(0, 0));
        copy_tensor_cuda_async(forward.szz_t, p.u_last_two.select(0, 3).select(0, 0));
        copy_tensor_cuda_async(forward.sxz_t, p.u_last_two.select(0, 4).select(0, 0));
        if (p.u_last_two.size(0) >= 8 && forward.exx_t.defined()) {
            copy_tensor_cuda_async(forward.exx_t, p.u_last_two.select(0, 5).select(0, 0));
            copy_tensor_cuda_async(forward.ezz_t, p.u_last_two.select(0, 6).select(0, 0));
            copy_tensor_cuda_async(forward.exz_t, p.u_last_two.select(0, 7).select(0, 0));
        }
    }

    // A strain-field source (the equation admits them) resolves to nullptr on
    // the bound reconstruction and is skipped; the reverse loop never reads
    // that grid, so the gradients are unaffected.
    static void uninject_forward_source(const State& s, const SolverContext& solver,
                                        WfView& for_view, const BackwardInputCore& p,
                                        IntSpan source_fields,
                                        int it, int forward_nsrc)
    {
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            float* field = das_mu2d_field_ptr(for_view.das, source_fields[isrc]);
            if (field == nullptr) continue;
            add_source_signed<<<s.source_config.grid, s.source_config.block>>>(
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
        LAUNCH_ELASTIC_STRESS_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view.el,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            s.grad_ctx,
            solver
        );

        float* field2[3] = {for_view.das.sxx, for_view.das.szz, for_view.das.sxz};
        for (int f = 2; f < 5; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                field2[f - 2],
                s.launch_config.grid,
                s.launch_config.block,
                bs,
                save_width,
                -solver.M,
                solver,
                f,
                f == 2,
                false
            );
        }

        LAUNCH_CALCULATE_GRAD_ELASTIC_BS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view.el,
            adj_view.el,
            carriers.fvx_prev.data_ptr<float>(),
            carriers.fvz_prev.data_ptr<float>(),
            s.models.vp.data_ptr<float>(),
            s.models.vs.data_ptr<float>(),
            s.models.rho.data_ptr<float>(),
            grads[0].data_ptr<float>(),
            grads[1].data_ptr<float>(),
            grads[2].data_ptr<float>(),
            s.grad_ctx,
            solver
        );

        VelPtrs v{for_view.das.vx, for_view.das.vz,
                  carriers.fvx_prev.data_ptr<float>(),
                  carriers.fvz_prev.data_ptr<float>()};
        fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it, adjoint_nsrc);

        // Stress-adjoint half (the hand-written monolithic runs all four
        // adjoint launches back to back; the p1/p2 split reproduces exactly
        // that sequence when both phases run).
        stress_adjoint_half(s, solver, adjoint, workspace, cpml_view);
    }

    static void bs_velocity_half(const State& s, const SolverContext& solver,
                          WfView& for_view, Wavefield& adjoint,
                          Workspace& workspace, ElasticCPMLPointer cpml_view,
                          BoundaryRuntime& boundary_runtime,
                          const GeneralBoundaryPointer& bs, int save_width,
                          ReconCarriers& carriers, Wavefield& forward,
                          int it, int nt)
    {
        velocity_adjoint_half(s, solver, adjoint, workspace, cpml_view);

        copy_tensor_cuda_async(carriers.fvz_prev, forward.vz_t);
        copy_tensor_cuda_async(carriers.fvx_prev, forward.vx_t);

        LAUNCH_ELASTIC_VELOCITY_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view.el,
            s.models.rho.data_ptr<float>(),
            s.grad_ctx,
            solver,
            nullptr, nullptr              // no carrier capture (plain reconstruction)
        );

        float* field1[2] = {for_view.das.vx, for_view.das.vz};
        for (int f = 0; f < 2; ++f) {
            boundary_runtime.restore_backward_2d_field(
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
                f == 1
            );
        }

        boundary_runtime.prefetch_next_backward_chunk_if_needed(it, nt);
    }

    // =================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING — sg_generic_backward_ckpt, per chunk:
    //   replay: velocity_substep / stress_substep / save_seg_velocities /
    //     inject_forward_sources;
    //   reverse: fix_rho_grad_at_sources / inject_residuals / vel_ptrs_from_seg /
    //     image_standalone / fix_rho_grad_at_receivers / (it > 0) plain_adjoint_step;
    //   after each chunk: export_seg_next_v.
    //   sg_generic_backward_recursive_ckpt, per reverse it: replay with
    //   capture_velocities, then vel_ptrs_from_carriers / image_standalone /
    //   fix_rho_grad_at_receivers / (it > 0) plain_adjoint_step.
    // =================================================================== //

    // Replay state: set 0 of the Python-bound forward_wavefields (zeroed by
    // the propagator per backward call), bound in FULL -- the 8 fields and
    // the 10 CPML memory tensors the replay steps through the PML -- exactly
    // the layout the hand-written driver used to allocate.  Every slot is a
    // full grid here, so the geometry is checked as well.  MANDATORY:
    // _forward_state_shapes("ckpt" / "recursive") derives CKPT_STATE_COUNT
    // slots from the forward slot shapes and Wrapper.backward binds them as
    // forward_wavefields on both checkpoint entries.
    static void bind_or_alloc_recon_ckpt(Wavefield& forward,
                                         const BackwardInputCore& p,
                                         const Buf& vp)
    {
        SWEEP_CHECK((int)p.forward_wavefields.size() >= CKPT_STATE_COUNT,
                    "das_mu2d/ckpt requires the propagator-bound replay state "
                    "(cuda_layout.base_nvar + pml_nvar = ", CKPT_STATE_COUNT,
                    " tensors per set), got ", p.forward_wavefields.size());
        auto state = wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT,
                                   "das_mu2d ckpt replay state");
        for (int i = 0; i < CKPT_STATE_COUNT; ++i)
            pool_slot_checked(state, i, vp, "das_mu2d ckpt replay state");
        forward.bind(state, true);
    }

    static void check_ckpt_aux_layout(const Wavefield&, const Wavefield&) {}

    // ---- seg / carrier plumbing (ckpt + recursive modes) ----

    // Per-segment velocity histories seg[c][k] = v_c(start + k), k in
    // [0, segment_len]: the N_VEL checkpoint_replay slots [vx, vz]
    // (cuda_layout.checkpoint_replay_shapes, (chunk + 1, B, 1, nz, nx) each,
    // allocated once next to the snapshots and never re-zeroed -- every row
    // the reverse pass reads was written by save_seg_velocities earlier in
    // the same segment).  MANDATORY in the chunked mode:
    // cuda_layout.checkpoint_replay_shapes declares N_VEL histories for mode
    // "ckpt" and _ensure_checkpoint_buffers allocates them next to the
    // snapshots.  Taken once per call at the longest segment; the skeleton
    // narrows the rows of a shorter last segment itself.
    static std::vector<Buf> seg_buffers(const BackwardInputCore& p,
                                                  const Buf& vp, int max_rows)
    {
        SWEEP_CHECK((int)p.checkpoint_replay.size() >= N_VEL,
                    "das_mu2d/ckpt requires the propagator-bound checkpoint_replay "
                    "(cuda_layout.checkpoint_replay_shapes, ", N_VEL,
                    " velocity histories), got ", p.checkpoint_replay.size());
        std::vector<int64_t> shape = vp.sizes().vec();   // (max_rows, B, 1, nz, nx)
        shape.insert(shape.begin(), static_cast<int64_t>(max_rows));
        std::vector<Buf> seg;
        for (int c = 0; c < N_VEL; ++c)
            seg.push_back(pool_required(p.checkpoint_replay, c, shape,
                                        "checkpoint_replay"));
        return seg;
    }

    static void save_seg_velocities(std::vector<Buf>& seg, Wavefield& forward,
                            int slot)
    {
        copy_tensor_cuda_async(seg[0].select(0, slot), forward.vx_t);
        copy_tensor_cuda_async(seg[1].select(0, slot), forward.vz_t);
    }

    static void inject_forward_sources(const State& s, const SolverContext& solver,
                                      WfView& for_view, const BackwardInputCore& p,
                                      IntSpan source_fields, int it)
    {
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            float* field = das_mu2d_field_ptr(for_view.das, source_fields[isrc]);
            if (field == nullptr) continue;
            add_source<<<s.source_config.grid, s.source_config.block>>>(
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
        VelPtrs v;
        v.vx_now = seg[0].select(0, now_offset).data_ptr<float>();
        v.vz_now = seg[1].select(0, now_offset).data_ptr<float>();
        v.vx_next = (next_offset >= 0) ? seg[0].select(0, next_offset).data_ptr<float>()
                                       : next_segment_v[0].data_ptr<float>();
        v.vz_next = (next_offset >= 0) ? seg[1].select(0, next_offset).data_ptr<float>()
                                       : next_segment_v[1].data_ptr<float>();
        return v;
    }

    static void export_seg_next_v(std::vector<Buf>& prev,
                                   const std::vector<Buf>& seg)
    {
        copy_tensor_cuda_async(prev[0], seg[0].select(0, 1));
        copy_tensor_cuda_async(prev[1], seg[1].select(0, 1));
    }

    static void capture_velocities(std::vector<Buf>& v, Wavefield& forward)
    {
        copy_tensor_cuda_async(v[0], forward.vx_t);
        copy_tensor_cuda_async(v[1], forward.vz_t);
    }

    static VelPtrs vel_ptrs_from_carriers(const std::vector<Buf>& current_v,
                                    const std::vector<Buf>& next_v)
    {
        VelPtrs v;
        v.vx_now = current_v[0].data_ptr<float>();
        v.vz_now = current_v[1].data_ptr<float>();
        v.vx_next = next_v[0].data_ptr<float>();
        v.vz_next = next_v[1].data_ptr<float>();
        return v;
    }
};

} // namespace das_mu2d
