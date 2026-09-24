// Driver traits for the 2-D elastic equation: the per-equation half of the
// staggered-family skeleton in ``common/sg_driver.cuh``.  Line-faithful
// transcription of the hand-written drivers; physics kernels untouched.
// The APM (Cao & Chen 2018) entry points stay hand-written in
// forward.cu/backward.cu — they refuse stepping/phasing and carry their own
// kernel variants, so there is nothing for the skeleton to share yet.
//
// REFERENCE EQUATION of the staggered family.  The other staggered-family
// members (elastic3d, das_mu2d, das_mu3d, elastic_tti_sg2d, elastic_tti_sg3d,
// elastic_vr2d) describe their deltas against this baseline; the properties
// below are what "same as elastic2d" means:
//   * constants: NDIM = 2, CKPT_NVAR = 15 (CKPT_COUNT_MSG / CKPT_RECURSIVE_COUNT_MSG), BS_NVAR = 5 (vx, vz, sxx, szz, sxz), CUT_MASK_BITS = 0xF (x_lo, x_hi, z_lo, z_hi);
//   * ADJ_WF_COUNT = 15, RECON_WF_COUNT = 7 (RECON_LIST_DESC = [vx, vz, sxx, szz, sxz, fvx_prev, fvz_prev]), N_VEL = 2, IMAGING_USES_NEXT_V = true (imaging consumes the v(t+1) carriers);
//   * models: parse_models reads {vp, vs, rho} and derives mu = rho*vs^2 and lambda = rho*(vp^2 - 2*vs^2) as torch tensors kept alive in Models (kernels hold raw pointers into them);
//   * State = Models + one SGradParam grad_ctx + launch/source/record configs + order + nx/nz/B; Workspace = ElasticAdjointWorkspaceTensor via bind_adjoint_workspace_required(p.adjoint_workspace, 2);
//   * validate_forward / validate_backward and zero_adjoint_if_first_segment / zero_adjoint_if_first_segment_bs are no-ops (the adjoint buffers arrive Python-zeroed);
//   * setup_ctx installs the per-edge free surface (set_per_edge(fs_faces, pad_lo, pad_hi)) and topo_rows / has_topo when p.has_topo;
//   * init_aux_slabs = elastic_init_aux_slabs (CPML aux slabs); alloc_cpml = cpml.bind(pml_vals, 2);
//   * allt_shape = (nt, 2, B, nz, nx): u_allt stores the velocities only (Vx, Vz);
//   * field_ptr = elastic_field_ptr(wf, 2, idx); view = wf.view();
//   * forward: bind(p.wavefields, true) -- MANDATORY, no self-allocating fallback; velocity_substep = LAUNCH_ELASTIC_VELOCITY (rho), stress_substep = LAUNCH_ELASTIC_STRESS (lambda, mu, u_this_t); both also replay in the ckpt/recursive modes;
//   * inject_source / record_field are plain add_source / record_kernel on the field selected by index; record_field writes record.select(0, irec);
//   * save_boundary_fields = save_forward_2d_field for the five fields at offset -M (field index f, flag f == 4); save_last_state copies the five into last_two[0..4];
//   * backward outputs: bind_grads takes exactly {grad_vp, grad_vs, grad_rho} from grads_out (accumulated "+=", never zeroed here), MANDATORY; adjoint_source_signs = elastic_adjoint_source_signs(adjoint_source, receiver_fields, 2);
//   * bind_or_alloc_adjoint binds p.adjoint_wavefields (MANDATORY);
//   * VelPtrs = {vx_now, vz_now, vx_next, vz_next}; vel_ptrs_from_u_forward reads u_forward.select(0, it) and u_forward.select(0, it + 1) (zero_velocity -- the skeleton's read-only adjoint_workspace slot SgCarrierSlots::FULL_ZERO -- when it + 1 == nt);
//   * fix_rho_grad_at_sources (velocity sources, field <= 1: add_body_force_rho_grad_correction) fires BEFORE inject_residuals; inject_residuals = add_source_signed of the residual per receiver field with that field's sign;
//   * fix_rho_grad_at_receivers = sub_receiver_rho_grad_correction at velocity receivers (stress receivers skipped) with imaging halo M;
//   * image_standalone = LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS (it == 0 in full mode; every reverse it in the ckpt/recursive modes);
//   * adjoint launch helpers: stress_adjoint_prepare (10 explicit imaging pointers) / stress_adjoint_apply / velocity_adjoint_prepare / velocity_adjoint_apply, with velocity_adjoint_half = the last two;
//   * full_mode_step: the vp/vs/rho imaging is fused into the STRESS_ADJOINT_PREPARE launch, then STRESS_ADJOINT_APPLY, VELOCITY_ADJOINT_PREPARE, VELOCITY_ADJOINT_APPLY, and the receiver-rho fix (fix_rho_grad_at_receivers) after the four launches;
//   * plain_adjoint_step = the same four launches with all-null imaging pointers (ckpt/recursive reverse sweeps);
//   * bs: ReconCarriers {fvx_prev, fvz_prev}; bind_or_alloc_recon takes the 7-tensor list (five fields bound with use_pml = false + two carriers), MANDATORY;
//   * seed_recon copies the five fields from u_last_two[0..4]; uninject_forward_source subtracts the forward source at the source fields (add_source_signed, sign -1);
//   * bs_stress_half order: STRESS_NOPML -> restore sxx/szz/sxz (fields 2..4, offset -M) -> STRESS_ADJOINT_PREPARE with the imaging fused (v(it) = for_view.v*, v(it+1) = carriers) -> fix_rho_grad_at_receivers -> STRESS_ADJOINT_APPLY;
//   * bs_velocity_half order: VELOCITY_ADJOINT_PREPARE -> VELOCITY_ADJOINT_APPLY -> elastic_capture_strips_2d (restore strips into the carriers) -> VELOCITY_NOPML (carrier write of every computed cell, then the velocity update) -> restore vx/vz (fields 0..1, offset -M) -> prefetch_next_backward_chunk_if_needed;
//   * ckpt: bind_or_alloc_recon_ckpt binds forward_wavefields set 0 (CKPT_STATE_COUNT = 15, the full base+pml bind), MANDATORY; check_ckpt_aux_layout compares the m_vxx aux shapes; WS_CARRIERS = 8 (the skeleton's velocity carriers follow the q*/p* scratch in adjoint_workspace, sg_driver.cuh SgCarrierSlots);
//   * seg buffers: seg_buffers = the two checkpoint_replay slots (vx / vz histories of shape (checkpoint_interval + 1, vp.size(0) * vp.size(1), 1, nz, nx), narrowed per chunk by the skeleton); save_seg_velocities fills a row, export_seg_next_v keeps row 1, vel_ptrs_from_seg picks now / next (next_segment_v past the end);
//   * recursive: inject_forward_sources replays the forward source per source field; capture_velocities / vel_ptrs_from_carriers feed the imaging from the captured v(it) / v(it+1) pair.
//
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/sg_driver.cuh.
#pragma once

#include <torch/extension.h>
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
#include "../../common/wavetypes.h"
#include "../../common/eq_driver.cuh"
#include "../../common/sg_driver.cuh"
#include "../../launch/config.h"

namespace elastic2d {

struct Driver {
    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // Constants, type aliases, and the prologue hooks shared by every entry
    // point.  Prologue call order per the timing map: validate_backward
    // (backward only), parse_models, setup_ctx, bind_or_alloc_* wavefields,
    // init_aux_slabs, alloc_cpml, bind_grads, make_workspace,
    // make_state, adjoint_source_signs.  The bind_or_alloc_* / bind_grads /
    // adjoint_source_signs hooks live in their entry-point sections below.
    // ===================================================================== //
    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "elastic2d";
    static constexpr int CKPT_NVAR = 15;
    static constexpr const char* CKPT_COUNT_MSG =
        "Elastic 2D checkpointing expects 15 checkpoint tensors";
    static constexpr int BS_NVAR = 5;   // vx, vz, sxx, szz, sxz

    static constexpr int CUT_MASK_BITS = 0xF;
    static constexpr const char* CUT_MASK_DESC = "bits 0..3 (x_lo, x_hi, z_lo, z_hi)";
    static constexpr int ADJ_WF_COUNT = 15;
    static constexpr int RECON_WF_COUNT = 7;
    static constexpr const char* RECON_LIST_DESC =
        "[vx, vz, sxx, szz, sxz, fvx_prev, fvz_prev]";
    static constexpr const char* CKPT_RECURSIVE_COUNT_MSG =
        "Elastic 2D recursive checkpointing expects 15 checkpoint tensors";
    static constexpr int N_VEL = 2;
    static constexpr bool IMAGING_USES_NEXT_V = true;   // imaging consumes v(t+1) carriers
    // Checkpoint modes: the replay state is one full base+pml struct
    // (forward_wavefields set 0, cuda_layout.checkpoint_state_nvar); the
    // adjoint_workspace pool holds the 8 q*/p* scratch slots first and the
    // skeleton's velocity carriers after them (sg_driver.cuh SgCarrierSlots).
    static constexpr int CKPT_STATE_COUNT = CKPT_NVAR;   // 15
    static constexpr int WS_CARRIERS = 8;                // qxx, qzz, qxz, qzx, pxx, pzz, pxz, pzx

    using Wavefield = ElasticWavefieldTensor;
    using WfView = ElasticWavefieldPointer;
    using CPML = ElasticCPMLTensor;

    // lambda/mu are the propagator's derived_models slots, filled once per
    // call by common/derived_models.h; the struct keeps them alive for the
    // whole call (kernels hold raw pointers into them).
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
        const auto lame = derived::lame(p, m.vp, m.vs, m.rho, "elastic2d::parse_models");
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

    using Workspace = ElasticAdjointWorkspaceTensor;

    static Workspace make_workspace(const BackwardInputCore& p, const Buf& vp)
    {
        Workspace workspace;
        bind_adjoint_workspace_required(workspace, p.adjoint_workspace, 2,
                                        "elastic2d backward");
        return workspace;
    }

    static void validate_forward(const ForwardInputCore&) {}
    static void validate_backward(const BackwardInputCore&, const char*) {}

    template <class P>
    static void setup_ctx(SolverContext& solver, const P& p)
    {
        solver.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
        if (p.has_topo) {
            solver.topo_rows = p.topo_rows.template data_ptr<int>();
            solver.has_topo = true;
        }
    }

    static void init_aux_slabs(SolverContext& solver, Wavefield& wf)
    {
        elastic_init_aux_slabs(solver, wf);
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
        return elastic_field_ptr(wf, 2, field_idx);
    }

    static WfView view(Wavefield& wf) { return wf.view(); }

    // ===================================================================== //
    // [2] FORWARD — sg_generic_forward, per it in [it_begin, it_end):
    //   velocity_substep -> stress_substep -> inject_source ->
    //   <checkpoint save (shared runtime)> -> save_boundary_fields ->
    //   record_field; after the loop: save_last_state.
    // ===================================================================== //
    // MANDATORY: _c.py Prop.forward always hands the compiled forward its
    // propagation state (persistent _slice_wavefield_buffers in full mode,
    // per-call _transient_forward_wavefields otherwise), sized by
    // cuda_layout.base_nvar + pml_nvar = CKPT_NVAR slots.
    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p,
                                      const Buf& vp)
    {
        SWEEP_CHECK((int)p.wavefields.size() == CKPT_NVAR,
                    "elastic2d/forward requires the propagator-bound wavefields "
                    "(cuda_layout.base_nvar + cuda_layout.pml_nvar = ", CKPT_NVAR,
                    " tensors), got ", p.wavefields.size());
        wf.bind(p.wavefields, true);
    }

    // (velocity_substep / stress_substep also replay in the ckpt/recursive
    // backward modes.)
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
        LAUNCH_ELASTIC_STRESS(
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
        float* fields[5] = {wf.vx, wf.vz, wf.sxx, wf.szz, wf.sxz};
        for (int f = 0; f < 5; ++f) {
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
                f == 4
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
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — sg_generic_backward (full storage),
    // per reverse it:
    //   fix_rho_grad_at_sources -> inject_residuals -> vel_ptrs_from_u_forward ->
    //   it == 0: image_standalone + fix_rho_grad_at_receivers, loop ends;
    //   it  > 0: full_mode_step (imaging + receiver-rho + adjoint step).
    // ===================================================================== //
    // MANDATORY, every backward mode: _ensure_wavefield_buffers allocates the
    // adjoint set whenever a gradient is asked for and Wrapper.backward binds
    // it (zeroed) as adjoint_wavefields.
    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInputCore& p,
                                      const Buf& vp)
    {
        SWEEP_CHECK((int)p.adjoint_wavefields.size() == ADJ_WF_COUNT,
                    "elastic2d/backward requires the propagator-bound "
                    "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar + "
                    "adjoint_extra_nvar = ", ADJ_WF_COUNT, " tensors), got ",
                    p.adjoint_wavefields.size());
        wf.bind(p.adjoint_wavefields, true);
    }

    static void zero_adjoint_if_first_segment(Wavefield&, bool) {}

    // The model-gradient accumulators, MANDATORY in every mode: _c.py
    // Wrapper.backward allocates them per backward call (_gradient_buffers,
    // one per model; cuda_layout.grads_out_has_wavelet is false here) and
    // binds them as grads_out.  They are accumulated "+=" and NOT zeroed
    // here — Python zeroes them once before the first segment.
    static void bind_grads(const BackwardInputCore& p, std::vector<Buf>& grads)
    {
        SWEEP_CHECK(p.grads_out.size() == 3,
                    "elastic2d/backward requires the propagator-bound grads_out "
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

    // (VelPtrs is also produced by the bs / seg / carrier selectors below.)
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

    // Body-force (velocity) sources: the rho imaging correlates the adjoint
    // velocity with v(it) - v(it+1), which at a source cell still contains the
    // raw injected amplitude; the true derivative has no such term.  Must run
    // BEFORE this step's receiver residuals are injected (see the hand-written
    // driver's comment about source+receiver-on-one-cell overshoot).
    static void fix_rho_grad_at_sources(const State& s, const SolverContext& solver,
                                WfView& adj_view, const BackwardInputCore& p,
                                IntSpan source_fields, int it,
                                std::vector<Buf>& grads)
    {
        if (it < 0 || it >= p.nt) return;
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            const int sfield = source_fields[isrc];
            if (sfield > 1) continue;                       // vx = 0, vz = 1
            float* adj_field = elastic_field_ptr(adj_view, 2, sfield);
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
            float* field = elastic_field_ptr(adj_view, 2, receiver_fields[irec]);
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

    static void image_standalone(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const VelPtrs& v,
                                 std::vector<Buf>& grads)
    {
        LAUNCH_CALCULATE_GRAD_ELASTIC_NOBS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            adj_view,
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

    // Undo the just-injected receiver residual from this reverse step's rho
    // imaging, at every velocity-receiver cell.  Stress receivers have no rho
    // term to correct.
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
            if (fv_now == nullptr) continue;              // stress receiver: no rho term
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

    // Adjoint-step launch helpers shared by full_mode_step,
    // plain_adjoint_step (full / ckpt / recursive modes) and bs_stress_half /
    // bs_velocity_half (bs mode).  Bodies are the launch statements verbatim; the
    // stress prepare takes the 10 imaging pointers explicitly (all-null for
    // plain_adjoint_step).
    static void stress_adjoint_prepare(const State& s, const SolverContext& solver,
                                       WfView& adj_view, Workspace& workspace,
                                       ElasticCPMLPointer cpml_view,
                                       const float* vx_now, const float* vz_now,
                                       const float* vx_next, const float* vz_next,
                                       const float* vp, const float* vs,
                                       const float* rho,
                                       float* g_vp, float* g_vs, float* g_rho)
    {
        LAUNCH_ELASTIC_STRESS_ADJOINT_PREPARE(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            adj_view,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            cpml_view,
            solver,
            workspace.qxx_t.data_ptr<float>(),
            workspace.qzz_t.data_ptr<float>(),
            workspace.qxz_t.data_ptr<float>(),
            workspace.qzx_t.data_ptr<float>(),
            s.grad_ctx,
            vx_now, vz_now, vx_next, vz_next,
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
        LAUNCH_ELASTIC_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace.qxx_t.data_ptr<float>(),
            workspace.qzz_t.data_ptr<float>(),
            workspace.qxz_t.data_ptr<float>(),
            workspace.qzx_t.data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    static void velocity_adjoint_prepare(const State& s, const SolverContext& solver,
                                         WfView& adj_view, Workspace& workspace,
                                         ElasticCPMLPointer cpml_view)
    {
        LAUNCH_ELASTIC_VELOCITY_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.models.rho.data_ptr<float>(),
            cpml_view,
            solver,
            workspace.pxx_t.data_ptr<float>(),
            workspace.pzz_t.data_ptr<float>(),
            workspace.pxz_t.data_ptr<float>(),
            workspace.pzx_t.data_ptr<float>()
        );
    }

    static void velocity_adjoint_apply(const State& s, const SolverContext& solver,
                                       WfView& adj_view, Workspace& workspace)
    {
        LAUNCH_ELASTIC_VELOCITY_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace.pxx_t.data_ptr<float>(),
            workspace.pzz_t.data_ptr<float>(),
            workspace.pxz_t.data_ptr<float>(),
            workspace.pzx_t.data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    static void velocity_adjoint_half(const State& s, const SolverContext& solver,
                                      WfView& adj_view, Workspace& workspace,
                                      ElasticCPMLPointer cpml_view)
    {
        velocity_adjoint_prepare(s, solver, adj_view, workspace, cpml_view);
        velocity_adjoint_apply(s, solver, adj_view, workspace);
    }

    // FULL-mode step: fold this reverse step's vp/vs/rho-gradient imaging
    // into the stress-adjoint-prepare kernel (it reads the un-mutated
    // post-source adjoint at entry, exactly what calculate_grad_elastic_nobs
    // would correlate), then run the remaining three adjoint launches.
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
                               v.vx_now, v.vz_now, v.vx_next, v.vz_next,
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

    // Null imaging pointers => behaviour byte-for-byte identical to the
    // hand-written apply_adjoint_step_2d without fusion arguments.
    // (Fires in the ckpt/recursive modes' reverse sweeps, not in FULL mode.)
    static void plain_adjoint_step(const State& s, const SolverContext& solver,
                                   Wavefield& adjoint, Workspace& workspace,
                                   ElasticCPMLPointer cpml_view)
    {
        auto adj_view = adjoint.view();
        stress_adjoint_prepare(s, solver, adj_view, workspace, cpml_view,
                               nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr);
        stress_adjoint_apply(s, solver, adj_view, workspace);
        velocity_adjoint_half(s, solver, adj_view, workspace, cpml_view);
    }

    // ===================================================================== //
    // [4] BACKWARD_BS — sg_generic_backward_bs, per reverse it (floor
    // max(it_lo, 1)):
    //   fix_rho_grad_at_sources / inject_residuals / uninject_forward_source
    //   [inject_step] -> bs_stress_half -> bs_velocity_half;
    //   before the loop (first segment): seed_recon from u_last_two.
    // ===================================================================== //
    static void zero_adjoint_if_first_segment_bs(Wavefield&, bool) {}

    struct ReconCarriers {
        Buf fvx_prev, fvz_prev;
    };

    static ReconCarriers bind_or_alloc_recon(Wavefield& forward,
                                             const BackwardInputCore& p,
                                             const Buf& vp)
    {
        // MANDATORY: the bs backward's reconstruction state is
        // cuda_layout.reconstruction_nvar (slot_table ELASTIC2D.recon = 7)
        // grids, allocated per backward call by _forward_state_buffers and
        // bound as forward_wavefields.
        ReconCarriers c;
        wavefields_required(p.forward_wavefields, RECON_WF_COUNT, vp,
                            "elastic2d/bs reconstruction list "
                            "[vx, vz, sxx, szz, sxz, fvx_prev, fvz_prev] "
                            "(cuda_layout.reconstruction_nvar)");
        forward.bind(std::vector<Buf>(p.forward_wavefields.begin(),
                                                p.forward_wavefields.begin() + 5),
                     /*use_pml=*/false);
        c.fvx_prev = p.forward_wavefields[5];
        c.fvz_prev = p.forward_wavefields[6];
        return c;
    }

    static void seed_recon(Wavefield& forward, const BackwardInputCore& p)
    {
        copy_tensor_cuda_async(forward.vx_t, p.u_last_two.select(0, 0).select(0, 0));
        copy_tensor_cuda_async(forward.vz_t, p.u_last_two.select(0, 1).select(0, 0));
        copy_tensor_cuda_async(forward.sxx_t, p.u_last_two.select(0, 2).select(0, 0));
        copy_tensor_cuda_async(forward.szz_t, p.u_last_two.select(0, 3).select(0, 0));
        copy_tensor_cuda_async(forward.sxz_t, p.u_last_two.select(0, 4).select(0, 0));
    }

    static void uninject_forward_source(const State& s, const SolverContext& solver,
                                        WfView& for_view, const BackwardInputCore& p,
                                        IntSpan source_fields,
                                        int it, int forward_nsrc)
    {
        for (int isrc = 0; isrc < source_fields.size(); ++isrc) {
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc]);
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

    // bs phase 1: stress reconstruction + restore + gradient imaging + the
    // stress-adjoint half.
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
            for_view,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            s.grad_ctx,
            solver
        );

        float* field2[3] = {for_view.sxx, for_view.szz, for_view.sxz};
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

        // Imaging folded into the stress-adjoint-prepare kernel (as in FULL
        // mode): it carries the calculate_grad_elastic_bs block verbatim and
        // already streams the adjoint stresses, so the standalone imaging
        // pass (18 field passes) is gone.  for_view.v* = v(it) and the
        // carriers = v(it+1) are final here, and prepare touches nothing the
        // imaging reads.
        stress_adjoint_prepare(s, solver, adj_view, workspace, cpml_view,
                               for_view.vx, for_view.vz,
                               carriers.fvx_prev.data_ptr<float>(),
                               carriers.fvz_prev.data_ptr<float>(),
                               s.models.vp.data_ptr<float>(),
                               s.models.vs.data_ptr<float>(),
                               s.models.rho.data_ptr<float>(),
                               grads[0].data_ptr<float>(),
                               grads[1].data_ptr<float>(),
                               grads[2].data_ptr<float>());

        // Receiver-cell rho correction AFTER the imaging '+=' (same per-cell
        // accumulation order as the standalone pass); operands unchanged.
        // Must stay BETWEEN the stress prepare and the stress apply.
        VelPtrs v{for_view.vx, for_view.vz,
                  carriers.fvx_prev.data_ptr<float>(),
                  carriers.fvz_prev.data_ptr<float>()};
        fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it, adjoint_nsrc);
        stress_adjoint_apply(s, solver, adj_view, workspace);
    }

    // bs phase 2: velocity-adjoint half, carrier capture, velocity
    // reconstruction + restore, prefetch.
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

        // Carrier capture v(it+1): the NOPML kernel stores every cell it
        // computes into the carriers before updating it; the restore strips
        // (the only other box cells the imaging reads) are copied here
        // first.  Replaces two full-field memcpys.
        {
            const int wxl = solver.cut_x_lo() ? 0 : save_width;
            const int wxh = solver.cut_x_hi() ? 0 : save_width;
            const int wzl = solver.cut_z_lo() ? 0 : save_width;
            const int wzh = solver.cut_z_hi() ? 0 : save_width;
            const int bw = solver.phys_x1() - solver.phys_x0();
            const int bh = solver.phys_z1() - solver.phys_z0() - wzl - wzh;
            const int n_strip = (wzl + wzh) * bw + (wxl + wxh) * (bh > 0 ? bh : 0);
            if (n_strip > 0) {
                dim3 strip_grid((n_strip + 255) / 256, solver.B);
                elastic_capture_strips_2d<<<strip_grid, 256>>>(
                    for_view.vx, for_view.vz,
                    carriers.fvx_prev.data_ptr<float>(),
                    carriers.fvz_prev.data_ptr<float>(),
                    solver.nx, solver.nz,
                    solver.phys_x0(), solver.phys_x1(), solver.phys_z0(), solver.phys_z1(),
                    wxl, wxh, wzl, wzh
                );
            }
        }

        LAUNCH_ELASTIC_VELOCITY_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            s.models.rho.data_ptr<float>(),
            s.grad_ctx,
            solver,
            carriers.fvx_prev.data_ptr<float>(),
            carriers.fvz_prev.data_ptr<float>()
        );

        float* field1[2] = {for_view.vx, for_view.vz};
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

    // ===================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING
    // sg_generic_backward_ckpt — per chunk (sg_backward_segment):
    //   replay: velocity_substep / stress_substep / save_seg_velocities /
    //           inject_forward_sources;
    //   reverse: fix_rho_grad_at_sources / inject_residuals / vel_ptrs_from_seg /
    //            image_standalone / fix_rho_grad_at_receivers /
    //            (it > 0) plain_adjoint_step;
    //   after each chunk: export_seg_next_v.
    // sg_generic_backward_recursive_ckpt — per reverse it:
    //   fix_rho_grad_at_sources / inject_residuals ->
    //   sg_replay_forward_to_time (substeps + capture_velocities) ->
    //   vel_ptrs_from_carriers / image_standalone / fix_rho_grad_at_receivers /
    //   (it > 0) plain_adjoint_step.
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
                    "elastic2d/ckpt requires the propagator-bound replay state "
                    "(cuda_layout.slots, the forward slot list: ", CKPT_STATE_COUNT,
                    " tensors per set), got ", p.forward_wavefields.size());
        forward.bind(wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT,
                                   "elastic2d ckpt replay state"), true);
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
    // [0, N_VEL) = vx, vz: (max_rows, B, 1, nz, nx) each, bound at the longest
    // chunk (cuda_layout.checkpoint_replay_shapes) and narrowed per chunk by
    // the skeleton; row 0 = v(start), row k = v(start + k).  Never re-zeroed:
    // every row a chunk reads it wrote first.  MANDATORY in the
    // chunked mode: cuda_layout.checkpoint_replay_shapes declares N_VEL
    // histories for mode "ckpt" and _ensure_checkpoint_buffers allocates them
    // next to the snapshots.
    static std::vector<Buf> seg_buffers(const BackwardInputCore& p,
                                                  const Buf& vp, int max_rows)
    {
        SWEEP_CHECK((int)p.checkpoint_replay.size() >= N_VEL,
                    "elastic2d/ckpt requires the propagator-bound checkpoint_replay "
                    "(cuda_layout.checkpoint_replay_shapes, ", N_VEL,
                    " velocity histories), got ", p.checkpoint_replay.size());
        std::vector<Buf> seg;
        seg.reserve(N_VEL);
        for (int c = 0; c < N_VEL; ++c)
            seg.push_back(pool_required(p.checkpoint_replay, c,
                                        {max_rows, vp.size(0) * vp.size(1), 1,
                                         vp.size(2), vp.size(3)}, "checkpoint_replay"));
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
            float* field = elastic_field_ptr(for_view, 2, source_fields[isrc]);
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

} // namespace elastic2d
