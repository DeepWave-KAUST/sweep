// Driver traits for the 2-D staggered-grid elastic TTI equation: the
// per-equation half of the staggered-family skeleton in
// ``common/sg_driver.cuh``.  Line-faithful transcription of the hand-written
// drivers; physics kernels untouched.  Deltas vs the elastic/das_mu family
// members worth naming:
//   * the model set is rho + 15 stiffness tensors (16 grads); the kernels
//     take a StiffnessPointer, rebuilt on demand from p.models / grads;
//   * three velocity components on a 2-D grid (TTI couples vy), so N_VEL = 3
//     and the adjoint-source signs use the 3-D field layout;
//   * the adjoint workspace is six plain scratch tensors, taken from
//     p.adjoint_workspace, which the propagator always binds (it declares
//     them through cuda_layout.backward_workspace_shapes); in ckpt mode the
//     pool continues with the velocity carriers the skeleton takes behind
//     them (WS_CARRIERS);
//   * ckpt: the replay state is set 0 of the Python-bound forward_wavefields
//     (CKPT_STATE_COUNT = 20, the full bind with CPML memory) and the
//     per-segment vx/vy/vz histories are the three checkpoint_replay slots
//     (seg_buffers);
//   * u_allt stores all 8 physical fields, not just the velocities;
//   * bs: ReconCarriers {fvx_next, fvy_next, fvz_next}; bind_or_alloc_recon
//     takes the 11-tensor p.forward_wavefields list (the 8 physical fields
//     bound through WavefieldTensor::bind_physical -- CPML memory left
//     undefined, the NOPML reverse kernels never touch it -- plus the three
//     carriers); the list is MANDATORY;
//   * per-mode entry validation keeps the hand-written message texts
//     (validate_backward hook);
//   * no recursive checkpointing: the forward refuses it and backward.cu
//     instantiates no recursive driver, so the recursive-only hooks
//     (capture_velocities, vel_ptrs_from_carriers, CKPT_RECURSIVE_COUNT_MSG) are
//     deliberately absent;
//   * no DD cut support, no aux slabs (the CPML memory lives in the
//     equation's own wavefield tensors).
//
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/sg_driver.cuh.
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>

#include "kernels.cuh"
#include "tensors.h"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/elastic.h"   // elastic_adjoint_source_signs
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../common/eq_driver.cuh"
#include "../../common/sg_driver.cuh"
#include "../../launch/config.h"

namespace elastic_tti_sg2d {

struct Driver {

    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    //     Prologue of every entry (skeleton call order): validate_backward
    //     (backward only), parse_models, setup_ctx, bind_or_alloc_* wavefields,
    //     init_aux_slabs, alloc_cpml, bind_grads, make_workspace,
    //     make_state, adjoint_source_signs.
    // ===================================================================== //

    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "elastic_tti_sg2d";
    static constexpr int CKPT_NVAR = 20;
    static constexpr const char* CKPT_COUNT_MSG =
        "ElasticTTISG checkpointing expects 20 checkpoint tensors";
    // Checkpoint replay state: one set of the Python-bound forward_wavefields,
    // in the forward's full bind order -- 8 fields + 12 CPML memory tensors,
    // all full-grid (no aux slabs in this equation).
    static constexpr int CKPT_STATE_COUNT = CKPT_NVAR;
    static constexpr int BS_NVAR = 8;   // vx, vy, vz, sxx, szz, syz, sxz, sxy
    static constexpr int CUT_MASK_BITS = 0x0;
    static constexpr const char* CUT_MASK_DESC =
        "no bits (ElasticTTISG kernels are not cut-aware)";
    static constexpr int ADJ_WF_COUNT = 20;
    static constexpr int RECON_WF_COUNT = 11;   // == cuda_layout.bs_reconstruction_nvar
    static constexpr const char* RECON_LIST_DESC =
        "[vx, vy, vz, sxx, szz, syz, sxz, sxy, fvx_next, fvy_next, fvz_next]";
    static constexpr int N_VEL = 3;
    static constexpr bool IMAGING_USES_NEXT_V = true;   // imaging consumes v(t+1) carriers
    static_assert(RECON_WF_COUNT == BS_NVAR + N_VEL,
                  "bs reconstruction list = the physical fields + one v(t+1) carrier per velocity");

    using Wavefield = WavefieldTensor;
    using WfView = WavefieldPointer;
    using CPML = ElasticCPMLTensor;

    struct Models {
        torch::Tensor rho;
        std::vector<torch::Tensor> all;   // rho + 15 stiffness tensors
    };

    template <class P>
    static Models parse_models(const P& p)
    {
        Models m;
        m.rho = p.models[0];
        m.all = p.models;
        return m;
    }

    struct State {
        Models models;
        StiffnessPointer model;
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
        s.model = stiffness_view(p.models);
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

    // Six scratch tensors, taken from the Python-side pool
    // (cuda_layout.backward_workspace_shapes, 6 per shot).  MANDATORY: the
    // layout declares that callable, so _ensure_adjoint_workspace_buffers
    // sizes the pool for this backward's memory mode and Wrapper.backward
    // binds it as adjoint_workspace on every gradient-bearing call.
    // Slot names follow the stress-adjoint kernels' parameter order.
    enum WorkspaceSlot : int { Q_VXX = 0, Q_VYX, Q_VZX, Q_VXZ, Q_VYZ, Q_VZZ, N_WORKSPACE };
    using Workspace = std::array<torch::Tensor, N_WORKSPACE>;
    // In ckpt mode the pool continues past these six with the velocity
    // carriers the skeleton takes at WS_CARRIERS + c (v(t): next_segment_v)
    // and WS_CARRIERS + N_VEL + c (v(t+1): prev_segment_next_v), sg_driver.cuh
    // SgCarrierSlots (which also checks the pool's exact per-mode count).
    static constexpr int WS_CARRIERS = N_WORKSPACE;

    static Workspace make_workspace(const BackwardInput& p, const torch::Tensor& rho)
    {
        SWEEP_CHECK((int)p.adjoint_workspace.size() >= N_WORKSPACE,
                    "elastic_tti_sg2d backward requires the propagator-bound "
                    "adjoint_workspace (cuda_layout.backward_workspace_shapes) "
                    "holding at least ", static_cast<int>(N_WORKSPACE),
                    " tensors, got ", p.adjoint_workspace.size());
        Workspace w;
        for (int i = 0; i < N_WORKSPACE; ++i)
            w[i] = pool_required(p.adjoint_workspace, i, rho, "adjoint_workspace");
        return w;
    }

    static void validate_forward(const ForwardInput& p)
    {
        SWEEP_CHECK(!p.models.empty(), "ElasticTTISG forward expects model tensors");
        SWEEP_CHECK(p.pml_vals.size() == 8, "ElasticTTISG forward expects cpmls PML profiles");
        if (p.use_checkpoint)
            SWEEP_CHECK(!p.use_recursive_checkpoint,
                        "ElasticTTISG recursive checkpointing is not implemented yet");
    }

    static void validate_backward(const BackwardInput& p, const char* mode)
    {
        if (std::strcmp(mode, "full") == 0) {
            SWEEP_CHECK(p.models.size() == 16, "ElasticTTISG backward expects prepared models");
            SWEEP_CHECK(p.pml_vals.size() == 8, "ElasticTTISG backward expects cpmls PML profiles");
            SWEEP_CHECK(p.u_forward.defined(), "ElasticTTISG full backward expects saved forward wavefields");
            SWEEP_CHECK(p.u_forward.dim() == 5 && p.u_forward.size(1) == 8,
                        "ElasticTTISG full backward expects u_forward with shape (nt, 8, B, nz, nx)");
        } else if (std::strcmp(mode, "bs") == 0) {
            SWEEP_CHECK(p.models.size() == 16, "ElasticTTISG boundary-saving backward expects prepared models");
            SWEEP_CHECK(p.pml_vals.size() == 8, "ElasticTTISG boundary-saving backward expects cpmls PML profiles");
            SWEEP_CHECK(p.u_last_two.defined(), "ElasticTTISG boundary-saving backward expects last-two wavefield tensor");
        } else {
            SWEEP_CHECK(p.models.size() == 16, "ElasticTTISG checkpoint backward expects prepared models");
            SWEEP_CHECK(p.pml_vals.size() == 8, "ElasticTTISG checkpoint backward expects cpmls PML profiles");
        }
    }

    template <class P>
    static void setup_ctx(SolverContext&, const P&) {}   // no per-edge / topo

    // The CPML memory variables live in the equation's own wavefield tensors;
    // no solver aux slabs are ever installed.
    static void init_aux_slabs(SolverContext&, Wavefield&) {}

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.allocate(p.pml_vals, 2);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, 8, d.B, d.nz, d.nx};   // all 8 physical fields
    }

    static float* field_ptr(WfView& wf, int field_idx)
    {
        return elastic_tti_sg2d::field_ptr(wf, field_idx);
    }

    static WfView view(Wavefield& wf) { return wf.view(); }

    // ===================================================================== //
    // [2] FORWARD — sg_generic_forward, per it in [it_begin, it_end):
    //     velocity_substep -> stress_substep -> inject_source ->
    //     <checkpoint save (shared runtime)> -> save_boundary_fields ->
    //     record_field; after the loop: save_last_state.
    // ===================================================================== //

    // MANDATORY: _c.py Prop.forward always hands the compiled forward its
    // propagation state (persistent _slice_wavefield_buffers in full mode,
    // per-call _transient_forward_wavefields otherwise), sized by
    // cuda_layout.base_nvar + pml_nvar = CKPT_NVAR slots.
    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& rho)
    {
        SWEEP_CHECK((int)p.wavefields.size() == CKPT_NVAR,
                    "elastic_tti_sg2d/forward requires the propagator-bound "
                    "wavefields (cuda_layout.base_nvar + cuda_layout.pml_nvar = ",
                    CKPT_NVAR, " tensors), got ", p.wavefields.size());
        wf.bind(p.wavefields);
    }

    // No-op: this equation writes u_allt from inside its stress kernel.
    // See the call site in sg_driver.cuh for why the hook exists.
    static void capture_allt(torch::Tensor&, WfView&, const SolverContext&, int) {}

    static void velocity_substep(const State& s, WfView& wf,
                                 ElasticCPMLPointer cpml_view,
                                 const SolverContext& solver)
    {
        LAUNCH_ELASTIC_TTI_SG_VELOCITY(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            wf,
            s.model,
            s.grad_ctx,
            cpml_view,
            solver
        );
    }

    static void stress_substep(const State& s, WfView& wf,
                               ElasticCPMLPointer cpml_view,
                               const SolverContext& solver, float* u_this_t)
    {
        LAUNCH_ELASTIC_TTI_SG_STRESS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            wf,
            s.model,
            u_this_t,
            s.grad_ctx,
            cpml_view,
            solver
        );
    }

    static void inject_source(const State& s, const SolverContext& solver,
                              float* field, const torch::Tensor& source,
                              const torch::Tensor& sources_loc, int it, int nsrc)
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
            wf.vx, wf.vy, wf.vz,
            wf.sxx, wf.szz, wf.syz, wf.sxz, wf.sxy
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
                             float* field, torch::Tensor& record, int irec,
                             const torch::Tensor& receivers_loc, int it, int nrec)
    {
        record_kernel<<<s.record_config.grid, s.record_config.block>>>(
            field,
            record[irec].data_ptr<float>(),
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
        copy_tensor_cuda_async(saver.last_two.select(0, 4).select(0, 0), wf.szz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 5).select(0, 0), wf.syz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 6).select(0, 0), wf.sxz_t);
        copy_tensor_cuda_async(saver.last_two.select(0, 7).select(0, 0), wf.sxy_t);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — sg_generic_backward, per reverse it:
    //     fix_rho_grad_at_sources -> inject_residuals -> vel_ptrs_from_u_forward ->
    //     it == 0: image_standalone + fix_rho_grad_at_receivers (loop ends);
    //     it  > 0: full_mode_step (imaging + receiver-rho + adjoint step).
    // ===================================================================== //

    // ---- shared adjoint halves (internal helpers, not skeleton hooks) ---- //
    // stress_adjoint_half / velocity_adjoint_half are shared by
    // plain_adjoint_step (full + ckpt modes, both halves), bs_stress_half (stress
    // half) and bs_velocity_half (velocity half).
private:
    static void stress_adjoint_half(const State& s, const SolverContext& solver,
                                    WfView& adj_view, Workspace& workspace,
                                    ElasticCPMLPointer cpml_view)
    {
        LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.model,
            cpml_view,
            solver,
            workspace[Q_VXX].data_ptr<float>(),
            workspace[Q_VYX].data_ptr<float>(),
            workspace[Q_VZX].data_ptr<float>(),
            workspace[Q_VXZ].data_ptr<float>(),
            workspace[Q_VYZ].data_ptr<float>(),
            workspace[Q_VZZ].data_ptr<float>()
        );
        LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace[Q_VXX].data_ptr<float>(),
            workspace[Q_VYX].data_ptr<float>(),
            workspace[Q_VZX].data_ptr<float>(),
            workspace[Q_VXZ].data_ptr<float>(),
            workspace[Q_VYZ].data_ptr<float>(),
            workspace[Q_VZZ].data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    static void velocity_adjoint_half(const State& s, const SolverContext& solver,
                                      WfView& adj_view, Workspace& workspace,
                                      ElasticCPMLPointer cpml_view)
    {
        LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.model,
            cpml_view,
            solver,
            workspace[Q_VXX].data_ptr<float>(),
            workspace[Q_VYX].data_ptr<float>(),
            workspace[Q_VZX].data_ptr<float>(),
            workspace[Q_VXZ].data_ptr<float>(),
            workspace[Q_VYZ].data_ptr<float>(),
            workspace[Q_VZZ].data_ptr<float>()
        );
        LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace[Q_VXX].data_ptr<float>(),
            workspace[Q_VYX].data_ptr<float>(),
            workspace[Q_VZX].data_ptr<float>(),
            workspace[Q_VXZ].data_ptr<float>(),
            workspace[Q_VYZ].data_ptr<float>(),
            workspace[Q_VZZ].data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }
public:

    // ---- backward hooks -------------------------------------------------- //

    // MANDATORY, every backward mode: _ensure_wavefield_buffers allocates the
    // adjoint set whenever a gradient is asked for and Wrapper.backward binds
    // it (zeroed) as adjoint_wavefields.
    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInput& p,
                                      const torch::Tensor& rho)
    {
        SWEEP_CHECK((int)p.adjoint_wavefields.size() == ADJ_WF_COUNT,
                    "elastic_tti_sg2d/backward requires the propagator-bound "
                    "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar + "
                    "adjoint_extra_nvar = ", ADJ_WF_COUNT, " tensors), got ",
                    p.adjoint_wavefields.size());
        wf.bind(p.adjoint_wavefields);
    }

    static void zero_adjoint_if_first_segment(Wavefield&, bool) {}      // 2-D relies on Python-zeroed buffers

    // grads_out from the propagator (one per prepared model -- rho plus the
    // 15 stiffnesses -- zeroed per backward on the Python side, accumulated
    // here).  MANDATORY: _c.py Wrapper.backward allocates one accumulator per
    // model on every backward (_gradient_buffers;
    // cuda_layout.grads_out_has_wavelet is false here).
    static void bind_grads(const BackwardInput& p, std::vector<torch::Tensor>& grads)
    {
        SWEEP_CHECK(p.grads_out.size() == p.models.size(),
                    "elastic_tti_sg2d/backward requires the propagator-bound "
                    "grads_out, one tensor per prepared model (",
                    p.models.size(), "), got ", p.grads_out.size());
        grads = p.grads_out;
    }

    static std::vector<float> adjoint_source_signs(
        const BackwardInput& p, const torch::Tensor& receiver_fields)
    {
        // 3-D field layout (vx = 0, vy = 1, vz = 2, then stresses).
        return elastic_adjoint_source_signs(p.adjoint_source, receiver_fields, 3);
    }

    struct VelPtrs {
        const float* now[3];
        const float* next[3];
    };

    static VelPtrs vel_ptrs_from_u_forward(const BackwardInput& p, int it,
                                             const torch::Tensor& zero_velocity)
    {
        const bool has_next = (it + 1 < static_cast<int>(p.nt));
        VelPtrs v;
        for (int c = 0; c < 3; ++c) {
            v.now[c] = p.u_forward.select(0, it).select(0, c).data_ptr<float>();
            v.next[c] = has_next
                ? p.u_forward.select(0, it + 1).select(0, c).data_ptr<float>()
                : zero_velocity.data_ptr<float>();
        }
        return v;
    }

    static void fix_rho_grad_at_sources(const State& s, const SolverContext& solver,
                                WfView& adj_view, const BackwardInput& p,
                                const torch::Tensor& source_fields, int it,
                                std::vector<torch::Tensor>& grads)
    {
        if (it < 0 || it >= static_cast<int>(p.nt)) return;
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            const int sfield = source_fields[isrc].item<int>();
            if (sfield > 2) continue;                       // vx = 0, vy = 1, vz = 2
            float* adj_field = elastic_tti_sg2d::field_ptr(adj_view, sfield);
            if (adj_field == nullptr) continue;
            add_body_force_rho_grad_correction<<<s.source_config.grid, s.source_config.block>>>(
                grads[0].data_ptr<float>(),
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
                                 WfView& adj_view, const BackwardInput& p,
                                 const torch::Tensor& receiver_fields,
                                 const std::vector<float>& adj_source_signs,
                                 int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.numel(); ++irec) {
            float* field = elastic_tti_sg2d::field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_signed<<<s.record_config.grid, s.record_config.block>>>(
                field,
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                adj_source_signs[irec],
                solver
            );
        }
    }

    // (image_standalone / fix_rho_grad_at_receivers are also reused by bs_stress_half and
    // the ckpt backward.)
    static void image_standalone(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const VelPtrs& v,
                                 std::vector<torch::Tensor>& grads)
    {
        auto grad_view = stiffness_grad_view(grads);
        LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG_NOBS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            adj_view,
            s.model,
            grad_view,
            v.now[0],
            v.now[1],
            v.now[2],
            v.next[0],
            v.next[1],
            v.next[2],
            s.grad_ctx,
            solver
        );
    }

    static void fix_rho_grad_at_receivers(const State& s, const SolverContext& solver,
                                  std::vector<torch::Tensor>& grads,
                                  const VelPtrs& v, const BackwardInput& p,
                                  const torch::Tensor& receiver_fields,
                                  int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.numel(); ++irec) {
            const int field = receiver_fields[irec].item<int>();
            if (field > 2) continue;                      // stress receiver: no rho term
            sub_receiver_rho_grad_correction<<<s.record_config.grid, s.record_config.block>>>(
                grads[0].data_ptr<float>(),
                v.now[field],
                v.next[field],
                s.models.rho.data_ptr<float>(),
                p.adjoint_source[irec].data_ptr<float>(),
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
                                std::vector<torch::Tensor>& grads,
                                const BackwardInput& p,
                                const torch::Tensor& receiver_fields,
                                int it, int adjoint_nsrc)
    {
        auto adj = adjoint.view();
        image_standalone(s, solver, adj, v, grads);
        fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it,
                          adjoint_nsrc);
        plain_adjoint_step(s, solver, adjoint, workspace, cpml_view);
    }

    static void plain_adjoint_step(const State& s, const SolverContext& solver,
                                   Wavefield& adjoint, Workspace& workspace,
                                   ElasticCPMLPointer cpml_view)
    {
        auto adj_view = adjoint.view();
        stress_adjoint_half(s, solver, adj_view, workspace, cpml_view);
        velocity_adjoint_half(s, solver, adj_view, workspace, cpml_view);
    }

    // ===================================================================== //
    // [4] BACKWARD_BS — sg_generic_backward_bs, per reverse it (floor
    //     max(it_lo, 1)): fix_rho_grad_at_sources / inject_residuals /
    //     uninject_forward_source [inject_step], then bs_stress_half (stress recon
    //     NOPML + strip restore + imaging + receiver-rho + stress-adjoint
    //     half), then bs_velocity_half (velocity-adjoint half + carrier capture +
    //     velocity recon NOPML + strip restore + prefetch).
    //     Before the loop (first segment): seed_recon from u_last_two.
    // ===================================================================== //

    // (no-op like zero_adjoint_if_first_segment above: 2-D relies on Python-zeroed buffers)
    static void zero_adjoint_if_first_segment_bs(Wavefield&, bool) {}

    struct ReconCarriers {
        torch::Tensor fvx_next, fvy_next, fvz_next;
    };

    // Reconstruction state = the 8 physical fields + the v(t+1) carriers,
    // RECON_LIST_DESC order.  MANDATORY: Python owns all RECON_WF_COUNT of
    // them -- cuda_layout.bs_reconstruction_nvar zeroed grids shaped like
    // rho, allocated per backward call by _forward_state_buffers and bound as
    // forward_wavefields.
    static ReconCarriers bind_or_alloc_recon(Wavefield& forward,
                                             const BackwardInput& p,
                                             const torch::Tensor& rho)
    {
        ReconCarriers c;
        wavefields_required(p.forward_wavefields, RECON_WF_COUNT, rho,
                            "elastic_tti_sg2d/bs reconstruction list "
                            "(cuda_layout.bs_reconstruction_nvar)");
        const auto& list = p.forward_wavefields;
        // [0..BS_NVAR) = vx, vy, vz, sxx, szz, syz, sxz, sxy (bind order),
        // physical fields only -- no CPML memory in the reconstruction.
        forward.bind_physical(std::vector<torch::Tensor>(
            list.begin(), list.begin() + BS_NVAR));
        c.fvx_next = list[BS_NVAR + 0];   // [8]  fvx_next
        c.fvy_next = list[BS_NVAR + 1];   // [9]  fvy_next
        c.fvz_next = list[BS_NVAR + 2];   // [10] fvz_next
        return c;
    }

    static void seed_recon(Wavefield& forward, const BackwardInput& p)
    {
        copy_tensor_cuda_async(forward.vx_t, p.u_last_two.select(0, 0).select(0, 0));
        copy_tensor_cuda_async(forward.vy_t, p.u_last_two.select(0, 1).select(0, 0));
        copy_tensor_cuda_async(forward.vz_t, p.u_last_two.select(0, 2).select(0, 0));
        copy_tensor_cuda_async(forward.sxx_t, p.u_last_two.select(0, 3).select(0, 0));
        copy_tensor_cuda_async(forward.szz_t, p.u_last_two.select(0, 4).select(0, 0));
        copy_tensor_cuda_async(forward.syz_t, p.u_last_two.select(0, 5).select(0, 0));
        copy_tensor_cuda_async(forward.sxz_t, p.u_last_two.select(0, 6).select(0, 0));
        copy_tensor_cuda_async(forward.sxy_t, p.u_last_two.select(0, 7).select(0, 0));
    }

    static void uninject_forward_source(const State& s, const SolverContext& solver,
                                        WfView& for_view, const BackwardInput& p,
                                        const torch::Tensor& source_fields,
                                        int it, int forward_nsrc)
    {
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            float* field = elastic_tti_sg2d::field_ptr(for_view, source_fields[isrc].item<int>());
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
                          std::vector<torch::Tensor>& grads,
                          ReconCarriers& carriers, const BackwardInput& p,
                          const torch::Tensor& receiver_fields,
                          int it, int adjoint_nsrc)
    {
        LAUNCH_ELASTIC_TTI_SG_STRESS_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            s.model,
            s.grad_ctx,
            solver
        );

        float* stress_fields[5] = {
            for_view.sxx, for_view.szz, for_view.syz, for_view.sxz, for_view.sxy
        };
        for (int f = 3; f < 8; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                stress_fields[f - 3],
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

        // Imaging and the receiver-rho correction share the same operands:
        // for_view v* is v(it), fv*_next is v(it+1) (overwritten in phase 2).
        {
            VelPtrs v;
            v.now[0] = for_view.vx;
            v.now[1] = for_view.vy;
            v.now[2] = for_view.vz;
            v.next[0] = carriers.fvx_next.data_ptr<float>();
            v.next[1] = carriers.fvy_next.data_ptr<float>();
            v.next[2] = carriers.fvz_next.data_ptr<float>();
            image_standalone(s, solver, adj_view, v, grads);
            fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it,
                              adjoint_nsrc);
        }

        // Stress-adjoint half (the hand-written monolithic runs all four
        // adjoint launches back to back; the p1/p2 split reproduces exactly
        // that sequence when both phases run).
        auto adjv = adjoint.view();
        stress_adjoint_half(s, solver, adjv, workspace, cpml_view);
    }

    static void bs_velocity_half(const State& s, const SolverContext& solver,
                          WfView& for_view, Wavefield& adjoint,
                          Workspace& workspace, ElasticCPMLPointer cpml_view,
                          BoundaryRuntime& boundary_runtime,
                          const GeneralBoundaryPointer& bs, int save_width,
                          ReconCarriers& carriers, Wavefield& forward,
                          int it, int nt)
    {
        auto adjv = adjoint.view();
        velocity_adjoint_half(s, solver, adjv, workspace, cpml_view);

        copy_tensor_cuda_async(carriers.fvx_next, forward.vx_t);
        copy_tensor_cuda_async(carriers.fvy_next, forward.vy_t);
        copy_tensor_cuda_async(carriers.fvz_next, forward.vz_t);

        LAUNCH_ELASTIC_TTI_SG_VELOCITY_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            s.model,
            s.grad_ctx,
            solver
        );

        float* velocity_fields[3] = {
            for_view.vx, for_view.vy, for_view.vz
        };
        for (int f = 0; f < 3; ++f) {
            boundary_runtime.restore_backward_2d_field(
                it,
                velocity_fields[f],
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
    // [5] CKPT + RECURSIVE PLUMBING — sg_generic_backward_ckpt, per chunk:
    //     replay: velocity_substep / stress_substep / save_seg_velocities /
    //     inject_forward_sources; reverse: fix_rho_grad_at_sources / inject_residuals /
    //     vel_ptrs_from_seg / image_standalone / fix_rho_grad_at_receivers /
    //     (it > 0) plain_adjoint_step; after each chunk: export_seg_next_v.
    //     (No recursive checkpointing in this equation: capture_velocities /
    //     vel_ptrs_from_carriers are deliberately absent -- see the header note.)
    // ===================================================================== //

    // Replay state: set 0 of the Python-bound forward_wavefields (zeroed by
    // the propagator per backward call), bound in FULL -- the 8 fields and
    // the 12 CPML memory tensors the replay steps through the PML -- exactly
    // the layout the hand-written driver used to allocate.  Every slot is a
    // full grid here, so the geometry is checked as well.  MANDATORY:
    // _forward_state_shapes("ckpt") derives CKPT_STATE_COUNT slots from the
    // forward slot shapes and Wrapper.backward binds them as
    // forward_wavefields on the checkpoint entry.
    static void bind_or_alloc_recon_ckpt(Wavefield& forward,
                                         const BackwardInput& p,
                                         const torch::Tensor& rho)
    {
        SWEEP_CHECK((int)p.forward_wavefields.size() >= CKPT_STATE_COUNT,
                    "elastic_tti_sg2d/ckpt requires the propagator-bound replay "
                    "state (cuda_layout.base_nvar + pml_nvar = ", CKPT_STATE_COUNT,
                    " tensors per set), got ", p.forward_wavefields.size());
        auto state = wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT,
                                   "elastic_tti_sg2d ckpt replay state");
        for (int i = 0; i < CKPT_STATE_COUNT; ++i)
            pool_slot_checked(state, i, rho, "elastic_tti_sg2d ckpt replay state");
        forward.bind(state);
    }

    static void check_ckpt_aux_layout(const Wavefield&, const Wavefield&) {}

    // Per-segment velocity histories seg[c][k] = v_c(start + k), k in
    // [0, segment_len]: the N_VEL checkpoint_replay slots [vx, vy, vz]
    // (cuda_layout.checkpoint_replay_shapes, (chunk + 1, B, 1, nz, nx) each,
    // allocated once next to the snapshots and never re-zeroed -- every row
    // the reverse pass reads was written by save_seg_velocities earlier in
    // the same segment).  MANDATORY in the chunked mode:
    // cuda_layout.checkpoint_replay_shapes declares N_VEL histories for mode
    // "ckpt" and _ensure_checkpoint_buffers allocates them next to the
    // snapshots.  Taken once per call at the longest segment; the skeleton
    // narrows the rows of a shorter last segment itself.
    static std::vector<torch::Tensor> seg_buffers(const BackwardInput& p,
                                                  const torch::Tensor& rho, int max_rows)
    {
        SWEEP_CHECK((int)p.checkpoint_replay.size() >= N_VEL,
                    "elastic_tti_sg2d/ckpt requires the propagator-bound "
                    "checkpoint_replay (cuda_layout.checkpoint_replay_shapes, ",
                    N_VEL, " velocity histories), got ", p.checkpoint_replay.size());
        std::vector<int64_t> seg_shape = rho.sizes().vec();   // (max_rows, B, 1, nz, nx)
        seg_shape.insert(seg_shape.begin(), static_cast<int64_t>(max_rows));
        std::vector<torch::Tensor> seg;
        for (int c = 0; c < N_VEL; ++c)
            seg.push_back(pool_required(p.checkpoint_replay, c, seg_shape, rho.options(),
                                        "checkpoint_replay"));
        return seg;
    }

    static void save_seg_velocities(std::vector<torch::Tensor>& seg, Wavefield& forward,
                            int slot)
    {
        copy_tensor_cuda_async(seg[0].select(0, slot), forward.vx_t);
        copy_tensor_cuda_async(seg[1].select(0, slot), forward.vy_t);
        copy_tensor_cuda_async(seg[2].select(0, slot), forward.vz_t);
    }

    static void inject_forward_sources(const State& s, const SolverContext& solver,
                                      WfView& for_view, const BackwardInput& p,
                                      const torch::Tensor& source_fields, int it)
    {
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            float* field = elastic_tti_sg2d::field_ptr(for_view, source_fields[isrc].item<int>());
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

    static VelPtrs vel_ptrs_from_seg(const std::vector<torch::Tensor>& seg,
                                int now_offset, int next_offset,
                                const std::vector<torch::Tensor>& next_segment_v)
    {
        VelPtrs v;
        for (int c = 0; c < 3; ++c) {
            v.now[c] = seg[c].select(0, now_offset).data_ptr<float>();
            v.next[c] = (next_offset >= 0)
                ? seg[c].select(0, next_offset).data_ptr<float>()
                : next_segment_v[c].data_ptr<float>();
        }
        return v;
    }

    static void export_seg_next_v(std::vector<torch::Tensor>& prev,
                                   const std::vector<torch::Tensor>& seg)
    {
        copy_tensor_cuda_async(prev[0], seg[0].select(0, 1));
        copy_tensor_cuda_async(prev[1], seg[1].select(0, 1));
        copy_tensor_cuda_async(prev[2], seg[2].select(0, 1));
    }
};

} // namespace elastic_tti_sg2d
