// Driver traits for the 3-D DAS-Mu equation: the per-equation half of the
// staggered-family skeleton in ``common/sg_driver.cuh``.  Line-faithful
// transcription of the hand-written drivers; physics kernels untouched.
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/sg_driver.cuh.
// Structurally this is das_mu2d at three dimensions (see that file's header
// for the family deltas: elastic-kernel reuse through elastic_view(), the
// identity aux slabs, no grad fusion, no DD cut support).  Deltas of the 3-D
// member worth naming:
//   * 15 physical fields (9 elastic + 6 strains) / 33 wavefield tensors /
//     18-tensor adjoint workspace, three velocity carriers (N_VEL = 3);
//   * boundary saving stores all 15 fields but restores only the elastic 9
//     (strains are record-only, and unlike 2-D they are never seeded from
//     last_two either -- the hand-written seed copies 9 fields);
//   * the reconstruction wavefield binds/allocates WITHOUT the CPML memory
//     tensors (use_pml = false; 2-D keeps them);
//   * the full backward zeroes the adjoint state after binding (2-D relies
//     on Python-zeroed buffers), mapped to zero_adjoint_if_first_segment on the first segment;
//   * the full/ckpt imaging kernel is the shared LAUNCH_CALCULATE_GRAD_
//     3DELASTIC_BS over a velocity-only view (2-D has a dedicated _NOBS
//     kernel).
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/das_mu.h"
#include "../../common/elastic.h"
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../common/eq_driver.cuh"
#include "../../common/sg_driver.cuh"
#include "../../launch/config.h"

namespace das_mu3d {

struct Driver {
    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // Prologue of every entry (sg_driver.cuh timing map, in call order):
    //   validate_backward (backward only), parse_models, setup_ctx,
    //   bind_or_alloc_* wavefields, init_aux_slabs, alloc_cpml,
    //   bind_grads / alloc_grads, make_workspace, make_state,
    //   signed_adjoint_sources.
    // ===================================================================== //

    static constexpr int NDIM = 3;
    static constexpr const char* NAME = "das_mu3d";
    static constexpr int CKPT_NVAR = 33;
    static constexpr const char* CKPT_COUNT_MSG =
        "DAS Mu 3D checkpointing expects 33 checkpoint tensors";
    static constexpr const char* CKPT_RECURSIVE_COUNT_MSG =
        "DAS Mu 3D recursive checkpointing expects 33 checkpoint tensors";
    // vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, exx, eyy, ezz, exy, exz, eyz
    static constexpr int BS_NVAR = 15;
    static constexpr int CUT_MASK_BITS = 0x0;
    static constexpr const char* CUT_MASK_DESC =
        "no bits (the borrowed elastic kernels are not cut-aware)";
    static constexpr int ADJ_WF_COUNT = 33;
    static constexpr int RECON_WF_COUNT = 15;
    static constexpr const char* RECON_LIST_DESC =
        "(the 15 base DAS-Mu field tensors; the reconstruction carries no "
        "CPML memory)";
    static constexpr int N_VEL = 3;
    static constexpr bool IMAGING_USES_NEXT_V = true;   // imaging consumes v(t+1) carriers

    using Wavefield = DasMuWavefieldTensor3D;

    // Per-step view pair: the custom stress/strain kernel and the field maps
    // read the DAS view; the borrowed elastic kernels read the adapter.
    struct WfView {
        DasMuWavefieldPointer3D das;
        ElasticWavefieldPointer el;
    };

    using CPML = ElasticCPMLTensor;

    struct Models {
        torch::Tensor vp, vs, rho, mu, lambda;
    };

    template <class P>
    static Models parse_models(const P& p)
    {
        Models m;
        m.vp = p.models[0];
        m.vs = p.models[1];
        m.rho = p.models[2];
        m.mu = m.rho * m.vs * m.vs;
        m.lambda = m.rho * (m.vp * m.vp - 2 * m.vs * m.vs);
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

    static Workspace make_workspace(const BackwardInput& p, const torch::Tensor& vp)
    {
        Workspace workspace;
        init_adjoint_workspace(workspace, p.adjoint_workspace, vp, 3);
        return workspace;
    }

    static void validate_forward(const ForwardInput&) {}

    static void validate_backward(const BackwardInput&, const char*) {}

    template <class P>
    static void setup_ctx(SolverContext&, const P&) {}   // no per-edge / topo

    // DAS Mu reuses the elastic kernels, which address the CPML memory
    // variables through the solver's aux slabs.  DAS keeps those tensors
    // full-domain -- its CUDALayoutSpec sets no pml_slot_axes -- so install
    // identity slabs here.  Left default constructed they are lo=hi=n=0,
    // tot() is 0, the aux row stride collapses and every (iz,iy) row aliases
    // the first: a data race whose output changes from run to run.
    static void init_aux_slabs(SolverContext& solver, Wavefield&)
    {
        TORCH_CHECK(solver.init_aux_slabs(solver.nz, solver.ny, solver.nx),
                    "DAS Mu 3D: full-grid CPML memory variables rejected by "
                    "init_aux_slabs");
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.allocate(p.pml_vals, 3);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, 3, d.B, d.nz, d.ny, d.nx};   // only the velocities
    }

    static float* field_ptr(WfView& wf, int field_idx)
    {
        return das_mu3d_field_ptr(wf.das, field_idx);
    }

    static WfView view(Wavefield& wf)
    {
        return {wf.view(), wf.elastic_view()};
    }

    // ===================================================================== //
    // [2] FORWARD — sg_generic_forward, per it in [it_begin, it_end):
    //   velocity_substep / stress_substep / inject_source /
    //   <checkpoint save> / save_boundary_fields / record_field;
    //   after the loop: save_last_state (final snapshot for backward_bs).
    // ===================================================================== //

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.wavefields.empty())
            wf.bind(p.wavefields, true);
        else
            wf.allocate(vp, true);
    }

    // (velocity_substep / stress_substep are also replayed by the ckpt and
    // recursive-ckpt backward modes.)
    static void velocity_substep(const State& s, WfView& wf,
                                 ElasticCPMLPointer cpml_view,
                                 const SolverContext& solver)
    {
        LAUNCH_3DELASTIC_VELOCITY(
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
        LAUNCH_DAS_MU3D_STRESS_STRAIN(
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
                              float* field, const torch::Tensor& source,
                              const torch::Tensor& sources_loc, int it, int nsrc)
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
        float* fields[15] = {
            wf.das.vx, wf.das.vy, wf.das.vz,
            wf.das.sxx, wf.das.syy, wf.das.szz,
            wf.das.sxy, wf.das.sxz, wf.das.syz,
            wf.das.exx, wf.das.eyy, wf.das.ezz,
            wf.das.exy, wf.das.exz, wf.das.eyz
        };
        for (int f = 0; f < 15; ++f) {
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
                f == 14
            );
        }
    }

    static void record_field(const State& s, const SolverContext& solver,
                             float* field, torch::Tensor& record, int irec,
                             const torch::Tensor& receivers_loc, int it, int nrec)
    {
        record_kernel_3d<<<s.record_config.grid, s.record_config.block>>>(
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
        saver.last_two_t.select(0, 0).select(0, 0).copy_(wf.vx_t);
        saver.last_two_t.select(0, 1).select(0, 0).copy_(wf.vy_t);
        saver.last_two_t.select(0, 2).select(0, 0).copy_(wf.vz_t);
        saver.last_two_t.select(0, 3).select(0, 0).copy_(wf.sxx_t);
        saver.last_two_t.select(0, 4).select(0, 0).copy_(wf.syy_t);
        saver.last_two_t.select(0, 5).select(0, 0).copy_(wf.szz_t);
        saver.last_two_t.select(0, 6).select(0, 0).copy_(wf.sxy_t);
        saver.last_two_t.select(0, 7).select(0, 0).copy_(wf.sxz_t);
        saver.last_two_t.select(0, 8).select(0, 0).copy_(wf.syz_t);
        saver.last_two_t.select(0, 9).select(0, 0).copy_(wf.exx_t);
        saver.last_two_t.select(0, 10).select(0, 0).copy_(wf.eyy_t);
        saver.last_two_t.select(0, 11).select(0, 0).copy_(wf.ezz_t);
        saver.last_two_t.select(0, 12).select(0, 0).copy_(wf.exy_t);
        saver.last_two_t.select(0, 13).select(0, 0).copy_(wf.exz_t);
        saver.last_two_t.select(0, 14).select(0, 0).copy_(wf.eyz_t);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — sg_generic_backward, per reverse it:
    //   fix_rho_grad_at_sources / inject_residuals / vel_ptrs_from_u_forward;
    //   it == 0: image_standalone + fix_rho_grad_at_receivers, loop ends;
    //   it  > 0: full_mode_step (imaging + receiver-rho + adjoint step, in
    //   the equation's exact fused order).
    // ===================================================================== //

    // The two adjoint halves (DAS stress/strain PREPARE + elastic STRESS
    // APPLY; elastic VELOCITY PREPARE + APPLY).  Shared by plain_adjoint_step
    // ([3], also ckpt/recursive) and by bs_stress_half / bs_velocity_half ([4]).
private:
    static void stress_adjoint_half(const State& s, const SolverContext& solver,
                                    Wavefield& adjoint, Workspace& workspace,
                                    ElasticCPMLPointer cpml_view)
    {
        auto adj_view = adjoint.view();
        auto elastic_adj_view = adjoint.elastic_view();
        LAUNCH_DAS_MU3D_STRESS_STRAIN_ADJOINT_PREPARE(
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
            workspace.qzz_t.data_ptr<float>()
        );
        LAUNCH_3DELASTIC_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            elastic_adj_view,
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

    static void velocity_adjoint_half(const State& s, const SolverContext& solver,
                                      Wavefield& adjoint, Workspace& workspace,
                                      ElasticCPMLPointer cpml_view)
    {
        auto elastic_adj_view = adjoint.elastic_view();
        LAUNCH_3DELASTIC_VELOCITY_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            elastic_adj_view,
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
        LAUNCH_3DELASTIC_VELOCITY_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            elastic_adj_view,
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
public:

    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.adjoint_wavefields.empty())
            wf.bind(p.adjoint_wavefields, true);
        else
            wf.allocate(vp, true);
    }

    static void zero_adjoint_if_first_segment(Wavefield& adjoint, bool first_segment)
    {
        // The hand-written full backward zeroes the adjoint state after
        // binding (2-D relies on Python-zeroed buffers).  FIRST segment only:
        // a continuation call must keep the carried adjoint state (legacy
        // monolithic calls always have bw_begin == nt).
        if (!first_segment) return;
        for (auto& tensor : adjoint.state_tensors())
            if (tensor.defined()) tensor.zero_();
    }

    static void bind_grads(const BackwardInput& p, std::vector<torch::Tensor>& grads)
    {
        if (!p.grads_out.empty()) {
            TORCH_CHECK(p.grads_out.size() == 3,
                        "elastic grads_out must hold exactly {grad_vp, grad_vs, "
                        "grad_rho}");
            grads = {p.grads_out[0], p.grads_out[1], p.grads_out[2]};
        } else {
            alloc_grads(p.models[0], grads);
        }
    }

    static void alloc_grads(const torch::Tensor& vp, std::vector<torch::Tensor>& grads)
    {
        grads = {torch::zeros_like(vp), torch::zeros_like(vp), torch::zeros_like(vp)};
    }

    static std::vector<torch::Tensor> signed_adjoint_sources(
        const BackwardInput& p, const torch::Tensor& receiver_fields)
    {
        return elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);
    }

    struct VelPtrs {
        torch::Tensor now_vx, now_vy, now_vz;   // Tensors: the imaging view
        const float* now[3];
        const float* next[3];
    };

    // (shared builder: also used by vel_ptrs_from_seg / vel_ptrs_from_carriers in [5])
    static VelPtrs _vel_ptrs(const torch::Tensor& nvx, const torch::Tensor& nvy,
                             const torch::Tensor& nvz,
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

    static VelPtrs vel_ptrs_from_u_forward(const BackwardInput& p, int it,
                                             const torch::Tensor& zero_velocity)
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

    // (fix_rho_grad_at_sources / inject_residuals also fire in backward_bs and the
    // ckpt/recursive modes.)
    static void fix_rho_grad_at_sources(const State& s, const SolverContext& solver,
                                WfView& adj_view, const BackwardInput& p,
                                const torch::Tensor& source_fields, int it,
                                std::vector<torch::Tensor>& grads)
    {
        if (it < 0 || it >= p.nt) return;
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            const int sfield = source_fields[isrc].item<int>();
            if (sfield > 2) continue;                  // vx = 0, vy = 1, vz = 2
            float* adj_field = das_mu3d_field_ptr(adj_view.das, sfield);
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
                                 WfView& adj_view, const BackwardInput& p,
                                 const torch::Tensor& receiver_fields,
                                 const std::vector<torch::Tensor>& adj_source_signed,
                                 int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.numel(); ++irec) {
            float* field = das_mu3d_field_ptr(adj_view.das, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<s.record_config.grid, s.record_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }
    }

    // A velocity-only wavefield view for the gradient kernel: the stress
    // slots alias vx (never read by the imaging).
    static ElasticWavefieldTensor make_velocity_view(const torch::Tensor& vx,
                                                     const torch::Tensor& vy,
                                                     const torch::Tensor& vz)
    {
        ElasticWavefieldTensor view;
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

    // (image_standalone / fix_rho_grad_at_receivers / plain_adjoint_step are also
    // used by the ckpt/recursive imaging.)
    static void image_standalone(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const VelPtrs& v,
                                 std::vector<torch::Tensor>& grads)
    {
        auto current_forward = make_velocity_view(v.now_vx, v.now_vy, v.now_vz);
        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            current_forward.view(),
            adj_view.el,
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
                                  std::vector<torch::Tensor>& grads,
                                  const VelPtrs& v, const BackwardInput& p,
                                  const torch::Tensor& receiver_fields,
                                  int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.numel(); ++irec) {
            const int field = receiver_fields[irec].item<int>();
            if (field > 2) continue;              // stress/strain receiver: no rho term
            sub_receiver_rho_grad_correction<<<s.record_config.grid, s.record_config.block>>>(
                grads[2].data_ptr<float>(),
                v.now[field],
                v.next[field],
                s.models.rho.data_ptr<float>(),
                p.adjoint_source[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                3,
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

    // ===================================================================== //
    // [4] BACKWARD_BS — sg_generic_backward_bs, per reverse it (floor
    //   max(it_lo, 1)): fix_rho_grad_at_sources / inject_residuals /
    //   uninject_forward_source [inject_step], then
    //   bs_stress_half (stress recon (NOPML) + strip restore + imaging +
    //   receiver-rho + stress-adjoint half), then
    //   bs_velocity_half (velocity-adjoint half + carrier capture + velocity recon
    //   (NOPML) + strip restore + prefetch);
    //   before the loop (first segment): seed_recon from u_last_two.
    // ===================================================================== //

    static void zero_adjoint_if_first_segment_bs(Wavefield&, bool) {}

    struct ReconCarriers {
        torch::Tensor fvx_prev, fvy_prev, fvz_prev;
    };

    // Unlike 2-D, the reconstruction wavefield lives WITHOUT the CPML memory
    // tensors (the NOPML recon kernels never touch them).
    static ReconCarriers bind_or_alloc_recon(Wavefield& forward,
                                             const BackwardInput& p,
                                             const torch::Tensor& vp)
    {
        ReconCarriers c;
        if (!p.forward_wavefields.empty())
            forward.bind(p.forward_wavefields, false);
        else
            forward.allocate(vp, false);
        c.fvx_prev = torch::zeros_like(vp);
        c.fvy_prev = torch::zeros_like(vp);
        c.fvz_prev = torch::zeros_like(vp);
        return c;
    }

    // Seed the 9 elastic fields from the last snapshot; the strains are never
    // seeded (record-only, and the restore below never brings them back).
    static void seed_recon(Wavefield& forward, const BackwardInput& p)
    {
        forward.vx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
        forward.vy_t.copy_(p.u_last_two.select(0, 1).select(0, 0));
        forward.vz_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
        forward.sxx_t.copy_(p.u_last_two.select(0, 3).select(0, 0));
        forward.syy_t.copy_(p.u_last_two.select(0, 4).select(0, 0));
        forward.szz_t.copy_(p.u_last_two.select(0, 5).select(0, 0));
        forward.sxy_t.copy_(p.u_last_two.select(0, 6).select(0, 0));
        forward.sxz_t.copy_(p.u_last_two.select(0, 7).select(0, 0));
        forward.syz_t.copy_(p.u_last_two.select(0, 8).select(0, 0));
    }

    static void uninject_forward_source(const State& s, const SolverContext& solver,
                                        WfView& for_view, const BackwardInput& p,
                                        const torch::Tensor& source_fields,
                                        const torch::Tensor& neg_forward_source,
                                        int it, int forward_nsrc)
    {
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            float* field = das_mu3d_field_ptr(for_view.das, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source_3d<<<s.source_config.grid, s.source_config.block>>>(
                field,
                neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
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
        LAUNCH_3DELASTIC_STRESS_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view.el,
            s.models.lambda.data_ptr<float>(),
            s.models.mu.data_ptr<float>(),
            s.grad_ctx,
            solver
        );

        float* field2[6] = {for_view.das.sxx, for_view.das.syy, for_view.das.szz,
                            for_view.das.sxy, for_view.das.sxz, for_view.das.syz};
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

        LAUNCH_CALCULATE_GRAD_3DELASTIC_BS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view.el,
            adj_view.el,
            carriers.fvx_prev.data_ptr<float>(),
            carriers.fvy_prev.data_ptr<float>(),
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

        // Same operands the imaging just correlated: for_view v* is v(it),
        // fv*_prev is v(it+1) (overwritten in phase 2).
        {
            VelPtrs v;
            v.now[0] = for_view.das.vx;
            v.now[1] = for_view.das.vy;
            v.now[2] = for_view.das.vz;
            v.next[0] = carriers.fvx_prev.data_ptr<float>();
            v.next[1] = carriers.fvy_prev.data_ptr<float>();
            v.next[2] = carriers.fvz_prev.data_ptr<float>();
            fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it,
                              adjoint_nsrc);
        }

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

        carriers.fvz_prev.copy_(forward.vz_t);
        carriers.fvy_prev.copy_(forward.vy_t);
        carriers.fvx_prev.copy_(forward.vx_t);

        LAUNCH_3DELASTIC_VELOCITY_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view.el,
            s.models.rho.data_ptr<float>(),
            s.grad_ctx,
            solver
        );

        float* field1[3] = {for_view.das.vx, for_view.das.vy, for_view.das.vz};
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
    // [5] CKPT + RECURSIVE PLUMBING — sg_generic_backward_ckpt, per chunk:
    //   replay: velocity_substep / stress_substep / save_seg_velocities /
    //   inject_forward_sources; reverse: fix_rho_grad_at_sources / inject_residuals /
    //   vel_ptrs_from_seg / image_standalone / fix_rho_grad_at_receivers /
    //   (it > 0) plain_adjoint_step; after each chunk: export_seg_next_v.
    //   Recursive: replay + capture_velocities (IMAGING_USES_NEXT_V eqs also capture v at
    //   it+1), then vel_ptrs_from_carriers at the imaging.
    // ===================================================================== //

    static void bind_or_alloc_recon_ckpt(Wavefield& forward,
                                         const BackwardInput& p,
                                         const torch::Tensor& vp)
    {
        if (!p.forward_wavefields.empty())
            forward.bind(p.forward_wavefields, true);
        else
            forward.allocate(vp, true);
    }

    static void alloc_recursive_start_state(Wavefield& wf, const BackwardInput&,
                                            const torch::Tensor& vp)
    {
        wf.allocate(vp, true);
    }

    static void check_ckpt_aux_layout(const Wavefield&, const Wavefield&) {}

    // ---- seg / carrier plumbing (ckpt + recursive modes) ----

    static std::vector<torch::Tensor> alloc_seg_buffers(const torch::Tensor& vp,
                                                        int segment_len)
    {
        auto seg_vx = torch::zeros({segment_len + 1, vp.size(0) * vp.size(1), 1,
                                    vp.size(2), vp.size(3), vp.size(4)}, vp.options());
        auto seg_vy = torch::zeros_like(seg_vx);
        auto seg_vz = torch::zeros_like(seg_vx);
        return {seg_vx, seg_vy, seg_vz};
    }

    static void save_seg_velocities(std::vector<torch::Tensor>& seg, Wavefield& forward,
                            int slot)
    {
        seg[0].select(0, slot).copy_(forward.vx_t);
        seg[1].select(0, slot).copy_(forward.vy_t);
        seg[2].select(0, slot).copy_(forward.vz_t);
    }

    static void inject_forward_sources(const State& s, const SolverContext& solver,
                                      WfView& for_view, const BackwardInput& p,
                                      const torch::Tensor& source_fields, int it)
    {
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            float* field = das_mu3d_field_ptr(for_view.das, source_fields[isrc].item<int>());
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

    static VelPtrs vel_ptrs_from_seg(const std::vector<torch::Tensor>& seg,
                                int now_offset, int next_offset,
                                const std::vector<torch::Tensor>& next_segment_v)
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

    static void export_seg_next_v(std::vector<torch::Tensor>& prev,
                                   const std::vector<torch::Tensor>& seg)
    {
        prev[0].copy_(seg[0].select(0, 1));
        prev[1].copy_(seg[1].select(0, 1));
        prev[2].copy_(seg[2].select(0, 1));
    }

    static void capture_velocities(std::vector<torch::Tensor>& v, Wavefield& forward)
    {
        v[0].copy_(forward.vx_t);
        v[1].copy_(forward.vy_t);
        v[2].copy_(forward.vz_t);
    }

    static VelPtrs vel_ptrs_from_carriers(const std::vector<torch::Tensor>& current_v,
                                    const std::vector<torch::Tensor>& next_v)
    {
        return _vel_ptrs(current_v[0], current_v[1], current_v[2],
                         next_v[0].data_ptr<float>(),
                         next_v[1].data_ptr<float>(),
                         next_v[2].data_ptr<float>());
    }
};

} // namespace das_mu3d
