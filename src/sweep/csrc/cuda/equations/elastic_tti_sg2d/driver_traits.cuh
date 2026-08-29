// Driver traits for the 2-D staggered-grid elastic TTI equation: the
// per-equation half of the staggered-family skeleton in
// ``common/sg_driver.cuh``.  Line-faithful transcription of the hand-written
// drivers; physics kernels untouched.  Deltas vs the elastic/das_mu family
// members worth naming:
//   * the model set is rho + 15 stiffness tensors (16 grads); the kernels
//     take a StiffnessPointer, rebuilt on demand from p.models / grads;
//   * three velocity components on a 2-D grid (TTI couples vy), so N_VEL = 3
//     and the signed adjoint sources use the 3-D field layout;
//   * the adjoint workspace is six plain scratch tensors, always allocated
//     internally (the hand-written drivers never read p.adjoint_workspace);
//   * u_allt stores all 8 physical fields, not just the velocities;
//   * the boundary-saving reconstruction wavefield is always allocated
//     internally (never bound from p.forward_wavefields);
//   * per-mode entry validation keeps the hand-written message texts
//     (validate_backward hook);
//   * no recursive checkpointing: the forward refuses it and backward.cu
//     instantiates no recursive driver, so the recursive-only hooks
//     (capture_velocities, carrier_vel_ptrs, CKPT_RECURSIVE_COUNT_MSG) are
//     deliberately absent;
//   * no DD cut support, no aux slabs (the CPML memory lives in the
//     equation's own wavefield tensors).
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
#include "../../common/elastic.h"   // elastic_signed_adjoint_sources
#include "../../common/cudautils.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../common/wavetypes.h"
#include "../../common/eq_driver.cuh"
#include "../../common/sg_driver.cuh"
#include "../../launch/config.h"

namespace elastic_tti_sg2d {

struct Driver {
    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "elastic_tti_sg2d";
    static constexpr int CKPT_NVAR = 20;
    static constexpr const char* CKPT_COUNT_MSG =
        "ElasticTTISG checkpointing expects 20 checkpoint tensors";
    static constexpr int BS_NVAR = 8;   // vx, vy, vz, sxx, szz, syz, sxz, sxy
    static constexpr int CUT_MASK_BITS = 0x0;
    static constexpr const char* CUT_MASK_DESC =
        "no bits (ElasticTTISG kernels are not cut-aware)";
    static constexpr int ADJ_WF_COUNT = 20;
    static constexpr int RECON_WF_COUNT = 20;
    static constexpr const char* RECON_LIST_DESC =
        "(the full 20-tensor ElasticTTISG wavefield list)";
    static constexpr int N_VEL = 3;
    static constexpr bool NEXT_V = true;   // imaging consumes v(t+1) carriers

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

    template <class P>
    static void setup_ctx(SolverContext&, const P&) {}   // no per-edge / topo

    static void validate_forward(const ForwardInput& p)
    {
        TORCH_CHECK(!p.models.empty(), "ElasticTTISG forward expects model tensors");
        TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTISG forward expects cpmls PML profiles");
        if (p.use_checkpoint)
            TORCH_CHECK(!p.use_recursive_checkpoint,
                        "ElasticTTISG recursive checkpointing is not implemented yet");
    }

    static void validate_backward(const BackwardInput& p, const char* mode)
    {
        if (std::strcmp(mode, "full") == 0) {
            TORCH_CHECK(p.models.size() == 16, "ElasticTTISG backward expects prepared models");
            TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTISG backward expects cpmls PML profiles");
            TORCH_CHECK(p.u_forward.defined(), "ElasticTTISG full backward expects saved forward wavefields");
            TORCH_CHECK(p.u_forward.dim() == 5 && p.u_forward.size(1) == 8,
                        "ElasticTTISG full backward expects u_forward with shape (nt, 8, B, nz, nx)");
        } else if (std::strcmp(mode, "bs") == 0) {
            TORCH_CHECK(p.models.size() == 16, "ElasticTTISG boundary-saving backward expects prepared models");
            TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTISG boundary-saving backward expects cpmls PML profiles");
            TORCH_CHECK(p.u_last_two.defined(), "ElasticTTISG boundary-saving backward expects last-two wavefield tensor");
        } else {
            TORCH_CHECK(p.models.size() == 16, "ElasticTTISG checkpoint backward expects prepared models");
            TORCH_CHECK(p.pml_vals.size() == 8, "ElasticTTISG checkpoint backward expects cpmls PML profiles");
        }
    }

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& rho)
    {
        if (!p.wavefields.empty())
            wf.bind(p.wavefields);
        else
            wf.allocate(rho);
    }

    static WfView view(Wavefield& wf) { return wf.view(); }

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

    static float* field_ptr(WfView& wf, int field_idx)
    {
        return elastic_tti_sg2d::field_ptr(wf, field_idx);
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
        saver.last_two_t.select(0, 0).select(0, 0).copy_(wf.vx_t);
        saver.last_two_t.select(0, 1).select(0, 0).copy_(wf.vy_t);
        saver.last_two_t.select(0, 2).select(0, 0).copy_(wf.vz_t);
        saver.last_two_t.select(0, 3).select(0, 0).copy_(wf.sxx_t);
        saver.last_two_t.select(0, 4).select(0, 0).copy_(wf.szz_t);
        saver.last_two_t.select(0, 5).select(0, 0).copy_(wf.syz_t);
        saver.last_two_t.select(0, 6).select(0, 0).copy_(wf.sxz_t);
        saver.last_two_t.select(0, 7).select(0, 0).copy_(wf.sxy_t);
    }

    // ---- backward hooks -------------------------------------------------- //

    // Six plain scratch tensors, always internal: the hand-written drivers
    // never read p.adjoint_workspace.
    using Workspace = std::array<torch::Tensor, 6>;

    static Workspace make_workspace(const BackwardInput&, const torch::Tensor& rho)
    {
        return Workspace{
            torch::zeros_like(rho),
            torch::zeros_like(rho),
            torch::zeros_like(rho),
            torch::zeros_like(rho),
            torch::zeros_like(rho),
            torch::zeros_like(rho),
        };
    }

    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInput& p,
                                      const torch::Tensor& rho)
    {
        if (!p.adjoint_wavefields.empty())
            wf.bind(p.adjoint_wavefields);
        else
            wf.allocate(rho);
    }

    static void prep_adjoint(Wavefield&, bool) {}      // 2-D relies on Python-zeroed buffers
    static void prep_adjoint_bs(Wavefield&, bool) {}

    // The hand-written drivers never read p.grads_out: gradients are always
    // freshly allocated, one per prepared model (rho + 15 stiffnesses).
    static void bind_grads(const BackwardInput& p, std::vector<torch::Tensor>& grads)
    {
        grads = zero_model_grads(p.models);
    }

    static std::vector<torch::Tensor> signed_adjoint_sources(
        const BackwardInput& p, const torch::Tensor& receiver_fields)
    {
        // 3-D field layout (vx = 0, vy = 1, vz = 2, then stresses).
        return elastic_signed_adjoint_sources(p.adjoint_source, receiver_fields, 3);
    }

    struct VelPtrs {
        const float* now[3];
        const float* next[3];
    };

    static VelPtrs select_forward_velocities(const BackwardInput& p, int it,
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

    static void undo_body_force(const State& s, const SolverContext& solver,
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
                                 const std::vector<torch::Tensor>& adj_source_signed,
                                 int it, int adjoint_nsrc)
    {
        for (int irec = 0; irec < receiver_fields.numel(); ++irec) {
            float* field = elastic_tti_sg2d::field_ptr(adj_view, receiver_fields[irec].item<int>());
            if (field == nullptr) continue;
            add_source<<<s.record_config.grid, s.record_config.block>>>(
                field,
                adj_source_signed[irec].data_ptr<float>(),
                p.adjoint_sources_loc.data_ptr<int>(),
                it,
                adjoint_nsrc,
                solver
            );
        }
    }

    static void uninject_forward_source(const State& s, const SolverContext& solver,
                                        WfView& for_view, const BackwardInput& p,
                                        const torch::Tensor& source_fields,
                                        const torch::Tensor& neg_forward_source,
                                        int it, int forward_nsrc)
    {
        for (int isrc = 0; isrc < source_fields.numel(); ++isrc) {
            float* field = elastic_tti_sg2d::field_ptr(for_view, source_fields[isrc].item<int>());
            if (field == nullptr) continue;
            add_source<<<s.source_config.grid, s.source_config.block>>>(
                field,
                neg_forward_source.data_ptr<float>(),
                p.forward_sources_loc.data_ptr<int>(),
                it,
                forward_nsrc,
                solver
            );
        }
    }

    static void inject_sources_fwd_bw(const State& s, const SolverContext& solver,
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

    static void undo_receiver_rho(const State& s, const SolverContext& solver,
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
    static void full_fused_step(const State& s, const SolverContext& solver,
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
        undo_receiver_rho(s, solver, grads, v, p, receiver_fields, it,
                          adjoint_nsrc);
        plain_adjoint_step(s, solver, adjoint, workspace, cpml_view);
    }

    static void plain_adjoint_step(const State& s, const SolverContext& solver,
                                   Wavefield& adjoint, Workspace& workspace,
                                   ElasticCPMLPointer cpml_view)
    {
        auto adj_view = adjoint.view();
        LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.model,
            cpml_view,
            solver,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>()
        );
        LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>(),
            s.grad_ctx,
            solver
        );
        LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            s.model,
            cpml_view,
            solver,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>()
        );
        LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adj_view,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    struct ReconCarriers {
        torch::Tensor fvx_next, fvy_next, fvz_next;
    };

    // The boundary-saving reconstruction wavefield is always allocated
    // internally: the hand-written driver never binds p.forward_wavefields.
    static ReconCarriers bind_or_alloc_recon(Wavefield& forward,
                                             const BackwardInput&,
                                             const torch::Tensor& rho)
    {
        ReconCarriers c;
        forward.allocate(rho);
        c.fvx_next = torch::zeros_like(rho);
        c.fvy_next = torch::zeros_like(rho);
        c.fvz_next = torch::zeros_like(rho);
        return c;
    }

    static void bind_or_alloc_recon_ckpt(Wavefield& forward,
                                         const BackwardInput& p,
                                         const torch::Tensor& rho)
    {
        if (!p.forward_wavefields.empty())
            forward.bind(p.forward_wavefields);
        else
            forward.allocate(rho);
    }

    static void alloc_recursive_start_state(Wavefield& wf, const BackwardInput&,
                                            const torch::Tensor& rho)
    {
        wf.allocate(rho);
    }

    static void check_ckpt_aux_layout(const Wavefield&, const Wavefield&) {}

    static void seed_recon(Wavefield& forward, const BackwardInput& p)
    {
        forward.vx_t.copy_(p.u_last_two.select(0, 0).select(0, 0));
        forward.vy_t.copy_(p.u_last_two.select(0, 1).select(0, 0));
        forward.vz_t.copy_(p.u_last_two.select(0, 2).select(0, 0));
        forward.sxx_t.copy_(p.u_last_two.select(0, 3).select(0, 0));
        forward.szz_t.copy_(p.u_last_two.select(0, 4).select(0, 0));
        forward.syz_t.copy_(p.u_last_two.select(0, 5).select(0, 0));
        forward.sxz_t.copy_(p.u_last_two.select(0, 6).select(0, 0));
        forward.sxy_t.copy_(p.u_last_two.select(0, 7).select(0, 0));
    }

    static void bs_phase1(const State& s, const SolverContext& solver,
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

        {
            auto grad_view = stiffness_grad_view(grads);
            LAUNCH_CALCULATE_GRAD_ELASTIC_TTI_SG_NOBS(
                s.order,
                s.launch_config.grid,
                s.launch_config.block,
                adj_view,
                s.model,
                grad_view,
                for_view.vx,
                for_view.vy,
                for_view.vz,
                carriers.fvx_next.data_ptr<float>(),
                carriers.fvy_next.data_ptr<float>(),
                carriers.fvz_next.data_ptr<float>(),
                s.grad_ctx,
                solver
            );
        }

        // Same operands the imaging just correlated: for_view v* is v(it),
        // fv*_next is v(it+1) (overwritten in phase 2).
        {
            VelPtrs v;
            v.now[0] = for_view.vx;
            v.now[1] = for_view.vy;
            v.now[2] = for_view.vz;
            v.next[0] = carriers.fvx_next.data_ptr<float>();
            v.next[1] = carriers.fvy_next.data_ptr<float>();
            v.next[2] = carriers.fvz_next.data_ptr<float>();
            undo_receiver_rho(s, solver, grads, v, p, receiver_fields, it,
                              adjoint_nsrc);
        }

        // Stress-adjoint half (the hand-written monolithic runs all four
        // adjoint launches back to back; the p1/p2 split reproduces exactly
        // that sequence when both phases run).
        auto adjv = adjoint.view();
        LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adjv,
            s.model,
            cpml_view,
            solver,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>()
        );
        LAUNCH_ELASTIC_TTI_SG_STRESS_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adjv,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>(),
            s.grad_ctx,
            solver
        );
    }

    static void bs_phase2(const State& s, const SolverContext& solver,
                          WfView& for_view, Wavefield& adjoint,
                          Workspace& workspace, ElasticCPMLPointer cpml_view,
                          BoundaryRuntime& boundary_runtime,
                          const GeneralBoundaryPointer& bs, int save_width,
                          ReconCarriers& carriers, Wavefield& forward,
                          int it, int nt)
    {
        auto adjv = adjoint.view();
        LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_PREPARE(
            s.order, s.launch_config.grid, s.launch_config.block,
            adjv,
            s.model,
            cpml_view,
            solver,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>()
        );
        LAUNCH_ELASTIC_TTI_SG_VELOCITY_ADJOINT_APPLY(
            s.order, s.launch_config.grid, s.launch_config.block,
            adjv,
            workspace[0].data_ptr<float>(),
            workspace[1].data_ptr<float>(),
            workspace[2].data_ptr<float>(),
            workspace[3].data_ptr<float>(),
            workspace[4].data_ptr<float>(),
            workspace[5].data_ptr<float>(),
            s.grad_ctx,
            solver
        );

        carriers.fvx_next.copy_(forward.vx_t);
        carriers.fvy_next.copy_(forward.vy_t);
        carriers.fvz_next.copy_(forward.vz_t);

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

    // ---- seg / carrier plumbing (ckpt mode) ----

    static std::vector<torch::Tensor> alloc_seg_buffers(const torch::Tensor& rho,
                                                        int segment_len)
    {
        std::vector<int64_t> seg_shape = rho.sizes().vec();
        seg_shape.insert(seg_shape.begin(), static_cast<int64_t>(segment_len + 1));
        auto seg_vx = torch::zeros(seg_shape, rho.options());
        auto seg_vy = torch::zeros(seg_shape, rho.options());
        auto seg_vz = torch::zeros(seg_shape, rho.options());
        return {seg_vx, seg_vy, seg_vz};
    }

    static void capture_seg(std::vector<torch::Tensor>& seg, Wavefield& forward,
                            int slot)
    {
        seg[0].select(0, slot).copy_(forward.vx_t);
        seg[1].select(0, slot).copy_(forward.vy_t);
        seg[2].select(0, slot).copy_(forward.vz_t);
    }

    static VelPtrs seg_vel_ptrs(const std::vector<torch::Tensor>& seg,
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

    static void store_prev_segment(std::vector<torch::Tensor>& prev,
                                   const std::vector<torch::Tensor>& seg)
    {
        prev[0].copy_(seg[0].select(0, 1));
        prev[1].copy_(seg[1].select(0, 1));
        prev[2].copy_(seg[2].select(0, 1));
    }
};

} // namespace elastic_tti_sg2d
