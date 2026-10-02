// Driver traits for the 2-D visco-elastic (GSLS) equation: the per-equation
// half of the staggered-family skeleton in ``common/sg_driver.cuh``, written
// as deltas against elastic2d (the family's reference; see the property list
// at the top of ../elastic2d/driver_traits.cuh -- everything not named here
// is inherited from it unchanged):
//   * models: the PREPARED set [lam_U, mu_U, rho, lam_U + 2 mu_U, P_0..P_{L-1},
//     M_0..M_{L-1}] (ViscoElastic.prepare_models) -- rho third, so elastic2d's
//     body-force / receiver rho corrections (grads[2], models.rho) apply as
//     they are -- plus eq_aux = one host float32 buffer holding the
//     trapezoidal memory weights [a_0..a_{L-1}, c_0..c_{L-1}]
//     (ViscoElastic.c_eq_aux).  No derived models;
//   * wavefields: [vx, vz, sxx, szz, sxz | 3L memory variables r_xx_l, r_zz_l,
//     r_xz_l | the 10 elastic CPML memories], so CKPT_NVAR = ADJ_WF_COUNT =
//     15 + 3L: the traits are a template on L and the entries dispatch on the
//     model count (4 + 2L);
//   * stress_substep is the visco kernel (memory update + the surface solve of
//     the normal strain rate); every other forward launch, the source /
//     receiver plumbing and the three stencil transposes of the adjoint
//     (stress apply, velocity prepare/apply) are elastic2d's;
//   * the surface solve's strain rates are forward state the gradient reads:
//     the full-mode u_allt frame is [vx | vz | strip] (allt_shape; strip =
//     (B, 2nx + 2nz): z-low row, z-high row, x-low column, x-high column),
//     the chunked backward keeps a strip history next to the vx / vz ones and
//     the recursive one a strip carrier (N_VEL = 3); the replay state writes
//     its strip into a scratch slot of checkpoint_replay;
//   * imaging: the gradient needs the memory-variable multipliers and the
//     transpose of the surface solve -- the stress-adjoint prepare itself -- so
//     it is FUSED there in every mode.  image_standalone launches that prepare
//     with the gradient (it mutates the adjoint: FS zeroing, the memory step,
//     the q* scratch, the CPML memories) and plain_adjoint_step runs the
//     remaining three launches.  The skeleton calls the two back to back in
//     every mode this driver serves (full at it == 0, chunk and recursive
//     checkpointing at every step), and full_mode_step is the same sequence,
//     so all three modes run one kernel sequence and agree bitwise;
//   * refused: boundary saving (the attenuation is dissipative: no stable
//     reverse reconstruction), stepped / phased / domain-decomposed calls,
//     topography, APM, ADCIG.
//
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/sg_driver.cuh.
#pragma once

#include <cuda_runtime.h>

#include <cstring>
#include <type_traits>
#include <vector>

#include "../../../core/runner.h"
#include "kernels.cuh"
#include "../elastic2d/driver_traits.cuh"

namespace visco_elastic2d {

// Relaxation mechanisms in a prepared model list [lam, mu, rho, lam2mu, P.., M..].
inline int n_sls_of(const BufList& models)
{
    const int64_t n = models.size();
    SWEEP_CHECK(n >= 6 && (n - 4) % 2 == 0,
                "visco_elastic2d expects the prepared models [lam, mu, rho, lam2mu, "
                "P_0..P_{L-1}, M_0..M_{L-1}] (ViscoElastic.prepare_models), got ", n);
    const int L = static_cast<int>((n - 4) / 2);
    SWEEP_CHECK(L >= 1 && L <= VISCO_ELASTIC2D_MAX_SLS,
                "visco_elastic2d supports 1..", VISCO_ELASTIC2D_MAX_SLS,
                " relaxation mechanisms, got ", L);
    return L;
}

// elastic2d's view plus the memory variables and the surface-strip target of
// the stress kernel (the replay scratch; null in the forward and the adjoint).
struct ViscoWfView : ElasticWavefieldPointer {
    ViscoElastic2dMemory mem{};
    float* strip = nullptr;
};

// elastic2d's 15-tensor bind plus the 3L memory variables; ``all`` keeps the
// bound order, which is the checkpoint order (cuda_layout.checkpoint_slot_axes).
struct ViscoWavefield : ElasticWavefieldTensor {
    std::vector<Buf> memory;
    std::vector<Buf> all;
    Buf strip;   // (B, 2nx + 2nz) replay scratch; undefined elsewhere

    void bind_visco(const std::vector<Buf>& tensors, int L, const char* who)
    {
        const size_t n_mem = static_cast<size_t>(3 * L);
        SWEEP_CHECK(tensors.size() == 15 + n_mem,
                    who, " expects ", 15 + n_mem, " bound wavefields [vx, vz, sxx, szz, "
                    "sxz, 3L memory variables, 10 CPML memories] (cuda_layout.base_nvar + "
                    "pml_nvar), got ", tensors.size());
        for (size_t i = 5; i < 5 + n_mem; ++i)
            SWEEP_CHECK(same_shape(tensors[i], tensors[0]) && tensors[i].is_contiguous(),
                        who, ": memory variable ", i - 5, " must be a contiguous full grid "
                        "shaped like vx");
        std::vector<Buf> el(tensors.begin(), tensors.begin() + 5);
        el.insert(el.end(), tensors.begin() + 5 + n_mem, tensors.end());
        bind(el, true);
        memory.assign(tensors.begin() + 5, tensors.begin() + 5 + n_mem);
        all = tensors;
    }

    ViscoElastic2dMemory memory_view() const
    {
        ViscoElastic2dMemory v{};
        const int L = static_cast<int>(memory.size() / 3);
        for (int l = 0; l < L; ++l) {
            v.rxx[l] = memory[3 * l + 0].data_ptr<float>();
            v.rzz[l] = memory[3 * l + 1].data_ptr<float>();
            v.rxz[l] = memory[3 * l + 2].data_ptr<float>();
        }
        return v;
    }

    // Hide elastic2d's 15-tensor lists: snapshots and zeroing cover the memory
    // variables too.
    std::vector<Buf> checkpoint_tensors() const { return all; }
    std::vector<Buf> state_tensors() const { return all; }
};

template <int L>
struct DriverL : elastic2d::Driver {
    static_assert(L >= 1 && L <= VISCO_ELASTIC2D_MAX_SLS, "1..4 relaxation mechanisms");
    using Base = elastic2d::Driver;

    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    // ===================================================================== //
    static constexpr const char* NAME = "visco_elastic2d";
    static constexpr int CKPT_NVAR = 15 + 3 * L;
    static constexpr const char* CKPT_COUNT_MSG =
        "visco_elastic2d checkpointing expects 15 + 3 * n_sls checkpoint tensors";
    static constexpr const char* CKPT_RECURSIVE_COUNT_MSG =
        "visco_elastic2d recursive checkpointing expects 15 + 3 * n_sls checkpoint tensors";
    static constexpr int ADJ_WF_COUNT = CKPT_NVAR;
    static constexpr int CKPT_STATE_COUNT = CKPT_NVAR;
    // vx, vz and the surface strip: the histories / carriers the imaging reads.
    static constexpr int N_VEL = 3;
    // checkpoint_replay: [0] the replay's strip scratch (both checkpoint modes);
    // [1-3] the chunked mode's vx / vz / strip histories.
    static constexpr int REPLAY_STRIP_SCRATCH = 0;
    static constexpr int REPLAY_SEG_FIRST = 1;

    using Wavefield = ViscoWavefield;
    using WfView = ViscoWfView;

    struct Models {
        Buf lam, mu, rho, lam2mu;
        Buf P[VISCO_ELASTIC2D_MAX_SLS], M[VISCO_ELASTIC2D_MAX_SLS];
    };

    template <class P>
    static Models parse_models(const P& p)
    {
        SWEEP_CHECK(n_sls_of(p.models) == L,
                    "visco_elastic2d: the ", L, "-mechanism driver was handed ",
                    p.models.size(), " models");
        const Buf& like = p.models[0];
        SWEEP_CHECK(like.dim() == 4, "visco_elastic2d models must be (B, 1, nz, nx)");
        for (int64_t i = 0; i < p.models.size(); ++i) {
            pool_slot_checked(p.models, static_cast<int>(i), like, "visco_elastic2d models");
            SWEEP_CHECK(p.models[i].is_contiguous(), "visco_elastic2d models must be contiguous");
        }
        Models m;
        m.lam = p.models[0];
        m.mu = p.models[1];
        m.rho = p.models[2];
        m.lam2mu = p.models[3];
        for (int l = 0; l < L; ++l) {
            m.P[l] = p.models[4 + l];
            m.M[l] = p.models[4 + L + l];
        }
        return m;
    }

    // elastic2d's State (its models.rho / lambda / mu are what the inherited
    // velocity kernels and rho corrections read) plus the visco coefficients,
    // the CPML profiles and -- backward only -- the stress-adjoint scratch the
    // fused prepare writes (adjoint_workspace[0-3], the same buffers
    // make_workspace binds for the apply).
    struct State : Base::State {
        ViscoElastic2dModel m{};
        ElasticCPMLPointer cpml{};
        float* qxx = nullptr;
        float* qzz = nullptr;
        float* qxz = nullptr;
        float* qzx = nullptr;
        long nv = 0;   // B * nz * nx: one velocity component of a u_allt frame
    };

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d, const Models& models,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        State s;
        s.models.rho = models.rho;
        s.models.lambda = models.lam;
        s.models.mu = models.mu;
        s.grad_ctx = SGradParam{1, 0, d.nx, p.M, p.grad_coes.template data_ptr<float>(),
                                p.spacing[0], 0.f, p.spacing[1]};
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        s.nx = d.nx;
        s.nz = d.nz;
        s.B = d.B;
        s.nv = static_cast<long>(d.B) * d.nz * d.nx;

        SWEEP_CHECK(p.eq_aux.size() == 1 && p.eq_aux[0].defined() && !p.eq_aux[0].is_cuda() &&
                    p.eq_aux[0].dtype() == BoundaryDtype::FP32 && p.eq_aux[0].is_contiguous() &&
                    p.eq_aux[0].numel() == 2 * L,
                    "visco_elastic2d expects eq_aux = {[a_0..a_{L-1}, c_0..c_{L-1}]} as one "
                    "host float32 buffer (ViscoElastic.c_eq_aux)");
        const float* w = p.eq_aux[0].template data_ptr<float>();
        s.m.lam = models.lam.template data_ptr<float>();
        s.m.mu = models.mu.template data_ptr<float>();
        s.m.lam2mu = models.lam2mu.template data_ptr<float>();
        for (int l = 0; l < L; ++l) {
            s.m.P[l] = models.P[l].template data_ptr<float>();
            s.m.Mm[l] = models.M[l].template data_ptr<float>();
            s.m.a[l] = w[l];
            s.m.c[l] = w[L + l];
        }
        s.m.L = L;

        ElasticCPMLTensor cpml;
        cpml.bind(p.pml_vals, 2);
        s.cpml = cpml.view();

        if constexpr (std::is_same_v<P, BackwardInputCore>) {
            ElasticAdjointWorkspaceTensor ws;
            bind_adjoint_workspace_required(ws, p.adjoint_workspace, 2,
                                            "visco_elastic2d backward");
            s.qxx = ws.qxx_t.data_ptr<float>();
            s.qzz = ws.qzz_t.data_ptr<float>();
            s.qxz = ws.qxz_t.data_ptr<float>();
            s.qzx = ws.qzx_t.data_ptr<float>();
        }
        return s;
    }

    template <class P>
    static void check_supported(const P& p, const char* where)
    {
        SWEEP_CHECK(!p.has_topo && !p.use_apm,
                    "ViscoElastic (", where, "): topography / APM is not supported");
        SWEEP_CHECK(p.cut_face_mask == 0 && p.step_phase == 0,
                    "ViscoElastic (", where, "): domain decomposition / phased calls are "
                    "not supported");
    }

    static void validate_forward(const ForwardInputCore& p)
    {
        check_supported(p, "forward");
        SWEEP_CHECK(p.it_begin == 0 && (p.it_end < 0 || p.it_end == static_cast<int>(p.nt)),
                    "ViscoElastic (forward): stepped calls are not supported");
        SWEEP_CHECK(!p.use_boundary_saving,
                    "ViscoElastic does not support boundary saving (the attenuation is "
                    "dissipative: no stable reverse reconstruction); use "
                    "memory=Full() or memory=Ckpt(...) (sweep.propagator.options)");
    }

    static void validate_backward(const BackwardInputCore& p, const char* mode)
    {
        const int n_sls = n_sls_of(p.models);
        SWEEP_CHECK(n_sls == L, "visco_elastic2d: the ", L, "-mechanism driver was handed ",
                    n_sls, " mechanisms");
        check_supported(p, mode);
        SWEEP_CHECK(!p.bw_stepped(), "ViscoElastic (", mode, "): stepped backward is not supported");
        SWEEP_CHECK(!p.compute_adcig, "ViscoElastic (", mode, "): ADCIG is not supported");
        if (std::strcmp(mode, "full") == 0) {
            const Buf& m0 = p.models[0];
            const int64_t B = m0.size(0) * m0.size(1), nz = m0.size(2), nx = m0.size(3);
            const int64_t frame = 2 * B * nz * nx + B * visco_elastic2d_strip_len(nx, nz);
            SWEEP_CHECK(p.u_forward.defined() && p.u_forward.dim() == 2 &&
                        p.u_forward.size(0) == static_cast<int64_t>(p.nt) &&
                        p.u_forward.size(1) == frame && p.u_forward.is_contiguous(),
                        "ViscoElastic backward: u_forward must be the (nt, 2 B nz nx + B (2nx + 2nz)) "
                        "[vx | vz | strip] history of the forward (cuda_layout.save_all_shape)");
        }
    }

    // [vx | vz | strip] per step: the surface solve's strain rates ride with
    // the velocities (see the header).
    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        const int64_t B = d.B;
        return {nt, 2 * B * d.nz * d.nx + B * visco_elastic2d_strip_len(d.nx, d.nz)};
    }

    static WfView view(Wavefield& wf)
    {
        WfView v;
        static_cast<ElasticWavefieldPointer&>(v) = wf.ElasticWavefieldTensor::view();
        v.mem = wf.memory_view();
        v.strip = ptr_or_null(wf.strip);
        return v;
    }

    // ===================================================================== //
    // [2] FORWARD
    // ===================================================================== //
    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p, const Buf&)
    {
        wf.bind_visco(p.wavefields, L, "visco_elastic2d/forward wavefields");
    }

    // u_this (full-mode forward) receives [vx | vz | strip] of this step; the
    // replay writes its strip into the bound scratch; otherwise nothing.
    static void stress_substep(const State& s, WfView& wf,
                               ElasticCPMLPointer cpml_view,
                               const SolverContext& solver, float* u_this_t)
    {
        float* strip = u_this_t ? u_this_t + 2 * s.nv : wf.strip;
        LAUNCH_VISCO_ELASTIC2D(
            visco_elastic2d_stress_kernel, s.order,
            s.launch_config.grid, s.launch_config.block,
            static_cast<const ElasticWavefieldPointer&>(wf), wf.mem, s.m,
            u_this_t, strip, s.grad_ctx, cpml_view, solver);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE
    // ===================================================================== //
    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInputCore& p, const Buf&)
    {
        wf.bind_visco(p.adjoint_wavefields, L, "visco_elastic2d/backward adjoint_wavefields");
    }

    // One accumulator per prepared model, in model order (grads[2] = rho).
    static void bind_grads(const BackwardInputCore& p, std::vector<Buf>& grads)
    {
        SWEEP_CHECK(p.grads_out.size() == 4 + 2 * L,
                    "visco_elastic2d/backward requires the propagator-bound grads_out "
                    "(one per prepared model, no wavelet slot), got ", p.grads_out.size());
        grads.assign(p.grads_out.begin(), p.grads_out.end());
    }

    struct VelPtrs : Base::VelPtrs {
        const float* strip_now = nullptr;   // the solved surface strain rates of step it
    };

    static VelPtrs vel_ptrs_from_u_forward(const BackwardInputCore& p, int it,
                                           const Buf& zero_velocity)
    {
        const long nv = static_cast<long>(zero_velocity.numel());   // B * nz * nx
        const float* now = p.u_forward.select(0, it).data_ptr<float>();
        VelPtrs v;
        v.vx_now = now;
        v.vz_now = now + nv;
        v.strip_now = now + 2 * nv;
        if (it + 1 < static_cast<int>(p.nt)) {
            const float* next = p.u_forward.select(0, it + 1).data_ptr<float>();
            v.vx_next = next;
            v.vz_next = next + nv;
        } else {
            v.vx_next = zero_velocity.data_ptr<float>();
            v.vz_next = zero_velocity.data_ptr<float>();
        }
        return v;
    }

    // The fused stress-adjoint prepare WITH this step's gradient (see the
    // header): mutates the adjoint, so it is always followed by
    // plain_adjoint_step (or by nothing, at it == 0).
    static void image_standalone(const State& s, const SolverContext& solver,
                                 WfView& adj_view, const VelPtrs& v,
                                 std::vector<Buf>& grads)
    {
        ViscoElastic2dGrad g{};
        g.fvx = v.vx_now;
        g.fvz = v.vz_now;
        g.fvx_next = v.vx_next;
        g.fvz_next = v.vz_next;
        g.fs_strain = v.strip_now;
        g.rho = s.models.rho.template data_ptr<float>();
        g.g_lam = grads[0].data_ptr<float>();
        g.g_mu = grads[1].data_ptr<float>();
        g.g_rho = grads[2].data_ptr<float>();
        g.g_lam2mu = grads[3].data_ptr<float>();
        for (int l = 0; l < L; ++l) {
            g.g_P[l] = grads[4 + l].data_ptr<float>();
            g.g_M[l] = grads[4 + L + l].data_ptr<float>();
        }
        LAUNCH_VISCO_ELASTIC2D(
            visco_elastic2d_stress_adjoint_prepare, s.order,
            s.launch_config.grid, s.launch_config.block,
            static_cast<const ElasticWavefieldPointer&>(adj_view), adj_view.mem, s.m,
            s.cpml, solver, s.qxx, s.qzz, s.qxz, s.qzx, s.grad_ctx, g);
    }

    // The remaining three adjoint launches after image_standalone's prepare.
    static void plain_adjoint_step(const State& s, const SolverContext& solver,
                                   Wavefield& adjoint, Workspace& workspace,
                                   ElasticCPMLPointer cpml_view)
    {
        WfView adj_view = view(adjoint);
        stress_adjoint_apply(s, solver, adj_view, workspace);
        velocity_adjoint_half(s, solver, adj_view, workspace, cpml_view);
    }

    // Same kernel sequence as the checkpoint modes' image_standalone /
    // fix_rho_grad_at_receivers / plain_adjoint_step.
    static void full_mode_step(const State& s, const SolverContext& solver,
                               Wavefield& adjoint, Workspace& workspace,
                               ElasticCPMLPointer cpml_view,
                               const VelPtrs& v,
                               std::vector<Buf>& grads,
                               const BackwardInputCore& p,
                               IntSpan receiver_fields,
                               int it, int adjoint_nsrc)
    {
        WfView adj_view = view(adjoint);
        image_standalone(s, solver, adj_view, v, grads);
        fix_rho_grad_at_receivers(s, solver, grads, v, p, receiver_fields, it, adjoint_nsrc);
        plain_adjoint_step(s, solver, adjoint, workspace, cpml_view);
    }

    // ===================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING
    // ===================================================================== //
    static void bind_or_alloc_recon_ckpt(Wavefield& forward, const BackwardInputCore& p,
                                         const Buf& vp)
    {
        SWEEP_CHECK((int)p.forward_wavefields.size() >= CKPT_STATE_COUNT,
                    "visco_elastic2d/ckpt requires the propagator-bound replay state "
                    "(the forward slot list: ", CKPT_STATE_COUNT, " tensors), got ",
                    p.forward_wavefields.size());
        forward.bind_visco(wavefield_set(p.forward_wavefields, 0, CKPT_STATE_COUNT,
                                         "visco_elastic2d ckpt replay state"),
                           L, "visco_elastic2d ckpt replay state");
        forward.strip = pool_required(p.checkpoint_replay, REPLAY_STRIP_SCRATCH,
                                      {vp.size(0) * vp.size(1),
                                       visco_elastic2d_strip_len(vp.size(3), vp.size(2))},
                                      "checkpoint_replay");
    }

    // checkpoint_replay [1-3]: vx / vz (max_rows, B, 1, nz, nx) and the strip
    // (max_rows, B, 2nx + 2nz) histories of one chunk.
    static std::vector<Buf> seg_buffers(const BackwardInputCore& p, const Buf& vp, int max_rows)
    {
        SWEEP_CHECK((int)p.checkpoint_replay.size() >= REPLAY_SEG_FIRST + N_VEL,
                    "visco_elastic2d/ckpt requires the propagator-bound checkpoint_replay "
                    "(cuda_layout.checkpoint_replay_shapes: strip scratch + vx / vz / strip "
                    "histories), got ", p.checkpoint_replay.size());
        const int64_t B = vp.size(0) * vp.size(1), nz = vp.size(2), nx = vp.size(3);
        const int64_t rows = max_rows;
        return {
            pool_required(p.checkpoint_replay, REPLAY_SEG_FIRST + 0, {rows, B, 1, nz, nx},
                          "checkpoint_replay"),
            pool_required(p.checkpoint_replay, REPLAY_SEG_FIRST + 1, {rows, B, 1, nz, nx},
                          "checkpoint_replay"),
            pool_required(p.checkpoint_replay, REPLAY_SEG_FIRST + 2,
                          {rows, B, visco_elastic2d_strip_len(nx, nz)}, "checkpoint_replay"),
        };
    }

    static void save_seg_velocities(std::vector<Buf>& seg, Wavefield& forward, int slot)
    {
        copy_tensor_cuda_async(seg[0].select(0, slot), forward.vx_t);
        copy_tensor_cuda_async(seg[1].select(0, slot), forward.vz_t);
        copy_tensor_cuda_async(seg[2].select(0, slot), forward.strip);
    }

    static VelPtrs vel_ptrs_from_seg(const std::vector<Buf>& seg,
                                     int now_offset, int next_offset,
                                     const std::vector<Buf>& next_segment_v)
    {
        VelPtrs v;
        static_cast<Base::VelPtrs&>(v) =
            Base::vel_ptrs_from_seg(seg, now_offset, next_offset, next_segment_v);
        v.strip_now = seg[2].select(0, now_offset).data_ptr<float>();
        return v;
    }

    // (export_seg_next_v is elastic2d's: the older chunk's imaging reads the
    // younger chunk's v(start + 1) only -- never its strip.)

    // The strip carrier is a model-shaped slot (sg_carriers); the strip fills
    // its head.
    static void capture_velocities(std::vector<Buf>& v, Wavefield& forward)
    {
        copy_tensor_cuda_async(v[0], forward.vx_t);
        copy_tensor_cuda_async(v[1], forward.vz_t);
        copy_tensor_cuda_async(v[2].view({-1}).narrow(0, 0, forward.strip.numel()),
                               forward.strip);
    }

    static VelPtrs vel_ptrs_from_carriers(const std::vector<Buf>& current_v,
                                          const std::vector<Buf>& next_v)
    {
        VelPtrs v;
        static_cast<Base::VelPtrs&>(v) = Base::vel_ptrs_from_carriers(current_v, next_v);
        v.strip_now = current_v[2].data_ptr<float>();
        return v;
    }
};

} // namespace visco_elastic2d
