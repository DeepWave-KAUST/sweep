// Driver traits for the 2-D variable-density VRZ equation: the per-equation
// half of the acoustic-family skeleton in ``common/eq_driver.cuh``.
// Line-faithful transcription of the hand-written forward/backward/backward_bs
// drivers; physics kernels untouched.  The unified chunk/recursive checkpoint
// backward keeps its hand-written form in backward.cu (it is a linear
// segment sweep, not the acoustic bisection this skeleton templates).
//
// Deltas from acoustic2d (the acoustic-family REFERENCE EQUATION; anything
// not listed here is the same as in acoustic2d/driver_traits.cuh):
//   * two models, [vp, z] (make_state checks the count); State keeps z and the derived inv_z = 1/z alive as tensors beside their raw pointers;
//   * TANGENT_PAD = 1: the boundary strips carry a tangential pad of M (matches the Python boundary_tangent_pad = so//2 and the persistent int8 buffers' per-step stride);
//   * ADJ_WF_COUNT = 9 (u triple + psi quad + psin pair, no zeta double-buffer): rotate_adjoint_buffers rotates via swap_pml instead of swap_aux, and bind_or_alloc_adjoint allocates with double_buffer_psi = true;
//   * ADCIG_IN_FULL_MODES = false, HAS_FUSED_FULL_IMG = false, BS_HAS_IT0_ADJOINT_TAIL = false (no RTM/ADCIG kernels, per-step gradient with no lag fusion, bs floor is it == 1);
//   * BwdWorkspace is non-empty: the time-invariant adjoint coefficients C0/Cx/Cz and the split-gradient scratch c_x/c_z/e_x/e_z; the seven grids come from the Python-bound p.adjoint_workspace (WorkspaceSlot: [0-3]=c_x,c_z,e_x,e_z, [4-6]=C0,Cx,Cz), which AcousticVRZ.cuda_layout.backward_workspace_shapes declares in every memory mode, so the binding is required; make_bwd_workspace zeroes the adjoint wavefield state and runs BUILD_VRZ_ADJOINT_COEFFS once;
//   * validate_forward checks 6 checkpoint tensors and a 1-D checkpoint_steps; validate_backward checks u_last_two (bs) or a (nt, 5, B, 1, nz, nx) u_forward (full);
//   * setup_ctx and init_aux_slabs are empty: no per-edge free surface, no topography, legacy full-grid CPML aux;
//   * allt_shape = (nt, 5, B, 1, nz, nx) (u, psix, psiz, zetax, zetaz), filled by capture_allt after the swap (rotate_buffers); the in-kernel u_thist stays disabled (false, nullptr);
//   * save_width = M + 1 regardless of abcn; boundary save/restore offset -M (strips sit M inside the pad) instead of 0;
//   * launch_step_range refuses sub-ranges (the kernels ignore ctx.x_base / x_limit), so there are no phase-split strips and no air-clear prepass;
//   * grads = the two model gradients {grad_vp, grad_z} (grads_out slot 0, the wavelet, is unused): no grad_wavelet (accumulate_source_grad is empty), no illumination/ADCIG (rtm_out_full / rtm_out_bs return nullptr, fused_grad_ptr is nullptr, bs_rtm_tap is empty, pack_outputs sets grads only);
//   * u_forward_ptr = the u slice u_forward.select(0, it)[0];
//   * adjoint_step = ACOUSTIC_VRZ2D_ADJOINT_FUSED with the C0/Cx/Cz coefficients; inject_adjoint_source injects the NEGATED residual (add_source_signed, sign -1, straight from p.adjoint_source -- no negated copy is built);
//   * image_step = CALCULATE_GRAD_VRZ2D_AUTO (two gradients, split scratch from the workspace; returns early without grads), no RTM kernel;
//   * seed_reconstruction also zeroes u_next, and its set_boundary_zeros calls do not pass the cut_mask;
//   * bs_recon_step order: ACOUSTIC_VRZ2D_NOPML -> forward-source add_source -> restore_backward_2d -> forward.swap() -> CALCULATE_GRAD_VRZ2D_AUTO on the post-swap u_now (acoustic2d: NOPML -> restore -> strip-source un-injection -> band imaging -> inject -> swap; VRZ needs no un-injection -- its inject lands before the restore, so a restored strip source cell is the saved true value);
//   * no ckpt/recursive hooks (section [5] is empty): backward.cu keeps the hand-written chunk/recursive checkpoint backward.
//
// What the skeleton ADDS for this equation (dormant on legacy calls, all
// defaults reproduce the monolithic drivers bit-for-bit): the stepped
// it_begin/it_end range on forward and on the full/bs backwards, Python-bound
// record_out/wavefields/grads_out continuation buffers -- the prerequisites
// domain decomposition needs.  NOT added: phase-split (these kernels are not
// ranged -- launch_step_range refuses sub-ranges loudly) and the VRZ coupling
// exchange phases of the 3-D sibling's backward (declared absent Python-side).
//
// Hook timing: see the HOOK TIMING MAP at the top of ../../common/eq_driver.cuh.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>

#include "kernels.cuh"
#include "../../common/common.cuh"
#include "../../common/context.h"
#include "../../common/acoustic.h"
#include "../../common/cudautils.h"
#include "../../common/derived_models.h"
#include "../../common/boundarysaver.cuh"
#include "../../common/boundary_runtime.cuh"
#include "../../../core/input_core.h"
#include "../../../core/outputs.h"
#include "../../common/eq_driver.cuh"
#include "../../launch/config.h"
#include "../../operators/laplace.cuh"
#include "../../operators/gradient.cuh"

namespace acoustic_vrz2d {

struct Driver {
    // ===================================================================== //
    // [1] IDENTITY & SHARED PLUMBING
    //     Constants, type aliases, State/BwdWorkspace/BsScratch types with
    //     their factories, validation, and allocation helpers.
    //     Prologue call order (timing map): validate_forward / (backward:
    //     validate_backward + bind_backward_outputs + rtm gate),
    //     bind_or_alloc_* wavefields, alloc_cpml, setup_ctx, init_aux_slabs,
    //     make_state, make_bwd_workspace.
    // ===================================================================== //

    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "acoustic_vrz2d";
    static constexpr int CKPT_NVAR = 6;
    static constexpr int BS_NVAR = 1;
    static constexpr int BS_LAST_TWO_NVAR = 2;
    // Boundary strips carry a tangential pad of M (matches the Python
    // boundary_tangent_pad = so//2): without it the internal FP32 staging is
    // smaller than the persistent int8 buffers' per-step stride and
    // quantize_step reads past it.
    static constexpr int TANGENT_PAD = 1;
    static constexpr int CUT_MASK_BITS = 0xF;
    static constexpr const char* CUT_MASK_DESC = "bits 0..3 (x_lo, x_hi, z_lo, z_hi)";
    // u triple + psi quad + psin pair: the VRZ adjoint rotates via swap_pml
    // (no zeta double-buffer, unlike plain acoustic's 11).
    static constexpr int ADJ_WF_COUNT = 9;
    static constexpr int RECON_WF_COUNT = 3;
    static constexpr bool ADCIG_IN_FULL_MODES = false;   // no RTM/ADCIG kernels
    static constexpr bool HAS_FUSED_FULL_IMG = false;    // per-step grad, no lag fusion
    static constexpr bool BS_HAS_IT0_ADJOINT_TAIL = false;        // bs floor is it == 1

    using Wavefield = AcousticWavefieldTensor;
    using CPML = AcousticCPMLTensor;

    struct State {
        Buf vp_t, z_t, inv_z_t;   // inv_z derived; kept alive here
        const float* vp;
        const float* z;
        const float* inv_z;
        LaplaceParam lap_ctx;
        GradParam grad_ctx;
        GradParam grad_ctx_x;
        GradParam grad_ctx_z;
        fdtd::LaunchConfig launch_config;
        fdtd::LaunchConfig source_config;
        fdtd::LaunchConfig record_config;
        int order;
        int M;
        int nx, nz, B;
    };

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d,
                            const SolverContext& /*ctx*/,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        SWEEP_CHECK(p.models.size() == 2,
                    "AcousticVRZ CUDA driver expects models [vp, z]");
        float dx = p.spacing[0];
        float dz = p.spacing[1];
        State s;
        s.vp_t = p.models[0];
        s.z_t = p.models[1];
        s.inv_z_t = derived::reciprocal(p, s.z_t, "acoustic_vrz2d::make_state");
        s.vp = s.vp_t.template data_ptr<float>();
        s.z = s.z_t.template data_ptr<float>();
        s.inv_z = s.inv_z_t.template data_ptr<float>();
        s.lap_ctx = LaplaceParam{d.nx, 1, p.M, p.lap_coes.template data_ptr<float>(), dx, 0.f, dz};
        s.grad_ctx = GradParam{1, 0, d.nx, p.M, p.grad_coes.template data_ptr<float>(), dx, 0.f, dz};
        s.grad_ctx_x = GradParam{1, 0, 0, p.M, p.grad_coes.template data_ptr<float>(), dx, 0.f, 0.f};
        s.grad_ctx_z = GradParam{1, 0, 0, p.M, p.grad_coes.template data_ptr<float>(), dz, 0.f, 0.f};
        s.launch_config = launch_config;
        s.source_config = source_config;
        s.record_config = record_config;
        s.order = eqdrv::stencil_order(p.M);
        s.M = p.M;
        s.nx = d.nx;
        s.nz = d.nz;
        s.B = d.B;
        return s;
    }

    struct BwdWorkspace {
        Buf C0, Cx, Cz;           // time-invariant adjoint coeffs
        Buf c_x, c_z, e_x, e_z;   // split gradient scratch (order>=6)
    };

    // Layout of p.adjoint_workspace, declared on the Python side by
    // AcousticVRZ.cuda_layout (backward_workspace_shapes, bound in all four
    // memory modes: full and bs run through this skeleton, ckpt and recursive
    // through the hand-written backward_ckpt in backward.cu, which takes the
    // same seven slots through this enum).  One padded grid per
    // shot each, the 2-D cut of the 3-D sibling's WorkspaceSlot
    // (acoustic_vrz3d/backward.cu) in the order the DD runner binds the
    // family (coupling grids first, adjoint coefficients last): the four c/e
    // coupling grids of the split gradient, then the three time-invariant
    // adjoint coefficients C0/Cx/Cz.  The pool is zero at backward entry --
    // the propagator zeroes it before every gradient-bearing forward -- which
    // is what a fresh zeros_like started from and what keeps every slot's
    // M-wide halo at zero: build_vrz_adjoint_coeffs (once, below) and
    // build_vrz_grad_fields (every imaging step, order >= 6 only) write the
    // interior, while the fused adjoint and calculate_grad_vrz2d read those
    // grids through an M-wide stencil that reaches into the halo.
    enum WorkspaceSlot : int {
        C_X = 0, C_Z, E_X, E_Z,      // λ·vp·∂p (c) and λ·vp²·z·∂p (e) coupling scratch
        COEF_C0, COEF_CX, COEF_CZ,   // vp², ∂ₓb·κ, ∂_z b·κ adjoint coefficients
        N_SLOTS                      // 7
    };

    // Exactly N_SLOTS.  The propagator allocates the pool for every
    // gradient-bearing forward (_ensure_adjoint_workspace_buffers over
    // AcousticVRZ.cuda_layout.backward_workspace_shapes, which returns these
    // seven grids in all four memory modes), so an empty or differently sized
    // pool means the Python declaration drifted.
    static BufList workspace_slots(const BackwardInputCore& p)
    {
        SWEEP_CHECK(p.adjoint_workspace.size() == N_SLOTS,
                    "acoustic_vrz2d/backward requires the propagator-bound "
                    "adjoint_workspace (cuda_layout.backward_workspace_shapes): ",
                    static_cast<int>(N_SLOTS), " tensors ([0-3]=c_x,c_z,e_x,e_z coupling, "
                    "[4-6]=C0,Cx,Cz adjoint coeffs), got ", p.adjoint_workspace.size());
        return p.adjoint_workspace;
    }

    // (helper: fired inside make_bwd_workspace)
    static void zero_wavefield_state(Wavefield& wf)
    {
        zero_tensor_device_async(wf.u_prev_t);
        zero_tensor_device_async(wf.u_now_t);
        zero_tensor_device_async(wf.u_next_t);
        zero_tensor_device_async(wf.psix_t);
        zero_tensor_device_async(wf.psiz_t);
        zero_tensor_device_async(wf.zetax_t);
        zero_tensor_device_async(wf.zetaz_t);
    }

    static BwdWorkspace make_bwd_workspace(const BackwardInputCore& p,
                                           const State& s,
                                           const SolverContext& ctx,
                                           Wavefield& adjoint)
    {
        zero_wavefield_state(adjoint);
        BwdWorkspace ws;
        // The Python-bound pool slots (WorkspaceSlot above); Buf
        // copies share storage, so the data_ptr() the kernels take hits the
        // bound buffer.
        const auto& pool = workspace_slots(p);
        ws.C0  = pool_required(pool, COEF_C0, s.vp_t, "adjoint_workspace");   // vp²       (time-invariant adjoint coeffs)
        ws.Cx  = pool_required(pool, COEF_CX, s.vp_t, "adjoint_workspace");   // ∂ₓb·κ
        ws.Cz  = pool_required(pool, COEF_CZ, s.vp_t, "adjoint_workspace");   // ∂_z b·κ
        ws.c_x = pool_required(pool, C_X, s.vp_t, "adjoint_workspace");       // split gradient scratch (order>=6 path)
        ws.c_z = pool_required(pool, C_Z, s.vp_t, "adjoint_workspace");
        ws.e_x = pool_required(pool, E_X, s.vp_t, "adjoint_workspace");
        ws.e_z = pool_required(pool, E_Z, s.vp_t, "adjoint_workspace");
        // Time-invariant adjoint transpose coefficients (vp², ∂ₓb·κ, ∂_z b·κ),
        // computed once so the fused adjoint kernel only multiplies by λ per step.
        BUILD_VRZ_ADJOINT_COEFFS(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            s.vp,
            s.z,
            s.inv_z,
            ws.C0.data_ptr<float>(),
            ws.Cx.data_ptr<float>(),
            ws.Cz.data_ptr<float>(),
            s.grad_ctx,
            ctx
        );
        return ws;
    }

    // (factory make_bs_scratch lives in section [4])
    struct BsScratch {};

    static void validate_forward(const ForwardInputCore& p)
    {
        if (p.use_checkpoint)
            SWEEP_CHECK(p.checkpoints.size() == 6,
                        "AcousticVRZ checkpointing expects 6 checkpoint tensors");
        if (p.use_recursive_checkpoint) {
            SWEEP_CHECK(p.checkpoint_steps.defined(),
                        "Recursive checkpointing expects checkpoint_steps");
            SWEEP_CHECK(p.checkpoint_steps.dim() == 1, "checkpoint_steps must be 1-D");
        }
    }

    static void validate_backward(const BackwardInputCore& p, bool need_recon)
    {
        if (need_recon) {
            SWEEP_CHECK(p.u_last_two.defined() && p.u_last_two.numel() > 0,
                        "AcousticVRZ backward_bs expects saved last_two wavefields.");
        } else {
            SWEEP_CHECK(p.u_forward.defined() && p.u_forward.numel() > 0,
                        "AcousticVRZ backward expects saved full forward wavefields.");
            SWEEP_CHECK(p.u_forward.dim() == 6 && p.u_forward.size(1) == 5,
                        "AcousticVRZ backward expects forward wavefields with "
                        "shape (nt, 5, B, 1, nz, nx).");
        }
    }

    template <class P>
    static void setup_ctx(SolverContext&, const P&) {}   // no per-edge / topo

    static void init_aux_slabs(SolverContext&, Wavefield&) {}   // legacy full-grid aux

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.bind(p.pml_vals, 2);
    }

    // The record this equation writes: one field, {N, nrec, nt}.  Moved here
    // verbatim from the skeleton -- same expression, same operands.
    static std::vector<int64_t> record_shape(const eqdrv::Dims& d,
                                             const ForwardInputCore& p)
    {
        return {d.N, p.receivers_loc.size(1), static_cast<int64_t>(p.nt)};
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, 5, d.B, 1, d.nz, d.nx};   // u, psix, psiz, zetax, zetaz
    }

    static int save_width(int /*abcn*/, int M) { return M + 1; }

    // ===================================================================== //
    // [2] FORWARD — generic_forward, per it in [it_begin, it_end):
    //     launch_step_range -> save_boundary_fwd -> inject_source_fwd ->
    //     record -> rotate_buffers -> capture_allt; after the loop:
    //     save_last_state.
    // ===================================================================== //

    // The propagator binds the forward wavefield state on EVERY call -- the
    // persistent save_all pool or the per-call transient set (_c.py
    // Wrapper.forward, ``params.wavefields = cp.forward_wavefields``, sized by
    // AcousticVRZ.cuda_layout base_nvar 3 + pml_nvar 6) -- and the stepped / DD
    // drivers rebind the same list, so there is no unbound caller to allocate for.
    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p,
                                      const Buf& /*vp*/)
    {
        SWEEP_CHECK(!p.wavefields.empty(),
                    "acoustic_vrz2d/forward requires the propagator-bound wavefields "
                    "(cuda_layout.base_nvar + pml_nvar = 9 tensors)");
        wf.bind(p.wavefields, 2, true);
    }

    static void launch_step_range(const State& s, const SolverContext& ctx,
                                  int xb, int xe,
                                  AcousticWavefieldPointer view,
                                  bool /*save_all*/, float* /*u_thist*/,
                                  AcousticCPMLPointer cpml)
    {
        // The VRZ2D kernels do not honour ctx.x_base/x_limit; the DD
        // phase-split strips would silently compute the wrong cells.
        SWEEP_CHECK(xb == 0 && xe == s.nx,
                    "acoustic_vrz2d kernels are not ranged: phase-split "
                    "(step_phase) strips are unsupported");
        // History is captured post-swap by capture_allt (tensor copies), so
        // the in-kernel u_this pointer stays disabled like the hand-written
        // driver's (false, nullptr).
        ACOUSTIC_VRZ2D(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            view,
            false,
            nullptr,
            s.vp,
            s.z,
            s.inv_z,
            s.lap_ctx,
            s.grad_ctx,
            s.grad_ctx_x,
            s.grad_ctx_z,
            cpml,
            ctx
        );
    }

    static void save_boundary_fwd(BoundaryRuntime& rt, const State& s,
                                  const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  int it_shifted, int nt_shifted,
                                  const GeneralBoundaryPointer& bs, int save_width)
    {
        rt.save_forward_2d(
            it_shifted,
            nt_shifted,
            view.u_now,
            s.launch_config.grid,
            s.launch_config.block,
            bs,
            save_width,
            -s.M,     // offset: strips sit M inside the pad
            ctx
        );
    }

    // (also used by the ckpt/recursive replay)
    static void inject_source_fwd(const State& s, const SolverContext& ctx,
                                  const AcousticWavefieldPointer& view,
                                  const ForwardInputCore& p, int it, int nsrc)
    {
        add_source<<<s.source_config.grid, s.source_config.block>>>(
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
                       Buf& record, const ForwardInputCore& p,
                       int it, int nrec)
    {
        record_kernel<<<s.record_config.grid, s.record_config.block>>>(
            view.u_next,
            record.data_ptr<float>(),
            p.receivers_loc.data_ptr<int>(),
            it,
            nrec,
            ctx
        );
    }

    static void rotate_buffers(Wavefield& wf) { wf.swap_pml(); }

    static void capture_allt(Buf& u_allt, Wavefield& wf, int it)
    {
        if (!u_allt.defined()) return;
        copy_tensor_cuda_async(u_allt.select(0, it).select(0, 0), wf.u_now_t);
        copy_tensor_cuda_async(u_allt.select(0, it).select(0, 1), wf.psix_t);
        copy_tensor_cuda_async(u_allt.select(0, it).select(0, 2), wf.psiz_t);
        copy_tensor_cuda_async(u_allt.select(0, it).select(0, 3), wf.zetax_t);
        copy_tensor_cuda_async(u_allt.select(0, it).select(0, 4), wf.zetaz_t);
    }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        copy_tensor_cuda_async(saver.last_two.select(1, 0), wf.u_prev_t);
        copy_tensor_cuda_async(saver.last_two.select(1, 1), wf.u_now_t);
    }

    // ===================================================================== //
    // [3] BACKWARD SHARED + FULL MODE — generic_backward, per reverse it:
    //     adjoint_step -> inject_adjoint_source -> rotate_adjoint_buffers ->
    //     accumulate_source_grad -> image_step.
    // ===================================================================== //

    static void bind_backward_outputs(const BackwardInputCore& p,
                                      std::vector<Buf>& grads,
                                      RTMOutputCore& /*illumination*/,
                                      bool /*want_adcig*/)
    {
        // 3-D sibling convention: models.size()+1 slots, slot 0 (wavelet)
        // unused -- VRZ computes no grad_wavelet.  _c.py builds grads_out for
        // every backward (_gradient_buffers from
        // cuda_layout.grads_out_has_wavelet = true plus one slot per model),
        // and the DD runner rebinds the same list, so it is never empty.
        SWEEP_CHECK(p.grads_out.size() == p.models.size() + 1,
                    "acoustic_vrz2d/backward requires the propagator-bound grads_out "
                    "(cuda_layout.grads_out_has_wavelet + one slot per model = "
                    "models.size()+1 tensors, slot 0 = grad_wavelet, unused for VRZ), got ",
                    p.grads_out.size());
        grads = {pool_required(p.grads_out, 1, p.models[0], "grads_out"),
                 pool_required(p.grads_out, 2, p.models[1], "grads_out")};
    }

    static void pack_outputs(BackwardOutputCore& out,
                             std::vector<Buf>& grads,
                             RTMOutputCore& /*illumination*/)
    {
        out.grads = {grads[0], grads[1]};   // {grad_vp, grad_z}
    }

    static RTMOutputCore* rtm_out_full(const BackwardInputCore&, RTMOutputCore&)
    { return nullptr; }

    static float* fused_grad_ptr(std::vector<Buf>&)
    { return nullptr; }   // HAS_FUSED_FULL_IMG == false: never consulted

    static const float* u_forward_ptr(const BackwardInputCore& p, int it)
    {
        return p.u_forward.select(0, it).select(0, 0).data_ptr<float>();
    }

    // _c.py binds cp.adjoint_wavefields on every backward
    // (_ensure_wavefield_buffers allocates them whenever the forward required a
    // gradient: base_nvar 3 + pml_nvar 6 + adjoint_extra_nvar 0 = 9 slots), and
    // the stepped / DD drivers rebind the same list.
    static void bind_or_alloc_adjoint(Wavefield& wf, const BackwardInputCore& p,
                                      const Buf& /*vp*/)
    {
        SWEEP_CHECK(!p.adjoint_wavefields.empty(),
                    "acoustic_vrz2d/backward requires the propagator-bound "
                    "adjoint_wavefields (cuda_layout.base_nvar + pml_nvar + "
                    "adjoint_extra_nvar = 9 tensors)");
        wf.bind(p.adjoint_wavefields, 2, true);
    }

    static void adjoint_step(const State& s, const SolverContext& ctx,
                             AcousticWavefieldPointer adj_view,
                             AcousticCPMLPointer cpml, BwdWorkspace& ws,
                             const float* /*grad_forward_img*/, float* /*grad_out*/)
    {
        ACOUSTIC_VRZ2D_ADJOINT_FUSED(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            adj_view,
            s.vp,
            s.z,
            s.inv_z,
            ws.C0.data_ptr<float>(),
            ws.Cx.data_ptr<float>(),
            ws.Cz.data_ptr<float>(),
            s.lap_ctx,
            s.grad_ctx,
            s.grad_ctx_x,
            s.grad_ctx_z,
            cpml,
            ctx
        );
    }

    static void inject_adjoint_source(const State& s, const SolverContext& ctx,
                                      const AcousticWavefieldPointer& adj_view,
                                      const BackwardInputCore& p, int it, int nsrc,
                                      BwdWorkspace&)
    {
        // The VRZ adjoint injects the NEGATED residual (driver-level sign):
        // sign -1 flips the sample's sign bit inside the kernel, bit-identical
        // to the negated copy this used to inject, with no tensor built.
        add_source_signed<<<s.record_config.grid, s.record_config.block>>>(
            adj_view.u_next,
            p.adjoint_source.data_ptr<float>(),
            p.adjoint_sources_loc.data_ptr<int>(),
            it,
            nsrc,
            -1.0f,
            ctx
        );
    }

    static void rotate_adjoint_buffers(Wavefield& wf)
    {
        wf.swap_pml();   // rotate u AND psi<->psin: race-free adjoint psi
    }

    static void accumulate_source_grad(const State&, const SolverContext&,
                                       Wavefield&, const BackwardInputCore&,
                                       std::vector<Buf>&, int, int)
    {}   // VRZ computes no grad_wavelet

    // (also used by the ckpt/recursive imaging)
    static void image_step(const State& s, const SolverContext& ctx,
                           const float* forward_ptr, int /*it*/, Wavefield& adjoint,
                           std::vector<Buf>* grads,
                           RTMOutputCore* /*rtm_out*/, BwdWorkspace& ws)
    {
        if (grads == nullptr) return;
        CALCULATE_GRAD_VRZ2D_AUTO(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            forward_ptr,
            adjoint.u_now_t.data_ptr<float>(),
            s.vp,
            s.z,
            s.inv_z,
            ws.c_x.data_ptr<float>(),
            ws.c_z.data_ptr<float>(),
            ws.e_x.data_ptr<float>(),
            ws.e_z.data_ptr<float>(),
            (*grads)[0].data_ptr<float>(),
            (*grads)[1].data_ptr<float>(),
            s.grad_ctx,
            s.lap_ctx,
            ctx
        );
    }

    // ===================================================================== //
    // [4] BACKWARD_BS — generic_backward_bs, per reverse it: the four
    //     shared adjoint hooks of section [3], then bs_recon_step ->
    //     bs_rtm_tap; before the loop (first segment):
    //     seed_reconstruction from u_last_two.
    // ===================================================================== //

    static RTMOutputCore* rtm_out_bs(const BackwardInputCore&, RTMOutputCore&)
    { return nullptr; }

    // The reconstruction grids come from the propagator on every
    // boundary-saving backward (_c.py _forward_state_buffers over
    // cp.forward_state_shapes, which is cuda_layout.reconstruction_nvar --
    // slot_table.ACOUSTIC_VRZ2D.recon = 3 -- in bs mode), and the DD runner
    // rebinds the same list.
    static void bind_or_alloc_recon(Wavefield& wf, const BackwardInputCore& p,
                                    const Buf& vp)
    {
        wavefields_required(p.forward_wavefields, RECON_WF_COUNT, vp,
                            "acoustic_vrz2d/backward_bs reconstruction "
                            "(cuda_layout.reconstruction_nvar)");
        wf.bind(p.forward_wavefields, 2, false);
    }

    // Seed from last_two, zero u_next, and zero the boundary/PML band so stale
    // PML values carried in u_last_two don't leak inward during reverse
    // propagation (see the hand-written driver's accuracy note).
    static void seed_reconstruction(const State& s, const SolverContext& ctx,
                                    Wavefield& forward, const BackwardInputCore& p)
    {
        copy_tensor_cuda_async(forward.u_prev_t, p.u_last_two.select(1, 1).squeeze(0));
        copy_tensor_cuda_async(forward.u_now_t, p.u_last_two.select(1, 0).squeeze(0));
        zero_tensor_device_async(forward.u_next_t);
        auto for_init = forward.view();
        set_boundary_zeros<<<s.launch_config.grid, s.launch_config.block>>>(
            for_init.u_prev, ctx.abcn + ctx.M, s.nx, s.nz,
            ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));
        set_boundary_zeros<<<s.launch_config.grid, s.launch_config.block>>>(
            for_init.u_now, ctx.abcn + ctx.M, s.nx, s.nz,
            ctx.fsLo(0), ctx.fsHi(0), ctx.fsLo(2), ctx.fsHi(2));
    }

    static BsScratch make_bs_scratch(const BackwardInputCore&, const Buf&)
    { return {}; }

    // VRZ bs reverse order: NOPML step, forward-source injection, strip
    // restore, swap, THEN the gradient on the post-swap u_now (both operands
    // co-resident at time it).
    // No illumination on this equation (rtm_out_bs returns nullptr), and its bs
    // loop has no it == 0 tail either -- present so the skeleton can call it.
    static void bs_illum_tail(const State&, const SolverContext&,
                              Wavefield&, RTMOutputCore&) {}

    static void bs_recon_step(const State& s, const SolverContext& ctx,
                                Wavefield& forward, Wavefield& adjoint,
                                BoundaryRuntime& boundary_runtime,
                                const GeneralBoundaryPointer& bs, int save_width,
                                AcousticCPMLPointer /*cpml*/,
                                const BackwardInputCore& p,
                                std::vector<Buf>& grads,
                                RTMOutputCore* /*rtm_out*/,
                                BwdWorkspace& ws,
                                BsScratch& /*scratch*/,
                                int it, int bs_it0)
    {
        auto for_view = forward.view();
        ACOUSTIC_VRZ2D_NOPML(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            for_view,
            s.vp,
            s.z,
            s.inv_z,
            s.lap_ctx,
            s.grad_ctx,
            ctx
        );
        add_source<<<s.source_config.grid, s.source_config.block>>>(
            for_view.u_next,
            p.forward_source.data_ptr<float>(),
            p.forward_sources_loc.data_ptr<int>(),
            it,
            (int)p.forward_sources_loc.size(1),
            ctx
        );
        boundary_runtime.restore_backward_2d(
            it - bs_it0,
            for_view.u_next,
            s.launch_config.grid,
            s.launch_config.block,
            bs,
            save_width,
            -s.M,     // offset
            ctx
        );
        forward.swap();
        CALCULATE_GRAD_VRZ2D_AUTO(
            s.order,
            s.launch_config.grid,
            s.launch_config.block,
            forward.u_now_t.data_ptr<float>(),
            adjoint.u_now_t.data_ptr<float>(),
            s.vp,
            s.z,
            s.inv_z,
            ws.c_x.data_ptr<float>(),
            ws.c_z.data_ptr<float>(),
            ws.e_x.data_ptr<float>(),
            ws.e_z.data_ptr<float>(),
            grads[0].data_ptr<float>(),
            grads[1].data_ptr<float>(),
            s.grad_ctx,
            s.lap_ctx,
            ctx
        );
    }

    static void bs_rtm_tap(const State&, const SolverContext&,
                              Wavefield&, Wavefield&, RTMOutputCore&, bool)
    {}   // no illumination/ADCIG kernels

    // ===================================================================== //
    // [5] CKPT + RECURSIVE PLUMBING
    //     No hooks here: the chunk/recursive checkpoint backward keeps its
    //     hand-written form in backward.cu (see the header comment above).
    // ===================================================================== //
};

} // namespace acoustic_vrz2d
