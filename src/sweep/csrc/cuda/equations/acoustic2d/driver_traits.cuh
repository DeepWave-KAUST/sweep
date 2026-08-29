// Driver traits for the 2-D acoustic equation: the per-equation half of the
// shared skeleton in ``common/eq_driver.cuh``.  Everything here is a
// line-faithful transcription of the launches the hand-written drivers made;
// the physics kernels are untouched.
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

namespace acoustic2d {

struct Driver {
    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "acoustic2d";
    static constexpr int CKPT_NVAR = 6;
    static constexpr int BS_NVAR = 1;           // the saver stores u only
    static constexpr int BS_LAST_TWO_NVAR = 2;  // u_prev, u_now
    static constexpr int TANGENT_PAD = 0;       // (x TANGENT_PAD*M; VRZ uses 1)

    using Wavefield = AcousticWavefieldTensor;
    using CPML = AcousticCPMLTensor;

    // Per-call bundle built once before the time loop: model pointer, operator
    // parameter blocks, launch configs, and the few scalars the hooks need.
    struct State {
        const float* vp;
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
        float dz = p.spacing[1];
        State s;
        s.vp = p.models[0].template data_ptr<float>();
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
        s.has_topo = p.has_topo;
        return s;
    }

    template <class P>
    static void setup_ctx(SolverContext& ctx, const P& p)
    {
        ctx.topo_rows    = p.has_topo ? p.topo_rows.template data_ptr<int>() : nullptr;
        ctx.has_topo     = p.has_topo;
        ctx.topo_category = nullptr;
        ctx.use_apm      = false;
        ctx.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
    }

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.wavefields.empty())
            wf.bind(p.wavefields, 2, true);
        else
            wf.allocate(vp, 2, true, /*double_buffer_psi=*/true);
    }

    static void init_aux_slabs(SolverContext& ctx, Wavefield& wf)
    {
        acoustic_init_aux_slabs(ctx, wf);
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.allocate(p.pml_vals, 2);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, d.B, d.nz, d.nx};
    }

    static int save_width(int abcn, int M) { return abcn > 0 ? M + 1 : M; }

    // One forward step over x in [xb, xe).
    // Pre-pass: clear air cells in a separate kernel launch so the
    // main acoustic2nd kernel only reads (never writes) air cells.
    // Eliminates intra-launch RAW race on PML aux fields that was
    // showing up as ~30% non-deterministic forward output across
    // processes (sweep VTI history pattern).  The air-clear range is
    // widened by the stencil halo M so a phase-split stencil launch
    // still only reads air cells cleared earlier THIS step (re-clearing
    // across phases writes the same zeros — idempotent).
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
            auto alc = fdtd::Wave2D::make(axe - axb, s.nz, s.B);
            acoustic2d_air_clear_kernel<<<alc.grid, alc.block>>>(
                view, save_all, u_thist, actx
            );
        }
        SolverContext sctx = ctx;
        sctx.x_base = xb;
        sctx.x_limit = xe;
        auto lc = fdtd::Wave2D::make(xe - xb, s.nz, s.B);
        ACOUSTIC2D(
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
        rt.save_forward_2d(
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
                       torch::Tensor& record, const ForwardInput& p,
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

    static void end_of_step(Wavefield& wf)
    {
        wf.swap_pml();   // rotate u AND psi<->psin: race-free psi double-buffer
    }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        saver.last_two_t.select(1, 0).copy_(wf.u_prev_t);
        saver.last_two_t.select(1, 1).copy_(wf.u_now_t);
    }
};

} // namespace acoustic2d
