// Driver traits for the 2-D elastic equation: the per-equation half of the
// staggered-family skeleton in ``common/sg_driver.cuh``.  Line-faithful
// transcription of the hand-written drivers; physics kernels untouched.
// The APM (Cao & Chen 2018) entry points stay hand-written in
// forward.cu/backward.cu — they refuse stepping/phasing and carry their own
// kernel variants, so there is nothing for the skeleton to share yet.
#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>

#include <algorithm>

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

namespace elastic2d {

struct Driver {
    static constexpr int NDIM = 2;
    static constexpr const char* NAME = "elastic2d";
    static constexpr int CKPT_NVAR = 15;
    static constexpr const char* CKPT_COUNT_MSG =
        "Elastic 2D checkpointing expects 15 checkpoint tensors";
    static constexpr int BS_NVAR = 5;   // vx, vz, sxx, szz, sxz

    using Wavefield = ElasticWavefieldTensor;
    using WfView = ElasticWavefieldPointer;
    using CPML = ElasticCPMLTensor;

    // lambda/mu are torch-derived tensors: the struct keeps them alive for
    // the whole call (kernels hold raw pointers into them).
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

    template <class P>
    static void setup_ctx(SolverContext& solver, const P& p)
    {
        solver.set_per_edge(p.fs_faces, p.pad_lo, p.pad_hi);
        if (p.has_topo) {
            solver.topo_rows = p.topo_rows.template data_ptr<int>();
            solver.has_topo = true;
        }
    }

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInput& p,
                                      const torch::Tensor& vp)
    {
        if (!p.wavefields.empty())
            wf.bind(p.wavefields, true);
        else
            wf.allocate(vp, 2);
    }

    static WfView view(Wavefield& wf) { return wf.view(); }

    static void init_aux_slabs(SolverContext& solver, Wavefield& wf)
    {
        elastic_init_aux_slabs(solver, wf);
    }

    template <class P>
    static void alloc_cpml(CPML& cpml, const P& p)
    {
        cpml.allocate(p.pml_vals, 2);
    }

    static std::vector<int64_t> allt_shape(const eqdrv::Dims& d, int64_t nt)
    {
        return {nt, 2, d.B, d.nz, d.nx};   // only Vx and Vz
    }

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

    static float* field_ptr(WfView& wf, int field_idx)
    {
        return elastic_field_ptr(wf, 2, field_idx);
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
        saver.last_two_t.select(0, 1).select(0, 0).copy_(wf.vz_t);
        saver.last_two_t.select(0, 2).select(0, 0).copy_(wf.sxx_t);
        saver.last_two_t.select(0, 3).select(0, 0).copy_(wf.szz_t);
        saver.last_two_t.select(0, 4).select(0, 0).copy_(wf.sxz_t);
    }
};

} // namespace elastic2d
