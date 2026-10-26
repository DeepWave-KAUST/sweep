// Driver traits for the 2-D elastic Born equation: elastic2d's driver with a
// second (scattered) field stepped next to the background one.  Deltas against
// elastic2d (the family's reference; see the property list at the top of
// ../elastic2d/driver_traits.cuh -- everything not named here is inherited):
//   * models: {vp, vs, rho, dvp, dvs, drho}; parse_models derives the
//     background Lame pair exactly as elastic2d and keeps the perturbations;
//   * forward wavefields (cuda_layout.base_nvar + pml_nvar = 30), bind order
//     [bg vx, vz, sxx, szz, sxz][sc vx, vz, sxx, szz, sxz][bg 10 CPML memories]
//     [sc 10 CPML memories]; field_ptr maps the eager field ids (bg 0..14,
//     sc 15..29) onto the two elastic views;
//   * velocity_substep / stress_substep = the fused Born kernels (kernels.cuh);
//   * everything the backward reads is the BACKGROUND's, in elastic2d's layout:
//     checkpoint_tensors (CKPT_NVAR = 15), save_boundary_fields (BS_NVAR = 5),
//     save_last_state and the u_allt velocities.
// The backward is elastic2d's (backward.cu): the adjoint of the scattered
// field obeys the background elastic equation and its imaging condition
// against the background field is exactly the gradient of (dvp, dvs, drho).
#pragma once

#include "kernels.cuh"
#include "../elastic2d/driver_traits.cuh"

namespace elastic_born2d {

constexpr int N_MODELS = 6;    // vp, vs, rho, dvp, dvs, drho
constexpr int N_FIELDS = 15;   // one elastic field set (eager / receiver field ids)

struct BornWfView {
    ElasticWavefieldPointer bg, sc;
};

struct BornWavefield {
    ElasticWavefieldTensor bg, sc;

    void bind_born(const std::vector<Buf>& t)
    {
        SWEEP_CHECK(t.size() == 2 * N_FIELDS,
                    "elastic_born2d/forward requires the propagator-bound wavefields "
                    "(cuda_layout.base_nvar + pml_nvar = 30 tensors), got ", t.size());
        std::vector<Buf> b(t.begin(), t.begin() + 5);
        std::vector<Buf> s(t.begin() + 5, t.begin() + 10);
        b.insert(b.end(), t.begin() + 10, t.begin() + 20);
        s.insert(s.end(), t.begin() + 20, t.begin() + 30);
        bg.bind(b, true);
        sc.bind(s, true);
        SWEEP_CHECK(same_shape(bg.m_vxx_t, sc.m_vxx_t) && same_shape(bg.m_vzz_t, sc.m_vzz_t),
                    "elastic_born2d: the background and scattered CPML memories must share "
                    "one slab layout");
    }

    BornWfView view() { return {bg.view(), sc.view()}; }

    // The checkpoint-mode backward replays the background alone.
    std::vector<Buf> checkpoint_tensors() const { return bg.checkpoint_tensors(); }
};

struct Driver : elastic2d::Driver {
    using Base = elastic2d::Driver;

    static constexpr const char* NAME = "elastic_born2d";
    static constexpr const char* CKPT_COUNT_MSG =
        "Elastic Born 2D checkpointing expects 15 checkpoint tensors (the background's)";

    using Wavefield = BornWavefield;
    using WfView = BornWfView;

    struct Models : Base::Models {
        Buf dvp, dvs, drho;
    };

    template <class P>
    static Models parse_models(const P& p)
    {
        SWEEP_CHECK(p.models.size() == N_MODELS,
                    "elastic_born2d expects the models {vp, vs, rho, dvp, dvs, drho}, got ",
                    p.models.size());
        Models m;
        static_cast<Base::Models&>(m) = Base::parse_models(p);
        m.dvp = p.models[3];
        m.dvs = p.models[4];
        m.drho = p.models[5];
        return m;
    }

    struct State : Base::State {
        Buf dvp, dvs, drho;
    };

    template <class P>
    static State make_state(const P& p, const eqdrv::Dims& d, const Models& models,
                            fdtd::LaunchConfig launch_config,
                            fdtd::LaunchConfig source_config,
                            fdtd::LaunchConfig record_config)
    {
        State s;
        static_cast<Base::State&>(s) = Base::make_state(p, d, models, launch_config,
                                                        source_config, record_config);
        s.dvp = models.dvp;
        s.dvs = models.dvs;
        s.drho = models.drho;
        return s;
    }

    static void bind_or_alloc_forward(Wavefield& wf, const ForwardInputCore& p, const Buf&)
    {
        wf.bind_born(p.wavefields);
    }

    static void init_aux_slabs(SolverContext& solver, Wavefield& wf)
    {
        elastic_init_aux_slabs(solver, wf.bg);
    }

    static float* field_ptr(WfView& wf, int field_idx)
    {
        if (field_idx >= 0 && field_idx < N_FIELDS) return elastic_field_ptr(wf.bg, 2, field_idx);
        if (field_idx >= N_FIELDS && field_idx < 2 * N_FIELDS)
            return elastic_field_ptr(wf.sc, 2, field_idx - N_FIELDS);
        return nullptr;
    }

    static WfView view(Wavefield& wf) { return wf.view(); }

    static void capture_allt(Buf&, WfView&, const SolverContext&, int) {}

    static void velocity_substep(const State& s, WfView& wf,
                                 ElasticCPMLPointer cpml_view,
                                 const SolverContext& solver)
    {
        LAUNCH_ELASTIC_BORN_VELOCITY(
            s.order, s.launch_config.grid, s.launch_config.block,
            wf.bg, wf.sc,
            s.models.rho.data_ptr<float>(),
            s.drho.data_ptr<float>(),
            s.grad_ctx, cpml_view, solver);
    }

    static void stress_substep(const State& s, WfView& wf,
                               ElasticCPMLPointer cpml_view,
                               const SolverContext& solver, float* u_this_t)
    {
        ElasticBornModels mdl{
            s.models.lambda.data_ptr<float>(), s.models.mu.data_ptr<float>(),
            s.models.vp.data_ptr<float>(), s.models.vs.data_ptr<float>(),
            s.models.rho.data_ptr<float>(),
            s.dvp.data_ptr<float>(), s.dvs.data_ptr<float>(), s.drho.data_ptr<float>()};
        LAUNCH_ELASTIC_BORN_STRESS(
            s.order, s.launch_config.grid, s.launch_config.block,
            wf.bg, wf.sc, mdl, u_this_t,
            s.grad_ctx, cpml_view, solver);
    }

    static void save_boundary_fields(BoundaryRuntime& rt, const State& s,
                                     const SolverContext& solver, WfView& wf,
                                     int it, int nt,
                                     const GeneralBoundaryPointer& bs,
                                     int save_width)
    {
        Base::save_boundary_fields(rt, s, solver, wf.bg, it, nt, bs, save_width);
    }

    static void save_last_state(EffectiveBoundarySaver& saver, Wavefield& wf)
    {
        Base::save_last_state(saver, wf.bg);
    }
};

} // namespace elastic_born2d
