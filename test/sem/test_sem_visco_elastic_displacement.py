"""ViscoElasticSEM2DDeformed: SPECFEM2D's visco-elastic displacement SEM.

test/sem/specfem_compare/run_compare_box.py checks it against SPECFEM2D's
ATTENUATION_VISCOELASTIC; here: ``Q = inf`` is the elastic step bit for bit, a finite Q
attenuates with distance, ``Qmu`` takes the shear wave down and ``Qkappa`` leaves it alone, the
absorbing layer (and only it) runs without attenuation, the gradients in the five models
exist, the two readings of the velocities are the moduli they say, and what it cannot do is
refused.
"""
import math

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, EagerOptions, Full
from sweep.sem import ElasticSEM2DDeformed, PropSEM, ViscoElasticSEM2DDeformed

SHAPE, DH, DT, NT = (61, 121), 5.0, 5e-4, 700
F64 = torch.float64
VP, VS, RHO = 2000.0, 1000.0, 2000.0


def _prop(equation=ViscoElasticSEM2DDeformed, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), "abcn": 16,
          "free_surface": True, "receiver_type": ["vx", "vz"], "source_type": ["vz"], **kw}
    if equation is ViscoElasticSEM2DDeformed:
        kw.setdefault("equation_options", {"f_ref": 15.0})
    return PropSEM(equation, shape=SHAPE, dh=DH, dt=DT, n_order=4, f_dominant=15.0, dev="cpu",
                   dtype=F64, impl="eager", **kw)


def _shot(prop, models, rec=((80.0, 30.0), (100.0, 30.0), (110.0, 30.0))):
    t = np.arange(NT) * DT - 0.08
    a = (np.pi * 15.0 * t) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    return prop(w, np.array([[60.3, 30.6]]), np.array([rec]), models=models)[0]


def _elastic(grad=False):
    return [torch.full(SHAPE, v, dtype=F64, requires_grad=grad) for v in (VP, VS, RHO)]


def _q(prop, qk, qmu, grad=False):
    # on the GLL nodes: an infinite Q interpolated from the grid would be nan
    return [torch.full(prop._gll_shape, q, dtype=F64, requires_grad=grad) for q in (qk, qmu)]


def _window_peak(trace, r, v):
    t = torch.arange(NT, dtype=F64) * DT - 0.08
    return float(trace[(t - r / v).abs() < 0.06].abs().max())


def test_infinite_q_is_the_elastic_step_bit_for_bit():
    p = _prop()
    inf = float("inf")
    assert torch.equal(_shot(p, _elastic() + _q(p, inf, inf)),
                       _shot(_prop(ElasticSEM2DDeformed), _elastic()))


def test_a_finite_q_attenuates_with_distance():
    # an explosion's P wave along x, absorbing sides: the edges' echoes would mix in. Read
    # as unrelaxed, vp makes the wave at 15 Hz slower than the elastic one and its near trace
    # larger (the ratio starts above 1); what Q does is the fall with distance,
    # exp(-pi f r / (Q v)) = 0.84 over the 150 m
    kw = dict(free_surface=False, source_type=["sxx", "szz"])
    rec = tuple((60.0 + r, 30.0) for r in (15.0, 30.0, 45.0))
    elastic = _shot(_prop(ElasticSEM2DDeformed, **kw), _elastic(), rec)
    p = _prop(**kw)
    lossy = _shot(p, _elastic() + _q(p, 20.0, 20.0), rec)
    ratio = [_window_peak(lossy[:, i, 0], r * DH, VP) / _window_peak(elastic[:, i, 0], r * DH, VP)
             for i, r in enumerate((15.0, 30.0, 45.0))]
    assert ratio[1] < ratio[0] and ratio[2] < ratio[1]
    assert 0.75 < ratio[2] / ratio[0] < 0.9


def test_qmu_takes_the_shear_wave_down_and_qkappa_leaves_it_alone():
    # a vertical force sends its shear wave sideways, 200 m to the receiver:
    # exp(-pi f t / Qmu) = 0.62
    kw = dict(free_surface=False)
    rec = ((100.0, 30.6),)
    peak = lambda seis: _window_peak(seis[:, 0, 1], 40.0 * DH, VS)
    elastic = peak(_shot(_prop(ElasticSEM2DDeformed, **kw), _elastic(), rec))
    p = _prop(**kw)
    shear = peak(_shot(p, _elastic() + _q(p, float("inf"), 20.0), rec)) / elastic
    bulk = peak(_shot(p, _elastic() + _q(p, 20.0, float("inf")), rec)) / elastic
    assert 0.55 < shear < 0.7
    assert abs(bulk - 1.0) < 0.02


def test_the_absorbing_layer_and_only_it_runs_without_attenuation():
    p = _prop()
    eq = p._build_eq(_elastic() + _q(p, 20.0, 20.0))
    E = eq.mesh.n_elements
    xc, zc = eq.mesh.x_e.mean((1, 2)), eq.mesh.z_e.mean((1, 2))
    pad = 16 * DH                                     # the layer, left, right and below
    inside = (xc > pad) & (xc < pad + (SHAPE[1] - 1) * DH) & (zc < (SHAPE[0] - 1) * DH)
    for w in eq._w1 + eq._w2:
        on = (w != 0).reshape(E, -1)
        assert torch.equal(on.all(1), inside) and torch.equal(on.any(1), inside)


def test_the_gradients_exist():
    p = _prop()
    models = _elastic(grad=True) + _q(p, 30.0, 20.0, grad=True)
    (_shot(p, models) ** 2).sum().backward()
    for m in models:
        assert torch.isfinite(m.grad).all() and m.grad.abs().max() > 0


def test_the_readings_of_the_velocities_are_the_moduli_they_say():
    p = _prop()
    eq = p._build_eq(_elastic() + _q(p, 30.0, 20.0))         # unrelaxed, the default
    for got, want in ((eq._mu, RHO * VS ** 2), (eq._lam2mu, RHO * VP ** 2)):
        assert torch.allclose(got, torch.full_like(got, want), rtol=1e-14, atol=0)
    # real_modulus: the memories' modulus mu_U (1 - sum_l w_l / (1 + sum w) / (1 + i w tau_l))
    # has the real part rho vs^2 at f_ref, and the bulk one rho (vp^2 - vs^2)
    p = _prop(equation_options={"f_ref": 15.0, "velocities": "real_modulus"})
    eq = p._build_eq(_elastic() + _q(p, 30.0, 20.0))
    om = 2.0 * math.pi * 15.0
    for strengths, modulus, want in ((eq._w2, eq._mu_U, RHO * VS ** 2),
                                     (eq._w1, eq._kappa_U, RHO * (VP ** 2 - VS ** 2))):
        w = [float(s.max()) for s in strengths]
        re = 1.0 - sum(wl / (1.0 + sum(w)) / (1.0 + (om * ts) ** 2)
                       for wl, ts in zip(w, eq.tau_sigma))
        assert abs(float(modulus.max()) * re / want - 1.0) < 1e-12


def test_what_it_cannot_do_is_refused():
    with pytest.raises(NotImplementedError, match="dissipative"):
        _prop(memory=BoundarySaving())
    with pytest.raises(NotImplementedError, match="impl='c'"):
        PropSEM(ViscoElasticSEM2DDeformed, shape=SHAPE, dh=DH, dt=DT, n_order=4, dev="cpu",
                impl="c")
    p = _prop(equation_options={"f_ref": 15.0, "velocities": "phase"})
    with pytest.raises(ValueError, match="'unrelaxed' or 'real_modulus'"):
        p._build_eq(_elastic() + _q(p, 30.0, 20.0))
    p = _prop()
    with pytest.raises(ValueError, match="Bound Q from below"):
        p._build_eq(_elastic() + _q(p, 30.0, 1.1))
