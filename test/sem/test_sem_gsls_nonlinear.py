"""The nonlinear constant-Q fit (``sweep.equations.gsls_nonlinear``, SPECFEM2D's USE_SOLVOPT)
and the SEM classes that take it (``fit="nonlinear"``).

* the fit: SPECFEM2D's own SolvOpt mechanisms for the same Q (``specfem_solvopt_driver.f90``,
  band f0 / 10 .. 10 f0 at f0 = 1 Hz) to its four digits, our residual no larger; a flatter
  Q(f) than the linear fit; the table against direct fits; positive strengths below the linear
  fit's ``q_min``; ``Q = inf`` gives zero strengths; the derivative in Q; refusals;
* ViscoElasticSEM2DDeformed and ViscoAcousticPotentialSEM2D with it: per-node relaxation
  times (bulk and shear their own), ``Q = inf`` still the elastic / potential step bit for
  bit, a Q the linear fit refuses, the gradient in Q; the velocity-stress class takes no fit.
"""
import numpy as np
import pytest
import torch

from sweep.equations.gsls_nonlinear import GSLSNonlinearFit
from sweep.equations.visco_elastic import GSLSFit, gsls_relaxation_times
from sweep.propagator import EagerOptions, Full
from sweep.sem import (AcousticPotentialSEM2D, ElasticSEM2DDeformed, PropSEM,
                       ViscoAcousticPotentialSEM2D, ViscoElasticSEM2DDeformed,
                       ViscoElasticVSSEM2DDeformed)

F64 = torch.float64
BAND = (0.1, 10.0)
# SPECFEM2D 8.1.0, USE_SOLVOPT: (n_sls, Q) -> (theta_l / 2 pi in Hz, w_l = (tau_eps / tau_sigma - 1) / N)
SPECFEM = {
    (2, 100.0): ([2.11448313652829e-01, 4.93248622440177e+00],
                 [2.13856460976348e-02, 2.18306019620124e-02]),
    (3, 10.0): ([1.30389075004573e-01, 1.12585597235583e+00, 9.80191274594000e+00],
                [1.80999570773249e-01, 1.61799220292460e-01, 2.48945166062010e-01]),
    (3, 2.0): ([2.08628361776763e-01, 1.71904176517293e+00, 1.53787679697824e+01],
               [1.11860216320606e+00, 1.66731970242161e+00, 5.04121323124959e+00]),
    (4, 30.0): ([8.42028709025617e-02, 4.81400336395260e-01, 2.19609177929358e+00,
                 1.26818589372108e+01],
                [5.12731456162829e-02, 3.48131724230574e-02, 3.58746965907198e-02,
                 5.83601570571110e-02]),
}


def _residual(tau, w, Q, n_freq):
    """SPECFEM2D's (and the fit's) residual ``Q Im m - Re m`` at its frequencies, squared."""
    f = BAND[0] * (BAND[1] / BAND[0]) ** (np.arange(n_freq) / (n_freq - 1))
    m = GSLSNonlinearFit.modulus_ratio(tau, w, f)
    return float(np.sum((Q * m.imag - m.real) ** 2))


def _q_error(tau, w, Q):
    f = np.geomspace(*BAND, 3000)
    m = GSLSNonlinearFit.modulus_ratio(tau, w, f)
    return float(np.abs(m.real / m.imag / Q - 1.0).max())


@pytest.mark.parametrize("key", list(SPECFEM))
def test_specfem2d_solvopt_gives_the_same_mechanisms(key):
    n, Q = key
    th, ws = SPECFEM[key]
    tau_s = 1.0 / (2.0 * np.pi * np.array(th))
    tau, w = GSLSNonlinearFit(n, *BAND).params_np(Q)
    assert np.abs(tau / tau_s - 1.0).max() < 3e-4
    assert np.abs(w / np.array(ws) - 1.0).max() < 3e-4
    assert _residual(tau, w, Q, 4 * n) <= _residual(tau_s, ws, Q, 4 * n)


@pytest.mark.parametrize("n,lin_err,nl_err", [(3, 0.05, 0.037), (4, 0.035, 0.0085)])
def test_a_flatter_q_than_the_linear_fit(n, lin_err, nl_err):
    # Q = 10 over the two decades, the largest error of Q(f): the linear fit at SPECFEM2D's
    # 2N - 1 points (5.8% and 4.0%) against this (3.5% and 0.8%)
    Q = 10.0
    lin = GSLSFit(gsls_relaxation_times(n, *BAND), *BAND, n_freq=2 * n - 1)
    e_lin = _q_error(lin.tau_sigma, lin.tau_np(Q), Q)
    e_nl = _q_error(*GSLSNonlinearFit(n, *BAND).params_np(Q), Q)
    assert e_lin > lin_err and e_nl < nl_err
    assert e_nl < 0.7 * e_lin


def test_the_table_is_the_direct_fit():
    fit = GSLSNonlinearFit(3, *BAND)
    Q = torch.tensor([1.3, 2.7, 9.0, 33.0, 150.0, 4e3], dtype=F64)
    tau, w = fit.params(Q)
    for i, q in enumerate(Q.tolist()):
        tau_d, w_d = fit.params_np(q)
        assert np.abs(np.array([float(t[i]) for t in tau]) / tau_d - 1.0).max() < 1e-6
        assert np.abs(np.array([float(x[i]) for x in w]) / w_d - 1.0).max() < 1e-6


def test_positive_strengths_where_the_linear_fit_has_none():
    lin = GSLSFit(gsls_relaxation_times(4, *BAND), *BAND, n_freq=200)
    q_lin = lin.min_valid_q()
    assert q_lin > 1.5
    tau, w = GSLSNonlinearFit(4, *BAND).params(torch.tensor([1.0, 1.5], dtype=F64))
    assert all(bool((x > 0).all()) for x in w) and all(bool((t > 0).all()) for t in tau)
    assert _q_error([float(t[1]) for t in tau], [float(x[1]) for x in w], 1.5) < 0.05


def test_infinite_q_and_the_derivative_in_q():
    fit = GSLSNonlinearFit(3, *BAND)
    Q = torch.tensor([3.0, 40.0, float("inf")], dtype=F64, requires_grad=True)
    tau, w = fit.params(Q)
    assert all(float(x[2].detach()) == 0.0 for x in w)
    loss = lambda tt, ww: sum((k + 1.0) * (100.0 * t + x) for k, (t, x) in enumerate(zip(tt, ww)))
    (g,) = torch.autograd.grad(loss(tau, w).sum(), Q)
    assert float(g[2]) == 0.0
    for i in range(2):
        h = 1e-6 * float(Q[i])
        val = lambda q: float(loss(*fit.params(torch.tensor([q], dtype=F64))))
        fd = (val(float(Q[i]) + h) - val(float(Q[i]) - h)) / (2.0 * h)
        assert abs(float(g[i]) - fd) <= 1e-6 * abs(fd)


def test_refusals():
    with pytest.raises(ValueError, match="n_sls >= 2"):
        GSLSNonlinearFit(1, *BAND)
    with pytest.raises(ValueError, match="cannot fit"):
        GSLSNonlinearFit(3, *BAND, n_freq=5)
    with pytest.raises(ValueError, match="not a smooth function of Q"):
        GSLSNonlinearFit(2, 1.0, 1000.0, n_freq=8)


# --------------------------------------------------------------------------- #
# the SEM classes
# --------------------------------------------------------------------------- #
SHAPE, DH, DT, NT = (41, 81), 5.0, 4e-4, 400
VP, VS, RHO = 2000.0, 1000.0, 2000.0


def _prop(eq, fit="nonlinear", abcn=12, **kw):
    opts = {"f_ref": 15.0, "fit": fit} if eq in (ViscoElasticSEM2DDeformed,
                                                ViscoAcousticPotentialSEM2D) else {}
    if eq is ViscoAcousticPotentialSEM2D:
        opts["n_fit"] = None
        kw.setdefault("density", RHO)
        kw.setdefault("receiver_type", ["p"])
        kw.setdefault("source_type", ["chi"])
    if eq is AcousticPotentialSEM2D:
        kw.setdefault("density", RHO)
        kw.setdefault("receiver_type", ["p"])
        kw.setdefault("source_type", ["chi"])
    kw.setdefault("receiver_type", ["vx", "vz"])
    kw.setdefault("source_type", ["vz"])
    return PropSEM(eq, shape=SHAPE, dh=DH, dt=DT, n_order=4, f_dominant=15.0, dev="cpu",
                   dtype=F64, impl="eager", memory=Full(), abcn=abcn, free_surface=True,
                   eager_options=EagerOptions(use_compile=False),
                   **({"equation_options": opts} if opts else {}), **kw)


def _shot(p, models):
    t = np.arange(NT) * DT - 0.08
    a = (np.pi * 15.0 * t) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    rec = np.array([[[x, 10.0] for x in (10.0, 30.0, 50.0, 70.0)]])
    return p(w, np.array([[40.3, 20.6]]), rec, models=models)[0]


def _elastic(grad=False):
    return [torch.full(SHAPE, v, dtype=F64, requires_grad=grad) for v in (VP, VS, RHO)]


def _q(p, *qs, grad=False):
    return [torch.full(p._gll_shape, q, dtype=F64, requires_grad=grad) for q in qs]


def test_the_displacement_class_takes_per_node_relaxation_times():
    p = _prop(ViscoElasticSEM2DDeformed)
    eq = p._build_eq(_elastic() + _q(p, 30.0, 8.0))
    assert eq.tau_sigma is None and len(eq._tau_k) == len(eq._tau_mu) == 3
    fit = GSLSNonlinearFit(3, 1.5, 150.0)
    for qv, taus, ws in ((30.0, eq._tau_k, eq._w1), (8.0, eq._tau_mu, eq._w2)):
        tau_d, w_d = fit.params_np(qv)
        assert all(tuple(t.shape) == p._gll_shape for t in taus)
        assert np.allclose([float(t.max()) for t in taus], tau_d, rtol=1e-6)
        assert np.allclose([float(t.min()) for t in taus], tau_d, rtol=1e-6)
        assert np.allclose([float(x.max()) for x in ws], w_d, rtol=1e-6)


@pytest.mark.parametrize("eq,base", [(ViscoElasticSEM2DDeformed, ElasticSEM2DDeformed),
                                     (ViscoAcousticPotentialSEM2D, AcousticPotentialSEM2D)])
def test_infinite_q_is_still_the_lossless_step_bit_for_bit(eq, base):
    p, inf = _prop(eq), float("inf")
    if eq is ViscoAcousticPotentialSEM2D:
        lossless, models = [_elastic()[0]], [_elastic()[0]] + _q(p, inf)
    else:
        lossless, models = _elastic(), _elastic() + _q(p, inf, inf)
    assert torch.equal(_shot(p, models), _shot(_prop(base), lossless))


def test_a_q_the_linear_fit_refuses_and_the_gradient_in_q():
    lin = _prop(ViscoElasticSEM2DDeformed, fit="linear")
    # the linear fit's strengths turn negative below Q = 1.52 (3 mechanisms, 5 points)
    with pytest.raises(ValueError, match="fit='nonlinear'"):
        lin._build_eq(_elastic() + _q(lin, 30.0, 1.4))
    p = _prop(ViscoElasticSEM2DDeformed)
    models = _elastic(grad=True) + _q(p, 30.0, 1.4, grad=True)
    out = _shot(p, models)
    assert bool(torch.isfinite(out).all())
    (out ** 2).sum().backward()
    for m in models:
        assert bool(torch.isfinite(m.grad).all()) and float(m.grad.abs().max()) > 0


def test_the_velocity_stress_class_takes_no_fit():
    with pytest.raises(TypeError, match="not options"):
        PropSEM(ViscoElasticVSSEM2DDeformed, shape=SHAPE, dh=DH, dt=DT, n_order=4, dev="cpu",
                impl="eager", equation_options={"fit": "nonlinear"})
