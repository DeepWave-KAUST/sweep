"""Terms without a derivative in a ``sweep.symbolic`` first-order system.

``r.dt = -(r + ...) / ts`` (a field times a model in its own equation: a
damping, stepped with the trapezoidal rule) and ``p.dt = ... - r`` (another
field at the same position: a coupling, read at the time centre of the step).
Pinned here: the refusals; the generated step against the same update written
by hand; a standard linear solid (Zener) against its exact Fourier solution,
second order in ``dt`` and stable for a relaxation time well below ``dt``; the
GSLS visco-elastic system against the shipped ``ViscoElastic``; the memory
modes (a damping refuses boundary saving, a coupling alone does not) and the
compiled step.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy
import torch

from conftest import requires_compile, ricker
from sweep.equations.visco_elastic import ViscoElastic
from sweep.propagator.options import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Eq, Field, Model

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = old


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def zener_pde(name="SymZener", pml="cpml", axes="zx"):
    """Acoustic waves in a standard linear solid: relaxed modulus K, unrelaxed
    K (1 + tau), stress relaxation time ts, one memory variable r."""
    K, rho, tau, ts = Model("K"), Model("rho"), Model("tau"), Model("ts")
    p = Field("p", time_order=1, source=True, receiver=True)
    v = {ax: Field(f"v{ax}", time_order=1, stagger=ax, receiver=True) for ax in axes}
    r = Field("r", time_order=1)
    div = sum(v[ax].deriv(ax) for ax in axes)
    return PDE([*[Eq(v[ax].dt, -p.deriv(ax) / rho) for ax in axes],
                Eq(r.dt, -(r + K * tau * div) / ts),
                Eq(p.dt, -K * (1 + tau) * div - r)], name=name, pml=pml)


def coupled_pde(name="SymCoupled"):
    """A coupling without a damping: p reads q, another field of its sub-step."""
    K, rho, a = Model("K"), Model("rho"), Model("a")
    p = Field("p", time_order=1, source=True, receiver=True)
    q = Field("q", time_order=1)
    vx, vz = Field("vx", time_order=1, stagger="x"), Field("vz", time_order=1, stagger="z")
    return PDE([Eq(vx.dt, -p.dx / rho), Eq(vz.dt, -p.dz / rho),
                Eq(q.dt, -K * (vx.dx + vz.dz)),
                Eq(p.dt, -K * (vx.dx + vz.dz) + a * q)], name=name)


# ---- lowering and refusals ------------------------------------------------------

def test_damping_and_coupling_are_written_as_terms():
    cls = zener_pde().build(2)
    assert cls.DAMPING_TERMS == ("-r/ts",)
    src = zener_pde().source(2)
    assert "_old_r = r" in src and "(_old_r + r)" in src          # p reads the mean of r
    assert "(1 - dt/(2*ts))/(1 + dt/(2*ts))" in src               # the trapezoidal factor
    assert not hasattr(coupled_pde().build(2), "DAMPING_TERMS") or not coupled_pde().build(2).DAMPING_TERMS


def _fields():
    rho, k = Model("rho"), Model("k")
    p = Field("p", time_order=1)
    r = Field("r", time_order=1)
    vx = Field("vx", time_order=1, stagger="x")
    return rho, k, p, r, vx


def _refused():
    rho, k, p, r, vx = _fields()
    w = Field("w", time_order=1)
    return [
        ([Eq(vx.dt, -p.dx / rho + k * p), Eq(p.dt, -vx.dx)], ValueError, "different x positions"),
        ([Eq(vx.dt, -p.dx / rho), Eq(p.dt, -vx.dx - r), Eq(r.dt, -r - k * vx.dx)],
         NotImplementedError, "updated later in the same sub-step"),
        ([Eq(vx.dt, -p.dx / rho), Eq(r.dt, -r * p), Eq(p.dt, -vx.dx)], ValueError, "not linear"),
        ([Eq(vx.dt, -p.dx / rho), Eq(p.dt, -vx.dx + k * w)], ValueError, "no equation of its own"),
    ]


@pytest.mark.parametrize("k", range(4))
def test_unsupported_local_terms_are_refused(k):
    equations, error, match = _refused()[k]
    with pytest.raises(error, match=match):
        PDE(equations, name="SymBad").build(2)


# ---- one step against the update written by hand --------------------------------

def test_zener_step_matches_reference():
    eq = zener_pde("SymZenerStep", pml="none").build(2)(8, device=DEV)
    eq.pd.set_spacing((1.0, 1.0))
    g = torch.Generator().manual_seed(4)
    W = dict(zip(eq.wavefields, (torch.randn(2, 1, 30, 34, generator=g).to(DEV) for _ in eq.wavefields)))
    K, rho, tau, ts = ((lo + (hi - lo) * torch.rand(30, 34, generator=g)).to(DEV)
                       for lo, hi in ((0.8, 1.2), (0.8, 1.2), (0.1, 0.3), (0.05, 2.0)))
    dt = 0.3
    out = dict(zip(eq.wavefields, eq.func(list(W.values()), [K, rho, tau, ts], dt, 1.0, None)))
    pd = eq.pd
    vx = W["vx"] - dt / rho * pd.x_forward(W["p"])
    vz = W["vz"] - dt / rho * pd.z_forward(W["p"])
    div = pd.x_backward(vx) + pd.z_backward(vz)
    # r.dt = -(r + K tau div)/ts by the trapezoidal rule; p reads the mean of r.
    r = ((1 - dt / (2 * ts)) * W["r"] - dt * K * tau * div / ts) / (1 + dt / (2 * ts))
    p = W["p"] + dt * (-K * (1 + tau) * div - (W["r"] + r) / 2)
    for name, ref in (("vx", vx), ("vz", vz), ("r", r), ("p", p)):
        assert _rel(out[name], ref) <= 1e-6, name


# ---- a standard linear solid against its exact solution -------------------------

def _zener_error(dt, ts, t=60.0, n=512, K=1.0, rho=1.0, tau=0.3):
    """A Gaussian pressure at rest on a line far from the edges, float64,
    against the exact solution of the semi-discrete-in-time system: the 3x3
    linear system per wavenumber, solved by its eigenvectors."""
    eq = zener_pde("SymZenerLine", pml="none", axes="x").build(2)(8)
    for name in ("kxf", "kxb", "kzf", "kzb"):
        setattr(eq.pd, name, getattr(eq.pd, name).double())
    x = np.arange(n) - n / 2
    p0 = np.exp(-(x / 6.0) ** 2)
    k = 2 * np.pi * np.fft.fftfreq(n)
    A = np.zeros((n, 3, 3), complex)                        # d/dt (P, V, R) = A (P, V, R)
    A[:, 0, 1], A[:, 0, 2] = -K * (1 + tau) * 1j * k, -1
    A[:, 1, 0] = -1j * k / rho
    A[:, 2, 1], A[:, 2, 2] = -K * tau * 1j * k / ts, -1 / ts
    w, V = np.linalg.eig(A)
    y0 = np.zeros((n, 3), complex)
    y0[:, 0] = np.fft.fft(p0)
    c = np.linalg.solve(V, y0[..., None])[..., 0]

    def exact(time):
        return np.einsum("nij,nj->ni", V, c * np.exp(w * time))

    steps = int(round(t / dt))
    v0 = np.real(np.fft.ifft(exact(-dt / 2)[:, 1] * np.exp(1j * k / 2)))    # v at x + 1/2, t = -dt/2
    line = {"p": p0, "vx": v0, "r": 0 * p0}
    fields = [torch.tensor(line[name], dtype=torch.float64).view(1, 1, 1, -1) for name in eq.wavefields]
    ones = torch.ones(1, n, dtype=torch.float64)
    with torch.no_grad():
        for _ in range(steps):
            fields = list(eq.func(fields, [K * ones, rho * ones, tau * ones, ts * ones], dt, 1.0, None))
    num = fields[eq.wavefields.index("p")][0, 0, 0].numpy()
    ana = np.real(np.fft.ifft(exact(steps * dt)[:, 0]))
    return np.linalg.norm(num - ana) / np.linalg.norm(ana)


def test_zener_matches_its_exact_solution():
    # Measured (CPU, float64), ts = 2: 7.0e-4 at dt 0.2, 1.76e-4 at dt 0.1.
    coarse, fine = _zener_error(0.2, 2.0), _zener_error(0.1, 2.0)
    assert fine <= 2.5e-4
    assert 3.6 <= coarse / fine <= 4.4


def test_fast_relaxation_stays_stable():
    # ts = 0.05, eight times below dt = 0.4 (forward Euler would need dt < 0.1):
    # measured 6.7e-3 at dt 0.4, 1.7e-3 at dt 0.2.
    coarse, fine = _zener_error(0.4, 0.05), _zener_error(0.2, 0.05)
    assert coarse <= 1e-2 and 3.6 <= coarse / fine <= 4.4


# ---- GSLS visco-elasticity against the shipped ViscoElastic ---------------------

def gsls_pde(tau_sigma, name="SymGSLS"):
    """ViscoElastic's system, models in its prepared form (unrelaxed Lamé
    parameters, rho, the relaxation strengths P_l = pi_R tp_l, M_l = mu_R ts_l)."""
    L = len(tau_sigma)
    lam, mu, rho, lam2mu = (Model(n) for n in ("lam", "mu", "rho", "lam2mu"))
    P, M = [Model(f"P{m}") for m in range(L)], [Model(f"M{m}") for m in range(L)]
    vx = Field("vx", time_order=1, stagger="x", source=True, receiver=True)
    vz = Field("vz", time_order=1, stagger="z", source=True, receiver=True)
    sxx = Field("sxx", time_order=1, source=True, receiver=True)
    szz = Field("szz", time_order=1, source=True, receiver=True)
    sxz = Field("sxz", time_order=1, stagger=("z", "x"), source=True)
    rxx = [Field(f"rxx{m}", time_order=1) for m in range(L)]
    rzz = [Field(f"rzz{m}", time_order=1) for m in range(L)]
    rxz = [Field(f"rxz{m}", time_order=1, stagger=("z", "x")) for m in range(L)]
    theta = vx.dx + vz.dz
    eqs = [Eq(vz.dt, (szz.dz + sxz.dx) / rho), Eq(vx.dt, (sxx.dx + sxz.dz) / rho)]
    for m, t in enumerate(tau_sigma):
        t = sympy.Float(t)
        eqs += [Eq(rxx[m].dt, -(rxx[m] + P[m] * theta - 2 * M[m] * vz.dz) / t),
                Eq(rzz[m].dt, -(rzz[m] + P[m] * theta - 2 * M[m] * vx.dx) / t),
                Eq(rxz[m].dt, -(rxz[m] + M[m] * (vx.dz + vz.dx)) / t)]
    eqs += [Eq(szz.dt, lam2mu * vz.dz + lam * vx.dx + sum(rzz)),
            Eq(sxx.dt, lam2mu * vx.dx + lam * vz.dz + sum(rxx)),
            Eq(sxz.dt, mu * (vx.dz + vz.dx) + sum(rxz))]
    return PDE(eqs, name=name)


@pytest.mark.parametrize("n_sls", [1, 3])
def test_gsls_matches_visco_elastic(n_sls):
    # The same scheme (trapezoidal memory variables, the stress reading their
    # mean); the products are associated differently. Measured (RTX 6000 Ada,
    # 3 SLS, nt 400): record 6.3e-7, gradients 0.9e-7 .. 8.3e-7 (vp .. Qs).
    hand = ViscoElastic(4, device=DEV, n_sls=n_sls)
    gen = gsls_pde(hand._tau_sigma_floats).build(2)(4, device=DEV)
    shape, nt, dt = (40, 48), 300, 1e-3
    g = torch.Generator().manual_seed(0)
    vp = 2000 + 600 * torch.rand(shape, generator=g)
    rho = 1800 + 400 * torch.rand(shape, generator=g)
    qp, qs = 20 + 30 * torch.rand(shape, generator=g), 10 + 20 * torch.rand(shape, generator=g)

    def run(eq, prepare):
        raw = [m.clone().to(DEV).requires_grad_(True) for m in (vp, vp / 1.9, rho, qp, qs)]
        solver = PropTorch(eq, shape=shape, dh=10.0, dt=dt, nt=nt, dev=DEV, impl="eager", abcn=12,
                           memory=Full(), source_type=["sxx", "szz"], receiver_type=["vx", "vz"],
                           eager_options=EagerOptions(use_compile=False))
        d = solver(ricker(nt, dt, fm=12.0, delay=0.08), np.array([[20, 6]]),
                   np.array([[[ix, 3] for ix in range(2, 46, 3)]]),
                   models=hand.prepare_models(raw) if prepare else raw)
        d.square().sum().backward()
        return d.detach(), [m.grad for m in raw]

    d_hand, g_hand = run(hand, False)
    d_gen, g_gen = run(gen, True)
    assert _rel(d_gen, d_hand) <= 5e-6
    for name, a, b in zip(("vp", "vs", "rho", "Qp", "Qs"), g_gen, g_hand):
        assert _rel(a, b) <= 1e-5, name


# ---- memory modes and compile --------------------------------------------------

SHAPE = (40, 48)
ZENER, COUPLED = zener_pde().build(2), coupled_pde().build(2)


def _bump():
    z, x = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    return torch.tensor(np.exp(-((z - 20) / 8.0) ** 2 - ((x - 24) / 8.0) ** 2), dtype=torch.float32)


def _zener_models():
    b = _bump()
    return [m.to(DEV).requires_grad_(True) for m in (1.0 + 0.2 * b, 1.0 + 0.1 * b, 0.2 + 0.1 * b, 1.0 + 0 * b)]


def _coupled_models():
    b = _bump()
    return [m.to(DEV).requires_grad_(True) for m in (1.0 + 0.2 * b, 1.0 + 0.1 * b, 0.05 + 0.02 * b)]


def _grads(cls, make_models, memory=None, nt=200, **compile_kw):
    models = make_models()
    dt = 0.2
    solver = PropTorch(cls(4, device=DEV), shape=SHAPE, dh=1.0, dt=dt, nt=nt, dev=DEV, impl="eager", abcn=12,
                       memory=Full() if memory is None else memory, source_type=["p"], receiver_type=["p"],
                       eager_options=EagerOptions(**{"use_compile": False, **compile_kw}))
    ring = [[int(24 + 18 * np.cos(a)), int(20 + 15 * np.sin(a))] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    d = solver(ricker(nt, dt, fm=0.05, delay=30.0), np.array([[18, 16]]), np.array([ring]), models=models,
               max_vel=1.2, pml_freq=0.05)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


def test_damping_runs_with_checkpoints_and_refuses_boundary_saving():
    d_ref, g_ref = _grads(ZENER, _zener_models)
    assert torch.isfinite(d_ref).all() and d_ref.abs().max() > 0
    d, g = _grads(ZENER, _zener_models, memory=Ckpt())
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a, b) <= 1e-5
    with pytest.raises(NotImplementedError, match=r"-r/ts"):
        _grads(ZENER, _zener_models, memory=BoundarySaving(storage="gpu" if DEV == "cuda" else "cpu"))


def test_coupling_alone_keeps_boundary_saving():
    # Stepped backwards, p's update reads the mean of q again (q is inverted
    # first in its sub-step), so the leapfrog stays reversible.
    d_ref, g_ref = _grads(COUPLED, _coupled_models)
    assert torch.isfinite(d_ref).all() and d_ref.abs().max() > 0
    d, g = _grads(COUPLED, _coupled_models, memory=BoundarySaving(storage="gpu" if DEV == "cuda" else "cpu"))
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a, b) <= 1e-4


@requires_compile
def test_local_terms_compile_in_one_graph():
    d_ref, g_ref = _grads(ZENER, _zener_models, nt=100)
    d_c, g_c = _grads(ZENER, _zener_models, nt=100, use_compile=True, compile_fullgraph=True,
                      compile_backend="aot_eager")
    assert _rel(d_c, d_ref) <= 1e-5
    for a, b in zip(g_ref, g_c):
        assert _rel(b, a) <= 1e-4
