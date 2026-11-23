"""A damping ``s * f.dt`` in a ``sweep.symbolic`` second-order equation.

``p.dt2 = vp**2 * laplace(p) - eta * p.dt`` takes ``p.dt`` centred in time,
``(p_next - p_prev) / (2 dt)``. Pinned here: the refusals; the generated step
against the update written by hand; zero damping reproducing the undamped class
bit for bit; the damped line against its exact solution, second order in
``dt``; the memory modes (boundary saving refused) and the compiled step.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy
import torch

from conftest import requires_compile, ricker
from sweep.propagator.options import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Eq, Field, Model, laplace

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = old


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


vp, eta = Model("vp"), Model("eta")
p = Field("p", source=True, receiver=True)


def damped_pde(name="SymDamped", damped=True, line=False, **kw):
    lap = p.dx2 if line else laplace(p)
    lhs = p.dt2 + eta * p.dt if damped else p.dt2
    models = [vp, eta] if damped else [vp]
    return PDE([Eq(lhs, vp**2 * lap)], name=name, models=models, **kw)


# ---- lowering and refusals ------------------------------------------------------

def test_damping_is_written_as_a_term():
    cls = damped_pde().build(2)
    assert cls.DAMPING_TERMS == ("-eta*p.dt",)
    src = damped_pde().source(2)
    assert "'1/(eta*dt/2 + 1)'" in src and "'-eta*dt/2 + 1'" in src
    assert damped_pde("SymPlain", damped=False).build(2).DAMPING_TERMS == ()


def _refused():
    q = Field("q")
    return [
        ([Eq(p.dt2, vp**2 * laplace(p) - eta * q.dt), Eq(q.dt2, laplace(q))],
         NotImplementedError, "first time derivative of another field"),
        ([Eq(p.dt2, vp**2 * laplace(p) - p * p.dt)], ValueError, "not linear"),
        ([Eq(p.dt2, vp**2 * laplace(p) - eta * p.dx2 * p.dt)], ValueError, "not linear"),
        ([Eq(p.dt2, vp**2 * laplace(p) - sympy.Symbol("c") * p.dt)], ValueError, "neither a Model"),
    ]


@pytest.mark.parametrize("k", range(4))
def test_unsupported_damping_is_refused(k):
    equations, error, match = _refused()[k]
    with pytest.raises(error, match=match):
        PDE(equations, name="SymBad").build(2)


# ---- one step against the update written by hand --------------------------------

def test_step_matches_reference():
    damped = damped_pde("SymDampedStep", pml="none").build(2)(8, device=DEV)
    plain = damped_pde("SymPlainStep", damped=False, pml="none").build(2)(8, device=DEV)
    g = torch.Generator().manual_seed(3)
    W = [torch.randn(2, 1, 30, 34, generator=g).to(DEV) for _ in range(2)]
    v = (0.8 + 0.4 * torch.rand(30, 34, generator=g)).to(DEV)
    e = (2.0 * torch.rand(30, 34, generator=g)).to(DEV)
    dt = 0.3
    out = damped.func(W, [v, e], dt, 1.0, None)[0]
    u = plain.func(W, [v], dt, 1.0, None)[0]                  # 2 p - p_prev + dt^2 vp^2 lap p
    # (p_next - 2 p + p_prev)/dt^2 = vp^2 lap p - eta (p_next - p_prev)/(2 dt)
    ref = (u + e * dt / 2 * W[1]) / (1 + e * dt / 2)
    assert _rel(out, ref) <= 1e-6


# ---- zero damping is the undamped step ------------------------------------------

SHAPE = (40, 48)


def _bump():
    z, x = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    return torch.tensor(np.exp(-((z - 20) / 8.0) ** 2 - ((x - 24) / 8.0) ** 2), dtype=torch.float32)


def _run(cls, models, memory=None, nt=200, free_surface=False, **compile_kw):
    models = [m.clone().to(DEV).requires_grad_(True) for m in models]
    dt = 0.2
    solver = PropTorch(cls(4, device=DEV), shape=SHAPE, dh=1.0, dt=dt, nt=nt, dev=DEV, impl="eager", abcn=12,
                       memory=Full() if memory is None else memory, free_surface=free_surface,
                       eager_options=EagerOptions(**{"use_compile": False, **compile_kw}))
    ring = [[int(24 + 18 * np.cos(a)), int(20 + 15 * np.sin(a))] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    d = solver(ricker(nt, dt, fm=0.05, delay=30.0), np.array([[18, 16]]), np.array([ring]), models=models,
               max_vel=1.2, pml_freq=0.05)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


def test_zero_damping_reproduces_the_undamped_class():
    # Both factors are exactly 1 at eta = 0, so every product is the undamped one.
    v = 1.0 + 0.2 * _bump()
    d_plain, (g_plain,) = _run(damped_pde("SymPlainRun", damped=False).build(2), [v])
    d_zero, (g_zero, _) = _run(damped_pde("SymZeroRun").build(2), [v, 0 * v])
    assert torch.equal(d_zero, d_plain)
    assert torch.equal(g_zero, g_plain)


# ---- a damped line against its exact solution -----------------------------------

def _line_error(dt, e, t=60.0, n=512, c=1.0):
    """A Gaussian at rest on a line far from the edges, float64, against the
    exact solution of the semi-discrete-in-time equation: per wavenumber,
    P'' + e P' = c^2 L(k) P with L(k) the symbol of the class's own stencil."""
    eq = damped_pde("SymDampedLine", pml="none", line=True).build(2)(8)
    k = eq.laplace_kernels
    eq.laplace_kernels = tuple(a.double() for a in k) if isinstance(k, tuple) else k.double()
    ones = torch.ones(1, n, dtype=torch.float64)

    def step(f, f_prev, dt, e):
        with torch.no_grad():
            out = eq.func([torch.as_tensor(f).view(1, 1, 1, -1), torch.as_tensor(f_prev).view(1, 1, 1, -1)],
                          [c * ones, e * ones], dt, 1.0, None)
        return out[0][0, 0, 0].numpy()

    delta = np.zeros(n)
    delta[n // 2] = 1.0
    stencil = step(delta, delta, 1.0, 0.0) - delta                # dt = 1, p_prev = p: lap of the impulse
    L = np.real(np.fft.fft(np.roll(stencil, -n // 2)))
    x = np.arange(n) - n / 2
    p0 = np.exp(-(x / 6.0) ** 2)
    disc = np.sqrt((e * e + 4 * c * c * L).astype(complex))
    r1, r2 = (-e + disc) / 2, (-e - disc) / 2
    P0 = np.fft.fft(p0)
    A, B = -r2 * P0 / (r1 - r2), r1 * P0 / (r1 - r2)

    def exact(time):
        return np.real(np.fft.ifft(A * np.exp(r1 * time) + B * np.exp(r2 * time)))

    f, f_prev = p0, exact(-dt)
    steps = int(round(t / dt))
    for _ in range(steps):
        f, f_prev = step(f, f_prev, dt, e), f
    ana = exact(steps * dt)
    return np.linalg.norm(f - ana) / np.linalg.norm(ana)


def test_damped_line_matches_its_exact_solution():
    # Measured (CPU, float64), eta = 0.05 (amplitude e^-1.5 at t = 60): 8.5e-4 at
    # dt 0.2, 2.12e-4 at dt 0.1.
    coarse, fine = _line_error(0.2, 0.05), _line_error(0.1, 0.05)
    assert fine <= 2.5e-4
    assert 3.6 <= coarse / fine <= 4.4


# ---- memory modes, free surface, compile ----------------------------------------

DAMPED = damped_pde().build(2)
DAMPED_FS = damped_pde("SymDampedFS", free_surface="zero").build(2)


def _models():
    b = _bump()
    return [1.0 + 0.2 * b, 0.02 + 0.03 * b]


def test_damping_runs_with_checkpoints_and_refuses_boundary_saving():
    for cls, fs in ((DAMPED, False), (DAMPED_FS, True)):
        d_ref, g_ref = _run(cls, _models(), free_surface=fs)
        assert torch.isfinite(d_ref).all() and d_ref.abs().max() > 0
        assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in g_ref)
        d, g = _run(cls, _models(), memory=Ckpt(), free_surface=fs)
        assert torch.equal(d, d_ref)
        for a, b in zip(g, g_ref):
            assert _rel(a, b) <= 1e-5
    with pytest.raises(NotImplementedError, match=r"eta\*p\.dt"):
        _run(DAMPED, _models(), memory=BoundarySaving(storage="gpu" if DEV == "cuda" else "cpu"))


@requires_compile
def test_damping_compiles_in_one_graph():
    d_ref, g_ref = _run(DAMPED, _models(), nt=100)
    d_c, g_c = _run(DAMPED, _models(), nt=100, use_compile=True, compile_fullgraph=True,
                    compile_backend="aot_eager")
    assert _rel(d_c, d_ref) <= 1e-5
    for a, b in zip(g_ref, g_c):
        assert _rel(b, a) <= 1e-4
