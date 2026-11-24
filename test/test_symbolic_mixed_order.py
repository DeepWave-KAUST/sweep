"""Memory fields: first-order fields in a ``sweep.symbolic`` second-order system.

``p.dt2 = vp**2 * (laplace(p) + r)``, ``r.dt = -r / tau + k * laplace(p)``, a
standard linear solid. ``r`` sits at the half time levels and ``p`` reads the
mean of the two around its time. Pinned here: the refusals; the generated step
against the update written by hand; a memory field that stays zero
reproducing the plain class bit for bit; the line against its exact solution,
second order in ``dt``; the memory modes (boundary saving refused), 3-D and
the compiled step.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
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


vp, tau, k = Model("vp"), Model("tau"), Model("k")
p = Field("p", source=True, receiver=True)
r = Field("r", time_order=1, receiver=True)


def sls_pde(name="SymSLS", line=False, decay=True, **kw):
    lap = p.dx2 if line else laplace(p)
    r_rhs = (-r / tau if decay else 0) + k * lap
    models = [vp, tau, k] if decay else [vp, k]
    return PDE([Eq(p.dt2, vp**2 * (lap + r)), Eq(r.dt, r_rhs)], name=name, models=models, **kw)


def plain_pde(name="SymPlainMem", **kw):
    return PDE([Eq(p.dt2, vp**2 * laplace(p))], name=name, models=[vp], **kw)


# ---- lowering and refusals ------------------------------------------------------

def test_memory_field_is_generated():
    cls = sls_pde().build(2)
    assert cls.MEMORY_FIELDS == ("r",)
    assert cls.DAMPING_TERMS == ("-r/tau",)
    assert [f.name for f in cls.FIELD_SPECS][:3] == ["p", "p_prev", "r"]
    src = sls_pde().source(2)
    assert "_r_mid = 0.5 * (r + _r_next)" in src
    assert "_w1" not in src                    # laplace(p) summed once for both equations
    assert sls_pde("SymSLSNoDecay", decay=False).build(2).DAMPING_TERMS == ()


def _refused():
    q = Field("q", time_order=1)
    s = Field("s", time_order=1, stagger="x")
    lap = laplace(p)
    return [
        ([Eq(p.dt2, vp**2 * (lap + s.dx)), Eq(s.dt, k * p.dx)], NotImplementedError, "staggered"),
        ([Eq(p.dt2, vp**2 * (lap + r.dx2)), Eq(r.dt, k * lap)], NotImplementedError, "a derivative of r"),
        ([Eq(p.dt2, vp**2 * (lap + r)), Eq(r.dt, k * lap + q), Eq(q.dt, k * lap)],
         NotImplementedError, "another memory field"),
        ([Eq(p.dt2, vp**2 * (lap + r)), Eq(r.dt, k * r * lap)], ValueError, "not linear"),
        ([Eq(p.dt2 + r.dt, vp**2 * lap), Eq(r.dt, k * lap)], NotImplementedError, "another field"),
        ([Eq(p.dt2, vp**2 * (lap + r)), Eq(r.dt, k * p.dt)], ValueError, "more than one time derivative"),
        ([Eq(p.dt2, vp**2 * (lap + r))], ValueError, "no equation of its own"),
    ]


@pytest.mark.parametrize("case", range(7), ids=["staggered", "derivative", "other-memory", "nonlinear",
                                                "other-dt", "p-dt", "no-equation"])
def test_unsupported_memory_fields_are_refused(case):
    equations, error, match = _refused()[case]
    with pytest.raises(error, match=match):
        PDE(equations, name="SymBadMem").build(2)


# ---- one step against the update written by hand --------------------------------

def test_step_matches_reference():
    sls = sls_pde("SymSLSStep", pml="none").build(2)(8, device=DEV)
    plain = plain_pde("SymPlainMemStep", pml="none").build(2)(8, device=DEV)
    g = torch.Generator().manual_seed(5)
    p_now, p_prev, r_now = (torch.randn(2, 1, 30, 34, generator=g).to(DEV) for _ in range(3))
    v = (0.8 + 0.4 * torch.rand(30, 34, generator=g)).to(DEV)
    t = (2.0 + torch.rand(30, 34, generator=g)).to(DEV)
    kk = (-0.1 * torch.rand(30, 34, generator=g)).to(DEV)
    dt = 0.3
    out = sls.func([p_now, p_prev, r_now], [v, t, kk], dt, 1.0, None)
    ones = torch.ones_like(v)
    lap = plain.func([p_now, p_now], [ones], 1.0, 1.0, None)[0] - p_now
    r_next = ((1 - dt / (2 * t)) * r_now + dt * kk * lap) / (1 + dt / (2 * t))
    p_next = 2 * p_now - p_prev + dt**2 * v**2 * (lap + (r_now + r_next) / 2)
    assert _rel(out[2], r_next) <= 1e-6
    assert _rel(out[0], p_next) <= 1e-6
    assert torch.equal(out[1], p_now)


# ---- a memory field that stays zero is the plain class --------------------------

SHAPE = (40, 48)


def _bump():
    z, x = np.meshgrid(np.arange(SHAPE[0]), np.arange(SHAPE[1]), indexing="ij")
    return torch.tensor(np.exp(-((z - 20) / 8.0) ** 2 - ((x - 24) / 8.0) ** 2), dtype=torch.float32)


def _run(cls, models, memory=None, nt=200, shape=SHAPE, free_surface=False, **compile_kw):
    models = [m.clone().to(DEV).requires_grad_(True) for m in models]
    dt = 0.2
    solver = PropTorch(cls(4, device=DEV), shape=shape, dh=1.0, dt=dt, nt=nt, dev=DEV, impl="eager", abcn=12,
                       memory=Full() if memory is None else memory, free_surface=free_surface,
                       eager_options=EagerOptions(**{"use_compile": False, **compile_kw}))
    rec = [[int(24 + 18 * np.cos(a)), int(20 + 15 * np.sin(a))] for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    if len(shape) == 3:                   # (x, y, z) in a (nz, ny, nx) grid
        rec = [[ix, 10, 4] for ix in range(3, shape[-1] - 3, 3)]
        src = [[10, 10, 8]]
    else:
        src = [[18, 16]]
    d = solver(ricker(nt, dt, fm=0.05, delay=30.0), np.array(src), np.array([rec]), models=models,
               max_vel=1.2, pml_freq=0.05)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


def test_zero_memory_reproduces_the_plain_class():
    # k = 0 keeps r exactly zero, and adding vp**2 dt**2 * 0 changes no bit.
    v = 1.0 + 0.2 * _bump()
    d_plain, (g_plain,) = _run(plain_pde().build(2), [v])
    d_zero, (g_zero, _, _) = _run(SLS, [v, 5.0 + 0 * v, 0 * v])
    assert torch.equal(d_zero, d_plain)
    assert torch.equal(g_zero, g_plain)


# ---- the line against its exact solution ----------------------------------------

def _line_error(dt, t_end=60.0, n=512, c=1.0, t_relax=5.0, kk=-0.04):
    """A Gaussian at rest on a line, float64, against the exact solution of the
    semi-discrete-in-time system: per wavenumber, with L(k) the symbol of the
    class's own stencil, P'' = c^2 (L P + R), R' = -R / tau + k L P."""
    eq = sls_pde("SymSLSLine", pml="none", line=True).build(2)(8)
    kern = eq.laplace_kernels
    eq.laplace_kernels = tuple(a.double() for a in kern) if isinstance(kern, tuple) else kern.double()
    ones = torch.ones(1, n, dtype=torch.float64)

    def step(f, f_prev, rr, dt, kv):
        w = [torch.as_tensor(a).view(1, 1, 1, -1) for a in (f, f_prev, rr)]
        with torch.no_grad():
            out = eq.func(w, [c * ones, t_relax * ones, kv * ones], dt, 1.0, None)
        return out[0][0, 0, 0].numpy(), out[2][0, 0, 0].numpy()

    delta = np.zeros(n)
    delta[n // 2] = 1.0
    stencil = step(delta, delta, 0 * delta, 1.0, 0.0)[0] - delta     # dt = 1, p_prev = p, k = 0
    L = np.real(np.fft.fft(np.roll(stencil, -n // 2)))
    x = np.arange(n) - n / 2
    P0 = np.fft.fft(np.exp(-(x / 6.0) ** 2))
    # y = (P, P', R) per wavenumber; y' = M y.
    M = np.zeros((n, 3, 3))
    M[:, 0, 1] = 1.0
    M[:, 1, 0], M[:, 1, 2] = c * c * L, c * c
    M[:, 2, 0], M[:, 2, 2] = kk * L, -1.0 / t_relax

    def exact(time):
        y = np.stack([scipy.linalg.expm(m * time)[:, 0] for m in M]) * P0[:, None]
        return np.real(np.fft.ifft(y[:, 0])), np.real(np.fft.ifft(y[:, 2]))

    f, f_prev, rr = exact(0.0)[0], exact(-dt)[0], exact(-dt / 2)[1]
    steps = int(round(t_end / dt))
    for _ in range(steps):
        (f_next, rr) = step(f, f_prev, rr, dt, kk)
        f, f_prev = f_next, f
    ana = exact(steps * dt)[0]
    return np.linalg.norm(f - ana) / np.linalg.norm(ana)


def test_line_matches_its_exact_solution():
    # Measured (CPU, float64), tau = 5, k tau = -0.2: 9.28e-4 at dt 0.2,
    # 2.32e-4 at dt 0.1.
    coarse, fine = _line_error(0.2), _line_error(0.1)
    assert fine <= 2.6e-4
    assert 3.6 <= coarse / fine <= 4.4


# ---- memory modes, free surface, 3-D, compile -----------------------------------

SLS = sls_pde().build(2)
SLS_FS = sls_pde("SymSLSFS", free_surface="zero").build(2)


def _models():
    b = _bump()
    return [1.0 + 0.2 * b, 4.0 + 2.0 * b, -0.03 - 0.02 * b]


def test_memory_runs_with_checkpoints_and_refuses_boundary_saving():
    d_fs, g_fs = _run(SLS_FS, _models(), free_surface=True)
    assert torch.isfinite(d_fs).all() and d_fs.abs().max() > 0
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in g_fs)
    d_ref, g_ref = _run(SLS, _models())
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in g_ref)
    d, g = _run(SLS, _models(), memory=Ckpt())
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a, b) <= 1e-5
    bs = BoundarySaving(storage="gpu" if DEV == "cuda" else "cpu")
    with pytest.raises(NotImplementedError, match=r"-r/tau"):
        _run(SLS, _models(), memory=bs)
    no_decay = sls_pde("SymSLSNoDecayRun", decay=False).build(2)
    with pytest.raises(NotImplementedError, match="memory field r sits at the half time levels"):
        _run(no_decay, [_models()[0], _models()[2]], memory=bs)


def test_zero_memory_reproduces_the_plain_class_3d():
    shape = (18, 20, 22)
    v = torch.full(shape, 1.0)
    d_plain, (g_plain,) = _run(plain_pde("SymPlainMem3").build(3), [v], nt=60, shape=shape)
    d_zero, (g_zero, _, _) = _run(sls_pde("SymSLS3").build(3), [v, 5.0 + 0 * v, 0 * v], nt=60, shape=shape)
    assert torch.equal(d_zero, d_plain)
    assert torch.equal(g_zero, g_plain)


@requires_compile
def test_memory_compiles_in_one_graph():
    d_ref, g_ref = _run(SLS, _models(), nt=100)
    d_c, g_c = _run(SLS, _models(), nt=100, use_compile=True, compile_fullgraph=True,
                    compile_backend="aot_eager")
    assert _rel(d_c, d_ref) <= 1e-5
    for a, b in zip(g_ref, g_c):
        assert _rel(b, a) <= 1e-4
