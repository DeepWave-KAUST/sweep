"""First, mixed and model derivatives in ``sweep.symbolic`` second-order equations.

``p.dx`` (the centred gradient with the CPML stretch of ``p.dx2``), ``p.dz.dx``
(the centred gradient of that stretched derivative along the other axis, as
``AcousticTTI`` takes its mixed term) and ``rho.dx`` (a model's centred
gradient, made once per rollout). Pinned here: the expression rules and the
refusals; the elliptic TTI system against the shipped ``AcousticTTI`` (at
``epsilon == delta`` its S term vanishes); variable density at constant
density against ``Acoustic`` and ``Acoustic3D``; variable density with a
mixed term against a manufactured solution (float64), second order in ``dt``;
the memory modes and the compiled step.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import sympy
import torch

from conftest import requires_compile, ricker
from sweep.equations import Acoustic, Acoustic3D
from sweep.equations.qP_tti import AcousticTTI
from sweep.propagator.options import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Coefficient, Eq, Field, Model, laplace

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = old


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


vp, rho, eps, theta, kap = Model("vp"), Model("rho"), Model("epsilon"), Model("theta"), Model("kap")
p = Field("p", source=True, receiver=True)


def var_density_pde(name="SymVarDen", mixed=False, **kw):
    """``p_tt = vp^2 rho div(grad(p) / rho)``, expanded; optionally a mixed term."""
    rhs = vp**2 * (laplace(p) - (rho.dx * p.dx + rho.dz * p.dz) / rho)
    if mixed:
        rhs += kap * p.dz.dx
    return PDE([Eq(p.dt2, rhs)], name=name, **kw)


def var_density_3d_pde(name="SymVarDen3D"):
    return PDE([Eq(p.dt2, vp**2 * (laplace(p) - (rho.dx * p.dx + rho.dy * p.dy + rho.dz * p.dz) / rho))],
               name=name)


def tti_pde(name="SymEllTTI"):
    """``AcousticTTI`` at ``delta == epsilon`` (no S term): elliptic TTI."""
    c, s = sympy.cos(theta), sympy.sin(theta)
    return PDE([Eq(p.dt2, vp**2 * (((1 + 2 * eps) * c**2 + s**2) * p.dx2 + ((1 + 2 * eps) * s**2 + c**2) * p.dz2
                                   - 2 * eps * sympy.sin(2 * theta) * p.dz.dx))], name=name)


# ---- expressions, lowering and refusals -----------------------------------------

def test_mixed_and_model_derivatives_are_written_as_terms():
    assert p.dx.dz == p.dz.dx and str(p.dx.dz) == "p.dz.dx"
    src = var_density_pde().source(2)
    assert "_dx_rho = once(invariants, '_dx_rho', lambda: self.gradient(rho, h, -1, kernels=self.grad_kernels))" in src
    assert "_gx_p = cpml_stretch(_dx_p, _psix, _ax, _bx)" in src      # the stretch of p.dx2, psi before its update
    src = tti_pde().source(2)
    assert "_dzx_p = self.gradient(_gx_p, h, -2, kernels=self.grad_kernels)" in src
    assert "_dz_p = self.gradient(p, h, -2" in var_density_pde("SymVDNone", pml="none").source(2)


def _refused():
    q = Field("q", receiver=True)
    vx = Field("vx", time_order=1, stagger="x")
    w = Field("w", time_order=1, source=True, receiver=True)
    return [
        (lambda: [Eq(p.dt2, vp**2 * p.dz2 + p.dx)], ValueError, "needs p's second derivative along x"),
        (lambda: [Eq(p.dt2, vp**2 * p.dz2 + p.dz.dx)], ValueError, "needs p's second derivative along x"),
        (lambda: [Eq(p.dt2, vp**2 * laplace(p) + q.dx), Eq(q.dt2, q.dz2)], ValueError, "needs q's second"),
        (lambda: [Eq(p.dt2, vp**2 * laplace(p) + p.dx2.dz)], NotImplementedError, "mixed derivatives"),
        (lambda: [Eq(p.dt2, vp**2 * laplace(p) + p.dz.dx.dx)], NotImplementedError, "above the second"),
        (lambda: [Eq(p.dt2, vp**2 * laplace(p) + rho.dx.dx * p.dx)], NotImplementedError, "first derivatives only"),
        (lambda: [Eq(p.dt2, vp**2 * laplace(p) + p.dx * p.dz)], ValueError, "not linear"),
        (lambda: [Eq(vx.dt, -w.dx), Eq(w.dt, -vx.dx + w.dz.dx)], NotImplementedError, "first-order system"),
        (lambda: [Eq(vx.dt, -w.dx / rho), Eq(w.dt, -rho.dx * vx.dx)], NotImplementedError, "first-order system"),
        (lambda: [Eq(p.dt2, vp**2 * laplace(p) + rho.dx)], ValueError, "contains no field"),
    ]


@pytest.mark.parametrize("k", range(10))
def test_unsupported_terms_are_refused(k):
    equations, error, match = _refused()[k]
    with pytest.raises(error, match=match):
        PDE(equations(), name="SymBad").build(2)


def test_a_coefficient_and_a_2d_build_refuse_what_they_cannot_take():
    with pytest.raises(ValueError, match="contains a derivative"):
        Coefficient("gx", rho.dx / rho)
    pde = var_density_3d_pde("SymVDy")
    with pytest.raises(ValueError, match="y-derivative in a 2-D build"):
        pde.build(2)


# ---- against the shipped equations ----------------------------------------------

SHAPE, NT, DT, DH = (40, 48), 250, 1e-3, 10.0


def _run(eq, models, shape=SHAPE, nt=NT, memory=None, src=(20, 20), **compile_kw):
    models = [m.clone().to(DEV).requires_grad_(True) for m in models]
    solver = PropTorch(eq, shape=shape, dh=DH, dt=DT, nt=nt, dev=DEV, impl="eager", abcn=16,
                       memory=Full() if memory is None else memory,
                       eager_options=EagerOptions(**{"use_compile": False, **compile_kw}))
    nx = shape[-1]
    if len(shape) == 2:
        sources, receivers = np.array([[src[1], src[0]]]), np.array([[[ix, 4] for ix in range(3, nx - 3, 4)]])
    else:
        sources = np.array([[src[1], src[1], src[0]]])
        receivers = np.array([[[ix, nx // 2, 4] for ix in range(3, nx - 3, 4)]])
    d = solver(ricker(nt, DT, fm=15.0, delay=0.08), sources, receivers, models=models)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


def _models(shape=SHAPE, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = 2000 + 500 * torch.rand(shape, generator=g)
    e = 0.1 + 0.1 * torch.rand(shape, generator=g)
    t = 0.3 + 0.4 * torch.rand(shape, generator=g)
    return v, e, t


def test_elliptic_tti_matches_acoustic_tti():
    # The same operators in the same places; the products of models and dt are
    # associated differently. Measured (RTX 6000 Ada): record 2.4e-6, gradients
    # 8.4e-7 .. 1.3e-6.
    v, e, t = _models()
    d_ref, g_ref = _run(AcousticTTI(4, device=DEV), [v, e, e, t])
    d_gen, g_gen = _run(tti_pde().build(2)(4, device=DEV), [v, e, t])
    assert _rel(d_gen, d_ref) <= 5e-6
    for name, a, b in (("vp", g_gen[0], g_ref[0]), ("epsilon", g_gen[1], g_ref[1] + g_ref[2]),
                       ("theta", g_gen[2], g_ref[3])):
        assert _rel(a, b) <= 5e-6, name


def test_constant_density_matches_acoustic():
    # rho.dx of a constant model is round-off (4e-7 of rho / h), not 0, so not
    # bit for bit. Measured: record 1.6e-6, vp gradient 6.8e-7.
    v, _, _ = _models()
    d_ref, (g_ref,) = _run(Acoustic(4, device=DEV), [v])
    d_gen, (g_gen, _) = _run(var_density_pde().build(2)(4, device=DEV), [v, 0 * v + 2000.0])
    assert _rel(d_gen, d_ref) <= 5e-6
    assert _rel(g_gen, g_ref) <= 5e-6


def test_constant_density_matches_acoustic3d():
    # In 3-D both take torch.gradient, which gives exactly 0 on a constant
    # model: bit for bit.
    shape = (18, 20, 22)
    v = 2000 + 500 * torch.rand(shape, generator=torch.Generator().manual_seed(1))
    d_ref, (g_ref,) = _run(Acoustic3D(4, device=DEV), [v], shape=shape, nt=60, src=(9, 10))
    d_gen, (g_gen, _) = _run(var_density_3d_pde().build(3)(4, device=DEV), [v, 0 * v + 2000.0],
                             shape=shape, nt=60, src=(9, 10))
    assert torch.equal(d_gen, d_ref)
    assert torch.equal(g_gen, g_ref)


# ---- a manufactured solution -----------------------------------------------------

def _exact(k):
    """The FD coefficients (rationals) of a float32 stencil, in float64: the
    float32 ones carry a residue (the 4th-order second derivative sums to
    7e-8, not 0) that 1 / h^2 makes the error floor of a float64 run."""
    flat = [float(Fraction(float(v)).limit_denominator(5000)) for v in k.flatten().tolist()]
    return torch.tensor(flat, dtype=torch.float64).view(k.shape)


def _manufactured_error(n, so=8, t_end=0.3, cfl=0.2):
    """``p = cos(7 t) exp(-r^2 / 0.01)`` on the unit square with smooth vp, rho,
    kap, its residual added as a forcing at each step; relative L2 error at
    t_end, away from the edges (where the model derivatives' halo is zero)."""
    X, Z, T = sympy.symbols("x z t")
    V = 1 + 0.2 * sympy.sin(2 * sympy.pi * X) * sympy.cos(2 * sympy.pi * Z)
    R = 1.5 + 0.5 * sympy.cos(2 * sympy.pi * (X - 0.3)) * sympy.sin(2 * sympy.pi * (Z + 0.1))
    K = 0.3 + 0.1 * sympy.sin(2 * sympy.pi * (X + Z))
    P = sympy.cos(7 * T) * sympy.exp(-((X - 0.5) ** 2 + (Z - 0.48) ** 2) / 0.1**2)
    rhs = (V**2 * (sympy.diff(P, X, 2) + sympy.diff(P, Z, 2)
                   - (sympy.diff(R, X) * sympy.diff(P, X) + sympy.diff(R, Z) * sympy.diff(P, Z)) / R)
           + K * sympy.diff(P, X, Z))
    forcing = sympy.lambdify((X, Z, T), sympy.diff(P, T, 2) - rhs, "numpy")
    exact = sympy.lambdify((X, Z, T), P, "numpy")

    eq = var_density_pde("SymVDManufactured", mixed=True, pml="none").build(2)(so)
    k = eq.laplace_kernels
    eq.laplace_kernels = tuple(_exact(a) for a in k) if isinstance(k, tuple) else _exact(k)
    eq.grad_kernels = {a: _exact(b) for a, b in eq.grad_kernels.items()}
    h = 1.0 / n
    zz, xx = np.meshgrid(np.arange(n) * h, np.arange(n) * h, indexing="ij")
    models = [torch.tensor(sympy.lambdify((X, Z), m, "numpy")(xx, zz), dtype=torch.float64) for m in (V, R, K)]
    dt = cfl * h
    steps = int(round(t_end / dt))

    def grid(a):
        return torch.tensor(a, dtype=torch.float64).view(1, 1, n, n)

    f, f_prev = grid(exact(xx, zz, 0.0)), grid(exact(xx, zz, -dt))
    with torch.no_grad():
        for i in range(steps):
            f, f_prev = eq.func([f, f_prev], models, dt, h, None)[0] + dt**2 * grid(forcing(xx, zz, i * dt)), f
    ana = exact(xx, zz, steps * dt)
    inner = slice(n // 8, -n // 8)
    return np.linalg.norm((f[0, 0].numpy() - ana)[inner, inner]) / np.linalg.norm(ana[inner, inner])


def test_manufactured_solution_converges():
    # Measured (CPU, float64, 8th order, dt = 0.2 h): 9.4e-5 at n 48, 2.3e-5 at
    # n 96, 5.8e-6 at n 192: second order (dt dominates).
    coarse, fine = _manufactured_error(48), _manufactured_error(96)
    assert fine <= 3e-5
    assert 3.6 <= coarse / fine <= 4.4


# ---- memory modes and compile --------------------------------------------------

VARDEN, TTI = var_density_pde().build(2), tti_pde().build(2)


def _vd_models():
    v, e, _ = _models(seed=2)
    return [v, 1800 + 4000 * e]


@pytest.mark.parametrize("which", ["varden", "tti"])
def test_checkpoints_and_boundary_saving_match_full(which):
    cls, models = (VARDEN, _vd_models()) if which == "varden" else (TTI, list(_models(seed=3)))
    d_ref, g_ref = _run(cls(4, device=DEV), models, nt=200)
    assert all(torch.isfinite(g).all() and g.abs().max() > 0 for g in g_ref)
    d, g = _run(cls(4, device=DEV), models, nt=200, memory=Ckpt())
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a, b) <= 1e-5
    # Boundary saving: the gradient as Full() but in the outermost M = 2 cells
    # next to an absorbing face, which re-run the step from zeroed CPML memory
    # (see sweep.propagator._eager_boundary_saving). A model derivative reads
    # the PML cells, so there the rho gradient is 7e-4 off (measured);
    # inside, 1.5e-7 .. 6.2e-7.
    d, g = _run(cls(4, device=DEV), models, nt=200, memory=BoundarySaving(storage="gpu" if DEV == "cuda" else "cpu"))
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a[2:-2, 2:-2], b[2:-2, 2:-2]) <= 1e-5


def test_cpml_strips_match_whole_memory():
    # The stretch reads psi from its strips. Measured: record bit for bit,
    # gradients 2e-9 .. 1.5e-7 (the backward sums in another order).
    v, e, t = _models(seed=4)
    d_ref, g_ref = _run(TTI(4, device=DEV), [v, e, t], nt=150)
    d, g = _run(TTI(4, device=DEV), [v, e, t], nt=150, cpml_strips=True)
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a, b) <= 1e-5


@requires_compile
def test_mixed_derivatives_compile_in_one_graph():
    d_ref, g_ref = _run(TTI(4, device=DEV), list(_models(seed=3)), nt=60)
    d_c, g_c = _run(TTI(4, device=DEV), list(_models(seed=3)), nt=60, use_compile=True, compile_fullgraph=True,
                    compile_backend="aot_eager")
    assert _rel(d_c, d_ref) <= 1e-5
    for a, b in zip(g_ref, g_c):
        assert _rel(b, a) <= 1e-4
