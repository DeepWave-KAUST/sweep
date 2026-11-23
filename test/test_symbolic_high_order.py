"""Derivatives of order 3 and higher in ``sweep.symbolic``.

``u.dx4``, ``u.dx2.dx2`` or ``u.deriv("x", 4)`` lower to
``sweep.operators.Derivative`` where their result lands on the field they
update: even orders of any field in a second-order equation (a plate's
flexural waves, ``u_tt = -kappa u_xxxx``), odd orders of the other fields in a
first-order system as staggered differences (a dispersive correction,
``p_t = ... v_xxx``), even orders of a first-order field in its own equation as
dissipative terms (hyperdiffusion, ``T_t = -beta T_xxxx``). Pinned here: the
expression layer, the refusals, each generated step against the same update
written by hand, the plate and hyperdiffusion against their exact solutions
(the Fourier factor ``cos(sqrt(kappa) k^2 t)`` and ``exp(-beta k^4 t)``), a
gradient against finite differences, the memory modes and the compiled step.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy
import torch

from conftest import requires_compile, ricker
from sweep.operators import Derivative
from sweep.propagator.options import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Eq, Field, Model, laplace
from sweep.symbolic._expr import D1, D2, Dn

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KW = dict(max_vel=1.0, pml_freq=0.1)        # profiles from these, not from the models


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32 = old


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


def plate_pde(name="SymPlate", pml="cpml", axes="zx"):
    kappa = Model("kappa")
    u = Field("u", source=True, receiver=True)
    return PDE([Eq(u.dt2, -kappa * sum(u.deriv(ax, 4) for ax in axes))], name=name, pml=pml)


def dispersive_pde(name="SymDispersive", pml="cpml"):
    """Long waves with a dispersive correction: omega^2 = c^2 k^2 (1 + beta k^2)."""
    rho, c, beta = Model("rho"), Model("c"), Model("beta")
    p = Field("p", time_order=1, source=True, receiver=True)
    vx = Field("vx", time_order=1, stagger="x", receiver=True)
    vz = Field("vz", time_order=1, stagger="z", receiver=True)
    return PDE([Eq(vx.dt, -p.dx / rho), Eq(vz.dt, -p.dz / rho),
                Eq(p.dt, -rho * c**2 * (vx.dx + vz.dz - beta * (vx.dx3 + vz.dz3)))], name=name, pml=pml)


def hyper_pde(name="SymHyper", pml="cpml", axes="zx", kappa_term=True):
    kappa, beta = Model("kappa"), Model("beta")
    t = Field("T", time_order=1, source=True, receiver=True)
    rhs = -beta * sum(t.deriv(ax, 4) for ax in axes)
    if kappa_term:
        rhs = kappa * laplace(t) + rhs
    return PDE([Eq(t.dt, rhs)], name=name, pml=pml)


# ---- the expression layer -------------------------------------------------------

def test_spellings_of_a_high_order_derivative():
    u = Field("u")
    assert u.dx3 == u.dx2.dx == u.dx.dx.dx == u.dx.dx2 == u.deriv("x", 3)
    assert isinstance(u.dx3, Dn)
    assert u.dz4 == u.dz2.dz2 == u.deriv("z", 4)
    assert isinstance(u.dx.dx, D2) and isinstance(u.deriv("x"), D1) and isinstance(u.deriv("x", 2), D2)
    assert (u.dx4.order, u.dx4.axis, str(u.dy5)) == (4, "x", "u.dy5")
    assert sympy.latex(u.dz4) == r"\frac{\partial^{4} u}{\partial z^{4}}"
    assert u.dx.dz == u.dz.dx                         # a mixed second derivative (Dm)
    with pytest.raises(NotImplementedError, match="mixed derivatives"):
        u.dx2.dz
    with pytest.raises(NotImplementedError, match="mixed derivatives"):
        u.dx2.deriv("z", 2)
    with pytest.raises(ValueError, match="integer >= 1"):
        u.deriv("x", 0)
    with pytest.raises(ValueError, match="axis"):
        u.deriv("t", 3)
    with pytest.raises(AttributeError, match="dq3"):
        u.dq3


def test_latex_and_terms():
    pde = hyper_pde("SymHyperTex")
    assert r"\frac{\partial^{4} T}{\partial x^{4}}" in pde.latex()
    cls = pde.build(2)
    assert cls.DISSIPATIVE_TERMS == ("kappa*laplace(T)", "-beta*(T.dx4 + T.dz4)")
    assert cls.HIGH_ORDER_STENCILS == ((4, None),)
    assert plate_pde("SymPlateCls").build(2).HIGH_ORDER_STENCILS == ((4, None),)
    disp = dispersive_pde("SymDispCls").build(2)
    assert disp.HIGH_ORDER_STENCILS == ((3, "backward"),)
    # No CPML memory for the third derivatives: one per first derivative only.
    assert sorted(n for n in disp(4).wavefields if n.startswith("m_")) == ["m_p_x", "m_p_z", "m_vx_x", "m_vz_z"]


u2, w2 = Field("u2", source=True, receiver=True), Field("w2")
kap = Model("kap")
q = Field("q", time_order=1, source=True, receiver=True)
qx = Field("qx", time_order=1, stagger="x")
qz = Field("qz", time_order=1, stagger="z")
qflat = Field("qflat", time_order=1)


@pytest.mark.parametrize("equations, error, match", [
    ([Eq(u2.dt2, kap * u2.dx3)], NotImplementedError, "odd derivative in a second-order equation"),
    ([Eq(u2.dt2, kap * u2.dx4 * u2)], ValueError, "not linear"),
    ([Eq(u2.dt2, kap * w2.dx4)], ValueError, "no equation of its own"),
    ([Eq(q.dt, -kap * q.dx3)], NotImplementedError, "odd derivative of q in its own equation"),
    ([Eq(qx.dt, -q.dx), Eq(q.dt, -qx.dx + kap * qx.dx4)], NotImplementedError, "even derivatives of its own field"),
    ([Eq(qflat.dt, -q.dx3), Eq(q.dt, -qflat.dx)], NotImplementedError, "collocated odd derivative"),
    ([Eq(qz.dt, -q.dx3), Eq(q.dt, -qz.dz)], ValueError, "different z positions"),
    ([Eq(q.dt, kap * q.dx4 * q.dx2)], ValueError, "not linear"),
], ids=["odd-second-order", "nonlinear", "no-equation", "odd-own-field", "even-other-field",
        "collocated-odd", "other-axis", "nonlinear-first-order"])
def test_unsupported_high_order_is_refused(equations, error, match):
    with pytest.raises(error, match=match):
        PDE(equations, name="SymBadHigh")


def test_y_derivative_is_refused_in_2d():
    pde = PDE([Eq(u2.dt2, kap * u2.dy4)], name="SymBadHighY")      # fine in 3-D
    pde.build(3)
    with pytest.raises(ValueError, match="y-derivative in a 2-D build"):
        pde.build(2)


# ---- each step against the update written by hand ---------------------------------

def _random(shape, lo, hi, g):
    return (lo + (hi - lo) * torch.rand(shape, generator=g)).to(DEV)


@pytest.mark.parametrize("so", [4, 8])
def test_plate_step_matches_reference(so):
    eq = plate_pde("SymPlateStep", pml="none").build(2)(so, device=DEV)
    g = torch.Generator().manual_seed(1)
    u, up = (torch.randn(2, 1, 30, 34, generator=g).to(DEV) for _ in range(2))
    kappa = _random((30, 34), 0.5, 1.5, g)
    dt, h = 0.05, (1.0, 1.3)
    out = eq.func([u, up], [kappa], dt, h, None)
    d4 = Derivative(4, so, device=DEV)
    ref = 2 * u - up - kappa * dt**2 * (d4.dz(u, h[0]) + d4.dx(u, h[1]))
    assert _rel(out[0], ref) <= 1e-6 and torch.equal(out[1], u)


def test_dispersive_step_matches_reference():
    eq = dispersive_pde("SymDispStep", pml="none").build(2)(8, device=DEV)
    eq.pd.set_spacing((1.0, 1.0))
    g = torch.Generator().manual_seed(2)
    W = dict(zip(eq.wavefields, (torch.randn(2, 1, 30, 34, generator=g).to(DEV) for _ in eq.wavefields)))
    rho, c, beta = (_random((30, 34), lo, hi, g) for lo, hi in ((0.8, 1.2), (0.8, 1.2), (0.1, 0.3)))
    dt = 0.05
    out = dict(zip(eq.wavefields, eq.func(list(W.values()), [rho, c, beta], dt, 1.0, None)))
    pd, d3 = eq.pd, Derivative(3, 8, stagger="backward", device=DEV)
    vx = W["vx"] - dt / rho * pd.x_forward(W["p"])
    vz = W["vz"] - dt / rho * pd.z_forward(W["p"])
    p = W["p"] - dt * rho * c**2 * (pd.x_backward(vx) + pd.z_backward(vz) - beta * (d3.dx(vx) + d3.dz(vz)))
    for name, ref in (("vx", vx), ("vz", vz), ("p", p)):
        assert _rel(out[name], ref) <= 1e-6, name


def test_hyperdiffusion_step_matches_reference():
    eq = hyper_pde("SymHyperStep", pml="none").build(2)(4, device=DEV)
    g = torch.Generator().manual_seed(3)
    t = torch.randn(2, 1, 30, 34, generator=g).to(DEV)
    kappa, beta = _random((30, 34), 0.1, 0.2, g), _random((30, 34), 0.01, 0.02, g)
    dt = 0.1
    (out,) = eq.func([t], [kappa, beta], dt, 1.0, None)
    d4 = Derivative(4, 4, device=DEV)
    lapz, lapx = eq.separable_d2_2d(t, eq.laplace_kernels, 1.0, 1.0)
    ref = t + dt * (kappa * (lapz + lapx) - beta * (d4.dz(t) + d4.dx(t)))
    assert _rel(out, ref) <= 1e-6


# ---- against exact solutions ------------------------------------------------------

def _gaussian_line(n=512, width=6.0):
    x = np.arange(n) - n / 2
    return np.exp(-(x / width) ** 2)


def _fourier(u0, factor):
    k = 2 * np.pi * np.fft.fftfreq(u0.size)
    return np.real(np.fft.ifft(np.fft.fft(u0) * factor(k)))


def _march(eq, fields, models, dt, steps):
    for d in eq._dn.values():
        d.to(dtype=torch.float64)
    with torch.no_grad():
        for _ in range(steps):
            fields = eq.func(fields, models, dt, 1.0, None)
    return fields[0][0, 0, 0].cpu().numpy()


def _plate_error(dt, so=8, t=30.0):
    """u_tt = -u_xxxx from a Gaussian at rest, on a line far from the edges,
    against the Fourier solution; float64."""
    eq = plate_pde("SymPlateLine", pml="none", axes="x").build(2)(so)
    u0, steps = _gaussian_line(), int(round(t / dt))

    def line(a):
        return torch.tensor(a, dtype=torch.float64).view(1, 1, 1, -1)

    num = _march(eq, [line(u0), line(_fourier(u0, lambda k: np.cos(-k**2 * dt)))],
                 [torch.ones(1, u0.size, dtype=torch.float64)], dt, steps)
    ana = _fourier(u0, lambda k: np.cos(k**2 * steps * dt))
    return np.linalg.norm(num - ana) / np.linalg.norm(ana)


def test_plate_matches_its_fourier_solution():
    # Leapfrog, second order in time. Measured (CPU, float64): 8.8e-5 at dt 0.2,
    # 1.9e-5 at dt 0.1. Stable for dt <= 2 h^2 / sqrt(kappa S4), S4 = 39.7 at
    # spatial order 8: dt <= 0.317.
    coarse, fine = _plate_error(0.2), _plate_error(0.1)
    assert fine <= 3e-5
    assert coarse / fine >= 3.5


def _hyper_error(dt, so=8, t=30.0):
    eq = hyper_pde("SymHyperLine", pml="none", axes="x", kappa_term=False).build(2)(so)
    u0, steps = _gaussian_line(), int(round(t / dt))
    num = _march(eq, [torch.tensor(u0, dtype=torch.float64).view(1, 1, 1, -1)],
                 [torch.ones(1, u0.size, dtype=torch.float64)], dt, steps)
    ana = _fourier(u0, lambda k: np.exp(-k**4 * steps * dt))
    return np.linalg.norm(num - ana) / np.linalg.norm(ana)


def test_hyperdiffusion_matches_its_fourier_solution():
    # Forward Euler, first order in time; stable for dt <= 2 h^4 / (beta S4),
    # 0.0504 at spatial order 8. Measured (float64): 2.15e-5 at dt 0.02,
    # 1.07e-5 at dt 0.01; at dt 0.055 it grows to 4e61 by t = 60.
    coarse, fine = _hyper_error(0.02), _hyper_error(0.01)
    assert fine <= 2e-5
    assert 1.7 <= coarse / fine <= 2.3


def test_hyperdiffusion_beyond_its_bound_blows_up():
    assert not np.isfinite(_hyper_error(0.055, t=60.0)) or _hyper_error(0.055, t=60.0) > 1e3


# ---- through the propagator -----------------------------------------------------

def _bump(shape, centre, width):
    z, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return torch.tensor(np.exp(-((z - centre[0]) / width) ** 2 - ((x - centre[1]) / width) ** 2),
                        dtype=torch.float32)


SHAPE = (48, 56)
RING = [[int(28 + 20 * np.cos(a)), int(24 + 18 * np.sin(a))] for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
PLATE = plate_pde().build(2)
DISPERSIVE = dispersive_pde().build(2)
HYPER = hyper_pde().build(2)


def _record(cls, models, src_type, rec_type, dt, memory=None, nt=200, fm=0.05, **compile_kw):
    solver = PropTorch(cls(4, device=DEV), shape=SHAPE, dh=1.0, dt=dt, nt=nt, dev=DEV, impl="eager", abcn=12,
                       memory=Full() if memory is None else memory, source_type=src_type, receiver_type=rec_type,
                       eager_options=EagerOptions(**{"use_compile": False, **compile_kw}))
    return solver(ricker(nt, dt, fm=fm, delay=1.5 / fm), np.array([[20, 18], [36, 30]]), np.array([RING, RING]),
                  models=models, **KW)


def _plate(models, **kw):
    return _record(PLATE, models, ["u"], ["u"], 0.1, **kw)


def _dispersive(models, **kw):
    return _record(DISPERSIVE, models, ["p"], ["p", "vx"], 0.2, **kw)


def _hyper(models, **kw):
    return _record(HYPER, models, ["T"], ["T"], 0.02, **{"nt": 150, "fm": 1.0, **kw})


def _plate_models(requires_grad=True):
    return [(1.0 + 0.3 * _bump(SHAPE, (24, 28), 8.0)).to(DEV).requires_grad_(requires_grad)]


def _dispersive_models(requires_grad=True):
    bump = _bump(SHAPE, (24, 28), 8.0)
    return [m.to(DEV).requires_grad_(requires_grad) for m in (1.0 + 0.1 * bump, 1.0 + 0.2 * bump, 0.5 + 0.2 * bump)]


def _hyper_models(requires_grad=True):
    bump = _bump(SHAPE, (24, 28), 8.0)
    return [m.to(DEV).requires_grad_(requires_grad) for m in (0.2 + 0.1 * bump, 0.5 + 0.2 * bump)]


def test_plate_gradient_matches_finite_differences():
    # J = 0.5 |d(kappa) - d_obs|^2 along a bump in kappa. Measured (RTX 6000 Ada):
    # 2.9e-4, 3.6e-4 and 1.6e-3 at 3x, 1x and 0.3x this step (the float32 record
    # sets the floor of the finite difference).
    bump = _bump(SHAPE, (24, 28), 8.0).to(DEV)
    with torch.no_grad():
        d_obs = _plate(_plate_models(False))
    init = torch.ones(SHAPE, device=DEV)

    def misfit(kappa):
        return 0.5 * (_plate([kappa]).double() - d_obs.double()).square().sum()

    kappa = init.clone().requires_grad_(True)
    misfit(kappa).backward()
    ad = (kappa.grad.double() * bump.double()).sum().item()
    h = 1e-2
    with torch.no_grad():
        fd = (misfit(init + h * bump) - misfit(init - h * bump)).item() / (2 * h)
    assert abs(fd - ad) <= 1e-3 * abs(ad)


def _grads(run, make_models, **kw):
    models = make_models()
    d = run(models, **kw)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


@pytest.mark.parametrize("run, make_models", [(_plate, _plate_models), (_dispersive, _dispersive_models)],
                         ids=["plate", "dispersive"])
def test_lossless_memory_modes(run, make_models):
    # Records bitwise; gradients measured (RTX 6000 Ada): checkpointing 2.5e-7 ..
    # 7.3e-7, boundary saving 4.3e-6 .. 8.6e-6 (the step run backwards in float32).
    d_ref, g_ref = _grads(run, make_models)
    assert torch.isfinite(d_ref).all() and d_ref.abs().max() > 0
    for memory in (Ckpt(), BoundarySaving(storage="gpu" if DEV == "cuda" else "cpu")):
        d, g = _grads(run, make_models, memory=memory)
        assert torch.equal(d, d_ref), memory
        for a, b in zip(g, g_ref):
            assert _rel(a, b) <= 1e-4, memory


def test_hyperdiffusion_refuses_boundary_saving():
    d_ref, g_ref = _grads(_hyper, _hyper_models)
    assert torch.isfinite(d_ref).all() and d_ref.abs().max() > 0
    d, g = _grads(_hyper, _hyper_models, memory=Ckpt())
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a, b) <= 1e-6
    with pytest.raises(NotImplementedError, match=r"T\.dx4 \+ T\.dz4"):
        _grads(_hyper, _hyper_models, memory=BoundarySaving(storage="gpu" if DEV == "cuda" else "cpu"))


@requires_compile
@pytest.mark.parametrize("run, make_models", [(_plate, _plate_models), (_dispersive, _dispersive_models),
                                              (_hyper, _hyper_models)], ids=["plate", "dispersive", "hyper"])
def test_high_order_compiles_in_one_graph(run, make_models):
    # aot_eager: Dynamo capture without a C++ toolchain. Measured (RTX 6000 Ada):
    # records 8e-8 .. 1.6e-6, gradients 1.6e-7 .. 1.8e-6.
    d_ref, g_ref = _grads(run, make_models, nt=100)
    d_c, g_c = _grads(run, make_models, nt=100, use_compile=True, compile_fullgraph=True,
                      compile_backend="aot_eager")
    assert _rel(d_c, d_ref) <= 1e-5
    for a, b in zip(g_ref, g_c):
        assert _rel(b, a) <= 1e-4
