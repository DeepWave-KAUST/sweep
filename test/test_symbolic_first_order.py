"""sweep.symbolic: first-order (staggered-grid) systems and coefficients.

* Restating Acoustic1st and the 2-D Elastic reproduces them bit for bit, record
  and gradient, uncompiled: spatial orders 4 and 8, with and without
  strip-held CPML memory, and under eager boundary saving (the generated
  interior sub-steps). The generated step takes the derivatives, the CPML
  corrections and the updates in their order, and makes the same products of
  models and dt once per rollout. One exception: the 2-D Elastic makes dt/rho
  twice, under the names of the density at vz and at vx, the generated step
  once, so its rho gradient moves by rounding.
* Uncompiled with Full(), the restated equations peak at no more CUDA memory
  than the shipped ones.
* The 3-D Elastic writes its statements in another order, which the generator
  does not follow: the records agree bit for bit, the gradients to a few 1e-7.
* The step compiles in one graph. Compiled, a gradient moves by ~1e-7: Inductor
  orders the backward by the operand order of each sum, and sympy keeps a sum
  in its canonical order rather than as written.
* A Coefficient is computed once per run (``prepare_models``), in second-order
  PDEs too.
* A field declared again under its name (a notebook cell run twice) is a new
  field: sympy's cache does not hand back the old declaration.
* What is not supported raises when the PDE is built, and a free surface when
  the propagator is built.
* Dissipative terms (``kappa * laplace(T)``, the viscous terms of the unsteady
  Stokes system): an impulse diffuses as the heat kernel, a Gaussian vortex as
  Lamb-Oseen's; the step matches one written out by hand; the gradient matches
  finite differences; strip-held CPML memory reproduces ``Full()`` bit for bit,
  checkpointing the record bit for bit and the gradient to rounding; boundary
  saving is refused.

Uncompiled CUDA runs need ``cudnn.deterministic`` (see test_symbolic_pde.py).
"""
from __future__ import annotations

import gc
import importlib.util

import numpy as np
import pytest
import sympy
import torch

from conftest import requires_compile, ricker
from sweep.equations import Acoustic1st, get_equation
from sweep.equations.elastic import Elastic
from sweep.equations._cpml import cpml_update
from sweep.propagator.options import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Coefficient, Eq, Field, Model, laplace

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DH, DT = 10.0, 1.0e-3
SHAPES = {2: (48, 64), 3: (24, 20, 28)}
NT = {2: 150, 3: 50}


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old


def acoustic1st_pde(name="SymAcoustic1st"):
    """Acoustic1st: the velocities sit half a cell BEFORE the pressure node."""
    vp, rho = Model("vp", unit="m/s"), Model("rho", unit="kg/m^3")
    p = Field("p", time_order=1, source=True, receiver=True)
    vx = Field("vx", time_order=1, stagger="-x", receiver=True)
    vz = Field("vz", time_order=1, stagger="-z", receiver=True)
    return PDE([Eq(vz.dt, -p.dz / rho),
                Eq(vx.dt, -p.dx / rho),
                Eq(p.dt, -rho * vp**2 * (vx.dx + vz.dz))], name=name)


def elastic2d_pde(name="SymElastic"):
    """Virieux's velocity-stress system, Lamé parameters computed once per run."""
    vp, vs, rho = Model("vp", unit="m/s"), Model("vs", unit="m/s"), Model("rho", unit="kg/m^3")
    lam = Coefficient("lam", rho * (vp**2 - 2 * vs**2))
    mu = Coefficient("mu", rho * vs**2)
    lam2mu = Coefficient("lam2mu", lam + 2 * mu)
    vx = Field("vx", time_order=1, stagger="x", source=True, receiver=True)
    vz = Field("vz", time_order=1, stagger="z", source=True, receiver=True)
    sxx = Field("sxx", time_order=1, source=True, receiver=True)
    szz = Field("szz", time_order=1, source=True, receiver=True)
    sxz = Field("sxz", time_order=1, stagger=("z", "x"), source=True)
    return PDE([Eq(vz.dt, (szz.dz + sxz.dx) / rho),
                Eq(vx.dt, (sxx.dx + sxz.dz) / rho),
                Eq(szz.dt, lam2mu * vz.dz + lam * vx.dx),
                Eq(sxx.dt, lam2mu * vx.dx + lam * vz.dz),
                Eq(sxz.dt, mu * (vx.dz + vz.dx))], name=name)


def elastic3d_pde():
    vp, vs, rho = Model("vp", unit="m/s"), Model("vs", unit="m/s"), Model("rho", unit="kg/m^3")
    lam = Coefficient("lam", rho * (vp**2 - 2 * vs**2))
    mu = Coefficient("mu", rho * vs**2)
    vx = Field("vx", time_order=1, stagger="x", source=True, receiver=True)
    vy = Field("vy", time_order=1, stagger="y", source=True, receiver=True)
    vz = Field("vz", time_order=1, stagger="z", source=True, receiver=True)
    sxx, syy, szz = (Field(n, time_order=1, source=True, receiver=True) for n in ("sxx", "syy", "szz"))
    sxy = Field("sxy", time_order=1, stagger=("x", "y"), source=True)
    sxz = Field("sxz", time_order=1, stagger=("x", "z"), source=True)
    syz = Field("syz", time_order=1, stagger=("y", "z"), source=True)
    div_v = vx.dx + vy.dy + vz.dz
    return PDE([Eq(vx.dt, (sxx.dx + sxy.dy + sxz.dz) / rho),
                Eq(vy.dt, (sxy.dx + syy.dy + syz.dz) / rho),
                Eq(vz.dt, (sxz.dx + syz.dy + szz.dz) / rho),
                Eq(sxx.dt, lam * div_v + 2 * mu * vx.dx),
                Eq(syy.dt, lam * div_v + 2 * mu * vy.dy),
                Eq(szz.dt, lam * div_v + 2 * mu * vz.dz),
                Eq(sxy.dt, mu * (vx.dy + vy.dx)),
                Eq(sxz.dt, mu * (vx.dz + vz.dx)),
                Eq(syz.dt, mu * (vy.dz + vz.dy))], name="SymElastic3D")


KINDS = {
    # kind: (hand-written class, generated class, ndim, source_type, receiver_type)
    "acoustic1st": (lambda so: Acoustic1st(so, device=DEV, backend="torch"),
                    acoustic1st_pde().build(2), 2, ["p"], ["p", "vx", "vz"]),
    "elastic": (lambda so: Elastic(so, device=DEV), elastic2d_pde().build(2), 2,
                ["sxx", "szz"], ["vx", "vz", "sxx"]),
    "elastic3d": (lambda so: get_equation("Elastic3D")(so, device=DEV), elastic3d_pde().build(3), 3,
                  ["sxx", "syy", "szz"], ["vx", "vy", "vz"]),
}


def _run(eq, ndim, src_type, rec_type, memory=None, strips=False, batch=1, **compile_kw):
    """``batch > 1``: that many shots, each with its own model ``(B, *shape)``."""
    shape, nt = SHAPES[ndim], NT[ndim]
    solver = PropTorch(eq, shape=shape, dh=DH, dt=DT, nt=nt, dev=DEV, impl="eager", abcn=12,
                       memory=Full() if memory is None else memory, source_type=src_type,
                       receiver_type=rec_type,
                       eager_options=EagerOptions(cpml_strips=strips, **{"use_compile": False, **compile_kw}))
    g = torch.Generator().manual_seed(0)
    mshape = shape if batch == 1 else (batch, *shape)
    vp = 2000 + 600 * torch.rand(mshape, generator=g)
    models = [vp]
    if len(eq.model_specs) == 3:
        models.append(vp / 1.9 + 50 * torch.rand(mshape, generator=g))
    models.append(1800 + 400 * torch.rand(mshape, generator=g))
    models = [m.to(DEV).requires_grad_(True) for m in models]
    if ndim == 2:
        src, rec = np.array([[20, 6]]), np.array([[[ix, 3] for ix in range(2, 62, 3)]])
    else:
        src, rec = np.array([[14, 10, 5]]), np.array([[[ix, iy, 2] for ix in range(2, 26, 4) for iy in (4, 15)]])
    if batch > 1:
        src = np.array([[20 + 10 * b, 6] for b in range(batch)])
        rec = np.repeat(rec, batch, axis=0)
    d = solver(ricker(nt, DT, fm=12.0, delay=0.08), src, rec, models=models)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


def _both(kind, order=4, **kw):
    hand, gen, ndim, src_type, rec_type = KINDS[kind]
    return (_run(hand(order), ndim, src_type, rec_type, **kw),
            _run(gen(order, device=DEV), ndim, src_type, rec_type, **kw))


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# Each pair of (kind, order), (kind, strips), (order, strips) met once.
BITWISE = [("acoustic1st", 4, False), ("acoustic1st", 8, True), ("elastic", 4, True), ("elastic", 8, False)]
# The uncompiled step makes its products of models and dt once per rollout
# (sweep.equations._invariants), the generated one under the product's text:
# Elastic makes dt/rho under two names, for vz and for vx (rho_z, rho_x, both
# rho), the generated step once for both. The rho gradient then accumulates
# through one node instead of two: 1.1e-8 .. 1.4e-8 (A100). Boundary saving
# rebuilds the gradient from the interior sub-steps, which make dt/rho in place
# in both, and stays bit for bit.
ROUNDED_GRADIENTS = {"elastic": {2}}


@pytest.mark.parametrize("kind, order, strips", BITWISE,
                         ids=[f"{k}-so{o}-strips{s}" for k, o, s in BITWISE])
def test_restated_first_order_is_bitwise(kind, order, strips):
    (d_ref, g_ref), (d_gen, g_gen) = _both(kind, order, strips=strips)
    assert torch.equal(d_ref, d_gen)
    for k, (a, b) in enumerate(zip(g_ref, g_gen)):
        if k in ROUNDED_GRADIENTS.get(kind, ()):
            assert _rel(b, a) <= 1e-7
        else:
            assert torch.equal(a, b)


def test_latex_lists_coefficients():
    pde = elastic2d_pde()
    tex = pde.latex()
    assert "\\frac{\\partial \\sigma_{xz}}{\\partial t} &= \\mu \\left(" in tex
    assert "\\frac{1}{\\rho} \\left(" in tex                     # not a fraction of fractions
    assert "\\mu &= \\rho v_{s}^{2}" in tex and "\\lambda &= \\rho \\left(v_{p}^{2} - 2 v_{s}^{2}\\right)" in tex
    assert tex.count(" \\\\\n") == 4 + len(pde.lower(2).coefficients)
    assert pde.latex(coefficients=False).count(" \\\\\n") == 4


def _run_shots(eq, src_type, rec_type, models, src, rec):
    """A 2-D run on the given models and geometry: record and model gradients."""
    solver = PropTorch(eq, shape=SHAPES[2], dh=DH, dt=DT, nt=NT[2], dev=DEV, impl="eager", abcn=12,
                       memory=Full(), source_type=src_type, receiver_type=rec_type,
                       eager_options=EagerOptions(use_compile=False))
    models = [m.detach().clone().requires_grad_(True) for m in models]
    d = solver(ricker(NT[2], DT, fm=12.0, delay=0.08), src, rec, models=models)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


@pytest.mark.parametrize("kind", ["acoustic1st", "elastic"])
def test_per_shot_batched_models(kind):
    """Two shots with their own models, (B, nz, nx): shot b is the single-shot
    run on model b, and, where the hand-written class takes batched models too
    (Elastic), the hand-written batched run."""
    hand, gen, _, src_type, rec_type = KINDS[kind]
    assert gen.supports_batched_models and not KINDS["elastic3d"][1].supports_batched_models
    g = torch.Generator().manual_seed(0)
    shape, batch = SHAPES[2], 2
    vp = 2000 + 600 * torch.rand((batch, *shape), generator=g)
    models = [vp] + ([vp / 1.9 + 50 * torch.rand((batch, *shape), generator=g)] if kind == "elastic" else [])
    models.append(1800 + 400 * torch.rand((batch, *shape), generator=g))
    models = [m.to(DEV) for m in models]
    src = np.array([[20, 6], [30, 6]])
    rec = np.repeat(np.array([[[ix, 3] for ix in range(2, 62, 3)]]), batch, axis=0)
    d, grads = _run_shots(gen(4, device=DEV), src_type, rec_type, models, src, rec)
    assert d.shape[0] == batch and all(gr.shape == (batch, *shape) for gr in grads)
    for b in range(batch):
        d_b, grads_b = _run_shots(gen(4, device=DEV), src_type, rec_type, [m[b] for m in models], src[b:b + 1], rec[b:b + 1])
        assert torch.equal(d[b], d_b[0])
        # The record is bit for bit; the model gradient of the batched run is
        # accumulated by the backward of a batch of two, in another order than
        # the single shot's: 2e-9 .. 5e-7 relative (CPU).
        for gr, gr_b in zip(grads, grads_b):
            assert _rel(gr[b], gr_b) <= 2e-6
    if getattr(hand(4), "supports_batched_models", False):
        d_ref, grads_ref = _run_shots(hand(4), src_type, rec_type, models, src, rec)
        assert torch.equal(d_ref, d)
        for k, (a, b) in enumerate(zip(grads_ref, grads)):
            if k in ROUNDED_GRADIENTS.get(kind, ()):
                assert _rel(b, a) <= 1e-7
            else:
                assert torch.equal(a, b)


@pytest.mark.parametrize("kind", ["acoustic1st", "elastic"])
def test_restated_first_order_boundary_saving_is_bitwise(kind):
    (d_ref, g_ref), (d_gen, g_gen) = _both(kind, memory=BoundarySaving(storage="gpu"))
    assert torch.equal(d_ref, d_gen)
    for a, b in zip(g_ref, g_gen):
        assert torch.equal(a, b)


def _peak_bytes(make_eq, kind):
    """Peak CUDA memory of an uncompiled Full() forward + backward above what was
    allocated before it."""
    _, _, ndim, src_type, rec_type = KINDS[kind]
    _run(make_eq(), ndim, src_type, rec_type)       # cuDNN / cuBLAS handles and workspaces
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _run(make_eq(), ndim, src_type, rec_type)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


@pytest.mark.skipif(DEV != "cuda", reason="the peak is read from the CUDA caching allocator")
@pytest.mark.parametrize("kind", ["acoustic1st", "elastic", "elastic3d"])
def test_restated_first_order_peak_memory(kind):
    # The generated step frees a sub-step's derivatives at its end, as the
    # shipped sub-step functions free theirs on returning, and computes the 3-D
    # divergence once for the three normal stresses (written out per stress,
    # autograd keeps two more fields a step). Half a field of slack for the
    # caching allocator's block reuse.
    hand, gen, ndim, _, _ = KINDS[kind]
    half_field = 2 * int(np.prod([n + 24 for n in SHAPES[ndim]]))  # fp32, abcn=12 a side
    assert _peak_bytes(lambda: gen(4, device=DEV), kind) <= _peak_bytes(lambda: hand(4), kind) + half_field


def test_restated_elastic3d():
    # Measured (A100): gradients 5.7e-8 .. 3.7e-7 relative with Full, 7.6e-8 ..
    # 1.9e-7 with boundary saving.
    (d_ref, g_ref), (d_gen, g_gen) = _both("elastic3d")
    assert torch.equal(d_ref, d_gen)
    for a, b in zip(g_ref, g_gen):
        assert _rel(b, a) <= 1e-6


@requires_compile
def test_first_order_compiles_in_one_graph():
    # aot_eager: Dynamo capture (the fullgraph check) without a C++ toolchain.
    _, gen, ndim, src_type, rec_type = KINDS["elastic"]
    d_ref, g_ref = _run(gen(4, device=DEV), ndim, src_type, rec_type)
    d_c, g_c = _run(gen(4, device=DEV), ndim, src_type, rec_type, use_compile=True,
                    compile_fullgraph=True, compile_backend="aot_eager")
    assert _rel(d_c, d_ref) <= 1e-6
    for a, b in zip(g_ref, g_c):
        assert _rel(b, a) <= 1e-5


def test_coefficients_are_computed_once():
    vp, eps = Model("vp", unit="m/s"), Model("eps")
    p = Field("p", source=True, receiver=True)
    vh = Coefficient("vh2", vp**2 * (1 + 2 * eps))
    inline = PDE([Eq(p.dt2, vp**2 * (1 + 2 * eps) * p.dx2 + vp**2 * p.dz2)], name="SymInline").build(2)
    hoisted = PDE([Eq(p.dt2, vh * p.dx2 + vp**2 * p.dz2)], name="SymHoisted").build(2)
    vp_t, eps_t = torch.full((4, 4), 2000.0), torch.full((4, 4), 0.1)
    prepared = hoisted(4).prepare_models([vp_t, eps_t])
    assert len(prepared) == 3 and torch.equal(prepared[2], vp_t**2 * (1 + 2 * eps_t))
    assert all(a is b for a, b in zip(hoisted(4).prepare_models(prepared), prepared))   # passed through

    def run(cls):
        solver = PropTorch(cls(4, device=DEV), shape=(40, 48), dh=DH, dt=DT, nt=100, dev=DEV,
                           impl="eager", abcn=12, eager_options=EagerOptions(use_compile=False))
        models = [torch.full((40, 48), 2000.0, device=DEV).requires_grad_(True),
                  torch.full((40, 48), 0.1, device=DEV).requires_grad_(True)]
        d = solver(ricker(100, DT), np.array([[24, 20]]), np.array([[[ix, 4] for ix in range(4, 44, 4)]]),
                   models=models)
        d.square().sum().backward()
        return d.detach(), [m.grad for m in models]

    (d_i, g_i), (d_h, g_h) = run(inline), run(hoisted)
    assert torch.equal(d_i, d_h)              # the same numbers at every step
    for a, b in zip(g_i, g_h):                # summed once instead of per step
        assert _rel(b, a) <= 1e-5


def test_redeclared_field_is_a_new_field():
    # Acoustic1st declares vz half a cell before the node, Elastic after it:
    # without its own identity the second vz.dz would be the first's.
    acoustic1st_pde("SymRedeclA")
    cls = elastic2d_pde("SymRedeclE").build(2)
    assert "m_vz_z" in cls(4).wavefields
    p = Field("p", time_order=1, source=True)
    vx = Field("vx", time_order=1, stagger="x", receiver=True)
    first = PDE([Eq(vx.dt, p.dx), Eq(p.dt, vx.dx)], name="SymRedecl1").build(2)
    p = Field("p", time_order=1, source=True, receiver=True)   # now also a receiver
    vx = Field("vx", time_order=1, stagger="x")
    second = PDE([Eq(vx.dt, p.dx), Eq(p.dt, vx.dx)], name="SymRedecl2").build(2)
    assert first.DEFAULT_RECEIVER_FIELDS == ("vx",) and second.DEFAULT_RECEIVER_FIELDS == ("p",)


def test_first_order_export(tmp_path):
    path = tmp_path / "sym_acoustic1st.py"
    acoustic1st_pde().export(path, ndim=2)
    spec = importlib.util.spec_from_file_location("sym_acoustic1st", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, gen, ndim, src_type, rec_type = KINDS["acoustic1st"]
    g_exp = _run(module.SymAcoustic1st(4, device=DEV), ndim, src_type, rec_type)[1]
    g_gen = _run(gen(4, device=DEV), ndim, src_type, rec_type)[1]
    for a, b in zip(g_exp, g_gen):
        assert torch.equal(a, b)


def test_first_order_free_surface_is_refused():
    with pytest.raises(NotImplementedError, match="takes no free surface"):
        PDE(acoustic1st_pde().equations, name="SymFS", free_surface="zero")
    with pytest.raises(NotImplementedError, match="takes no free surface"):
        PropTorch(KINDS["acoustic1st"][1](4, device=DEV), shape=(20, 24), dh=DH, dt=DT, nt=5, dev=DEV,
                  impl="eager", abcn=8, free_surface=True)


vp, rho = Model("vp"), Model("rho")
p1 = Field("p", time_order=1, source=True, receiver=True)
ux = Field("ux", time_order=1, stagger="x")
ux_before = Field("ux_before", time_order=1, stagger="-x")
uz = Field("uz", time_order=1, stagger="z")
wx = Field("wx", time_order=1, stagger="x")
flat = Field("flat", time_order=1)
p2 = Field("q", source=True, receiver=True)
GOOD_P = Eq(p1.dt, -rho * vp**2 * (ux.dx + uz.dz))


@pytest.mark.parametrize("equations, error, match", [
    ([Eq(flat.dt, -p1.dx / rho), Eq(p1.dt, flat.dx)], NotImplementedError, "collocated"),
    ([Eq(ux.dt, -ux_before.dx / rho), Eq(ux_before.dt, ux.dx), Eq(p1.dt, ux.dx)], ValueError, "whole cell"),
    ([Eq(uz.dt, -p1.dx / rho), Eq(ux.dt, -p1.dx / rho), GOOD_P], ValueError, "different z positions"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(uz.dt, p1.dz2), GOOD_P], NotImplementedError, "its own field only"),
    ([Eq(ux.dt, -p1.dx / rho + ux * laplace(ux)), Eq(uz.dt, -p1.dz / rho), GOOD_P], ValueError, "not linear"),
    ([Eq(ux.dt, -p1.dx / rho + p1.dx * ux.dx2), Eq(uz.dt, -p1.dz / rho), GOOD_P], ValueError, "not linear"),
    ([Eq(p1.dt, vp * laplace(p1) + vp)], ValueError, "contains no field"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(uz.dt, -p1.dz / rho), GOOD_P, Eq(p2.dt2, vp**2 * laplace(p2))],
     NotImplementedError, "mixes first-order"),
    ([Eq(ux.dt, ux.dx), Eq(p1.dt, ux.dx)], ValueError, "derivative of itself"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(p1.dt, ux.dx), Eq(ux_before.dt, -p1.dx / rho)], NotImplementedError,
     "two positions"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(p1.dt, -rho * vp**2 * (ux.dx + wx.dx)), Eq(wx.dt, -p1.dx / rho)],
     NotImplementedError, "two sub-steps"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(uz.dt, -p1.dz / rho), Eq(p1.dt, vp * (ux.dx + uz.dz) + vp)], ValueError,
     "contains no field"),
], ids=["collocated", "whole-cell", "other-axis", "second-derivative", "nonlinear-dissipation",
        "dissipation-times-derivative", "dissipation-source-term", "mixed-order",
        "itself", "two-positions", "two-substeps", "source-term"])
def test_unsupported_first_order_is_refused(equations, error, match):
    with pytest.raises(error, match=match):
        PDE(equations, name="SymBad1st")


def test_declarations_are_checked():
    with pytest.raises(ValueError, match="first-order fields"):
        Field("u", stagger="x")
    with pytest.raises(ValueError, match="stagger entry"):
        Field("u", time_order=1, stagger="w")
    with pytest.raises(ValueError, match="twice"):
        Field("u", time_order=1, stagger=("x", "-x"))
    with pytest.raises(ValueError, match="neither a Model nor a Coefficient"):
        Coefficient("c", vp * p1)
    with pytest.raises(ValueError, match="uses no model"):
        Coefficient("c", sympy.Integer(2))
    with pytest.raises(ValueError, match="reserved"):
        Coefficient("pd", vp)
    with pytest.raises(ValueError, match="staggered along y in a 2-D build"):
        uy = Field("uy", time_order=1, stagger="y")
        PDE([Eq(uy.dt, -p1.dy / rho), Eq(p1.dt, uy.dy)], name="SymY1st").build(2)
    with pytest.raises(ValueError, match="y-derivative in a 2-D build"):
        PDE([Eq(ux.dt, -p1.dx / rho + vp * ux.dy2), Eq(p1.dt, ux.dx)], name="SymY2nd").build(2)


# ---- dissipative terms -------------------------------------------------------------

def heat_pde(name="SymHeat"):
    kappa = Model("kappa", unit="m^2/s")
    t = Field("T", time_order=1, source=True, receiver=True)
    return PDE([Eq(t.dt, kappa * laplace(t))], name=name)


def stokes_pde(name="SymStokes", pml="cpml", per_axis=False):
    """Unsteady Stokes flow, Chorin's artificial compressibility."""
    rho, nu, c = Model("rho"), Model("nu"), Model("c")
    vx = Field("vx", time_order=1, stagger="x", source=True, receiver=True)
    vz = Field("vz", time_order=1, stagger="z", receiver=True)
    p = Field("p", time_order=1, receiver=True)
    visc = (lambda f: f.dx2 + f.dz2) if per_axis else laplace
    return PDE([Eq(vx.dt, -p.dx / rho + nu * visc(vx)),
                Eq(vz.dt, -p.dz / rho + nu * visc(vz)),
                Eq(p.dt, -rho * c**2 * (vx.dx + vz.dz))], name=name, pml=pml)


STOKES = stokes_pde().build(2)
# The PML profiles from max_vel and pml_freq rather than from the models: a
# finite difference of a model then sees the model only.
STOKES_KW = dict(max_vel=1.0, pml_freq=0.1)


def test_dissipative_class():
    assert STOKES.DISSIPATIVE_TERMS == ("nu*laplace(vx)", "nu*laplace(vz)")
    assert not hasattr(STOKES, "interior_substeps")                 # no lossless sub-steps
    assert callable(KINDS["acoustic1st"][1].interior_substeps)     # a lossless system keeps them
    assert KINDS["acoustic1st"][1](4).supports_boundary_saving_eager
    # Per axis or as a Laplacian, one step.
    def step(src):
        return src[src.index("    def func"):]
    assert step(stokes_pde("SymStokesSrc", per_axis=True).source(2)) == step(stokes_pde("SymStokesSrc").source(2))
    heat = heat_pde("SymHeatSrc").source(2)
    assert "(T,) = wavefields" in heat and "pd = self.pd" not in heat


def _heat_kernel_run(ndim, n, nt, r=0.05, so=8):
    """An impulse at the centre, kappa = h = 1, dt = r: the snapshot after step
    nt - 1 (nt - 1 steps of diffusion since the injection) and the heat kernel
    exp(-|x|^2 / (4 t)) / (4 pi t)^(ndim / 2) at t = (nt - 1) dt."""
    shape, c = (n,) * ndim, n // 2
    solver = PropTorch(heat_pde(f"SymHeat{ndim}D").build(ndim)(so, device=DEV), shape=shape, dh=1.0, dt=r, nt=nt,
                       dev=DEV, impl="eager", abcn=8, memory=Full(), eager_options=EagerOptions(use_compile=False))
    w = np.zeros(nt, np.float32)
    w[0] = 1.0
    with torch.no_grad():
        _, snap = solver(torch.from_numpy(w), np.array([[c] * ndim]), np.array([[[c + 3] + [c] * (ndim - 1)]]),
                         models=[torch.ones(shape, device=DEV)], return_wavefield=True, snapshot_times=[nt - 1],
                         max_vel=1.0, pml_freq=1.0)
    pad = (snap.shape[-1] - n) // 2
    num = snap[0, 0, 0, 0][(slice(pad, pad + n),) * ndim].double().numpy()
    t = (nt - 1) * r
    r2 = sum(a**2 for a in np.meshgrid(*[np.arange(n) - c] * ndim, indexing="ij"))
    return num, np.exp(-r2 / (4 * t)) / (4 * np.pi * t) ** (ndim / 2)


@pytest.mark.parametrize("ndim, n, nt, tol", [(2, 81, 401, 2.5e-3), (3, 41, 201, 8e-3)], ids=["2d", "3d"])
def test_heat_kernel(ndim, n, nt, tol):
    # Forward Euler: the error is O(dt / t). Measured (A100 and CPU): 2-D 1.5e-3,
    # 3-D 4.8e-3; the heat to 2.5e-6 and 1.2e-5.
    num, ana = _heat_kernel_run(ndim, n, nt)
    assert np.linalg.norm(num - ana) / np.linalg.norm(ana) <= tol
    assert abs(num.sum() - 1) <= 1e-4                  # the heat stays


def _lamb_oseen_run(steps=180, dt=0.05):
    """With rho constant the pressure drops out of the vorticity, which then
    diffuses as heat: a Gaussian vortex (circulation 2 pi) stays one, the
    Lamb-Oseen vortex. The propagator starts from rest, so the step is driven
    directly from it; the edges, where the padding cuts the far field, are 64
    cells from the vortex. Returns the vorticity's relative L2 error within 3
    core radii, its circulation within r = 40 over the analytic one, minus 1,
    and max |p| within r = 30 over rho v_max^2."""
    n, nu, t0 = 128, 1.0, 9.0                             # a = 6 cells -> 8.5 at t = 9
    eq = stokes_pde("SymStokesVortex", pml="none").build(2)(8, device=DEV)
    eq.pd.set_spacing(1.0)
    zc = (n - 1) / 2                                      # a cell corner
    z, x = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64), indexing="ij")

    def velocity(zz, xx):                                 # (vx, vz) = (-v_theta dz / r, v_theta dx / r)
        r2 = (zz - zc) ** 2 + (xx - zc) ** 2              # > 0 on the staggered points
        f = -np.expm1(-r2 / (4 * nu * t0)) / r2
        return -f * (zz - zc), f * (xx - zc)

    init = {"vx": velocity(z, x + 0.5)[0], "vz": velocity(z + 0.5, x)[1], "p": np.zeros((n, n))}
    fields = [torch.tensor(init[name], dtype=torch.float32, device=DEV)[None, None] for name in eq.wavefields]
    vals = {"rho": 1.0, "nu": nu, "c": 2.0}
    models = [torch.full((n, n), vals[m], device=DEV) for m in eq.models]
    with torch.no_grad():
        for _ in range(steps):
            fields = eq.func(fields, models, dt, 1.0, None)
    f = dict(zip(eq.wavefields, fields))
    w = (eq.pd.z_forward(f["vx"]) - eq.pd.x_forward(f["vz"]))[0, 0].double().cpu().numpy()
    a2 = 4 * nu * (t0 + steps * dt)
    r2 = (z + 0.5 - zc) ** 2 + (x + 0.5 - zc) ** 2
    ana = -2 / a2 * np.exp(-r2 / a2)
    core = r2 < 9 * a2
    # p, away from the sound the edges send in (c t = 18 cells)
    p = f["p"][0, 0].double().cpu().numpy()[(z - zc) ** 2 + (x - zc) ** 2 < 30**2]
    return (np.linalg.norm((w - ana)[core]) / np.linalg.norm(ana[core]),
            w[r2 < 40**2].sum() / (-2 * np.pi) - 1,
            np.abs(p).max() / (0.638 / np.sqrt(4 * nu * t0)) ** 2)


def test_stokes_vortex_diffuses_as_heat():
    # Measured (A100 and CPU): 8.5e-4, half that at dt / 2 (forward Euler);
    # circulation 3e-7; p 1.5e-6.
    err, circulation, p = _lamb_oseen_run()
    assert err <= 2e-3
    assert abs(circulation) <= 1e-4
    assert p <= 1e-4


def test_dissipative_step_matches_reference():
    eq = STOKES(4, device=DEV)
    shape, dh, dt = (20, 24), 1.0, 0.1
    PropTorch(eq, shape=shape, dh=dh, dt=dt, nt=3, dev=DEV, impl="eager", abcn=8, source_type=["vx"],
              receiver_type=["p"], eager_options=EagerOptions(use_compile=False))(
        ricker(3, dt), np.array([[10, 10]]), np.array([[[5, 5]]]),
        models=[torch.ones(shape, device=DEV)] * 3, **STOKES_KW)            # sets the profiles and spacing
    nz, nx = eq.b[0].shape[-2], eq.b[4].shape[-1]
    g = torch.Generator().manual_seed(5)
    fields = [torch.randn(2, 1, nz, nx, generator=g).to(DEV) for _ in eq.wavefields]
    W = dict(zip(eq.wavefields, fields))
    # nu dt / h^2 ~ 0.05 and c dt / h ~ 0.1: every term moves the fields.
    rho_m, nu_m, c_m = [(lo + (hi - lo) * torch.rand(nz, nx, generator=g)).to(DEV)
                        for lo, hi in ((0.8, 1.2), (0.3, 0.7), (0.8, 1.2))]
    out = dict(zip(eq.wavefields, eq.func(fields, [rho_m, nu_m, c_m], dt, dh, None)))

    pd, (az, bz, azh, bzh, ax, bx, axh, bxh) = eq.pd, eq.b
    ref = {}

    def lap(u):
        d2z, d2x = eq.separable_d2_2d(u, eq.laplace_kernels, dh, dh)
        return d2z + d2x

    px, ref["m_p_x"] = cpml_update(pd.x_forward(W["p"]), W["m_p_x"], axh, bxh)
    pz, ref["m_p_z"] = cpml_update(pd.z_forward(W["p"]), W["m_p_z"], azh, bzh)
    ref["vx"] = W["vx"] + dt * (-px / rho_m + nu_m * lap(W["vx"]))
    ref["vz"] = W["vz"] + dt * (-pz / rho_m + nu_m * lap(W["vz"]))
    vxx, ref["m_vx_x"] = cpml_update(pd.x_backward(ref["vx"]), W["m_vx_x"], ax, bx)
    vzz, ref["m_vz_z"] = cpml_update(pd.z_backward(ref["vz"]), W["m_vz_z"], az, bz)
    ref["p"] = W["p"] - dt * rho_m * c_m**2 * (vxx + vzz)
    assert sorted(ref) == sorted(eq.wavefields)
    for name in eq.wavefields:
        assert (out[name] - ref[name]).abs().max() <= 1e-5 * ref[name].abs().max(), name


def _bump(shape, centre, width):
    z, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    return torch.tensor(np.exp(-((z - centre[0]) / width) ** 2 - ((x - centre[1]) / width) ** 2), dtype=torch.float32)


def _stokes_record(models, memory=None, strips=False, nt=240, **compile_kw):
    """Stokes on the canonical 48 x 56 grid: nu dt / h^2 = 0.02, c dt / h = 0.2,
    a force on vx, p and vx recorded on a ring."""
    shape = (48, 56)
    solver = PropTorch(STOKES(4, device=DEV), shape=shape, dh=1.0, dt=0.2, nt=nt, dev=DEV, impl="eager", abcn=12,
                       memory=Full() if memory is None else memory, source_type=["vx"], receiver_type=["p", "vx"],
                       eager_options=EagerOptions(cpml_strips=strips, **{"use_compile": False, **compile_kw}))
    ring = [[int(28 + 20 * np.cos(a)), int(24 + 18 * np.sin(a))] for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
    return solver(ricker(nt, 0.2, fm=0.1, delay=15.0), np.array([[10, 10], [46, 38]]), np.array([ring, ring]),
                  models=models, **STOKES_KW)


def _stokes_models(requires_grad=True):
    bump = _bump((48, 56), (24, 28), 8.0)
    return [m.to(DEV).requires_grad_(requires_grad)
            for m in (1.0 + 0.1 * bump, 0.1 * (1 + 0.5 * bump), 1.0 + 0.1 * bump)]


def test_dissipative_gradient_matches_finite_differences():
    # J = 0.5 |d(nu) - d_obs|^2 along a bump in nu, the viscous term's own
    # coefficient. Measured (A100 and CPU): 5.8e-6 at this step; 1.3e-5 and
    # 2.1e-4 at 3x and 1/3 of it.
    bump = _bump((48, 56), (24, 28), 8.0).to(DEV)
    with torch.no_grad():
        d_obs = _stokes_record(_stokes_models(False))
    init = [torch.ones(48, 56, device=DEV), torch.full((48, 56), 0.1, device=DEV), torch.ones(48, 56, device=DEV)]

    def misfit(nu):
        return 0.5 * (_stokes_record([init[0], nu, init[2]]).double() - d_obs.double()).square().sum()

    nu = init[1].clone().requires_grad_(True)
    misfit(nu).backward()
    ad = (nu.grad.double() * bump.double()).sum().item()
    h = 1e-2 * 0.1
    with torch.no_grad():
        fd = (misfit(init[1] + h * bump) - misfit(init[1] - h * bump)).item() / (2 * h)
    assert abs(fd - ad) <= 1e-4 * abs(ad)


def test_dissipative_memory_modes():
    def run(**kw):
        models = _stokes_models()
        d = _stokes_record(models, nt=120, **kw)          # the wavelet peaks at step 75
        d.square().sum().backward()
        return d.detach(), [m.grad for m in models]

    d_ref, g_ref = run()
    d, g = run(strips=True)
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert torch.equal(a, b)
    # Each checkpointed segment makes the step's products of models and dt
    # anew (sweep.equations._invariants), as for the shipped equations: the
    # gradient accumulates through one node per segment instead of one per
    # rollout. Measured: up to 2.5e-7 (A100).
    d, g = run(memory=Ckpt())
    assert torch.equal(d, d_ref)
    for a, b in zip(g, g_ref):
        assert _rel(a, b) <= 1e-6
    with pytest.raises(NotImplementedError, match=r"dissipative or damping terms nu\*laplace\(vx\), nu\*laplace\(vz\) do not allow"):
        run(memory=BoundarySaving(storage="gpu"))


@requires_compile
def test_dissipative_compiles_in_one_graph():
    # aot_eager: Dynamo capture (the fullgraph check) without a C++ toolchain.
    # Measured (A100 and CPU): record 1.4e-7, gradients 2e-7 .. 3.9e-7 (inductor
    # on the A100: 1.5e-7, 2.9e-7 .. 4.1e-7).
    def run(**kw):
        models = _stokes_models()
        d = _stokes_record(models, nt=120, **kw)
        d.square().sum().backward()
        return d.detach(), [m.grad for m in models]

    d_ref, g_ref = run()
    d_c, g_c = run(use_compile=True, compile_fullgraph=True, compile_backend="aot_eager")
    assert _rel(d_c, d_ref) <= 1e-6
    for a, b in zip(g_ref, g_c):
        assert _rel(b, a) <= 1e-5
