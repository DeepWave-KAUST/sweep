"""sweep.symbolic: first-order (staggered-grid) systems and coefficients.

* Restating Acoustic1st and the 2-D Elastic reproduces them bit for bit, record
  and gradient, uncompiled: spatial orders 4 and 8, with and without
  strip-held CPML memory, and under eager boundary saving (the generated
  interior sub-steps). The generated step takes the derivatives, the CPML
  corrections and the updates in their order.
* The restated equations peak at no more CUDA memory than the shipped ones.
* The 3-D Elastic writes its statements in another order, which the generator
  does not follow: the records agree bit for bit, the gradients to ~1e-7.
* The step compiles in one graph. Compiled, a gradient moves by ~1e-7: Inductor
  orders the backward by the operand order of each sum, and sympy keeps a sum
  in its canonical order rather than as written.
* A Coefficient is computed once per run (``prepare_models``), in second-order
  PDEs too.
* A field declared again under its name (a notebook cell run twice) is a new
  field: sympy's cache does not hand back the old declaration.
* What is not supported raises when the PDE is built, and a free surface when
  the propagator is built.

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
from sweep.propagator.options import BoundarySaving, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Coefficient, Eq, Field, Model, laplace

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DH, DT = 10.0, 1.0e-3
SHAPES = {2: (48, 64), 3: (24, 20, 28)}
NT = {2: 150, 3: 80}


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


def _run(eq, ndim, src_type, rec_type, memory=None, strips=False, **compile_kw):
    shape, nt = SHAPES[ndim], NT[ndim]
    solver = PropTorch(eq, shape=shape, dh=DH, dt=DT, nt=nt, dev=DEV, impl="eager", abcn=12,
                       memory=Full() if memory is None else memory, source_type=src_type,
                       receiver_type=rec_type,
                       eager_options=EagerOptions(cpml_strips=strips, **{"use_compile": False, **compile_kw}))
    g = torch.Generator().manual_seed(0)
    vp = 2000 + 600 * torch.rand(shape, generator=g)
    models = [vp]
    if len(eq.model_specs) == 3:
        models.append(vp / 1.9 + 50 * torch.rand(shape, generator=g))
    models.append(1800 + 400 * torch.rand(shape, generator=g))
    models = [m.to(DEV).requires_grad_(True) for m in models]
    if ndim == 2:
        src, rec = np.array([[20, 6]]), np.array([[[ix, 3] for ix in range(2, 62, 3)]])
    else:
        src, rec = np.array([[14, 10, 5]]), np.array([[[ix, iy, 2] for ix in range(2, 26, 4) for iy in (4, 15)]])
    d = solver(ricker(nt, DT, fm=12.0, delay=0.08), src, rec, models=models)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


def _both(kind, order=4, **kw):
    hand, gen, ndim, src_type, rec_type = KINDS[kind]
    return (_run(hand(order), ndim, src_type, rec_type, **kw),
            _run(gen(order, device=DEV), ndim, src_type, rec_type, **kw))


def _rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


BITWISE = [(kind, order, strips) for kind in ("acoustic1st", "elastic") for order in (4, 8)
           for strips in (False, True)]


@pytest.mark.parametrize("kind, order, strips", BITWISE,
                         ids=[f"{k}-so{o}-strips{s}" for k, o, s in BITWISE])
def test_restated_first_order_is_bitwise(kind, order, strips):
    (d_ref, g_ref), (d_gen, g_gen) = _both(kind, order, strips=strips)
    assert torch.equal(d_ref, d_gen)
    for a, b in zip(g_ref, g_gen):
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
    # Measured: gradients 5e-8 .. 8.4e-7 relative (A100, Ada; Full and boundary saving).
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
    ([Eq(ux.dt, -p1.dx / rho + 0.1 * ux), Eq(uz.dt, -p1.dz / rho), GOOD_P], NotImplementedError,
     "without a derivative"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(uz.dt, p1.dz2), GOOD_P], NotImplementedError, "first derivatives only"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(uz.dt, -p1.dz / rho), GOOD_P, Eq(p2.dt2, vp**2 * laplace(p2))],
     NotImplementedError, "mixes first-order"),
    ([Eq(ux.dt, ux.dx), Eq(p1.dt, ux.dx)], ValueError, "derivative of itself"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(p1.dt, ux.dx), Eq(ux_before.dt, -p1.dx / rho)], NotImplementedError,
     "two positions"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(p1.dt, -rho * vp**2 * (ux.dx + wx.dx)), Eq(wx.dt, -p1.dx / rho)],
     NotImplementedError, "two sub-steps"),
    ([Eq(ux.dt, -p1.dx / rho), Eq(uz.dt, -p1.dz / rho), Eq(p1.dt, vp * (ux.dx + uz.dz) + vp)], ValueError,
     "contains no field"),
], ids=["collocated", "whole-cell", "other-axis", "bare-field", "second-derivative", "mixed-order",
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
