"""sweep.symbolic: wave equations written as PDEs, lowered to eager equation classes.

* A PDE that restates a shipped equation reproduces it bit for bit, record and
  gradient: ``p.dt2 = vp**2 laplace(p)`` against Acoustic (2-D) and Acoustic3D,
  without and with a free surface, per edge, on a topography, with strip-held
  CPML memory. The generated step issues the shipped equations' operations in
  their order; another order of the same operations already moves an
  uncompiled gradient by ~1e-7 (autograd accumulates in graph order).
* The restated Acoustic peaks at no more CUDA memory than the shipped one.
* A generated coupled two-field step computes what its PDE says, against a
  reference assembled here from the same operators.
* Boundary saving matches Full() on that system, and its compiled step captures
  in one graph (``compile_fullgraph=True`` raises on a graph break).
* Everything outside the supported form raises when the PDE is built, and a
  free surface the PDE does not declare raises when the propagator is built.

Uncompiled CUDA runs need ``cudnn.deterministic``: cuDNN otherwise picks
non-deterministic backward algorithms and Acoustic alone differs from itself by
~3e-7 run to run.
"""
from __future__ import annotations

import gc
import importlib.util

import numpy as np
import pytest
import sympy
import torch

from conftest import requires_compile, ricker
from sweep.equations import Acoustic, Acoustic3D
from sweep.equations._cpml import cpml_axis_update
from sweep.propagator.options import BoundarySaving, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.symbolic import PDE, Eq, Field, Model, laplace

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DH, DT = 10.0, 1.0e-3

vp = Model("vp", unit="m/s")
eps, delta = Model("eps"), Model("delta")
p = Field("p", source=True, receiver=True)
q = Field("q", receiver=True)

ACOUSTIC = PDE([Eq(p.dt2, vp**2 * laplace(p))], name="SymAcoustic", free_surface="zero")
# Coupled, per-axis coefficients, a cross-field derivative, a bare-field term
# and a function in a coefficient. Checked step by step only: the bare q in the
# p equation has no counterpart in the q equation and lets a mode grow.
TWO_FIELD = PDE([
    Eq(p.dt2, vp**2 * (1 + 2 * eps) * p.dx2 + vp**2 * sympy.sqrt(1 + 2 * delta) * q.dz2 - 0.5 * vp * q),
    Eq(q.dt2, vp**2 * sympy.sqrt(1 + 2 * delta) * p.dx2 + vp**2 * laplace(q)),
], name="SymTwoField", free_surface="zero")
# A stable coupled system (pseudo-acoustic VTI, eps >= delta) for the runs.
VTI = PDE([
    Eq(p.dt2, vp**2 * (1 + 2 * eps) * p.dx2 + vp**2 * sympy.sqrt(1 + 2 * delta) * q.dz2),
    Eq(q.dt2, vp**2 * sympy.sqrt(1 + 2 * delta) * p.dx2 + vp**2 * q.dz2),
], name="SymVTI", free_surface="zero")


@pytest.fixture(autouse=True)
def _deterministic_cudnn():
    old = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    yield
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old


SHAPES = {2: (48, 64), 3: (24, 20, 28)}


def _run(cls, ndim, order, fs, strips, topography=None):
    shape = SHAPES[ndim]
    nt = 150 if ndim == 2 else 80
    kw = {} if topography is None else {"topography": topography}
    solver = PropTorch(cls(order, device=DEV), shape=shape, dh=DH, dt=DT, nt=nt, dev=DEV, impl="eager",
                       abcn=12, free_surface=fs, memory=Full(),
                       eager_options=EagerOptions(use_compile=False, cpml_strips=strips), **kw)
    g = torch.Generator().manual_seed(0)
    model = (2000 + 300 * torch.rand(shape, generator=g)).to(DEV).requires_grad_(True)
    if ndim == 2:
        src, rec = np.array([[20, 6]]), np.array([[[ix, 3] for ix in range(2, 62, 3)]])
    else:
        src, rec = np.array([[14, 10, 5]]), np.array([[[ix, iy, 2] for ix in range(2, 26, 4) for iy in (4, 15)]])
    d = solver(ricker(nt, DT, fm=12.0, delay=0.08), src, rec, models=[model])
    d.square().sum().backward()
    return d.detach(), model.grad


ACOUSTIC_CASES = [
    *[(2, order, fs, strips, None) for order in (4, 8) for fs in (False, True, ["top", "left"]) for strips in (False, True)],
    (2, 4, True, False, np.r_[np.zeros(30), np.arange(34) // 4].astype(np.int64)),
    # 3-D: the two corners; 2-D covers the free-surface x strips product.
    (3, 4, False, False, None),
    (3, 4, True, True, None),
]


@pytest.mark.parametrize("ndim, order, fs, strips, topography", ACOUSTIC_CASES,
                         ids=[f"{c[0]}d-so{c[1]}-fs{c[2]}-strips{c[3]}-topo{c[4] is not None}" for c in ACOUSTIC_CASES])
def test_restated_acoustic_is_bitwise(ndim, order, fs, strips, topography):
    hand = Acoustic if ndim == 2 else Acoustic3D
    gen = ACOUSTIC.build(ndim)
    d_ref, g_ref = _run(hand, ndim, order, fs, strips, topography)
    d_gen, g_gen = _run(gen, ndim, order, fs, strips, topography)
    assert torch.equal(d_ref, d_gen)
    assert torch.equal(g_ref, g_gen)


def _run_per_shot(cls, batch=2):
    """Two shots, each with its own 2-D model ``(B, nz, nx)``."""
    shape, nt = SHAPES[2], 150
    solver = PropTorch(cls(4, device=DEV), shape=shape, dh=DH, dt=DT, nt=nt, dev=DEV, impl="eager",
                       abcn=12, memory=Full(), eager_options=EagerOptions(use_compile=False))
    g = torch.Generator().manual_seed(0)
    model = (2000 + 300 * torch.rand((batch, *shape), generator=g)).to(DEV).requires_grad_(True)
    src = np.array([[20, 6], [40, 6]])
    rec = np.array([[[ix, 3] for ix in range(2, 62, 3)]] * batch)
    d = solver(ricker(nt, DT, fm=12.0, delay=0.08), src, rec, models=[model])
    d.square().sum().backward()
    return d.detach(), model.grad


def test_per_shot_batched_models_match_acoustic():
    gen = ACOUSTIC.build(2)
    assert gen.supports_batched_models and not ACOUSTIC.build(3).supports_batched_models
    d_ref, g_ref = _run_per_shot(Acoustic)
    d_gen, g_gen = _run_per_shot(gen)
    assert d_gen.shape[0] == 2 and g_gen.shape == g_ref.shape
    assert torch.equal(d_ref, d_gen)
    assert torch.equal(g_ref, g_gen)


def test_latex():
    assert ACOUSTIC.latex() == ("\\begin{aligned}\n"
                                "\\frac{\\partial^{2} p}{\\partial t^{2}} &= v_{p}^{2} \\nabla^{2} p\n"
                                "\\end{aligned}")
    assert ACOUSTIC._repr_latex_() == f"${ACOUSTIC.latex()}$"
    assert ACOUSTIC.equations[0].latex() == "\\frac{\\partial^{2} p}{\\partial t^{2}} = v_{p}^{2} \\nabla^{2} p"
    tex = VTI.latex(symbols={"p": r"\phi", vp: "c"})
    assert tex.count(" \\\\\n") == 1
    assert "\\frac{\\partial^{2} \\phi}{\\partial t^{2}} &= " in tex and "c^{2} \\sqrt{2 \\delta + 1}" in tex
    assert "\\left(2 \\epsilon + 1\\right)" in tex and "vp" not in tex


def _peak_bytes(cls, ndim):
    """Peak CUDA memory of an uncompiled Full() forward + backward, free
    surface on, above what was allocated before it."""
    _run(cls, ndim, 4, True, False)  # cuDNN / cuBLAS handles and workspaces
    gc.collect()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _run(cls, ndim, 4, True, False)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


@pytest.mark.skipif(DEV != "cuda", reason="the peak is read from the CUDA caching allocator")
@pytest.mark.parametrize("ndim", [2, 3])
def test_restated_acoustic_peak_memory(ndim):
    # A step that keeps its temporaries to the end holds them through the free
    # surface's new outputs: 4 fields above Acoustic, 7 above Acoustic3D here.
    # Half a field of slack for the caching allocator's block reuse.
    hand = Acoustic if ndim == 2 else Acoustic3D
    half_field = 2 * int(np.prod([n + 24 for n in SHAPES[ndim]]))  # fp32, abcn=12 a side
    assert _peak_bytes(ACOUSTIC.build(ndim), ndim) <= _peak_bytes(hand, ndim) + half_field


def _initialised(cls, shape=(20, 24)):
    """An equation whose CPML profiles the propagator has built."""
    eq = cls(4, device=DEV)
    solver = PropTorch(eq, shape=shape, dh=DH, dt=DT, nt=5, dev=DEV, impl="eager", abcn=8,
                       eager_options=EagerOptions(use_compile=False))
    nmod = len(eq.model_specs)
    with torch.no_grad():
        solver(ricker(5, DT), np.array([[10, 5]]), np.array([[[3, 3]]]),
               models=[torch.full(shape, v, device=DEV) for v in (2000.0, 0.1, 0.05)[:nmod]])
    return eq


def test_two_field_step_matches_reference():
    eq = _initialised(TWO_FIELD.build(2))
    nz, nx = eq.b[0].shape[1], eq.b[3].shape[2]
    g = torch.Generator().manual_seed(3)
    fields = [torch.randn(2, 1, nz, nx, generator=g).to(DEV) for _ in eq.wavefields]
    W = dict(zip(eq.wavefields, fields))
    vpm = (2000 + 300 * torch.rand(nz, nx, generator=g)).to(DEV)
    epsm, deltam = (0.1 * torch.rand(nz, nx, generator=g)).to(DEV), (0.05 * torch.rand(nz, nx, generator=g)).to(DEV)
    out = dict(zip(eq.wavefields, eq.func(fields, [vpm, epsm, deltam], DT, DH, None)))

    az, bz, dbz, ax, bx, dbx = eq.b
    prof = {"z": (az, bz, dbz, -2), "x": (ax, bx, dbx, -1)}
    ref = {}

    def d2(f, axis):
        lap = dict(zip("zx", eq.separable_d2_2d(W[f], eq.laplace_kernels, DH, DH)))[axis]
        a, b, db, i = prof[axis]
        du = eq.gradient(W[f], DH, i, kernels=eq.grad_kernels)
        c, ref[f"psi{axis}_{f}"], ref[f"zeta{axis}_{f}"] = cpml_axis_update(
            lap, du, W[f"psi{axis}_{f}"], W[f"zeta{axis}_{f}"], a, b, db, DH, i, eq.gradient, eq.grad_kernels)
        return c

    pxx, qzz, qxx = d2("p", "x"), d2("q", "z"), d2("q", "x")
    s = torch.sqrt(1 + 2 * deltam)
    ref["p"] = 2 * W["p"] - W["p_prev"] + DT**2 * (vpm**2 * (1 + 2 * epsm) * pxx + vpm**2 * s * qzz - 0.5 * vpm * W["q"])
    ref["q"] = 2 * W["q"] - W["q_prev"] + DT**2 * (vpm**2 * s * pxx + vpm**2 * (qzz + qxx))
    ref["p_prev"], ref["q_prev"] = W["p"], W["q"]
    assert sorted(ref) == sorted(eq.wavefields)
    for name in eq.wavefields:
        scale = ref[name].abs().max()
        assert (out[name] - ref[name]).abs().max() <= 1e-5 * scale, name


def _vti_grads(memory, use_compile=False, fullgraph=False, backend=None):
    nz, nx, nt = 48, 56, 120
    solver = PropTorch(VTI.build(2)(4, device=DEV), shape=(nz, nx), dh=DH, dt=1.5e-3, nt=nt, dev=DEV,
                       impl="eager", abcn=30, free_surface=True, memory=memory,
                       eager_options=EagerOptions(use_compile=use_compile, compile_fullgraph=fullgraph,
                                                  compile_backend=backend))
    z = torch.arange(nz, dtype=torch.float32)[:, None].expand(nz, nx)
    models = [(1800 + 600 * z / (nz - 1)).to(DEV), torch.full((nz, nx), 0.05, device=DEV),
              torch.full((nz, nx), 0.03, device=DEV)]
    models = [m.clone().requires_grad_(True) for m in models]
    d = solver(ricker(nt, 1.5e-3), np.array([[nx // 2, nz // 4]]), np.array([[[ix, 2] for ix in range(0, nx, 6)]]),
               models=models)
    d.square().sum().backward()
    return d.detach(), [m.grad for m in models]


def _masked_rel(a, b, nz=48, nx=56):
    # The source cell's own gradient is the least stable number of the map.
    mask = torch.ones_like(a)
    mask[nz // 4 - 2:nz // 4 + 3, nx // 2 - 2:nx // 2 + 3] = 0
    return ((a - b).abs() * mask).max() / (a.abs() * mask).max()


def test_coupled_boundary_saving_matches_full():
    # Measured 2e-5 .. 9e-5 (Ada, compiled, every model); Acoustic on the same
    # case 1.4e-5 with this free surface.
    _, g_full = _vti_grads(Full())
    _, g_bs = _vti_grads(BoundarySaving(storage="gpu"))
    for a, b in zip(g_full, g_bs):
        assert _masked_rel(a, b) <= 1e-3


@requires_compile
def test_coupled_compiles_in_one_graph():
    # aot_eager: Dynamo capture (the fullgraph check) without a C++ toolchain.
    # Compiled, the stencils are slices summed in another order. Measured (Ada):
    # record 5.8e-6, gradients 6e-6 .. 2.6e-5 (inductor: 1.5e-5, 8e-6 .. 2.6e-5).
    d_ref, g_ref = _vti_grads(Full())
    d_c, g_c = _vti_grads(Full(), use_compile=True, fullgraph=True, backend="aot_eager")
    assert (d_c - d_ref).abs().max() <= 1e-4 * d_ref.abs().max()
    for a, b in zip(g_ref, g_c):
        assert _masked_rel(a, b) <= 3e-4


@pytest.fixture
def _restore_registry():
    """Other modules read the global equation registry; leave it as found."""
    from sweep.equations import _registry
    saved = [dict(d) for d in (_registry._REGISTRY, _registry._CANONICAL, _registry._METHOD)]
    yield
    for d, old in zip((_registry._REGISTRY, _registry._CANONICAL, _registry._METHOD), saved):
        d.clear()
        d.update(old)


def test_export_and_register(tmp_path, _restore_registry):
    path = tmp_path / "sym_exported.py"
    ACOUSTIC.export(path, ndim=2)
    spec = importlib.util.spec_from_file_location("sym_exported", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert torch.equal(_run(module.SymAcoustic, 2, 4, True, False)[1], _run(ACOUSTIC.build(2), 2, 4, True, False)[1])

    from sweep.equations import get_equation
    cls = PDE(ACOUSTIC.equations, name="SymRegisteredAcoustic", free_surface="zero").build(2, register=True)
    assert get_equation("SymRegisteredAcoustic") is cls
    # A rebuild (a notebook cell run again) is a redefinition, not a clash.
    again = PDE(ACOUSTIC.equations, name="SymRegisteredAcoustic", free_surface="zero").build(2, register=True)
    assert get_equation("SymRegisteredAcoustic") is again
    with pytest.raises(ValueError, match="already registered"):
        PDE(ACOUSTIC.equations, name="Acoustic").build(2, register=True)


def test_no_pml_has_no_memory_fields():
    eq = PDE(ACOUSTIC.equations, name="SymAcousticNoPML", pml="none").build(2)(4)
    assert eq.wavefields == ["p", "p_prev"]
    assert eq.models == ["vp"]


def test_undeclared_free_surface_is_refused():
    cls = PDE(ACOUSTIC.equations, name="SymAcousticNoFS").build(2)
    with pytest.raises(NotImplementedError, match="without a free surface"):
        PropTorch(cls(4, device=DEV), shape=(20, 24), dh=DH, dt=DT, nt=5, dev=DEV, impl="eager", abcn=8,
                  free_surface=True)
    with pytest.raises(NotImplementedError, match="free_surface='zero'"):
        PropTorch(cls(4, device=DEV), shape=(20, 24), dh=DH, dt=DT, nt=5, dev=DEV, impl="eager", abcn=8,
                  topography=np.zeros(24, dtype=np.int64))


@pytest.mark.parametrize("equations, error, match", [
    ([Eq(p.dt2, vp**2 * p.dx)], NotImplementedError, "first spatial derivative"),
    ([Eq(p.dt, vp * p.dx2)], ValueError, "declared with time_order=2"),
    ([Eq(p.dt2, vp**2 * laplace(p) - 0.1 * q.dt)], NotImplementedError, "first time derivative of another field"),
    ([Eq(p.dt2, p * p.dx2)], ValueError, "not linear"),
    ([Eq(p.dt2, vp**2 * laplace(p) + vp)], ValueError, "contains no field"),
    ([Eq(p.dt2, sympy.Symbol("c") * laplace(p))], ValueError, "neither a Model, a Coefficient nor a number"),
    ([Eq(p.dt2, vp**2 * laplace(q))], ValueError, "no equation of its own"),
    ([Eq(p.dt2, vp**2 * laplace(p)), Eq(p.dt2, vp**2 * p.dx2)], ValueError, "more than one equation"),
    ([Eq(vp, laplace(p))], ValueError, "no time derivative"),
    ([Eq(q.dt2, vp**2 * laplace(q))], ValueError, "no field has source=True"),
], ids=["d1", "dt1", "other-dt", "nonlinear", "source-term", "unknown-symbol", "orphan-field",
        "two-equations", "no-dt", "no-source"])
def test_unsupported_pde_is_refused(equations, error, match):
    with pytest.raises(error, match=match):
        PDE(equations, name="SymBad")


def test_build_checks_dimension_names_and_models():
    pde = PDE([Eq(p.dt2, vp**2 * (p.dx2 + p.dy2 + p.dz2))], name="SymY")
    with pytest.raises(ValueError, match="y-derivative in a 2-D build"):
        pde.build(2)
    assert pde.build(3)(4).wavefields[:2] == ["p", "p_prev"]
    with pytest.raises(ValueError, match="used twice"):
        PDE([Eq(p.dt2, Model("p_prev") * laplace(p))], name="SymClash").build(2)
    with pytest.raises(ValueError, match="misses eps"):
        PDE([Eq(p.dt2, vp**2 * (1 + 2 * eps) * laplace(p))], name="SymM", models=[vp])
    with pytest.raises(ValueError, match="which no equation uses"):
        PDE(ACOUSTIC.equations, name="SymM", models=[vp, eps])
    with pytest.raises(ValueError, match="taken by the generated module"):
        PDE(ACOUSTIC.equations, name="FieldSpec")
    with pytest.raises(ValueError, match="reserved"):
        Field("dt")
    with pytest.raises(NotImplementedError, match="time_order"):
        Field("u", time_order=3)
    with pytest.raises(NotImplementedError, match="'torch' only"):
        ACOUSTIC.build(2)(4, backend="jax")
