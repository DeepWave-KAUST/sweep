"""The 2-D velocity-stress family on CUDA (impl='c') against eager.

ElasticVSSEM2DDeformed and its subclasses ElasticVTISEM2DDeformed,
ElasticTTISEM2DDeformed and ViscoElasticVSSEM2DDeformed run the kernels of
``sem2d/vs2d.cu``: the per-node 2x2 inverse Jacobian, the C-PML on the divergence
split by direction and on the velocity gradients, the stiffnesses (4 / 6) or the
GSLS memories in the stress update. The mesh has a topography and an interior
shear, so all four metric terms are non-zero.

* records and the gradients in every model against eager, cos >= 1-1e-4, with the
  C-PML on three sides (and the isotropic class bare);
* a record long enough to go through the C-PML and back: relative L2, records 1e-5
  and gradients 1e-4;
* PropSEM(topography=..., impl='c') takes the CUDA path;
* the strong free surface (free_top=True, which PropSEM never sets) is refused.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM, SEMPropagator
from sweep.sem.cpml.elastic_vs import build_simple_elastic_cpml
from sweep.sem.equations.elastic_vs_deformed import ElasticVSSEM2DDeformed
from sweep.sem.equations.elastic_vti_vs_deformed import (ElasticTTISEM2DDeformed,
                                                         ElasticVTISEM2DDeformed)
from sweep.sem.equations.visco_elastic_vs_deformed import ViscoElasticVSSEM2DDeformed
from sweep.sem.mesh.quad9 import quad9_from_control_nodes
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEZ = 12, 8
XMAX, ZMAX = 1200.0, 800.0
DT, NT, F0, TSH = 5e-4, 400, 15.0, 0.08
# every point at a column centre: the Quad9 locator takes the column from x/dx_e,
# and the interior shear moves a column by at most 20 m
SRC = (550.0, 300.0)
REC = [(float(x), 160.0) for x in np.arange(150.0, 1100.0, 100.0)]
PAYLOAD = (1.0e6, 0.7e6)
SIDES = ("left", "right", "bottom")


def _curved():
    """Gaussian valley on top, graded columns, and an interior x shear that
    vanishes on the sides, the top and the bottom."""
    d = torch.float64
    x = torch.linspace(0.0, XMAX, 2 * NEX + 1, dtype=d)
    f = torch.linspace(0.0, 1.0, 2 * NEZ + 1, dtype=d)
    F, X = torch.meshgrid(f, x, indexing="ij")
    top = 60.0 * torch.exp(-(((X - XMAX * 0.45) / 220.0) ** 2))
    Z = top + F * (ZMAX - top)
    Xs = X + 20.0 * torch.sin(np.pi * X / XMAX) * torch.sin(np.pi * F)
    control = torch.zeros(NEZ, NEX, 3, 3, 2, dtype=d)
    for ez in range(NEZ):
        for ex in range(NEX):
            for i in range(3):
                for j in range(3):
                    control[ez, ex, i, j, 0] = Xs[2 * ez + i, 2 * ex + j]
                    control[ez, ex, i, j, 1] = Z[2 * ez + i, 2 * ex + j]
    return quad9_from_control_nodes(control, n_order=4, dtype=torch.float32)


MODELS = {
    ElasticVSSEM2DDeformed: ("vp", "vs", "rho"),
    ElasticVTISEM2DDeformed: ("vp", "vs", "rho", "epsilon", "delta"),
    ElasticTTISEM2DDeformed: ("vp", "vs", "rho", "epsilon", "delta", "theta"),
    ViscoElasticVSSEM2DDeformed: ("vp", "vs", "rho", "Qp", "Qs"),
}


def _models(m, names):
    x, z = m.x_e / XMAX, m.z_e / ZMAX
    v = {"vp": 2400.0 + 300.0 * z, "vs": 1250.0 + 150.0 * x, "rho": 2000.0 + 150.0 * z,
         "epsilon": 0.2 + 0.05 * torch.sin(3.0 * x), "delta": 0.1 + 0.03 * x,
         "theta": 0.5 + 0.1 * z, "Qp": 40.0 + 30.0 * z, "Qs": 25.0 + 20.0 * x}
    return [v[n].to(torch.float32).contiguous() for n in names]


def _eq(cls, m, cpml=True, grad=False):
    names = MODELS[cls]
    ms = _models(m, names)
    if grad:
        ms = [t.requires_grad_(True) for t in ms]
    kw = {}
    if cpml:
        kw["cpml"] = build_simple_elastic_cpml(m, dt=DT, sides=SIDES, pml_thickness_elements=2,
                                               c_max=3300.0, f_dominant=F0)
    if cls is ViscoElasticVSSEM2DDeformed:
        kw["f_ref"] = F0
    return cls.build(m, *ms, **kw), dict(zip(names, ms))


def _wav(nt=NT):
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _prop(eq, use_cuda, nt=NT, src=SRC):
    return (SEMCudaPropagator if use_cuda else SEMPropagator).build(
        eq, dt=DT, nt=nt, source_xz=[src], receiver_xz=REC,
        source_kinds=["force"], source_payloads=[PAYLOAD], pml_guard="off")


def _weighted_loss(o):
    loss = 0.0
    for j in range(o.shape[-1]):                 # every component
        s = o[..., j].T
        W = torch.linspace(0.4 + 0.1 * j, 1.6 - 0.1 * j, s.numel(),
                           device=s.device).reshape(s.shape)
        loss = loss + (W * s).sum()
    return loss


def test_the_mesh_exercises_every_metric_term():
    m = _curved()
    J = m.inv_J
    diag = float(J[..., 0, 0].abs().max())
    for a in range(2):
        for j in range(2):
            assert float(J[..., a, j].abs().max()) > 0.02 * diag, (a, j)


CASES = [(ElasticVSSEM2DDeformed, True), (ElasticVSSEM2DDeformed, False),
         (ElasticVTISEM2DDeformed, True), (ElasticTTISEM2DDeformed, True),
         (ViscoElasticVSSEM2DDeformed, True)]


@pytest.mark.parametrize("cls,cpml", CASES,
                         ids=[f"{c.__name__}-{'cpml' if p else 'bare'}" for c, p in CASES])
def test_records_and_gradients_match_eager(cls, cpml):
    def run(uc):
        eq, ms = _eq(cls, _curved(), cpml=cpml, grad=True)
        o = _prop(eq, uc).forward(_wav())
        _weighted_loss(o).backward()
        return o.detach().cpu(), {k: t.grad.cpu() for k, t in ms.items()}
    oe, ge = run(False)
    oc, gc = run(True)
    for j, nm in enumerate(("v_x", "v_z")):
        c = _cos(oc[..., j].T, oe[..., j].T)
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"
    for name in ge:
        c = _cos(gc[name], ge[name])
        assert c >= 1 - 1e-4, f"grad {name} cos={c:.6f}"


@pytest.mark.parametrize("cls", [ElasticVSSEM2DDeformed, ElasticTTISEM2DDeformed,
                                 ViscoElasticVSSEM2DDeformed])
def test_cpml_matches_eager_in_the_layer(cls):
    """A source next to the bottom layer and a record long enough to go through
    it and come back. Relative L2, records and every gradient."""
    nt = 900
    src = (550.0, ZMAX - 230.0)

    def run(uc):
        eq, ms = _eq(cls, _curved(), grad=True)
        o = _prop(eq, uc, nt=nt, src=src).forward(_wav(nt))
        o.square().sum().backward()
        return o.detach().cpu(), {k: t.grad.cpu() for k, t in ms.items()}
    oe, ge = run(False)
    oc, gc = run(True)
    assert _rel(oc, oe) < 1e-5, f"record rel {_rel(oc, oe):.2e}"
    for name in ge:
        assert _rel(gc[name], ge[name]) < 1e-4, f"grad {name} rel {_rel(gc[name], ge[name]):.2e}"


def test_free_top_is_refused():
    m = _curved()
    vp, vs, rho = _models(m, ("vp", "vs", "rho"))
    eq = ElasticVSSEM2DDeformed.build(m, vp, vs, rho, free_top=True)
    with pytest.raises(NotImplementedError, match="free_top"):
        _prop(eq, True).forward(_wav())


# ------------------------ PropSEM(topography=...) -------------------------- #
NZ, NX = 41, 61
DH, PDT, PNT, AB = 10.0, 5e-4, 400, 3
PSRC = np.array([[30, 6]], dtype=np.int64)
PREC = np.stack([np.arange(6, NX - 6, 3), np.full(len(range(6, NX - 6, 3)), 8)], -1)[None]


def _run_propsem(cls, impl, extra):
    top = 40.0 * np.exp(-(((np.arange(NX) * DH - NX * DH * 0.5) / 120.0) ** 2))
    p = PropSEM(cls, shape=(NZ, NX), dh=DH, dt=PDT, nt=PNT, n_order=4, free_surface=True,
                abcn=AB * 4, f_dominant=15.0, topography=top / DH, dev="cuda",
                dtype=torch.float32, impl=impl)
    assert p.impl == impl
    z = torch.arange(NZ, dtype=torch.float32)[:, None].expand(NZ, NX)
    x = torch.arange(NX, dtype=torch.float32)[None, :].expand(NZ, NX)
    vp = 2200.0 + 15.0 * z + 120.0 * torch.sin(x / 7.0)
    vals = (vp, vp / 1.8, torch.full((NZ, NX), 2100.0)) + tuple(f(z, x) for f in extra)
    ms = [t.contiguous().cuda().requires_grad_(True) for t in vals]
    t = np.arange(PNT) * PDT
    a = (np.pi * 15.0 * (t - 0.08)) ** 2
    wav = torch.tensor(1e10 * (1 - 2 * a) * np.exp(-a), dtype=torch.float32, device="cuda")
    s = p(wav, PSRC, PREC, models=ms)
    W = torch.linspace(0.4, 1.6, s.numel(), device=s.device).reshape(s.shape)
    (W * s).sum().backward()
    return (s.detach().cpu().numpy(),) + tuple(m.grad.cpu().numpy() for m in ms)


PROPSEM_CASES = [
    (ElasticVSSEM2DDeformed, ()),
    (ElasticTTISEM2DDeformed, (lambda z, x: 0.2 + 0 * z, lambda z, x: 0.1 + 0.002 * x,
                               lambda z, x: 0.4 + 0.01 * z)),
    (ViscoElasticVSSEM2DDeformed, (lambda z, x: 40.0 + z, lambda z, x: 25.0 + 0.5 * x)),
]


@pytest.mark.parametrize("cls,extra", PROPSEM_CASES, ids=[c.__name__ for c, _ in PROPSEM_CASES])
def test_propsem_topography_takes_the_cuda_path(cls, extra):
    se, *ge = _run_propsem(cls, "eager", extra)
    sc, *gc = _run_propsem(cls, "c", extra)
    assert _cos(sc, se) >= 1 - 1e-4, f"fwd cos={_cos(sc, se):.6f}"
    for k, (e, c) in enumerate(zip(ge, gc)):
        assert _cos(c, e) >= 1 - 1e-4, f"grad {k} cos={_cos(c, e):.6f}"
