"""3-D velocity-stress elastic on a DEFORMED Hex27 mesh, impl='c' vs eager.

ElasticVSSEM3DDeformed: topography with a natural free surface on top and the
simple C-PML on the other five sides (or bare). The interior is also sheared in
x and y, so all nine inv_J streams are non-zero and a transposed metric index
cannot hide behind a zero cross term (a topography-only mesh leaves six of them
at zero).

* seis_vx/vy/vz and dL/d{vp,vs,rho} vs eager full-tape autograd: cos >= 1-1e-4;
* a record long enough to go through the C-PML and back: relative L2, records
  1e-5 and gradients 1e-4, as in test_sem_c_elastic_vs3d.py;
* a flat Hex27 reproduces the Hex8 CUDA path (ElasticVSSEM3D), records and
  gradients, to fp32 round-off;
* PropSEM(topography=..., impl='c') takes the CUDA path and agrees with eager.

Force source scaled up (1e6) to lift the gradients off the fp32 noise floor.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM, SEMPropagator
from sweep.sem.cpml.elastic_vs3d import build_simple_elastic_cpml_3d
from sweep.sem.equations.elastic_vs3d import ElasticVSSEM3D
from sweep.sem.equations.elastic_vs3d_deformed import ElasticVSSEM3DDeformed
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.hex27 import (_pack_control, build_axis_aligned_hex27,
                                  hex27_from_control_nodes)
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEY, NEZ = 7, 6, 6
XMAX, YMAX, ZMAX = 700.0, 600.0, 600.0
DT, NT, F0, TSH = 4e-4, 180, 16.0, 0.07
# Every point sits at a column centre: the Hex27 locator takes the column from
# x/dx_e, y/dy_e (uniform columns), and the interior shear below moves a column
# by at most 22 m, well inside its 100 m width.
SRC = (350.0, 250.0, 250.0)
REC = [(x, y, 170.0) for y in (150.0, 350.0)
       for x in (150.0, 250.0, 350.0, 450.0, 550.0)]
PAYLOAD = (1.0e6, 0.7e6, 0.5e6)
SIDES = ("left", "right", "front", "back", "bottom")


def _curved_mesh():
    """Gaussian valley on top, graded columns, plus an interior x/y shear that
    vanishes on every outer face (the C-PML sides stay planar)."""
    nc = lambda ne: 2 * ne + 1
    d = torch.float64
    x = torch.linspace(0.0, XMAX, nc(NEX), dtype=d)
    y = torch.linspace(0.0, YMAX, nc(NEY), dtype=d)
    f = torch.linspace(0.0, 1.0, nc(NEZ), dtype=d)
    F, Y, X = torch.meshgrid(f, y, x, indexing="ij")
    top = 60.0 * torch.exp(-(((X - XMAX * 0.5) / 170.0) ** 2
                             + ((Y - YMAX * 0.5) / 150.0) ** 2))
    Z = top + F * (ZMAX - top)
    bump = (torch.sin(np.pi * X / XMAX) * torch.sin(np.pi * Y / YMAX)
            * torch.sin(np.pi * F))
    Xs = X + 22.0 * bump * torch.cos(2 * np.pi * Y / YMAX)
    Ys = Y + 18.0 * bump * torch.sin(2 * np.pi * X / XMAX)
    ctl = _pack_control(Xs.contiguous(), Ys.contiguous(), Z.contiguous(),
                        NEZ, NEY, NEX)
    return hex27_from_control_nodes(ctl, NEX, NEY, NEZ, n_order=4,
                                    dtype=torch.float32)


def _models(m, grad):
    vp = (2400.0 + 300.0 * (m.z_e / ZMAX)).to(torch.float32).contiguous()
    vs = (1200.0 + 200.0 * (m.x_e / XMAX)).to(torch.float32).contiguous()
    rho = (2000.0 + 150.0 * (m.y_e / YMAX)).to(torch.float32).contiguous()
    if grad:
        vp, vs, rho = (vp.requires_grad_(True), vs.requires_grad_(True),
                       rho.requires_grad_(True))
    return vp, vs, rho


def _eq(cls, m, cpml, grad=False):
    vp, vs, rho = _models(m, grad)
    kw = {}
    if cpml:
        kw["cpml"] = build_simple_elastic_cpml_3d(
            m, dt=DT, sides=SIDES, pml_thickness_elements=2,
            c_max=2700.0, f_dominant=F0)
    return cls.build(m, vp, vs, rho, **kw), (vp, vs, rho)


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


def test_the_mesh_exercises_every_metric_stream():
    """Guard against a vacuous comparison: each of the nine inv_J streams must
    carry a non-trivial share of the diagonal's size somewhere."""
    J = _curved_mesh().inv_J
    diag = float(J[..., 0, 0].abs().max())
    for a in range(3):
        for j in range(3):
            assert float(J[..., a, j].abs().max()) > 0.02 * diag, (a, j)


@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
def test_forward_matches_eager(cpml):
    eq, _ = _eq(ElasticVSSEM3DDeformed, _curved_mesh(), cpml)
    oe = _prop(eq, False).forward(_wav())
    oc = _prop(eq, True).forward(_wav())
    for j, nm in enumerate(("v_x", "v_y", "v_z")):
        c = _cos(oc[..., j].T.cpu(), oe[..., j].T.cpu())
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"


@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
def test_grads_match_eager(cpml):
    def run(uc):
        eq, (vp, vs, rho) = _eq(ElasticVSSEM3DDeformed, _curved_mesh(), cpml,
                                grad=True)
        _weighted_loss(_prop(eq, uc).forward(_wav())).backward()
        return vp.grad.cpu(), vs.grad.cpu(), rho.grad.cpu()
    ge = run(False)
    gc = run(True)
    for name, e, c in zip(("vp", "vs", "rho"), ge, gc):
        assert _cos(c, e) >= 1 - 1e-4, f"grad {name} cos={_cos(c, e):.6f}"


def test_cpml_matches_eager_in_the_layer():
    """The C-PML terms on the curved mesh: a source next to the bottom layer and
    a record long enough to go through it and come back (the cosine tests end
    before the wave reaches the layer). Relative L2, records and every
    gradient."""
    nt = 450
    src = (350.0, 250.0, ZMAX - 260.0)

    def run(uc):
        eq, (vp, vs, rho) = _eq(ElasticVSSEM3DDeformed, _curved_mesh(), True,
                                grad=True)
        o = _prop(eq, uc, nt=nt, src=src).forward(_wav(nt))
        o.square().sum().backward()
        return o.detach().cpu(), vp.grad.cpu(), vs.grad.cpu(), rho.grad.cpu()
    oe, *ge = run(False)
    oc, *gc = run(True)
    assert _rel(oc, oe) < 1e-5, f"record rel {_rel(oc, oe):.2e}"
    for name, e, c in zip(("vp", "vs", "rho"), ge, gc):
        assert _rel(c, e) < 1e-4, f"grad {name} rel {_rel(c, e):.2e}"


def test_flat_hex27_matches_the_hex8_cuda_path():
    """Same global node set, same C-PML profile: the deformed kernels on an
    axis-aligned Hex27 against the Hex8 kernels, both on CUDA. The zero cross
    terms still enter the sums, so this is fp32 round-off, not bitwise."""
    h8 = build_hex8_mesh(NEX, NEY, NEZ, XMAX, YMAX, ZMAX, n_order=4,
                         dtype=torch.float32)
    h27 = build_axis_aligned_hex27(NEX, NEY, NEZ, XMAX, YMAX, ZMAX, n_order=4,
                                   dtype=torch.float32)
    assert torch.equal(h8.ibool, h27.ibool)

    def run(cls, m):
        eq, (vp, vs, rho) = _eq(cls, m, True, grad=True)
        o = _prop(eq, True).forward(_wav())
        _weighted_loss(o).backward()
        return o.detach().cpu(), vp.grad.cpu(), vs.grad.cpu(), rho.grad.cpu()
    o8, *g8 = run(ElasticVSSEM3D, h8)
    o27, *g27 = run(ElasticVSSEM3DDeformed, h27)
    assert _rel(o27, o8) < 1e-5, f"record rel {_rel(o27, o8):.2e}"
    for name, a, b in zip(("vp", "vs", "rho"), g8, g27):
        assert _rel(b, a) < 1e-4, f"grad {name} rel {_rel(b, a):.2e}"


# ------------------------ PropSEM(topography=...) -------------------------- #
NZ, NY, NX = 21, 17, 25
DH, PDT, PNT, AB = 20.0, 1.0e-3, 200, 3
PSRC = np.array([[12, 8, 4]], dtype=np.int64)
_rix = np.arange(4, NX - 4, 2)
PREC = np.stack([_rix, np.full_like(_rix, 8), np.full_like(_rix, 5)], -1)[None]


def _ztop_rows():
    Y, X = torch.meshgrid(torch.arange(NY, dtype=torch.float64) * DH,
                          torch.arange(NX, dtype=torch.float64) * DH,
                          indexing="ij")
    top = 50.0 * torch.exp(-(((X - NX * DH * 0.5) / 140.0) ** 2
                             + ((Y - NY * DH * 0.5) / 120.0) ** 2))
    return (top / DH).numpy()


def _run_propsem(impl):
    p = PropSEM(ElasticVSSEM3DDeformed, shape=(NZ, NY, NX), dh=DH, dt=PDT,
                nt=PNT, n_order=4, free_surface=True, abcn=AB * 4,
                f_dominant=8.0, topography=_ztop_rows(), dev="cuda",
                dtype=torch.float32, impl=impl)
    assert p.impl == impl
    z = torch.arange(NZ, dtype=torch.float32)[:, None, None]
    x = torch.arange(NX, dtype=torch.float32)[None, None, :]
    shape = (NZ, NY, NX)
    vp = (2200.0 + 15.0 * z + 120.0 * torch.sin(x / 7.0)).expand(shape)
    vs = (vp / 1.8)
    rho = torch.full(shape, 2100.0)
    ms = [t.contiguous().cuda().requires_grad_(True) for t in (vp, vs, rho)]
    t = np.arange(PNT) * PDT
    a = (np.pi * 8.0 * (t - 0.15)) ** 2
    wav = torch.tensor(1e10 * (1 - 2 * a) * np.exp(-a), dtype=torch.float32,
                       device="cuda")
    s = p(wav, PSRC, PREC, models=ms)
    W = torch.linspace(0.4, 1.6, s.numel(), device=s.device).reshape(s.shape)
    (W * s).sum().backward()
    return (s.detach().cpu().numpy(),) + tuple(m.grad.cpu().numpy() for m in ms)


def test_propsem_topography_takes_the_cuda_path():
    se, *ge = _run_propsem("eager")
    sc, *gc = _run_propsem("c")
    assert _cos(sc, se) >= 1 - 1e-4, f"fwd cos={_cos(sc, se):.6f}"
    for name, e, c in zip(("vp", "vs", "rho"), ge, gc):
        assert _cos(c, e) >= 1 - 1e-4, f"grad {name} cos={_cos(c, e):.6f}"
