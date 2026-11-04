"""VTI / TTI media on the 3-D SEM, impl='c' vs eager.

ElasticVTISEM3D, ElasticTTISEM3D and their Hex27 versions run the velocity-stress
kernels with the Voigt stiffnesses (nine for VTI, 21 for TTI) in place of the
Lame set; the kernels return the stiffness gradients and autograd takes them on
to the Thomsen parameters through the class's ``thomsen_to_cij3d``.

* records and the gradients in every model (``vp, vs, rho, epsilon, delta, gamma``
  and, for TTI, the tilt ``theta`` and azimuth ``phi``) against eager, cos >=
  1-1e-4: Hex8 with the C-PML, and a curved Hex27 (a topography and an interior
  shear, so that all nine metric terms are non-zero) with the C-PML;
* a record long enough to go through the C-PML and back, TTI on the curved
  Hex27: relative L2, records 1e-5 and gradients 1e-4;
* a TTI medium without anisotropy runs the isotropic kernels' numbers (a check of
  the Voigt order and of the stress-row mapping against an independent path);
* PropSEM(topography=..., impl='c') takes the CUDA path for the TTI class.

Force source scaled up (1e6) to lift the gradients off the fp32 noise floor.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM, SEMPropagator
from sweep.sem.cpml.elastic_vs3d import build_simple_elastic_cpml_3d
from sweep.sem.equations.elastic_tti_vs3d import (ElasticTTISEM3D, ElasticTTISEM3DDeformed,
                                                  ElasticVTISEM3D, ElasticVTISEM3DDeformed)
from sweep.sem.equations.elastic_vs3d_deformed import ElasticVSSEM3DDeformed
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.hex27 import _pack_control, hex27_from_control_nodes
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEY, NEZ = 7, 6, 6
XMAX, YMAX, ZMAX = 700.0, 600.0, 600.0
DT, NT, F0, TSH = 4e-4, 180, 16.0, 0.07
# every point at a column centre: the Hex27 locator takes the column from x/dx_e,
# y/dy_e, and the interior shear moves a column by at most 22 m
SRC = (350.0, 250.0, 250.0)
REC = [(x, y, 170.0) for y in (150.0, 350.0)
       for x in (150.0, 250.0, 350.0, 450.0, 550.0)]
PAYLOAD = (1.0e6, 0.7e6, 0.5e6)
SIDES = ("left", "right", "front", "back", "bottom")
VTI_MODELS = ("vp", "vs", "rho", "epsilon", "delta", "gamma")
TTI_MODELS = VTI_MODELS + ("theta", "phi")


def _hex8():
    return build_hex8_mesh(NEX, NEY, NEZ, XMAX, YMAX, ZMAX, n_order=4,
                           dtype=torch.float32)


def _curved():
    """Gaussian valley on top, graded columns, and an interior x/y shear that
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


def _models(m, names, aniso=True):
    x, y, z = m.x_e / XMAX, m.y_e / YMAX, m.z_e / ZMAX
    a = 1.0 if aniso else 0.0
    v = {"vp": 2400.0 + 300.0 * z, "vs": 1250.0 + 150.0 * x, "rho": 2000.0 + 150.0 * y,
         "epsilon": a * (0.2 + 0.05 * torch.sin(3.0 * x)), "delta": a * (0.1 + 0.03 * x),
         "gamma": a * (0.15 + 0.03 * y), "theta": 0.5 + 0.1 * z, "phi": 0.7 + 0.2 * x}
    return [v[n].to(torch.float32).contiguous() for n in names]


def _cpml(m):
    return build_simple_elastic_cpml_3d(m, dt=DT, sides=SIDES, pml_thickness_elements=2,
                                        c_max=3300.0, f_dominant=F0)


def _eq(cls, m, grad=False, aniso=True):
    names = TTI_MODELS if issubclass(cls, ElasticTTISEM3D) and not issubclass(
        cls, ElasticVTISEM3D) else VTI_MODELS
    ms = _models(m, names, aniso)
    if grad:
        ms = [t.requires_grad_(True) for t in ms]
    return cls.build(m, *ms, cpml=_cpml(m)), dict(zip(names, ms))


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


CASES = [(ElasticVTISEM3D, _hex8), (ElasticTTISEM3D, _hex8),
         (ElasticVTISEM3DDeformed, _curved), (ElasticTTISEM3DDeformed, _curved)]


@pytest.mark.parametrize("cls,mesh", CASES, ids=[c.__name__ for c, _ in CASES])
def test_records_and_gradients_match_eager(cls, mesh):
    def run(uc):
        eq, ms = _eq(cls, mesh(), grad=True)
        o = _prop(eq, uc).forward(_wav())
        _weighted_loss(o).backward()
        return o.detach().cpu(), {k: t.grad.cpu() for k, t in ms.items()}
    oe, ge = run(False)
    oc, gc = run(True)
    for j, nm in enumerate(("v_x", "v_y", "v_z")):
        c = _cos(oc[..., j].T, oe[..., j].T)
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"
    for name in ge:
        c = _cos(gc[name], ge[name])
        assert c >= 1 - 1e-4, f"grad {name} cos={c:.6f}"


def test_tti_cpml_matches_eager_in_the_layer():
    """The C-PML terms with the 21 stiffnesses on the curved mesh: a source next
    to the bottom layer and a record long enough to go through it and come back.
    Relative L2, records and every gradient."""
    nt = 450
    src = (350.0, 250.0, ZMAX - 260.0)

    def run(uc):
        eq, ms = _eq(ElasticTTISEM3DDeformed, _curved(), grad=True)
        o = _prop(eq, uc, nt=nt, src=src).forward(_wav(nt))
        o.square().sum().backward()
        return o.detach().cpu(), {k: t.grad.cpu() for k, t in ms.items()}
    oe, ge = run(False)
    oc, gc = run(True)
    assert _rel(oc, oe) < 1e-5, f"record rel {_rel(oc, oe):.2e}"
    for name in ge:
        assert _rel(gc[name], ge[name]) < 1e-4, f"grad {name} rel {_rel(gc[name], ge[name]):.2e}"


def test_no_anisotropy_runs_the_isotropic_numbers():
    """epsilon = delta = gamma = 0 with a tilt: the 21-stiffness kernels against the
    Lame-set kernels on the same curved mesh, both on CUDA."""
    m = _curved()
    eq, _ = _eq(ElasticTTISEM3DDeformed, m, aniso=False)
    vp, vs, rho = _models(m, ("vp", "vs", "rho"))
    iso = ElasticVSSEM3DDeformed.build(m, vp, vs, rho, cpml=_cpml(m))
    a = _prop(eq, True).forward(_wav()).cpu()
    b = _prop(iso, True).forward(_wav()).cpu()
    assert _rel(a, b) < 1e-5, f"rel {_rel(a, b):.2e}"


# ------------------------ PropSEM(topography=...) -------------------------- #
NZ, NY, NX = 21, 17, 25
DH, PDT, PNT, AB = 20.0, 8.0e-4, 250, 3
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
    p = PropSEM(ElasticTTISEM3DDeformed, shape=(NZ, NY, NX), dh=DH, dt=PDT,
                nt=PNT, n_order=4, free_surface=True, abcn=AB * 4,
                f_dominant=8.0, topography=_ztop_rows(), dev="cuda",
                dtype=torch.float32, impl=impl)
    assert p.impl == impl
    shape = (NZ, NY, NX)
    z = torch.arange(NZ, dtype=torch.float32)[:, None, None].expand(shape)
    x = torch.arange(NX, dtype=torch.float32)[None, None, :].expand(shape)
    vp = 2200.0 + 15.0 * z + 120.0 * torch.sin(x / 7.0)
    vals = (vp, vp / 1.8, torch.full(shape, 2100.0), 0.2 + 0.0 * z, 0.1 + 0.002 * x,
            0.12 + 0.0 * z, 0.4 + 0.01 * z, 0.6 + 0.01 * x)
    ms = [t.contiguous().cuda().requires_grad_(True) for t in vals]
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
    for name, e, c in zip(TTI_MODELS, ge, gc):
        assert _cos(c, e) >= 1 - 1e-4, f"grad {name} cos={_cos(c, e):.6f}"
