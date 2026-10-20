"""P5 parity: CUDA (impl='c') 3-D acoustic forward + adjoint vs eager
(AcousticSEM3D, axis-aligned Hex8, velocity form).

Forward gathers and dL/dvp vs eager full-tape autograd: cos >= 1-1e-4.
Heterogeneous vp, free surface (top dirichlet) on/off, interior receivers.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.cpml.acoustic3d import build_cpml3d
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEY, NEZ = 8, 6, 7
XMAX, YMAX, ZMAX = 800.0, 600.0, 700.0
DT, NT, F0, TSH = 5e-4, 200, 14.0, 0.08
SRC = (XMAX * 0.5, YMAX * 0.45, 240.0)
REC = [(float(x), YMAX * 0.55, 150.0)
       for x in np.linspace(120.0, XMAX - 120.0, 12)]


def _eq(dirichlet, grad=False, cpml=False):
    m = build_hex8_mesh(NEX, NEY, NEZ, XMAX, YMAX, ZMAX, n_order=4,
                        dtype=torch.float32)
    vp = (2000.0 + 500.0 * (m.z_e / ZMAX) + 120.0 * torch.sin(m.x_e / 90.0)
          ).to(torch.float32).contiguous()
    if grad:
        vp = vp.requires_grad_(True)
    kw = {}
    if cpml:
        kw["cpml"] = build_cpml3d(m, dt=DT, sides=("left", "right", "front",
                                                   "back", "bottom"),
                                  pml_thickness_elements=2, c_max=2700.0,
                                  f_dominant=F0)
    eq = AcousticSEM3D.build(m, vp, dirichlet=dirichlet, **kw)
    return eq, vp


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _prop(eq, use_cuda):
    return (SEMCudaPropagator if use_cuda else SEMPropagator).build(
        eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC)


@pytest.mark.parametrize("dirichlet", [(), ("top",)], ids=["neumann", "fs"])
def test_acoustic3d_forward_matches_eager(dirichlet):
    eq, _ = _eq(dirichlet)
    oe = _prop(eq, False).forward(_wav())[..., 0].T
    oc = _prop(eq, True).forward(_wav())[..., 0].T
    c = _cos(oc.cpu(), oe.cpu())
    assert c >= 1 - 1e-4, f"forward cos={c:.6f}"


@pytest.mark.parametrize("dirichlet", [(), ("top",)], ids=["neumann", "fs"])
def test_acoustic3d_grads_match_eager(dirichlet):
    def run(uc):
        eq, vp = _eq(dirichlet, grad=True)
        o = _prop(eq, uc).forward(_wav())
        s = o[..., 0].T
        W = torch.linspace(0.4, 1.6, s.numel(), device=s.device).reshape(s.shape)
        (W * s).sum().backward()
        return vp.grad.cpu()
    ge = run(False)
    gc = run(True)
    c = _cos(gc, ge)
    assert c >= 1 - 1e-4, f"grad vp cos={c:.6f}"


@pytest.mark.parametrize("dirichlet", [(), ("top",)], ids=["neumann", "fs"])
def test_acoustic3d_cpml_forward_matches_eager(dirichlet):
    eq, _ = _eq(dirichlet, cpml=True)
    oe = _prop(eq, False).forward(_wav())[..., 0].T
    oc = _prop(eq, True).forward(_wav())[..., 0].T
    c = _cos(oc.cpu(), oe.cpu())
    assert c >= 1 - 1e-4, f"cpml forward cos={c:.6f}"


@pytest.mark.parametrize("dirichlet", [(), ("top",)], ids=["neumann", "fs"])
def test_acoustic3d_cpml_grads_match_eager(dirichlet):
    def run(uc):
        eq, vp = _eq(dirichlet, grad=True, cpml=True)
        o = _prop(eq, uc).forward(_wav())
        s = o[..., 0].T
        W = torch.linspace(0.4, 1.6, s.numel(), device=s.device).reshape(s.shape)
        (W * s).sum().backward()
        return vp.grad.cpu()
    ge = run(False)
    gc = run(True)
    c = _cos(gc, ge)
    assert c >= 1 - 1e-4, f"cpml grad vp cos={c:.6f}"
