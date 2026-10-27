"""3-D acoustic on a DEFORMED Hex27 mesh with TOPOGRAPHY, impl='c' vs eager.

AcousticSEM3DDeformed (full per-node 3x3 inv_J chain rule) on a curved-top
mesh: the CUDA deformed-metric kernels must match the eager equation on both
the forward gathers and dL/dvp with 3-direction C-PML, and the CUDA
boundary-saving path must equal the CUDA save-all path. The bare (reflecting
box) kernel solves div(c² grad u), not the eager c² Laplace u, and is refused.

All comparisons: cos >= 1-1e-4.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.hex27 import build_topography_hex27
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
from sweep.sem.cpml.acoustic3d import build_cpml3d
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEY, NEZ = 8, 6, 7
XMAX, YMAX, ZMAX = 800.0, 600.0, 700.0
DT, NT, F0, TSH = 5e-4, 200, 14.0, 0.08
SRC = (XMAX * 0.5, YMAX * 0.45, 300.0)
REC = [(float(x), YMAX * 0.55, 220.0)
       for x in np.linspace(230.0, XMAX - 230.0, 12)]


def _z_top(x, y):
    # a smooth hill: up to ~70 m of relief on the top face (SEM down-positive)
    return 70.0 * torch.exp(-(((x - XMAX * 0.5) / 180.0) ** 2
                              + ((y - YMAX * 0.5) / 150.0) ** 2))


def _mesh():
    return build_topography_hex27(NEX, NEY, NEZ, XMAX, YMAX, ZMAX, _z_top,
                                  n_order=4, dtype=torch.float32)


def _eq(dirichlet, grad=False, cpml=False):
    m = _mesh()
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
    eq = AcousticSEM3DDeformed.build(m, vp, dirichlet=dirichlet, **kw)
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


def _grad(eq, vp, use_cuda, mem=None):
    o = _prop(eq, use_cuda).forward(_wav(), memory=mem)
    s = o[..., 0].T
    W = torch.linspace(0.4, 1.6, s.numel(), device=s.device).reshape(s.shape)
    (W * s).sum().backward()
    return vp.grad.cpu()


def test_deformed3d_bare_is_refused():
    eq, _ = _eq(("top",))
    with pytest.raises(NotImplementedError, match="without a C-PML"):
        _prop(eq, True).forward(_wav())


@pytest.mark.parametrize("dirichlet", [(), ("top",)], ids=["neumann", "fs"])
def test_deformed3d_forward_matches_eager(dirichlet):
    eq, _ = _eq(dirichlet, cpml=True)
    oe = _prop(eq, False).forward(_wav())[..., 0].T
    oc = _prop(eq, True).forward(_wav())[..., 0].T
    c = _cos(oc.cpu(), oe.cpu())
    assert c >= 1 - 1e-4, f"forward cos={c:.6f}"


@pytest.mark.parametrize("dirichlet", [(), ("top",)], ids=["neumann", "fs"])
def test_deformed3d_grads_match_eager(dirichlet):
    ge = _grad(*_eq(dirichlet, grad=True, cpml=True), False)
    gc = _grad(*_eq(dirichlet, grad=True, cpml=True), True)
    c = _cos(gc, ge)
    assert c >= 1 - 1e-4, f"grad vp cos={c:.6f}"


def test_deformed3d_boundary_saving_matches_full():
    """CUDA boundary saving vs CUDA save-all AND vs eager full tape (fs on)."""
    gc_full = _grad(*_eq(("top",), grad=True, cpml=True), True, mem=None)
    gc_bs = _grad(*_eq(("top",), grad=True, cpml=True), True, mem="boundary")
    ge = _grad(*_eq(("top",), grad=True, cpml=True), False)
    assert _cos(gc_bs, gc_full) >= 1 - 1e-4, \
        f"c-BS vs c-full cos={_cos(gc_bs, gc_full):.6f}"
    assert _cos(gc_bs, ge) >= 1 - 1e-4, f"c-BS vs eager cos={_cos(gc_bs, ge):.6f}"
