"""P3a: acoustic-potential C-PML forward parity — CUDA (impl='c') vs eager.

The Komatitsch-Martin double-convolution (psi/zeta memory) forward must match the
eager AcousticPotentialSEM2D CPML step to cos >= 1-1e-6, on flat + topography.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEZ, XMAX, ZMAX = 30, 22, 1400.0, 1000.0
DT, NT, F0, TSH = 8e-4, 400, 12.0, 0.10
VP, RHO = 2200.0, 1000.0
SRC = (XMAX * 0.5, 200.0)
REC = [(float(x), 60.0) for x in np.linspace(200.0, XMAX - 200.0, 24)]


def _hill(x):
    return 40.0 * (1.0 - torch.exp(-((x - XMAX / 2) ** 2) / (2 * 180.0 ** 2)))


def _mesh(kind):
    if kind == "flat":
        return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX, n_order=4, dtype=torch.float32)
    return build_topography_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_bottom=ZMAX, z_top_fn=_hill, n_order=4, dtype=torch.float32)


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _build(mesh, grad):
    shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp = torch.full(shp, VP, dtype=torch.float32)
    rho = (RHO * (1.0 + 0.3 * (mesh.z_e / ZMAX))).to(torch.float32) if grad \
        else torch.full(shp, RHO, dtype=torch.float32)
    if grad:
        vp, rho = vp.requires_grad_(True), rho.requires_grad_(True)
    cp = build_cpml(mesh, dt=DT, sides=("left", "right", "bottom"),
                    pml_thickness_elements=8, c_max=VP, f_dominant=F0)
    eq = AcousticPotentialSEM2D.build(mesh, vp, rho, dirichlet=("top",), cpml=cp)
    return eq, vp, rho


@pytest.mark.parametrize("kind", ["flat", "topo"])
def test_acoustic_cpml_forward_matches_eager(kind):
    eq, _, _ = _build(_mesh(kind), grad=False)
    kw = dict(dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC)
    e = SEMPropagator.build(eq, **kw).forward(_wav())[..., 0].T.cpu()
    c = SEMCudaPropagator.build(eq, **kw).forward(_wav())[..., 0].T.cpu()
    assert _cos(c, e) >= 1 - 1e-6, f"cos={_cos(c, e):.8f}"


@pytest.mark.parametrize("kind", ["flat", "topo"])
def test_acoustic_cpml_grads_match_eager(kind):
    def run(uc):
        eq, vp, rho = _build(_mesh(kind), grad=True)
        prop = (SEMCudaPropagator if uc else SEMPropagator).build(
            eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC)
        seis = prop.forward(_wav())[..., 0].T
        W = torch.linspace(0.3, 1.7, seis.numel(), device=seis.device).reshape(seis.shape)
        (W * seis).sum().backward()
        return vp.grad.cpu(), rho.grad.cpu()
    ve, re = run(False)
    vc, rc = run(True)
    assert _cos(vc, ve) >= 1 - 1e-4, f"grad vp cos={_cos(vc, ve):.6f}"
    assert _cos(rc, re) >= 1 - 1e-4, f"grad rho cos={_cos(rc, re):.6f}"
