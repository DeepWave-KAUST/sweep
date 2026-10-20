"""Phase 0 parity: sweep.sem CUDA (impl='c') acoustic-potential forward vs eager.

Bar (T2, per the fp32-only / cos~1e-6 project decision): CUDA-fp32 vs eager cast to
fp32, cos >= 1-1e-6 and max_rel <= 1e-5, across the acoustic PARITY MATRIX
{flat, topography} x {homogeneous, heterogeneous-rho} x {free-surface on, off}. The
topography + asymmetric (nex != nez) + off-centre-source cases would fail a silent
(eta,xi) axis transpose in the ported stiffness kernel.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEZ, XMAX, ZMAX = 24, 16, 1200.0, 800.0     # nex != nez -> asymmetric
DT, NT, F0, TSH = 8e-4, 250, 14.0, 0.09
VP0, RHO0 = 2000.0, 1000.0
SRC = (XMAX * 0.42, 120.0)                        # off-centre -> breaks x<->z symmetry
REC = [(float(x), 60.0) for x in np.linspace(150.0, XMAX - 150.0, 16)]


def _hill(x):
    return 60.0 * (1.0 - torch.exp(-((x - XMAX / 2) ** 2) / (2 * 150.0 ** 2)))


def _mesh(kind):
    if kind == "flat":
        return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                        n_order=4, dtype=torch.float32)
    return build_topography_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_bottom=ZMAX,
                                  z_top_fn=_hill, n_order=4, dtype=torch.float32)


def _models(mesh, hetero):
    shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp = torch.full(shp, VP0, dtype=torch.float32)
    rho = (torch.full(shp, RHO0, dtype=torch.float32) if not hetero
           else (RHO0 * (1.0 + 0.4 * (mesh.z_e / ZMAX))).to(torch.float32))
    return vp, rho


def _wavelet():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


@pytest.mark.parametrize("mesh_kind", ["flat", "topo"])
@pytest.mark.parametrize("hetero", [False, True])
@pytest.mark.parametrize("fs", [True, False])
def test_acoustic_forward_matches_eager(mesh_kind, hetero, fs):
    mesh = _mesh(mesh_kind)
    vp, rho = _models(mesh, hetero)
    eq = AcousticPotentialSEM2D.build(mesh, vp, rho,
                                      dirichlet=("top",) if fs else ())
    kw = dict(dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC)
    wav = _wavelet()

    seis_e = SEMPropagator.build(eq, **kw).forward(wav)[..., 0].T.reshape(-1).double()
    seis_c = SEMCudaPropagator.build(eq, **kw).forward(wav)[..., 0].T.cpu().reshape(-1).double()

    cos = float((seis_c @ seis_e) / (seis_c.norm() * seis_e.norm() + 1e-30))
    rel = float((seis_c - seis_e).norm() / (seis_e.norm() + 1e-30))
    assert cos >= 1 - 1e-6, f"cos={cos:.8f} rel={rel:.2e}"
    assert rel <= 1e-5, f"rel={rel:.2e} cos={cos:.8f}"
