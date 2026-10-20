"""Phase 1 grad parity: CUDA (impl='c') acoustic adjoint vs eager full-tape autograd.

For a weighted-sum loss on the seismogram, dL/dvp_e, dL/drho_e, dL/dwavelet from the
CUDA adjoint Function must match eager PyTorch autograd. The CUDA backward returns
grads w.r.t. inv_M/inv_rho/wavelet; PyTorch chains inv_M/inv_rho through build() to
vp_e/rho_e. Heterogeneous rho + topography + free-surface exercise every grad path.
fp32 bar: cos >= 0.999.
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

NEX, NEZ, XMAX, ZMAX = 20, 14, 1000.0, 700.0
DT, NT, F0, TSH = 8e-4, 160, 14.0, 0.09
VP0, RHO0 = 2000.0, 1000.0
SRC = (XMAX * 0.44, 110.0)
REC = [(float(x), 55.0) for x in np.linspace(130.0, XMAX - 130.0, 14)]


def _hill(x):
    return 50.0 * (1.0 - torch.exp(-((x - XMAX / 2) ** 2) / (2 * 130.0 ** 2)))


def _mesh(kind):
    if kind == "flat":
        return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                        n_order=4, dtype=torch.float32)
    return build_topography_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_bottom=ZMAX,
                                  z_top_fn=_hill, n_order=4, dtype=torch.float32)


def _wavelet():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def _grads(kind, fs, use_cuda):
    mesh = _mesh(kind)
    shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp = torch.full(shp, VP0, dtype=torch.float32).requires_grad_(True)
    rho = (RHO0 * (1.0 + 0.4 * (mesh.z_e / ZMAX))).to(torch.float32).requires_grad_(True)
    wav = _wavelet().requires_grad_(True)
    eq = AcousticPotentialSEM2D.build(mesh, vp, rho, dirichlet=("top",) if fs else ())
    prop = (SEMCudaPropagator if use_cuda else SEMPropagator).build(
        eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC)
    seis = prop.forward(wav)[..., 0].T
    W = torch.linspace(0.3, 1.7, seis.numel(), device=seis.device).reshape(seis.shape)
    (W * seis).sum().backward()
    return vp.grad.cpu(), rho.grad.cpu(), wav.grad.cpu()


@pytest.mark.parametrize("mesh_kind", ["flat", "topo"])
@pytest.mark.parametrize("fs", [True, False])
def test_acoustic_grads_match_eager(mesh_kind, fs):
    gvp_e, grho_e, gwav_e = _grads(mesh_kind, fs, use_cuda=False)
    gvp_c, grho_c, gwav_c = _grads(mesh_kind, fs, use_cuda=True)
    assert _cos(gvp_c, gvp_e) >= 0.999, f"grad vp cos={_cos(gvp_c, gvp_e):.6f}"
    assert _cos(grho_c, grho_e) >= 0.999, f"grad rho cos={_cos(grho_c, grho_e):.6f}"
    assert _cos(gwav_c, gwav_e) >= 0.999, f"grad wav cos={_cos(gwav_c, gwav_e):.6f}"
