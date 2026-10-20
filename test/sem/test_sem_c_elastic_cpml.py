"""P3e parity: CUDA (impl='c') elastic C-PML forward + adjoint vs eager
(ElasticSEM2DDeformed + build_elastic_cpml).

Forward gathers (vx,vz) and gradients dL/dvp, dL/dvs, dL/drho vs eager full-tape
autograd: cos >= 1-1e-4. Moment-tensor source, heterogeneous vs/rho, flat/topo.
Source amplitude scaled up (1e6) to lift the gradients off the fp32 noise floor
(see test_sem_c_elastic.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9
from sweep.sem.equations.elastic2d_deformed import ElasticSEM2DDeformed
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEZ, XMAX, ZMAX = 22, 16, 1100.0, 780.0
DT, NT, F0, TSH = 6e-4, 220, 14.0, 0.10
VP0, VS0, RHO0 = 2400.0, 1200.0, 2000.0
SRC = (XMAX * 0.46, 130.0)
REC = [(float(x), 60.0) for x in np.linspace(300.0, XMAX - 300.0, 16)]
MT = [(1e6, 1e6, 0.0)]
SIDES = ("left", "right", "bottom")
NPML = 5


def _hill(x):
    return 55.0 * (1.0 - torch.exp(-((x - XMAX / 2) ** 2) / (2 * 140.0 ** 2)))


def _mesh(kind):
    if kind == "flat":
        return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX, n_order=4, dtype=torch.float32)
    return build_topography_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_bottom=ZMAX, z_top_fn=_hill, n_order=4, dtype=torch.float32)


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _models(mesh, grad):
    shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
    vp = torch.full(shp, VP0, dtype=torch.float32)
    vs = (VS0 * (1.0 + 0.3 * (mesh.z_e / ZMAX))).to(torch.float32)
    rho = (RHO0 * (1.0 + 0.2 * (mesh.z_e / ZMAX))).to(torch.float32)
    if grad:
        vp, vs, rho = vp.requires_grad_(True), vs.requires_grad_(True), rho.requires_grad_(True)
    return vp, vs, rho


def _prop(kind, use_cuda, grad):
    mesh = _mesh(kind)
    vp, vs, rho = _models(mesh, grad)
    cpml = build_elastic_cpml(mesh, dt=DT, sides=SIDES,
                              pml_thickness_elements=NPML, c_max=VP0 * 1.1,
                              f_dominant=F0)
    eq = ElasticSEM2DDeformed.build(mesh, vp, vs, rho, cpml=cpml)
    prop = (SEMCudaPropagator if use_cuda else SEMPropagator).build(
        eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC,
        source_kinds=["moment_tensor"], source_payloads=MT)
    return prop, (vp, vs, rho)


@pytest.mark.parametrize("kind", ["flat", "topo"])
def test_elastic_cpml_forward_matches_eager(kind):
    pe, _ = _prop(kind, False, False); oe = pe.forward(_wav())
    pc, _ = _prop(kind, True, False); oc = pc.forward(_wav())
    assert _cos(oc[..., 0].T.cpu(), oe[..., 0].T.cpu()) >= 1 - 1e-4, \
        f"seis_x cos={_cos(oc['seis_x'].cpu(), oe['seis_x'].cpu()):.6f}"
    assert _cos(oc[..., 1].T.cpu(), oe[..., 1].T.cpu()) >= 1 - 1e-4, \
        f"seis_z cos={_cos(oc['seis_z'].cpu(), oe['seis_z'].cpu()):.6f}"


@pytest.mark.parametrize("kind", ["flat", "topo"])
def test_elastic_cpml_grads_match_eager(kind):
    def run(uc):
        prop, (vp, vs, rho) = _prop(kind, uc, True)
        o = prop.forward(_wav())
        sx, sz = o[..., 0].T, o[..., 1].T
        Wx = torch.linspace(0.4, 1.6, sx.numel(), device=sx.device).reshape(sx.shape)
        Wz = torch.linspace(0.6, 1.4, sz.numel(), device=sz.device).reshape(sz.shape)
        ((Wx * sx).sum() + (Wz * sz).sum()).backward()
        return vp.grad.cpu(), vs.grad.cpu(), rho.grad.cpu()
    ge = run(False)
    gc = run(True)
    for name, e, c in zip(("vp", "vs", "rho"), ge, gc):
        assert _cos(c, e) >= 1 - 1e-4, f"grad {name} cos={_cos(c, e):.6f}"
