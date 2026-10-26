"""P8 parity: CUDA (impl='c') 3-D velocity-stress elastic (ElasticVSSEM3D)
vs eager, bare + simple per-axis CPML.

seis_vx/vy/vz and dL/d{vp,vs,rho} vs eager full-tape autograd: cos >= 1-1e-4.
Force source scaled up (1e6) to lift the gradients off the fp32 noise floor.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.equations.elastic_vs3d import ElasticVSSEM3D
from sweep.sem.cpml.elastic_vs3d import build_simple_elastic_cpml_3d
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEY, NEZ = 6, 5, 6
XMAX, YMAX, ZMAX = 600.0, 500.0, 600.0
DT, NT, F0, TSH = 4e-4, 160, 16.0, 0.07
SRC = (XMAX * 0.5, YMAX * 0.45, 220.0)
REC = [(float(x), YMAX * 0.55, 150.0)
       for x in np.linspace(120.0, XMAX - 120.0, 8)]
PAYLOAD = (1.0e6, 0.7e6, 0.5e6)
SIDES = ("left", "right", "front", "back", "bottom")


def _eq(cpml, grad=False):
    m = build_hex8_mesh(NEX, NEY, NEZ, XMAX, YMAX, ZMAX, n_order=4,
                        dtype=torch.float32)
    vp = (2400.0 + 300.0 * (m.z_e / ZMAX)).to(torch.float32).contiguous()
    vs = (1200.0 + 200.0 * (m.x_e / XMAX)).to(torch.float32).contiguous()
    rho = (2000.0 + 150.0 * (m.y_e / YMAX)).to(torch.float32).contiguous()
    if grad:
        vp, vs, rho = (vp.requires_grad_(True), vs.requires_grad_(True),
                       rho.requires_grad_(True))
    kw = {}
    if cpml:
        kw["cpml"] = build_simple_elastic_cpml_3d(
            m, dt=DT, sides=SIDES, pml_thickness_elements=2,
            c_max=2700.0, f_dominant=F0)
    return ElasticVSSEM3D.build(m, vp, vs, rho, **kw), (vp, vs, rho)


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
        eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC,
        source_kinds=["force"], source_payloads=[PAYLOAD], pml_guard="off")


@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
def test_elastic_vs3d_forward_matches_eager(cpml):
    eq, _ = _eq(cpml)
    oe = _prop(eq, False).forward(_wav())
    oc = _prop(eq, True).forward(_wav())
    for j, nm in enumerate(('v_x', 'v_y', 'v_z')):  # component order
        c = _cos(oc[..., j].T.cpu(), oe[..., j].T.cpu())
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"


@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
def test_elastic_vs3d_grads_match_eager(cpml):
    def run(uc):
        eq, (vp, vs, rho) = _eq(cpml, grad=True)
        o = _prop(eq, uc).forward(_wav())
        loss = 0.0
        for j in range(o.shape[-1]):             # every component
            s = o[..., j].T
            W = torch.linspace(0.4 + 0.1 * j, 1.6 - 0.1 * j, s.numel(),
                               device=s.device).reshape(s.shape)
            loss = loss + (W * s).sum()
        loss.backward()
        return vp.grad.cpu(), vs.grad.cpu(), rho.grad.cpu()
    ge = run(False)
    gc = run(True)
    for name, e, c in zip(("vp", "vs", "rho"), ge, gc):
        assert _cos(c, e) >= 1 - 1e-4, f"grad {name} cos={_cos(c, e):.6f}"
