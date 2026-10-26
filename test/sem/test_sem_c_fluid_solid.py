"""P4 parity: CUDA (impl='c') coupled fluid-solid forward vs eager
(AcousticElasticSEM2D), bare + dual-CPML, marine geometry (water over solid,
free surface top, near-interface pressure source, receivers in both domains).

seis_chi / seis_x / seis_z: cos >= 1-1e-4 vs eager (same eq object through
SEMPropagator vs SEMCudaPropagator).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9
from sweep.sem.equations.acoustic_elastic2d import AcousticElasticSEM2D
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NEX, NEZ, DE = 20, 16, 50.0
XMAX, ZMAX, Z_IF = NEX * DE, NEZ * DE, 8 * DE     # interface at 400 m
DT, NT, F0, TSH = 6e-4, 260, 12.0, 0.10
SRC = (XMAX * 0.45, 350.0)                        # water, near the interface
REC = ([(float(x), 200.0) for x in np.linspace(300.0, XMAX - 300.0, 6)]     # fluid
       + [(float(x), 600.0) for x in np.linspace(300.0, XMAX - 300.0, 6)])  # solid
SIDES = ("left", "right", "bottom")
NPML = 4


def _eq(cpml, grad=False):
    m = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XMAX, z_max=ZMAX,
                                 n_order=4, dtype=torch.float32)
    shp = (m.n_elements, m.ngll, m.ngll)
    fl = (m.z_e.flatten(1).mean(1) < Z_IF).reshape(-1, 1, 1).expand(shp)
    vp = torch.where(fl, 1500.0, 3000.0).float().contiguous()
    vs = torch.where(fl, 0.0, 1732.0).float().contiguous()
    rho = torch.where(fl, 1000.0, 2200.0).float().contiguous()
    if grad:
        vp = vp.requires_grad_(True)
        vs = vs.requires_grad_(True)
        rho = rho.requires_grad_(True)
    kw = {}
    if cpml:
        kw["cpml_ac"] = build_cpml(m, dt=DT, sides=SIDES,
                                   pml_thickness_elements=NPML,
                                   c_max=1500.0, f_dominant=F0)
        kw["cpml_el"] = build_elastic_cpml(m, dt=DT, sides=SIDES,
                                           pml_thickness_elements=NPML,
                                           c_max=3000.0, f_dominant=F0)
    eq = AcousticElasticSEM2D.build(m, vp, vs, rho, dirichlet=("top",), **kw)
    return (eq, (vp, vs, rho)) if grad else eq


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - TSH)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)


def _cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _run(cpml, use_cuda):
    eq = _eq(cpml)
    prop = (SEMCudaPropagator if use_cuda else SEMPropagator).build(
        eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC,
        source_kinds=["scalar"], source_payloads=[(1.0,)],
        pml_guard="off")
    return prop.forward(_wav())


@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
def test_fluid_solid_forward_matches_eager(cpml):
    oe = _run(cpml, False)
    oc = _run(cpml, True)
    for j, nm in enumerate(("chi", "u_x", "u_z")):   # component order
        c = _cos(oc[..., j].T.cpu(), oe[..., j].T.cpu())
        assert c >= 1 - 1e-4, f"{nm} cos={c:.6f}"


@pytest.mark.parametrize("cpml", [False, True], ids=["bare", "cpml"])
def test_fluid_solid_grads_match_eager(cpml):
    def run(use_cuda):
        eq, (vp, vs, rho) = _eq(cpml, grad=True)
        prop = (SEMCudaPropagator if use_cuda else SEMPropagator).build(
            eq, dt=DT, nt=NT, source_xz=[SRC], receiver_xz=REC,
            source_kinds=["scalar"], source_payloads=[(1e6,)],
            pml_guard="off")
        o = prop.forward(_wav())
        loss = 0.0
        for j in range(o.shape[-1]):                  # chi, u_x, u_z
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
