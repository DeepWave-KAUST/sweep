"""CPML for the deformed (Quad9) acoustic equation.

Two checks:
1. Degeneracy — on a flat axis-aligned Quad9 mesh, ``AcousticSEM2DDeformed``
   with CPML must reproduce ``AcousticSEM2D`` (Quad4) with CPML to machine
   precision (validates the inv_J directional operators reduce to the
   axis-aligned ones in the absorbing band).
2. Stability/absorption — on a genuine topography (Gaussian-hill) mesh, a long
   run stays finite and the propagating-wave energy is absorbed to ~0.
"""
import math

import pytest
import torch

from sweep.sem.mesh.quad4 import build_quad4_mesh
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.equations.acoustic2d import AcousticSEM2D
from sweep.sem.equations.acoustic2d_deformed import AcousticSEM2DDeformed
from sweep.sem.propagator import SEMPropagator

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ricker(nt, dt, f0=12.0, t0=0.09, dtype=torch.float64):
    t = torch.arange(nt, device=DEV, dtype=dtype) * dt - t0
    a = (math.pi * f0 * t) ** 2
    return (1 - 2 * a) * torch.exp(-a)


def test_deformed_cpml_degenerates_to_quad4():
    nex, nez, no, dt, nt = 24, 18, 4, 1.2e-3, 400
    xm, zm = nex * no * 10.0, nez * no * 10.0
    wav = _ricker(nt, dt)
    src = [(xm / 2, zm * 0.5)]
    rec = [(xm * 0.3, 40.0), (xm * 0.7, 40.0)]

    def run(mesh, eq_cls):
        vp = 1800 + 600 * mesh.z_e / zm
        cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                          pml_thickness_elements=6, c_max=2400., f_dominant=12.)
        eq = eq_cls.build(mesh, vp, dirichlet=("top",), cpml=cpml)
        prop = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=src, receiver_xz=rec)
        return prop.forward(wav)[..., 0].T

    m4 = build_quad4_mesh(nex=nex, nez=nez, x_max=xm, z_max=zm, n_order=no,
                          dtype=torch.float64, device=DEV)
    m9 = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=xm, z_max=zm,
                                  n_order=no, dtype=torch.float64, device=DEV)
    s4 = run(m4, AcousticSEM2D)
    s9 = run(m9, AcousticSEM2DDeformed)
    rel = (s9 - s4).norm() / s4.norm()
    assert rel < 1e-10, rel.item()


@pytest.mark.skipif(DEV != "cuda", reason="long-nt run; CUDA recommended")
def test_deformed_cpml_topography_stable():
    nex, nez, no, dt, nt = 30, 22, 4, 1.0e-3, 3000
    xm, zb = nex * no * 10.0, nez * no * 10.0
    hill = lambda x: -60.0 * torch.exp(-((x - xm / 2) / 180.0) ** 2)
    mesh = build_topography_quad9(nex, nez, xm, zb, hill, n_order=no,
                                  dtype=torch.float32, device=DEV)
    vp = 1800 + 600 * mesh.z_e / zb
    cpml = build_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                      pml_thickness_elements=8, c_max=2400., f_dominant=12.)
    eq = AcousticSEM2DDeformed.build(mesh, vp, dirichlet=("top",), cpml=cpml)
    wav = _ricker(nt, dt, dtype=torch.float32)
    sp = SEMPropagator.build(eq, dt=dt, nt=1, source_xz=[(xm / 2, zb * 0.5)],
                             receiver_xz=[(xm * 0.3, 30.0)]
                             )._build_source_patterns(mesh.n_global, DEV, torch.float32)[0]
    st = eq.init_state(dt, DEV, torch.float32)
    peak = 0.0
    with torch.no_grad():
        for n in range(nt):
            st = eq.step(st, [sp * wav[n]], dt)        # positional src
            u = st[0]                                  # field 0 = u
            e = float(((u - u.mean()) ** 2).sum())     # propagating-wave energy
            peak = max(peak, e)
        late = e
    assert math.isfinite(late)
    assert late / (peak + 1e-30) < 1e-3                # wave absorbed, no blow-up
