"""SPECFEM2D-style C-PML for the 2nd-order *displacement* elastic SEM (Quad9).

The 2nd-order displacement elastic CPML is notoriously prone to a late-time
instability unless the acceleration/mass damping term is included (see
``sweep.sem.cpml.elastic``).  These tests gate three things:

1. **Stability + absorption** on a flat mesh: a long run stays finite and the
   wavefield energy decays to ~0 (a reflecting boundary would keep energy
   bouncing).  The natural free top + 3-side CPML means the run also exercises
   the free-surface/CPML corners, the classic failure mode.
2. **Stability** on a genuine topography (Gaussian-hill) mesh.
3. **Autograd**: the model gradient flows through the CPML recursion and matches
   a finite-difference reference (FWI-ready).
"""
import math

import pytest
import torch

from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.equations.elastic2d_deformed import ElasticSEM2DDeformed
from sweep.sem.propagator import SEMPropagator

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _ricker(nt, dt, f0=12.0, t0=0.09, scale=1e6, dtype=torch.float64):
    t = torch.arange(nt, device=DEV, dtype=dtype) * dt - t0
    a = (math.pi * f0 * t) ** 2
    return (1 - 2 * a) * torch.exp(-a) * scale


def _full(mesh, v, dtype=torch.float32):
    return torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), float(v),
                      dtype=dtype, device=DEV)


def _energy_decay(eq, mesh, dt, nt, dtype):
    """Run a point (moment-tensor) source and return (peak, late, finite)."""
    sp = SEMPropagator.build(
        eq, dt=dt, nt=1, source_xz=[(mesh.x_max / 2, mesh.z_max * 0.45)],
        receiver_xz=[(mesh.x_max * 0.3, 30.0)],
        source_kinds=["moment_tensor"], source_payloads=[(1.0, 1.0, 0.0)],
    )._build_source_patterns(mesh.n_global, DEV, dtype)
    spx, spz = sp[0], sp[1]              # source fields u_x, u_z, in order
    wav = _ricker(nt, dt, dtype=dtype)
    st = eq.init_state(dt, DEV, dtype)
    peak, late = 0.0, 0.0
    with torch.inference_mode():         # cheaper per step than no_grad
        for n in range(nt):
            st = eq.step(st, [spx * wav[n], spz * wav[n]], dt)
            late = float((st[0] ** 2 + st[1] ** 2).sum())
            peak = max(peak, late)
            if not math.isfinite(late):
                break
    return peak, late, math.isfinite(late)


def test_elastic_cpml_flat_stable_and_absorbs():
    nex, nez, no, dt, nt = 30, 22, 4, 1.0e-3, 6000
    mesh = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=nex * no * 10.0,
                                    z_max=nez * no * 10.0, n_order=no,
                                    dtype=torch.float32, device=DEV)
    cpml = build_elastic_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                              pml_thickness_elements=8, c_max=2600., f_dominant=12.)
    eq = ElasticSEM2DDeformed.build(mesh, _full(mesh, 2400.), _full(mesh, 1300.),
                                    _full(mesh, 2000.), cpml=cpml)
    peak, late, finite = _energy_decay(eq, mesh, dt, nt, torch.float32)
    assert finite, "elastic CPML blew up (non-finite energy)"
    assert late / (peak + 1e-30) < 1e-4, f"poor absorption: late/peak={late/peak:.2e}"


@pytest.mark.skipif(DEV != "cuda", reason="long-nt run; CUDA recommended")
def test_elastic_cpml_topography_stable():
    nex, nez, no, dt, nt = 30, 22, 4, 1.0e-3, 5000
    xm, zb = nex * no * 10.0, nez * no * 10.0
    hill = lambda x: -60.0 * torch.exp(-((x - xm / 2) / 180.0) ** 2)
    mesh = build_topography_quad9(nex, nez, xm, zb, hill, n_order=no,
                                  dtype=torch.float32, device=DEV)
    cpml = build_elastic_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                              pml_thickness_elements=8, c_max=2600., f_dominant=12.)
    eq = ElasticSEM2DDeformed.build(mesh, _full(mesh, 2400.), _full(mesh, 1300.),
                                    _full(mesh, 2000.), cpml=cpml)
    peak, late, finite = _energy_decay(eq, mesh, dt, nt, torch.float32)
    assert finite, "elastic CPML blew up on topography"
    assert late / (peak + 1e-30) < 1e-3, f"poor absorption: late/peak={late/peak:.2e}"


def test_elastic_cpml_autograd_matches_fd():
    nex, nez, no, dt, nt = 20, 16, 4, 1.0e-3, 400
    xm, zm = nex * no * 10.0, nez * no * 10.0
    mesh = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=xm, z_max=zm,
                                    n_order=no, dtype=torch.float64, device=DEV)
    vs0, rho0 = _full(mesh, 1300., torch.float64), _full(mesh, 2000., torch.float64)
    wav = _ricker(nt, dt, t0=0.08, dtype=torch.float64)

    def loss_of(vp_e):
        cpml = build_elastic_cpml(mesh, dt=dt, sides=("left", "right", "bottom"),
                                  pml_thickness_elements=6, c_max=2600., f_dominant=12.)
        eq = ElasticSEM2DDeformed.build(mesh, vp_e, vs0, rho0, cpml=cpml)
        prop = SEMPropagator.build(
            eq, dt=dt, nt=nt, source_xz=[(xm / 2, zm * 0.45)],
            receiver_xz=[(xm * 0.35, 40.0), (xm * 0.65, 40.0)],
            source_kinds=["moment_tensor"], source_payloads=[(1.0, 1.0, 0.0)])
        o = prop.forward(wav)
        return (o[..., 0].T ** 2 + o[..., 1].T ** 2).sum()

    vp = _full(mesh, 2400., torch.float64).requires_grad_(True)
    loss = loss_of(vp)
    loss.backward()
    g = vp.grad
    assert torch.isfinite(g).all() and g.abs().max() > 0

    idx = (mesh.n_elements // 2, 2, 2)
    eps = 1e-2
    vp_p = _full(mesh, 2400., torch.float64); vp_p[idx] += eps
    vp_m = _full(mesh, 2400., torch.float64); vp_m[idx] -= eps
    with torch.no_grad():
        fd = float((loss_of(vp_p) - loss_of(vp_m)) / (2 * eps))
    ad = float(g[idx])
    assert abs(ad - fd) / (abs(fd) + 1e-30) < 5e-2, f"grad FD mismatch: {ad:.3e} vs {fd:.3e}"
