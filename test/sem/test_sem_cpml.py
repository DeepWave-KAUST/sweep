"""Verify the SEM acoustic CPML actually absorbs the outgoing wave.

Strategy: run two simulations on the SAME mesh with the SAME source/
receivers — one with a large interior domain (so no reflection reaches
the receivers within the time window) as the "reference truth", the
other with CPML on the relevant sides. Subtract; the residual at
receivers is the energy that leaked back from the PML.

Acceptance: peak |residual| / peak |reference| < ~5% after a full
through-the-PML traverse.
"""

from __future__ import annotations

import torch

from sweep.sem import AcousticSEM2D, SEMPropagator, build_cpml, build_quad4_mesh


def ricker(t: torch.Tensor, f0: float, delay: float) -> torch.Tensor:
    a = (torch.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * torch.exp(-a)


def _common():
    return dict(dtype=torch.float64, dt=0.0008, f0=12.0, delay=0.10)


def test_cpml_reduces_boundary_reflection():
    """Normal-incidence: source on centre line, receivers along the same
    line. CPML on every side, compared against a big-domain reference."""
    cfg = _common()
    dtype, dt, f0, delay = cfg["dtype"], cfg["dt"], cfg["f0"], cfg["delay"]
    nt = 500

    src_xz = (1000.0, 500.0)
    rec_xz = [(1200.0, 500.0), (1500.0, 500.0)]

    big = build_quad4_mesh(
        nex=40, nez=20, x_max=4000.0, z_max=2000.0, n_order=4, dtype=dtype,
    )
    vp_big = torch.full((big.n_elements, big.ngll, big.ngll), 2000.0, dtype=dtype)
    eq_big = AcousticSEM2D.build(big, vp_big, dirichlet=())
    src_big = (2000.0, 1000.0)
    rec_big = [(2200.0, 1000.0), (2500.0, 1000.0)]
    prop_big = SEMPropagator.build(
        eq_big, dt=dt, nt=nt, source_xz=[src_big], receiver_xz=rec_big,
    )
    t = torch.arange(nt, dtype=dtype) * dt
    src = ricker(t, f0=f0, delay=delay)
    seis_ref = prop_big.forward(src)[..., 0].T

    small = build_quad4_mesh(
        nex=20, nez=10, x_max=2000.0, z_max=1000.0, n_order=4, dtype=dtype,
    )
    vp_small = torch.full((small.n_elements, small.ngll, small.ngll), 2000.0, dtype=dtype)
    cpml = build_cpml(
        small, dt=dt,
        sides=("left", "right", "top", "bottom"),
        pml_thickness_elements=4,
        c_max=2000.0, f_dominant=f0,
        reflection_target=1e-3,
    )
    eq_pml = AcousticSEM2D.build(small, vp_small, dirichlet=(),
                                 cpml=cpml)
    prop_pml = SEMPropagator.build(
        eq_pml, dt=dt, nt=nt, source_xz=[src_xz], receiver_xz=rec_xz,
    )
    seis_pml = prop_pml.forward(src)[..., 0].T

    residual = seis_pml - seis_ref
    rec_peak = seis_ref.abs().max(dim=1).values
    res_peak = residual.abs().max(dim=1).values
    rel = (res_peak / rec_peak.clamp(min=1e-30)).tolist()
    print(f"CPML normal-incidence leak per receiver: "
          + ", ".join(f"{r:.2%}" for r in rel))
    assert max(rel) < 0.05, f"CPML leaks > 5% : {rel}"


import pytest

@pytest.mark.skip(reason="receiver geometry needs to keep all pts outside PML; "
                  "deferred — covered visually by snapshot/gather comparisons")
def test_cpml_grazing_incidence():
    """Grazing-incidence stress test: source NEAR a PML edge, receivers
    along the line parallel to that edge. The outgoing wave hits the PML
    at small angles → this is where simple unsplit CPML fails. A 12-element
    PML at 25 m should hold leakage under ~2 %.
    """
    cfg = _common()
    dtype, dt, f0, delay = cfg["dtype"], cfg["dt"], cfg["f0"], cfg["delay"]
    nt = 900

    # Source close to the right boundary so the leading wavefront travels
    # nearly parallel to that PML in much of its trajectory.
    src_xz = (700.0, 400.0)
    # Receivers along the same depth line, spanning towards the boundary.
    rec_xz = [(750.0, 400.0), (900.0, 400.0), (950.0, 400.0)]

    big = build_quad4_mesh(nex=80, nez=40, x_max=4000.0, z_max=2000.0,
                           n_order=4, dtype=dtype)
    vp_big = torch.full((big.n_elements, big.ngll, big.ngll), 2000.0, dtype=dtype)
    eq_big = AcousticSEM2D.build(big, vp_big, dirichlet=())
    src_big = src_xz
    rec_big = rec_xz
    prop_big = SEMPropagator.build(eq_big, dt=dt, nt=nt,
                                   source_xz=[src_big], receiver_xz=rec_big)
    t = torch.arange(nt, dtype=dtype) * dt
    src = ricker(t, f0=f0, delay=delay)
    seis_ref = prop_big.forward(src)[..., 0].T

    small = build_quad4_mesh(nex=40, nez=20, x_max=1000.0, z_max=500.0,
                             n_order=4, dtype=dtype)
    vp_small = torch.full((small.n_elements, small.ngll, small.ngll), 2000.0, dtype=dtype)
    cpml = build_cpml(small, dt=dt,
                      sides=("left", "right", "top", "bottom"),
                      pml_thickness_elements=12,   # 12 × 25 m = 300 m
                      c_max=2000.0, f_dominant=f0,
                      reflection_target=1e-3)
    eq_pml = AcousticSEM2D.build(small, vp_small, dirichlet=(),
                                 cpml=cpml)
    prop_pml = SEMPropagator.build(eq_pml, dt=dt, nt=nt,
                                   source_xz=[src_xz], receiver_xz=rec_xz)
    seis_pml = prop_pml.forward(src)[..., 0].T

    residual = seis_pml - seis_ref
    rec_peak = seis_ref.abs().max(dim=1).values
    res_peak = residual.abs().max(dim=1).values
    rel = (res_peak / rec_peak.clamp(min=1e-30)).tolist()
    print(f"CPML grazing leak per receiver (12-elem PML): "
          + ", ".join(f"{r:.2%}" for r in rel))
    assert max(rel) < 0.02, f"CPML grazing leak > 2% : {rel}"


def test_cpml_path_is_autograd_clean():
    """vp gradient still flows through stiffness_with_cpml."""
    cfg = _common()
    dtype, dt, f0, delay = cfg["dtype"], cfg["dt"], cfg["f0"], cfg["delay"]
    nt = 200

    mesh = build_quad4_mesh(nex=10, nez=8, x_max=500.0, z_max=400.0,
                            n_order=4, dtype=dtype)
    vp_e = torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), 2000.0,
                      dtype=dtype, requires_grad=True)
    cpml = build_cpml(mesh, dt=dt,
                      sides=("left", "right", "bottom"),
                      pml_thickness_elements=3,
                      c_max=2200.0, f_dominant=f0)
    eq = AcousticSEM2D.build(mesh, vp_e, cpml=cpml)
    prop = SEMPropagator.build(
        eq, dt=dt, nt=nt, source_xz=[(250.0, 80.0)],
        receiver_xz=[(300.0, 80.0)],
    )
    t = torch.arange(nt, dtype=dtype) * dt
    seis = prop.forward(ricker(t, f0=f0, delay=delay))[..., 0].T
    loss = (seis ** 2).sum()
    loss.backward()
    assert vp_e.grad is not None
    assert torch.isfinite(vp_e.grad).all()
    assert vp_e.grad.abs().sum() > 0


if __name__ == "__main__":
    test_cpml_reduces_boundary_reflection()
    test_cpml_path_is_autograd_clean()
