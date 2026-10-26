"""VTI (Thomsen) velocity-stress SEM: degeneracy + stability + autograd.

The headline correctness check is that ``epsilon = delta = 0`` reduces the VTI
constitutive to isotropy, so ``ElasticVTISEM2DDeformed`` must reproduce
``ElasticVSSEM2DDeformed`` to machine precision. Then a genuinely anisotropic
run must stay finite and yield a finite, nonzero model gradient (autograd).
"""
import math

import pytest
import torch

from sweep.sem import (
    ElasticVSSEM2DDeformed,
    ElasticVTISEM2DDeformed,
    SEMPropagator,
    build_axis_aligned_quad9,
    thomsen_to_vti_cij,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VP, VS, RHO = 2400.0, 1300.0, 2000.0


def _ricker(nt, dt, f0=12.0, t0=0.10):
    t = torch.arange(nt, dtype=torch.float64) * dt - t0
    a = (math.pi * f0 * t) ** 2
    return (1 - 2 * a) * torch.exp(-a)


def _full(mesh, val):
    return torch.full((mesh.n_elements, mesh.ngll, mesh.ngll), val,
                      dtype=torch.float64, device=DEV)


def test_thomsen_cij_isotropic_limit():
    vp = torch.tensor([2400.0], dtype=torch.float64)
    vs = torch.tensor([1300.0], dtype=torch.float64)
    rho = torch.tensor([2000.0], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    c11, c13, c33, c55 = thomsen_to_vti_cij(vp, vs, zero, zero, rho)
    lam2mu = rho * vp * vp
    mu = rho * vs * vs
    lam = lam2mu - 2 * mu
    assert torch.allclose(c11, lam2mu, rtol=1e-12)
    assert torch.allclose(c33, lam2mu, rtol=1e-12)
    assert torch.allclose(c55, mu, rtol=1e-12)
    assert torch.allclose(c13, lam, rtol=1e-9)


def test_vti_degenerates_to_isotropic():
    mesh = build_axis_aligned_quad9(nex=10, nez=8, x_max=1000.0, z_max=800.0,
                                    n_order=4, dtype=torch.float64, device=DEV)
    vp, vs, rho = _full(mesh, VP), _full(mesh, VS), _full(mesh, RHO)
    eps, delta = _full(mesh, 0.0), _full(mesh, 0.0)
    dt, nt = 1e-3, 150
    wav = _ricker(nt, dt).to(DEV)
    src_xz = [(500.0, 400.0)]
    rec_xz = [(300.0, 400.0), (700.0, 400.0), (500.0, 250.0)]
    mt = ["moment_tensor"]; pay = [(1.0, 1.0, 0.0)]

    eq_iso = ElasticVSSEM2DDeformed.build(mesh, vp, vs, rho)
    eq_vti = ElasticVTISEM2DDeformed.build(mesh, vp, vs, rho, eps, delta)
    out_iso = SEMPropagator.build(eq_iso, dt=dt, nt=nt, source_xz=src_xz,
                                  receiver_xz=rec_xz, source_kinds=mt,
                                  source_payloads=pay).forward(wav)
    out_vti = SEMPropagator.build(eq_vti, dt=dt, nt=nt, source_xz=src_xz,
                                  receiver_xz=rec_xz, source_kinds=mt,
                                  source_payloads=pay).forward(wav)
    for j, key in enumerate(('v_x', 'v_z')):     # component order
        a, b = out_vti[..., j].T, out_iso[..., j].T
        rel = (a - b).norm() / (b.norm() + 1e-30)
        assert rel < 1e-9, f"{key} VTI(eps=delta=0) vs isotropic rel={rel:.2e}"


def test_vti_stable_and_differentiable():
    mesh = build_axis_aligned_quad9(nex=10, nez=8, x_max=1000.0, z_max=800.0,
                                    n_order=4, dtype=torch.float64, device=DEV)
    vp0, vs, rho = _full(mesh, VP), _full(mesh, VS), _full(mesh, RHO)
    eps, delta = _full(mesh, 0.12), _full(mesh, 0.06)   # genuinely anisotropic
    dt, nt = 1e-3, 200
    wav = _ricker(nt, dt).to(DEV)
    src_xz = [(500.0, 400.0)]; rec_xz = [(300.0, 400.0), (700.0, 400.0)]
    mt = ["moment_tensor"]; pay = [(1.0, 1.0, 0.0)]

    vp = vp0.clone().requires_grad_(True)
    eq = ElasticVTISEM2DDeformed.build(mesh, vp, vs, rho, eps, delta)
    out = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=src_xz,
                              receiver_xz=rec_xz, source_kinds=mt,
                              source_payloads=pay).forward(wav)
    seis = out[..., 0].T
    assert torch.isfinite(seis).all()
    assert seis.abs().max() > 0
    loss = (seis ** 2).sum()
    loss.backward()
    assert torch.isfinite(vp.grad).all()
    assert vp.grad.abs().max() > 0
