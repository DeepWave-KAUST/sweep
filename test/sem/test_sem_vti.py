"""VTI and TTI (Thomsen) velocity-stress SEM: degeneracy + stability + autograd.

The headline correctness check is that ``epsilon = delta = 0`` reduces the VTI
constitutive to isotropy, so ``ElasticVTISEM2DDeformed`` must reproduce
``ElasticVSSEM2DDeformed`` to machine precision. Then a genuinely anisotropic
run must stay finite and yield a finite, nonzero model gradient (autograd).
"""
import math

import pytest
import torch

from sweep.sem import (
    ElasticTTISEM2DDeformed,
    ElasticVSSEM2DDeformed,
    ElasticVTISEM2DDeformed,
    SEMPropagator,
    build_axis_aligned_quad9,
    thomsen_to_tti_cij,
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


def test_tti_cij_are_the_fd_elastic_tti_2nd_ones():
    from sweep.equations import ElasticTTI2nd
    vp, vs, rho = (torch.tensor([v], dtype=torch.float64) for v in (VP, VS, RHO))
    eps, delta = torch.tensor([0.21], dtype=torch.float64), torch.tensor([-0.07], dtype=torch.float64)
    vh, eta = vp * torch.sqrt(1 + 2 * eps), (eps - delta) / (1 + 2 * delta)
    for deg in (0.0, 17.0, 45.0, -60.0, 90.0, 133.0):
        theta = torch.tensor([math.radians(deg)], dtype=torch.float64)
        c11, c13, c15, c33, c35, c55 = thomsen_to_tti_cij(vp, vs, eps, delta, rho, theta)
        _, f11, f33, f13, f55, f15, f35 = ElasticTTI2nd().prepare_models(
            [vh, vs, rho, eps, eta, theta])
        for ours, fd in ((c11, f11), (c13, f13), (c15, f15), (c33, f33), (c35, f35),
                         (c55, f55)):
            assert torch.allclose(ours, fd, rtol=1e-12, atol=1e-12 * float(f11)), deg


def _tti_records(theta_deg, rec_xz, nt=260):
    mesh = build_axis_aligned_quad9(nex=11, nez=11, x_max=1100.0, z_max=1100.0,
                                    n_order=4, dtype=torch.float64, device=DEV)
    eq = ElasticTTISEM2DDeformed.build(
        mesh, _full(mesh, VP), _full(mesh, VS), _full(mesh, RHO), _full(mesh, 0.25),
        _full(mesh, 0.1), _full(mesh, math.radians(theta_deg)))
    dt = 1e-3
    # an explosion at the centre of the diagonal's middle element: the box,
    # mesh and source are symmetric under x <-> z
    return SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=[(550.0, 550.0)],
                               receiver_xz=rec_xz, source_kinds=["moment_tensor"],
                               source_payloads=[(1.0, 1.0, 0.0)]).forward(
                                   _ricker(nt, dt).to(DEV))


def test_tti_at_zero_tilt_is_vti():
    mesh = build_axis_aligned_quad9(nex=10, nez=8, x_max=1000.0, z_max=800.0,
                                    n_order=4, dtype=torch.float64, device=DEV)
    args = (_full(mesh, VP), _full(mesh, VS), _full(mesh, RHO), _full(mesh, 0.12),
            _full(mesh, 0.06))
    dt, nt = 1e-3, 150
    kw = dict(dt=dt, nt=nt, source_xz=[(510.0, 390.0)], receiver_xz=[(300.0, 400.0)],
              source_kinds=["moment_tensor"], source_payloads=[(1.0, 0.5, 0.3)])
    vti = SEMPropagator.build(ElasticVTISEM2DDeformed.build(mesh, *args), **kw)
    tti = SEMPropagator.build(ElasticTTISEM2DDeformed.build(mesh, *args, _full(mesh, 0.0)), **kw)
    wav = _ricker(nt, dt).to(DEV)
    a, b = tti.forward(wav), vti.forward(wav)
    # the stress updates agree to the bit (c15 = c35 = 0 add zeros); the CUDA
    # assembly's atomics are what is left
    assert (a - b).abs().max() <= 1e-12 * b.abs().max()


def test_tti_mirrors_into_the_complementary_tilt():
    # x <-> z maps the tilt theta to 90 deg - theta (the cij swap 11 <-> 33,
    # 15 <-> 35), so the records at (x, z) under theta are those at (z, x) under
    # 90 - theta with v_x and v_z exchanged -- which a c15 / c35 put in the
    # wrong place of the stress update breaks
    rec = [(850.0, 550.0), (790.0, 330.0), (300.0, 910.0), (610.0, 140.0)]
    a = _tti_records(30.0, rec)
    b = _tti_records(60.0, [(z, x) for x, z in rec])
    assert a.abs().max() > 0
    scale = a.abs().max()
    assert (a[..., 0] - b[..., 1]).abs().max() < 1e-10 * scale
    assert (a[..., 1] - b[..., 0]).abs().max() < 1e-10 * scale
    # and a tilted medium is not the swap of itself
    c = _tti_records(30.0, [(z, x) for x, z in rec])
    assert (a[..., 0] - c[..., 1]).abs().max() > 1e-2 * scale


def test_tti_tilt_is_differentiable():
    mesh = build_axis_aligned_quad9(nex=10, nez=8, x_max=1000.0, z_max=800.0,
                                    n_order=4, dtype=torch.float64, device=DEV)
    theta = _full(mesh, 0.4).requires_grad_(True)
    eq = ElasticTTISEM2DDeformed.build(mesh, _full(mesh, VP), _full(mesh, VS),
                                       _full(mesh, RHO), _full(mesh, 0.12),
                                       _full(mesh, 0.06), theta)
    dt, nt = 1e-3, 200
    seis = SEMPropagator.build(eq, dt=dt, nt=nt, source_xz=[(510.0, 390.0)],
                               receiver_xz=[(300.0, 400.0), (700.0, 400.0)],
                               source_kinds=["moment_tensor"],
                               source_payloads=[(1.0, 1.0, 0.0)]).forward(
                                   _ricker(nt, dt).to(DEV))
    (seis ** 2).sum().backward()
    assert torch.isfinite(theta.grad).all()
    assert theta.grad.abs().max() > 0


def test_tti_boundary_saving_matches_the_full_tape():
    # the six stiffnesses are the leaves the boundary-saved backward substitutes
    # (bs_model_attrs): one left out would leave its model's gradient wrong.
    # Exact up to the C-PML band too: the per-step VJP pins the velocity
    # update to the ring (boundary_saving.pin_to_ring; before that ~1e-5 on
    # the whole grid, ~1e-7 four cells in, the isotropic class too)
    from sweep.propagator import BoundarySaving, Full
    from sweep.sem import PropSEM
    shape = (25, 41)

    def run(memory):
        p = PropSEM(ElasticTTISEM2DDeformed, shape=shape, dh=10.0, dt=6e-4, n_order=4, abcn=8,
                    f_dominant=15.0, dev="cpu", dtype=torch.float64, impl="eager",
                    source_type=["sxx", "szz"], receiver_type=["vx", "vz"],
                    free_surface=True, memory=memory)
        g = torch.Generator().manual_seed(3)
        ms = [(v * (1 + 0.05 * torch.rand(shape, generator=g, dtype=torch.float64)))
              .requires_grad_(True) for v in (VP, VS, RHO, 0.12, 0.06, 0.4)]
        rx = torch.arange(3, 38, 3, dtype=torch.float64)
        r = p(_ricker(160, 6e-4, f0=15.0, t0=0.08), torch.tensor([[12.0, 5.0]]),
              torch.stack([rx, torch.full_like(rx, 4.0)], 1)[None], models=ms)
        (r / r.detach().abs().max()).square().sum().backward()
        return r.detach(), [m.grad for m in ms]

    r0, g0 = run(Full())
    r1, g1 = run(BoundarySaving())
    assert (r1 - r0).abs().max() <= 1e-12 * r0.abs().max()
    for a, b in zip(g1, g0):
        assert b.abs().max() > 0
        assert (a - b).norm() <= 1e-12 * b.norm()
