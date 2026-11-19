"""PseudoelasticP (Sun & Alkhalifah 2021 pure P-mode), eager.

1. the projection leaves a staggered FD gradient field unchanged (it is built from the stencil's symbol);
2. with vs = 0 the record equals Elastic(vs = 0) to round-off, without and with the free surface;
3. a shear-modulus-only interface reflects P at oblique incidence: the pure P-mode record carries that
   reflection like Elastic, while the acoustic record (vs = 0) has none;
4. eager only, boundary saving refused, free surface on a solid top refused.
"""
import numpy as np
import pytest
import torch

import sweep.equations as E
from sweep.equations import Elastic, PseudoelasticP
from sweep.propagator.options import BoundarySaving, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

UNCOMPILED = EagerOptions(use_compile=False)
DH, DT = 10.0, 1.0e-3


def _prop(eq, shape, nt, fs=False, abcn=12, memory=None, **kw):
    return PropTorch(eq, shape=shape, abcn=abcn, dh=DH, dt=DT, impl="eager", eager_options=UNCOMPILED,
                     memory=memory or Full(), free_surface=fs, source_type=kw.pop("source_type", ["sxx", "szz"]),
                     receiver_type=["sxx"], **kw)


def _wavelet(nt, f=20.0):
    t = np.arange(nt) * DT - 0.06
    return torch.tensor(1.0e3 * ricker(t, f=f), dtype=torch.float32)


def test_projection_leaves_an_fd_gradient_field_unchanged():
    nz, nx = 48, 56
    eq = PseudoelasticP(spatial_order=4, device="cpu")
    _prop(eq, (nz, nx), 10)                       # sets the stencil spacing
    torch.manual_seed(0)
    p = torch.randn(1, 1, nz + 24, nx + 24)
    p[..., :6, :] = 0; p[..., -6:, :] = 0; p[..., :, :6] = 0; p[..., :, -6:] = 0   # the stencil's zero padding = periodic
    vx, vz = eq.pd.x_forward(p), eq.pd.z_forward(p)
    px, pz = eq._project(vx, vz, DH, DH)
    assert torch.allclose(px, vx, atol=1e-5 * vx.abs().max()), (px - vx).abs().max()
    assert torch.allclose(pz, vz, atol=1e-5 * vz.abs().max())
    # and it is a projector: applying it twice changes nothing more
    torch.manual_seed(1)
    ax, az = torch.randn_like(vx), torch.randn_like(vz)
    qx, qz = eq._project(ax, az, DH, DH)
    rx, rz = eq._project(qx, qz, DH, DH)
    assert torch.allclose(rx, qx, atol=1e-5) and torch.allclose(rz, qz, atol=1e-5)
    assert (qx - ax).abs().max() > 0.1          # a random field is not curl-free


@pytest.mark.parametrize("fs", [False, True], ids=["no_fs", "fs"])
def test_vs_zero_reproduces_elastic(fs):
    nz, nx, nt = 60, 80, 260
    vp = torch.full((nz, nx), 1500.0); vp[30:] = 1900.0
    rho = torch.full((nz, nx), 1000.0); rho[30:] = 2000.0
    vs = torch.zeros(nz, nx)
    src = torch.tensor([[nx // 2, 20]]); rec_x = torch.arange(6, nx - 6, 3)
    rec = torch.stack([rec_x, torch.full_like(rec_x, 4)], -1)[None]
    w = _wavelet(nt)
    out = []
    for cls in (Elastic, PseudoelasticP):
        prop = _prop(cls(spatial_order=4, device="cpu"), (nz, nx), nt, fs=fs)
        out.append(prop(w, src, rec, models=[vp, vs, rho])[0, ..., 0].detach())
    ref, got = out
    assert ref.abs().max() > 0
    # with the free surface the near-surface rows use the order-2 image stencil, whose symbol differs from the
    # interior one the projector is built from: a few 1e-4 instead of 3e-5
    assert (got - ref).norm() / ref.norm() < (2e-3 if fs else 2e-4)


def test_shear_only_interface_reflects_p_at_oblique_incidence():
    """vp and rho are constant, vs jumps at 500 m: the acoustic impedance is continuous, so the acoustic
    record has no reflection; Elastic and the pure P-mode do (the PP AVO term in the shear modulus)."""
    nz, nx, nt = 90, 160, 700
    vp = torch.full((nz, nx), 2000.0); rho = torch.full((nz, nx), 2000.0)
    vs = torch.full((nz, nx), 800.0); vs[50:] = 1200.0
    src = torch.tensor([[12, 8]])
    rec_x = torch.arange(44, 84, 4)              # offsets 320-720 m
    rec = torch.stack([rec_x, torch.full_like(rec_x, 8)], -1)[None]
    w = _wavelet(nt, f=15.0)
    recs = {}
    for name, cls, vs_m in (("elastic", Elastic, vs), ("pure_p", PseudoelasticP, vs), ("acoustic", Elastic, vs * 0)):
        prop = _prop(cls(spatial_order=4, device="cpu"), (nz, nx), nt, abcn=16)
        recs[name] = prop(w, src, rec, models=[vp, vs_m, rho])[0, ..., 0].detach().numpy()
    t = np.arange(nt) * DT
    x = (rec_x.numpy() - 12) * DH; d = (50 - 8) * DH
    t_ref = (np.sqrt(x ** 2 + (2 * d) ** 2)) / 2000.0 + 0.06    # PP reflection, wavelet delay 0.06 s
    t_dir = x / 2000.0 + 0.06
    assert np.all(t_ref - t_dir > 0.15)                           # the windows do not overlap
    def energy(r):
        e = 0.0
        for j in range(len(x)):
            win = (t > t_ref[j] - 0.04) & (t < t_ref[j] + 0.08)
            e += float((r[win, j] ** 2).sum())
        return e
    e_el, e_pp, e_ac = energy(recs["elastic"]), energy(recs["pure_p"]), energy(recs["acoustic"])
    assert e_ac < 0.02 * e_el, (e_ac, e_el)
    # the pure P-mode PP reflection is ~25 % stronger than the elastic one at this (125 %) shear-modulus
    # jump: it loses no energy to the converted S wave (the radiation patterns agree to first order)
    assert 0.8 < e_pp / e_el < 2.0, (e_pp, e_el)


def test_eager_only_and_refusals():
    assert not PseudoelasticP.supports_torch_binding()
    assert "PseudoelasticP" in E.list_equations() and E.get_equation("PureP") is PseudoelasticP
    nz, nx = 40, 48
    vp0 = torch.full((nz, nx), 2000.0, requires_grad=True)
    prop = _prop(PseudoelasticP(spatial_order=4, device="cpu"), (nz, nx), 20, memory=BoundarySaving())
    with pytest.raises(NotImplementedError):            # refused when the reverse driver is picked
        prop(_wavelet(20), torch.tensor([[nx // 2, 10]]), torch.tensor([[[10, 4]]]),
             models=[vp0, torch.zeros(nz, nx), torch.full((nz, nx), 2000.0)]).square().sum().backward()
    # a solid along the surface row is refused when the free surface is on
    vp = torch.full((nz, nx), 2000.0); vs = torch.full((nz, nx), 800.0); rho = torch.full((nz, nx), 2000.0)
    prop = _prop(PseudoelasticP(spatial_order=4, device="cpu"), (nz, nx), 20, fs=True)
    with pytest.raises(ValueError):
        prop(_wavelet(20), torch.tensor([[nx // 2, 10]]), torch.tensor([[[10, 4]]]), models=[vp, vs, rho])
