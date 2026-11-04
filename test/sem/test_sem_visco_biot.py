"""ViscoBiotVSSEM2DDeformed: the Biot SEM with a GSLS frame.

``Q = inf`` is BiotVSSEM2DDeformed bit for bit; a dry solid is ViscoElasticVSSEM2DDeformed to
round-off (which test/sem/specfem_compare/run_compare_visco.py compares with SPECFEM2D); the
gradients in the frame's ``Qp`` and ``Qs`` in a porous rock are the derivatives of the
records; boundary saving is refused.
"""
import math

import numpy as np
import pytest
import torch

from sweep.equations.biot import biot_from_rock
from sweep.equations.biot_params import biot_elastic
from sweep.propagator import BoundarySaving, EagerOptions, Full
from sweep.sem import (BiotVSSEM2DDeformed, PropSEM, ViscoBiotVSSEM2DDeformed,
                       ViscoElasticVSSEM2DDeformed)

F64 = torch.float64
ROCK = biot_from_rock(k_s=12.2e9, rho_s=2650.0, k_f=1.985e9, rho_f=880.0, eta=1e-3, k_d=9.6e9,
                      mu=5.1e9, phi=0.1, kappa=1e-9, tortuosity=2.0)


def _prop(eq, shape, dt, src, rec, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), **kw}
    return PropSEM(eq, shape=shape, dh=10.0, dt=dt, n_order=4, f_dominant=12.0, dev="cpu",
                   impl="eager", dtype=F64, source_type=src, receiver_type=rec, **kw)


def _ricker(nt, dt, f0=12.0):
    a = (math.pi * f0 * (np.arange(nt) * dt - 1.2 / f0)) ** 2
    return torch.from_numpy((1 - 2 * a) * np.exp(-a))


def _layered(mesh, tops, media):
    """GLL-native coefficients: an element takes ``media[k]`` when its centre lies between
    the depths ``tops[k]`` and ``tops[k + 1]``."""
    zc = mesh.z_e.mean((1, 2))
    k = torch.bucketize(zc, torch.tensor(tops[1:], dtype=zc.dtype))
    E, n = mesh.n_elements, mesh.ngll
    return [torch.stack([torch.as_tensor(m[i], dtype=F64) for m in media])[k]
            .view(E, 1, 1).expand(E, n, n).contiguous() for i in range(len(media[0]))]


SRC = np.array([[20.0, 10.0]])
REC = np.stack([np.arange(4.0, 37.0, 4.0), np.full(9, 22.0)], 1)[None]


def test_infinite_q_is_biot_bit_for_bit():
    shape, dt, nt = (33, 41), 5e-4, 200
    kw = dict(abcn=8, free_surface=True)
    biot = _prop(BiotVSSEM2DDeformed, shape, dt, ["sxx", "szz"], ["vx", "vz", "p"], **kw)
    visco = _prop(ViscoBiotVSSEM2DDeformed, shape, dt, ["sxx", "szz"], ["vx", "vz", "p"], **kw)
    coef = [torch.full(shape, float(c), dtype=F64) for c in ROCK]
    inf = torch.full((visco.mesh.n_elements, visco.mesh.ngll, visco.mesh.ngll), math.inf,
                     dtype=F64)
    w = _ricker(nt, dt)
    with torch.no_grad():
        assert torch.equal(visco(w, SRC, REC, models=coef + [inf, inf]),
                           biot(w, SRC, REC, models=coef))


def test_a_dry_solid_is_the_visco_elastic_sem():
    # no pore fluid (M = 0): the drained P modulus is H and the frame is the visco-elastic
    # solid, so the records are ViscoElasticVSSEM2DDeformed's
    shape, dt, nt = (33, 41), 5e-4, 300
    kw = dict(abcn=0, free_surface=True, equation_options={"f_ref": 12.0})
    rocks = [(2500.0, 1400.0, 2200.0, 25.0, 15.0), (3000.0, 1600.0, 2200.0, 60.0, 40.0)]
    biot = _prop(ViscoBiotVSSEM2DDeformed, shape, dt, ["sxx", "szz"], ["vx", "vz", "p"], **kw)
    elas = _prop(ViscoElasticVSSEM2DDeformed, shape, dt, ["sxx", "szz"], ["vx", "vz"], **kw)
    w = _ricker(nt, dt)
    with torch.no_grad():
        db = biot(w, SRC, REC, models=_layered(
            biot.mesh, [0.0, 150.0], [list(biot_elastic(*r[:3])) + list(r[3:]) for r in rocks]))
        de = elas(w, SRC, REC, models=_layered(elas.mesh, [0.0, 150.0], rocks))
    assert not db[..., 2].any()
    err = float((db[..., :2] - de).abs().max() / de.abs().max())
    assert err < 1e-11, err


@pytest.mark.parametrize("which", [8, 9], ids=["Qp", "Qs"])
def test_gradients_in_the_frame_q_are_the_derivative_of_the_records(which):
    """A directional derivative against central differences over three steps."""
    torch.manual_seed(0)
    shape, dt, nt = (21, 41), 5e-4, 160
    p = _prop(ViscoBiotVSSEM2DDeformed, shape, dt, ["sxx", "szz"], ["vx", "vz", "p"], abcn=8,
              free_surface=True)
    z = torch.linspace(0.0, 1.0, shape[0], dtype=F64)[:, None].expand(shape)
    ms = [torch.full(shape, float(c), dtype=F64) for c in ROCK] + [20.0 + 10.0 * z, 15.0 + 5.0 * z]
    ms = [m.clone().requires_grad_(True) for m in ms]
    w = _ricker(nt, dt)
    src = np.array([[20.0, 6.0]])
    rec = np.stack([np.arange(4.0, 37.0, 4.0), np.full(9, 3.0)], 1)[None]
    loss = lambda models: p(w, src, rec, models=models).square().sum()
    loss(ms).backward()
    d = torch.randn(shape, dtype=F64)
    d[:2] = d[-2:] = 0.0
    d[:, :2] = d[:, -2:] = 0.0
    analytic = float((ms[which].grad * d).sum())
    scale = float(ms[which].detach().abs().mean())
    errs = []
    for h in (1e-3, 1e-4, 1e-5):
        plus = [m.detach().clone() for m in ms]
        minus = [m.detach().clone() for m in ms]
        plus[which] += h * scale * d
        minus[which] -= h * scale * d
        with torch.no_grad():
            fd = float(loss(plus) - loss(minus)) / (2 * h * scale)
        errs.append(abs(fd - analytic) / abs(analytic))
    assert min(errs) < 1e-7, errs


def test_boundary_saving_is_refused():
    with pytest.raises(NotImplementedError, match="boundary saving"):
        _prop(ViscoBiotVSSEM2DDeformed, (21, 41), 5e-4, ["sxx", "szz"], ["vx"],
              memory=BoundarySaving())
