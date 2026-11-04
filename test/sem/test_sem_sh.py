"""ElasticSHSEM2DDeformed: SH waves (SPECFEM2D's P_SV = .false.).

test/sem/specfem_compare/run_compare_box.py checks it against SPECFEM2D; here: its surface is
traction-free (a free top doubles the field instead of zeroing it), it is the potential form's
operators with ``vp = vs`` and the density ``1 / (rho vs^2)``, the boundary-saved gradients are
the full tape's, the gradients in ``vs`` and ``rho`` exist, and what it cannot do is refused.
"""
import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, EagerOptions, Full
from sweep.sem import AcousticPotentialSEM2D, ElasticSHSEM2DDeformed, PropSEM

SHAPE, DT, NT = (41, 81), 6e-4, 400
F64 = torch.float64


def _prop(equation=ElasticSHSEM2DDeformed, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), "abcn": 16,
          "free_surface": True, **kw}
    return PropSEM(equation, shape=SHAPE, dh=10.0, dt=DT, n_order=4, f_dominant=10.0, dev="cpu",
                   dtype=F64, impl="eager", **kw)


def _shot(prop, models, z_rec=0.0):
    t = np.arange(NT) * DT - 0.1
    a = (np.pi * 10.0 * t) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    rx = np.arange(10.0, 71.0, 10.0)
    rec = np.stack([rx, np.full_like(rx, z_rec)], 1)[None]
    return prop(w, np.array([[40.3, 20.6]]), rec, models=models)


def _models(grad=False):
    z = torch.linspace(0.0, 1.0, SHAPE[0], dtype=F64)[:, None].expand(SHAPE)
    return [(1300.0 + 200.0 * z).clone().requires_grad_(grad),
            torch.full(SHAPE, 2100.0, dtype=F64, requires_grad=grad)]


def test_the_surface_is_traction_free():
    # on the free top the SH field is not zero (as an acoustic pressure-release surface is)
    seis = _shot(_prop(receiver_type=["u_y"]), _models(), z_rec=0.0)
    assert seis.abs().max() > 0
    # and it is the potential form's operators, with no Dirichlet nodes
    vs, rho = _models()
    sh = _prop()._build_eq([vs, rho])
    pot = AcousticPotentialSEM2D.build(sh.mesh, sh.vp_e, sh.rho_e, dirichlet=(), cpml=sh.cpml)
    assert sh.dirichlet_idx.numel() == 0
    assert torch.equal(sh._inv_M, pot._inv_M) and torch.equal(sh._inv_rho, pot._inv_rho)


def test_boundary_saving_gives_the_full_tape():
    vs0, rho0 = _models(grad=True)
    _shot(_prop(receiver_type=["vy"]), [vs0, rho0]).square().sum().backward()
    vs1, rho1 = _models(grad=True)
    _shot(_prop(receiver_type=["vy"], memory=BoundarySaving()), [vs1, rho1]).square().sum().backward()
    for a, b in ((vs1.grad, vs0.grad), (rho1.grad, rho0.grad)):
        assert torch.isfinite(b).all() and b.abs().max() > 0
        assert (a - b).norm() <= 1e-9 * b.norm()


def test_what_it_cannot_do_is_refused():
    with pytest.raises(NotImplementedError, match="impl='c'"):
        PropSEM(ElasticSHSEM2DDeformed, shape=SHAPE, dh=10.0, dt=DT, n_order=4, dev="cpu",
                impl="c")
