"""ViscoAcousticPotentialSEM2D: SPECFEM2D's visco-acoustic potential SEM.

test/sem/specfem_compare/run_compare_box.py checks it against SPECFEM2D's
ATTENUATION_VISCOACOUSTIC; here: ``Qkappa = inf`` is the potential step bit for bit, a finite Q
attenuates with distance, the gradients in ``vp`` and ``Qkappa`` exist, the unrelaxed reading
of ``vp`` is the mass matrix's, and what it cannot do is refused.
"""
import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, EagerOptions, Full
from sweep.sem import AcousticPotentialSEM2D, PropSEM, ViscoAcousticPotentialSEM2D

SHAPE, DT, NT = (61, 121), 4e-4, 700
F64 = torch.float64
ALL = {"top": True, "bottom": True, "left": True, "right": True}


def _prop(equation=ViscoAcousticPotentialSEM2D, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), "abcn": 0,
          "free_surface": ALL, **kw}
    if equation is ViscoAcousticPotentialSEM2D:
        kw.setdefault("equation_options", {"f_ref": 15.0})
    return PropSEM(equation, shape=SHAPE, dh=5.0, dt=DT, n_order=4, f_dominant=15.0, dev="cpu",
                   dtype=F64, impl="eager", density=1020.0, receiver_type=["p"], **kw)


def _shot(prop, models):
    t = np.arange(NT) * DT - 0.08
    a = (np.pi * 15.0 * t) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    rec = np.array([[[30.0 + r, 30.0] for r in (15.0, 35.0, 55.0)]])
    return prop(w, np.array([[30.3, 30.6]]), rec, models=models)[0, :, :, 0]


def _vp():
    return torch.full(SHAPE, 1500.0, dtype=F64)


def test_infinite_q_is_the_potential_step_bit_for_bit():
    p = _prop()
    qinf = torch.full(p._gll_shape, float("inf"), dtype=F64)
    assert torch.equal(_shot(p, [_vp(), qinf]), _shot(_prop(AcousticPotentialSEM2D), [_vp()]))


def test_a_finite_q_attenuates_with_distance():
    # absorbing sides and the direct pulse alone: the edges' echoes would mix in
    kw = dict(abcn=16, free_surface=False)
    elastic = _shot(_prop(AcousticPotentialSEM2D, **kw), [_vp()])
    lossy = _shot(_prop(**kw), [_vp(), torch.full(SHAPE, 20.0, dtype=F64)])
    t = torch.arange(NT, dtype=F64) * DT - 0.08
    ratio = []
    for i, r in enumerate((75.0, 175.0, 275.0)):
        win = (t - r / 1500.0).abs() < 0.06
        ratio.append(float(lossy[win, i].abs().max() / elastic[win, i].abs().max()))
    assert ratio[0] < 1.0 and ratio[1] < ratio[0] and ratio[2] < ratio[1]


def test_the_gradients_exist():
    vp, qk = _vp().requires_grad_(True), torch.full(SHAPE, 30.0, dtype=F64, requires_grad=True)
    (_shot(_prop(), [vp, qk]) ** 2).sum().backward()
    for g in (vp.grad, qk.grad):
        assert torch.isfinite(g).all() and g.abs().max() > 0


def test_the_unrelaxed_reading_is_the_mass_matrix_modulus():
    p = _prop(equation_options={"f_ref": 15.0, "velocities": "unrelaxed"})
    eq = p._build_eq([_vp(), torch.full(SHAPE, 30.0, dtype=F64)])
    assert torch.allclose(eq.kappa_node, torch.full_like(eq.kappa_node, 1020.0 * 1500.0 ** 2),
                          rtol=1e-14, atol=0)
    # the phase velocity at f_ref puts the unrelaxed modulus above rho vp^2
    eq = _prop()._build_eq([_vp(), torch.full(SHAPE, 30.0, dtype=F64)])
    assert float(eq.kappa_node.min()) > 1020.0 * 1500.0 ** 2


def test_what_it_cannot_do_is_refused():
    with pytest.raises(NotImplementedError, match="dissipative"):
        _prop(memory=BoundarySaving(), abcn=8, free_surface=True)
    with pytest.raises(NotImplementedError, match="impl='c'"):
        PropSEM(ViscoAcousticPotentialSEM2D, shape=SHAPE, dh=5.0, dt=DT, n_order=4,
                dev="cpu", impl="c")
    with pytest.raises(ValueError, match="Bound Q from below"):
        _shot(_prop(), [_vp(), torch.full(SHAPE, 1.1, dtype=F64)])
