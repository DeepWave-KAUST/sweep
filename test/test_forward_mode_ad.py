"""Forward-mode AD (Jacobian-vector products) through the eager propagator.

``torch.autograd.forward_ad`` carries a direction h through the uncompiled eager
step with memory=Full(): the tangent of the record is J h, the Born record of h,
from the plain equation and with no autograd tape. A Hessian-vector product
then needs first derivatives only: the gradient of <d, s.detach()> +
<s, r.detach()> with s = J h and r the residual (reverse over forward).
The paths that cannot carry a tangent -- the compiled step, checkpointing,
eager boundary saving, impl='c', PropSEM, the qP adjoint-mode step, a
registered gradient-mode imaging condition -- refuse a dual tensor instead of
dropping its tangent: the compiled steps used to return a tangent of zero
without a word.
"""
import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from conftest import requires_binding, requires_compile
from sweep.equations import Acoustic, AcousticVTI, Elastic, ElasticTTI, ViscoElastic
from sweep.propagator import BoundarySaving, Full
from sweep.propagator._second_order import CreateGraphGuard
from sweep.propagator.options import Ckpt, EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
NT = 120
SRC = np.array([[8, 3], [19, 3]])
REC = np.repeat(np.array([[[ix, 2] for ix in range(2, 26, 2)]]), 2, axis=0)
BASE = {"vp": 2000.0, "vs": 1150.0, "rho": 2100.0, "Qp": 60.0, "Qs": 40.0, "epsilon": 0.1,
        "delta": 0.05, "vp0": 2000.0, "vs0": 1150.0, "gamma": 0.05, "theta": 0.3, "phi": 0.2}


def _wavelet(device="cpu"):
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.04
    a = (np.pi * 25.0 * t) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=device)


def _prop(equation, memory, impl="eager", dev="cpu", **eager):
    kw = dict(shape=SHAPE, dh=10.0, dt=1e-3, nt=NT, abcn=8, dev=dev, impl=impl, memory=memory)
    if impl == "eager":
        kw["eager_options"] = EagerOptions(**{"use_compile": False, **eager})
    return PropTorch(equation, **kw)


def _models(names, device="cpu"):
    z = torch.arange(SHAPE[0], dtype=torch.float32, device=device)[:, None] * torch.ones(SHAPE, device=device)
    return [BASE[n] * (1 + 0.004 * z) for n in names]


def _directions(models, seed=0, ring=True):
    g = torch.Generator().manual_seed(seed)
    mask = torch.ones(SHAPE)
    if ring:
        mask = torch.zeros(SHAPE)
        mask[2:-2, 2:-2] = 1.0       # off the outer cells, whose gradient is cropped (EdgePadding)
    return [(torch.randn(SHAPE, generator=g) * mask).to(m.device) * 1e-3 * m.abs().mean() for m in models]


def _exact_jvp(prop, models, hs):
    """J h from reverse mode twice (the transpose of a vector-Jacobian product)."""
    ms = [m.clone().requires_grad_(True) for m in models]
    d = prop(_wavelet(models[0].device), SRC, REC, models=ms)
    u = torch.zeros_like(d, requires_grad=True)
    gs = torch.autograd.grad(d, ms, grad_outputs=u, create_graph=True)
    return torch.autograd.grad(sum((g * h).sum() for g, h in zip(gs, hs)), u)[0]


def _forward_jvp(prop, models, hs):
    with torch.no_grad(), fwAD.dual_level():
        out = prop(_wavelet(models[0].device), SRC, REC,
                   models=[fwAD.make_dual(m, h) for m, h in zip(models, hs)])
        primal, tangent = fwAD.unpack_dual(out)
    return primal, tangent


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("make", [
    lambda: Acoustic(spatial_order=4, device="cpu"),
    lambda: Elastic(spatial_order=4, device="cpu"),
    lambda: ViscoElastic(spatial_order=4, device="cpu"),      # the GSLS strengths' jvp
    lambda: ElasticTTI(spatial_order=4, device="cpu"),        # the receiver footprint's jvp
], ids=["acoustic", "elastic", "viscoelastic", "elastic_tti"])
def test_forward_mode_tangent_is_the_jacobian_vector_product(make):
    prop = _prop(make(), Full())
    models = _models(prop.equation.models)
    hs = _directions(models)
    primal, tangent = _forward_jvp(prop, models, hs)
    with torch.no_grad():
        plain = prop(_wavelet(), SRC, REC, models=models)
    assert torch.equal(primal, plain)                       # the primal is the plain forward
    assert _rel(tangent, _exact_jvp(prop, models, hs)) < 1e-5


def test_edge_cells_are_padded_into_the_pml_in_the_tangent_too():
    """A direction on the outermost cells moves the edge-replicated PML strip
    (EdgePadding's jvp is the replicate padding): the tangent is the derivative
    of the record, as a central difference of two forwards sees it."""
    prop = _prop(Acoustic(spatial_order=4, device="cpu"), Full())
    (vp,) = _models(["vp"])
    edge = torch.zeros(SHAPE)
    edge[:2] = 1.0
    edge[-2:] = 1.0
    edge[:, :2] = 1.0
    edge[:, -2:] = 1.0
    h = edge * torch.randn(SHAPE, generator=torch.Generator().manual_seed(1)) * 2.0
    _, tangent = _forward_jvp(prop, [vp], [h])
    with torch.no_grad():
        diff = (prop(_wavelet(), SRC, REC, models=[vp + h]) - prop(_wavelet(), SRC, REC, models=[vp - h])) / 2
    assert _rel(tangent, diff) < 5e-3


def test_reverse_over_forward_hessian_vector_product_matches_double_backward():
    prop = _prop(Acoustic(spatial_order=4, device="cpu"), Full())
    (v0,) = _models(["vp"])
    v_true = v0.clone()
    v_true[10:14, 10:18] += 150.0
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    (h,) = _directions([v0])

    m = v0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(0.5 * ((prop(_wavelet(), SRC, REC, models=[m]) - d_obs) ** 2).sum(), m,
                               create_graph=True)
    (hv_db,) = torch.autograd.grad((g * h).sum(), m)

    m = v0.clone().requires_grad_(True)
    with fwAD.dual_level():
        d, s = fwAD.unpack_dual(prop(_wavelet(), SRC, REC, models=[fwAD.make_dual(m, h)]))
        r = d - d_obs
        psi = (d * s.detach()).sum() + (s * r.detach()).sum()
    (hv_rf,) = torch.autograd.grad(psi, m)
    assert _rel(hv_rf, hv_db) < 1e-5


def test_guard_carries_the_tangent_of_a_leaf_dual():
    with fwAD.dual_level():
        x = fwAD.make_dual(torch.ones(3), torch.arange(3.0))
        (y,) = CreateGraphGuard.apply(lambda ctx: None, x)
        assert torch.equal(fwAD.unpack_dual(y).tangent, torch.arange(3.0))


@pytest.mark.parametrize("memory,what", [(Ckpt, r"memory=Ckpt\(\)"), (BoundarySaving, r"memory=BoundarySaving\(\)")],
                         ids=["ckpt", "boundary_saving"])
def test_memory_strategies_without_a_tangent_refuse_dual_models(memory, what):
    prop = _prop(Acoustic(spatial_order=4, device="cpu"), memory())
    models = _models(["vp"])
    with pytest.raises(RuntimeError, match=what + r" does not carry forward-mode tangents.*memory=Full\(\)"):
        _forward_jvp(prop, models, _directions(models))


@requires_compile
def test_compiled_step_refuses_dual_models_instead_of_dropping_the_tangent():
    prop = _prop(Acoustic(spatial_order=4, device="cpu"), Full(), use_compile=True)
    models = _models(["vp"])
    with pytest.raises(RuntimeError, match=r"torch.compile does not carry forward-mode.*use_compile=False"):
        _forward_jvp(prop, models, _directions(models))


@pytest.mark.parametrize("s_gradient,why", [("adjoint", "hand-written adjoint"),
                                             ("frozen", "frozen linearisation, not J h")])
def test_qp_adjoint_and_frozen_modes_refuse_dual_models(s_gradient, why):
    models = _models(["vp", "epsilon", "delta"])
    prop = _prop(AcousticVTI(spatial_order=4, device="cpu", s_gradient=s_gradient), Full())
    with pytest.raises(RuntimeError, match=r"AcousticVTI does not carry forward-mode.*" + why
                                           + r".*s_gradient='exact'"):
        _forward_jvp(prop, models, _directions(models))


def test_qp_exact_mode_carries_j_h():
    models = _models(["vp", "epsilon", "delta"])
    hs = _directions(models)
    prop = _prop(AcousticVTI(spatial_order=4, device="cpu", s_gradient="exact"), Full())
    _, tangent = _forward_jvp(prop, models, hs)
    assert _rel(tangent, _exact_jvp(prop, models, hs)) < 1e-4    # the double vjp: J h of the same map


def test_registered_gradient_imaging_condition_refuses_dual_models():
    prop = _prop(Acoustic(spatial_order=4, device="cpu"), Full())
    prop.register_gradient("vp", lambda fwd, adj, m, dt, h: fwd["h1"] * adj["h1"], mode="gradient")
    models = _models(["vp"])
    with pytest.raises(RuntimeError, match=r"imaging condition does not carry forward-mode.*clear_gradients"):
        _forward_jvp(prop, models, _directions(models))


def test_propsem_refuses_dual_models():
    from sweep.sem import AcousticSEM2D, PropSEM
    prop = PropSEM(AcousticSEM2D, shape=(25, 33), dh=10.0, dt=6e-4, n_order=4, f_dominant=15.0,
                   impl="eager", abcn=8)
    t = np.arange(120) * 6e-4
    a = (np.pi * 15.0 * (t - 0.08)) ** 2
    w = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"PropSEM does not carry forward-mode"):
        with torch.no_grad(), fwAD.dual_level():
            vp = fwAD.make_dual(torch.full((25, 33), 2000.0), torch.ones(25, 33))
            prop(w, np.array([[16, 8]]), np.array([[[ix, 4] for ix in range(4, 29, 4)]]), models=[vp])


@requires_binding()
def test_impl_c_refuses_dual_models():
    prop = _prop(Acoustic(spatial_order=4, device="cuda"), Full(), impl="c", dev="cuda")
    models = _models(["vp"], device="cuda")
    with pytest.raises(RuntimeError, match=r"impl='c' does not carry forward-mode"):
        _forward_jvp(prop, models, _directions(models))
