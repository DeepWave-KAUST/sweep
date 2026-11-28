"""sweep.autodiff: Jacobian-, Gauss-Newton- and Hessian-vector products of a misfit.

Every route against the exact product by double backward through the uncompiled
eager step (the reference): 'forward' (reverse over forward-mode AD) to
round-off, 'difference' to its step's truncation; 'auto' takes 'forward' where
the propagator carries a tangent and 'difference' where it does not (Ckpt,
the compiled step, impl='c'), so one call works with every propagator.
"""
import numpy as np
import pytest
import torch

from conftest import requires_binding
from sweep.autodiff import (gauss_newton_vector_product, hessian_vector_product, hessian_vector_products,
                            jacobian_vector_product)
from sweep.equations import Acoustic, AcousticBorn, AcousticVTI, AcousticVTIBorn, Elastic, ElasticBorn
from sweep.propagator import BoundarySaving, Full
from sweep.propagator.options import Ckpt, EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
NT = 150
SRC = np.array([[8, 3], [19, 3]])
REC = np.repeat(np.array([[[ix, 2] for ix in range(2, 26, 2)]]), 2, axis=0)


def _wavelet(device="cpu"):
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.04
    a = (np.pi * 25.0 * t) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=device)


def _prop(equation=None, memory=None, impl="eager", dev="cpu", **eager):
    kw = dict(shape=SHAPE, dh=10.0, dt=1e-3, nt=NT, abcn=8, dev=dev, impl=impl, memory=memory or Full())
    if impl == "eager":
        kw["eager_options"] = EagerOptions(**{"use_compile": False, **eager})
    return PropTorch(equation or Acoustic(spatial_order=4, device=dev), **kw)


def _setup(device="cpu"):
    z = torch.arange(SHAPE[0], dtype=torch.float32, device=device)[:, None] * torch.ones(SHAPE, device=device)
    v0 = 2000.0 + 8.0 * z
    v_true = v0.clone()
    v_true[10:14, 10:18] += 120.0
    inner = torch.zeros(SHAPE, device=device)
    inner[1:-1, 1:-1] = 1.0            # off the outermost ring, whose gradient is cropped (EdgePadding)
    h = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0)).to(device) * inner
    return v0, v_true, h


def _rel(a, b):
    return float((a - b).norm() / b.norm())


LOSSES = {
    "l2": lambda d_obs: (lambda d: 0.5 * ((d - d_obs) ** 2).sum()),
    "cauchy": lambda d_obs: (lambda d: torch.log1p(((d - d_obs) / 0.05) ** 2).sum()),   # not quadratic
}


@pytest.mark.parametrize("loss_name", list(LOSSES))
def test_routes_agree_with_double_backward(loss_name):
    prop = _prop()
    v0, v_true, h = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES[loss_name](d_obs)
    args = (prop, _wavelet(), SRC, REC, [v0], [h], loss)
    (ref,) = hessian_vector_product(*args, route="double_backward")
    (fwd,) = hessian_vector_product(*args, route="forward")
    (auto,) = hessian_vector_product(*args)
    assert _rel(fwd, ref) < 1e-5                       # exact for any twice-differentiable loss
    assert torch.equal(auto, fwd)                      # eager, uncompiled, Full: the forward route
    if loss_name == "l2":                              # a strongly curved loss needs a smaller step
        (dif,) = hessian_vector_product(*args, route="difference")
        assert _rel(dif, ref) < 1e-2


def test_double_backward_is_the_symmetric_product_on_the_outermost_ring():
    """There the gradient is cropped (C) but the model replicated (E): double
    backward gives C H C^T, forward mode and differences C H E."""
    prop = _prop()
    v0, v_true, _ = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    ring = torch.ones(SHAPE)
    ring[1:-1, 1:-1] = 0.0
    gen = torch.Generator().manual_seed(1)
    u, v = (torch.randn(SHAPE, generator=gen) * ring for _ in range(2))
    (hu,) = hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [u], loss, route="double_backward")
    (hv,) = hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [v], loss, route="double_backward")
    vhu, uhv = float((v * hu).sum()), float((u * hv).sum())
    assert abs(vhu - uhv) <= 1e-4 * abs(uhv)
    (fu,) = hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [u], loss, route="forward")
    assert _rel(fu, hu) > 5e-2


def test_gauss_newton_is_the_double_vjp_product():
    prop = _prop()
    v0, v_true, h = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    m = v0.clone().requires_grad_(True)
    d = prop(_wavelet(), SRC, REC, models=[m])
    u = torch.zeros_like(d, requires_grad=True)
    (jtu,) = torch.autograd.grad(d, m, grad_outputs=u, create_graph=True)
    (jh,) = torch.autograd.grad((jtu * h).sum(), u)
    (ref,) = torch.autograd.grad((prop(_wavelet(), SRC, REC, models=[m]) * jh).sum(), m)   # J^T J h
    (gn,) = gauss_newton_vector_product(prop, _wavelet(), SRC, REC, [v0], [h], loss, route="forward")
    (gn_dif,) = gauss_newton_vector_product(prop, _wavelet(), SRC, REC, [v0], [h], loss, route="difference")
    assert _rel(gn, ref) < 1e-5
    assert _rel(gn_dif, ref) < 1e-2


def test_jacobian_vector_product_and_held_models():
    prop = _prop(Elastic(spatial_order=4, device="cpu"))
    z = torch.arange(SHAPE[0], dtype=torch.float32)[:, None] * torch.ones(SHAPE)
    models = [1800.0 + 8.0 * z, 1000.0 + 4.0 * z, 2000.0 + 2.0 * z]
    _, _, h = _setup()
    record, jh = jacobian_vector_product(prop, _wavelet(), SRC, REC, models, [h, None, None])
    with torch.no_grad():
        assert torch.equal(record, prop(_wavelet(), SRC, REC, models=models))
    _, jh_dif = jacobian_vector_product(prop, _wavelet(), SRC, REC, models, [h, None, None], route="difference")
    assert _rel(jh_dif, jh) < 1e-2


class _RefusesBackgroundGradient:
    """An eager propagator that refuses the gradient of its first model, as the
    impl='c' Born equations refuse that of their background models."""

    def __init__(self, prop):
        self.prop, self.equation = prop, prop.equation

    def __call__(self, wavelet, sources, receivers, models):
        if torch.is_grad_enabled() and models[0].requires_grad:
            raise NotImplementedError("the background model requires grad")
        return self.prop(wavelet, sources, receivers, models=models)


def test_a_model_held_fixed_is_not_differentiated_and_a_zero_model_steps_by_one():
    """LSRTM's normal operator: the background held fixed (None), the
    reflectivity starting at zero.  The Born record is linear in mp, so with
    an l2 loss H h along mp is J^T J h, and differences give it to round-off."""
    born = _prop(AcousticBorn(spatial_order=4, device="cpu"))
    v0, v_true, h = _setup()
    mp0 = torch.zeros(SHAPE)
    with torch.no_grad():
        d_obs = born(_wavelet(), SRC, REC, models=[v_true, 0.1 * h])
    loss = LOSSES["l2"](d_obs)
    args = (_wavelet(), SRC, REC, [v0, mp0], [None, h])
    _, jh = jacobian_vector_product(born, *args)                       # forward mode
    gn = gauss_newton_vector_product(born, *args, loss)
    held = _RefusesBackgroundGradient(born)                             # takes 'difference'
    _, jh_dif = jacobian_vector_product(held, *args)
    gn_dif = gauss_newton_vector_product(held, *args, loss)
    hv_dif = hessian_vector_product(held, *args, loss)
    assert gn[0] is None and gn_dif[0] is None and hv_dif[0] is None
    assert _rel(jh_dif, jh) < 1e-4
    assert _rel(gn_dif[1], gn[1]) < 1e-4 and _rel(hv_dif[1], gn[1]) < 1e-3
    # a zero direction keeps a model a variable: its entry is the cross term
    cross = hessian_vector_product(born, *args[:3], [v0, mp0], [torch.zeros(SHAPE), h], loss)
    assert torch.isfinite(cross[0]).all() and cross[0].abs().max() > 0


@pytest.mark.parametrize("memory", [Ckpt, BoundarySaving], ids=["ckpt", "boundary_saving"])
def test_auto_falls_back_to_differences_where_no_tangent_is_carried(memory):
    v0, v_true, h = _setup()
    reference = _prop()
    other = _prop(memory=memory())
    with torch.no_grad():
        d_obs = reference(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    (ref,) = hessian_vector_product(reference, _wavelet(), SRC, REC, [v0], [h], loss)
    (auto,) = hessian_vector_product(other, _wavelet(), SRC, REC, [v0], [h], loss)
    assert _rel(auto, ref) < 1e-2
    with pytest.raises(RuntimeError, match="forward-mode"):
        hessian_vector_product(other, _wavelet(), SRC, REC, [v0], [h], loss, route="forward")


@pytest.mark.parametrize("mode", ["adjoint", "frozen", "exact"])
def test_qp_refuses_curvature_products_and_differences(mode):
    """Where epsilon differs from delta the qP record and gradient are not
    smooth functions of the model: H h (every route), the Gauss-Newton product
    and products by differences are refused; J h runs by forward mode in
    'exact'."""
    prop = _prop(AcousticVTI(spatial_order=4, device="cpu", s_gradient=mode))
    v0, _, h = _setup()
    loss = lambda d: 0.5 * (d ** 2).sum()
    args = (prop, _wavelet(), SRC, REC, [v0, 0.1 + 0 * v0, 0.05 + 0 * v0], [h, None, None])
    for route in ("auto", "forward", "double_backward", "difference"):
        with pytest.raises(RuntimeError, match="no reliable Hessian-vector product"):
            hessian_vector_product(*args, loss, route=route)
        if route != "double_backward":
            with pytest.raises(RuntimeError, match="no reliable Gauss-Newton product"):
                gauss_newton_vector_product(*args, loss, route=route)
    with pytest.raises(RuntimeError, match="no reliable product by central differences"):
        jacobian_vector_product(*args, route="difference")
    if mode == "exact":
        _, jh = jacobian_vector_product(*args)                  # forward mode carries J h
        assert torch.isfinite(jh).all() and jh.abs().max() > 0
    else:                                                       # no tangent: 'auto' takes differences
        with pytest.raises(RuntimeError, match="no reliable product by central differences"):
            jacobian_vector_product(*args)


def test_qp_born_refuses_every_product():
    """The scattered field's S_n follows that field's own gradient: the record
    is not linear in mp, and at mp = 0 a tangent would propagate the scattered
    field elliptically, so forward mode is refused too."""
    v0, _, h = _setup()
    models = [v0, 0.1 + 0 * v0, 0.05 + 0 * v0, torch.zeros(SHAPE)]
    directions = [None, None, None, h]
    for mode in ("adjoint", "exact"):
        prop = _prop(AcousticVTIBorn(spatial_order=4, device="cpu", s_gradient=mode))
        with pytest.raises(RuntimeError, match="does not carry forward-mode|no reliable product"):
            jacobian_vector_product(prop, _wavelet(), SRC, REC, models, directions)
        with pytest.raises(RuntimeError, match="no reliable Gauss-Newton product.*Born operator"):
            gauss_newton_vector_product(prop, _wavelet(), SRC, REC, models, directions,
                                        lambda d: 0.5 * (d ** 2).sum())


def test_a_float64_direction_is_cast_to_its_model():
    """A direction built with numpy is float64; on the forward route it used to
    reach the float32 step as a float64 tangent and fail in the source
    injection."""
    prop = _prop()
    v0, v_true, h = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    _, jh = jacobian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h])
    _, jh64 = jacobian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h.double()])
    assert jh64.dtype == torch.float32 and torch.equal(jh64, jh)
    for fn in (gauss_newton_vector_product, hessian_vector_product):
        (ref,) = fn(prop, _wavelet(), SRC, REC, [v0], [h], loss)
        (out,) = fn(prop, _wavelet(), SRC, REC, [v0], [h.double()], loss)
        assert out.dtype == torch.float32 and torch.equal(out, ref)


def test_bad_inputs_are_refused():
    prop = _prop()
    v0, _, h = _setup()
    loss = lambda d: (d ** 2).sum()
    with pytest.raises(ValueError, match="one direction per model"):
        hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h, h], loss)
    with pytest.raises(ValueError, match="nothing to differentiate"):
        hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [None], loss)
    with pytest.raises(ValueError, match="nothing to differentiate"):
        hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [torch.zeros(SHAPE)], loss,
                               route="difference")
    with pytest.raises(ValueError, match="not finite"):
        hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h * float("nan")], loss,
                               route="difference")
    with pytest.raises(ValueError, match="route"):
        hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h], loss, route="newton")
    with pytest.raises(ValueError, match="route"):
        jacobian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h], route="double_backward")


@requires_binding()
def test_impl_c_with_boundary_saving_takes_differences():
    v0, v_true, h = (t.cuda() for t in _setup())
    c = _prop(impl="c", dev="cuda", memory=BoundarySaving())
    with torch.no_grad():
        d_obs = c(_wavelet("cuda"), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    (hv,) = hessian_vector_product(c, _wavelet("cuda"), SRC, REC, [v0], [h], loss)
    (hv2,) = hessian_vector_product(c, _wavelet("cuda"), SRC, REC, [v0], [h], loss, rel_step=3e-3)
    assert torch.isfinite(hv).all() and _rel(hv, hv2) < 2e-2


@requires_binding()
def test_impl_c_born_with_its_background_held_fixed():
    """impl='c' ElasticBorn returns the gradient of the perturbation only, and
    AcousticBorn under checkpointing, and they refuse a background model that
    requires grad; a held model does not."""
    v0, v_true, h = (t.cuda() for t in _setup())
    acoustic = _prop(AcousticBorn(spatial_order=4, device="cuda"), impl="c", dev="cuda", memory=Ckpt())
    with torch.no_grad():
        d_obs = acoustic(_wavelet("cuda"), SRC, REC, models=[v_true, 0.1 * h])
    loss = LOSSES["l2"](d_obs)
    args = (_wavelet("cuda"), SRC, REC, [v0, torch.zeros_like(v0)], [None, h])
    gn = gauss_newton_vector_product(acoustic, *args, loss)
    hv = hessian_vector_product(acoustic, *args, loss)
    assert gn[0] is None and hv[0] is None
    assert torch.isfinite(gn[1]).all() and _rel(hv[1], gn[1]) < 1e-2    # linear in mp: H = J^T J

    elastic = _prop(ElasticBorn(spatial_order=4, device="cuda"), impl="c", dev="cuda", memory=BoundarySaving())
    background = [v0 * 0.9, v0 * 0.5, 0 * v0 + 2000.0]
    with torch.no_grad():
        d_obs = elastic(_wavelet("cuda"), SRC, REC, models=background + [0.1 * h, 0.05 * h, 0.1 * h])
    loss = LOSSES["l2"](d_obs)
    perturbation = [torch.zeros_like(v0)] * 3
    gn = gauss_newton_vector_product(elastic, _wavelet("cuda"), SRC, REC, background + perturbation,
                                     [None, None, None, h, h, h], loss)
    assert gn[:3] == [None, None, None] and all(torch.isfinite(x).all() for x in gn[3:])


# --------------------------------------------------------------------------- #
# K directions at once (hessian_vector_products)
# --------------------------------------------------------------------------- #
def _stack(k=3, device="cpu"):
    """K directions off the outermost ring, stacked on a leading axis."""
    inner = torch.zeros(SHAPE)
    inner[1:-1, 1:-1] = 1.0
    return (torch.randn(k, *SHAPE, generator=torch.Generator().manual_seed(1)) * inner).to(device)


@pytest.mark.parametrize("route", ["double_backward", "forward", "difference"])
def test_stacked_directions_give_the_products_one_at_a_time(route):
    """The K directions run as copies of the survey in one call (a model per
    shot, the copies' shots one after another) -- the products of K calls."""
    prop = _prop()
    v0, v_true, _ = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["cauchy"](d_obs)
    hs = _stack()
    (stacked,) = hessian_vector_products(prop, _wavelet(), SRC, REC, [v0], [hs], loss, route=route)
    one = torch.stack([hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h], loss, route=route)[0]
                       for h in hs])
    assert stacked.shape == hs.shape
    assert _rel(stacked, one) < 1e-5


def test_stacked_directions_in_batches_and_with_a_model_held_fixed():
    prop = _prop(Elastic(spatial_order=4, device="cpu"))
    z = torch.arange(SHAPE[0], dtype=torch.float32)[:, None] * torch.ones(SHAPE)
    models = [1800.0 + 8.0 * z, 1000.0 + 4.0 * z, 2000.0 + 2.0 * z]
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[models[0] * 1.03, models[1], models[2]])
    loss = LOSSES["l2"](d_obs)
    hs = _stack()
    out = hessian_vector_products(prop, _wavelet(), SRC, REC, models, [hs, None, 0.5 * hs], loss,
                                  route="double_backward", batch=2)
    assert out[1] is None
    for k, h in enumerate(hs):
        one = hessian_vector_product(prop, _wavelet(), SRC, REC, models, [h, None, 0.5 * h], loss,
                                     route="double_backward")
        assert _rel(out[0][k], one[0]) < 1e-5 and _rel(out[2][k], one[2]) < 1e-5


def test_stacked_directions_run_one_call_each_where_models_are_per_shot_already():
    prop = _prop()
    v0, v_true, _ = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    v0s = v0.expand(2, *SHAPE).clone()
    hs = _stack()[:, None].expand(3, 2, *SHAPE).clone()
    (stacked,) = hessian_vector_products(prop, _wavelet(), SRC, REC, [v0s], [hs], loss, route="double_backward")
    one = torch.stack([hessian_vector_product(prop, _wavelet(), SRC, REC, [v0s], [h], loss,
                                              route="double_backward")[0] for h in hs])
    assert torch.equal(stacked, one)


def test_stacked_directions_are_checked():
    prop = _prop()
    v0, _, h = _setup()
    loss = LOSSES["l2"](torch.zeros(()))
    with pytest.raises(ValueError, match="stacked on a leading axis"):
        hessian_vector_products(prop, _wavelet(), SRC, REC, [v0], [h], loss)
    with pytest.raises(ValueError, match="an earlier one"):
        hessian_vector_products(prop, _wavelet(), SRC, REC, [v0, v0], [h[None], torch.stack([h, h])], loss)
    with pytest.raises(ValueError, match="batch must be a positive"):
        hessian_vector_products(prop, _wavelet(), SRC, REC, [v0], [h[None]], loss, batch=0)


@requires_binding()
def test_stacked_differences_on_impl_c():
    """impl='c' runs the copies' gradients in one call too (per-shot models)."""
    v0, v_true, _ = (t.cuda() for t in _setup())
    c = _prop(impl="c", dev="cuda", memory=BoundarySaving())
    with torch.no_grad():
        d_obs = c(_wavelet("cuda"), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    hs = _stack(device="cuda")
    (stacked,) = hessian_vector_products(c, _wavelet("cuda"), SRC, REC, [v0], [hs], loss)
    one = torch.stack([hessian_vector_product(c, _wavelet("cuda"), SRC, REC, [v0], [h], loss)[0] for h in hs])
    assert _rel(stacked, one) < 1e-4


def test_slice_stencils_give_the_convolutions_values(monkeypatch):
    """The double backward builds its graph with the stencils as slices
    (slice_stencils); the record and the gradient are those of the convolutions."""
    from sweep.operators import torch as ops

    calls = []
    apply_taps = ops._apply_taps
    monkeypatch.setattr(ops, "_apply_taps", lambda u, taps: calls.append(1) or apply_taps(u, taps))
    prop = _prop()
    v0, v_true, _ = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)

    def value_and_grad():
        v = v0.clone().requires_grad_(True)
        record = prop(_wavelet(), SRC, REC, models=[v])
        return record.detach(), torch.autograd.grad(loss(record), v)[0]

    d0, g0 = value_and_grad()
    assert not calls
    with ops.slice_stencils():
        d1, g1 = value_and_grad()
    assert calls
    # float32 round-off of another summation order (bit-identical on a V100); the
    # residual is 1.5 % of the record here, so the gradient's share is ~70x larger
    assert _rel(d1, d0) < 5e-6 and _rel(g1, g0) < 3e-4
    with pytest.raises(RuntimeError, match="inside"):
        with ops.slice_stencils():
            raise RuntimeError("inside")
    calls.clear()
    value_and_grad()
    assert not calls                      # off again after the exception



def test_double_backward_keeps_the_convolutions_unless_asked(monkeypatch):
    """The double backward builds its graph with the convolutions whatever the
    number of shots or directions; slice_stencils() is the caller's choice,
    and gives the same products."""
    from sweep.operators import torch as ops

    calls = []
    apply_taps = ops._apply_taps
    monkeypatch.setattr(ops, "_apply_taps", lambda u, taps: calls.append(1) or apply_taps(u, taps))
    prop = _prop()
    v0, v_true, h = _setup()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v_true])
    loss = LOSSES["l2"](d_obs)
    hessian_vector_product(prop, _wavelet(), SRC, REC, [v0], [h], loss, route="double_backward")
    (conv,) = hessian_vector_products(prop, _wavelet(), SRC, REC, [v0], [_stack()], loss, route="double_backward")
    assert not calls
    with ops.slice_stencils():
        (sliced,) = hessian_vector_products(prop, _wavelet(), SRC, REC, [v0], [_stack()], loss,
                                            route="double_backward")
    assert calls
    assert _rel(sliced, conv) < 1e-4
