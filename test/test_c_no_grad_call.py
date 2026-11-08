"""Under ``torch.no_grad()`` an impl='c' call prepares no backward, whatever the
inputs' ``requires_grad`` says.

``requires_grad`` is a tensor property, blind to grad mode. A per-shot wavelet
``(nshots, nt)`` that requires grad reaches the compiled forward as a view
(``wavelet[:, None, :]``), and a view keeps ``requires_grad=True`` under no_grad;
inside ``autograd.Function.forward`` ``ctx.needs_input_grad`` does not see the
outer no_grad either (torch 2.9). So such a call was taken for a gradient call:
it saved the boundary ring and allocated the adjoint wavefields (boundary
saving) or the whole ``nt`` history (``Full()``), and refused what only a
backward needs (the wavelet gradient of AcousticVRZ / AcousticBorn, boundary
saving under ``topography=``). A line search or a QC forward on a wavelet being
inverted for paid all of it, or could not run.

The caller now decides whether the call will run a backward --
``torch.is_grad_enabled()`` and the inputs' ``requires_grad`` -- before
``Wrapper.apply`` and hands that to ``Wrapper.forward``.
"""
import numpy as np
import pytest
import torch

from conftest import capture, requires_binding, ricker
from gradient_cases import model_arrays

from sweep.equations import Acoustic, AcousticBorn, AcousticVRZ
from sweep.propagator import _c
from sweep.propagator.options import BoundarySaving, Full
from sweep.propagator.torch import PropTorch

pytestmark = requires_binding("acoustic2d_forward")

SHAPE, NT, DT, DH, ABCN = (40, 48), 120, 1e-3, 10.0, 10


def _prop(eq_cls, memory, **kw):
    eq = eq_cls(spatial_order=4, device="cuda", backend="torch")
    prop = PropTorch(eq, backend="torch", impl="c", shape=SHAPE, dh=DH, dt=DT, nt=NT, abcn=ABCN,
                     dev="cuda", memory=memory, **kw)
    assert prop.impl == "c"
    return prop


def _inputs(prop, surface=None, wavelet_grad=True):
    """Two shots with a per-shot wavelet (nshots, nt) -- the form that arrives
    as a view -- and the equation's models, none of which requires grad."""
    surf = np.zeros(SHAPE[1], np.int64) if surface is None else surface
    xs = np.array([12, 36])
    src = np.stack([xs, surf[xs] + 4], -1)
    rx = np.arange(4, SHAPE[1] - 4, 4)
    rec = np.stack([np.stack([rx, surf[rx] + 1], -1)] * len(xs))
    w = ricker(NT, DT, fm=15.0, delay=0.06)
    wav = torch.tensor(np.stack([w, 0.5 * w]), device="cuda", requires_grad=wavelet_grad)
    models = [torch.tensor(a, device="cuda") for a in model_arrays(list(prop.equation.models), SHAPE)]
    return wav, src, rec, models


def test_boundary_saving_call_without_grad_saves_nothing():
    prop = _prop(Acoustic, BoundarySaving())
    impl = prop._backend_impl
    cap = capture(prop)
    wav, src, rec, models = _inputs(prop)
    with torch.no_grad():
        out = prop(wav, src, rec, models=models)
    fp = cap["params"]
    assert not fp.use_boundary_saving and not fp.save_all_wavefields, (
        "a call that runs no backward saved the forward for one")
    assert impl.adjoint_wavefields == () and impl.boundary_gpu_full == () and impl.last_two.numel() == 0, (
        "a call that runs no backward allocated the adjoint wavefields / boundary ring")
    ref = prop(wav.detach(), src, rec, models=models)
    assert torch.equal(out, ref)

    # The control: with grad on the same call does save, so the check above is
    # not vacuous.
    out = prop(wav, src, rec, models=models)
    assert cap["params"].use_boundary_saving and impl.adjoint_wavefields
    out.pow(2).sum().backward()
    assert wav.grad is not None and float(wav.grad.abs().max()) > 0


def test_full_call_without_grad_keeps_no_history(monkeypatch):
    prop = _prop(Acoustic, Full())
    histories = []
    orig = _c._history_buffer
    monkeypatch.setattr(_c, "_history_buffer", lambda *a, **k: histories.append(a) or orig(*a, **k))
    cap = capture(prop)
    wav, src, rec, models = _inputs(prop)
    with torch.no_grad():
        prop(wav, src, rec, models=models)
    assert not cap["params"].save_all_wavefields and not histories, (
        "a call that runs no backward allocated the nt-long forward history")
    assert prop._backend_impl.forward_wavefields == ()

    prop(wav, src, rec, models=models).pow(2).sum().backward()      # the control
    assert len(histories) == 1


@pytest.mark.parametrize("eq_cls", (AcousticVRZ, AcousticBorn), ids=lambda c: c.__name__)
def test_no_grad_call_is_not_refused_for_a_wavelet_gradient(eq_cls):
    """These compiled backwards return no wavelet gradient and refuse a wavelet
    that asks for one -- which a call under no_grad does not."""
    prop = _prop(eq_cls, BoundarySaving())
    wav, src, rec, models = _inputs(prop)
    with torch.no_grad():
        out = prop(wav, src, rec, models=models)
    assert torch.isfinite(out).all() and float(out.abs().max()) > 0
    with pytest.raises(NotImplementedError, match="wavelet gradient"):
        prop(wav, src, rec, models=models)                           # the control


def test_no_grad_call_with_topography_is_not_refused():
    """Boundary saving is refused under topography= because its backward comes
    out wrong; a call under no_grad runs no backward."""
    x = np.arange(SHAPE[1], dtype=np.float32)
    surf = (5.0 * np.exp(-((x - SHAPE[1] / 2) ** 2) / 200.0)).round().astype(np.int64)
    prop = _prop(Acoustic, BoundarySaving(), topography=surf, topo_method="image")
    wav, src, rec, models = _inputs(prop, surface=surf)
    with torch.no_grad():
        prop(wav, src, rec, models=models)
    with pytest.raises(NotImplementedError, match="boundary saving cannot be combined"):
        prop(wav, src, rec, models=models)                           # the control


def test_no_grad_call_with_more_shots_after_a_gradient_call():
    """A call that saves nothing binds no boundary storage, so a no_grad call
    with more shots than the gradient call before it no longer narrows the
    persistent ring past its end ("start (0) + length (2) exceeds dimension
    size (1)") -- the batch-growth failure the skill listed as a pitfall."""
    prop = _prop(Acoustic, BoundarySaving())
    wav, src, rec, models = _inputs(prop, wavelet_grad=False)
    vp = models[0].clone().requires_grad_(True)
    prop(wav[:1], src[:1], rec[:1], models=[vp]).pow(2).sum().backward()
    with torch.no_grad():
        out = prop(wav, src, rec, models=models)
    with torch.no_grad():
        ref = _prop(Acoustic, BoundarySaving())(wav, src, rec, models=models)
    assert torch.equal(out, ref)
