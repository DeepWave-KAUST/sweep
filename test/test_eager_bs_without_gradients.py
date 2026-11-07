"""Eager boundary saving on a call autograd does not record.

Such a call (grad mode off, or no model or wavelet requiring grad) has no
backward pass, so there is no boundary to save: under ``BoundarySaving()`` it
runs the plain forward steps, as ``Full()`` does and as ``impl='c'`` skips its
boundary, and records what ``Full()`` records bit for bit.  It used to save
every step's ring into storage sized for the whole rollout.
"""
import numpy as np
import pytest
import torch

import sweep.propagator._eager_boundary_saving as EBS
from conftest import requires_compile
from sweep.equations import Acoustic, Elastic
from sweep.propagator import BoundarySaving, Full
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
SRC = np.array([[10, 4], [18, 5]])
REC_XZ = np.repeat(np.array([[[ix, 3] for ix in range(2, 26, 3)]]), 2, axis=0)
NT = 41


def _wavelet(nt=NT):
    t = np.arange(nt, dtype=np.float32) * 1e-3 - 0.012
    a = (np.pi * 30.0 * t) ** 2
    w = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)
    return np.stack([w, 0.7 * np.roll(w, 3)])                      # one per shot


def _prop(eq, memory, **eager):
    kw = dict(source_type=["szz"], receiver_type=["vz", "vx"]) if eq is Elastic else {}
    return PropTorch(eq(spatial_order=4, device="cpu"), shape=SHAPE, dh=10.0, dt=1e-3,
                     nt=NT, abcn=6, dev="cpu", impl="eager", memory=memory,
                     eager_options=EagerOptions(**eager), **kw)


def _models(eq, grad):
    ramp = torch.linspace(0, 1, SHAPE[0]).view(-1, 1)
    values = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0}
    return [(values[n] * (1 + 0.05 * ramp)).expand(SHAPE).clone().requires_grad_(grad)
            for n in eq(spatial_order=4, device="cpu").models]


def _counting(monkeypatch):
    """Count what boundary saving does: forward calls of its autograd
    Function, ring rows saved, ring storages allocated."""
    counts = {"calls": 0, "saved": 0, "rings": 0}
    step = EBS._BoundarySaveStep.forward

    def step_counted(*args):
        counts["calls"] += 1
        return step(*args)

    monkeypatch.setattr(EBS._BoundarySaveStep, "forward", staticmethod(step_counted))
    save = EBS.save_step
    monkeypatch.setattr(EBS, "save_step",
                        lambda *a, **kw: counts.__setitem__("saved", counts["saved"] + 1) or save(*a, **kw))
    state = EBS.ReconState
    monkeypatch.setattr(EBS, "ReconState",
                        lambda *a, **kw: counts.__setitem__("rings", counts["rings"] + 1) or state(*a, **kw))
    return counts


CALLS = {
    # grad mode off, the models requiring grad: a line search over them
    "no_grad_line_search": (False, True, False),
    # grad mode off, nothing requiring grad
    "no_grad": (False, False, False),
    # grad mode on, nothing requiring grad: forward modelling
    "nothing_requires_grad": (True, False, False),
}


def _compile_options(compiled):
    if compiled:
        return dict(use_compile=True, compile_backend="aot_eager")
    return dict(use_compile=False)


@pytest.mark.parametrize("compiled", [False, pytest.param(True, marks=requires_compile)],
                         ids=["uncompiled", "compiled"])
@pytest.mark.parametrize("eq", [Acoustic, Elastic], ids=["acoustic", "elastic"])
@pytest.mark.parametrize("call", list(CALLS))
def test_a_call_without_gradients_saves_no_boundary(monkeypatch, call, eq, compiled):
    grad_mode, models_grad, wavelet_grad = CALLS[call]
    counts = _counting(monkeypatch)
    opts = _compile_options(compiled)
    bs = _prop(eq, BoundarySaving(storage="gpu"), **opts)
    full = _prop(eq, Full(), **opts)
    models = _models(eq, models_grad)
    wav = torch.tensor(_wavelet(), requires_grad=wavelet_grad)
    with torch.set_grad_enabled(grad_mode):
        out = bs(wav, SRC, REC_XZ, models=models)
        ref = full(wav, SRC, REC_XZ, models=models)
    assert counts == {"calls": 0, "saved": 0, "rings": 0}
    assert not out.requires_grad
    assert torch.equal(out, ref) and out.abs().max() > 0


@pytest.mark.parametrize("compiled", [False, pytest.param(True, marks=requires_compile)],
                         ids=["uncompiled", "compiled"])
def test_the_same_propagator_saves_the_boundary_for_a_gradient(monkeypatch, compiled):
    """A call without gradients leaves no state behind: the next call with
    gradients saves its boundary and gets the gradient it got before."""
    counts = _counting(monkeypatch)
    bs = _prop(Acoustic, BoundarySaving(storage="gpu"), **_compile_options(compiled))

    def gradient():
        models = _models(Acoustic, True)
        wav = torch.tensor(_wavelet(), requires_grad=True)
        bs(wav, SRC, REC_XZ, models=models).square().sum().backward()
        return [m.grad for m in models] + [wav.grad]

    before = gradient()
    first = dict(counts)
    assert first["calls"] > 0 and first["saved"] > 0 and first["rings"] == 1
    with torch.no_grad():
        bs(torch.tensor(_wavelet()), SRC, REC_XZ, models=_models(Acoustic, True))
    assert counts == first
    after = gradient()
    assert counts == {name: 2 * n for name, n in first.items()}
    for a, b in zip(after, before):
        assert torch.equal(a, b)
