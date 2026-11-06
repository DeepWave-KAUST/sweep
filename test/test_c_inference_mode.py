"""impl='c' runs under ``torch.inference_mode()``.

Any call there failed in the C adapter before the forward ran, with "Inference
tensors do not track version counter": the host copies of the index tensors the
drivers read (``source_field_indices``, ``receiver_field_indices``,
``checkpoint_steps``) are cached per params object while the tensor keeps its
identity and its ``_version``, and ``_CompiledPropagator.forward`` builds those
tensors per call -- under inference_mode they are inference tensors, which have
no version counter, and reading ``_version`` raises. They are now copied on
every call and never cached, so an inference_mode call runs the forward-only
path a ``torch.no_grad()`` call runs.
"""
import numpy as np
import pytest
import torch

from conftest import requires_binding, ricker
from gradient_cases import model_arrays

from sweep.equations import Acoustic
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch

pytestmark = requires_binding("acoustic2d_forward")

SHAPE, NT, DT, DH, ABCN = (40, 48), 120, 1e-3, 10.0, 10
MEMORIES = {"bs": BoundarySaving, "full": Full, "ckpt": lambda: Ckpt(mode="chunk", chunks=4)}

XS = np.array([12, 36])
SRC = np.stack([XS, np.full_like(XS, 4)], -1)
RX = np.arange(4, SHAPE[1] - 4, 4)
REC = np.stack([np.stack([RX, np.ones_like(RX)], -1)] * len(XS))


def _prop(memory):
    eq = Acoustic(spatial_order=4, device="cuda", backend="torch")
    prop = PropTorch(eq, backend="torch", impl="c", shape=SHAPE, dh=DH, dt=DT, nt=NT,
                     abcn=ABCN, dev="cuda", memory=MEMORIES[memory]())
    assert prop.impl == "c"
    return prop


def _inputs(requires_grad=False):
    wav = torch.tensor(ricker(NT, DT, fm=15.0, delay=0.06), device="cuda")
    vp = torch.tensor(model_arrays(["vp"], SHAPE)[0], device="cuda",
                      requires_grad=requires_grad)
    return wav, vp


def _no_grad_record(memory):
    wav, vp = _inputs()
    with torch.no_grad():
        return _prop(memory)(wav, SRC, REC, models=[vp])


def _inference_record(prop):
    with torch.inference_mode():
        wav, vp = _inputs()
        out = prop(wav, SRC, REC, models=[vp])
    assert out.is_inference()
    return out


def _grad_call(prop):
    wav, vp = _inputs(requires_grad=True)
    out = prop(wav, SRC, REC, models=[vp])
    out.pow(2).sum().backward()
    return out.detach(), vp.grad


@pytest.mark.parametrize("memory", MEMORIES)
def test_inference_mode_forward_is_the_no_grad_forward(memory):
    """Inputs made inside inference_mode, as in an inference-only script."""
    ref = _no_grad_record(memory)
    assert torch.equal(_inference_record(_prop(memory)), ref)


@pytest.mark.parametrize("memory", MEMORIES)
def test_inference_mode_on_a_model_that_requires_grad(memory):
    """The model being inverted (made outside, requires_grad) evaluated under
    inference_mode, e.g. for QC: nothing is differentiated."""
    ref = _no_grad_record(memory)
    wav, vp = _inputs(requires_grad=True)
    with torch.inference_mode():
        out = _prop(memory)(wav, SRC, REC, models=[vp])
    assert torch.equal(out, ref)


@pytest.mark.parametrize("memory", MEMORIES)
def test_differentiable_calls_around_inference_mode_calls(memory):
    """inference_mode, grad, inference_mode, grad on one propagator. An
    inference tensor kept in the propagator's state would fail a later
    differentiable call (no in-place write outside inference_mode, no
    save_for_backward); a grad call that left something behind would change
    the next record. Each record and gradient is a fresh propagator's."""
    ref = _no_grad_record(memory)
    ref_out, ref_grad = _grad_call(_prop(memory))
    assert float(ref_grad.abs().max()) > 0
    prop = _prop(memory)
    for _ in range(2):
        assert torch.equal(_inference_record(prop), ref)
        out, grad = _grad_call(prop)
        assert torch.equal(out, ref_out) and torch.equal(grad, ref_grad)
