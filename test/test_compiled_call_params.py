"""The hazard `CompiledCallParams` introduces, and the one it removes.

Collapsing 51 positional arguments into one object is safe for everything that
was already returning `None` from `backward`. It creates exactly one new way to
be wrong, and it is silent:

    a tensor that requires grad, placed in a field of CompiledCallParams,
    is invisible to autograd and receives no gradient -- no error, no warning.

The bit-exact gate cannot see this. It compares this build against a recorded
baseline, and a tensor that never carried a gradient in EITHER build produces
identical numbers in both. Only an assertion about the call itself catches it,
which is what these tests are.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="the compiled path needs CUDA")

NZ, NX, NT = 48, 56, 120


def _run_once(capture: list):
    """One compiled forward+backward, capturing what reached ``Warpper.apply``."""
    from sweep.equations.acoustic import Acoustic
    from sweep.propagator.torch import PropTorch
    from sweep.propagator import _c as c_mod

    dev = torch.device("cuda")
    eq = Acoustic(spatial_order=4, device=dev, backend="torch")
    prop = PropTorch(eq, backend="torch", impl="c", shape=(NZ, NX), dev=dev,
                     dh=10.0, dt=0.001, source_type=["p"], receiver_type=["p"],
                     abcn=10, nt=NT, B=1, allow_growth=True, use_ckpt=False,
                     boundary_saving_config={"enabled": False})

    orig_apply = c_mod.Warpper.apply

    def spy(*args, **kwargs):
        capture.append(args)
        return orig_apply(*args, **kwargs)

    t = torch.arange(NT, dtype=torch.float32, device=dev) * 0.001 - 0.06
    # Geometry is numpy and ordered (x, z) -- see solver_gradient_mode_suite.
    # sources (nshots, 2) with a (nt,) wavelet is the naive single-shot form.
    wavelet = ((1 - 2 * (np.pi * 15 * t) ** 2) * torch.exp(-(np.pi * 15 * t) ** 2)
               ).reshape(NT).clone().requires_grad_(True)
    sources = np.array([[NX // 2, NZ // 4]], dtype=np.int32)
    rec_x = np.arange(5, NX - 5, dtype=np.int32)
    receivers = np.stack([rec_x, np.full_like(rec_x, NZ // 4)], axis=-1)[None, ...]
    vp = torch.full((1, NZ, NX), 2000.0, device=dev).requires_grad_(True)

    c_mod.Warpper.apply = staticmethod(spy)
    try:
        rec = prop(wavelet, sources, receivers, models=[vp])
        rec = rec[0] if isinstance(rec, (tuple, list)) else rec
        rec.pow(2).mean().backward()
        torch.cuda.synchronize(dev)
    finally:
        c_mod.Warpper.apply = orig_apply
    return wavelet, vp


def test_no_differentiable_tensor_hides_in_the_params_object():
    """The new failure mode, asserted directly.

    A tensor needing a gradient must be a positional input to the autograd
    Function. If one is ever moved into `CompiledCallParams` for tidiness, its
    gradient vanishes without a trace -- so refuse the arrangement here.
    """
    capture: list = []
    _run_once(capture)
    assert capture, "Warpper.apply was never called -- the test proves nothing"
    params = capture[0][0]

    offenders = []
    for name in type(params).__dataclass_fields__:
        val = getattr(params, name)
        for t in (val if isinstance(val, (tuple, list)) else [val]):
            if isinstance(t, torch.Tensor) and t.requires_grad:
                offenders.append(name)
    assert not offenders, (
        f"these CompiledCallParams fields carry requires_grad tensors and will "
        f"silently receive no gradient: {sorted(set(offenders))}")


def test_only_the_wavelet_and_models_are_positional():
    """Guards the collapse itself.

    ``backward`` now returns ``(None, wavelet_grad, *model_grads)``. That is
    correct only while the positional inputs are exactly params, wavelet, and
    the models -- so pin the arity rather than trusting a future edit to notice.
    """
    capture: list = []
    _run_once(capture)
    args = capture[0]
    assert len(args) == 3, (
        f"Warpper.apply got {len(args)} positional args, expected "
        f"(params, wavelet, *models) with one model")
    assert type(args[0]).__name__ == "CompiledCallParams"
    assert isinstance(args[1], torch.Tensor) and args[1].requires_grad


def test_gradients_still_reach_the_wavelet_and_the_model():
    """The collapse must not quietly drop either differentiable input."""
    wavelet, vp = _run_once([])
    for name, t in (("wavelet", wavelet), ("vp", vp)):
        assert t.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(t.grad).all(), f"{name} gradient has NaN/Inf"
        assert t.grad.abs().sum() > 0, f"{name} gradient is all zeros"
