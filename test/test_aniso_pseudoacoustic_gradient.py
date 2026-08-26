"""qP VTI/TTI gradients must exist. They used to be 100% NaN.

`AcousticVTI` and `AcousticTTI` computed their anisotropy weight as

    sk = numerator * ((denominator + 1e-26) ** -1)

which is finite in the forward and catastrophic in the backward. Autograd's
`pow` backward evaluates `x ** -2`; in fp32, `(1e-26) ** -2` overflows to `+inf`.
Wherever the numerator is also zero -- the entire grid at t=0, the quiet region
ahead of the wavefront, the PML corners -- the chain rule computes `0 * inf`,
and one NaN poisons every cell of the model gradient.

It went unnoticed because nothing ever took a gradient through these two: the
tests that touch them only construct them, and the VTI notebook runs under
`torch.no_grad()`. The forward was fine, so nothing looked wrong.

fp64 is not an escape: the eager propagator casts models and wavefields to
float32, so a float64 model still returns a float32 record and still NaNs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

NZ, NX, NT, DT, DH = 48, 56, 200, 1.5e-3, 10.0


def _run(name, grad_on=0):
    import sweep.equations as E
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    dev = torch.device("cuda")
    eq = getattr(E, name)(spatial_order=4, device=dev, backend="torch")
    prop = PropTorch(eq, backend="torch", impl="eager",
                     eager_options=EagerOptions(use_compile=False),
                     shape=(NZ, NX), dev=dev, dh=DH, dt=DT,
                     source_type=["h1"], receiver_type=["h1"],
                     abcn=30, pml_type="cpmlr", nt=NT, B=1,
                     allow_growth=True, use_ckpt=False)

    z = np.arange(NZ)[:, None] * np.ones((1, NX))
    vp = (1800.0 + 600.0 * z / (NZ - 1)).astype(np.float32)
    models = [torch.tensor(vp, device=dev),
              torch.full((NZ, NX), 0.05, device=dev),   # epsilon, near-isotropic
              torch.full((NZ, NX), 0.03, device=dev)]   # delta < epsilon: eta > 0
    if name == "AcousticTTI":
        models.append(torch.zeros((NZ, NX), device=dev))   # theta = 0
    models[grad_on].requires_grad_(True)

    t = torch.arange(NT, dtype=torch.float32, device=dev) * DT - 0.06
    wav = ((1 - 2 * (np.pi * 15 * t) ** 2) * torch.exp(-(np.pi * 15 * t) ** 2)).reshape(NT)
    src = np.array([[NX // 2, 6]], dtype=np.int32)
    rx = np.arange(6, NX - 6, dtype=np.int32)
    rec = np.stack([rx, np.full_like(rx, 6)], axis=-1)[None, ...]

    out = prop(wav, src, rec, models=models)
    out = out[0] if isinstance(out, (tuple, list)) else out
    loss = out.pow(2).mean()
    loss.backward()
    torch.cuda.synchronize(dev)
    return out.detach(), float(loss.detach()), models[grad_on].grad


@pytest.mark.parametrize("name", ["AcousticVTI", "AcousticTTI"])
def test_model_gradient_is_finite(name):
    rec, loss, grad = _run(name)
    assert torch.isfinite(rec).all(), "forward record is not finite"
    assert grad is not None, "no gradient reached vp"
    n_nan = int(torch.isnan(grad).sum())
    assert n_nan == 0, f"{name}: dJ/dvp is NaN in {n_nan}/{grad.numel()} cells"
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0, "gradient is identically zero"


@pytest.mark.parametrize("name", ["AcousticVTI", "AcousticTTI"])
@pytest.mark.parametrize("grad_on", [1, 2])
def test_anisotropy_parameter_gradients_are_finite(name, grad_on):
    """epsilon and delta flow through `sk` directly, which is where the
    overflow was -- so they are the most exposed of all."""
    _, _, grad = _run(name, grad_on=grad_on)
    assert grad is not None and torch.isfinite(grad).all()


def test_tilt_zero_tti_reproduces_vti():
    """A sanity check on the fix: with theta = 0 the TTI weight reduces to the
    VTI one, so both must give the same loss."""
    _, loss_vti, _ = _run("AcousticVTI")
    _, loss_tti, _ = _run("AcousticTTI")
    assert loss_vti == pytest.approx(loss_tti, rel=1e-6)


def test_the_reciprocal_power_form_is_what_overflowed():
    """Pins the MECHANISM, so the fix cannot be reverted as a style preference.

    Nothing about the propagator is needed to show it: in fp32 the pow backward
    needs den**-2, which is +inf for the 1e-26 floor, and 0 * inf is NaN.
    """
    num = torch.zeros(4, requires_grad=True)
    den = torch.zeros(4, requires_grad=True)
    (num * ((den + 1e-26) ** -1)).sum().backward()
    assert torch.isnan(den.grad).all(), "the reciprocal-power form no longer overflows?"

    num2 = torch.zeros(4, requires_grad=True)
    den2 = torch.zeros(4, requires_grad=True)
    (num2 / (den2 + 1e-26)).sum().backward()
    assert torch.isfinite(den2.grad).all(), "the division form must stay finite"
