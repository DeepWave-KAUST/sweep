"""``ModelParallel`` refuses a wavelet that asks for its gradient.

The domain-decomposed backward returns the model gradients only, for every
equation: ``_DDForward.backward`` hands autograd ``None`` for the wavelet.
Until this check a DD run left ``wavelet.grad`` at None without a word -- even
for ``Acoustic``, whose single-domain ``impl='c'`` backward does return one.
The refusal fires at the top of ``forward``, before the one-time capture.
(world=1: the dd_propagator path without NCCL, as in test_dd_tail_two_tile.py.)
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.parallel import MeshTopology
from sweep.propagator.options import BoundarySaving
from sweep.propagator.torch import PropTorch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="ModelParallel builds CUDA tile solvers")

SHAPE, DH, DT, NT, ABCN = (48, 51), 10.0, 6e-4, 120, 8


def _ddp():
    from sweep.parallel import ModelParallel
    dev = torch.device("cuda:0")
    prop = PropTorch(Acoustic(spatial_order=4, device=dev, backend="torch"), backend="torch",
                     impl="c", shape=SHAPE, dh=DH, dt=DT, nt=NT, abcn=ABCN, dev=dev,
                     memory=BoundarySaving())
    return ModelParallel(prop, MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0))


def _inputs(wavelet_grad, vp_grad):
    dev = torch.device("cuda:0")
    t = np.arange(NT, dtype=np.float32) * DT - 0.04
    a = (np.pi * 20.0 * t) ** 2
    wav = torch.tensor((1 - 2 * a) * np.exp(-a), device=dev, requires_grad=wavelet_grad)
    vp = torch.full(SHAPE, 2000.0, device=dev, requires_grad=vp_grad)
    src = np.array([[[SHAPE[1] // 2, 4]]], np.int32)
    rec = np.array([[[x, 4] for x in range(2, SHAPE[1] - 2, 4)]], np.int32)
    return wav, src, rec, vp


@pytest.mark.parametrize("vp_grad", [True, False])
def test_dd_refuses_the_wavelet_gradient(vp_grad):
    wav, src, rec, vp = _inputs(True, vp_grad)
    with pytest.raises(NotImplementedError, match="wavelet gradient"):
        _ddp()(wav, src, rec, models=[vp])


def test_dd_model_gradient_without_the_wavelet():
    """The control: a wavelet that asks for nothing still gets the model
    gradient, and under no_grad a requires_grad wavelet is not refused --
    nothing is asked of the backward there."""
    ddp = _ddp()
    wav, src, rec, vp = _inputs(False, True)
    (ddp(wav, src, rec, models=[vp]).double() ** 2).sum().backward()
    assert wav.grad is None
    assert torch.isfinite(vp.grad).all() and float(vp.grad.abs().max()) > 0
    wav, _, _, vp = _inputs(True, True)
    with torch.no_grad():
        ddp(wav, src, rec, models=[vp])
