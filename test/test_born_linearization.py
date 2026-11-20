"""AcousticBorn is the linearisation of Acoustic: its scattered record for
mp = 2 h / vp is J h, the tangent of the Acoustic record along h.

Acoustic takes spatial_order first-derivative kernels in its CPML
(``init_grad_kernels``) while AcousticBorn took ``torch.gradient``, so the two
were different discretisations: J h was off by ~1e-2 in a thin PML, and every
Born route to a Gauss-Newton or Hessian product inherited it.  The Born pair
now builds the same kernels, on every device.
"""
import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fwAD

from sweep.equations import Acoustic, AcousticBorn
from sweep.propagator import Full
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
NT = 300
SRC = np.array([[14, 6]])
REC = np.array([[[ix, 2] for ix in range(2, 26, 2)]])


def _wavelet(device):
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.06
    a = (np.pi * 15.0 * t) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=device)


def _prop(equation, receiver, device):
    return PropTorch(equation, shape=SHAPE, dh=10.0, dt=1e-3, nt=NT, abcn=8, dev=device, impl="eager",
                     memory=Full(), source_type=["h1"], receiver_type=[receiver],
                     eager_options=EagerOptions(use_compile=False))


def _check(device):
    acoustic = Acoustic(spatial_order=4, device=device)
    born = AcousticBorn(spatial_order=4, device=device)
    assert acoustic.grad_kernels is not None and born.grad_kernels is not None
    for axis, kernel in acoustic.grad_kernels.items():
        assert torch.equal(born.grad_kernels[axis], kernel)
    z = torch.arange(SHAPE[0], dtype=torch.float32, device=device)[:, None] * torch.ones(SHAPE, device=device)
    vp = 2000.0 + 8.0 * z
    ring = torch.zeros(SHAPE, device=device)
    ring[2:-2, 2:-2] = 1.0           # off the outer cells, whose gradient is cropped (EdgePadding)
    h = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0)).to(device) * ring * 2.0
    w = _wavelet(device)
    with torch.no_grad(), fwAD.dual_level():
        jh = fwAD.unpack_dual(_prop(acoustic, "h1", device)(w, SRC, REC, models=[fwAD.make_dual(vp, h)])).tangent
    with torch.no_grad():
        scattered = _prop(born, "sh1", device)(w, SRC, REC, models=[vp, 2 * h / vp])
    return float((scattered - jh).norm() / jh.norm())


def test_born_record_is_the_tangent_of_the_acoustic_record():
    assert _check("cpu") < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_born_record_is_the_tangent_of_the_acoustic_record_on_cuda():
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        assert _check("cuda") < 1e-5
    finally:
        torch.backends.cudnn.allow_tf32 = prev
