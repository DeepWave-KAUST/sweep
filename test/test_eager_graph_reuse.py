"""An eager propagator runs again while a graph of an earlier call is alive.

Every rollout started from a cached zero state that the next call zeroed in
place.  A graph that had saved it -- every checkpointed chunk (Ckpt, the eager
default), the qP adjoint-mode step and Biot's flow decay under Full() -- then
failed in backward with "modified by an inplace operation", so
``out1 = prop(m1); out2 = prop(m2); out1.backward()`` and a Hessian-vector
product taken after a few more forwards did not run.  Each call now starts
from fresh zeros: the numbers are bit-identical, and the cache that pinned
``nfields x shots x grid`` for the propagator's lifetime is gone.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, AcousticVTI, Biot
from sweep.equations.biot import biot_from_rock
from sweep.propagator import Full
from sweep.propagator.options import Ckpt, EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE = (24, 28)
NT = 120
SRC = np.array([[8, 4], [19, 4]])
REC = np.repeat(np.array([[[ix, 2] for ix in range(2, 26, 2)]]), 2, axis=0)


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.04
    a = (np.pi * 25.0 * t) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))


def _prop(equation, memory, **kw):
    return PropTorch(equation, shape=SHAPE, dh=10.0, dt=1e-3, nt=NT, abcn=8, dev="cpu", impl="eager",
                     memory=memory, eager_options=EagerOptions(use_compile=False), **kw)


def _acoustic_models(shift=0.0):
    z = torch.arange(SHAPE[0], dtype=torch.float32)[:, None] * torch.ones(SHAPE)
    return [2000.0 + 8.0 * z + shift]


def _vti_models(shift=0.0):
    z = torch.arange(SHAPE[0], dtype=torch.float32)[:, None] * torch.ones(SHAPE)
    return [2000.0 + 8.0 * z + shift, 0.1 + 0 * z, 0.05 + 0 * z]


def _biot_models(shift=0.0):
    z = torch.arange(SHAPE[0], dtype=torch.float32)[:, None] * torch.ones(SHAPE)
    lower = (z > SHAPE[0] // 2).float()
    coef = biot_from_rock(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3,
                          k_d=8e9 + 6e9 * lower + shift * 1e7, mu=6e9 + 5e9 * lower, phi=0.25 - 0.1 * lower,
                          kappa=1e-13, tortuosity=2.0)
    return [c.float().expand(SHAPE).contiguous() if isinstance(c, torch.Tensor) else torch.full(SHAPE, float(c))
            for c in coef]


CASES = {
    "acoustic_full": (lambda: Acoustic(spatial_order=4, device="cpu"), Full, _acoustic_models, {}),
    "acoustic_ckpt": (lambda: Acoustic(spatial_order=4, device="cpu"), Ckpt, _acoustic_models, {}),
    "qp_vti_full": (lambda: AcousticVTI(spatial_order=4, device="cpu"), Full, _vti_models, {}),
    "biot_full": (lambda: Biot(device="cpu"), Full, _biot_models,
                  dict(source_type=["sxx", "szz"], receiver_type=["vz"])),
}


def _gradient(prop, models, interleave=0):
    ms = [m.clone().requires_grad_(True) for m in models]
    out = prop(_wavelet(), SRC, REC, models=ms)
    for k in range(interleave):          # other calls while this graph is alive
        prop(_wavelet(), SRC, REC, models=[m + (k + 1.0) for m in models])
    return torch.autograd.grad((out ** 2).sum(), ms)


@pytest.mark.parametrize("case", list(CASES))
def test_backward_after_more_forwards_is_bit_identical(case):
    make, memory, models_of, kw = CASES[case]
    prop = _prop(make(), memory(), **kw)
    models = models_of()
    lone = _gradient(prop, models)
    interleaved = _gradient(prop, models, interleave=2)
    assert all(torch.equal(a, b) for a, b in zip(lone, interleaved))


def test_hessian_vector_product_after_more_forwards_under_checkpointing():
    prop = _prop(Acoustic(spatial_order=4, device="cpu"), Ckpt())
    (v0,) = _acoustic_models()
    with torch.no_grad():
        d_obs = prop(_wavelet(), SRC, REC, models=[v0 + 30.0])
    h = torch.randn(SHAPE, generator=torch.Generator().manual_seed(0))

    def hv(interleave):
        m = v0.clone().requires_grad_(True)
        loss = 0.5 * ((prop(_wavelet(), SRC, REC, models=[m]) - d_obs) ** 2).sum()
        (g,) = torch.autograd.grad(loss, m, create_graph=True)
        for k in range(interleave):
            with torch.no_grad():
                prop(_wavelet(), SRC, REC, models=[v0 + k])
        return torch.autograd.grad((g * h).sum(), m)[0]

    assert torch.equal(hv(0), hv(3))
