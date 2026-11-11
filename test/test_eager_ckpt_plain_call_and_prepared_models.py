"""Eager chunk checkpointing: a call autograd does not record takes the plain
rollout, and the chunks take the models prepared once per call.

Both used to cost time for nothing. A checkpointing solver cut every rollout
into chunks, also without gradients; and each chunk prepared the models again,
in its forward and in its recompute. For ElasticTTI, whose preparation rotates
the stiffness tensor, a call without gradients ran 7x slower than under full
storage, and most of a gradient's time went to the preparation.

ViscoAcoustic caches its fractional grids per prepared exponent, which carries
the Q gradient. Prepared once, the exponent stayed the same object for the
whole call, so the grids, built in the first step, were built in the first
chunk's forward but read from the cache in its recompute: the backward raised
CheckpointError.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Elastic, ElasticTTI, ViscoAcoustic
from sweep.propagator import Full
from sweep.propagator.options import Ckpt, EagerOptions
from sweep.propagator.torch import PropTorch

NZ, NX, NT, CHUNKS = 24, 28, 40, 8
VALUES = {"vp": 2500.0, "vs": 1400.0, "rho": 2000.0, "vp0": 2500.0, "vs0": 1400.0, "epsilon": 0.12,
          "delta": 0.05, "gamma": 0.06, "theta": 0.4, "phi": 0.0, "Q": 50.0, "omega": 2 * np.pi * 15.0}


def _prop(cls, memory):
    # Uncompiled: what is tested is the rollout around the step, and compiling
    # the step costs seconds per propagator on a CPU runner.
    return PropTorch(cls(spatial_order=4, device="cpu"), backend="torch", impl="eager", shape=(NZ, NX),
                     dh=10.0, dt=1e-3, nt=NT, abcn=6, B=1, dev="cpu", memory=memory,
                     eager_options=EagerOptions(use_compile=False))


def _models(cls, requires_grad=True):
    z = torch.linspace(0.0, 1.0, NZ)[:, None]
    x = torch.linspace(0.0, 1.0, NX)[None, :]
    vary = 1.0 + 0.04 * z + 0.02 * x
    return [(VALUES[n] * vary).requires_grad_(requires_grad) for n in cls(spatial_order=4).models]


def _shoot(prop, models):
    src = np.array([[NX // 2, 6]], np.int64)
    rxx = np.arange(4, NX - 4, 3, np.int64)
    rec = np.stack([rxx, np.full(rxx.size, 4)], -1)[None]
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.008
    a = np.pi * 25.0 * t
    wav = torch.tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2)).astype(np.float32))
    return prop(wav, src, rec, models=models)


@pytest.mark.parametrize("cls", [Elastic, ElasticTTI])
def test_a_call_without_gradients_takes_the_plain_rollout(cls, monkeypatch):
    prop = _prop(cls, Ckpt(chunks=CHUNKS))
    impl = type(prop._backend_impl)
    chunked = []
    real = impl._rollout_ckpt
    monkeypatch.setattr(impl, "_rollout_ckpt", lambda self, *a, **k: chunked.append(1) or real(self, *a, **k))
    with torch.no_grad():
        out = _shoot(prop, _models(cls))
        ref = _shoot(_prop(cls, Full()), _models(cls))
    assert not chunked
    assert torch.equal(out, ref)
    _shoot(prop, _models(cls)).square().sum().backward()
    assert chunked == [1]


@pytest.mark.parametrize("cls", [Elastic, ElasticTTI, ViscoAcoustic])
def test_the_chunks_take_the_models_prepared_once(cls):
    prop = _prop(cls, Ckpt(chunks=CHUNKS))
    eq = prop._backend_impl.equation
    calls = []
    real = eq.prepare_models
    eq.prepare_models = lambda models: calls.append(1) or real(models)
    ms = _models(cls)
    rec = _shoot(prop, ms)
    rec.square().sum().backward()
    assert len(calls) == 1, len(calls)          # was 1 + chunks + recomputes: 11 here

    full = _models(cls)
    ref = _shoot(_prop(cls, Full()), full)
    ref.square().sum().backward()
    assert torch.equal(rec, ref)
    nonzero = 0
    for name, a, b in zip(cls(spatial_order=4).models, ms, full):
        if float(b.grad.norm()) == 0:           # e.g. ElasticTTI's gamma, an SH parameter in 2-D
            assert float(a.grad.abs().max()) == 0, name
            continue
        nonzero += 1
        assert float((a.grad - b.grad).norm() / b.grad.norm()) < 1e-5, name
    assert nonzero >= 3
