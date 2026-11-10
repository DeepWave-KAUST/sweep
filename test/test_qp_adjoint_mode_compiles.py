"""The qP equations' default ``s_gradient='adjoint'`` gradient compiles.

Its backward used to re-run the update under ``requires_grad_()`` +
``torch.autograd.grad``, which torch.compile cannot trace: Dynamo broke the
graph there, and the whole backward of a compiled step ran uncompiled (the
eager qP gradients took 12-32x their forward). ``aot_eager`` traces what
Inductor traces, without a C++ compiler.
"""
import numpy as np
import pytest
import torch
from torch._dynamo.utils import counters

from sweep.equations import AcousticTTI, AcousticTTIBorn, AcousticVTI, AcousticVTIBorn
from sweep.propagator import Full
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

SHAPE, ABCN, NT, DT, DH = (24, 28), 8, 24, 0.0015, 10.0


def _models(names):
    nz, nx = SHAPE
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
    bump = np.exp(-(((z - 0.55) / 0.2) ** 2 + ((x - 0.45) / 0.25) ** 2))
    vals = {"vp": 2000.0 + 400.0 * z + 200.0 * bump, "epsilon": 0.2 + 0.05 * bump,
            "delta": 0.06 + 0.02 * x, "theta": 0.35 + 0.3 * x, "mp": 0.1 * bump}
    return [np.ascontiguousarray(vals[n] + 0 * z + 0 * x, dtype=np.float32) for n in names]


def _shoot(prop, models):
    nz, nx = SHAPE
    src = np.array([[nx // 2, nz // 4]], dtype=np.int64)
    rx = np.arange(3, nx - 3, 3, dtype=np.int64)
    rec = np.stack([rx, np.full_like(rx, 2)], -1)[None]
    t = np.arange(NT) * DT - 0.02
    a = (np.pi * 25.0 * t) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))
    return prop(wav, src, rec, models=models)


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI, AcousticVTIBorn, AcousticTTIBorn])
def test_the_adjoint_mode_gradient_compiles_without_a_graph_break(cls):
    torch._dynamo.reset()
    counters.clear()
    eq = cls(spatial_order=4, device="cpu", backend="torch", s_gradient="adjoint")
    prop = PropTorch(eq, backend="torch", impl="eager", shape=SHAPE, abcn=ABCN, dh=DH, dt=DT, dev="cpu",
                     nt=NT, B=1, memory=Full(),
                     eager_options=EagerOptions(use_compile=True, compile_backend="aot_eager"))
    ms = [torch.tensor(a, requires_grad=True) for a in _models(eq.models)]
    _shoot(prop, ms).pow(2).sum().backward()
    assert prop._backend_impl.use_compile                  # it did not fall back to uncompiled
    assert not counters["graph_break"], dict(counters["graph_break"])
    for name, m in zip(eq.models, ms):
        assert m.grad is not None and bool(torch.isfinite(m.grad).all()), name
        assert float(m.grad.abs().max()) > 0, name
