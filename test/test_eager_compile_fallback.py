"""A compiler that cannot build the eager step costs speed, not the run.

``EagerOptions.use_compile`` defaults to True, so a plain CPU forward goes
through torch.compile, and Inductor's CPU backend shells out to g++. With g++ 9
(Ubuntu 20.04) it fails on ``-std=c++20`` and the very first forward raised
``InductorError: CppCompileError``. The step now falls back to running
uncompiled, warns once, and returns exactly what ``use_compile=False`` does.
"""
import warnings

import numpy as np
import pytest
import torch
from torch._inductor.exc import CppCompileError

from sweep.equations import Acoustic
from sweep.propagator import BoundarySaving
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

DEV = torch.device("cpu")
SHAPE = (40, 48)
WAVELET = np.zeros(120, dtype=np.float32)
WAVELET[5] = 1.0
SRC = np.array([[20, 4]])
REC = np.array([[[ix, 4] for ix in range(0, 48, 4)]])


def _compile_that_cannot_build(calls):
    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            raise CppCompileError(["g++"], "g++: error: unrecognized command line option '-std=c++20'")
        return compiled
    return fake_compile


def _run(use_compile, memory=None):
    kw = {} if memory is None else {"memory": memory}
    solver = PropTorch(Acoustic(device=DEV), shape=SHAPE, dh=10.0, dt=1e-3, dev=DEV, impl="eager",
                       eager_options=EagerOptions(use_compile=use_compile), **kw)
    vp = torch.full(SHAPE, 2000.0, requires_grad=True)
    out = solver(WAVELET, SRC, REC, models=[vp])
    out.square().sum().backward()
    return solver, out.detach(), vp.grad


@pytest.mark.parametrize("memory", [None, BoundarySaving()], ids=["full", "boundary_saving"])
def test_falls_back_to_the_uncompiled_step(monkeypatch, memory):
    _, ref_out, ref_grad = _run(False, memory)

    calls = []
    monkeypatch.setattr(torch, "compile", _compile_that_cannot_build(calls))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver, out, grad = _run(True, memory)

    assert calls == [1], "the compiled step is tried once, then never again"
    assert solver.use_compile is False
    fallback = [w for w in caught if "torch.compile could not build" in str(w.message)]
    assert len(fallback) == 1 and issubclass(fallback[0].category, RuntimeWarning)
    assert "CppCompileError" in str(fallback[0].message)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


def test_an_error_in_the_step_itself_is_not_swallowed(monkeypatch):
    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            raise ValueError("bad model")
        return compiled
    monkeypatch.setattr(torch, "compile", fake_compile)
    with pytest.raises(ValueError, match="bad model"):
        _run(True)
