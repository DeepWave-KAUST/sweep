"""A dropped eager boundary-saving propagator must free its memory.

With ``use_compile=True`` the compiled step held the propagator strongly and was
stored in every ``_BoundarySaveStep`` ctx; the propagator's cached record carries
a ``grad_fn`` into that graph.  That closed a reference cycle through the C++
autograd graph, which gc cannot see, so every propagator built this way kept its
last rollout's boundary ring and workspace on the device after it was dropped
(26-42 MiB per instance at the 25 m Marmousi size of the memory notebook).
"""
import gc
import weakref

import numpy as np
import pytest
import torch

from conftest import requires_compile
from sweep.equations import Acoustic
from sweep.propagator import BoundarySaving
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHAPE, NT = (40, 48), 60
WAVELET = np.zeros(NT, dtype=np.float32)
WAVELET[5] = 1.0
SRC = np.array([[24, 4]])
REC = np.array([[[ix, 4] for ix in range(0, 48, 4)]])


def _solver(use_compile):
    # aot_eager: compiled, but needs no C++ toolchain on a CPU-only runner.
    return PropTorch(Acoustic(device=DEV), shape=SHAPE, dh=10.0, dt=1e-3, dev=DEV, impl="eager",
                     nt=NT, abcn=8, memory=BoundarySaving(storage="gpu"),
                     eager_options=EagerOptions(use_compile=use_compile, compile_backend="aot_eager"))


def _forward(solver):
    vp = torch.full(SHAPE, 2000.0, device=DEV, requires_grad=True)
    return solver(WAVELET, SRC.copy(), REC.copy(), models=[vp])


@pytest.mark.parametrize("use_compile", [False, pytest.param(True, marks=requires_compile)], ids=["uncompiled", "compiled"])
def test_dropped_propagator_is_freed(use_compile):
    gc.collect()
    base = torch.cuda.memory_allocated() if DEV.type == "cuda" else 0
    solver = _solver(use_compile)
    _forward(solver).square().sum().backward()
    # The eager implementation behind the PropTorch facade is what holds the
    # workspace and what the compiled step captured.
    ref = weakref.ref(solver._backend_impl)
    del solver
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    gc.collect()
    assert ref() is None, "the eager propagator outlived its last reference"
    if DEV.type == "cuda":
        assert torch.cuda.memory_allocated() - base < 2**20


def test_second_backward_of_one_rollout_is_refused():
    out = _forward(_solver(False))
    out.square().sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="one backward pass per forward"):
        out.square().sum().backward()
