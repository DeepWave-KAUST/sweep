"""The compile modes that replay the compiled eager step as CUDA graphs
(``reduce-overhead``, ``max-autotune``) are refused on a CUDA device.

A replay writes its outputs to the buffers of the replay before, which the time
loop still holds: the records of the earlier calls until the rollout ends, and
the fields a call passes on.  Every rollout read them back and failed ("accessing
tensor output of CUDAGraphs that has been overwritten by a subsequent run"),
under ``Full()`` and ``Ckpt()`` with and without gradients and under boundary
saving without them.  On a CPU there are no CUDA graphs to replay.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator import BoundarySaving, Ckpt, Full
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

from conftest import requires_compile

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
SHAPE, NT = (24, 20), 16


def _solver(dev, memory=Full, **eager):
    return PropTorch(Acoustic(4, device=dev), shape=SHAPE, dh=10.0, dt=1e-3, nt=NT, dev=dev, impl="eager",
                     abcn=4, memory=memory(), eager_options=EagerOptions(**eager))


@requires_compile
@gpu
@pytest.mark.parametrize("memory", [Full, Ckpt, BoundarySaving])
@pytest.mark.parametrize("mode", ["reduce-overhead", "max-autotune"])
def test_cuda_graph_modes_are_refused_on_a_gpu(mode, memory):
    with pytest.raises(ValueError, match="CUDA graphs.*'max-autotune-no-cudagraphs'"):
        _solver("cuda", memory, compile_mode=mode)


@requires_compile
@gpu
@pytest.mark.parametrize("mode", ["default", "max-autotune-no-cudagraphs"])
def test_the_other_modes_build_on_a_gpu(mode):
    _solver("cuda", compile_mode=mode)


@requires_compile
@gpu
def test_uncompiled_the_mode_is_not_refused():
    """Nothing is compiled, so nothing replays: the steps run as before."""
    solver = _solver("cuda", compile_mode="reduce-overhead", use_compile=False)
    wavelet = np.zeros(NT, dtype=np.float32)
    wavelet[2] = 1.0
    src, rec = np.array([[12, 4]]), np.array([[[ix, 4] for ix in range(2, 18, 4)]])
    record = solver(wavelet, src, rec, models=[torch.full(SHAPE, 2000.0, device="cuda")])
    assert torch.isfinite(record).all() and record.abs().max() > 0


def test_on_a_cpu_the_mode_is_not_refused():
    _solver("cpu", compile_mode="reduce-overhead")
