"""Two forwards before one backward on the eager propagator.

The initial wavefield came from a workspace cache that every call zeroed in
place. Chunk checkpointing, the default memory strategy, saves each chunk's
inputs, the first chunk's being that buffer, so a second forward (recorded or
not) bumped the version of a tensor the first one's backward still needed, and
``(loss1 + loss2).backward()`` raised "modified by an inplace operation".
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator import BoundarySaving, EagerOptions, Full, PropTorch

from conftest import ricker

SHAPE, NT, DT, DH = (32, 40), 64, 0.0015, 10.0
SRC1 = np.array([[20, 8]], dtype=np.int64)
SRC2 = np.array([[13, 8]], dtype=np.int64)
_RX = np.arange(2, 38, 6)
REC = np.stack([_RX, np.full(_RX.size, 2)], -1)[None].astype(np.int64)

MEMORIES = [None, Full(), BoundarySaving()]
IDS = ["default_ckpt", "full", "boundary"]


def _prop(memory):
    eq = Acoustic(spatial_order=4, device="cpu", backend="torch")
    kw = {} if memory is None else {"memory": memory}
    return PropTorch(eq, impl="eager", shape=SHAPE, dev="cpu", dh=DH, dt=DT,
                     abcn=20, nt=NT, eager_options=EagerOptions(use_compile=False), **kw)


def _vp():
    vp = np.full(SHAPE, 2000.0, dtype=np.float32)
    vp[12:] += 300.0
    return torch.from_numpy(vp).requires_grad_(True)


WAV = torch.as_tensor(ricker(NT, DT, 10.0, 0.06), dtype=torch.float32)


def _grad(prop, *shots):
    vp = _vp()
    sum(prop(WAV, s, REC, models=[vp]).square().sum() for s in shots).backward()
    return vp.grad


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("memory", MEMORIES, ids=IDS)
def test_two_recorded_forwards_then_one_backward(memory):
    prop = _prop(memory)
    g1, g2 = _grad(prop, SRC1), _grad(prop, SRC2)
    assert _rel(_grad(prop, SRC1, SRC2), g1 + g2) < 1e-6


@pytest.mark.parametrize("memory", MEMORIES, ids=IDS)
def test_a_no_grad_forward_before_the_backward(memory):
    """An unrecorded call between a forward and its backward."""
    prop = _prop(memory)
    ref = _grad(prop, SRC1)
    vp = _vp()
    loss = prop(WAV, SRC1, REC, models=[vp]).square().sum()
    with torch.no_grad():
        prop(WAV, SRC2, REC, models=[vp])
    loss.backward()
    assert _rel(vp.grad, ref) < 1e-6
