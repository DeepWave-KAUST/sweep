"""What a SECOND backward over the same graph does, on each impl.

Written to pin an invariant for a refactor that moves tensor allocation out of
C++, and it found something instead: `impl='c'` has never supported
`loss.backward(retain_graph=True)` followed by another `backward()`. The
compiled path releases its params object -- wavefields, PML profiles, the
forward source -- as soon as the first backward has produced the gradient,
which is deliberate and is what keeps a propagation's buffers from outliving
the gradient. The five per-attribute `del ctx.*` lines that do it predate the
params-object refactor, so this is not new; nothing in the suite had ever taken
that path.

It surfaced as `AttributeError: 'WrapperBackward' object has no attribute 'cp'`
from the middle of the unpack, which tells a user nothing. It now refuses with
a message that names the reason and the two ways out.

`impl='eager'` keeps its graph and IS repeatable, so that half is a real
bit-exactness check -- and it is the half that matters for the allocation
refactor: an eager buffer zeroed per FORWARD rather than per BACKWARD would
show up here and nowhere else, because gate/bitgate.py runs a forward before
every backward.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import requires_binding

from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


def _reflective_setup(dev):
    """A model that actually produces a gradient.

    Deliberately not a uniform half-space with a short record: that gives
    max|grad| == 0 exactly, and comparing two zero tensors passes whatever the
    buffers did. Measured here: max|record| 2.6, max|grad| 2.0, 18436 of 19200
    cells non-zero.
    """
    nz, nx, nt, dt = 120, 160, 600, 1e-3
    vp_np = np.full((nz, nx), 2000.0, np.float32)
    vp_np[60:] = 2600.0                       # a reflector, so there is signal
    wav = torch.as_tensor(
        np.asarray(ricker(np.arange(nt) * dt - 0.1, 12.0), dtype=np.float32), device=dev)
    # [x, z], which is the convention everywhere else in test/. Writing it the
    # other way round does not fault on impl='c' -- it just indexes a different
    # cell -- while eager asserts "index out of bounds", so the two backends
    # disagree about whether a transposed geometry is an error at all.
    src = np.array([[[nx // 2, 10]]], dtype=np.int32)
    rec = np.array([[[i, 6] for i in range(8, nx - 8, 4)]], dtype=np.int32)
    return (nz, nx), nt, dt, vp_np, wav, src, rec


def _run(impl, dev):
    from sweep.equations import Acoustic

    shape, nt, dt, vp_np, wav, src, rec = _reflective_setup(dev)
    prop = PropTorch(Acoustic(spatial_order=4, device=dev, backend="torch"),
                     backend="torch", impl=impl, shape=shape, dh=10.0, dt=dt,
                     nt=nt, abcn=20, dev=dev)
    vp = torch.tensor(vp_np, device=dev, requires_grad=True)
    record = prop(wav, src, rec, models=[vp])
    return vp, (record.double() ** 2).sum()


@requires_binding("acoustic2d_forward")
def test_c_refuses_a_second_backward_and_says_why():
    dev = torch.device("cuda:0")
    vp, loss = _run("c", dev)

    loss.backward(retain_graph=True)
    assert float(vp.grad.abs().max()) > 1e-3, "degenerate setup: the first backward gave nothing"

    with pytest.raises(RuntimeError) as exc:
        loss.backward()
    msg = str(exc.value)
    # The point of the check is the message, so assert on it rather than on the
    # type: an AttributeError from inside the unpack is also a RuntimeError's
    # cousin and tells the user nothing.
    assert "second backward" in msg and "retain_graph" in msg, msg
    assert "impl='eager'" in msg, "the refusal should name the way out: " + msg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_eager_is_repeatable_bitwise():
    dev = torch.device("cuda:0")
    vp, loss = _run("eager", dev)

    loss.backward(retain_graph=True)
    first = vp.grad.detach().clone()
    vp.grad = None
    loss.backward()                            # no forward in between
    second = vp.grad.detach()

    assert float(first.abs().max()) > 1e-3, f"degenerate setup, max|grad| = {float(first.abs().max())}"
    assert torch.equal(first, second), (
        f"the second backward differs: max|d| = {float((first - second).abs().max()):.3e}. "
        f"Something the backward reads is zeroed per forward rather than per backward.")
