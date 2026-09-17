"""The adjoint workspace is allocated only when a backward is coming.

`_ensure_wavefield_buffers` has always been called with
`need_adjoint=requires_backward`; the workspace allocation on the very next
line was unconditional. So a pure-forward propagator -- no model or wavelet
with requires_grad -- allocated its equation's full backward workspace on the
first call and zeroed it on every call after, for buffers that only a backward
reads. Elastic declares 8 of them, Elastic3D 18, each a padded grid.

This matters beyond the waste: the direction this branch exists for moves the
C++ side's per-backward-call scratch into this same pool. That is only
acceptable if forward-only users do not pay for it, and the way to make that
hold is to make it a tested property first.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import requires_binding

from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


def _elastic(dev):
    """Elastic declares backward_workspace_nvar=8, so it is the equation
    whose workspace is both real and visible."""
    from sweep.equations import Elastic

    nz, nx, nt, dt = 48, 64, 40, 1e-3
    g = np.linspace(0, 1, nz * nx, dtype=np.float32).reshape(nz, nx)
    models = [2200.0 + 400.0 * g, 1200.0 + 200.0 * g, 2000.0 + 100.0 * g]
    wav = torch.as_tensor(
        np.asarray(ricker(np.arange(nt) * dt - 0.02, 20.0), dtype=np.float32), device=dev)
    src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
    rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 4)]], dtype=np.int32)
    prop = PropTorch(Elastic(spatial_order=4, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=(nz, nx), dh=10.0, dt=dt, nt=nt,
                     abcn=10, source_type=["sxx", "szz"], receiver_type=["vx", "vz"],
                     dev=dev)
    assert int(prop._cuda_layout().backward_workspace_nvar) == 8
    return prop, models, wav, src, rec


@requires_binding("elastic2d_forward")
def test_a_forward_only_propagator_allocates_no_workspace():
    dev = torch.device("cuda:0")
    prop, models, wav, src, rec = _elastic(dev)
    m = [torch.tensor(x, device=dev) for x in models]          # no requires_grad
    prop(wav, src, rec, models=m)
    torch.cuda.synchronize()
    assert prop.adjoint_workspace == (), (
        f"{len(prop.adjoint_workspace)} workspace tensors live on a propagator that "
        f"will never run a backward")


@requires_binding("elastic2d_forward")
def test_a_gradient_forward_still_gets_its_workspace():
    """The other direction: laziness must not turn into absence."""
    dev = torch.device("cuda:0")
    prop, models, wav, src, rec = _elastic(dev)
    m = [torch.tensor(x, device=dev, requires_grad=True) for x in models]
    rec_out = prop(wav, src, rec, models=m)
    assert len(prop.adjoint_workspace) == 8
    (rec_out.double() ** 2).sum().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in m)


@requires_binding("elastic2d_forward")
def test_forward_only_after_a_gradient_call_keeps_the_pool():
    """A propagator reused across both modes must not thrash: the pool
    allocated for a gradient call stays for a later forward-only call."""
    dev = torch.device("cuda:0")
    prop, models, wav, src, rec = _elastic(dev)
    m = [torch.tensor(x, device=dev, requires_grad=True) for x in models]
    (prop(wav, src, rec, models=m).double() ** 2).sum().backward()
    pool = tuple(t.data_ptr() for t in prop.adjoint_workspace)
    assert len(pool) == 8
    m2 = [torch.tensor(x, device=dev) for x in models]
    prop(wav, src, rec, models=m2)
    assert tuple(t.data_ptr() for t in prop.adjoint_workspace) == pool
