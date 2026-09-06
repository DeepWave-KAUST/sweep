"""The wavefield-snapshot buffer must not outlive the call that made it.

``return_wavefield=True`` used to take its buffer from ``_workspace_cache``,
which has no eviction policy, so a buffer whose size grows linearly in the
number of snapshot steps -- and the default is EVERY step -- stayed pinned to
the propagator for its whole lifetime. Because it was workspace, it then had to
be cloned on the way out, so the path cost two copies of it at once.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch

NZ, NX, NT = 24, 28, 12


@pytest.fixture()
def prop():
    return PropTorch(Acoustic(spatial_order=4, device="cpu"), backend="torch",
                     impl="eager", shape=(NZ, NX), dh=10.0, dt=1e-3, nt=NT,
                     abcn=6, B=1, dev="cpu", source_type=["h1"],
                     receiver_type=["h1"],
                     # snapshots need the full tape; chunk checkpointing, the
                     # eager default, refuses return_wavefield outright
                     use_ckpt=False)


def _shoot(prop):
    src = np.array([[NX // 2, 4]], np.int64)
    rxx = np.arange(4, NX - 4, 3, np.int64)
    rec = np.stack([rxx, np.full(rxx.size, 4)], -1)[None]
    t = np.arange(NT, dtype=np.float32) * 1e-3 - 0.004
    a = np.pi * 30.0 * t
    wav = torch.tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2)).astype(np.float32))
    vp = torch.full((NZ, NX), 2000.0)
    return prop(wav, src, rec, models=[vp], return_wavefield=True)


def test_the_buffer_is_not_retained_after_the_call(prop):
    _shoot(prop)
    cache = getattr(prop._backend_impl, "_workspace_cache", {})
    # the cache is keyed by (kind, shape, device, dtype), not by kind alone
    kinds = {k[0] for k in cache}
    assert "snapshots" not in kinds, (
        "the snapshot buffer is pinned to the propagator; it grows with the "
        "number of snapshot steps and nothing ever evicts it"
    )


def test_the_returned_buffer_is_the_one_that_was_filled(prop):
    """Returning workspace directly would hand the caller a buffer the next
    call overwrites; returning a per-call allocation must not."""
    _, snaps = _shoot(prop)
    cache_ptrs = {t.data_ptr() for t in
                  getattr(prop._backend_impl, "_workspace_cache", {}).values()}
    assert snaps.data_ptr() not in cache_ptrs


def test_two_calls_do_not_alias(prop):
    _, first = _shoot(prop)
    baseline = first.clone()
    _, second = _shoot(prop)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first, baseline), "the second call overwrote the first result"


def test_the_snapshots_are_real(prop):
    record, snaps = _shoot(prop)
    assert snaps.shape[0] == NT and snaps.dtype == torch.float32
    assert snaps.abs().max() > 0, "every snapshot is zero"
    assert torch.equal(_shoot(prop)[1], snaps), "two identical calls disagree"
