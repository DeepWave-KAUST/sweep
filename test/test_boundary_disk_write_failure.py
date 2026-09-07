"""A boundary-disk WRITE that fails must raise, not return a wrong gradient.

The write path cannot report a failure by propagating it: the writers are
detached threads, and the enqueue runs inside a ``cudaLaunchHostFunc``
callback, which terminates the process if an exception escapes it. So both
sites used to catch, print to stderr, and call ``boundary_disk_writer_done()``
anyway -- decrementing the very counter ``wait_for_boundary_disk_writes()``
waits on. The barrier then returned as if the data had landed, and because
``_allocate_boundary_disk_files`` creates the files SPARSE with
``handle.truncate()``, the backward read zeros for anything never written.

Measured before the fix, with a second problem run against boundary files still
holding the first one's data: every write failed, nothing raised, and the
gradient came back ``1.46e-02`` off (cosine 0.9935) from the same problem on
gpu-direct storage. ENOSPC on a scratch filesystem is not hypothetical, and a
1.5 % gradient error that announces itself only on stderr is the worst kind.

The failure is injected by making the boundary files read-only after they have
been created, which is what ``write_boundary_file_chunk``'s own
"Failed to open boundary disk file for writing" check is there to catch -- no
root, no filesystem tricks, and it exercises the real error path rather than a
test hook compiled into the hot path.
"""
import glob
import os
import stat

import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.options import BoundarySaving
from sweep.propagator.torch import PropTorch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="boundary disk staging is a CUDA path")

DH, DT, NT, N, ABCN = 10.0, 6e-4, 300, 90, 12


def _wavelet(dev):
    t = np.arange(NT, dtype=np.float32) * DT - 0.035
    a = np.pi * 12.0 * t
    return torch.as_tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32),
                           device=dev)


def _geometry():
    src = np.array([[[(N + 3) // 3, N // 4]]], dtype=np.int64)
    rec = np.array([[[ix, 3] for ix in range(3, N, 4)]], dtype=np.int64)
    return src, rec


def _prop(dev, memory):
    return PropTorch(Acoustic(spatial_order=4, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=(N, N + 3), dh=DH, dt=DT,
                     nt=NT, abcn=ABCN, dev=dev, memory=memory)


def _grad(prop, vp, dev):
    wav, (src, rec) = _wavelet(dev), _geometry()
    leaf = torch.tensor(vp, device=dev, requires_grad=True)
    (prop(wav, src, rec, models=[leaf]).double() ** 2).sum().backward()
    return leaf.grad.detach().clone()


def _chmod_boundary_files(root, mode):
    for path in glob.glob(os.path.join(root, "*.bin")):
        os.chmod(path, mode)


def test_failed_boundary_disk_write_raises(tmp_path):
    dev = torch.device("cuda:0")
    vp_a = np.full((N, N + 3), 1800.0, dtype=np.float32)
    vp_a[N // 2:] += 300.0
    # A DIFFERENT model for the second run: with the same one, the stale bytes
    # the failed writes leave behind are the CORRECT bytes and the gradient
    # comes back right for the wrong reason -- which is exactly how the first
    # attempt at this test passed against the unfixed code.
    vp_b = np.full((N, N + 3), 2400.0, dtype=np.float32)
    vp_b[: N // 3] -= 500.0

    prop = _prop(dev, BoundarySaving(storage="disk", disk_dir=str(tmp_path)))
    _grad(prop, vp_a, dev)                      # creates and fills the files
    root = prop._boundary_disk_root
    assert root and glob.glob(os.path.join(root, "*.bin")), "no boundary files were created"

    _chmod_boundary_files(root, stat.S_IRUSR)
    try:
        with pytest.raises(RuntimeError, match="boundary disk file"):
            _grad(prop, vp_b, dev)
    finally:
        # the propagator's own cleanup unlinks these
        _chmod_boundary_files(root, stat.S_IRUSR | stat.S_IWUSR)


def test_boundary_disk_write_error_does_not_leak_into_the_next_run(tmp_path):
    """Reporting the failure clears it: a later, healthy run must not inherit it."""
    dev = torch.device("cuda:0")
    vp = np.full((N, N + 3), 1800.0, dtype=np.float32)
    vp[N // 2:] += 300.0
    vp_b = np.full((N, N + 3), 2400.0, dtype=np.float32)

    prop = _prop(dev, BoundarySaving(storage="disk", disk_dir=str(tmp_path)))
    _grad(prop, vp, dev)
    root = prop._boundary_disk_root
    _chmod_boundary_files(root, stat.S_IRUSR)
    try:
        with pytest.raises(RuntimeError, match="boundary disk file"):
            _grad(prop, vp_b, dev)
    finally:
        _chmod_boundary_files(root, stat.S_IRUSR | stat.S_IWUSR)

    # writable again: the same propagator must now work, and agree bit-for-bit
    # with gpu-direct storage on the same problem.
    got = _grad(prop, vp_b, dev)
    want = _grad(_prop(dev, BoundarySaving(storage="gpu")), vp_b, dev)
    assert torch.equal(got, want), (
        f"disk gradient differs from gpu-direct after a recovered write failure; "
        f"max|d|={float((got - want).abs().max()):.3e}")
