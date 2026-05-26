"""Regression test for the checkpoint batch-shrink contiguity bug.

``_CompiledPropagator._ensure_checkpoint_buffers`` previously allocated
checkpoint tensors with the *largest* batch dimension ever seen (cached
in ``self.B``) and ``_slice_checkpoint_buffers`` returned
``t[:, :batch_size]``.  When ``batch_size`` is strictly smaller than the
cached ``B`` the slice is a non-contiguous view, and the CUDA kernel
raises::

    RuntimeError: acoustic2d checkpoint tensor must be contiguous

Triggering sequence used in FWI with source-encoding probes:

1. Call ``solver(...)`` with naive multi-shot at ``batch_size=N`` so the
   checkpoint buffer is allocated at batch N.
2. Call the *same* solver with source-encoded input
   (``sources=(1, nsrc, ndim)``) → ``batch_size=1`` → non-contiguous
   slice → kernel raises.

The fix re-allocates the checkpoint buffer when the active batch size
changes, mirroring how wavefield buffers expose a contiguous leading-dim
slice.
"""

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA",
)


def _build_solver(nz, nx, nt, dh, dt, dev):
    from sweep.equations import Acoustic
    from sweep.propagator.options import CkptOptions, CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    equation = Acoustic(spatial_order=4, device=dev, backend="torch")
    solver = PropTorch(
        equation,
        shape=(nz, nx),
        dev=dev,
        dh=dh,
        dt=dt,
        nt=nt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=30,
        backend="cuda",
        cuda_options=CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="chunk", chunks=20, storage="gpu"),
            )
        ),
    )
    return solver


def test_checkpoint_batch_shrink_remains_contiguous():
    from sweep.signal import ricker

    dev = torch.device("cuda")
    nz, nx, nt = 48, 56, 120
    dh, dt, freq, delay = 10.0, 1.5e-3, 10.0, 0.06

    t = np.arange(nt, dtype=np.float32) * dt
    wavelet = ricker(t - delay, f=freq).astype(np.float32)

    vp = torch.full((nz, nx), 2000.0, device=dev).requires_grad_(True)

    # Step 1: naive multi-shot at batch=4 -> caches checkpoint buffer at B=4.
    nshots = 4
    sources_a = np.stack(
        [np.linspace(8, nx - 8, nshots, dtype=np.int64),
         np.full(nshots, 4, dtype=np.int64)],
        axis=1,
    )
    rec_x = np.linspace(4, nx - 4, 16, dtype=np.int64)
    rec_z = np.full_like(rec_x, 2)
    receivers_a = np.stack([rec_x, rec_z], axis=1)
    receivers_a = np.broadcast_to(receivers_a, (nshots, 16, 2)).copy()

    solver = _build_solver(nz, nx, nt, dh, dt, dev)
    syn_a = solver(wavelet, sources_a, receivers_a, models=[vp])
    syn_a.pow(2).mean().backward()
    torch.cuda.synchronize()

    # Step 2: same solver, source-encoded super-shot (batch=1, nsrc>1).
    # This is what fails before the fix.
    nsrc = nshots
    sources_b = sources_a[None, ...]                          # (1, nsrc, 2)
    receivers_b = receivers_a[:1]                              # (1, nrec, 2)
    wavelet_b = np.broadcast_to(wavelet, (nsrc, nt)).copy()    # (nsrc, nt)

    if vp.grad is not None:
        vp.grad = None
    syn_b = solver(wavelet_b, sources_b, receivers_b, models=[vp])
    loss_b = syn_b.pow(2).mean()
    loss_b.backward()
    torch.cuda.synchronize()

    assert torch.isfinite(vp.grad).all(), "gradient has NaN/inf"
    assert vp.grad.abs().max().item() > 0.0, "gradient is identically zero"
