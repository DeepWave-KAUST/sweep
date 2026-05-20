"""Regression test for the cuda:1 (non-default device) CUDAGuard bug.

Before the fix in bug/cuda-guard-missing-non-default-device, forward/backward
CUDA kernels in most equations (acoustic*, elastic*, das_mu*) launched on the
current default device, which is cuda:0 unless the caller explicitly sets it.
A user passing tensors on cuda:1 would hit "illegal memory access" because
the kernels ran on cuda:0 while tensors lived on cuda:1.

This test runs a tiny acoustic forward + backward on every visible CUDA
device.  If the fix regresses (CUDAGuard removed from an entry function),
the kernel launch on cuda:1 raises.
"""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires >=2 visible CUDA devices",
)


def _tiny_acoustic_run(device):
    """Forward + backward an acoustic propagator on `device`.

    Returns the per-iteration loss so a caller can sanity-check finite values.
    """
    import numpy as np

    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch
    from sweep.signal import ricker

    shape = (48, 56)
    dh, dt, nt = 10.0, 0.0015, 120
    freq, delay = 10.0, 0.06

    vp_np = np.full(shape, 2000.0, dtype=np.float32)
    sources = np.array([[shape[1] // 2, 4]], dtype=np.int64)
    rec_x = np.arange(0, shape[1], 6, dtype=np.int64)
    receivers = np.stack([rec_x, np.full(rec_x.shape, 2, dtype=np.int64)], axis=1)[None]

    t = np.arange(nt, dtype=np.float32) * dt
    wavelet = ricker(t - delay, f=freq).astype(np.float32)

    equation = Acoustic(spatial_order=4, device=device, backend="torch")
    solver = PropTorch(
        equation,
        shape=shape,
        dh=dh,
        dt=dt,
        dev=device,
        abcn=30,
        source_type=["h1"],
        receiver_type=["h1"],
        pml_type="cpmlr",
        backend="torch",
        impl="c",
    )

    vp_true = torch.tensor(vp_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        observed = solver(wavelet, sources, receivers, models=[vp_true]).detach()

    vp = torch.tensor(vp_np * 0.95, dtype=torch.float32, device=device, requires_grad=True)
    predicted = solver(wavelet, sources, receivers, models=[vp])
    loss = (predicted - observed).pow(2).mean()
    loss.backward()
    assert vp.grad is not None
    return float(loss.detach().cpu()), float(vp.grad.norm().cpu())


@pytest.mark.parametrize("device_str", [f"cuda:{i}" for i in range(torch.cuda.device_count())])
def test_acoustic_runs_on_each_visible_device(device_str):
    """Forward + backward must succeed on every CUDA device, not just cuda:0."""
    device = torch.device(device_str)
    loss, gnorm = _tiny_acoustic_run(device)
    assert torch.isfinite(torch.tensor(loss)), f"loss not finite on {device_str}: {loss}"
    assert torch.isfinite(torch.tensor(gnorm)), f"grad norm not finite on {device_str}: {gnorm}"
