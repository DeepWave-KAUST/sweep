"""Regression tests for ``sweep.equations._free_surface._concat``.

Two failure modes the helper has had to balance:

1. numpy >= 1.25 exposes ``.device`` on ``np.ndarray`` per the Array API,
   so a generic duck-type ``hasattr(first, "device")`` check would
   mis-route plain ndarrays into ``torch.cat`` — guarded by
   :func:`test_concat_with_numpy_array`.
2. Under ``torch.compile`` on CUDA the elastic free-surface step traces
   ``_concat`` and used to bake an ``np.concatenate`` fallback into the
   compiled graph, raising "can't convert cuda tensor to numpy" at
   runtime — guarded by :func:`test_concat_with_cuda_compile`.
"""

import numpy as np
import pytest
import torch

from sweep.equations._free_surface import _concat


# ---------------------------------------------------------------------------
# Unit dispatch
# ---------------------------------------------------------------------------


def test_concat_with_numpy_array():
    """numpy inputs must dispatch to ``np.concatenate`` and stay numpy."""
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(6, 12, dtype=np.float32).reshape(2, 3)
    out = _concat([a, b], axis=0)
    assert isinstance(out, np.ndarray), f"expected ndarray, got {type(out).__name__}"
    assert out.shape == (4, 3)
    np.testing.assert_array_equal(out, np.concatenate([a, b], axis=0))


def test_concat_with_torch_cpu_tensor():
    """torch CPU tensors must dispatch to ``torch.cat`` and stay tensors.

    Sanity check for the dispatch path even on machines without CUDA, so
    a regression that breaks ``torch.cat`` selection is caught in CI."""
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
    out = _concat([a, b], axis=0)
    assert isinstance(out, torch.Tensor), f"expected Tensor, got {type(out).__name__}"
    assert out.shape == (4, 3)
    assert torch.equal(out, torch.cat([a, b], dim=0))


# ---------------------------------------------------------------------------
# Integration: torch.compile + CUDA on the elastic free-surface step.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile path")
def test_concat_with_cuda_compile():
    """Flat elastic free-surface with ``use_compile=True`` on CUDA.

    Before the fix this raised inside the compiled graph:
        TypeError: can't convert cuda:0 device type tensor to numpy.
    because ``_concat`` was specialised onto its ``np.concatenate`` branch
    while the wavefield was a CUDA tensor.
    """
    from sweep.equations import Elastic
    from sweep.propagator.torch import PropTorch
    from sweep.signal import ricker

    nz, nx, nt, dt = 60, 80, 80, 1e-3
    t = np.arange(nt, dtype=np.float32) * dt - 0.05
    wavelet = torch.tensor((1e3 * ricker(t, f=10.0)).astype(np.float32)).cuda()
    vp = torch.full((nz, nx), 2000.0).cuda()
    vs = torch.full((nz, nx), 1200.0).cuda()
    rho = torch.full((nz, nx), 1500.0).cuda()
    sources = torch.from_numpy(np.array([[nx // 2, 6]], dtype=np.int64)).cuda()
    receivers = torch.from_numpy(np.array([[nx // 2, 2]], dtype=np.int64)[None, ...]).cuda()

    prop = PropTorch(
        Elastic(spatial_order=4, device="cuda", backend="torch"),
        shape=(nz, nx),
        free_surface=True,
        topography=None,
        abcn=20,
        dh=10.0,
        dt=dt,
        use_ckpt=False,
        impl="eager",
        # use_compile=True is the EAGER_DEFAULTS — make it explicit so a
        # future flip of the default doesn't silently neuter this test.
        eager_options={"use_compile": True},
    )
    record = prop(wavelet, sources, receivers, models=[vp, vs, rho])

    assert torch.isfinite(record).all(), "non-finite values in record"
    assert float(record.abs().sum()) > 0.0, "record is all-zero — propagator didn't fire"
