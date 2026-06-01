"""Phase 2 test: CUDA forward parity with Python eager forward.

Build the same propagator twice -- once with ``impl='eager'`` (Python
torch step) and once with ``impl='c'`` (CUDA kernel) -- run the same
shot through both, compare the records. Relative L2 should be <1e-4.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.equations import (
    ElasticVRR,
    compute_vector_reflectivity,
)
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEVICE = "cuda"

# Canonical small-grid config (matches Phase 1 tests).
NZ, NX = 48, 56
DH = 10.0
DT = 1.5e-3
NT = 120
ABCN = 30
SO = 4
FREQ = 10.0
DELAY = 0.06


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32)).to(DEVICE)


def _geometry():
    # impl='c' parses sources/receivers via numpy.copy(); pass numpy.
    sources = np.array([[NX // 2, NZ // 4]], dtype=np.int64)
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    receivers = np.stack([rec_x, np.full_like(rec_x, 2)], axis=-1)[None, ...]
    return sources, receivers


def _make_prop(impl):
    eq = ElasticVRR(spatial_order=SO, device=DEVICE, backend="torch")
    return PropTorch(
        eq, shape=(NZ, NX),
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl=impl,
        eager_options={"use_compile": False} if impl == "eager" else None,
    )


def _layered_models():
    z = torch.arange(NZ, device=DEVICE, dtype=torch.float32).view(NZ, 1)
    vp = (1800.0 + 8.0 * z).expand(NZ, NX).contiguous()
    vs = vp / 1.73
    rho = (1000.0 + 5.0 * z).expand(NZ, NX).contiguous().clone()
    rho[NZ // 2 :, :] += 200.0
    Rp_x, Rp_z, Rs_x, Rs_z = compute_vector_reflectivity(vp, vs, rho, h=DH)
    return [vp, vs, Rp_x, Rp_z, Rs_x, Rs_z]


def test_cuda_forward_matches_eager():
    """impl='c' record should match impl='eager' record to ~1e-4 rel L2."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    models = _layered_models()
    sources, receivers = _geometry()
    wavelet = _wavelet()

    prop_eager = _make_prop("eager")
    syn_eager = prop_eager(wavelet, sources, receivers, models=models)

    prop_c = _make_prop("c")
    syn_c = prop_c(wavelet, sources, receivers, models=models)

    print(f"eager shape: {tuple(syn_eager.shape)} max={syn_eager.abs().max().item():.4e}")
    print(f"c     shape: {tuple(syn_c.shape)} max={syn_c.abs().max().item():.4e}")

    rel = ((syn_eager - syn_c).pow(2).sum().sqrt() /
           syn_eager.pow(2).sum().sqrt().clamp_min(1e-30)).item()
    print(f"rel L2: {rel:.4e}")
    assert rel < 1e-3, f"impl=c vs impl=eager rel L2 = {rel:.4e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
