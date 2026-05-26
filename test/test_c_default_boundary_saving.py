"""Regression tests for the impl='c' default memory strategy.

Defaults switched from chunked checkpointing to boundary saving (GPU
storage) so that the C backend's lifetime buffers are sized for the
batch the user actually runs at.  These tests pin the new behaviour and
verify that:

1. ``PropTorch(impl='c')`` with no memory option ends up in
   boundary-saving / GPU mode (``use_ckpt`` flipped off, boundary
   saving enabled).
2. Explicitly opting in to checkpointing via ``cuda_options`` still
   works (the default is opt-out).
3. 2-D RTM does not raise even when the default would have boundary
   saving on (the rtm path silently forces full-wavefield mode).
"""

import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="impl='c' requires CUDA.",
)


_SHAPE = (80, 96)
_DH = 12.5
_DT = 1e-3
_NT = 200
_ABCN = 50


def _build_solver(**solver_kwargs):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eq = Acoustic(spatial_order=4, device=dev, backend="torch")
    return PropTorch(
        eq,
        shape=_SHAPE,
        dh=_DH,
        dt=_DT,
        nt=_NT,
        abcn=_ABCN,
        dev=dev,
        impl="c",
        **solver_kwargs,
    )


@cuda_only
def test_c_default_is_boundary_saving_gpu():
    solver = _build_solver()
    impl = solver._backend_impl
    assert impl.use_ckpt is False, "impl='c' default should switch off chunked ckpt."
    assert impl.boundary_saving_config["enabled"] is True, "boundary saving should be on."
    assert impl.boundary_saving_config["storage"] == "gpu"


@cuda_only
def test_c_explicit_ckpt_opt_in():
    solver = _build_solver(cuda_options={"memory": {"strategy": "ckpt"}})
    impl = solver._backend_impl
    assert impl.use_ckpt is True
    assert impl.boundary_saving_config["enabled"] is False


@cuda_only
def test_c_rtm_2d_no_longer_raises():
    """RTM 2-D used to require explicit ``use_ckpt=False``; now it just runs."""

    solver = _build_solver()  # default → boundary saving on
    if solver._backend_impl.rtm_func is None:
        pytest.skip("Acoustic RTM not available in this binding.")

    nz, nx = _SHAPE
    nt = _NT
    t = np.arange(nt, dtype=np.float32) * _DT - 0.05
    x = np.pi * 10.0 * t
    wavelet = ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)

    sources = np.array([[nx // 2, 4]], dtype=np.int64)
    rx = np.arange(8, nx - 8, 2, dtype=np.int64)
    receivers = np.stack(
        [rx, np.full(len(rx), 6, dtype=np.int64)],
        axis=-1,
    )[None, ...]
    adjoint_source = np.zeros(
        (1, nt, receivers.shape[1], 1), dtype=np.float32
    )

    dev = torch.device("cuda")
    vp = torch.full((nz, nx), 1800.0, dtype=torch.float32, device=dev)

    # Should not raise — the 2-D RTM path silently forces full mode.
    syn, image, src_illum, _ = solver.rtm(
        wavelet, sources, receivers, adjoint_source=adjoint_source, models=[vp]
    )
    assert syn.shape[0] == 1
    # Image is returned in the full padded layout (B, 1, nz_full, nx_full).
    assert image.ndim == 4
    assert image.shape[0] == 1
    # Solver did NOT mutate its persistent flag — boundary saving stays on.
    assert solver._backend_impl.use_ckpt is False
    assert solver._backend_impl.boundary_saving_config["enabled"] is True
