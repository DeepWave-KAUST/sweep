"""Unit tests for the unified source/receiver/wavelet IO contract.

Verifies :meth:`PropBase._normalize_io` accepts the three supported input modes
(A1, A2, B) and rejects everything else with informative errors.  Also asserts
that ``source_encoding`` is no longer accepted as a public kwarg on the
propagator ``forward`` entry points.
"""
import inspect

import numpy as np
import pytest


@pytest.fixture
def prop():
    """Build a minimal 2-D acoustic PropTorch ('eager' impl) for shape checks.

    The eager backend is cheap to construct and exercises the same
    ``_normalize_io`` path shared by all backends.
    """
    import torch
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch

    eq = Acoustic(spatial_order=4, backend="torch")
    return PropTorch(
        eq,
        shape=(20, 24),
        dh=10.0,
        dt=1e-3,
        dev=torch.device("cpu"),
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=5,
        impl="eager",
        use_ckpt=False,
    )._backend_impl


def _arr(shape, dtype=np.int64):
    return np.zeros(shape, dtype=dtype)


# ---------------------------------------------------------------------------
# Accepted modes
# ---------------------------------------------------------------------------

def test_mode_A1_shared_wavelet(prop):
    nshots, nrec, nt = 4, 6, 50
    mode, B, nsrc, nr, enc = prop._normalize_io(
        wavelet=np.zeros(nt, dtype=np.float32),
        sources=_arr((nshots, 2)),
        receivers=_arr((nshots, nrec, 2)),
    )
    assert mode == "A1" and B == nshots and nsrc == 1 and nr == nrec and enc is False


def test_mode_A2_per_shot_wavelet(prop):
    nshots, nrec, nt = 3, 5, 40
    mode, B, nsrc, nr, enc = prop._normalize_io(
        wavelet=np.zeros((nshots, nt), dtype=np.float32),
        sources=_arr((nshots, 2)),
        receivers=_arr((nshots, nrec, 2)),
    )
    assert mode == "A2" and B == nshots and nsrc == 1 and nr == nrec and enc is False


def test_mode_B_shared_wavelet(prop):
    n_super_src, nrec, nt = 8, 6, 40
    mode, B, nsrc, nr, enc = prop._normalize_io(
        wavelet=np.zeros(nt, dtype=np.float32),
        sources=_arr((1, n_super_src, 2)),
        receivers=_arr((1, nrec, 2)),
    )
    assert mode == "B" and B == 1 and nsrc == n_super_src and nr == nrec and enc is True


def test_mode_B_per_source_wavelet(prop):
    n_super_src, nrec, nt = 5, 6, 40
    mode, B, nsrc, nr, enc = prop._normalize_io(
        wavelet=np.zeros((n_super_src, nt), dtype=np.float32),
        sources=_arr((1, n_super_src, 2)),
        receivers=_arr((1, nrec, 2)),
    )
    assert mode == "B" and B == 1 and nsrc == n_super_src and enc is True


# ---------------------------------------------------------------------------
# Rejected shapes
# ---------------------------------------------------------------------------

def test_reject_shared_2d_receivers(prop):
    """The old shared-receivers form (nrec, dim) is no longer accepted."""
    with pytest.raises(ValueError, match="receivers must have shape"):
        prop._normalize_io(
            wavelet=np.zeros(50, dtype=np.float32),
            sources=_arr((3, 2)),
            receivers=_arr((6, 2)),  # 2-D — should be 3-D
        )


def test_reject_receivers_batch_mismatch(prop):
    with pytest.raises(ValueError, match="receivers batch"):
        prop._normalize_io(
            wavelet=np.zeros(50, dtype=np.float32),
            sources=_arr((4, 2)),
            receivers=_arr((3, 6, 2)),
        )


def test_reject_A2_wavelet_wrong_batch(prop):
    with pytest.raises(ValueError, match="wavelet must have shape"):
        prop._normalize_io(
            wavelet=np.zeros((5, 50), dtype=np.float32),  # 5 != nshots=3
            sources=_arr((3, 2)),
            receivers=_arr((3, 6, 2)),
        )


def test_reject_B_sources_batch_not_one(prop):
    with pytest.raises(ValueError, match="sources in source-encoding"):
        prop._normalize_io(
            wavelet=np.zeros(50, dtype=np.float32),
            sources=_arr((2, 4, 2)),  # batch must be 1
            receivers=_arr((2, 6, 2)),
        )


def test_reject_B_wavelet_nsrc_mismatch(prop):
    with pytest.raises(ValueError, match="wavelet must have shape"):
        prop._normalize_io(
            wavelet=np.zeros((4, 50), dtype=np.float32),  # nsrc disagrees
            sources=_arr((1, 5, 2)),
            receivers=_arr((1, 6, 2)),
        )


def test_reject_wrong_ndim_sources(prop):
    with pytest.raises(ValueError, match="sources must have shape"):
        prop._normalize_io(
            wavelet=np.zeros(50, dtype=np.float32),
            sources=_arr((3,)),  # 1-D
            receivers=_arr((3, 6, 2)),
        )


# ---------------------------------------------------------------------------
# Removed kwarg
# ---------------------------------------------------------------------------

def test_source_encoding_kwarg_removed_torch():
    from sweep.propagator.torch import PropTorch
    sig = inspect.signature(PropTorch.forward)
    assert "source_encoding" not in sig.parameters


def test_source_encoding_kwarg_removed_jax():
    pytest.importorskip("jax")
    from sweep.propagator.jax import PropJax
    sig = inspect.signature(PropJax.forward)
    assert "source_encoding" not in sig.parameters
