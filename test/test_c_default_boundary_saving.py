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
def test_c_rtm_entry_is_removed():
    """solver.rtm() is gone entirely -- no method, no tombstone.

    The entry was full-wavefield-only in 2-D, had no bit-exact gate, its last
    real caller (sweep-tasks RTM) migrated to the forward+backward recipe
    (inner-product loss + compute_illumination, see notebook 08), and the docs
    described a notebook as using it when the notebook deliberately did not.
    Pinned so it cannot quietly return without a design discussion.
    """
    solver = _build_solver()
    assert not hasattr(solver, "rtm")
    assert not hasattr(solver._backend_impl, "rtm")


