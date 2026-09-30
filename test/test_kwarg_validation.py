"""A typo'd keyword must be an error, not a silently kept default.

Before this fix, PropTorch(..., free_surfce=True) constructed a flat-surface
propagator with no warning on BOTH impls (the leftover died unread in
PropBase.__init__'s **kwargs), and forward(..., pml_freqs=15) silently kept
pml_freq=25 (init_abc reads its extras via .get). The only keys that DID
error were ones colliding with the other impl's option set -- exactly the
wrong ones to single out.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sweep.equations.acoustic import Acoustic
from sweep.propagator.torch import PropTorch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _build(impl="eager", **extra):
    eq = Acoustic(spatial_order=4, device=DEV, backend="torch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PropTorch(eq, backend="torch", impl=impl, shape=(48, 56),
                         dev=DEV, dh=10.0, dt=1e-3, source_type=["p"],
                         receiver_type=["p"], abcn=20, nt=50, B=1, **extra)


@pytest.mark.parametrize("impl", ["eager"] + (["c"] if DEV == "cuda" else []))
def test_construction_typo_raises_and_names_the_key(impl):
    with pytest.raises(TypeError, match="free_surfce"):
        _build(impl=impl, free_surfce=True)


def test_legacy_spellings_are_still_consumed_not_rejected():
    # the deprecated loose keywords must keep working -- they are popped
    # BEFORE the leftover check.
    p = _build(boundary_saving_config={"enabled": True}, transfer_interval=8)
    assert p.memory_strategy == "boundary"
    # (the eager backend then applies its own transfer cadence -- what matters
    # here is that the loose key was consumed, not rejected)


def test_model_parallel_kwarg_is_still_accepted():
    assert _build(model_parallel=None).model_parallel is None


def test_call_time_typo_raises_and_names_the_key():
    p = _build()
    wavelet = torch.zeros(50, device=DEV)
    src = np.array([[[24, 28]]], dtype=np.int64)
    rec = np.array([[[10, 28]]], dtype=np.int64)
    with pytest.raises(TypeError, match="pml_freqs"):
        p(wavelet, src, rec, models=[torch.full((48, 56), 1500.0, device=DEV)],
          pml_freqs=15)


def test_call_time_recognised_extras_still_work():
    p = _build()
    wavelet = torch.zeros(50, device=DEV)
    src = np.array([[[24, 28]]], dtype=np.int64)
    rec = np.array([[[10, 28]]], dtype=np.int64)
    out = p(wavelet, src, rec, models=[torch.full((48, 56), 1500.0, device=DEV)],
            max_vel=4000.0, pml_freq=20.0)
    assert out.shape[-2] == 50 or out.shape[1] == 50
