"""PropJax public call entry point.

Regression guard for a name collision that silently disabled the whole jax
front-end: ``PropBase.__init__`` binds the per-edge PML widths to
``self.pad`` (a tuple), so a *method* named ``pad`` on ``PropJax`` was
shadowed and every ``prop(...)`` raised
``TypeError: 'tuple' object is not callable``. The model-padding helper is
therefore called ``pad_model``; these tests fail loudly if either name moves
back on top of the other.

Deliberately torch-free so it runs in jax-only environments.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from sweep.equations import Acoustic
from sweep.propagator.jax import PropJax

NZ, NX = 40, 48
DH, DT, NT, ABCN = 10.0, 1e-3, 60, 12


def _solver():
    eq = Acoustic(spatial_order=4, backend="jax")
    return PropJax(eq, (NZ, NX), dh=DH, dt=DT, nt=NT, abcn=ABCN,
                   source_type=["h1"], receiver_type=["h1"],
                   backend="jax", use_ckpt=False)


def _geom():
    t = np.arange(NT, dtype=np.float32) * DT - 0.03
    wavelet = (1e3 * (1 - 2 * (np.pi * 20.0 * t) ** 2)
               * np.exp(-((np.pi * 20.0 * t) ** 2))).astype(np.float32)
    src = np.array([[NX // 2, NZ // 2]], dtype=np.int32)
    rec = np.array([[[NX // 4, NZ // 3], [3 * NX // 4, NZ // 3]]], dtype=np.int32)
    return wavelet, src, rec


def test_pad_names_do_not_shadow():
    prop = _solver()
    # the PML widths stay a plain tuple on ``pad`` (public, used by base/_c)
    assert isinstance(prop.pad, tuple), f"prop.pad must stay the width tuple, got {type(prop.pad)}"
    assert len(prop.pad) == 2 * prop.ndim
    # the model padder must remain reachable as a callable under its own name
    assert callable(getattr(prop, "pad_model", None)), "PropJax.pad_model is missing/not callable"


def test_public_call_runs():
    """``prop(...)`` — the documented entry point — must work, not raise
    ``TypeError: 'tuple' object is not callable``."""
    prop = _solver()
    wavelet, src, rec = _geom()
    models = [jnp.asarray(np.full((NZ, NX), 2000.0, dtype=np.float32))]
    out = prop(wavelet, src, rec, models=models)
    out = out[0] if isinstance(out, (tuple, list)) else out
    rec_arr = np.asarray(out)
    assert rec_arr.shape[0] == 1 and rec_arr.shape[1] == NT
    assert np.isfinite(rec_arr).all()
    assert np.abs(rec_arr).max() > 0.0


def test_public_call_matches_call_forward():
    """The public entry differs from ``__call_forward__`` only by padding the
    models itself, so both must produce the same record."""
    prop = _solver()
    wavelet, src, rec = _geom()
    vp = np.full((NZ, NX), 2000.0, dtype=np.float32)

    a = prop(wavelet, src, rec, models=[jnp.asarray(vp)])
    a = np.asarray(a[0] if isinstance(a, (tuple, list)) else a)
    b = prop.__call_forward__(wavelet, src, rec, models=[jnp.asarray(vp)])
    b = np.asarray(b[0] if isinstance(b, (tuple, list)) else b)

    assert a.shape == b.shape
    denom = max(float(np.abs(b).max()), 1e-30)
    assert float(np.abs(a - b).max()) / denom < 1e-5
