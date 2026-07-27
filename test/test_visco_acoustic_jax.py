"""ViscoAcoustic on the JAX backend.

Regression: the wavenumber grid was built inside ``func`` from traced values --
``float(np.asarray(hz))`` on a scan-carried tracer, then ``str(vp.device)`` --
so every forward died with TracerArrayConversionError.  ``k`` is now built in
``init_abc`` from the concrete runtime shape and grid spacing, and kept as
numpy on jax: the same rule ``init_abc`` already applies to the PML profiles,
because a jnp array cached on the instance leaks as a tracer into the next
trace.

The torch path lives in ``test_visco_acoustic.py``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "sweep").exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

NZ, NX = 40, 48
DH, DT, NT, ABCN = 10.0, 1e-3, 80, 15
VP, Q_VAL, F0 = 2000.0, 30.0, 15.0


def _geom():
    t = np.arange(NT) * DT - 1.0 / F0
    a = (np.pi * F0 * t) ** 2
    w = ((1 - 2 * a) * np.exp(-a)).astype(np.float32)[None, :]
    # sweep coordinates are (x, z)
    src = np.array([[[NX // 2, 4]]], dtype=np.int32)
    rx = np.arange(4, NX - 4, 4, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, 2, np.int32)], -1)[None]
    return w, src, rec


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def _vp():
    return np.full((NZ, NX), VP, dtype=np.float32)


def _q():
    return np.full((NZ, NX), Q_VAL, dtype=np.float32)


def _om():
    return np.full((NZ, NX), 2 * np.pi * F0, dtype=np.float32)


def _run_visco(phase_shift=True, amplitude_damping=True, n=1):
    from sweep.equations import ViscoAcoustic
    from sweep.propagator.jax import PropJax

    eq = ViscoAcoustic(4, "cpu", "jax",
                       phase_shift=phase_shift, amplitude_damping=amplitude_damping)
    prop = PropJax(eq, (NZ, NX), dh=DH, dt=DT, abcn=ABCN, use_ckpt=False)
    prop.set_parameters([_vp(), _q(), _om()])
    w, src, rec = _geom()
    models = [jnp.asarray(_vp()), jnp.asarray(_q()), jnp.asarray(_om())]
    outs = []
    for _ in range(n):
        o = prop.__call_forward__(w, src, rec, models=models)
        outs.append(np.asarray(o[0] if isinstance(o, (tuple, list)) else o))
    return (outs[0], eq) if n == 1 else (outs, eq)


def test_forward_runs_and_is_finite():
    d, _ = _run_visco()
    assert np.isfinite(d).all()


def test_wavenumbers_stay_numpy_on_jax():
    """A jnp array cached on the instance would leak as a tracer into the next
    trace -- ``k`` must stay numpy, like the PML profiles."""
    _, eq = _run_visco()
    assert isinstance(eq.k, np.ndarray)
    expected = tuple(s + 2 * ABCN + eq.so for s in (NZ, NX))
    assert tuple(eq.k.shape[-2:]) == expected


def test_repeated_forward_on_one_instance():
    outs, _ = _run_visco(n=3)
    for i, d in enumerate(outs[1:], start=2):
        assert np.isfinite(d).all(), f"forward #{i} produced non-finite output"
        assert _rel_l2(d, outs[0]) == 0.0, f"forward #{i} drifted from #1"


def test_both_toggles_off_matches_acoustic():
    from sweep.equations import Acoustic
    from sweep.propagator.jax import PropJax

    d_visco, _ = _run_visco(phase_shift=False, amplitude_damping=False)

    ea = Acoustic(4, "cpu", "jax")
    pa = PropJax(ea, (NZ, NX), dh=DH, dt=DT, abcn=ABCN, use_ckpt=False)
    pa.set_parameters([_vp()])
    w, src, rec = _geom()
    o = pa.__call_forward__(w, src, rec, models=[jnp.asarray(_vp())])
    d_ac = np.asarray(o[0] if isinstance(o, (tuple, list)) else o)

    assert _rel_l2(d_visco, d_ac) == 0.0


def test_toggles_are_independent():
    out = {}
    for ps in (False, True):
        for ad in (False, True):
            out[(ps, ad)] = _run_visco(phase_shift=ps, amplitude_damping=ad)[0]
    keys = list(out)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            assert _rel_l2(out[a], out[b]) > 1e-6, f"{a} and {b} are identical"
