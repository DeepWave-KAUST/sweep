"""jax.jit tracing fixes for the JAX propagator path.

Two pre-existing bugs, both jit-tracing related:

1. ``edge_pad`` was a ``jax.custom_vjp`` WITHOUT ``nondiff_argnums``: a
   pure-forward ``jax.jit`` (no grad) traces the primal with every argument
   lifted to a tracer — including the Python ``pad_width`` tuple — and
   ``jnp.pad`` raises ConcretizationTypeError.  (Under grad / jit(grad) the
   fwd/bwd rules run instead of the primal, which is why only the *simpler*
   transform failed.)  Fixed with ``nondiff_argnums=(1,)``.

2. A ``PropJax`` instance could only ever be traced once: ``init_abc`` runs
   inside the user's jit trace and cached jnp-converted PML profiles (tracers)
   on the equation, guarded by a pure-Python cache key — the second trace hit
   the cache and read the first trace's stale tracer (UnexpectedTracerError,
   leaked value created in ``to_backend``).  Fixed by keeping the JAX profiles
   as numpy (trace-safe closure constants in every trace).

Runs on CPU or GPU.  Import the source tree with ``PYTHONPATH=src``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "sweep").exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sweep.equations import Acoustic  # noqa: E402
from sweep.propagator.jax import PropJax  # noqa: E402
from sweep.utils.jax import edge_pad  # noqa: E402


# ---- edge_pad under every transform -----------------------------------------

PW = ((1, 1), (2, 2))


def test_edge_pad_pure_forward_jit():
    u = jnp.arange(16.0).reshape(4, 4)
    ref = np.pad(np.asarray(u), PW, mode="edge")
    assert np.array_equal(edge_pad(u, PW), ref)                       # eager
    assert np.array_equal(jax.jit(lambda x: edge_pad(x, PW))(u), ref)  # jit


def test_edge_pad_grad_consistent():
    u = jnp.arange(16.0).reshape(4, 4)

    def loss(x):
        return (edge_pad(x, PW) * 2.0).sum()

    g = jax.grad(loss)(u)
    gj = jax.jit(jax.grad(loss))(u)
    # bwd slices the interior of the output cotangent back out.
    gref = (2.0 * np.ones((6, 8)))[1:5, 2:6]
    assert np.array_equal(g, gref)
    assert np.array_equal(gj, gref)


# ---- one propagator instance, many traces -----------------------------------

def _setup():
    nz, nx = 48, 56
    vp = (1800.0 + 600.0 * np.linspace(0, 1, nz)[:, None]
          * np.ones((1, nx))).astype(np.float32)
    t = np.arange(120, dtype=np.float32) * 0.0015
    x = np.pi * 10.0 * (t - 0.06)
    w = ((1 - 2 * x * x) * np.exp(-x * x)).astype(np.float32)[None, :]
    src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 6, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, 2, np.int32)], -1)[None]
    prop = PropJax(Acoustic(4, "cpu", "jax"), (nz, nx),
                   dh=10.0, dt=0.0015, abcn=30, use_ckpt=False)
    prop.set_parameters([vp])
    return prop, jnp.asarray(vp), w, src, rec


def test_propagator_traced_multiple_times():
    """init_abc's cached PML profiles must survive re-tracing: two distinct
    jit wrappers, a reused jit function, and forward-then-grad on ONE
    instance."""
    prop, m0, w, src, rec = _setup()
    r1 = jax.jit(lambda m: prop.__call_forward__(w, src, rec, models=[m]))(m0)
    r2 = jax.jit(lambda m: prop.__call_forward__(w, src, rec, models=[m]))(m0)
    f = jax.jit(lambda m: prop.__call_forward__(w, src, rec, models=[m]))
    r3 = f(m0)
    r4 = f(m0)
    for r in (r2, r3, r4):
        assert np.array_equal(np.asarray(r1), np.asarray(r))
    g = jax.jit(jax.grad(
        lambda m: jnp.sum(prop(w, src, rec, models=[m]) ** 2)))(m0)
    assert bool(jnp.isfinite(g).all()) and float(jnp.linalg.norm(g)) > 0


def _rel_l2(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)


def test_pure_forward_jit_via_call():
    """With edge_pad fixed, jit of the plain __call__ (model padding included)
    works — no __call_forward__ detour needed.  jit vs eager differ only by
    XLA float reassociation (~1e-7 rel after 120 steps)."""
    prop, m0, w, src, rec = _setup()
    r_jit = jax.jit(lambda m: prop(w, src, rec, models=[m]))(m0)
    r_eager = prop(w, src, rec, models=[m0])
    assert _rel_l2(r_jit, r_eager) < 1e-5


def test_jit_gradient_matches_eager():
    """Numerics guard: the numpy PML profiles must not change values (up to
    XLA float reassociation)."""
    prop, m0, w, src, rec = _setup()

    def loss(m):
        return jnp.sum(prop(w, src, rec, models=[m]) ** 2)

    g_eager = jax.grad(loss)(m0)
    g_jit = jax.jit(jax.grad(loss))(m0)
    assert _rel_l2(g_jit, g_eager) < 1e-5
