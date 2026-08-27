"""RSG checkerboard stencil on the jax front-end.

Lives in its own file (no torch import) so it actually runs in jax-only
environments — ``test_elastic_tti.py`` is gated behind ``importorskip('torch')``.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from sweep.equations import ElasticTTI, ElasticTTISG
from sweep.sources.jax import SourceJax
from sweep.receivers.jax import ReceiverJax


def test_stencil_is_backend_neutral():
    k = ElasticTTI(spatial_order=4, backend="jax", device="cpu").source_receiver_stencil
    assert isinstance(k, np.ndarray), f"stencil must stay backend-neutral, got {type(k)}"
    assert k.shape == (3, 3) and abs(float(k.sum()) - 1.0) < 1e-6
    sign = np.array([[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]])
    assert abs(float((k * sign).sum())) < 1e-7
    assert ElasticTTISG(spatial_order=4, backend="jax", device="cpu").source_receiver_stencil is None
    assert ElasticTTI(spatial_order=4, backend="jax", device="cpu",
                      checkerboard_smoothing=False).source_receiver_stencil is None


def test_source_jax_spreads_with_stencil():
    k = ElasticTTI(spatial_order=4, backend="jax", device="cpu").source_receiver_stencil
    shape = (1, 1, 16, 16)
    coords = jnp.asarray(np.array([[[8, 8]]], dtype=np.int32))
    s = SourceJax(coords, shape, False, False, spread_kernel=k)
    assert s.spread_offsets is not None and len(s.spread_offsets) == 9
    out = np.asarray(s(jnp.zeros(shape), jnp.asarray(np.array([1.0], np.float32))))
    assert abs(out.sum() - 1.0) < 1e-5           # injection is conserved
    assert abs(out[0, 0, 8, 8] - 0.25) < 1e-5    # binomial weights, not a spike
    assert abs(out[0, 0, 7, 8] - 0.125) < 1e-5

    plain = np.asarray(SourceJax(coords, shape, False, False)(
        jnp.zeros(shape), jnp.asarray(np.array([1.0], np.float32))))
    assert abs(plain[0, 0, 8, 8] - 1.0) < 1e-6   # default stays single-cell


def test_receiver_jax_rejects_checkerboard():
    k = ElasticTTI(spatial_order=4, backend="jax", device="cpu").source_receiver_stencil
    shape = (1, 1, 16, 16)
    coords = jnp.asarray(np.array([[[8, 8]]], dtype=np.int32))
    zz, xx = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    checker = jnp.asarray(((-1.0) ** (zz + xx)).astype(np.float32).reshape(shape))

    r = ReceiverJax(coords, gather_kernel=k)
    assert r.gather_offsets is not None and len(r.gather_offsets) == 9
    assert abs(float(np.asarray(r(checker)).ravel()[0])) < 1e-6
    # the plain gather sees the full spurious amplitude
    assert abs(float(np.asarray(ReceiverJax(coords)(checker)).ravel()[0])) > 0.9


def test_rsg_traces_match_sg_with_smoothing_jax():
    """End-to-end on the jax propagator: smoothed RSG must track the
    axis-aligned SG reference; raw sampling must not."""
    from sweep.propagator.jax import PropJax

    shape = (100, 100)
    dh, dt, nt, abcn = 5.0, 5e-4, 200, 16
    f0, delay = 20.0, 0.05
    t = np.arange(nt, dtype=np.float32) * dt - delay
    wavelet = (1e3 * (1 - 2 * (np.pi * f0 * t) ** 2)
               * np.exp(-((np.pi * f0 * t) ** 2))).astype(np.float32)
    nz, nx = shape
    src = np.array([[nx // 2, nz // 2]], dtype=np.int32)
    rec = np.array([[[nx // 2, nz // 2 + 20], [nx // 2 + 20, nz // 2]]], dtype=np.int32)
    vals = [2500.0, 1300.0, 2000.0, 0.20, 0.08, 0.0, 0.5236, 0.0]

    def _run(eq):
        prop = PropJax(eq, shape, dh=dh, dt=dt, nt=nt, abcn=abcn, use_ckpt=False,
                       pml_type=eq.default_pml_type, backend="jax",
                       source_type=["vz"], receiver_type=["vz"])
        models = [jnp.asarray(np.full(shape, v, np.float32)) for v in vals]
        out = prop.__call_forward__(wavelet, src, rec, models=models)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return np.asarray(out).squeeze(0).squeeze(-1).T

    ref = _run(ElasticTTISG(spatial_order=4, backend="jax", device="cpu"))
    on = _run(ElasticTTI(spatial_order=4, backend="jax", device="cpu"))
    off = _run(ElasticTTI(spatial_order=4, backend="jax", device="cpu",
                          checkerboard_smoothing=False))

    for k_ in range(ref.shape[0]):
        a, b = ref[k_], on[k_]
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
        assert cos > 0.98, f"receiver {k_}: smoothed RSG vs SG cos={cos:.4f}"
    a, b = ref[1], off[1]
    rel_off = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    assert rel_off > 0.3, f"legacy raw sampling unexpectedly clean: rel={rel_off:.3f}"
