"""JAX boundary-saving wavefield reconstruction.

Validates the JAX boundary-saving memory strategy (``PropJax(memory=
MemoryOptions(strategy='boundary'))``) against the plain-autodiff scan
gradient, across the (exact) reverse-driver families — the JAX twin of
``test_eager_boundary_saving.py``:

* ``swap2nd``  — 2nd-order schemes (Acoustic / Acoustic3D): reverse reuses
  ``func`` with the time levels swapped.
* ``substep``  — 1st-order leapfrog (Acoustic1st / Elastic): reverse reuses the
  forward ``interior_substeps`` composed in reverse order at ``-dt``.

1st-order equations without ``interior_substeps`` are unsupported (no
approximate fallback) — the propagator raises.

Runs on CPU or GPU (whatever JAX picks up).  Import the source tree with
``PYTHONPATH=src`` or an editable install.
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

from sweep.equations import Acoustic, Acoustic3D, Acoustic1st, Elastic  # noqa: E402
from sweep.propagator.jax import PropJax  # noqa: E402
from sweep.propagator.options import BoundaryOptions, MemoryOptions  # noqa: E402

DT, DH, NT, SO, ABCN = 0.0015, 10.0, 64, 4, 20
BOUNDARY = MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="gpu"))


def _ricker(nt, dt, freq=10.0, delay=0.06, amp=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return (amp * (1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def _ramp(shape, top, bot):
    nz = shape[0]
    r = top + (bot - top) * np.linspace(0, 1, nz, dtype=np.float32)
    view = (slice(None),) + (None,) * (len(shape) - 1)
    return np.broadcast_to(r[view], shape).astype(np.float32).copy()


def _box(arr, v):
    out = arr.copy()
    sl = tuple(slice(s // 3, max(s // 3 + 2, 2 * s // 3)) if i == 0
               else slice(s // 4, max(s // 4 + 2, 3 * s // 4))
               for i, s in enumerate(arr.shape))
    out[sl] += v
    return out


def _models(kind, shape):
    vp_i = _ramp(shape, 1800.0, 2400.0)
    vp_t = _box(vp_i, 180.0)
    if kind == "acoustic":
        return [vp_t], [vp_i]
    if kind == "acoustic1st":
        rho = _ramp(shape, 1000.0, 1200.0)
        return [vp_t, rho], [vp_i, rho]
    # elastic
    vs_i = (vp_i / 1.73).astype(np.float32)
    vs_t = (vp_t / 1.73).astype(np.float32)
    rho = _ramp(shape, 1000.0, 1200.0)
    return [vp_t, vs_t, rho], [vp_i, vs_i, rho]


# A stress source's velocity records are ~1e-8 at unit amplitude (impedance
# scaling), under the fp32 noise floor — boost the elastic wavelet.
CASES = {
    "acoustic2d": dict(cls=Acoustic, ndim=2, st=["h1"], rt=["h1"], kind="acoustic",
                       driver="swap2nd", amp=1.0),
    "acoustic3d": dict(cls=Acoustic3D, ndim=3, st=["h1"], rt=["h1"], kind="acoustic",
                       driver="swap2nd", amp=1.0),
    "acoustic1st": dict(cls=Acoustic1st, ndim=2, st=["p"], rt=["p"], kind="acoustic1st",
                        driver="substep", amp=1.0),
    "elastic2d": dict(cls=Elastic, ndim=2, st=["sxx", "szz"], rt=["vx", "vz"],
                      kind="elastic", driver="substep", amp=1e6),
}


def _shape(ndim):
    return (32, 40) if ndim == 2 else (20, 18, 20)


def _geometry(ndim):
    radius = SO // 2
    shape = _shape(ndim)
    if ndim == 2:
        nz, nx = shape
        s = np.array([[[nx // 2, max(1, nz // 4)]]], dtype=np.int32)
        rx = np.arange(2, nx - 2, 6, dtype=np.int32)
        r = np.stack([rx, np.full(rx.size, radius, np.int32)], -1)[None]
        return s, r
    nz, ny, nx = shape
    s = np.array([[[nx // 2, ny // 2, max(1, nz // 4)]]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 6, dtype=np.int32)
    ry = np.arange(2, ny - 2, 6, dtype=np.int32)
    gy, gx = np.meshgrid(ry, rx, indexing="ij")
    r = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, radius, np.int32)], -1)[None]
    return s, r


def _build(cfg, *, memory=None, mode=None, use_ckpt=False, nt=NT):
    eq = cfg["cls"](spatial_order=SO, device="cpu", backend="jax")
    prop = PropJax(
        eq, _shape(cfg["ndim"]), memory=memory, use_ckpt=use_ckpt,
        dh=DH, dt=DT, source_type=cfg["st"], receiver_type=cfg["rt"],
        abcn=ABCN, pml_type=eq.default_pml_type, nt=nt, B=1,
    )
    if mode is not None:
        prop.use_ckpt = False
        prop.enable_boundary_saving(True, mode=mode)
    return prop


def _setup(cfg, nt=NT):
    shape = _shape(cfg["ndim"])
    true_m, init_m = _models(cfg["kind"], shape)
    wavelet = _ricker(nt, DT, amp=cfg["amp"])[None, :]
    src, rec = _geometry(cfg["ndim"])
    full = _build(cfg, nt=nt)
    obs = full(wavelet, src, rec, models=[jnp.asarray(a) for a in true_m])
    return init_m, wavelet, src, rec, obs


def _first_grad(prop, init_models, observed, wavelet, src, rec):
    rest = [jnp.asarray(a) for a in init_models[1:]]

    def loss(m0):
        out = prop(wavelet, src, rec, models=[m0] + rest)
        return jnp.mean((out - observed) ** 2)

    return jax.grad(loss)(jnp.asarray(init_models[0]))


def _cosine(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    na = max(np.linalg.norm(a), 1e-30)
    nb = max(np.linalg.norm(b), 1e-30)
    return float(np.dot(a / na, b / nb))


def _rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


@pytest.mark.parametrize("name", list(CASES))
def test_boundary_saving_matches_plain_autodiff(name):
    """Boundary-saving gradient == plain-autodiff gradient to ~fp32 precision.

    This also guards the reconstruction physics: the reverse drivers reuse
    ``func`` (swap2nd) or ``interior_substeps`` (substep), so a broken
    reconstruction would surface here."""
    cfg = CASES[name]
    init_m, wavelet, src, rec, obs = _setup(cfg)
    g_full = _first_grad(_build(cfg), init_m, obs, wavelet, src, rec)
    g_bs = _first_grad(_build(cfg, memory=BOUNDARY), init_m, obs, wavelet, src, rec)
    assert bool(jnp.isfinite(g_bs).all())
    assert float(jnp.linalg.norm(g_full)) > 0
    cos = _cosine(g_full, g_bs)
    rel = float(jnp.linalg.norm(g_bs - g_full) / jnp.linalg.norm(g_full))
    assert cos > 0.999, f"{name}: cosine {cos} (rel_l2 {rel})"
    assert rel < 1e-2, f"{name}: rel_l2 {rel} (cosine {cos})"


@pytest.mark.parametrize("storage_dtype,tol", [("fp16", 1e-2), ("bf16", 3e-2)])
def test_low_precision_ring_storage(storage_dtype, tol):
    """fp16/bf16 ring compression: compute stays fp32, only the saved strip is
    down-cast, so the gradient stays within the quantization floor."""
    cfg = CASES["acoustic2d"]
    init_m, wavelet, src, rec, obs = _setup(cfg)
    memory = MemoryOptions(strategy="boundary",
                           boundary=BoundaryOptions(storage_dtype=storage_dtype))
    g_full = _first_grad(_build(cfg), init_m, obs, wavelet, src, rec)
    g_bs = _first_grad(_build(cfg, memory=memory), init_m, obs, wavelet, src, rec)
    cos = _cosine(g_full, g_bs)
    rel = float(jnp.linalg.norm(g_bs - g_full) / jnp.linalg.norm(g_full))
    assert cos > 0.999, f"{storage_dtype}: cosine {cos}"
    assert rel < tol, f"{storage_dtype}: rel_l2 {rel}"


def test_multi_shot_batch():
    """Naive multi-shot (mode A2): the BS scan carries the batch dim untouched."""
    cfg = CASES["acoustic2d"]
    nz, nx = _shape(2)
    true_m, init_m = _models("acoustic", (nz, nx))
    wavelet = np.repeat(_ricker(NT, DT)[None, :], 2, axis=0)
    src = np.array([[nx // 3, nz // 4], [2 * nx // 3, nz // 4]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 6, dtype=np.int32)
    r1 = np.stack([rx, np.full(rx.size, SO // 2, np.int32)], -1)
    rec = np.stack([r1, r1], 0)

    full = _build(cfg)
    obs = full(wavelet, src, rec, models=[jnp.asarray(true_m[0])])
    g_full = _first_grad(_build(cfg), init_m, obs, wavelet, src, rec)
    g_bs = _first_grad(_build(cfg, memory=BOUNDARY), init_m, obs, wavelet, src, rec)
    assert _cosine(g_full, g_bs) > 0.999


def test_wavelet_gradient():
    """Source-inversion path: the custom VJP assembles per-step wavelet
    cotangents from the reverse scan — compare against plain autodiff."""
    cfg = CASES["acoustic2d"]
    init_m, wavelet, src, rec, obs = _setup(cfg)
    models = [jnp.asarray(init_m[0])]

    def make_loss(prop):
        def loss(w):
            return jnp.mean((prop(w, src, rec, models=models) - obs) ** 2)
        return loss

    g_full = jax.grad(make_loss(_build(cfg)))(jnp.asarray(wavelet))
    g_bs = jax.grad(make_loss(_build(cfg, memory=BOUNDARY)))(jnp.asarray(wavelet))
    assert float(jnp.linalg.norm(g_full)) > 0
    assert _cosine(g_full, g_bs) > 0.999
    rel = float(jnp.linalg.norm(g_bs - g_full) / jnp.linalg.norm(g_full))
    assert rel < 1e-2, f"wavelet grad rel_l2 {rel}"


def test_source_encoding_mode():
    """Mode B (3-D ``sources=(1, nsrc, ndim)``, per-source wavelets): the BS
    custom VJP must match plain autodiff.  The JAX source has no explicit
    encoding branch — ``SourceJax.multiwavelet`` covers it through fancy-index
    broadcasting ((1,) shots vs nsrc points); this test guards that behavior
    (verified against torch eager at rel ~5e-7 / grad cosine 1.0)."""
    cfg = CASES["acoustic2d"]
    nz, nx = _shape(2)
    true_m, init_m = _models("acoustic", (nz, nx))
    nsrc = 3
    scales = np.array([1.0, -0.7, 0.45], np.float32)
    wavelet = scales[:, None] * _ricker(NT, DT)[None, :]
    src = np.array([[[nx // 4, 3], [nx // 2, 3], [3 * nx // 4, 3]]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 6, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, SO // 2, np.int32)], -1)[None]

    obs = _build(cfg)(wavelet, src, rec, models=[jnp.asarray(true_m[0])])
    g_full = _first_grad(_build(cfg), init_m, obs, wavelet, src, rec)
    g_bs = _first_grad(_build(cfg, memory=BOUNDARY), init_m, obs, wavelet, src, rec)
    r_full = _build(cfg)(wavelet, src, rec, models=[jnp.asarray(init_m[0])])
    r_bs = _build(cfg, memory=BOUNDARY)(wavelet, src, rec, models=[jnp.asarray(init_m[0])])
    assert _rel_l2(r_bs, r_full) < 1e-6
    assert float(jnp.linalg.norm(g_full)) > 0
    assert _cosine(g_full, g_bs) > 0.999


def test_scan_unroll_preserves_gradients():
    """scan_unroll only restructures the loop — gradients must agree up to
    XLA float reassociation (bit-identical on GPU, reassociated on CPU), on
    the plain path and through the BS custom VJP alike."""
    cfg = CASES["acoustic2d"]
    init_m, wavelet, src, rec, obs = _setup(cfg)

    def grad_with(unroll, memory):
        eq = cfg["cls"](spatial_order=SO, device="cpu", backend="jax")
        prop = PropJax(eq, _shape(2), memory=memory, use_ckpt=False,
                       dh=DH, dt=DT, source_type=cfg["st"], receiver_type=cfg["rt"],
                       abcn=ABCN, nt=NT, B=1, scan_unroll=unroll)
        return _first_grad(prop, init_m, obs, wavelet, src, rec)

    for memory in (None, BOUNDARY):
        g1 = grad_with(1, memory)
        g4 = grad_with(4, memory)
        rel = _rel_l2(g4, g1)
        assert rel < 1e-5, f"unroll changed the gradient (memory={memory}): rel {rel}"


def test_forward_matches_plain_path():
    """The BS forward is the same physics as the plain scan — records match."""
    cfg = CASES["acoustic2d"]
    init_m, wavelet, src, rec, _ = _setup(cfg)
    models = [jnp.asarray(init_m[0])]
    r_full = _build(cfg)(wavelet, src, rec, models=models)
    r_bs = _build(cfg, memory=BOUNDARY)(wavelet, src, rec, models=models)
    rel = float(jnp.linalg.norm(r_bs - r_full) / (jnp.linalg.norm(r_full) + 1e-30))
    assert rel < 1e-6


def test_memory_analysis_shows_reduction():
    """XLA-accounted temp memory of the gradient must drop by >5x vs the plain
    scan tape (compile-time accounting — nothing executed, so a grid where the
    ring perimeter is actually small relative to the area is cheap to check)."""
    nz, nx, nt = 128, 128, 256
    eq_kw = dict(spatial_order=SO, device="cpu", backend="jax")
    prop_kw = dict(dh=DH, dt=DT, source_type=["h1"], receiver_type=["h1"],
                   abcn=ABCN, nt=nt, B=1)
    vp = _ramp((nz, nx), 1800.0, 2400.0)
    wavelet = _ricker(nt, DT)[None, :]
    src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 6, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, SO // 2, np.int32)], -1)[None]

    def temp_bytes(memory):
        prop = PropJax(Acoustic(**eq_kw), (nz, nx), memory=memory, **prop_kw)

        def loss(m0):
            return jnp.mean(prop(wavelet, src, rec, models=[m0]) ** 2)

        return jax.jit(jax.grad(loss)).lower(jnp.asarray(vp)).compile() \
            .memory_analysis().temp_size_in_bytes

    full = temp_bytes(None)
    bs = temp_bytes(BOUNDARY)
    assert bs * 5 < full, f"temp memory: full {full} vs bs {bs}"


def test_driver_auto_dispatch():
    """2nd-order -> swap2nd, 1st-order with substeps -> substep."""
    from sweep.propagator._jax_boundary_saving import resolve_reverse_mode
    assert resolve_reverse_mode(Acoustic(SO, "cpu", "jax")) == "swap2nd"
    eq1 = Acoustic1st(SO, "cpu", "jax")
    assert resolve_reverse_mode(eq1) == "substep"


def test_forced_substep_without_hook_raises():
    with pytest.raises(ValueError, match="interior_substeps"):
        prop = _build(CASES["acoustic2d"], memory=BOUNDARY, mode="substep")
        init_m, wavelet, src, rec, _ = _setup(CASES["acoustic2d"])
        prop(wavelet, src, rec, models=[jnp.asarray(init_m[0])])


def test_int8_storage_rejected():
    with pytest.raises(NotImplementedError, match="int8"):
        _build(CASES["acoustic2d"],
               memory=MemoryOptions(strategy="boundary",
                                    boundary=BoundaryOptions(storage_dtype="int8")))


def test_cpu_ring_storage_rejected():
    with pytest.raises(ValueError, match="storage"):
        _build(CASES["acoustic2d"],
               memory=MemoryOptions(strategy="boundary",
                                    boundary=BoundaryOptions(storage="cpu")))


def test_mutually_exclusive_with_ckpt():
    prop = _build(CASES["acoustic2d"], memory=BOUNDARY)
    prop.use_ckpt = True
    init_m, wavelet, src, rec, _ = _setup(CASES["acoustic2d"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        prop(wavelet, src, rec, models=[jnp.asarray(init_m[0])])


def test_return_wavefield_rejected():
    prop = _build(CASES["acoustic2d"], memory=BOUNDARY)
    init_m, wavelet, src, rec, _ = _setup(CASES["acoustic2d"])
    with pytest.raises(ValueError, match="return_wavefield"):
        prop(wavelet, src, rec, models=[jnp.asarray(init_m[0])],
             return_wavefield=True)
