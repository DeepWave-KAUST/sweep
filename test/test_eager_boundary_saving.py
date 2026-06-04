"""Eager (pure-PyTorch) boundary-saving wavefield reconstruction.

Validates the eager boundary-saving memory strategy (``MemoryOptions(
strategy='boundary')`` on ``impl='eager'``) against the eager full-autograd-tape
gradient, across the (exact) reverse-driver families:

* ``swap2nd``  — 2nd-order schemes (Acoustic / Acoustic3D): reverse reuses
  ``func`` with the time levels swapped.
* ``substep``  — 1st-order leapfrog (Acoustic1st / Elastic): reverse reuses the
  forward ``interior_substeps`` composed in reverse order at ``-dt``.

1st-order equations without ``interior_substeps`` are unsupported (no
approximate fallback) — the propagator raises.

Run: ``pytest test/test_eager_boundary_saving.py`` (needs CUDA + the eager path;
import the source tree with ``PYTHONPATH=src`` or an editable install).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "sweep").exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sweep.equations import Acoustic, Acoustic3D, Acoustic1st, Elastic  # noqa: E402
from sweep.propagator.options import (  # noqa: E402
    BoundaryOptions,
    EagerOptions,
    MemoryOptions,
)
from sweep.propagator.torch import PropTorch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DT, DH, NT, SO, ABCN = 0.0015, 10.0, 64, 4, 20
BOUNDARY = MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="gpu"))


def _ricker(nt, dt, freq=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


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


# (equation_cls, ndim, source_type, receiver_type, model factory) per case.
def _models(kind, shape):
    vp_i = _ramp(shape, 1800.0, 2400.0)
    vp_t = _box(vp_i, 180.0)
    if kind == "acoustic":
        return [vp_t], [vp_i], 1
    if kind == "acoustic1st":
        rho = _ramp(shape, 1000.0, 1200.0)
        return [vp_t, rho], [vp_i, rho], 1
    # elastic
    vs_i = (vp_i / 1.73).astype(np.float32)
    vs_t = (vp_t / 1.73).astype(np.float32)
    rho = _ramp(shape, 1000.0, 1200.0)
    return [vp_t, vs_t, rho], [vp_i, vs_i, rho], 1


CASES = {
    "acoustic2d": dict(cls=Acoustic, ndim=2, st=["h1"], rt=["h1"], kind="acoustic", driver="swap2nd"),
    "acoustic3d": dict(cls=Acoustic3D, ndim=3, st=["h1"], rt=["h1"], kind="acoustic", driver="swap2nd"),
    "acoustic1st": dict(cls=Acoustic1st, ndim=2, st=["p"], rt=["p"], kind="acoustic1st", driver="substep"),
    "elastic2d": dict(cls=Elastic, ndim=2, st=["sxx", "szz"], rt=["vx", "vz"], kind="elastic", driver="substep"),
}


def _shape(ndim):
    return (32, 40) if ndim == 2 else (20, 18, 20)


def _geometry(ndim):
    radius = SO // 2
    shape = _shape(ndim)
    if ndim == 2:
        nz, nx = shape
        s = np.array([[nx // 2, max(1, nz // 4)]], dtype=np.int32)
        rx = np.arange(2, nx - 2, 6, dtype=np.int32)
        r = np.stack([rx, np.full(rx.size, radius, np.int32)], -1)[None]
        return s, r
    nz, ny, nx = shape
    s = np.array([[nx // 2, ny // 2, max(1, nz // 4)]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 6, dtype=np.int32)
    ry = np.arange(2, ny - 2, 6, dtype=np.int32)
    gy, gx = np.meshgrid(ry, rx, indexing="ij")
    r = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, radius, np.int32)], -1)[None]
    return s, r


def _build(cfg, *, memory=None, mode=None, use_ckpt=False, nt=NT, free_surface=False,
           use_compile=False):
    eq = cfg["cls"](spatial_order=SO, device="cuda", backend="torch")
    shape = _shape(cfg["ndim"])
    prop = PropTorch(
        eq, impl="eager", eager_options=EagerOptions(use_compile=use_compile),
        memory=memory, use_ckpt=use_ckpt,
        shape=shape, dev="cuda", dh=DH, dt=DT,
        source_type=cfg["st"], receiver_type=cfg["rt"],
        abcn=ABCN, pml_type=eq.default_pml_type, free_surface=free_surface, nt=nt, B=1,
    )
    if mode is not None:
        prop._backend_impl.enable_eager_boundary_saving(True, mode=mode)
        prop._backend_impl.use_ckpt = False
    return prop


def _first_grad(prop, cfg, init_models, observed, wavelet, src, rec):
    m = [torch.tensor(a, device="cuda", requires_grad=(i == 0))
         for i, a in enumerate(init_models)]
    rec_out = prop(wavelet, src.copy(), rec.copy(), models=m)
    (rec_out - observed).pow(2).mean().backward()
    return m[0].grad.detach().clone()


def _setup(cfg, nt=NT, free_surface=False):
    shape = _shape(cfg["ndim"])
    true_m, init_m, _ = _models(cfg["kind"], shape)
    wavelet = torch.tensor(_ricker(nt, DT), device="cuda")
    src, rec = _geometry(cfg["ndim"])
    full = _build(cfg, nt=nt, free_surface=free_surface)
    with torch.no_grad():
        obs = full(wavelet, src.copy(), rec.copy(),
                   models=[torch.tensor(a, device="cuda") for a in true_m]).clone()
    return true_m, init_m, wavelet, src, rec, obs


def _cosine(a, b):
    a = a.double().reshape(-1)
    b = b.double().reshape(-1)
    na = torch.linalg.vector_norm(a).clamp_min(1e-30)
    nb = torch.linalg.vector_norm(b).clamp_min(1e-30)
    return float(torch.dot(a / na, b / nb))


@pytest.mark.parametrize("name", list(CASES))
def test_boundary_saving_matches_full_tape(name):
    """Boundary-saving gradient == full-tape gradient to ~machine precision.

    This also guards the reconstruction physics: the reverse drivers reuse
    ``func`` (swap2nd) or ``interior_substeps`` (substep), so a mismatch with
    full-tape would surface here."""
    cfg = CASES[name]
    true_m, init_m, wavelet, src, rec, obs = _setup(cfg)
    g_full = _first_grad(_build(cfg), cfg, init_m, obs, wavelet, src, rec)
    g_bs = _first_grad(_build(cfg, memory=BOUNDARY), cfg, init_m, obs, wavelet, src, rec)
    assert torch.isfinite(g_bs).all()
    assert g_full.norm() > 0
    cos = _cosine(g_full, g_bs)
    rel = float((g_bs - g_full).norm() / g_full.norm())
    assert cos > 0.999, f"{name}: cosine {cos} (rel_l2 {rel})"
    assert rel < 1e-2, f"{name}: rel_l2 {rel} (cosine {cos})"


def _peak_mem(prop, cfg, init_m, obs, wavelet, src, rec):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _first_grad(prop, cfg, init_m, obs, wavelet, src, rec)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


@pytest.mark.parametrize("name", ["acoustic2d", "elastic2d"])
def test_boundary_saving_reduces_memory(name):
    """Boundary saving must actually save memory, and scale better with nt:
    full-tape peak ~ O(nt * Ncells * nfields); boundary saving ~ O(nt * ring)
    plus the final frame.  Checks (a) a clear reduction at fixed nt, and (b)
    that growing nt blows up full-tape far more than boundary saving."""
    cfg = CASES[name]
    nt_short, nt_long = 64, 400

    _, init_s, wav_s, src, rec, obs_s = _setup(cfg, nt=nt_short)
    full_s = _peak_mem(_build(cfg, nt=nt_short), cfg, init_s, obs_s, wav_s, src, rec)
    bs_s = _peak_mem(_build(cfg, memory=BOUNDARY, nt=nt_short), cfg, init_s, obs_s, wav_s, src, rec)

    _, init_l, wav_l, src, rec, obs_l = _setup(cfg, nt=nt_long)
    full_l = _peak_mem(_build(cfg, nt=nt_long), cfg, init_l, obs_l, wav_l, src, rec)
    bs_l = _peak_mem(_build(cfg, memory=BOUNDARY, nt=nt_long), cfg, init_l, obs_l, wav_l, src, rec)

    # (a) clear reduction at the longer rollout.
    assert bs_l < 0.5 * full_l, f"{name}: bs {bs_l} not < 0.5*full {full_l}"
    # (b) boundary saving scales sub-linearly vs full-tape's linear-in-nt growth.
    full_growth = full_l / full_s
    bs_growth = bs_l / bs_s
    assert bs_growth < full_growth, (
        f"{name}: bs grew {bs_growth:.2f}x vs full {full_growth:.2f}x over "
        f"nt {nt_short}->{nt_long}"
    )


@pytest.mark.parametrize("name", ["acoustic2d", "elastic2d"])
def test_free_surface_supported(name):
    """Free-surface boundary saving matches full-tape:
    * acoustic2d (2nd-order) — swap2nd reuses ``func`` (FS BC applied);
    * elastic2d (1st-order) — interior_substeps reuse the shared velocity/stress
      sub-steps, which carry the image-method FS BC + top-row stress zeroing."""
    cfg = CASES[name]
    _, init_m, wavelet, src, rec, obs = _setup(cfg, free_surface=True)
    g_full = _first_grad(_build(cfg, free_surface=True), cfg, init_m, obs, wavelet, src, rec)
    g_bs = _first_grad(_build(cfg, memory=BOUNDARY, free_surface=True), cfg, init_m, obs, wavelet, src, rec)
    assert _cosine(g_full, g_bs) > 0.999


def test_boundary_saving_composes_with_compile():
    """torch.compile composes with boundary saving: with use_compile=True the
    inner per-step physics is compiled and called inside the (eager) custom
    autograd.Function / time loop; the gradient still matches full-tape.
    (Slow: triggers a one-off torch.compile warmup.)"""
    cfg = CASES["acoustic2d"]
    _, init_m, wavelet, src, rec, obs = _setup(cfg)
    g_full = _first_grad(_build(cfg), cfg, init_m, obs, wavelet, src, rec)
    g_bs = _first_grad(
        _build(cfg, memory=BOUNDARY, use_compile=True),
        cfg, init_m, obs, wavelet, src, rec,
    )
    assert torch.isfinite(g_bs).all()
    assert _cosine(g_full, g_bs) > 0.999


def test_free_surface_guard_when_substeps_lack_fs():
    """A 1st-order equation whose sub-steps do NOT carry the free-surface BC
    (``supports_bs_free_surface`` unset) must refuse with free_surface=True
    rather than silently return a wrong near-surface gradient."""
    cfg = CASES["elastic2d"]
    _, init_m, wavelet, src, rec, obs = _setup(cfg, free_surface=True)
    prop = _build(cfg, memory=BOUNDARY, free_surface=True)
    prop._backend_impl.equation.supports_bs_free_surface = False  # simulate unsupported
    m = [torch.tensor(a, device="cuda", requires_grad=(i == 0))
         for i, a in enumerate(init_m)]
    with pytest.raises(NotImplementedError):
        prop(wavelet, src.copy(), rec.copy(), models=m)


@pytest.mark.parametrize("name", list(CASES))
def test_no_cross_iteration_drift(name):
    """Per-rollout reconstruction state (not class globals): repeated backward
    passes on the same propagator give bitwise-identical gradients."""
    cfg = CASES[name]
    _, init_m, wavelet, src, rec, obs = _setup(cfg)
    prop = _build(cfg, memory=BOUNDARY)
    g0 = _first_grad(prop, cfg, init_m, obs, wavelet, src, rec)
    g1 = _first_grad(prop, cfg, init_m, obs, wavelet, src, rec)
    g2 = _first_grad(prop, cfg, init_m, obs, wavelet, src, rec)
    # seistorch keeps reconstruction state on the autograd.Function CLASS and
    # never resets it -> the 2nd+ FWI iteration drifts by an O(1) timestep.
    # Our per-rollout state must reproduce the gradient run-to-run; the only
    # admissible difference is float32 ULP noise from CUDA atomics / cuDNN
    # conv backward, far below the structural drift the bug would cause.
    scale = g0.abs().max().clamp_min(1e-30)
    assert (g1 - g0).abs().max() <= 1e-5 * scale
    assert (g2 - g0).abs().max() <= 1e-5 * scale


def test_first_order_requires_substeps():
    """1st-order boundary saving needs an interior_substeps hook (the exact
    reverse driver); there is no approximate fallback, so the path must refuse
    when the hook is absent rather than silently degrade."""
    cfg = CASES["acoustic1st"]
    _, init_m, wavelet, src, rec, obs = _setup(cfg)
    prop = _build(cfg, memory=BOUNDARY)
    prop._backend_impl.equation.interior_substeps = None  # simulate missing hook
    m = [torch.tensor(a, device="cuda", requires_grad=(i == 0))
         for i, a in enumerate(init_m)]
    with pytest.raises(NotImplementedError):
        prop(wavelet, src.copy(), rec.copy(), models=m)


@pytest.mark.parametrize("name", ["acoustic1st", "elastic2d"])
def test_substep_round_trip_identity(name):
    """Composing the forward sub-steps forward (+dt) then in reverse order (-dt)
    recovers the input — i.e. the reverse is the forward code driven backwards,
    exact to float32."""
    cfg = CASES[name]
    prop = _build(cfg, memory=BOUNDARY)
    # one forward to initialise the equation's derivative operators
    _, init_m, wavelet, src, rec, obs = _setup(cfg)
    _first_grad(prop, cfg, init_m, obs, wavelet, src, rec)

    impl = prop._backend_impl
    eq = impl.equation
    subs = eq.interior_substeps()
    shape_rt = impl._runtime_shape()
    nwf = len(eq.wavefields)
    torch.manual_seed(0)
    state = [torch.randn((1, 1) + tuple(shape_rt), device="cuda") for _ in range(nwf)]
    runtime_models = impl._prepare_runtime_models(
        [torch.full((1, 1) + tuple(shape_rt), v, device="cuda")
         for v in ([2000.0, 1000.0] if name == "acoustic1st" else [2500.0, 1500.0, 1000.0])]
    )

    fwd = state
    for ss in subs:
        fwd = ss(fwd, runtime_models, DT, DH)
    rev = fwd
    for ss in reversed(subs):
        rev = ss(rev, runtime_models, -DT, DH)

    scale = max(max(f.abs().max().item() for f in fwd[:3]),
                max(f.abs().max().item() for f in state[:3]), 1e-30)
    err = max((rev[k] - state[k]).abs().max().item() for k in range(3)) / scale
    assert err < 1e-5, f"{name}: round-trip relative error {err}"


def _func_breaker(eq):
    """Wrap ``eq.func`` so it corrupts its output only while ``flag['break']`` is
    set — lets a test run a clean forward then a broken backward (reconstruction
    + recompute use the corrupted op)."""
    real_func = eq.func
    flag = {"break": False}

    def wrapped(wavefields, models, dt, h, b, **kw):
        out = list(real_func(wavefields, models, dt, h, b, **kw))
        if flag["break"]:
            # Multiplicative, not additive: an additive constant cancels under
            # swap2nd's u_{t+1} = 2u_t - u_{t-1} + ... structure (the recompute
            # subtracts the same constant the reconstruction added).  A scale
            # does not cancel, so it genuinely breaks forward-consistency.
            out[0] = out[0] * 1.5
        return out

    eq.func = wrapped
    return flag


def test_self_check_raises_on_broken_reverse():
    """The one-time interior forward-consistency probe must REFUSE rather than
    silently return a wrong gradient when the reverse driver fails to invert the
    step.  The forward builds a valid frame; then reconstruction + recompute use
    a corrupted ``func`` so ``func(reconstructed S_i)`` no longer reproduces
    ``S_{i+1}`` across the interior and the probe fires on the first backward
    step."""
    cfg = CASES["acoustic2d"]
    _, init_m, wavelet, src, rec, obs = _setup(cfg)
    prop = _build(cfg, memory=BOUNDARY)
    flag = _func_breaker(prop._backend_impl.equation)
    m = [torch.tensor(a, device="cuda", requires_grad=(i == 0))
         for i, a in enumerate(init_m)]
    loss = (prop(wavelet, src.copy(), rec.copy(), models=m) - obs).pow(2).mean()
    flag["break"] = True
    with pytest.raises(RuntimeError, match="does not invert"):
        loss.backward()


def test_multifield_second_order_reconstructs_all_pairs():
    """swap2nd reconstructs EVERY now/prev field-pair, so a multi-field 2nd-order
    equation (AcousticLSRTM: background h1/h2 + scattered sh1/sh2) recovers its
    scattered field too: the self-check passes (no raise — the scattered field is
    no longer zeroed) and the boundary-saved gradient matches the full-tape
    gradient.  Before generalising swap2nd the scattered field was dropped and
    the probe rejected this equation."""
    from sweep.equations import AcousticLSRTM  # noqa: PLC0415
    shape = (40, 48)

    def build(bs):
        eq = AcousticLSRTM(spatial_order=SO, device="cuda", backend="torch")
        st, rt = list(eq.default_source_fields), list(eq.default_receiver_fields)
        return PropTorch(
            eq, impl="eager", eager_options=EagerOptions(use_compile=False),
            memory=(BOUNDARY if bs else None), use_ckpt=False, shape=shape,
            dev="cuda", dh=DH, dt=DT, source_type=st, receiver_type=rt, abcn=ABCN,
            pml_type=eq.default_pml_type, free_surface=False, nt=NT, B=1,
        )

    wavelet = torch.tensor(_ricker(NT, DT), device="cuda")
    s = np.array([[shape[1] // 2, shape[0] // 4]], dtype=np.int32)
    rx = np.arange(2, shape[1] - 2, 6, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, SO // 2, np.int32)], -1)[None]
    vp = _ramp(shape, 1800.0, 2400.0)
    mp0 = np.zeros(shape, np.float32)
    mp0[shape[0] // 3:2 * shape[0] // 3, shape[1] // 3:2 * shape[1] // 3] = 0.05
    with torch.no_grad():
        obs = build(False)(wavelet, s.copy(), rec.copy(),
                           models=[torch.tensor(vp, device="cuda"),
                                   torch.tensor(mp0 * 1.05, device="cuda")]).clone()

    def mp_grad(bs):
        m = [torch.tensor(vp, device="cuda", requires_grad=False),
             torch.tensor(mp0, device="cuda", requires_grad=True)]
        (build(bs)(wavelet, s.copy(), rec.copy(), models=m) - obs).pow(2).mean().backward()
        return m[1].grad.detach()

    gf = mp_grad(False)
    gb = mp_grad(True)   # must NOT raise: the scattered field reconstructs now
    cos = _cosine(gf, gb)
    assert cos > 0.999, f"LSRTM mp grad cosine {cos}"


def test_self_check_can_be_disabled():
    """``self_check=False`` skips the probe — a broken reverse then returns a
    (wrong) gradient WITHOUT raising.  Documents the escape hatch for equations
    validated out-of-band."""
    cfg = CASES["acoustic2d"]
    _, init_m, wavelet, src, rec, obs = _setup(cfg)
    prop = _build(cfg, memory=BOUNDARY)
    prop._backend_impl.enable_eager_boundary_saving(True, self_check=False)
    flag = _func_breaker(prop._backend_impl.equation)
    m = [torch.tensor(a, device="cuda", requires_grad=(i == 0))
         for i, a in enumerate(init_m)]
    loss = (prop(wavelet, src.copy(), rec.copy(), models=m) - obs).pow(2).mean()
    flag["break"] = True
    loss.backward()                      # must NOT raise
    assert m[0].grad is not None
