"""ViscoAcoustic on ``impl='c'`` (CUDA): forward + gradients vs the eager
reference, per-edge free surfaces included, across full / ckpt / recursive
checkpoint memory modes.

Geometry DISCRIMINATES (central source, receiver grid over the physical
interior, long NT — see test_free_surface_per_edge_cuda.py for why), and the
models are heterogeneous in BOTH vp and Q (uniform models mask gradient
defects).

The closed-box config (all 4 faces free, no PML anywhere) isolates the
stencil + FFT damping from the pre-existing eager-vs-c PML-region divergence:
there c must match eager to ~1e-5, while PML configs sit at the same ~2e-3
level as plain Acoustic.

``omega`` is a real model parameter ((vp/omega)^(2*gamma) anchors the power
law); its gradient is compared like every other model gradient.
"""
import numpy as np
import pytest
import torch

from sweep.equations import ViscoAcoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import (
    CUDAOptions, MemoryOptions, BoundaryOptions, CkptOptions,
)
from sweep.signal import ricker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="visco_acoustic2d impl='c' requires CUDA",
)

DEV = "cuda"
NZ, NX, DH, DT, NT, SO, ABCN = 64, 64, 8.0, 1.0e-3, 450, 4, 16
PHYS = NZ - 2 * (ABCN + SO)
F0 = 10.0


def _wav():
    t = np.arange(NT, dtype=np.float32) * DT - 0.05
    return torch.tensor((1e3 * ricker(t, f=F0)).astype(np.float32), device=DEV)


def _src_rec():
    src = np.array([[PHYS // 2, PHYS // 2]], dtype=np.int64)
    g = np.arange(2, PHYS - 1, 4, dtype=np.int64)
    rx, rz = np.meshgrid(g, g, indexing="xy")
    rec = np.stack([rx.ravel(), rz.ravel()], -1)[None]
    return src, rec


def _models(req=False):
    d = np.linspace(0, 1, NZ, dtype=np.float32)
    vp = np.broadcast_to((2200 + 700 * d)[:, None], (NZ, NX)).astype(np.float32).copy()
    Q = np.broadcast_to((25 + 30 * d)[:, None], (NZ, NX)).astype(np.float32).copy()
    return [torch.tensor(vp, device=DEV, requires_grad=req),
            torch.tensor(Q, device=DEV, requires_grad=req),
            torch.full((NZ, NX), float(2 * np.pi * F0), device=DEV,
                       requires_grad=req)]


def _prop(fs, mode, phase=True, amp=True):
    eq = ViscoAcoustic(spatial_order=SO, device=DEV, backend="torch",
                       phase_shift=phase, amplitude_damping=amp)
    common = dict(shape=(NZ, NX), free_surface=fs, abcn=ABCN, dh=DH, dt=DT)
    if mode == "eager":
        # use_compile=False: the compiled eager step is the IMPRECISE side —
        # inductor's fused pow/ln backward perturbs the Q-gradient cotangents
        # at ~1e-6, which the ln(vp/omega)-weighted dB2/dQ chain amplifies to
        # ~5e-3 (NCQ).  The plain eager step matches the hand-written CUDA
        # adjoint at ~5e-7, so it is the reference.
        return PropTorch(eq, backend="torch", impl="eager", use_ckpt=False,
                         use_compile=False, **common)
    if mode == "full":
        return PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **common)
    if mode == "default":
        return PropTorch(eq, backend="torch", impl="c", **common)
    if mode == "ckpt":
        return PropTorch(eq, backend="torch", impl="c",
                         cuda_options=CUDAOptions(memory=MemoryOptions(
                             strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=10))),
                         **common)
    if mode == "rec":
        return PropTorch(eq, backend="torch", impl="c",
                         cuda_options=CUDAOptions(memory=MemoryOptions(
                             strategy="ckpt", ckpt=CkptOptions(mode="recursive", count=8))),
                         **common)
    raise ValueError(mode)


def _rel(a, b):
    a = a.detach().flatten().double(); b = b.detach().flatten().double()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def _cos(a, b):
    a = a.detach().flatten().double(); b = b.detach().flatten().double()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-30)).item()


def _record(fs, mode, phase=True, amp=True):
    src, rec = _src_rec()
    return _prop(fs, mode, phase, amp)(_wav(), src, rec,
                                       models=_models(False)).detach()


def _grads(fs, mode, phase=True, amp=True):
    src, rec = _src_rec()
    p = _prop(fs, mode, phase, amp)
    m = _models(True)
    w = _wav().requires_grad_(True)
    (p(w, src, rec, models=m) ** 2).sum().backward()

    def g(x):  # unused params (e.g. omega with amp off) have grad None
        return (x.grad if x.grad is not None else torch.zeros_like(x)).detach().clone()
    return [g(w)] + [g(x) for x in m]


ALL4 = ["top", "bottom", "left", "right"]


@pytest.mark.parametrize("fs,tag", [
    (False, "noFS"), (True, "top"), (["left"], "left"), (ALL4, "all4")])
@pytest.mark.parametrize("phase,amp", [
    (True, True), (True, False), (False, True), (False, False)])
def test_forward_matches_eager(fs, tag, phase, amp):
    e = _record(fs, "eager", phase, amp)
    c = _record(fs, "full", phase, amp)
    tol = 1e-4 if tag == "all4" else 6e-3   # PML band = pre-existing c-vs-eager floor
    assert _rel(c, e) < tol, (tag, phase, amp, _rel(c, e))


def test_forward_discriminates():
    assert _rel(_record(["left"], "eager"), _record(True, "eager")) > 0.10


@pytest.mark.parametrize("fs,tag", [(True, "top"), (["left"], "left"), (ALL4, "all4")])
def test_gradients_match_eager(fs, tag):
    ge = _grads(fs, "eager")
    gc = _grads(fs, "full")
    # NCQ: the reference frequency omega is a REAL parameter of the
    # constant-Q model ((vp/omega)^(2*gamma)); its gradient is genuine, so it
    # is compared like every other model gradient.
    for name, a, b in zip(["wavelet", "vp", "Q", "omega"], gc, ge[:4]):
        assert _cos(a, b) > 0.9999, (tag, name, _cos(a, b))
        assert _rel(a, b) < 6e-3, (tag, name, _rel(a, b))


@pytest.mark.parametrize("mode", ["ckpt", "rec", "default"])
def test_ckpt_modes_match_full(mode):
    gf = _grads(True, "full")
    gm = _grads(True, mode)
    for name, a, b in zip(["wavelet", "vp", "Q"], gm[:3], gf[:3]):
        assert _rel(a, b) < 1e-5, (mode, name, _rel(a, b))


def test_default_strategy_is_full():
    # implicit default must silently fall back to FULL storage (visco has no
    # BS; ckpt stays available as an explicit request)
    p = _prop(True, "default")
    b = p._backend_impl
    assert getattr(b, "use_ckpt", None) is False
    assert not dict(getattr(b, "boundary_saving_config", {})).get("enabled", False)


def test_explicit_boundary_raises():
    eq = ViscoAcoustic(spatial_order=SO, device=DEV, backend="torch")
    with pytest.raises(NotImplementedError):
        PropTorch(eq, backend="torch", impl="c", shape=(NZ, NX),
                  free_surface=True, abcn=ABCN, dh=DH, dt=DT,
                  cuda_options=CUDAOptions(memory=MemoryOptions(
                      strategy="boundary",
                      boundary=BoundaryOptions(storage="gpu"))))


def test_rtm_runs_on_full_storage():
    src, rec = _src_rec()
    m = _models(False)
    p = _prop(True, "full")
    obs = p(_wav(), src, rec, models=m).detach()
    syn, image, src_illum, rec_illum = p.rtm(_wav(), src, rec, obs, models=m)
    assert torch.isfinite(image).all() and image.abs().max() > 0
    assert torch.isfinite(src_illum).all()


# --------------------------------------------------------------------------
# toggle matrix: records + gradients for every (phase_shift, amplitude_damping)
# --------------------------------------------------------------------------
TOGGLES = [(True, True), (True, False), (False, True), (False, False)]


@pytest.mark.parametrize("phase,amp", TOGGLES)
def test_gradients_match_eager_toggles(phase, amp):
    """c-vs-eager gradient consistency for each attenuation-term combo.

    Physical structure asserted too: with amp off, omega never enters the
    equation (grad exactly zero on BOTH backends); with phase AND amp off it
    degenerates to acoustic, so Q's grad is exactly zero as well.  With amp
    on, omega's true gradient is ~0 by the constant-Q cancellation (module
    docstring) — magnitude-checked, never compared pointwise."""
    ge = _grads(True, "eager", phase, amp)
    gc = _grads(True, "full", phase, amp)
    for name, a, b in zip(["wavelet", "vp"], gc[:2], ge[:2]):
        assert _cos(a, b) > 0.9999, (phase, amp, name, _cos(a, b))
        assert _rel(a, b) < 6e-3, (phase, amp, name, _rel(a, b))
    if phase or amp:
        assert _cos(gc[2], ge[2]) > 0.9999 and _rel(gc[2], ge[2]) < 6e-3
        # NCQ: the reference frequency is a REAL parameter — (vp/omega)^(2g)
        # gives omega a genuine gradient; compare it like the others.
        assert _cos(gc[3], ge[3]) > 0.9999 and _rel(gc[3], ge[3]) < 6e-3
    else:
        assert ge[2].norm() == 0 and gc[2].norm() == 0      # Q dead
        assert ge[3].norm() == 0 and gc[3].norm() == 0      # omega dead


def test_toggles_discriminate_on_c():
    """Each toggle must genuinely change the impl='c' record (guards against a
    silently-ignored toggle passing the c-vs-eager comparisons trivially)."""
    r = {t: _record(True, "full", *t) for t in TOGGLES}
    assert _rel(r[(True, True)], r[(False, False)]) > 0.05
    assert _rel(r[(True, False)], r[(False, False)]) > 0.02   # phase alone moves it
    assert _rel(r[(False, True)], r[(False, False)]) > 0.02   # damping alone moves it


def test_toggles_off_matches_acoustic_c():
    """(phase off, amp off) must reproduce Acoustic impl='c' BIT-EXACTLY:
    same CPML kernels, vp_step is the vp tensor itself, no damping launch.
    Mirrors the eager anchor test_both_toggles_off_matches_acoustic."""
    from sweep.equations import Acoustic
    src, rec = _src_rec()
    m = _models(True)

    pv = _prop(True, "full", phase=False, amp=False)
    wv = _wav().requires_grad_(True)
    rv = pv(wv, src, rec, models=m)

    ea = Acoustic(spatial_order=SO, device=DEV, backend="torch")
    pa = PropTorch(ea, backend="torch", impl="c", use_ckpt=False,
                   boundary_saving_config={"enabled": False},
                   shape=(NZ, NX), free_surface=True, abcn=ABCN, dh=DH, dt=DT)
    va = m[0].detach().clone().requires_grad_(True)
    wa = _wav().requires_grad_(True)
    ra = pa(wa, src, rec, models=[va])

    assert torch.equal(rv.detach(), ra.detach())
    (rv ** 2).sum().backward()
    (ra ** 2).sum().backward()
    assert torch.equal(wv.grad, wa.grad)
    assert torch.equal(m[0].grad, va.grad)


def test_ncq_closed_box_grads_tight():
    """The sharp NCQ assertion: with no PML (closed box) the hand-written CUDA
    adjoint must match the UNCOMPILED eager autograd at float precision for
    every gradient including Q (wavelet/vp ~1e-6, Q ~1e-6; the compiled eager
    step would sit ~5e-3 away on Q — see _prop)."""
    ge = _grads(ALL4, "eager")
    gc = _grads(ALL4, "full")
    for name, a, b in zip(["wavelet", "vp", "Q"], gc[:3], ge[:3]):
        assert _cos(a, b) > 0.999999, (name, _cos(a, b))
        assert _rel(a, b) < 5e-5, (name, _rel(a, b))
