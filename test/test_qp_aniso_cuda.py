"""impl='c' (qp_vti2d / qp_tti2d, qp_vti_born2d / qp_tti_born2d) against eager.

The pure-qP operators take S_n from the wavefield's gradient direction, so the
discrete step is nonlinear in p; ``s_gradient`` picks the backward:

* ``'frozen'`` -- the transpose with S_n held at the forward field: a linear,
  well-conditioned adjoint; impl='c' matches eager to fp32 round-off in full
  and boundary-saving mode.
* ``'adjoint'`` -- the adjoint field carries its own S_n (nonlinear in lambda):
  matches eager to the mode's own fp32 floor (eager gpu against eager cpu).
* ``'exact'`` -- the exact discrete derivative: ill-conditioned (percent-level
  fp32 floor); full storage only.

With epsilon == delta, S_n == 0 and every mode is the same linear adjoint, which
must agree to round-off (that is the CPML / stencil / imaging check).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

from sweep.equations import AcousticTTI, AcousticTTIBorn, AcousticVTI, AcousticVTIBorn  # noqa: E402
from sweep.propagator.options import BoundarySaving, Ckpt, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="impl='c' runs on CUDA")

SHAPE, ABCN, NT, DT, DH = (40, 48), 16, 200, 0.0015, 10.0


def _models(names, eps_eq_delta=False):
    nz, nx = SHAPE
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
    bump = np.exp(-(((z - 0.55) / 0.18) ** 2 + ((x - 0.45) / 0.22) ** 2)).astype(np.float32)
    ramp = 0.6 * z + 0.4 * x
    vals = {"vp": 2000.0 + 500.0 * ramp + 300.0 * bump,
            "epsilon": 0.20 + 0.08 * (ramp - 0.5) + 0.04 * bump,
            "delta": 0.06 + 0.04 * (0.5 - x + 0.3 * z) + 0.012 * bump,
            "theta": 0.35 + 0.45 * x - 0.2 * z + 0.15 * bump,
            "mp": np.zeros(SHAPE, np.float32)}
    if eps_eq_delta:
        vals["delta"] = vals["epsilon"]
    return [np.ascontiguousarray(vals[n] + 0 * z + 0 * x, dtype=np.float32) for n in names]


def _wavelet():
    t = np.arange(NT) * DT - 0.06
    a = (np.pi * 10.0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def _geometry():
    nz, nx = SHAPE
    src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rx = np.arange(3, nx - 3, 4, dtype=np.int32)
    return src, np.stack([rx, np.full_like(rx, 2)], -1)[None]


def _prop(cls, impl, memory, s_gradient, dev="cuda"):
    eq = cls(spatial_order=4, device=dev, backend="torch", s_gradient=s_gradient)
    kw = dict(shape=SHAPE, abcn=ABCN, dh=DH, dt=DT, dev=dev, nt=NT, B=1)
    if impl == "eager":
        return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    return PropTorch(eq, impl="c", memory=memory, **kw)


def _run(cls, impl, memory, s_gradient, eps_eq_delta=False, dev="cuda", grads=None):
    prop = _prop(cls, impl, memory, s_gradient, dev)
    names = cls(spatial_order=4).models
    grads = names if grads is None else grads
    wav = torch.tensor(_wavelet(), device=dev, requires_grad="wavelet" in grads)
    ms = [torch.tensor(a, device=dev, requires_grad=n in grads) for n, a in zip(names, _models(names, eps_eq_delta))]
    src, rec = _geometry()
    d = prop(wav, src, rec, models=ms)
    if all(not m.requires_grad for m in ms) and not wav.requires_grad:
        return d.detach().double().cpu(), {}
    d.pow(2).sum().backward()
    out = {"wavelet": wav.grad} if wav.requires_grad else {}
    out.update({n: m.grad for n, m in zip(names, ms) if m.requires_grad})
    return d.detach().double().cpu(), {k: v.detach().double().cpu() for k, v in out.items()}


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _cos(a, b):
    return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
def test_forward_record_matches_eager(cls):
    rc, _ = _run(cls, "c", Full(), "frozen", grads=[])
    re, _ = _run(cls, "eager", None, "frozen", grads=[])
    assert float(re.abs().max()) > 1e-3
    assert _rel(rc, re) < 1e-4


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
def test_forward_is_the_same_in_every_mode(cls):
    recs = [_run(cls, "eager", None, sg, grads=[])[0] for sg in ("adjoint", "frozen", "exact")]
    assert torch.equal(recs[0], recs[1]) and torch.equal(recs[0], recs[2])


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
@pytest.mark.parametrize("s_gradient", ["adjoint", "frozen", "exact"])
def test_linear_case_every_mode_matches_eager(cls, s_gradient):
    """epsilon == delta: S_n == 0, the adjoint is linear in every mode."""
    _, ge = _run(cls, "eager", None, s_gradient, eps_eq_delta=True)
    _, gc = _run(cls, "c", Full(), s_gradient, eps_eq_delta=True)
    for k in ge:
        assert _rel(gc[k], ge[k]) < 5e-4, (k, _rel(gc[k], ge[k]))


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
def test_frozen_full_and_bs_match_eager(cls):
    _, ge = _run(cls, "eager", None, "frozen")
    for mem in (Full(), BoundarySaving(storage="gpu"), BoundarySaving(storage="cpu", transfer_interval=4)):
        _, gc = _run(cls, "c", mem, "frozen")
        for k in ge:
            assert _rel(gc[k], ge[k]) < 2e-3, (type(mem).__name__, k, _rel(gc[k], ge[k]))
            assert _cos(gc[k], ge[k]) > 0.99999


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
@pytest.mark.parametrize("s_gradient", ["adjoint", "frozen"])
def test_boundary_saving_reproduces_full(cls, s_gradient):
    _, gf = _run(cls, "c", Full(), s_gradient)
    _, gb = _run(cls, "c", BoundarySaving(storage="gpu"), s_gradient)
    for k in gf:
        assert _rel(gb[k], gf[k]) < 1e-3, (k, _rel(gb[k], gf[k]))


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI, AcousticVTIBorn])
@pytest.mark.parametrize("s_gradient", ["adjoint", "frozen"])
def test_int8_boundary_storage_is_refused(cls, s_gradient):
    """int8 frames move the qP gradients by 8-15% away from the source."""
    grads = ["mp"] if "mp" in cls(spatial_order=4).models else None
    with pytest.raises(NotImplementedError, match="int8 boundary storage"):
        _run(cls, "c", BoundarySaving(storage="gpu", storage_dtype="int8"), s_gradient, grads=grads)


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
def test_adjoint_mode_matches_eager_to_its_floor(cls):
    """The adjoint recursion is nonlinear in lambda: compare against eager's own
    gpu / cpu spread instead of a fixed tolerance."""
    _, gg = _run(cls, "eager", None, "adjoint")
    _, gcpu = _run(cls, "eager", None, "adjoint", dev="cpu")
    _, gc = _run(cls, "c", Full(), "adjoint")
    for k in gg:
        floor = _rel(gcpu[k], gg[k])
        assert _rel(gc[k], gg[k]) < max(3 * floor, 1e-4), (k, _rel(gc[k], gg[k]), floor)


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTIBorn])
def test_exact_with_boundary_saving_is_refused(cls):
    with pytest.raises(NotImplementedError, match="s_gradient='exact'"):
        names = cls(spatial_order=4).models
        _run(cls, "c", BoundarySaving(storage="gpu"), "exact", grads=["mp"] if "mp" in names else None)


def test_checkpointing_is_refused():
    with pytest.raises(Exception, match="no checkpoint backward"):
        _run(AcousticVTI, "c", Ckpt(mode="chunk", chunks=20), "frozen")


def _born_mp_gradient(cls, impl, memory, s_gradient, dev="cuda"):
    """The reflectivity gradient at mp = 0 (the RTM image) of 0.5 |d - d_obs|^2,
    d_obs the eager Born data of a flat reflector."""
    names = cls(spatial_order=4).models
    mp_true = np.zeros(SHAPE, np.float32)
    mp_true[int(0.6 * SHAPE[0])] = 0.2
    bg = _models(names[:-1])
    wav = torch.tensor(_wavelet(), device="cuda")
    src, rec = _geometry()
    with torch.no_grad():
        d_obs = _prop(cls, "eager", None, s_gradient)(
            wav, src, rec, models=[torch.tensor(a, device="cuda") for a in bg + [mp_true]])
    prop = _prop(cls, impl, memory, s_gradient, dev)
    mp = torch.zeros(SHAPE, device=dev, requires_grad=True)
    d = prop(wav.to(dev), src, rec, models=[torch.tensor(a, device=dev) for a in bg] + [mp])
    (0.5 * (d - d_obs.to(dev)).pow(2).sum()).backward()
    return mp.grad.detach().double().cpu()


@pytest.mark.parametrize("cls", [AcousticVTIBorn, AcousticTTIBorn])
def test_born_reflectivity_gradient_frozen(cls):
    """'frozen' is linear in the residual: impl='c' matches eager to round-off."""
    ge = _born_mp_gradient(cls, "eager", None, "frozen")
    gf = _born_mp_gradient(cls, "c", Full(), "frozen")
    gb = _born_mp_gradient(cls, "c", BoundarySaving(storage="gpu"), "frozen")
    assert float(ge.abs().max()) > 0
    assert _cos(gf, ge) > 0.9999, _cos(gf, ge)
    assert _rel(gb, gf) < 1e-3, _rel(gb, gf)


@pytest.mark.parametrize("cls", [AcousticVTIBorn, AcousticTTIBorn])
def test_born_reflectivity_gradient_adjoint(cls):
    """'adjoint' is nonlinear in the residual (percent-level fp32 spread for the
    Born pair): compare against eager's own gpu / cpu spread."""
    ge = _born_mp_gradient(cls, "eager", None, "adjoint")
    gcpu = _born_mp_gradient(cls, "eager", None, "adjoint", dev="cpu")
    gf = _born_mp_gradient(cls, "c", Full(), "adjoint")
    gb = _born_mp_gradient(cls, "c", BoundarySaving(storage="gpu"), "adjoint")
    floor = _rel(gcpu, ge)
    assert float(ge.abs().max()) > 0
    assert _rel(gf, ge) < max(3 * floor, 1e-4), (_rel(gf, ge), floor)
    assert _rel(gb, gf) < 1e-3, _rel(gb, gf)


@pytest.mark.parametrize("cls", [AcousticVTI, AcousticTTI])
@pytest.mark.parametrize("s_gradient", ["adjoint", "frozen"])
def test_eager_compiled_matches_uncompiled(cls, s_gradient):
    """eager's default (torch.compile) runs every s_gradient; stop_gradient used
    to send torch tensors to jax under dynamo."""
    names = cls(spatial_order=4).models
    outs = []
    for use_compile in (False, True):
        eq = cls(spatial_order=4, device="cuda", backend="torch", s_gradient=s_gradient)
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=use_compile), memory=Full(),
                         shape=SHAPE, abcn=ABCN, dh=DH, dt=DT, dev="cuda", nt=NT, B=1)
        ms = [torch.tensor(a, device="cuda", requires_grad=True) for a in _models(names)]
        src, rec = _geometry()
        d = prop(torch.tensor(_wavelet(), device="cuda"), src, rec, models=ms)
        d.pow(2).sum().backward()
        outs.append([d.detach().double().cpu()] + [m.grad.detach().double().cpu() for m in ms])
    assert _rel(outs[1][0], outs[0][0]) < 1e-4
    for n, a, b in zip(names, outs[1][1:], outs[0][1:]):
        assert _cos(a, b) > 0.999, (n, _cos(a, b))


def test_born_refuses_background_gradients():
    with pytest.raises(NotImplementedError, match="reflectivity"):
        _run(AcousticVTIBorn, "c", Full(), "adjoint", grads=["vp", "mp"])
