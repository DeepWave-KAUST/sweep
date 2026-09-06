"""Pin the illumination path's invariants before anything changes it.

No gate tier sets ``compute_illumination``: grep the whole gate for "illum" and
it returns nothing. So the entire illumination path -- an extra per-timestep
grid pass over the wavefields, launched from the same drivers that build the
gradient -- is unprotected, and the one invariant that actually matters is
unstated anywhere:

    **turning illumination on must not move the gradient by one bit.**

Illumination is a diagnostic output. The FWI gradient comes from
``calculate_grad`` / the fused reconstruction imaging, which is separate code.
If enabling a diagnostic ever changed the answer, every illumination-compensated
inversion in the repository would be quietly wrong -- and nothing would say so.
"""
import numpy as np
import pytest
import torch

from conftest import requires_binding
from sweep.equations import Acoustic, Acoustic3D
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch

NT, DT, DH, ABCN, ORDER = 200, 1.2e-3, 10.0, 8, 4
CASES = (("acoustic2d", Acoustic, (80, 100), 2),
         ("acoustic3d", Acoustic3D, (48, 48, 48), 3))
STRATEGIES = (("full", Full()), ("boundary", BoundarySaving(storage="gpu")),
              ("ckpt", Ckpt(mode="chunk", chunks=4)))


def _heterogeneous(shape):
    """A ramp plus a bump. A homogeneous model hides operator-adjoint defects,
    and it also makes every relative comparison here meaningless."""
    idx = np.indices(shape).astype(np.float32)
    bump = sum((i - s / 2) ** 2 / s for i, s in zip(idx, shape))
    return (2500.0 + 4.0 * idx[0] + 60.0 * np.sin(bump)).astype(np.float32)


def _run(cls, shape, ndim, memory, illum, dev="cuda"):
    prop = PropTorch(cls(spatial_order=ORDER, device=dev), backend="torch", impl="c",
                     shape=shape, dh=DH, dt=DT, nt=NT, abcn=ABCN, B=1, dev=dev,
                     source_type=["h1"], receiver_type=["h1"], memory=memory)
    prop.compute_illumination = bool(illum)
    nz, nx = shape[0], shape[-1]
    src = np.array([[nx // 2, nz // 3] if ndim == 2 else
                    [nx // 2, shape[1] // 2, nz // 3]], np.int64)
    rxx = np.arange(4, nx - 4, 4, np.int64)
    cols = ([rxx, np.full(rxx.size, 4)] if ndim == 2 else
            [rxx, np.full(rxx.size, shape[1] // 2), np.full(rxx.size, 4)])
    rec = np.stack(cols, -1)[None]
    t = np.arange(NT, dtype=np.float32) * DT - 0.012
    a = np.pi * 15.0 * t
    wav = torch.tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32), device=dev)
    vp = torch.tensor(_heterogeneous(shape), device=dev, requires_grad=True)
    out = prop(wav, src, rec, models=[vp])
    out.pow(2).mean().backward()
    return {"record": out.detach(), "grad": vp.grad.detach(),
            "src_illum": getattr(prop, "source_illumination", None),
            "rec_illum": getattr(prop, "receiver_illumination", None)}


@requires_binding("acoustic2d_backward_bs", "acoustic3d_backward_bs")
@pytest.mark.parametrize("name,cls,shape,ndim", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("label,memory", STRATEGIES, ids=[s[0] for s in STRATEGIES])
def test_illumination_does_not_move_the_gradient(name, cls, shape, ndim, label, memory):
    """The invariant, measured against this configuration's own noise floor.

    Bit-exactness is the right bar for five of the six cells and the wrong bar
    for one: acoustic3d + chunk checkpointing is not run-to-run deterministic
    (measured here: max|d| 5.0e-9 on |grad|max 3.1e-1, rel 1.6e-8, with
    identical settings and illumination off in both runs). Demanding bit
    equality there would fail on the checkpoint path's own reordering and say
    nothing about illumination.

    So the test measures the floor instead of assuming one: run the SAME
    configuration twice with illumination off, then require that turning
    illumination on moves the gradient no more than that. A tolerance that is
    measured in the same process cannot drift with the hardware, and it keeps
    the strict bar wherever the path really is deterministic.
    """
    offs = [_run(cls, shape, ndim, memory, illum=False) for _ in range(3)]
    on = _run(cls, shape, ndim, memory, illum=True)

    assert torch.equal(offs[0]["record"], on["record"]), (
        f"{name}/{label}: the forward record moved when illumination was enabled")

    # The floor is a SPREAD, not one sample: a single off-vs-off pair of a
    # non-deterministic path lands anywhere in its distribution, so comparing
    # against one draw fails about as often as it passes.
    floor = max(float((a["grad"] - b["grad"]).abs().max())
                for i, a in enumerate(offs) for b in offs[i + 1:])
    moved = max(float((o["grad"] - on["grad"]).abs().max()) for o in offs)
    scale = float(offs[0]["grad"].abs().max())
    assert moved <= floor, (
        f"{name}/{label}: enabling illumination moved the vp GRADIENT by "
        f"{moved:.3e} (rel {moved / max(scale, 1e-30):.2e}), more than this "
        f"configuration's own run-to-run spread of {floor:.3e}. Illumination is "
        f"a diagnostic output and must not touch the gradient.")
    if floor == 0.0:
        assert torch.equal(offs[0]["grad"], on["grad"]), (
            f"{name}/{label}: this path is bit-reproducible with illumination "
            f"off, so enabling it must change nothing at all")


@requires_binding("acoustic2d_backward_bs", "acoustic3d_backward_bs")
@pytest.mark.parametrize("name,cls,shape,ndim", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("label,memory", STRATEGIES, ids=[s[0] for s in STRATEGIES])
def test_illumination_is_off_by_default_and_populated_when_asked(
        name, cls, shape, ndim, label, memory):
    off = _run(cls, shape, ndim, memory, illum=False)
    assert off["src_illum"] is None and off["rec_illum"] is None, (
        f"{name}/{label}: illumination came back without being requested")

    on = _run(cls, shape, ndim, memory, illum=True)
    for which in ("src_illum", "rec_illum"):
        t = on[which]
        assert isinstance(t, torch.Tensor), f"{name}/{label}: {which} is {type(t)}"
        assert tuple(t.shape) == tuple(shape), (
            f"{name}/{label}: {which} has shape {tuple(t.shape)}, model is {tuple(shape)}")
        assert torch.isfinite(t).all(), f"{name}/{label}: {which} is not finite"
        assert float(t.min()) >= 0.0, (
            f"{name}/{label}: {which} is a sum of squares and cannot be negative")
        assert float(t.max()) > 0.0, f"{name}/{label}: {which} is identically zero"
