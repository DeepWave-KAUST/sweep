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
    # Six off-runs -> 15 pairs. Four runs (6 pairs) drew a floor of 1.6e-9 on
    # [ckpt-acoustic3d] in a 2026-09-24 gate, BELOW the 3.0e-9 minimum the 15
    # pairs quoted below had shown, and a genuine 9.4e-9 on/off difference then
    # read as 5.9x. The floor is a max over pairs, so more pairs can only raise
    # it toward the spread's real top; sensitivity to a real coupling -- a
    # systematic shift, not a draw -- is untouched. Cost: two ~0.5 s runs.
    offs = [_run(cls, shape, ndim, memory, illum=False) for _ in range(6)]
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
    # ALLOWANCE: the noisy cell's own spread runs 3.0e-9..2.4e-8 over 15 pairs --
    # a factor of 8 between draws -- so a floor built from a handful of pairs can
    # under-estimate it by several times. 4x covers that. It costs nothing in
    # sensitivity: a real coupling would be a systematic shift, and measured over
    # 18 on-vs-off pairs the maximum came out at 0.86x the off-vs-off maximum,
    # i.e. drawn from the same distribution.
    assert moved <= 4.0 * floor, (
        f"{name}/{label}: enabling illumination moved the vp GRADIENT by "
        f"{moved:.3e} (rel {moved / max(scale, 1e-30):.2e}), more than this "
        f"configuration's own run-to-run spread of {floor:.3e} (x4). Illumination is "
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


@requires_binding("acoustic2d_backward_bs", "acoustic3d_backward_bs")
@pytest.mark.parametrize("name,cls,shape,ndim", CASES, ids=[c[0] for c in CASES])
def test_illumination_does_not_depend_on_the_memory_strategy(name, cls, shape, ndim):
    """`memory=` is a space/time trade. It must not change WHAT you get back.

    It used to. One kernel accumulated the source illumination for both paths,
    but they handed it different fields: the full store, which for acoustic *is*
    ``u_tt = vp^2*Lap(u)``, versus boundary saving's reconstructed raw pressure.
    So the same attribute returned ``sum_t u_tt^2`` or ``sum_t u^2`` -- about 3e10
    apart (measured 5.24e18 vs 1.59e8 in 2-D) -- decided by a memory knob, with
    nothing in the docstring to say which.

    The bar is set by the gradient, not by a hand-picked tolerance. Both
    quantities are built from the same reconstructed wavefield, so boundary
    saving's fp32 reconstruction error is their common floor; squaring doubles a
    relative error, so the illumination may be ~2x the gradient's. 20x is that
    with an order of margin, and it still catches the old behaviour by seven.
    """
    full = _run(cls, shape, ndim, Full(), illum=True)
    bs = _run(cls, shape, ndim, BoundarySaving(storage="gpu"), illum=True)

    def rel(a, b):
        return float((a - b).abs().max()) / max(float(a.abs().max()), 1e-30)

    grad_rel = rel(full["grad"], bs["grad"])
    illum_rel = rel(full["src_illum"], bs["src_illum"])
    bar = max(20.0 * grad_rel, 1e-9)
    assert illum_rel <= bar, (
        f"{name}: source_illumination differs by rel {illum_rel:.3e} between "
        f"Full() and BoundarySaving(), against a gradient difference of "
        f"{grad_rel:.3e} between the same two runs. The memory strategy is "
        f"changing what the quantity IS, not just how it was stored.")


@requires_binding("acoustic2d_backward_bs", "acoustic3d_backward_bs")
@pytest.mark.parametrize("name,cls,shape,ndim", CASES, ids=[c[0] for c in CASES])
def test_receiver_illumination_no_longer_misses_the_it0_step(name, cls, shape, ndim):
    """It used to be short by exactly one term, and now it is short by none.

    ``receiver_illumination`` is ``sum_t lambda^2``. The store-based loop
    accumulates it for ``it = nt-1 .. 0``; the boundary-saving loop floors at
    ``it == 1``, so it summed one ``lambda(0)^2`` fewer -- and ``it == 0`` sits
    straight after the last residual injection, where lambda is largest.
    Measured 1.66e-02 (2-D) and 1.80e-03 (3-D) against the full store.

    ``eq_driver``'s it == 0 tail now calls ``bs_illum_tail``, which accumulates
    the receiver term only: the forward field is not reconstructed at it == 0,
    so the source term cannot be closed there -- but the same comparison bounds
    its contribution below 2.7e-8, which is why only one half of this is fixable
    and the other half does not matter.

    The bar is **bit equality**, not a tolerance, and that is the point: if the
    diagnosis had been wrong in any part, adding one term would have left a
    residual instead of landing exactly on the store's value.
    """
    full = _run(cls, shape, ndim, Full(), illum=True)
    ckpt = _run(cls, shape, ndim, Ckpt(mode="chunk", chunks=4), illum=True)
    bs = _run(cls, shape, ndim, BoundarySaving(storage="gpu"), illum=True)

    assert torch.equal(full["rec_illum"], ckpt["rec_illum"]), (
        f"{name}: Full() and Ckpt() disagree on receiver_illumination -- they "
        f"share image_step, so this is a different defect from the one below")
    assert torch.equal(full["rec_illum"], bs["rec_illum"]), (
        f"{name}: boundary saving's receiver_illumination differs from the "
        f"store's by max|d| "
        f"{float((full['rec_illum'] - bs['rec_illum']).abs().max()):.3e}. It was "
        f"short by exactly lambda(0)^2 before eq_driver grew its it == 0 tail; "
        f"a residual here means that is no longer the whole story.")
