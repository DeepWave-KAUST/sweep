"""Shot-batched impl='c': B shots in ONE kernel sequence.

Semantics under test (the contract PropSEM already documents for the loop):

* the batch is only a scheduling change -- shot k of the batched call must
  equal an INDEPENDENT single-shot run of shot k (per-shot assertions, so a
  cross-shot contamination through a mis-strided buffer cannot average out);
* a batch-shared model's gradient is the SUM of the per-shot gradients;
* per-shot models (``models=[vp_B]`` with ``vp_B`` of shape ``(B, nz, nx)``)
  give every shot its own physics, degenerating to the shared case when the
  batch axis is dropped;
* eager and impl='c' agree under both model modes.

Tolerances follow test_sem_backend_parity: both backends scatter with CUDA
atomics, so every bar is the measured run-to-run floor of the same
configuration, not a fixed number.

Trap this file must not fall into: if the batched fast path silently
DECLINES (a guard condition fails) the comparison becomes loop-vs-loop and
passes vacuously. Every test that claims to exercise the batch asserts, via a
spy on ``sem_c_forward_shot_batch``, that the fast path actually ran.

C-PML convention for per-shot models: ONE profile for the whole batch, built
with the batch-max vp, on BOTH backends (the batched kernels read a single
per-mesh profile). The per-shot tests therefore give every shot the same vp
maximum so "independent single runs" build the identical profile and remain a
valid reference.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem import prop_c
from sweep.sem.equations import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NT, B = 4, 25.0, 8.0, 120, 4
NZ, NX = 25, 33
VP = 2400.0
DT = 0.30 * (0.172 * NO * DH) / (VP * 1.2)
SRC_AMP = 1e12                     # keep fields O(1); see the parity module
FLOOR_FACTOR = 25.0
ABS_CEILING = 1e-4

SRC = np.stack([np.linspace(6, NX - 7, B).astype(np.int64),
                np.full(B, 3, dtype=np.int64)], -1)          # (B, 2), x varies
RIX = np.arange(5, NX - 5)
REC = np.stack([RIX, np.full_like(RIX, 2)], -1)[None]        # (1, nrec, 2)


def _wav(per_shot=False):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    w = ((1 - 2 * a) * np.exp(-a)) * SRC_AMP
    w = torch.tensor(w.astype(np.float32), device="cuda")
    if not per_shot:
        return w
    scale = torch.linspace(0.5, 1.5, B, device="cuda")[:, None]
    return w[None] * scale                                   # (B, nt) distinct


def _vp_shared(grad=False):
    v = torch.full((NZ, NX), VP, device="cuda")
    return v.requires_grad_(True) if grad else v


def _vp_pershot(grad=False):
    """(B, NZ, NX): a strong SHALLOW low-velocity anomaly at a different x per
    shot. Negative, so every shot's max-vp (and hence the batch's shared
    C-PML profile) equals the plain VP an independent single-shot build would
    use; shallow and strong (-700 m/s under the receiver line), so shot b's
    record visibly depends on model b -- the discrimination the broadcast
    falsifier below needs (a deep +300 bump moved the records by only ~7e-4,
    within an order of magnitude of the tolerance)."""
    z = torch.arange(NZ, device="cuda")[:, None]
    x = torch.arange(NX, device="cuda")[None, :]
    out = []
    for b in range(B):
        cx = 6 + b * ((NX - 12) // max(B - 1, 1))
        bump = -700.0 * torch.exp(-((x - cx) ** 2 + (z - 8) ** 2) / 50.0)
        out.append(VP + bump)
    v = torch.stack(out)
    return v.requires_grad_(True) if grad else v


def _prop(impl, memory=None):
    return S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=3 * NO, f_dominant=F0, memory=memory,
                     dev="cuda", impl=impl)


def _spy(monkeypatch):
    calls = {"n": 0}
    orig = prop_c.sem_c_forward_shot_batch

    def wrapper(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(prop_c, "sem_c_forward_shot_batch", wrapper)
    return calls


def _rel(a, b, scale=None):
    a, b = a.double(), b.double()
    d = float(a.abs().max()) if scale is None else float(scale)
    return float((a - b).abs().max() / d) if d else 0.0


def _single_runs(impl, vp_rows, wav):
    """The reference semantics: B independent single-shot calls, stacked.
    ``vp_rows`` is one shared tensor or a (B, NZ, NX) stack."""
    outs = []
    for b in range(B):
        v = vp_rows if vp_rows.dim() == 2 else vp_rows[b]
        w = wav if wav.dim() == 1 else wav[b]
        p = _prop(impl)
        with torch.no_grad():
            outs.append(p(w, SRC[b:b + 1], REC, models=[v]))
    return torch.cat(outs, 0)


# ------------------------------ shared model ------------------------------ #

@pytest.mark.parametrize("per_shot_wavelet", [False, True],
                         ids=["shared-wavelet", "pershot-wavelet"])
def test_batched_forward_matches_independent_runs(monkeypatch, per_shot_wavelet):
    calls = _spy(monkeypatch)
    wav = _wav(per_shot_wavelet)
    p = _prop("c")
    with torch.no_grad():
        got = p(wav, SRC, REC, models=[_vp_shared()])
        again = p(wav, SRC, REC, models=[_vp_shared()])
    assert calls["n"] == 2, "the batched fast path did not run"
    assert got.shape == (B, NT, REC.shape[1], 1)
    ref = _single_runs("c", _vp_shared(), wav)
    floor = _rel(got, again)
    scale = float(got.abs().max())
    for b in range(B):                    # per-shot: contamination cannot hide
        d = _rel(got[b], ref[b], scale)
        assert d <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"shot {b}: batched vs independent {d:.2e}, floor {floor:.2e}")


def test_shots_differ_from_each_other():
    """Falsifier for the falsifier: if a stride bug made every batch slot
    replay shot 0, the per-shot comparison above would only catch it through
    the reference. Assert directly that the shots are distinct."""
    p = _prop("c")
    with torch.no_grad():
        got = p(_wav(), SRC, REC, models=[_vp_shared()])
    for b in range(1, B):
        assert _rel(got[0], got[b]) > 1e-3, f"shot {b} equals shot 0"


def test_batched_gradient_is_the_sum_of_per_shot_gradients(monkeypatch):
    calls = _spy(monkeypatch)
    wav = _wav()
    vp = _vp_shared(grad=True)
    p = _prop("c")
    s = p(wav, SRC, REC, models=[vp])
    (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30)).backward()
    assert calls["n"] == 1, "the batched fast path did not run"
    g_batch = vp.grad.clone()

    # floor: the same batched call, run again
    vp2 = _vp_shared(grad=True)
    s2 = _prop("c")(wav, SRC, REC, models=[vp2])
    (s2 ** 2).sum().div(s2.detach().pow(2).sum().clamp_min(1e-30)).backward()
    floor = _rel(g_batch, vp2.grad)

    # reference: per-shot loop through the SAME loss. One shared vp tensor
    # reused across B independent calls accumulates the summed gradient --
    # exactly the semantics the batch must reproduce.
    vp_ref = _vp_shared(grad=True)
    seis = []
    for b in range(B):
        seis.append(_prop("c")(wav, SRC[b:b + 1], REC, models=[vp_ref]))
    sref = torch.cat(seis, 0)
    (sref ** 2).sum().div(sref.detach().pow(2).sum().clamp_min(1e-30)).backward()
    d = _rel(g_batch, vp_ref.grad)
    assert d <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"batched grad vs per-shot sum {d:.2e}, floor {floor:.2e}")


def test_batched_forward_matches_eager(monkeypatch):
    calls = _spy(monkeypatch)
    wav = _wav()
    with torch.no_grad():
        c1 = _prop("c")(wav, SRC, REC, models=[_vp_shared()])
        c2 = _prop("c")(wav, SRC, REC, models=[_vp_shared()])
        e = _prop("eager")(wav, SRC, REC, models=[_vp_shared()])
    assert calls["n"] == 2
    floor, got = _rel(c1, c2), _rel(c1, e)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"batched c vs eager {got:.2e}, floor {floor:.2e}")


def test_different_receiver_rows_fall_back_to_the_loop(monkeypatch):
    """Distinct receiver geometries per shot are outside the batched kernel's
    contract (one rec_G table); the call must still be CORRECT via the loop."""
    calls = _spy(monkeypatch)
    rec2 = np.concatenate([REC, REC + np.array([0, 1])], 0)  # (2, nrec, 2)
    src2 = SRC[:2]
    p = _prop("c")
    with torch.no_grad():
        got = p(_wav(), src2, rec2, models=[_vp_shared()])
    assert calls["n"] == 0, "batched path must decline per-shot receivers"
    with torch.no_grad():
        r0 = _prop("c")(_wav(), src2[0:1], rec2[0:1], models=[_vp_shared()])
        r1 = _prop("c")(_wav(), src2[1:2], rec2[1:2], models=[_vp_shared()])
    ref = torch.cat([r0, r1], 0)
    scale = float(got.abs().max())
    assert _rel(got, ref, scale) <= ABS_CEILING


# ------------------------------ per-shot models --------------------------- #

def test_pershot_models_match_independent_runs(monkeypatch):
    calls = _spy(monkeypatch)
    wav = _wav()
    vp_b = _vp_pershot()
    p = _prop("c")
    with torch.no_grad():
        got = p(wav, SRC, REC, models=[vp_b])
        again = p(wav, SRC, REC, models=[vp_b])
    assert calls["n"] == 2, "the batched fast path did not run"
    ref = _single_runs("c", vp_b, wav)
    floor = _rel(got, again)
    scale = float(got.abs().max())
    for b in range(B):
        d = _rel(got[b], ref[b], scale)
        assert d <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"shot {b}: per-shot-model batch vs independent {d:.2e}, "
            f"floor {floor:.2e}")


def test_pershot_models_are_not_broadcast():
    """If the material stride were dropped (Bm read as 1), every shot would
    see model 0. Shot b's record must differ from a broadcast-model run."""
    wav = _wav()
    vp_b = _vp_pershot()
    with torch.no_grad():
        got = _prop("c")(wav, SRC, REC, models=[vp_b])
        b0 = _prop("c")(wav, SRC, REC, models=[vp_b[0]])   # model 0 broadcast
    assert _rel(got[1:], b0[1:]) > 1e-3, (
        "per-shot models produced the same records as broadcasting model 0 "
        "-- the material stride is being ignored")


def test_pershot_models_match_eager(monkeypatch):
    calls = _spy(monkeypatch)
    wav = _wav()
    vp_b = _vp_pershot()
    with torch.no_grad():
        c1 = _prop("c")(wav, SRC, REC, models=[vp_b])
        c2 = _prop("c")(wav, SRC, REC, models=[vp_b])
        e = _prop("eager")(wav, SRC, REC, models=[vp_b])
    assert calls["n"] == 2
    floor, got = _rel(c1, c2), _rel(c1, e)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"per-shot-model batched c vs eager {got:.2e}, floor {floor:.2e}")


def test_pershot_model_gradients_stay_per_shot(monkeypatch):
    """Zeroing every loss term except shot k's must leave d/dvp[j], j != k,
    at (numerically) zero: with per-shot models the shots share NO material
    tensor, so cross-shot gradient leakage can only come from a stride bug."""
    calls = _spy(monkeypatch)
    wav = _wav()
    k = 2
    vp_b = _vp_pershot(grad=True)
    s = _prop("c")(wav, SRC, REC, models=[vp_b])
    (s[k] ** 2).sum().div(s[k].detach().pow(2).sum().clamp_min(1e-30)).backward()
    assert calls["n"] == 1
    g = vp_b.grad
    gk = float(g[k].abs().max())
    assert gk > 0.0, "shot k's own gradient vanished"
    for j in range(B):
        if j == k:
            continue
        leak = float(g[j].abs().max()) / gk
        assert leak < 1e-6, f"gradient leaked into shot {j}: {leak:.2e}"


# ---------------------------- boundary saving ----------------------------- #

def _grad(memory=None, vp=None, wav=None, src=None):
    vp = _vp_shared(grad=True) if vp is None else vp
    wav = _wav() if wav is None else wav
    src = SRC if src is None else src
    s = _prop("c", memory)(wav, src, REC, models=[vp])
    (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30)).backward()
    return vp.grad.clone()


def test_batched_bs_grad_matches_batched_saveall(monkeypatch):
    """BoundarySaving() on the BATCHED path: the fp32 band ring reconstructs
    the C-PML states exactly, so the gradient must equal the save-all batched
    gradient at the atomics floor -- while storing (nt, B, nband) instead of
    (nt, B, ng) x 5."""
    calls = _spy(monkeypatch)
    g_bs = _grad(memory=BoundarySaving())
    g_bs2 = _grad(memory=BoundarySaving())        # run-to-run floor
    g_full = _grad(memory=None)
    assert calls["n"] == 3, "the batched fast path did not run for BS"
    floor, got = _rel(g_bs, g_bs2), _rel(g_bs, g_full)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"batched BS vs batched save-all {got:.2e}, floor {floor:.2e}")


def test_batched_bs_grad_matches_looped_bs():
    """Reference semantics: B independent single-shot BS runs accumulating
    into one shared vp."""
    g_batch = _grad(memory=BoundarySaving())
    floor = _rel(g_batch, _grad(memory=BoundarySaving()))
    vp_ref = _vp_shared(grad=True)
    wav = _wav()
    seis = [_prop("c", BoundarySaving())(wav, SRC[b:b + 1], REC,
                                         models=[vp_ref]) for b in range(B)]
    sref = torch.cat(seis, 0)
    (sref ** 2).sum().div(sref.detach().pow(2).sum().clamp_min(1e-30)).backward()
    got = _rel(g_batch, vp_ref.grad)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"batched BS vs looped BS {got:.2e}, floor {floor:.2e}")


def test_batched_bs_actually_uses_the_ring(monkeypatch):
    """If the bs flag were dropped on the batched path, BoundarySaving()
    would silently run the full save-all tape -- the exact silent-fallback
    this suite exists to prevent. _band_alloc is only reached when the ring
    is really allocated."""
    calls = {"n": 0}
    orig = prop_c._band_alloc

    def wrapper(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(prop_c, "_band_alloc", wrapper)
    _grad(memory=BoundarySaving())
    assert calls["n"] == 1, "BoundarySaving() did not allocate the band ring"
    calls["n"] = 0
    _grad(memory=None)
    assert calls["n"] == 0


def test_batched_bs_pershot_models(monkeypatch):
    """BS composes with per-shot models: grads match the per-shot save-all
    batch at the floor, and stay isolated per shot."""
    calls = _spy(monkeypatch)
    wav = _wav()
    vp_b = _vp_pershot(grad=True)
    s = _prop("c", BoundarySaving())(wav, SRC, REC, models=[vp_b])
    norm = s.detach().pow(2).sum().clamp_min(1e-30)
    (s ** 2).sum().div(norm).backward()
    assert calls["n"] == 1
    g_bs = vp_b.grad.clone()

    vp_b2 = _vp_pershot(grad=True)                # floor
    s2 = _prop("c", BoundarySaving())(wav, SRC, REC, models=[vp_b2])
    (s2 ** 2).sum().div(norm).backward()
    floor = _rel(g_bs, vp_b2.grad)

    vp_b3 = _vp_pershot(grad=True)                # save-all reference
    s3 = _prop("c")(wav, SRC, REC, models=[vp_b3])
    (s3 ** 2).sum().div(norm).backward()
    got = _rel(g_bs, vp_b3.grad)
    assert got <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"per-shot-model batched BS vs save-all {got:.2e}, floor {floor:.2e}")


def test_batched_bs_int8_scales_are_per_shot():
    """The int8 ring quantises against an amax scale. Batched, that scale
    must be PER (field, step, SHOT): with wavelet amplitudes spanning 1e3
    across the batch, a batch-shared scale (set by the loudest shot) would
    quantise the quietest shot's band to zero and destroy its gradient. The
    reference is the per-shot loop, where each shot trivially owns its scale.

    Uses per-shot MODELS so the gradients stay per-shot -- a summed shared
    gradient would be dominated by the loud shots and could hide the damage.
    """
    scale = torch.logspace(-3, 0, B, device="cuda")[:, None]      # 1e-3 .. 1
    wav_b = _wav()[None] * scale                                  # (B, nt)
    vp_b = _vp_pershot(grad=True)
    int8 = BoundarySaving(storage_dtype="int8")
    s = _prop("c", int8)(wav_b, SRC, REC, models=[vp_b])
    norm = s.detach().pow(2).sum().clamp_min(1e-30)
    (s ** 2).sum().div(norm).backward()
    g_batch = vp_b.grad

    vp_ref = _vp_pershot(grad=True)
    for b in range(B):
        sb = _prop("c", int8)(wav_b[b], SRC[b:b + 1], REC, models=[vp_ref[b]])
        (sb ** 2).sum().div(norm).backward()
    for b in range(B):
        ref = vp_ref.grad[b]
        scale_b = float(ref.abs().max())
        d = _rel(g_batch[b], ref, scale_b)
        assert d <= 5e-3, (
            f"shot {b} (amplitude {float(scale[b]):.0e}): batched int8 BS "
            f"deviates {d:.2e} from its own-scale loop reference -- the "
            f"quantisation scale is not per-shot")


@pytest.mark.parametrize("store", ["bf16", "int8"])
def test_batched_bs_low_precision_rings_stay_close(store):
    """bf16/int8 rings are approximations; hold them to the measured band
    (eager: cos 0.9999997 / 0.999988) rather than the atomics floor. Also
    pin that they actually differ from the fp32 ring -- a low-precision
    request that silently ran fp32 would pass any closeness bound."""
    g32 = _grad(memory=BoundarySaving())
    gq = _grad(memory=BoundarySaving(storage_dtype=store))
    a, b = g32.double().flatten(), gq.double().flatten()
    cos = float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))
    assert cos >= (0.9999 if store == "bf16" else 0.999), f"{store}: cos={cos}"
    assert _rel(g32, gq) > 0, f"{store} ring is bit-identical to fp32 -- fell back?"


def test_pershot_model_gradients_match_independent_runs(monkeypatch):
    calls = _spy(monkeypatch)
    wav = _wav()
    vp_b = _vp_pershot(grad=True)
    s = _prop("c")(wav, SRC, REC, models=[vp_b])
    norm = s.detach().pow(2).sum().clamp_min(1e-30)
    (s ** 2).sum().div(norm).backward()
    assert calls["n"] == 1
    g_batch = vp_b.grad.clone()

    vp_b2 = _vp_pershot(grad=True)                # floor: batched, again
    s2 = _prop("c")(wav, SRC, REC, models=[vp_b2])
    (s2 ** 2).sum().div(norm).backward()
    floor = _rel(g_batch, vp_b2.grad)

    vp_ref = _vp_pershot(grad=True)               # independent single shots
    for b in range(B):
        sb = _prop("c")(wav, SRC[b:b + 1], REC, models=[vp_ref[b]])
        (sb ** 2).sum().div(norm).backward()
    d = _rel(g_batch, vp_ref.grad)
    assert d <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
        f"per-shot-model batched grads vs independent {d:.2e}, "
        f"floor {floor:.2e}")
