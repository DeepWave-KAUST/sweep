"""Shot-batched impl='c' COUPLED FLUID-SOLID path: batched vs per-shot loop.

Discipline (same as test_sem_shot_batch.py, the acoustic twin):

* a spy on ``sem_c_forward_shot_batch`` proves the fast path actually ran --
  without it every "parity" test silently degenerates to loop-vs-loop;
* the reference is B INDEPENDENT single-shot calls (or the per-shot loop,
  forced by monkeypatching ``sem_c_shot_batchable`` off), so a batch-axis
  stride bug cannot cancel out of both sides;
* thresholds are anchored to the run-to-run nondeterminism floor of the
  batched call itself (atomics reorder), not to magic constants.

The fs batch shares ONE model by design (no Bm); per-shot models must
therefore DECLINE the fast path and still be correct via the loop.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem import prop_c
from sweep.sem.equations import AcousticElasticSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NT, DT = 4, 25.0, 6.0, 200, 0.0016
NZ, NX, NW, B = 49, 81, 8, 3
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4

SRC = np.stack([np.linspace(10, NX - 11, B).astype(np.int64),
                np.full(B, 3, dtype=np.int64)], -1)           # in-water, x varies
RIX = np.arange(6, NX - 6, 3)
REC = np.stack([RIX, np.full_like(RIX, NW + 2)], -1)[None]    # OBS, solid side


def _wav(per_shot=False):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                     device="cuda")
    if not per_shot:
        return w
    return w[None] * torch.linspace(0.5, 1.5, B, device="cuda")[:, None]


def _prop(impl="c", memory=None):
    return S.PropSEM(AcousticElasticSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=4 * NO, f_dominant=F0, memory=memory,
                     dev="cuda", impl=impl,
                     receiver_type=["p", "vx", "vz"])


def _models(p, grad=False, pershot=False):
    m = p.mesh
    fl = (m.z_e.flatten(1).mean(1) < NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    vp = torch.where(fl, 1500.0, 2200.0).float().cuda()
    vs = torch.where(fl, 0.0, 1200.0).float().cuda()
    rho = torch.where(fl, 1000.0, 2100.0).float().cuda()
    if pershot:
        # Per-shot vp on the GRID (leading B axis, what _resolve_model_batch
        # recognises): a strong shallow solid anomaly under a different x per
        # shot, negative so the batch-max profile equals the plain build's.
        # vs / rho stay ELEMENT-level shared -- the fluid/solid split is
        # driven by vs, and a grid-level vs step would interpolate into the
        # interface elements and be rejected as mixed.
        z = torch.arange(NZ, device="cuda")[:, None]
        x = torch.arange(NX, device="cuda")[None, :]
        rows = []
        for b in range(B):
            cx = 10 + b * ((NX - 20) // max(B - 1, 1))
            bump = -500.0 * torch.exp(-((x - cx) ** 2 + (z - NW - 4) ** 2) / 50.0)
            base = torch.where(z < NW, 1500.0, 2200.0).expand(NZ, NX).float()
            rows.append(base + bump * (z >= NW))
        ms = [torch.stack(rows).float(), vs, rho]
    else:
        ms = [vp, vs, rho]
    return [t.requires_grad_(True) for t in ms] if grad else ms


def _spy(monkeypatch):
    calls = {"n": 0}
    orig = prop_c.sem_c_forward_shot_batch

    def wrapper(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(prop_c, "sem_c_forward_shot_batch", wrapper)
    return calls


def _no_batch(monkeypatch):
    """Force the per-shot loop (the reference semantics)."""
    monkeypatch.setattr(prop_c, "sem_c_shot_batchable", lambda eq: False)


def _rel(a, b, scale=None):
    a, b = a.double(), b.double()
    d = float(a.abs().max()) if scale is None else float(scale)
    return float((a - b).abs().max() / d) if d else 0.0


def _loss(s):
    return (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30))


# ------------------------------ forward ----------------------------------- #

@pytest.mark.parametrize("per_shot_wavelet", [False, True],
                         ids=["shared-wavelet", "pershot-wavelet"])
def test_fs_batched_forward_matches_independent_runs(monkeypatch,
                                                     per_shot_wavelet):
    calls = _spy(monkeypatch)
    wav = _wav(per_shot_wavelet)
    p = _prop()
    with torch.no_grad():
        got = p(wav, SRC, REC, models=_models(p))
        again = p(wav, SRC, REC, models=_models(p))
    assert calls["n"] == 2, "the batched fast path did not run"
    assert got.shape == (B, NT, REC.shape[1], 3)   # p, vx, vz ride the batch
    outs = []
    for b in range(B):
        w = wav if wav.dim() == 1 else wav[b]
        pb = _prop()
        with torch.no_grad():
            outs.append(pb(w, SRC[b:b + 1], REC, models=_models(pb)))
    ref = torch.cat(outs, 0)
    floor = _rel(got, again)
    scale = float(got.abs().max())
    for b in range(B):                    # per-shot: contamination cannot hide
        d = _rel(got[b], ref[b], scale)
        assert d <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"shot {b}: batched vs independent {d:.2e}, floor {floor:.2e}")


def test_fs_shots_differ_from_each_other():
    """If a stride bug replayed shot 0 into every batch slot, the parity test
    above could only catch it through the reference. Assert directly."""
    p = _prop()
    with torch.no_grad():
        got = p(_wav(), SRC, REC, models=_models(p))
    for b in range(1, B):
        assert _rel(got[0], got[b]) > 1e-3, f"shot {b} equals shot 0"


# ------------------------------ gradients --------------------------------- #

@pytest.mark.parametrize("memory", [None, BoundarySaving()],
                         ids=["save-all", "boundary"])
def test_fs_batched_gradients_match_the_loop(monkeypatch, memory):
    """Shared model: the batched per-shot gradients summed on the Python side
    must equal the per-shot loop's autograd accumulation -- for ALL of vp,
    vs, rho (they chain to every material the fs backward emits)."""
    calls = _spy(monkeypatch)
    wav = _wav()
    p = _prop(memory=memory)
    ms = _models(p, grad=True)
    s = p(wav, SRC, REC, models=ms)
    _loss(s).backward()
    assert calls["n"] == 1, "the batched fast path did not run"
    g_batch = [t.grad.clone() for t in ms]

    # floor: the same batched call, run again
    p2 = _prop(memory=memory)
    ms2 = _models(p2, grad=True)
    _loss(p2(wav, SRC, REC, models=ms2)).backward()
    floors = [_rel(a, t.grad) for a, t in zip(g_batch, ms2)]

    # reference: the per-shot loop through the SAME loss
    _no_batch(monkeypatch)
    p3 = _prop(memory=memory)
    ms3 = _models(p3, grad=True)
    _loss(p3(wav, SRC, REC, models=ms3)).backward()
    for nm, gb, fl, t in zip(("vp", "vs", "rho"), g_batch, floors, ms3):
        d = _rel(gb, t.grad)
        assert d <= max(FLOOR_FACTOR * fl, ABS_CEILING), (
            f"{nm}: batched grad vs loop {d:.2e}, floor {fl:.2e}")


def test_fs_batched_bs_grad_matches_batched_saveall(monkeypatch):
    """Boundary saving only changes WHAT IS STORED; on the batched path the
    rings and the save-all histories are disjoint storage, so agreement here
    falsifies the batched ring layout end to end."""
    calls = _spy(monkeypatch)
    wav = _wav()
    p = _prop()
    ms = _models(p, grad=True)
    _loss(p(wav, SRC, REC, models=ms)).backward()
    p2 = _prop(memory=BoundarySaving())
    ms2 = _models(p2, grad=True)
    _loss(p2(wav, SRC, REC, models=ms2)).backward()
    assert calls["n"] == 2
    # floor: batched save-all, run again
    p3 = _prop()
    ms3 = _models(p3, grad=True)
    _loss(p3(wav, SRC, REC, models=ms3)).backward()
    for nm, a, b, f in zip(("vp", "vs", "rho"), ms, ms2, ms3):
        fl = _rel(a.grad, f.grad)
        d = _rel(a.grad, b.grad)
        assert d <= max(FLOOR_FACTOR * fl, ABS_CEILING), (
            f"{nm}: bs vs save-all (batched) {d:.2e}, floor {fl:.2e}")


# ------------------------------ fallbacks --------------------------------- #

def test_fs_pershot_models_decline_the_batch(monkeypatch):
    """The fs kernels share one model (no Bm). A per-shot model stack must
    fall back to the per-shot loop -- and still be CORRECT (the guard that
    would otherwise silently broadcast model 0 over the batch)."""
    calls = _spy(monkeypatch)
    wav = _wav()
    p = _prop()
    with torch.no_grad():
        got = p(wav, SRC, REC, models=_models(p, pershot=True))
    assert calls["n"] == 0, "fs must decline the batch for per-shot models"
    outs = []
    ms = _models(p, pershot=True)
    for b in range(B):
        pb = _prop()
        with torch.no_grad():
            outs.append(pb(wav, SRC[b:b + 1], REC,
                           models=[ms[0][b], ms[1], ms[2]]))
    ref = torch.cat(outs, 0)
    scale = float(got.abs().max())
    for b in range(B):
        d = _rel(got[b], ref[b], scale)
        assert d <= ABS_CEILING, f"shot {b}: loop fallback wrong ({d:.2e})"
