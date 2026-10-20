"""Shot-batched impl='c' STANDALONE ELASTIC C-PML path: batched vs loop.

Same discipline as the acoustic / fluid-solid twins (see
test_sem_shot_batch.py): a spy proves the fast path ran, references are
independent runs or the forced per-shot loop, thresholds anchor to the
batched call's own run-to-run atomics floor. The elastic batch shares ONE
model (no Bm); per-shot model stacks must decline the batch.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem import prop_c
from sweep.sem.equations import ElasticSEM2DDeformed
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NT, DT = 4, 25.0, 8.0, 200, 0.0012
NZ, NX, B = 49, 81, 3
FLOOR_FACTOR, ABS_CEILING = 25.0, 1e-4

SRC = np.stack([np.linspace(10, NX - 11, B).astype(np.int64),
                np.full(B, 3, dtype=np.int64)], -1)
RIX = np.arange(6, NX - 6, 3)
REC = np.stack([RIX, np.full_like(RIX, 2)], -1)[None]


def _wav(per_shot=False):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                     device="cuda")
    if not per_shot:
        return w
    return w[None] * torch.linspace(0.5, 1.5, B, device="cuda")[:, None]


def _prop(memory=None):
    return S.PropSEM(ElasticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=4 * NO, f_dominant=F0, memory=memory,
                     dev="cuda", impl="c")


def _models(grad=False, pershot=False):
    z = torch.linspace(0, 1, NZ)[:, None].expand(NZ, NX)
    vp = (2000 + 800 * z).contiguous().float().cuda()
    vs = (1150 + 500 * z).contiguous().float().cuda()
    rho = (2000 + 300 * z).contiguous().float().cuda()
    if pershot:
        # negative shallow anomaly at a different x per shot, so every
        # shot's max-vp (hence the batch profile) matches the shared build
        x = torch.arange(NX, device="cuda")[None, :]
        zz = torch.arange(NZ, device="cuda")[:, None]
        rows = []
        for b in range(B):
            cx = 10 + b * ((NX - 20) // max(B - 1, 1))
            rows.append(vp - 400.0 * torch.exp(
                -((x - cx) ** 2 + (zz - 8) ** 2) / 50.0))
        vp = torch.stack(rows).contiguous()
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


def _rel(a, b, scale=None):
    a, b = a.double(), b.double()
    d = float(a.abs().max()) if scale is None else float(scale)
    return float((a - b).abs().max() / d) if d else 0.0


def _loss(s):
    return (s ** 2).sum().div(s.detach().pow(2).sum().clamp_min(1e-30))


@pytest.mark.parametrize("per_shot_wavelet", [False, True],
                         ids=["shared-wavelet", "pershot-wavelet"])
def test_elastic_batched_forward_matches_independent_runs(monkeypatch,
                                                          per_shot_wavelet):
    calls = _spy(monkeypatch)
    wav = _wav(per_shot_wavelet)
    with torch.no_grad():
        got = _prop()(wav, SRC, REC, models=_models())
        again = _prop()(wav, SRC, REC, models=_models())
    assert calls["n"] == 2, "the batched fast path did not run"
    assert got.shape == (B, NT, REC.shape[1], 2)     # u_x, u_z
    outs = []
    for b in range(B):
        w = wav if wav.dim() == 1 else wav[b]
        with torch.no_grad():
            outs.append(_prop()(w, SRC[b:b + 1], REC, models=_models()))
    ref = torch.cat(outs, 0)
    floor = _rel(got, again)
    scale = float(got.abs().max())
    for b in range(B):
        d = _rel(got[b], ref[b], scale)
        assert d <= max(FLOOR_FACTOR * floor, ABS_CEILING), (
            f"shot {b}: batched vs independent {d:.2e}, floor {floor:.2e}")


def test_elastic_shots_differ_from_each_other():
    with torch.no_grad():
        got = _prop()(_wav(), SRC, REC, models=_models())
    for b in range(1, B):
        assert _rel(got[0], got[b]) > 1e-3, f"shot {b} equals shot 0"


@pytest.mark.parametrize("memory", [None, BoundarySaving()],
                         ids=["save-all", "boundary"])
def test_elastic_batched_gradients_match_the_loop(monkeypatch, memory):
    calls = _spy(monkeypatch)
    wav = _wav()
    ms = _models(grad=True)
    _loss(_prop(memory)(wav, SRC, REC, models=ms)).backward()
    assert calls["n"] == 1, "the batched fast path did not run"

    ms2 = _models(grad=True)          # floor: same batched call again
    _loss(_prop(memory)(wav, SRC, REC, models=ms2)).backward()
    floors = [_rel(a.grad, t.grad) for a, t in zip(ms, ms2)]

    monkeypatch.setattr(prop_c, "sem_c_shot_batchable", lambda eq: False)
    ms3 = _models(grad=True)
    _loss(_prop(memory)(wav, SRC, REC, models=ms3)).backward()
    for nm, a, fl, t in zip(("vp", "vs", "rho"), ms, floors, ms3):
        d = _rel(a.grad, t.grad)
        assert d <= max(FLOOR_FACTOR * fl, ABS_CEILING), (
            f"{nm}: batched grad vs loop {d:.2e}, floor {fl:.2e}")


def test_elastic_batched_bs_grad_matches_batched_saveall(monkeypatch):
    """The rings and the save-all histories are disjoint storage, so
    agreement falsifies the batched elastic ring layout end to end."""
    calls = _spy(monkeypatch)
    wav = _wav()
    ms = _models(grad=True)
    _loss(_prop()(wav, SRC, REC, models=ms)).backward()
    ms2 = _models(grad=True)
    _loss(_prop(BoundarySaving())(wav, SRC, REC, models=ms2)).backward()
    assert calls["n"] == 2
    ms3 = _models(grad=True)          # floor: batched save-all again
    _loss(_prop()(wav, SRC, REC, models=ms3)).backward()
    for nm, a, b, f in zip(("vp", "vs", "rho"), ms, ms2, ms3):
        fl = _rel(a.grad, f.grad)
        d = _rel(a.grad, b.grad)
        assert d <= max(FLOOR_FACTOR * fl, ABS_CEILING), (
            f"{nm}: bs vs save-all (batched) {d:.2e}, floor {fl:.2e}")


def test_elastic_pershot_models_decline_the_batch(monkeypatch):
    """No Bm on the elastic kernels: a per-shot vp stack must fall back to
    the per-shot loop and still be correct."""
    calls = _spy(monkeypatch)
    wav = _wav()
    ms = _models(pershot=True)
    with torch.no_grad():
        got = _prop()(wav, SRC, REC, models=ms)
    assert calls["n"] == 0, "elastic must decline the batch for per-shot models"
    outs = []
    for b in range(B):
        with torch.no_grad():
            outs.append(_prop()(wav, SRC[b:b + 1], REC,
                                models=[ms[0][b], ms[1], ms[2]]))
    ref = torch.cat(outs, 0)
    scale = float(got.abs().max())
    for b in range(B):
        d = _rel(got[b], ref[b], scale)
        assert d <= ABS_CEILING, f"shot {b}: loop fallback wrong ({d:.2e})"
