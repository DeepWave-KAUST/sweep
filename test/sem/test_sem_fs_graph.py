"""CUDA-graphed gradients for the SHOT-BATCHED fluid-solid path (Stage C).

One ``SEMGraphGradRunner(prop, batch=props)`` capture covers every shot of the
batch -- the composition that makes graphing the coupled path practical
(per-shot graphs would pin one private memory pool each). The reference is the
UNGRAPHED batched call through the same Function, so the comparison isolates
exactly what capture/replay adds; thresholds are anchored to the ungraphed
run-to-run atomicAdd floor.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem as S
from sweep.sem import prop_c
from sweep.sem.equations import AcousticElasticSEM2D
from sweep.sem.prop_c import SEMCudaPropagator, sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + the compiled sweep._C SEM binding")

NO, DH, F0, NT, DT = 4, 25.0, 6.0, 200, 0.0016
NZ, NX, NW, B = 49, 81, 8, 3

SRC = np.stack([np.linspace(10, NX - 11, B).astype(np.int64),
                np.full(B, 3, dtype=np.int64)], -1)
RIX = np.arange(6, NX - 6, 3)
REC = np.stack([RIX, np.full_like(RIX, NW + 2)], -1)[None]


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                        device="cuda")


def _propsem():
    return S.PropSEM(AcousticElasticSEM2D, shape=(NZ, NX), dh=DH, dt=DT,
                     nt=NT, n_order=NO, free_surface=("top",),
                     abcn=4 * NO, f_dominant=F0,
                     dev="cuda", impl="c", receiver_type=["p", "vx", "vz"])


def _models(p, grad=True, scale=1.0):
    m = p.mesh
    fl = (m.z_e.flatten(1).mean(1) < NW * DH).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    vp = torch.where(fl, 1500.0, 2200.0 * scale).float().cuda()
    vs = torch.where(fl, 0.0, 1200.0).float().cuda()
    rho = torch.where(fl, 1000.0, 2100.0).float().cuda()
    ms = [vp, vs, rho]
    return [t.requires_grad_(True) for t in ms] if grad else ms


def _harvest(p):
    """One throwaway call populates the propagator's stamp caches."""
    with torch.no_grad():
        p(_wav(), SRC, REC, models=_models(p, grad=False))
    return p


def _shot_props(p, eq):
    """The per-shot SEMCudaPropagators of the batch, exactly as the PropSEM
    fast path builds them (kind/payload derivation included)."""
    kind = "scalar" if p.is_acoustic else "force"
    payload = (1.0,) if p.is_acoustic else p._source_vector(None)
    return [SEMCudaPropagator(
        eq=eq, dt=p.dt, nt=p.nt, sources=p._c_src_stamps[b],
        receivers=p._c_rec_stamps[0],
        source_kinds=[kind] * len(p._c_src_stamps[b]),
        source_payloads=[payload] * len(p._c_src_stamps[b]),
        receiver_type=p._recv_base) for b in range(B)]


def _grads(out_bnrt, models):
    """Weighted linear loss (a plain sum leaves fs grads near the denormal
    floor); ``out_bnrt`` is (B, nrec, nt, ncomp)."""
    W = torch.linspace(0.4, 1.6, out_bnrt.numel(),
                       device=out_bnrt.device).reshape(out_bnrt.shape)
    for m in models:
        m.grad = None
    (W * out_bnrt).sum().div(out_bnrt.detach().abs().max().clamp_min(1e-30)
                             ).backward()
    return [m.grad.detach().clone() for m in models]


def _cos(a, b):
    aa, bb = a.reshape(-1).double(), b.reshape(-1).double()
    na, nb = float(aa.norm()), float(bb.norm())
    return float(aa @ bb) / (na * nb) if na and nb else float("nan")


@pytest.mark.parametrize("memory", [None, "boundary"],
                         ids=["save-all", "boundary"])
def test_fs_graphed_batch_gradients_match_ungraphed(memory):
    from sweep.sem.graph_runner import SEMGraphGradRunner
    p = _harvest(_propsem())
    wav = _wav()

    def ungraphed():
        ms = _models(p)
        eq, _ = p._resolve_model_batch(ms)
        out = prop_c.sem_c_forward_shot_batch(_shot_props(p, eq), wav,
                                              memory=memory)
        return _grads(out, ms)

    ref, ref2 = ungraphed(), ungraphed()          # second run = atomics floor

    ms3 = _models(p)
    eq3, _ = p._resolve_model_batch(ms3)
    props = _shot_props(p, eq3)
    runner = SEMGraphGradRunner(props[0], memory=memory, batch=props)
    out = runner(wav, eq=eq3)                     # (B, nt, nrec, ncomp)
    got = _grads(out.transpose(1, 2), ms3)

    for nm, r, r2, g in zip(("vp", "vs", "rho"), ref, ref2, got):
        assert g is not None and torch.isfinite(g).all(), nm
        floor, cg = _cos(r, r2), _cos(r, g)
        assert cg >= floor - 1e-6, (
            f"{nm}: graphed cos={cg:.8f} under the ungraphed floor {floor:.8f}")


def test_fs_graphed_batch_tracks_a_model_change():
    """Replays against two different models must give two different gradients
    -- the property an FWI loop needs from a captured graph."""
    from sweep.sem.graph_runner import SEMGraphGradRunner
    p = _harvest(_propsem())
    wav = _wav()
    ms0 = _models(p)
    eq0, _ = p._resolve_model_batch(ms0)
    props = _shot_props(p, eq0)
    runner = SEMGraphGradRunner(props[0], memory="boundary", batch=props)

    g0 = _grads(runner(wav, eq=eq0).transpose(1, 2), ms0)
    ms1 = _models(p, scale=1.15)
    eq1, _ = p._resolve_model_batch(ms1)
    g1 = _grads(runner(wav, eq=eq1).transpose(1, 2), ms1)

    moved = max(float((b - a).norm() / (a.norm() + 1e-300))
                for a, b in zip(g0, g1))
    assert moved > 1e-3, (
        f"a 15% vp change moved the graphed gradient by only {moved:.2e} "
        "relative -- the graph is not seeing the update")
