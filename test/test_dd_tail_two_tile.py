"""Boundary tail truncation × domain decomposition (two x-tiles, one GPU).

Composition gates on top of the proven bit-exact DD backward protocol of
test_dd_backward_two_tile.py:

  1. DD(tail) vs single-domain(tail) — BITWISE.  Both pipelines truncate the
     reverse sweep at the same bs_stop, so the truncated partial sums must
     stay bitwise equal exactly like the untruncated precedent.  Any adjoint
     source is valid here (the gate is composition, not physics — the
     steady-state validity argument lives in test_boundary_tail_truncation).
  2. The per-tile boundary ring allocates tail-shrunk (nt_saved = K), i.e.
     the DD memory win is real, not just a skipped loop.
  3. int8 storage smoke: the tail-shrunk uint8 ring + per-block scale ring
     save/restore without shape errors and the gradient stays close to the
     fp32 DD(tail) gradient (int8 has its own quantization floor, so no
     bitwise/rel gate here — fp32 carries the numerical criteria).
  4. ModelParallel world=1 inherits tail_steps from the wrapped prop's
     config and reproduces the single-domain truncated gradient (the
     config-plumbing + reverse-loop-stop path of dd_propagator, minus NCCL;
     the 2-rank end-to-end lives in dd_tail_nccl_check.py).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT / "test")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sweep.equations import Acoustic  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

from test_dd_backward_two_tile import (  # noqa: E402
    ABCN, M, NT, NXP, NZ, NX, PAD, REC_GX, REC_GZ, SO, SRC_GX, SRC_GZ, DT,
    X_HI_BIT, X_LO_BIT, TileState, capture_both, replay_dd_forward, ricker,
    run_public_once, vp_global,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

TAIL = 60                       # bs_it0 = 60, reverse loop stops at 61
BS_IT0 = NT - TAIL
BS_STOP = BS_IT0 + 1


def make_prop_tail(shape, topo=None, tail=None, storage_dtype="fp32"):
    equation = Acoustic(spatial_order=SO, device=DEV, backend="torch")
    cfg = {
        "enabled": True, "storage": "gpu",
        "storage_dtype": storage_dtype,
        "transfer_interval": 1, "pinned_memory": False,
    }
    if tail:
        cfg["tail_steps"] = tail
    kwargs = dict(
        backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0, dt=DT,
        source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
        free_surface=False, pml_type="cpmlr", nt=NT, B=1, use_ckpt=False,
        boundary_saving_config=cfg,
    )
    if topo is not None:
        kwargs["model_parallel"] = topo
    return PropTorch(equation, **kwargs)


def build_reference_tail(tail):
    sources = np.array([[[SRC_GX, SRC_GZ]]], dtype=np.int32)
    receivers = np.array([[[gx, REC_GZ] for gx in REC_GX]], dtype=np.int32)
    prop = make_prop_tail((NZ, NX), tail=tail)
    cap = capture_both(prop)
    run_public_once(prop, ricker(NT, DT), sources, receivers, vp_global())
    return TileState(cap)


def build_tiles_tail(residual_raw, tail, storage_dtype="fp32"):
    vp = vp_global()
    wavelet = ricker(NT, DT)
    tiles = []
    for xi in range(2):
        topo = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * NXP
        vp_tile = vp[:, x0:x0 + NXP].copy()

        owns_src = (x0 <= SRC_GX < x0 + NXP)
        if owns_src:
            tile_sources = np.array([[[SRC_GX - x0, SRC_GZ]]], dtype=np.int32)
            tile_wavelet = wavelet
        else:
            tile_sources = np.array([[[1, 1]]], dtype=np.int32)
            tile_wavelet = np.zeros_like(wavelet)

        own_rec = [gx for gx in REC_GX if x0 <= gx < x0 + NXP]
        idxs = [REC_GX.index(gx) for gx in own_rec]
        tile_receivers = np.array(
            [[[gx - x0, REC_GZ] for gx in own_rec]], dtype=np.int32
        )

        prop = make_prop_tail((NZ, NXP), topo=topo, tail=tail,
                              storage_dtype=storage_dtype)
        cap = capture_both(prop)
        run_public_once(prop, tile_wavelet, tile_sources, tile_receivers, vp_tile)
        t = TileState(cap)
        t.owns_src = owns_src
        t.cut_face_mask = X_HI_BIT if xi == 0 else X_LO_BIT
        t.x_off = xi * NXP
        t.lo = prop.padding[0] + M
        t.hi = t.lo + NXP

        adj = torch.zeros_like(t.bp.adjoint_source)
        assert adj.shape == (1, len(idxs), NT)
        for j, gi in enumerate(idxs):
            adj[:, j] = residual_raw[:, gi]
        t.bp.adjoint_source = adj
        tiles.append(t)
    return tiles


def replay_dd_backward_tail(tiles, stop):
    """Truncated Design-B reverse loop: identical to the historical loop when
    stop == 0; with stop > 0 no segment below stop is issued (the C++ driver
    would refuse to restore saved slot -1) and the final exchange is skipped
    exactly like the it == 0 break."""
    for t in tiles:
        t.zero_backward_state()
        t.bp.cut_face_mask = t.cut_face_mask
    r0, r1 = tiles[0].bwd_runner(), tiles[1].bwd_runner()
    lo0, hi0, lo1, hi1 = tiles[0].lo, tiles[0].hi, tiles[1].lo, tiles[1].hi
    with torch.no_grad():
        for it in range(NT - 1, stop - 1, -1):
            r0.run_segment(it + 1, it)
            r1.run_segment(it + 1, it)
            if it == stop:
                break
            l0, l1 = r0.lambda_now, r1.lambda_now
            l0[..., hi0:hi0 + M] = l1[..., lo1:lo1 + M]
            l1[..., lo1 - M:lo1] = l0[..., hi0 - M:hi0]
            f0, f1 = r0.recon_u_now, r1.recon_u_now
            f0[..., hi0:hi0 + M] = f1[..., lo1:lo1 + M]
            f1[..., lo1 - M:lo1] = f0[..., hi0 - M:hi0]


def _run_dd_tail(tail, stop, storage_dtype="fp32"):
    """Full DD pipeline (capture -> DD forward replay -> truncated backward),
    returning (tiles, reference TileState with the SAME tail)."""
    ref = build_reference_tail(tail)
    residual_raw = ref.fwd_record_raw.clone()
    ref.bp.adjoint_source = residual_raw

    tiles = build_tiles_tail(residual_raw, tail, storage_dtype=storage_dtype)
    replay_dd_forward(tiles, ref.fp.models[0])
    replay_dd_backward_tail(tiles, stop)
    return tiles, ref


def _mono_truncated_grads(ref):
    ref.zero_backward_state()
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj
    ref.bp.forward_wavefields = ref.recon
    ref.bwd_func(ref.bp)
    return ref


# ------------------------------------------------------------------ gates
@cuda_only
def test_dd_tail_bitwise_vs_single_domain():
    tiles, ref = _run_dd_tail(TAIL, BS_STOP)
    assert int(ref.bp.boundary_tail_steps) == TAIL   # plumbing reached the C params
    _mono_truncated_grads(ref)

    t0, t1 = tiles
    own0 = slice(t0.lo, t0.lo + NXP)
    own1 = slice(t1.lo, t1.lo + NXP)
    g_ref = ref.gbufs[1]
    assert g_ref.abs().max() > 0, "degenerate reference (all-zero gradient)"

    assert torch.equal(t0.gbufs[1][..., own0], g_ref[..., PAD:PAD + NXP]), \
        "tile0 grad_vp differs (bitwise, truncated)"
    assert torch.equal(t1.gbufs[1][..., own1], g_ref[..., PAD + NXP:PAD + 2 * NXP]), \
        "tile1 grad_vp differs (bitwise, truncated)"
    for k, name in enumerate(["source_illum", "receiver_illum"]):
        assert torch.equal(t0.ibufs[k][..., own0], ref.ibufs[k][..., PAD:PAD + NXP]), \
            f"tile0 {name} differs (bitwise, truncated)"
        assert torch.equal(
            t1.ibufs[k][..., own1], ref.ibufs[k][..., PAD + NXP:PAD + 2 * NXP]
        ), f"tile1 {name} differs (bitwise, truncated)"
    assert t1.owns_src and torch.equal(t1.gbufs[0], ref.gbufs[0]), \
        "grad_wavelet differs (bitwise, truncated)"


@cuda_only
def test_dd_tail_ring_is_shrunk():
    # per-step ring layout => numel scales exactly with the saved step count
    full = make_prop_tail((NZ, NXP),
                          topo=MeshTopology(py=1, px=2, shot_groups=1,
                                            world_size=2, rank=0))
    cap_f = capture_both(full)
    tailp = make_prop_tail((NZ, NXP),
                           topo=MeshTopology(py=1, px=2, shot_groups=1,
                                             world_size=2, rank=0),
                           tail=TAIL)
    cap_t = capture_both(tailp)
    src = np.array([[[1, 1]]], dtype=np.int32)
    rec = np.array([[[4, REC_GZ]]], dtype=np.int32)
    vp_tile = vp_global()[:, :NXP].copy()
    run_public_once(full, ricker(NT, DT), src, rec, vp_tile)
    run_public_once(tailp, ricker(NT, DT), src, rec, vp_tile)
    n_full = sum(t.numel() for t in cap_f["fp"].boundary_gpu)
    n_tail = sum(t.numel() for t in cap_t["fp"].boundary_gpu)
    assert n_full > 0 and n_tail > 0
    assert n_tail * NT == n_full * TAIL, \
        f"boundary ring not tail-shrunk: {n_tail} vs {n_full} (K={TAIL}, nt={NT})"


@cuda_only
def test_dd_tail_int8_smoke():
    tiles8, _ = _run_dd_tail(TAIL, BS_STOP, storage_dtype="int8")
    tiles32, _ = _run_dd_tail(TAIL, BS_STOP, storage_dtype="fp32")
    for t8, t32 in zip(tiles8, tiles32):
        g8 = t8.gbufs[1][..., t8.lo:t8.lo + NXP].double().flatten()
        g32 = t32.gbufs[1][..., t32.lo:t32.lo + NXP].double().flatten()
        assert torch.isfinite(g8).all()
        cos = torch.nn.functional.cosine_similarity(g8, g32, dim=0).item()
        assert cos > 0.999, f"int8 tail gradient drifted: cos={cos}"


@cuda_only
def test_model_parallel_world1_inherits_tail():
    # dd_propagator plumbing without NCCL: world=1 tile == global domain, so
    # the ModelParallel truncated gradient must match the plain single-domain
    # truncated gradient.  (2-rank end-to-end: dd_tail_nccl_check.py.)
    from sweep.parallel import ModelParallel

    wavelet = ricker(NT, DT)
    src = np.array([[[SRC_GX, SRC_GZ]]], dtype=np.int32)
    rec = np.array([[[gx, REC_GZ] for gx in REC_GX]], dtype=np.int32)
    vp_np = vp_global()

    def mono_grad():
        p = make_prop_tail((NZ, NX), tail=TAIL)
        vp = torch.tensor(vp_np, device=DEV, requires_grad=True)
        syn = p(wavelet, src, rec, models=[vp])
        syn.backward(gradient=syn.detach())
        return vp.grad.detach().clone()

    def ddp_grad():
        p = make_prop_tail((NZ, NX), tail=TAIL)
        topo = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
        ddp = ModelParallel(p, topo)
        assert ddp._btail == TAIL, "tail_steps not inherited by ModelParallel"
        vp = torch.tensor(vp_np, device=DEV, requires_grad=True)
        rec_t = ddp.forward(wavelet, src, rec, models=[vp])
        rec_t.backward(gradient=rec_t.detach())
        assert int(ddp.bp.boundary_tail_steps) == TAIL
        return vp.grad.detach().clone()

    g_mono, g_ddp = mono_grad(), ddp_grad()
    assert g_mono.abs().max() > 0
    mad = (g_mono - g_ddp).abs().max().item()
    rel = mad / (g_mono.abs().max().item() + 1e-30)
    assert rel < 1e-6, f"world=1 ModelParallel tail grad differs: rel={rel:.3e}"
