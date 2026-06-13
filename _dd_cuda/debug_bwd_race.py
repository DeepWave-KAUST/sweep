"""Localize the nondeterminism in test_dd_backward_two_tile_bitexact.

Splits the pipeline:
  A. DD forward replay determinism: rings + last_two bitwise across 2 replays
  B. DD backward determinism: grads bitwise across 2 backward replays on the
     SAME forward products
  C. DD vs reference compare (the failing assertion) + diff localization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))

import numpy as np
import torch

import test_dd_backward_two_tile as T


def ring_snapshot(tiles):
    snap = []
    for t in tiles:
        snap.append({
            "ring": [b.clone() for b in t.fp.boundary_gpu],
            "last_two": t.fp.last_two.clone(),
        })
    return snap


def cmp_ring(a, b, tag):
    ok = True
    for i, (ta, tb) in enumerate(zip(a["ring"], b["ring"])):
        if not torch.equal(ta, tb):
            d = (ta != tb)
            print(f"  {tag} ring[{i}] differs: {int(d.sum())} elems")
            ok = False
    if not torch.equal(a["last_two"], b["last_two"]):
        d = (a["last_two"] != b["last_two"])
        print(f"  {tag} last_two differs: {int(d.sum())} elems")
        ok = False
    return ok


def main():
    ref = T.build_reference()
    residual_raw = ref.fwd_record_raw.clone()
    ref.bp.adjoint_source = residual_raw

    # reference monolithic backward (as in test)
    ref.zero_backward_state()
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj
    ref.bp.forward_wavefields = ref.recon
    ref.bwd_func(ref.bp)
    g_ref = ref.gbufs[1].clone()

    tiles = T.build_tiles(residual_raw)

    # ---- A. forward replay determinism ----
    T.replay_dd_forward(tiles, ref.fp.models[0])
    snap1 = ring_snapshot(tiles)
    T.replay_dd_forward(tiles, ref.fp.models[0])
    snap2 = ring_snapshot(tiles)
    fwd_ok = True
    for xi in range(2):
        fwd_ok &= cmp_ring(snap1[xi], snap2[xi], f"tile{xi}")
    print(f"A. DD forward replay deterministic (rings+last_two): {fwd_ok}")

    # ---- B. backward determinism on same forward products ----
    T.replay_dd_backward(tiles)
    gb1 = [[g.clone() for g in t.gbufs] for t in tiles]
    T.replay_dd_backward(tiles)
    gb2 = [[g.clone() for g in t.gbufs] for t in tiles]
    bwd_ok = True
    for xi in range(2):
        for j, (a, b) in enumerate(zip(gb1[xi], gb2[xi])):
            if not torch.equal(a, b):
                d = (a != b)
                print(f"  tile{xi} gbufs[{j}] differs across backward replays: "
                      f"{int(d.sum())} elems, max abs diff "
                      f"{(a - b).abs().max().item():.3e}")
                bwd_ok = False
    print(f"B. DD backward deterministic (same forward products): {bwd_ok}")

    # ---- C. DD vs reference ----
    own = slice(T.PAD, T.PAD + T.NXP)
    for xi, t in enumerate(tiles):
        sl = slice(T.PAD + xi * T.NXP, T.PAD + (xi + 1) * T.NXP)
        a = t.gbufs[1][..., own]
        b = g_ref[..., sl]
        eq = torch.equal(a, b)
        print(f"C. tile{xi} grad_vp bitwise == ref: {eq}")
        if not eq:
            d = (a != b).squeeze()
            zz, xx = torch.nonzero(d, as_tuple=True)
            print(f"   {int(d.sum())} differing cells; "
                  f"z range [{zz.min().item()},{zz.max().item()}], "
                  f"x range [{xx.min().item()},{xx.max().item()}] "
                  f"(tile-local owned coords, 0..{T.NXP-1})")
            print(f"   max abs diff {(a-b).abs().max().item():.3e}, "
                  f"max |ref| {b.abs().max().item():.3e}")


if __name__ == "__main__":
    main()

def dump(tag):
    """Dump ref grad + tile grads to disk for cross-process comparison."""
    ref = T.build_reference()
    residual_raw = ref.fwd_record_raw.clone()
    ref.bp.adjoint_source = residual_raw
    ref.zero_backward_state()
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj
    ref.bp.forward_wavefields = ref.recon
    ref.bwd_func(ref.bp)
    g_ref = ref.gbufs[1].clone()

    tiles = T.build_tiles(residual_raw)
    T.replay_dd_forward(tiles, ref.fp.models[0])
    T.replay_dd_backward(tiles)
    torch.save({
        "g_ref": g_ref.cpu(),
        "g_t0": tiles[0].gbufs[1].cpu(),
        "g_t1": tiles[1].gbufs[1].cpu(),
        "ring_t0": [b.cpu() for b in tiles[0].fp.boundary_gpu],
        "lt_t0": tiles[0].fp.last_two.cpu(),
        "ring_ref": [b.cpu() for b in ref.fp.boundary_gpu],
        "lt_ref": ref.fp.last_two.cpu(),
    }, f"/tmp/dd_dump_{tag}.pt")
