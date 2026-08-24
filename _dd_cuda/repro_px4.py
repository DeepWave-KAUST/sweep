"""Reproduce the V100 px=4 ulp mismatch on one GPU with manual halo copies.

Same geometry as dd_nccl_check px=4 small: 48 x 112, src at (56, 12) — on
the tile1/tile2 cut — nt=120, abcn=8. If this is bitwise vs single domain,
the NCCL/multi-GPU path is the culprit; if not, the 3-cut manual logic is.

PYTHONPATH=src python _dd_cuda/repro_px4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "test"))

from sweep.equations import Acoustic
from sweep.parallel import MeshTopology
from sweep.propagator.torch import PropTorch
from sweep.propagator._stepped import SteppedBindingRunner, acoustic_psi_pairs

DEV = torch.device("cuda")
NZ, NX, NT, DT, SO, ABCN = 48, 112, 120, 0.0015, 4, 8
M = SO // 2
PAD = ABCN + M
PX = 4
NXP = NX // PX


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def make_prop(shape, topo=None):
    eq = Acoustic(spatial_order=SO, device=DEV, backend="torch")
    kw = dict(backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0, dt=DT,
              source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
              free_surface=False, pml_type="cpmlr", nt=NT, B=1,
              use_ckpt=False, boundary_saving_config={"enabled": False})
    if topo is not None:
        kw["model_parallel"] = topo
    return PropTorch(eq, **kw)


def capture(prop):
    cap = {}
    impl = prop._backend_impl
    orig = impl.forward_func

    def wrapper(p):
        out = orig(p)
        cap["params"], cap["raw_out"] = p, out
        return out

    impl.forward_func = wrapper
    cap["func"] = orig
    return cap


def runner_for(prop, wavelet, sources, receivers, vp_np):
    cap = capture(prop)
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=[torch.tensor(vp_np, device=DEV)])
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(9)]
    for t in L:
        t.zero_()
    p.record_out = torch.zeros_like(cap["raw_out"][2])
    return SteppedBindingRunner(func, p, L, acoustic_psi_pairs(2)), p.record_out


def main():
    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp = np.broadcast_to(vp, (NZ, NX)).copy()
    vp[NZ // 3:NZ // 2, NX // 3:2 * NX // 3] += 180.0
    wavelet = ricker(NT, DT)
    src = (NX // 2, NZ // 4)
    rec_gx = list(range(2, NX - 2, 6))

    # reference
    rf, rec_f = runner_for(
        make_prop((NZ, NX)), wavelet,
        np.array([[[src[0], src[1]]]], dtype=np.int32),
        np.array([[[gx, 2] for gx in rec_gx]], dtype=np.int32), vp,
    )
    with torch.no_grad():
        rf.run_to(NT)

    K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    W = K * M
    overlap = len(sys.argv) > 2 and sys.argv[2] == "overlap"
    if overlap:
        assert K == 1, "overlap mode is per-step exchange (K=1) only"

    # tiles. With K-step exchange, a source within W of a cut must ALSO be
    # injected by the neighbouring tile (mirror injection into its halo/pad)
    # so the un-refreshed halo evolves with the source between exchanges.
    tiles = []
    for xi in range(PX):
        topo = MeshTopology(py=1, px=PX, shot_groups=1, world_size=PX, rank=xi)
        x0 = xi * NXP
        vt = vp[:, x0:x0 + NXP].copy()
        local = src[0] - x0
        injects = -W <= local < NXP + W  # owned or within the W-halo
        ts = np.array([[[1, 1]]], dtype=np.int32)  # placeholder, fixed below
        tw = wavelet if injects else np.zeros_like(wavelet)
        own = [gx for gx in rec_gx if x0 <= gx < x0 + NXP]
        tr = (np.array([[[gx - x0, 2] for gx in own]], dtype=np.int32)
              if own else np.array([[[1, 1]]], dtype=np.int32))
        r, rec = runner_for(make_prop((NZ, NXP), topo=topo), tw, ts, tr, vt)
        if injects:
            # overwrite the (already pad-shifted) placeholder with the true
            # run-grid coordinates; halo coords (local<0 or >=NXP) are legal
            r.p.sources_loc[..., 0] = PAD + local
            r.p.sources_loc[..., 1] = PAD + src[1]
        tiles.append(r)

    assert PAD >= W, "halo width K*M must fit inside the symmetric pad"
    lo, hi = PAD, PAD + NXP
    if overlap:
        # SPECFEM-style phase-split: phase 1 computes the cut strips of
        # u_next, the halo copy moves them (manual, no NCCL needed on one
        # GPU), phase 2 computes the interior + tail.  The copied strips
        # are pre-source; the mirror injection (phase-2 add_source into the
        # halo column) re-adds the same wavelet with the same single FP add
        # as the owner tile, so this stays bitwise vs the legacy loop.
        for xi, r in enumerate(tiles):
            r.p.cut_face_mask = (1 if xi > 0 else 0) | (2 if xi < PX - 1 else 0)
        with torch.no_grad():
            for it in range(NT):
                for r in tiles:
                    r.run_phase(it + 1, 1)
                for a in range(PX - 1):
                    fa, fb = tiles[a].u_next, tiles[a + 1].u_next
                    fa[..., hi:hi + W] = fb[..., lo:lo + W]
                    fb[..., lo - W:lo] = fa[..., hi - W:hi]
                for r in tiles:
                    r.run_phase(it + 1, 2)
    else:
        with torch.no_grad():
            for it in range(NT):
                for r in tiles:
                    r.run_to(it + 1)
                if (it + 1) % K == 0 or it == NT - 1:
                    for a in range(PX - 1):
                        for fa, fb in ((tiles[a].u_now, tiles[a + 1].u_now),
                                       (tiles[a].u_prev, tiles[a + 1].u_prev)):
                            fa[..., hi:hi + W] = fb[..., lo:lo + W]
                            fb[..., lo - W:lo] = fa[..., hi - W:hi]

    u_full = rf.u_now
    ok = True
    for xi, r in enumerate(tiles):
        ref = u_full[..., PAD + xi * NXP: PAD + xi * NXP + NXP]
        got = r.u_now[..., lo:hi]
        bit = torch.equal(got, ref)
        mad = (got - ref).abs().max().item()
        print(f"tile {xi}: bitexact={bit} max|diff|={mad:.3e}")
        ok &= bit
    print(f"LOCAL_PX4[{'overlap' if overlap else 'legacy'} K={K}]:",
          "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
