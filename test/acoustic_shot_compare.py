#!/usr/bin/env python3
"""Shot gathers of the isotropic Acoustic: eager against impl='c', and each
against a reference without boundary (eager on the model edge-padded by PAD
cells on every side, so nothing comes back within nt), i.e. what each
absorbing layer sends back.

Model: vp of the seistorch qP LSRTM example (qp_grad_figures_mu.mu_model),
151 x 301 cells of 10 m, no free surface, receivers every cell at z = 1,
18 Hz Ricker, dt 1 ms, 2001 steps.  Rows: a shot in the middle and one near
the left edge.  Columns: eager | c | c - eager | eager - ref | c - ref; the
three differences share one colour range, GAIN x tighter than the gathers'.

    PYTHONPATH=src python test/acoustic_shot_compare.py --out DIR
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))

import qp_aniso_consistency as Q  # noqa: E402
from qp_grad_figures_mu import DH, DT, FM, NX, NZ, mu_model  # noqa: E402
from sweep.equations import Acoustic  # noqa: E402
from sweep.propagator.options import EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

ABCN, PAD, NT, DELAY, GAIN = 30, 420, 2001, 0.1, 100.0


def gather(vp, impl, sx, off=0):
    nz, nx = vp.shape
    eq = Acoustic(spatial_order=4, device="cuda", backend="torch")
    kw = dict(shape=(nz, nx), abcn=ABCN, dh=DH, dt=DT, dev="cuda", nt=NT, B=1)
    if impl == "eager":
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=Full(), **kw)
    src = np.array([[sx + off, 1 + off]], np.int32)
    rx = np.arange(NX, dtype=np.int32)
    rec = np.stack([rx + off, np.full_like(rx, 1 + off)], -1)[None]
    wav = torch.tensor(Q.ricker(NT, DT, FM, DELAY), device="cuda")
    with torch.no_grad():
        d = prop(wav, src, rec, models=[torch.tensor(vp, device="cuda")])
    return d.double().cpu().numpy()[0, :, :, 0] if d.ndim == 4 else d.double().cpu().numpy()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", nargs="+", type=int, default=[150, 30])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vp = mu_model()["vp"]
    big = np.pad(vp, PAD, mode="edge").astype(np.float32)
    ext = [0, NX * DH / 1000, NT * DT, 0]
    fig, axes = plt.subplots(len(args.shots), 5, figsize=(22, 6.0 * len(args.shots)), squeeze=False)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    saved = {}
    for r, sx in enumerate(args.shots):
        e = gather(vp, "eager", sx)
        c = gather(vp, "c", sx)
        ref = gather(big, "eager", sx, off=PAD)
        nz_ = e[e != 0]
        lo, hi = np.percentile(nz_, [2, 98])
        dlo, dhi = lo / GAIN, hi / GAIN
        n = lambda a: np.linalg.norm(a) / np.linalg.norm(ref)
        panels = [(e, "eager", lo, hi), (c, "c", lo, hi),
                  (c - e, f"c-eager x{GAIN:.0f}  ({n(c - e):.0e})", dlo, dhi),
                  (e - ref, f"eager-ref x{GAIN:.0f}  ({n(e - ref):.0e})", dlo, dhi),
                  (c - ref, f"c-ref x{GAIN:.0f}  ({n(c - ref):.0e})", dlo, dhi)]
        for col, (a, t, vmin, vmax) in enumerate(panels):
            ax = axes[r, col]
            ax.imshow(a, cmap="gray", vmin=vmin, vmax=vmax, extent=ext, aspect="auto")
            ax.set_title(f"({next(letters)}) shot x={sx * DH / 1000:.1f} km, {t}", fontsize=10)
            ax.set_xlabel("x (km)")
            if col == 0:
                ax.set_ylabel("t (s)")
        for k, a in (("eager", e), ("c", c), ("ref", ref)):
            saved[f"x{sx}|{k}"] = a.astype(np.float32)
        print(f"shot {sx}: |c-eager| {n(c - e):.2e}  |eager-ref| {n(e - ref):.2e}  |c-ref| {n(c - ref):.2e} "
              "(L2 over the gather, relative to the reference gather)", flush=True)
    fig.suptitle("Acoustic shot gathers, no free surface, abcn 30; differences x"
                 f"{GAIN:.0f}, (norm) = L2 relative to the reference gather", fontsize=12)
    fig.tight_layout()
    path = out / "acoustic_shot_compare.png"
    fig.savefig(path, dpi=300)
    np.savez_compressed(out / "acoustic_shot_compare.npz", **saved)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
