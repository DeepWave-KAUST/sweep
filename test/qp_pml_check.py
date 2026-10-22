#!/usr/bin/env python3
"""impl='c' against eager at the absorbing boundary of the qP equations.

The linear check (epsilon == delta != 0) and the FWI gradients on the Mu model
(epsilon 0.15 layer reaching the left edge) show c-full and eager parting late
and at the edges, but not with epsilon = delta = 0.  Here, in a homogeneous
medium with a centred source and receivers on every interior cell:

  * c-full against eager, per time window, split edge band / rest;
  * each against a reference without boundary: eager on a box large enough
    that nothing comes back within nt, cut to the same window -- the energy of
    (run - reference) is what the absorbing layer sends back.

Media: iso (eps = delta = 0), eq (eps = delta = 0.15, S_n = 0),
aniso (eps 0.15, delta 0.08); VTI and TTI (theta 30 deg); the isotropic
Acoustic (same CPML memories) as the reference behaviour of the c backend.

    PYTHONPATH=src python test/qp_pml_check.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))

import qp_aniso_consistency as Q  # noqa: E402
from sweep.equations import Acoustic, AcousticTTI, AcousticVTI  # noqa: E402
from sweep.propagator.options import EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

N, BIG, ABCN, NT, DT, DH, FM = 80, 400, 20, 700, 1e-3, 10.0, 15.0
MEDIA = {"iso": (0.0, 0.0), "eq": (0.15, 0.15), "aniso": (0.15, 0.08)}


def run(cls, so, media, impl, n, theta):
    extra = {} if cls is Acoustic else {"s_gradient": "frozen"}
    eq = cls(spatial_order=so, device="cuda", backend="torch", **extra)
    kw = dict(shape=(n, n), abcn=ABCN, dh=DH, dt=DT, dev="cuda", nt=NT, B=1)
    if impl == "eager":
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=Full(), **kw)
    e, d = MEDIA[media]
    vals = {"vp": 2500.0, "epsilon": e, "delta": d, "theta": theta}
    ms = [torch.full((n, n), vals[k], device="cuda") for k in eq.models]
    off = (n - N) // 2
    src = np.array([[n // 2, n // 2]], np.int32)
    zz, xx = np.meshgrid(np.arange(N) + off, np.arange(N) + off, indexing="ij")
    rec = np.stack([xx.ravel(), zz.ravel()], -1).astype(np.int32)[None]
    wav = torch.tensor(Q.ricker(NT, DT, FM, 0.08), device="cuda")
    with torch.no_grad():
        d = prop(wav, src, rec, models=ms)
    return d.double().cpu().numpy()[0].reshape(NT, N, N)


def energy(a, t0, t1, mask=None):
    x = a[t0:t1] if mask is None else a[t0:t1][:, mask]
    return float(np.sqrt((x ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=int, default=4)
    args = ap.parse_args()
    so = args.so
    edge = np.zeros((N, N), bool)
    w = so // 2 + 4
    edge[:w] = edge[-w:] = True
    edge[:, :w] = edge[:, -w:] = True
    # the direct wave reaches the box edge at ~ (N/2 dh) / v + delay; split there
    t_hit = int(((N / 2) * DH / (2500 * np.sqrt(1.3)) + 0.08) / DT)
    wins = [(0, t_hit), (t_hit, t_hit + 150), (t_hit + 150, NT)]
    print(f"so {so}, {N}x{N}, abcn {ABCN}, nt {NT}; windows {wins} (the wave reaches the box edge ~ step {t_hit})",
          flush=True)
    for cls, theta in ((Acoustic, 0.0), (AcousticVTI, 0.0), (AcousticTTI, np.deg2rad(30.0))):
        for media in (["iso"] if cls is Acoustic else MEDIA):
            c = run(cls, so, media, "c", N, theta)
            e = run(cls, so, media, "eager", N, theta)
            ref = run(cls, so, media, "eager", BIG, theta)
            tot = energy(ref, 0, NT)
            print(f"\n== {cls.__name__} {media}", flush=True)
            for (t0, t1) in wins:
                ce = energy(c - e, t0, t1) / max(energy(e, t0, t1), 1e-30)
                ce_edge = energy(c - e, t0, t1, edge) ** 2 / max(energy(c - e, t0, t1) ** 2, 1e-300)
                rc = energy(c - ref, t0, t1) / tot
                re_ = energy(e - ref, t0, t1) / tot
                print(f"  steps {t0:4d}-{t1:4d}: c vs eager rel {ce:.1e} (edge share {ce_edge:.2f}) | "
                      f"boundary return / total: c {rc:.2e}  eager {re_:.2e}", flush=True)


if __name__ == "__main__":
    main()
