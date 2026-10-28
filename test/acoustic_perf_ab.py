#!/usr/bin/env python3
"""Wall time of one forward + backward of the second-order acoustic family's
impl='c' propagators (Acoustic, Acoustic3D, AcousticVRZ(3D), AcousticBorn(3D),
the LSRTM ones also with the RWI vp gradient, ViscoAcoustic), for an A/B of two source trees
(run once per tree, PYTHONPATH pointing at it): production-like sizes, one
shot, the memory modes given (a mode a case refuses or runs out of memory in is
reported and skipped).

Prints one line per (equation, mode): median and min over --reps timed runs
after --warmup untimed ones, with torch.cuda.synchronize around each.

    PYTHONPATH=<tree>/src python test/acoustic_perf_ab.py --tag post --modes full bs
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from sweep.equations import (Acoustic, Acoustic3D, AcousticBorn, AcousticBorn3D, AcousticVRZ,
                             AcousticVRZ3D, ViscoAcoustic)
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch

# name: (class, shape, abcn, spatial order, nt, models needing a gradient)
CASES = {
    "acoustic2d": (Acoustic, (800, 1600), 50, 8, 1500, ("vp",)),
    "acoustic3d": (Acoustic3D, (160, 160, 160), 30, 8, 300, ("vp",)),
    "vrz2d": (AcousticVRZ, (800, 1600), 50, 8, 1500, ("vp", "z")),
    "vrz3d": (AcousticVRZ3D, (160, 160, 160), 30, 8, 300, ("vp", "z")),
    "born2d": (AcousticBorn, (800, 1600), 50, 8, 1500, ("mp",)),
    "born2d_rwi": (AcousticBorn, (800, 1600), 50, 8, 1500, ("mp", "vp")),
    "born3d": (AcousticBorn3D, (160, 160, 160), 30, 8, 300, ("mp",)),
    "born3d_rwi": (AcousticBorn3D, (160, 160, 160), 30, 8, 300, ("mp", "vp")),
    "visco2d": (ViscoAcoustic, (800, 1600), 50, 8, 1500, ("vp",)),
}
MODES = {"full": lambda: Full(), "bs": lambda: BoundarySaving(storage="gpu"),
         "ckpt": lambda: Ckpt(mode="chunk", chunks=20)}


def one(cls, shape, abcn, so, nt, grads, mode):
    eq = cls(spatial_order=so, device="cuda", backend="torch")
    prop = PropTorch(eq, impl="c", memory=MODES[mode](), shape=shape, abcn=abcn, dh=10.0, dt=1e-3,
                     dev="cuda", nt=nt, B=1)
    nd = len(shape)
    vp = torch.full(shape, 2500.0, device="cuda")
    vp += torch.linspace(0, 1000, shape[0], device="cuda").view(-1, *([1] * (nd - 1)))
    gen = torch.Generator(device="cuda").manual_seed(3)
    vals = {"vp": vp, "z": vp * 2000.0, "mp": 0.05 * torch.randn(shape, device="cuda", generator=gen),
            "Q": torch.full(shape, 80.0, device="cuda"), "omega": torch.full(shape, 2 * np.pi * 15.0, device="cuda")}
    models = [vals[n].clone().requires_grad_(n in grads) for n in eq.models]
    if nd == 2:
        src = np.array([[shape[1] // 2, 5]], np.int64)
        rx = np.arange(5, shape[1] - 5, 2, dtype=np.int64)
        rec = np.stack([rx, np.full_like(rx, 5)], -1)[None]
    else:
        src = np.array([[shape[2] // 2, shape[1] // 2, 5]], np.int64)
        g = np.arange(5, shape[2] - 5, 6, dtype=np.int64)
        xx, yy = np.meshgrid(g, g, indexing="ij")
        rec = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 5)], -1)[None]
    t = np.arange(nt) * 1e-3 - 0.1
    a = (np.pi * 15.0 * t) ** 2
    wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device="cuda")

    def step():
        for m in models:
            m.grad = None
        d = prop(wav, src, rec, models=models)
        d.pow(2).sum().backward()

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--modes", nargs="+", default=["full", "bs"])
    ap.add_argument("--rwi-modes", nargs="+", default=["full", "bs"],
                    help="modes for the *_rwi cases (ckpt carries no RWI vp gradient)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    for name in args.cases:
        cls, shape, abcn, so, nt, grads = CASES[name]
        for mode in (args.rwi_modes if name.endswith("_rwi") else args.modes):
            try:
                step = one(cls, shape, abcn, so, nt, grads, mode)
                for _ in range(args.warmup):
                    step()
                ts = []
                for _ in range(args.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    step()
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
            except (RuntimeError, NotImplementedError) as exc:   # incl. CUDA out of memory
                print(f"{args.tag:5s} {name:11s} {mode:4s} SKIPPED {type(exc).__name__}: {str(exc)[:100]}",
                      flush=True)
                step = None
                torch.cuda.empty_cache()
                continue
            print(f"{args.tag:5s} {name:11s} {mode:4s} median {np.median(ts):7.3f} s  min {min(ts):7.3f} s  "
                  f"({args.reps} reps)", flush=True)
            del step
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
