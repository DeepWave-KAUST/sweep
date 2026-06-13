"""Single-GPU NON-DD regression bench: normal public prop() path.

Times the ordinary monolithic forward and the forward+backward FWI iteration
(no model_parallel, no stepped driver) so we can A/B the current DD branch
build against the origin/dev build and confirm the gated DD additions did not
slow the single-card path.

  PYTHONPATH=<worktree>/src python _dd_cuda/bench_single_gpu.py --eq acoustic2d ...

Reports a SGBENCH line per (eq,size,so,bs): pure-forward ms and fwd+bwd ms
(best of repeats), plus cv. Warm-up >= 1.6 s wall (RTX 6000 Ada clock ramp).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

# Select the build under test via PYTHONPATH=<worktree>/src — do NOT auto-insert
# this file's own worktree, or the A/B (current branch vs origin/dev) would
# always load the current branch regardless of PYTHONPATH.
import sweep  # noqa: E402
from sweep.equations import Acoustic, Acoustic3D, Elastic, Elastic3D  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


EQ = {
    "acoustic2d": (Acoustic, 2, ["h1"], ["h1"], "cpmlr", 1),
    "acoustic3d": (Acoustic3D, 3, ["h1"], ["h1"], "cpmlr", 1),
    "elastic2d": (Elastic, 2, ["sxx", "szz"], ["vx", "vz"], "cpmls", 3),
    "elastic3d": (Elastic3D, 3, ["sxx", "syy", "szz"], ["vx", "vy", "vz"],
                  "cpmls", 3),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", required=True, choices=list(EQ))
    ap.add_argument("--nz", type=int, default=1024)
    ap.add_argument("--ny", type=int, default=192)
    ap.add_argument("--nx", type=int, default=1024)
    ap.add_argument("--nt", type=int, default=500)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.0004)
    ap.add_argument("--bs", default="gpu", choices=("gpu", "none"))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    eq_cls, ndim, stype, rtype, pml, nmodel = EQ[args.eq]
    shape = (args.nz, args.nx) if ndim == 2 else (args.nz, args.ny, args.nx)

    if args.eq.startswith("acoustic"):
        models_np = [np.full(shape, 2500.0, dtype=np.float32)]
    else:
        models_np = [np.full(shape, 3000.0, dtype=np.float32),
                     np.full(shape, 1730.0, dtype=np.float32),
                     np.full(shape, 2200.0, dtype=np.float32)]

    if ndim == 2:
        src = np.array([[[shape[1] // 2, shape[0] // 4]]], dtype=np.int32)
        rec = np.array([[[ix, 2] for ix in range(2, shape[1] - 2, 4)]],
                       dtype=np.int32)
    else:
        src = np.array([[[shape[2] // 2, shape[1] // 2, shape[0] // 4]]],
                       dtype=np.int32)
        rec = np.array([[[ix, shape[1] // 2, 2]
                         for ix in range(2, shape[2] - 2, 4)]], dtype=np.int32)
    wavelet = ricker(args.nt, args.dt, scale=1.0e6 if "elastic" in args.eq else 1.0)

    bs_cfg = ({"enabled": True, "storage": "gpu", "transfer_interval": 1,
               "pinned_memory": False} if args.bs == "gpu"
              else {"enabled": False})
    eq = eq_cls(spatial_order=args.so, device=dev, backend="torch")
    prop = PropTorch(
        eq, backend="torch", impl="c", shape=shape, dev=dev, dh=10.0,
        dt=args.dt, source_type=stype, receiver_type=rtype, abcn=args.abcn,
        free_surface=False, pml_type=pml, nt=args.nt, B=1, use_ckpt=False,
        boundary_saving_config=bs_cfg,
    )

    def fwd_only():
        with torch.no_grad():
            prop(wavelet, src, rec,
                 models=[torch.tensor(m, device=dev) for m in models_np])
        torch.cuda.synchronize()

    def fwd_bwd():
        models = [torch.tensor(m, device=dev, requires_grad=True)
                  for m in models_np]
        syn = prop(wavelet, src, rec, models=models)
        r = syn[0] if isinstance(syn, (tuple, list)) else syn
        r.pow(2).sum().backward()
        torch.cuda.synchronize()

    # warm-up >= 1.6 s (clock ramp)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.6:
        fwd_bwd()

    def timeit(fn, n):
        ts = []
        for _ in range(n):
            a = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - a)
        return min(ts), float(np.std(ts) / np.mean(ts))

    f_best, f_cv = timeit(fwd_only, args.repeats)
    fb_best, fb_cv = timeit(fwd_bwd, args.repeats)
    peak = torch.cuda.max_memory_allocated() / 2**30
    grid = "x".join(str(s) for s in shape)
    # parents[2] = the worktree dir (…/src/sweep/__init__.py); if it resolves
    # to site-packages the PYTHONPATH override failed — visible in the tag.
    build_tag = Path(sweep.__file__).resolve().parents[2].name
    print(f"SGBENCH build={build_tag} eq={args.eq} grid={grid} nt={args.nt} "
          f"so={args.so} bs={args.bs} fwd={f_best * 1e3:.2f}ms(cv{f_cv:.3f}) "
          f"fwd_bwd={fb_best * 1e3:.2f}ms(cv{fb_cv:.3f}) "
          f"peak_mem={peak:.2f}GB")


if __name__ == "__main__":
    main()
