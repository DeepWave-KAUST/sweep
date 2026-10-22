#!/usr/bin/env python3
"""Toy FWI with the pure-qP VTI / TTI equations: which s_gradient is usable?

True model = smooth background + a Gaussian anomaly (vp or epsilon); start
from the background; a few shots and receivers along the surface; L2 misfit;
the gradient is masked near the surface (sources, receivers) and each update
moves the largest cell by a fixed amount (vp m/s or epsilon), with a short
backtracking line search on the true misfit.  For each s_gradient the misfit
history and the model error in the anomaly region are printed, and the final
models saved.

    python test/qp_toy_fwi.py --eq vti --impl eager --iters 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sweep.equations import AcousticTTI, AcousticVTI  # noqa: E402
from sweep.propagator.options import BoundarySaving, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

EQ = {"vti": AcousticVTI, "tti": AcousticTTI}
NZ, NX, DH, ABCN, NT, DT = 80, 100, 10.0, 20, 700, 0.001


def ricker(nt, dt, f=12.0, delay=0.08):
    t = np.arange(nt) * dt - delay
    a = (np.pi * f * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def models(names, anomaly):
    z = np.linspace(0, 1, NZ)[:, None]
    x = np.linspace(0, 1, NX)[None, :]
    bump = np.exp(-(((z - 0.55) / 0.10) ** 2 + ((x - 0.5) / 0.12) ** 2))
    base = {"vp": 2000.0 + 600.0 * z + 0 * x, "epsilon": 0.15 + 0.05 * z + 0 * x,
            "delta": 0.05 + 0.02 * z + 0 * x, "theta": 0.35 + 0.1 * x + 0 * z}
    true = dict(base)
    if anomaly == "vp":
        true["vp"] = base["vp"] + 200.0 * bump
    else:
        true["epsilon"] = base["epsilon"] + 0.08 * bump
    f32 = lambda d: [np.ascontiguousarray(d[n], dtype=np.float32) for n in names]
    return f32(base), f32(true), bump > 0.3


def geometry():
    src = np.array([[x, 3] for x in np.linspace(10, NX - 11, 5).astype(int)], dtype=np.int32)
    rx = np.arange(2, NX - 2, 2, dtype=np.int32)
    rec = np.stack([rx, np.full_like(rx, 3)], -1)
    return src, np.repeat(rec[None], len(src), 0)


def make(cls, impl, s_gradient, dev, memory):
    eq = cls(spatial_order=4, device=dev, backend="torch", s_gradient=s_gradient)
    kw = dict(shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, dev=dev, nt=NT, B=5)
    if impl == "eager":
        return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    mem = {"full": Full(), "bs-fp32": BoundarySaving(storage="gpu")}[memory]
    return PropTorch(eq, impl="c", memory=mem, **kw)


def misfit(prop, wav, src, rec, ms, d_obs, grad):
    ts = [torch.tensor(m, device="cuda", requires_grad=grad) for m in ms]
    d = prop(wav, src, rec, models=ts)
    J = 0.5 * (d - d_obs).pow(2).sum()
    if not grad:
        return float(J), None
    J.backward()
    return float(J), [t.grad.detach().cpu().numpy() for t in ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", default="vti")
    ap.add_argument("--anomaly", default="vp", choices=["vp", "epsilon"])
    ap.add_argument("--impl", default="eager")
    ap.add_argument("--memory", default="full")
    ap.add_argument("--modes", nargs="+", default=["adjoint", "frozen", "exact"])
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--out", default="qp_toy_fwi", help="output file prefix")
    args = ap.parse_args()
    cls = EQ[args.eq]
    names = cls(spatial_order=4).models
    k = names.index(args.anomaly)
    m0, mt, region = models(names, args.anomaly)
    src, rec = geometry()
    wav = torch.tensor(ricker(NT, DT), device="cuda")
    obs_prop = make(cls, "eager", "exact", torch.device("cuda"), "full")
    with torch.no_grad():
        d_obs = obs_prop(wav, src, rec, models=[torch.tensor(m, device="cuda") for m in mt])
    mask = np.ones((NZ, NX), np.float32)
    mask[:8] = 0.0                                  # sources and receivers at z = 3
    mask[8:14] = np.linspace(0, 1, 6)[:, None]
    step0 = 20.0 if args.anomaly == "vp" else 0.01
    result = {}
    for mode in args.modes:
        prop = make(cls, args.impl, mode, torch.device("cuda"), args.memory)
        m = [a.copy() for a in m0]
        hist = []
        t0 = time.perf_counter()
        for it in range(args.iters):
            J, g = misfit(prop, wav, src, rec, m, d_obs, True)
            err = float(np.linalg.norm((m[k] - mt[k])[region]) / np.linalg.norm((m0[k] - mt[k])[region]))
            hist.append((J, err))
            d = -g[k] * mask
            d /= np.abs(d).max()
            # backtracking on the true misfit
            step, accepted = step0, False
            for _ in range(4):
                trial = [a.copy() for a in m]
                trial[k] = trial[k] + step * d
                Jt, _ = misfit(prop, wav, src, rec, trial, d_obs, False)
                if Jt < J:
                    m, accepted = trial, True
                    break
                step *= 0.5
            print(f"[{args.eq}/{args.anomaly}/{args.impl}-{args.memory}/{mode}] it {it:2d}  J/J0 {J / hist[0][0]:.4f}"
                  f"  model err {err:.4f}  step {step if accepted else 0:g}", flush=True)
            if not accepted:
                break
        J, _ = misfit(prop, wav, src, rec, m, d_obs, False)
        err = float(np.linalg.norm((m[k] - mt[k])[region]) / np.linalg.norm((m0[k] - mt[k])[region]))
        hist.append((J, err))
        print(f"[{args.eq}/{args.anomaly}/{args.impl}-{args.memory}/{mode}] final J/J0 {J / hist[0][0]:.4f}"
              f"  model err {err:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
        result[mode] = {"hist": hist}
        np.save(f"{args.out}_{args.eq}_{args.anomaly}_{args.impl}-{args.memory}_{mode}.npy", m[k])
    np.save(f"{args.out}_{args.eq}_{args.anomaly}_true.npy", mt[k])
    np.save(f"{args.out}_{args.eq}_{args.anomaly}_init.npy", m0[k])
    Path(f"{args.out}_{args.eq}_{args.anomaly}_{args.impl}-{args.memory}.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
