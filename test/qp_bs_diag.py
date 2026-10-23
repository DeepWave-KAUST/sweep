#!/usr/bin/env python3
"""Where do the large impl='c' deviations of matrix_frozen.json come from?

(A) boundary saving (fp32) against full storage: 1.2e-2 (delta) on vti/so8/long/eps<delta;
(B) c-full against eager: 19x eager's gpu/cpu spread (theta) on tti/so8/canonical/eps<delta.

For each case, three anisotropies (eps>delta, eps<delta and eps=delta, where S_n == 0
and the step is linear) and these runs:

    eager-gpu, eager-gpu on models x (1 + 1e-7 noise), eager-cpu,
    c-full, c-full on the perturbed models, c-bs-gpu (fp32)

The perturbed runs give the conditioning of the case (what a 1e-7 relative model
change does); the per-time-window record errors show where the forward drifts;
the spatial split of the gradient differences (CPML frame of boundary saving /
around the source / the rest) shows where the backward differs.

    PYTHONPATH=src python test/qp_bs_diag.py --out diag.npz
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
from sweep.propagator.options import BoundarySaving, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

CASES = [("vti", 8, "long"), ("tti", 8, "canonical"), ("vti", 4, "long"), ("tti", 8, "long")]
ANISO = ["eps>delta", "eps<delta", "eps=delta"]
RUNS = ["eager-gpu", "eager-gpu-pert", "eager-cpu", "c-full", "c-full-pert", "c-bs-gpu"]


def arrays(names, shape, aniso, pert, seed=0):
    ms = Q.model_arrays(names, shape, "eps>delta" if aniso == "eps=delta" else aniso)
    if aniso == "eps=delta":
        ms[names.index("delta")] = ms[names.index("epsilon")].copy()
    if pert:
        rng = np.random.default_rng(seed)
        ms = [(a * (1.0 + 1e-7 * rng.standard_normal(a.shape))).astype(np.float32) for a in ms]
    return ms


def run(cls, so, cfg, aniso, label, sg):
    dev = torch.device("cpu" if label == "eager-cpu" else "cuda")
    eq = cls(spatial_order=so, device=dev, backend="torch", s_gradient=sg)
    kw = dict(shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev=dev, nt=cfg["nt"], B=1)
    if label.startswith("eager"):
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=BoundarySaving(storage="gpu") if label == "c-bs-gpu" else Full(), **kw)
    names = eq.models
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=dev, requires_grad=True)
    ms = [torch.tensor(a, device=dev, requires_grad=True)
          for a in arrays(names, cfg["shape"], aniso, label.endswith("-pert"))]
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    d = prop(wav, src, rec, models=ms)
    d.pow(2).sum().backward()
    out = {"record": d.detach().double().cpu().numpy(), "wavelet": wav.grad.detach().double().cpu().numpy()}
    out.update({n: m.grad.detach().double().cpu().numpy() for n, m in zip(names, ms)})
    return out


def rel(a, b):
    nb = np.linalg.norm(b)
    return np.linalg.norm(a - b) / nb if nb > 0 else float("nan")


def windows(c, r, n=6):
    """Record rel error per time window, normalised by the window's own reference norm."""
    nt = r.shape[1]
    edges = np.linspace(0, nt, n + 1).astype(int)
    return [rel(c[:, a:b], r[:, a:b]) for a, b in zip(edges[:-1], edges[1:])], \
           [float(np.abs(r[:, a:b]).max()) for a, b in zip(edges[:-1], edges[1:])]


def split(diff, shape, so, src):
    """Fraction of |diff|^2 in the boundary-saving frame (M + 1 cells inside the box),
    within 6 cells of the source, and elsewhere."""
    nz, nx = shape
    w = so // 2 + 1
    z = np.arange(nz)[:, None] * np.ones((1, nx))
    x = np.arange(nx)[None, :] * np.ones((nz, 1))
    frame = (z < w) | (z >= nz - w) | (x < w) | (x >= nx - w)
    sx, sz = src[0]
    near = ((z - sz) ** 2 + (x - sx) ** 2 <= 36) & ~frame
    e = diff ** 2
    tot = e.sum()
    if tot == 0:
        return (0.0, 0.0, 0.0)
    return (e[frame].sum() / tot, e[near].sum() / tot, e[~frame & ~near].sum() / tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-gradient", default="frozen")
    ap.add_argument("--cases", nargs="+", default=[f"{e}/so{s}/{c}" for e, s, c in CASES])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    saved = {}
    for case in args.cases:
        e, so, cname = case.split("/")
        so = int(so[2:])
        cls, cfg = Q.EQUATIONS[e], Q.CONFIGS[cname]
        src, _ = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
        for an in ANISO:
            tag = f"{e}/so{so}/{cname}/{an}/{args.s_gradient}"
            res = {lab: run(cls, so, cfg, an, lab, args.s_gradient) for lab in RUNS}
            keys = list(res["eager-gpu"])
            print(f"\n== {tag}", flush=True)
            print(f"{'pair':34s} " + " ".join(f"{k:>9s}" for k in keys))
            pairs = [("eager-cpu", "eager-gpu"), ("eager-gpu-pert", "eager-gpu"), ("c-full", "eager-gpu"),
                     ("c-full-pert", "c-full"), ("c-bs-gpu", "c-full"), ("c-bs-gpu", "eager-gpu")]
            for a, b in pairs:
                print(f"{a + ' vs ' + b:34s} " + " ".join(f"{rel(res[a][k], res[b][k]):9.1e}" for k in keys),
                      flush=True)
            r = res["eager-gpu"]["record"][0]
            for a in ("eager-cpu", "eager-gpu-pert", "c-full"):
                errs, peaks = windows(res[a]["record"][0], r)
                print(f"  record windows {a:15s} rel " + " ".join(f"{v:8.1e}" for v in errs), flush=True)
            print(f"  record windows {'(peak |d|)':15s}     " + " ".join(f"{v:8.1e}" for v in peaks), flush=True)
            for a, b in (("c-bs-gpu", "c-full"), ("c-full", "eager-gpu"), ("eager-cpu", "eager-gpu")):
                fr = {k: split(res[a][k] - res[b][k], cfg["shape"], so, src) for k in keys if k not in ("record", "wavelet")}
                print(f"  grad diff {a} - {b}: fraction in (frame, source, rest) " +
                      "  ".join(f"{k} {f[0]:.2f}/{f[1]:.2f}/{f[2]:.2f}" for k, f in fr.items()), flush=True)
            for lab, out in res.items():
                for k, v in out.items():
                    saved[f"{tag}|{lab}|{k}"] = v
    if args.out:
        np.savez_compressed(args.out, **saved)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
