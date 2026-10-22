#!/usr/bin/env python3
"""impl='c' against eager on the linear qP case (epsilon == delta, S_n == 0).

bs_diag_frozen.npz: with epsilon == delta, so 8, the record of impl='c' sits
2e-5..6e-5 off eager from the first time window, 20-50 x eager's own gpu / cpu
spread and the 1e-7 model-perturbation floor (so 4: 5e-6).  Is it qP-specific
or shared with the isotropic Acoustic (same CPML memories, same lap_coes /
grad_coes path), and where / when does it start?

Receivers on every interior cell give the wavefield at each step; per case:
the relative difference over time and its spatial split (source, edge band,
rest) at the step it first exceeds 10 x the eager floor.

    PYTHONPATH=src python test/qp_so8_linear_check.py --out lin.npz

--c-center gives the eager runs impl='c''s Laplacian centre coefficient,
-2 * (fp32 sum of the fp32 coefficients) (propagator/_c.py), in place of the
eager kernel's fp64 -2 * sum rounded once to fp32; at so 8 / 10 the two differ
by one ulp.
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
from sweep.operators.torch import register_stencil  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.scalars import fd_coefficients  # noqa: E402

# name -> (class, anisotropy: None (isotropic), "zero" (eps = delta = 0), "equal" (eps = delta != 0))
EQS = {"acoustic": (Acoustic, None), "vti-0": (AcousticVTI, "zero"), "vti-eq": (AcousticVTI, "equal"),
       "tti-0": (AcousticTTI, "zero"), "tti-eq": (AcousticTTI, "equal")}


def arrays(names, shape, aniso):
    ms = Q.model_arrays(names, shape, "eps>delta")
    if aniso == "zero":
        for n in ("epsilon", "delta"):
            ms[names.index(n)] = np.zeros(shape, np.float32)
    elif aniso == "equal":
        ms[names.index("delta")] = ms[names.index("epsilon")].copy()
    return ms


def dense_receivers(shape, so):
    nz, nx = shape
    m = so // 2
    zz, xx = np.meshgrid(np.arange(m, nz - m), np.arange(m, nx - m), indexing="ij")
    return np.stack([xx.ravel(), zz.ravel()], -1).astype(np.int32)[None], (nz - 2 * m, nx - 2 * m)


def use_c_centre(eq, so):
    coes = torch.from_numpy(fd_coefficients(2, so)).float()
    centre = -(torch.sum(coes) * 2)
    for k in eq.laplace_kernels:
        k.view(-1)[so // 2] = centre.to(k.device)
        register_stencil(k)


def run(cls, aniso, so, cfg, label, c_centre=False):
    dev = torch.device("cpu" if label == "eager-cpu" else "cuda")
    kw = dict(shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev=dev, nt=cfg["nt"], B=1)
    eq = cls(spatial_order=so, device=dev, backend="torch") if aniso is None else \
        cls(spatial_order=so, device=dev, backend="torch", s_gradient="frozen")
    if label.startswith("eager"):
        if c_centre:
            use_c_centre(eq, so)
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=Full(), **kw)
    ms = [torch.tensor(a, device=dev) for a in arrays(eq.models, cfg["shape"], aniso)]
    if label == "eager-gpu-pert":
        ms = [torch.tensor(a, device=dev) for a in Q.perturb([m.cpu().numpy() for m in ms])]
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=dev)
    src, _ = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    rec, inner = dense_receivers(cfg["shape"], so)
    with torch.no_grad():
        d = prop(wav, src, rec, models=ms)
    return d.double().cpu().numpy()[0].reshape(cfg["nt"], *inner), src


def per_step_rel(a, r):
    num = np.sqrt(((a - r) ** 2).sum((1, 2)))
    den = np.sqrt((r ** 2).sum((1, 2)))
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def split(diff, src_iz, src_ix, edge):
    nz, nx = diff.shape
    z = np.arange(nz)[:, None]
    x = np.arange(nx)[None, :]
    band = (z < edge) | (z >= nz - edge) | (x < edge) | (x >= nx - edge)
    near = ((z - src_iz) ** 2 + (x - src_ix) ** 2 <= 9) & ~band
    e = diff ** 2
    t = e.sum()
    return (e[near].sum() / t, e[band].sum() / t, e[~near & ~band].sum() / t) if t > 0 else (0, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eqs", nargs="+", default=list(EQS))
    ap.add_argument("--orders", nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--configs", nargs="+", default=["canonical", "long"])
    ap.add_argument("--c-center", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    saved = {}
    for cname in args.configs:
        cfg = Q.CONFIGS[cname]
        for so in args.orders:
            for name in args.eqs:
                cls, aniso = EQS[name]
                tag = f"{name}/so{so}/{cname}"
                res = {}
                for lab in ("eager-gpu", "eager-cpu", "eager-gpu-pert", "c-full"):
                    res[lab], src = run(cls, aniso, so, cfg, lab, args.c_center)
                ref = res["eager-gpu"]
                rel = {lab: float(np.linalg.norm(res[lab] - ref) / np.linalg.norm(ref)) for lab in res if lab != "eager-gpu"}
                steps = {lab: per_step_rel(res[lab], ref) for lab in rel}
                floor = np.fmax(steps["eager-cpu"], steps["eager-gpu-pert"])
                print(f"\n== {tag}   whole wavefield rel: " + "  ".join(f"{k} {v:.1e}" for k, v in rel.items()),
                      flush=True)
                nt = cfg["nt"]
                marks = sorted({2, 5, 10, 20, nt // 4, nt // 2, nt - 1})
                for lab in ("c-full", "eager-cpu", "eager-gpu-pert"):
                    print(f"  per step {lab:15s} " + " ".join(f"t{t}:{steps[lab][t]:.1e}" for t in marks), flush=True)
                bad = np.nonzero(steps["c-full"] > 10 * floor)[0]
                m = so // 2
                sx, sz = int(src[0][0]) - m, int(src[0][1]) - m
                if len(bad):
                    t0 = int(bad[0])
                    for t in sorted({t0, min(t0 + 5, nt - 1), nt // 2}):
                        f = split(res["c-full"][t] - ref[t], sz, sx, edge=m + 2)
                        print(f"  c-full - eager at t={t:4d} (rel {steps['c-full'][t]:.1e}, floor {floor[t]:.1e}): "
                              f"fraction source/edge/rest {f[0]:.2f}/{f[1]:.2f}/{f[2]:.2f}", flush=True)
                else:
                    print("  c-full within 10 x floor at every step", flush=True)
                if so == 8:   # the maps of the order in question only (the others for the trend)
                    for lab in ("eager-gpu", "c-full"):
                        saved[f"{tag}|{lab}"] = res[lab].astype(np.float32)
    if args.out:
        np.savez_compressed(args.out, **saved)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
