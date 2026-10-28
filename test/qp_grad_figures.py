#!/usr/bin/env python3
"""Gradient comparison figures: eager against impl='c' (full, boundary saving).

One figure per (equation, s_gradient): rows = model gradients, columns =
eager | c-full | c-bs (fp32, gpu) | c-full - eager | c-bs - c-full.  The three
gradients share one colour range, the two differences another; colour limits
are the signed 2/98 percentiles of the non-zero samples away from the source
and below the receiver rows.  Titles of the difference panels give rel_l2 off
the source (r > R_SRC cells).

FWI-type misfit 0.5 |d - d_obs|^2: d_obs from the background plus a Gaussian
anomaly in every model, the gradient at the background.  Born: d_obs the
eager Born data of a dipping reflector, the gradient at mp = mp_true / 2.

    PYTHONPATH=src python test/qp_grad_figures.py --out DIR
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

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))

import qp_aniso_consistency as Q  # noqa: E402
import qp_born_consistency as B  # noqa: E402
from sweep.equations import AcousticTTI, AcousticTTIBorn, AcousticVTI, AcousticVTIBorn  # noqa: E402
from sweep.propagator.options import BoundarySaving, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

RUNS = ("eager", "c-full", "c-bs")


def models(names, shape, aniso, anomaly):
    """Q.model_arrays without its bump (the background) or with a stronger one (the truth)."""
    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
    ramp = 0.6 * z + 0.4 * x
    bump = np.exp(-(((z - 0.55) / 0.12) ** 2 + ((x - 0.5) / 0.12) ** 2)).astype(np.float32) * anomaly
    e0, de = Q.ANISO[aniso]["eps"]
    d0, dd = Q.ANISO[aniso]["delta"]
    vals = {"vp": 2000.0 + 500.0 * ramp + 200.0 * bump, "epsilon": e0 + de * (ramp - 0.5) + 0.05 * bump,
            "delta": d0 + dd * (0.5 - x + 0.3 * z) + 0.03 * bump, "theta": 0.35 + 0.45 * x - 0.2 * z + 0.2 * bump}
    return [np.ascontiguousarray(vals[n] + 0 * z + 0 * x, dtype=np.float32) for n in names]


def build(cls, so, sg, cfg, label):
    eq = cls(spatial_order=so, device="cuda", backend="torch", s_gradient=sg)
    kw = dict(shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev="cuda", nt=cfg["nt"], B=1)
    if label == "eager":
        return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    return PropTorch(eq, impl="c", memory=BoundarySaving(storage="gpu") if label == "c-bs" else Full(), **kw)


def fwi_grads(cls, so, sg, cfg, aniso):
    names = cls(spatial_order=so).models
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device="cuda")
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    with torch.no_grad():
        d_obs = build(cls, so, sg, cfg, "eager")(
            wav, src, rec, models=[torch.tensor(a, device="cuda") for a in models(names, cfg["shape"], aniso, 1.0)])
    out = {}
    for lab in RUNS:
        ms = [torch.tensor(a, device="cuda", requires_grad=True) for a in models(names, cfg["shape"], aniso, 0.0)]
        d = build(cls, so, sg, cfg, lab)(wav, src, rec, models=ms)
        (0.5 * (d - d_obs).pow(2).sum()).backward()
        out[lab] = {n: m.grad.double().cpu().numpy() for n, m in zip(names, ms)}
    return names, out, src


def born_grads(cls, so, sg, cfg, aniso):
    names = cls(spatial_order=so).models
    bg = models(names[:-1], cfg["shape"], aniso, 0.0)
    mp_true = B.reflectivity(cfg["shape"])
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device="cuda")
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    bgt = [torch.tensor(a, device="cuda") for a in bg]
    with torch.no_grad():
        d_obs = build(cls, so, sg, cfg, "eager")(wav, src, rec, models=bgt + [torch.tensor(mp_true, device="cuda")])
    out = {}
    for lab in RUNS:
        mp = torch.tensor(0.5 * mp_true, device="cuda", requires_grad=True)
        d = build(cls, so, sg, cfg, lab)(wav, src, rec, models=bgt + [mp])
        (0.5 * (d - d_obs).pow(2).sum()).backward()
        out[lab] = {"mp": mp.grad.double().cpu().numpy()}
    return ["mp"], out, src


def limits(arrs, keep):
    v = np.concatenate([a[keep].ravel() for a in arrs])
    v = v[v != 0]
    return np.percentile(v, [2, 98]) if v.size else (-1.0, 1.0)


def figure(title, names, g, src, cfg, so, path):
    nz, nx = cfg["shape"]
    mask = Q.off_source_mask(cfg["shape"], src).numpy()
    keep = mask.copy()
    keep[: so // 2 + 4] = False                       # receiver rows
    ext = [0, nx * cfg["dh"] / 1000, nz * cfg["dh"] / 1000, 0]
    fig, axes = plt.subplots(len(names), 5, figsize=(17, 2.9 * len(names) + 0.6), squeeze=False)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, n in enumerate(names):
        e, f, b = g["eager"][n], g["c-full"][n], g["c-bs"][n]
        lo, hi = limits([e], keep)
        dlo, dhi = limits([f - e, b - f], keep)
        rel = lambda a, ref: np.linalg.norm((a - ref)[mask]) / np.linalg.norm(ref[mask])
        panels = [(e, f"{n} eager", lo, hi), (f, f"{n} c-full", lo, hi), (b, f"{n} c-bs", lo, hi),
                  (f - e, f"full-eager {rel(f, e):.0e}", dlo, dhi), (b - f, f"bs-full {rel(b, f):.0e}", dlo, dhi)]
        for c, (a, t, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(a, cmap="seismic", vmin=vmin, vmax=vmax, extent=ext, aspect="auto")
            ax.set_title(f"({next(letters)}) {t}", fontsize=10)
            if c == 0:
                ax.set_ylabel("z (km)")
            if r == len(names) - 1:
                ax.set_xlabel("x (km)")
            if c in (2, 4):
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format="%.1e")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="long")
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--aniso", default="eps>delta")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Q.CONFIGS[args.config]
    tag = f"so{args.so}_{args.config}"
    saved = {}
    for eq, cls, fn in (("vti", AcousticVTI, fwi_grads), ("tti", AcousticTTI, fwi_grads),
                        ("born-vti", AcousticVTIBorn, born_grads), ("born-tti", AcousticTTIBorn, born_grads)):
        for sg in ("adjoint", "frozen"):
            names, g, src = fn(cls, args.so, sg, cfg, args.aniso)
            title = f"{eq.upper()} {sg}, so {args.so}, {cfg['shape'][0]}x{cfg['shape'][1]}, {cfg['nt']} steps"
            figure(title, names, g, src, cfg, args.so, out / f"grad_{eq}_{sg}_{tag}.png")
            for lab, d in g.items():
                for n, a in d.items():
                    saved[f"{eq}|{sg}|{lab}|{n}"] = a.astype(np.float32)
    np.savez_compressed(out / f"grads_{tag}.npz", **saved)


if __name__ == "__main__":
    main()
