#!/usr/bin/env python3
"""Gradient comparison on the seistorch qP LSRTM example model (Mu et al. 2020
style; seistorch jax branch, examples/lsrtm/qP/generate_model.py rebuilt here):
eager against impl='c' full and boundary saving (fp32, gpu).

151 x 301 cells of 10 m, three tilted layers (30 / 45 / 60 deg, epsilon 0.15,
delta 0.08, isotropic elsewhere), smoothed background (gaussian sigma 2),
m_true = 2 (vp - vp_smooth) / vp_smooth.  Sources and receivers at z = 1, 18 Hz
Ricker, dt 1 ms.  Eager full mode keeps every step, so the record is cut to
NT steps and a few shots are summed one at a time.

Figures:
  * born: the LSRTM gradient at m = 0 (the RTM image) of 0.5 |d - d_obs|^2,
    d_obs the eager TTI Born data of m_true; rows VTI adjoint, TTI adjoint,
    TTI frozen;
  * fwi-tti-<mode>: the TTI gradient of vp, epsilon, delta, theta at the
    smoothed vp (epsilon, delta, theta true), d_obs the eager TTI data of the
    true vp.
Columns: eager | c-full | c-bs | c-full - eager | c-bs - c-full.  Colour
limits and rel_l2 below the first interface (row TOP); each difference has its
own colour range.  Difference titles: rel_l2, and the floor: eager on models x
(1 + 1e-7 noise) against eager.

    PYTHONPATH=src python test/qp_grad_figures_mu.py --out DIR
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
from matplotlib.path import Path as MPath  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))

import qp_aniso_consistency as Q  # noqa: E402
from sweep.equations import AcousticTTI, AcousticTTIBorn, AcousticVTIBorn  # noqa: E402
from sweep.propagator.options import BoundarySaving, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

NZ, NX, DH, DT, FM, DELAY, ABCN = 151, 301, 10.0, 1e-3, 18.0, 0.1, 30
RUNS = ("eager", "c-full", "c-bs", "eager-pert")
TOP = 30        # rows above the first interface (z = 260 m): left out of limits and rel_l2


def _fill(a, poly, value):
    yy, xx = np.meshgrid(np.arange(a.shape[1]), np.arange(a.shape[0]))
    inside = MPath(poly).contains_points(np.vstack((yy.ravel(), xx.ravel())).T).reshape(a.shape)
    a[inside] = value
    return a


def mu_model():
    """generate_model.py of the seistorch example: vp, theta (deg), epsilon, delta."""
    yat = lambda x0, y0, x1, th: y0 - np.tan(np.deg2rad(th)) * (x1 - x0)
    vp = np.zeros((NZ, NX), np.float32)
    th = np.zeros((NZ, NX), np.float32)
    vp = _fill(vp, [(-1, -1), (NX, -1), (NX, 25), (-1, 25)], 2000)
    vp = _fill(vp, [(-1, 125), (NX, 125), (NX, NZ), (-1, NZ)], 4000)
    vp = _fill(vp, [(-1, 100), (100, 100), (110, 126), (-1, 126)], 3500)
    c1, c3 = (100, 100), (110, 126)
    for theta, x2, x4, v in ((30, 150, 160, 3750), (45, 170, 190, 3000), (60, 185, 215, 2750)):
        c2, c4 = (x2, yat(*c1, x2, theta)), (x4, yat(*c3, x4, theta))
        vp = _fill(vp, [c1, c2, c4, c3], v)
        th = _fill(th, [c1, c2, c4, c3], theta)
        c1, c3 = c2, c4
    vp[:26] = 2000.0
    vp[vp == 0.0] = 2500.0
    eps = np.zeros((NZ, NX), np.float32)
    dl = np.zeros((NZ, NX), np.float32)
    for v in (3500, 3750, 3000, 2750):
        eps[vp == v] = 0.15
        dl[vp == v] = 0.08
    vps = gaussian_filter(vp.copy(), sigma=2).astype(np.float32)
    m = (2 * (vp - vps) / vps).astype(np.float32)
    return dict(vp=vp, vp_smooth=vps, theta=np.deg2rad(th).astype(np.float32), epsilon=eps, delta=dl, m=m)


def geometry(shots):
    src = np.array([[x, 1] for x in shots], np.int32)
    rx = np.arange(0, NX, 2, dtype=np.int32)
    rec = np.stack([rx, np.ones_like(rx)], -1)[None]
    return src, rec


def prop(cls, sg, label, nt):
    eq = cls(spatial_order=4, device="cuda", backend="torch", s_gradient=sg)
    kw = dict(shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, dev="cuda", nt=nt, B=1)
    if label.startswith("eager"):
        return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    return PropTorch(eq, impl="c", memory=BoundarySaving(storage="gpu") if label == "c-bs" else Full(), **kw)


def gradients(cls, sg, nt, shots, obs_cls, obs_models, models, grad_names):
    """Sum over shots of the gradient of 0.5 |d - d_obs|^2 for every run.

    d_obs is modelled by the same implementation as the run (eager for the
    eager runs, c-full for c-full and c-bs): impl='c' and eager absorb the
    late returns from the CPML differently (qp_pml_check.py), and a d_obs of
    the other implementation puts that difference into the residual."""
    wav = torch.tensor(Q.ricker(nt, DT, FM, DELAY), device="cuda")
    out = {lab: {n: 0.0 for n in grad_names} for lab in RUNS}
    names = cls(spatial_order=4).models
    for x in shots:
        src, rec = geometry([x])
        obs_ms = [torch.tensor(a, device="cuda") for a in obs_models]
        with torch.no_grad():
            d_obs_by = {"eager": prop(obs_cls, sg, "eager", nt)(wav, src, rec, models=obs_ms),
                        "c": prop(obs_cls, sg, "c-full", nt)(wav, src, rec, models=obs_ms)}
        for lab in RUNS:
            d_obs = d_obs_by["eager" if lab.startswith("eager") else "c"]
            arrays = Q.perturb(models, seed=x) if lab == "eager-pert" else models
            ms = [torch.tensor(a, device="cuda", requires_grad=n in grad_names) for n, a in zip(names, arrays)]
            d = prop(cls, sg, lab, nt)(wav, src, rec, models=ms)
            (0.5 * (d - d_obs).pow(2).sum()).backward()
            for n, m in zip(names, ms):
                if n in grad_names:
                    out[lab][n] = out[lab][n] + m.grad.double().cpu().numpy()
            del d, ms
            torch.cuda.empty_cache()
        print(f"  shot x={x} done", flush=True)
    return out


def pct(arrs):
    v = np.concatenate([a[TOP:].ravel() for a in arrs])
    v = v[v != 0]
    return np.percentile(v, [2, 98]) if v.size else (-1.0, 1.0)


def rel(a, ref):
    return np.linalg.norm((a - ref)[TOP:]) / np.linalg.norm(ref[TOP:])


def figure(title, rows, path):
    """rows: list of (label, {run: array})."""
    ext = [0, NX * DH / 1000, NZ * DH / 1000, 0]
    fig, axes = plt.subplots(len(rows), 5, figsize=(22, 2.7 * len(rows) + 0.7), squeeze=False)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for r, (lab, g) in enumerate(rows):
        e, f, b, p = g["eager"], g["c-full"], g["c-bs"], g["eager-pert"]
        lo, hi = pct([e])
        d1, d2 = pct([f - e]), pct([b - f])
        fl = rel(p, e)
        panels = [(e, f"{lab} eager", lo, hi), (f, f"{lab} c-full", lo, hi), (b, f"{lab} c-bs", lo, hi),
                  (f - e, f"full-eager {rel(f, e):.0e} (floor {fl:.0e})", *d1),
                  (b - f, f"bs-full {rel(b, f):.0e}", *d2)]
        for c, (a, t, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(a, cmap="seismic", vmin=vmin, vmax=vmax, extent=ext, aspect="auto")
            ax.set_title(f"({next(letters)}) {t}", fontsize=10)
            if c == 0:
                ax.set_ylabel("z (km)")
            if r == len(rows) - 1:
                ax.set_xlabel("x (km)")
            if c in (2, 3, 4):
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format="%.1e")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def model_figure(md, path):
    try:
        from sweepx.tasks.viz.colormaps import VP_CMAP
    except ImportError:
        VP_CMAP = "viridis"
    ext = [0, NX * DH / 1000, NZ * DH / 1000, 0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 5.6))
    items = [(md["vp"], "vp (m/s)", VP_CMAP), (md["vp_smooth"], "vp smooth (m/s)", VP_CMAP),
             (md["m"], "m true", "gray"), (np.rad2deg(md["theta"]), "theta (deg)", "viridis"),
             (md["epsilon"], "epsilon", "viridis"), (md["delta"], "delta", "viridis")]
    for ax, (a, t, cm), l in zip(axes.ravel(), items, "abcdef"):
        kw = dict(vmin=1500, vmax=4000) if "vp" in t else (
            dict(vmin=np.percentile(a[a != 0], 2), vmax=np.percentile(a[a != 0], 98)) if t == "m true" else {})
        fig.colorbar(ax.imshow(a, cmap=cm, extent=ext, aspect="auto", **kw), ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(f"({l}) {t}", fontsize=10)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--nt", type=int, default=1201)
    ap.add_argument("--shots", nargs="+", type=int, default=[30, 90, 150, 210, 270])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = mu_model()
    model_figure(md, out / "mu_model.png")
    tag = f"nt{args.nt}_{len(args.shots)}shots"
    saved = {}

    # Born: the RTM image (m = 0); d_obs the eager TTI Born data of m_true
    obs = [md["vp_smooth"], md["epsilon"], md["delta"], md["theta"], md["m"]]
    rows = []
    for cls, sg, lab in ((AcousticVTIBorn, "adjoint", "VTI adjoint"), (AcousticTTIBorn, "adjoint", "TTI adjoint"),
                         (AcousticTTIBorn, "frozen", "TTI frozen")):
        bg = [md["vp_smooth"], md["epsilon"], md["delta"]] + ([md["theta"]] if cls is AcousticTTIBorn else [])
        print(f"born {lab}", flush=True)
        g = gradients(cls, sg, args.nt, args.shots, AcousticTTIBorn, obs, bg + [np.zeros_like(md["m"])], ["mp"])
        rows.append((f"m {lab}", {k: v["mp"] for k, v in g.items()}))
        for k, v in g.items():
            saved[f"born|{lab}|{k}"] = v["mp"].astype(np.float32)
    figure(f"LSRTM gradient at m = 0 (TTI Born data), {len(args.shots)} shots, {args.nt} steps", rows,
           out / f"grad_born_mu_{tag}.png")

    # TTI FWI: smoothed vp, true epsilon / delta / theta; d_obs from the true vp
    names = ["vp", "epsilon", "delta", "theta"]
    true = [md["vp"], md["epsilon"], md["delta"], md["theta"]]
    init = [md["vp_smooth"], md["epsilon"], md["delta"], md["theta"]]
    for sg in ("adjoint", "frozen"):
        print(f"fwi tti {sg}", flush=True)
        g = gradients(AcousticTTI, sg, args.nt, args.shots, AcousticTTI, true, init, names)
        figure(f"TTI {sg} FWI gradient, {len(args.shots)} shots, {args.nt} steps",
               [(n, {k: v[n] for k, v in g.items()}) for n in names], out / f"grad_fwi_tti_{sg}_mu_{tag}.png")
        for k, v in g.items():
            for n in names:
                saved[f"fwi-tti|{sg}|{k}|{n}"] = v[n].astype(np.float32)
    np.savez_compressed(out / f"grads_mu_{tag}.npz", **saved)


if __name__ == "__main__":
    main()
