#!/usr/bin/env python3
"""How good a descent direction is each qP gradient, on an FWI misfit?

d_obs from a model with a vp / epsilon anomaly, J(m) = 0.5 |d(m) - d_obs|^2 at
the smooth model.  For s_gradient in (exact, frozen), all in float64 eager:
  * cos(g, g_exact),
  * the true directional derivative of J along -g/|g| (central FD of the fp64
    forward) against the linear prediction -<g_exact, g>/|g|,
  * J after a line-search step along -g (best of a few step lengths).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))

import qp_aniso_consistency as Q  # noqa: E402
import qp_aniso_diag as D  # noqa: E402


def forward64(cls, so, cfg, models, s_gradient, need_grad, d_obs=None):
    prop = D.eager64(cls, so, cfg, s_gradient)
    cuda = torch.device("cuda")
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=cuda,
                       dtype=torch.float64)
    ms = [torch.tensor(a, device=cuda, dtype=torch.float64, requires_grad=need_grad) for a in models]
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    d = prop(wav, src, rec, models=ms)
    if d_obs is None:
        return d.detach()
    J = 0.5 * (d - d_obs).pow(2).sum()
    if not need_grad:
        return float(J)
    J.backward()
    return float(J), [m.grad.detach().double() for m in ms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", nargs="+", default=["vti", "tti"])
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--config", default="long")
    ap.add_argument("--aniso", nargs="+", default=["eps>delta"])
    ap.add_argument("--save", default="qp_fwi_dir", help="output file prefix")
    args = ap.parse_args()
    cfg = Q.CONFIGS[args.config]
    nz, nx = cfg["shape"]
    z = np.linspace(0, 1, nz)[:, None]
    x = np.linspace(0, 1, nx)[None, :]
    bump = np.exp(-(((z - 0.6) / 0.12) ** 2 + ((x - 0.5) / 0.15) ** 2))
    for e in args.eq:
        cls = Q.EQUATIONS[e]
        names = cls(spatial_order=args.so).models
        for an in args.aniso:
            m0 = Q.model_arrays(names, cfg["shape"], an)
            for target in ("vp", "epsilon"):
                mt = [a.copy() for a in m0]
                i = names.index(target)
                mt[i] = (mt[i] + (80.0 if target == "vp" else 0.05) * bump).astype(np.float32)
                d_obs = forward64(cls, args.so, cfg, mt, "exact", False)
                J0, g_ex = forward64(cls, args.so, cfg, m0, "exact", True, d_obs)
                _, g_fr = forward64(cls, args.so, cfg, m0, "frozen", True, d_obs)
                print(f"\n=== {e} so{args.so} {args.config} {an}: anomaly in {target}, J0={J0:.6e}", flush=True)
                # one model at a time: the anomaly's own parameter
                for k, n in enumerate(names):
                    ge, gf = g_ex[k].flatten(), g_fr[k].flatten()
                    cos = float(ge @ gf / (ge.norm() * gf.norm()))
                    print(f"  d/d{n:8s} cos(frozen, exact) = {cos:+.4f}   |frozen|/|exact| = {float(gf.norm() / ge.norm()):.3f}",
                          flush=True)
                k = names.index(target)
                np.savez(f"{args.save}_{e}_{target}.npz", exact=g_ex[k].cpu().numpy(),
                         frozen=g_fr[k].cpu().numpy(), anomaly=(mt[k] - m0[k]))
                for label, g in (("exact", g_ex[k]), ("frozen", g_fr[k])):
                    ga = g.abs().flatten()
                    top = torch.sort(ga ** 2, descending=True).values
                    share = float(top[: max(1, top.numel() // 100)].sum() / top.sum())
                    # a direction whose largest cell moves by `scale` (vp m/s, epsilon)
                    dirn = (-g / g.abs().max()).cpu().numpy()
                    scale = 20.0 if target == "vp" else 0.01
                    h = 0.25 * scale
                    mp = [a.copy() for a in m0]
                    mm = [a.copy() for a in m0]
                    mp[k] = m0[k] + h * dirn
                    mm[k] = m0[k] - h * dirn
                    Jp = forward64(cls, args.so, cfg, mp, "exact", False, d_obs)
                    Jm = forward64(cls, args.so, cfg, mm, "exact", False, d_obs)
                    fd = (Jp - Jm) / (2 * h)
                    lin = float((g_ex[k] * torch.tensor(dirn, device=g.device)).sum())
                    best, best_s = J0, 0.0
                    for step in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
                        ms = [a.copy() for a in m0]
                        ms[k] = m0[k] + step * scale * dirn
                        Js = forward64(cls, args.so, cfg, ms, "exact", False, d_obs)
                        if Js < best:
                            best, best_s = Js, step
                    print(f"  -g_{label:6s}: top-1% cells hold {share:.2f} of |g|^2;"
                          f"  dJ/ds FD {fd:+.4e} (exact-grad linear {lin:+.4e});"
                          f"  best J/J0 {best / J0:.4f} at {best_s * scale:g} max-cell step", flush=True)


if __name__ == "__main__":
    main()
