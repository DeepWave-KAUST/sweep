#!/usr/bin/env python3
"""eager vs impl='c' consistency of the qP Born (LSRTM) pair.

``AcousticVTILSRTM`` / ``AcousticTTILSRTM`` run on ``qp_vti_lsrtm2d`` /
``qp_tti_lsrtm2d`` in full and boundary-saving mode.  The background models
are fixed (no grad, as in LSRTM), the reflectivity ``mp`` requires grad; the
misfit is 0.5 |d(mp) - d_obs|^2 with d_obs the eager Born data of a reflector.
Evaluated at mp = 0 (the RTM gradient) and at mp = 0.5 mp_true (a non-zero
scattered field).  Judged as qp_aniso_consistency.py: against the case's floor
(eager cpu, and eager gpu on background models x (1 + 1e-7 noise)), the
reflectivity gradient off the source.

    PYTHONPATH=src python test/qp_born_consistency.py --out born.json
"""
from __future__ import annotations

import argparse
import json
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
from sweep.equations import AcousticTTILSRTM, AcousticVTILSRTM  # noqa: E402
from sweep.propagator.options import EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

EQUATIONS = {"vti": AcousticVTILSRTM, "tti": AcousticTTILSRTM}
MODES = dict(Q.MODES)   # eager-cpu: eager's own device spread, the floor of the comparison


def reflectivity(shape):
    nz, nx = shape
    z = np.arange(nz)[:, None] * np.ones((1, nx))
    x = np.arange(nx)[None, :] * np.ones((nz, 1))
    mp = np.zeros(shape, np.float32)
    mp[np.abs(z - (0.6 * nz + 0.15 * (x - nx / 2))) < 1.0] = 0.2      # a dipping reflector
    return mp


def build(cls, so, label, sg, dev, cfg):
    eq = cls(spatial_order=so, device=dev, backend="torch", s_gradient=sg)
    kw = dict(shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev=dev, nt=cfg["nt"], B=1)
    if label.startswith("eager"):
        return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    return PropTorch(eq, impl="c", memory=MODES[label][1](), **kw)


def run(cls, so, label, sg, cfg, aniso, mp_np, d_obs):
    dev = torch.device("cpu" if label == "eager-cpu" else "cuda")
    perturbed = label == "eager-gpu-pert"
    if d_obs is not None:
        d_obs = d_obs.to(dev)
    prop = build(cls, so, label, sg, dev, cfg)
    names = cls(spatial_order=so).models
    bg = Q.model_arrays(names[:-1], cfg["shape"], aniso)
    if perturbed:
        bg = Q.perturb(bg)
    ms = [torch.tensor(a, device=dev) for a in bg]
    mp = torch.tensor(mp_np, device=dev, requires_grad=d_obs is not None)
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=dev)
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    d = prop(wav, src, rec, models=ms + [mp])
    if d_obs is None:
        return d.detach()
    loss = 0.5 * (d - d_obs).pow(2).sum()
    loss.backward()
    return {"record": d.detach().double().cpu(), "mp": mp.grad.detach().double().cpu()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", nargs="+", default=["vti", "tti"])
    ap.add_argument("--orders", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--configs", nargs="+", default=["canonical", "long"])
    ap.add_argument("--aniso", nargs="+", default=["eps>delta", "eps<delta"])
    ap.add_argument("--s-gradient", nargs="+", default=["adjoint"], choices=["adjoint", "frozen", "exact"])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for sg in args.s_gradient:
        for cname in args.configs:
            cfg = Q.CONFIGS[cname]
            mp_true = reflectivity(cfg["shape"])
            for e in args.eq:
                cls = EQUATIONS[e]
                for so in args.orders:
                    for an in args.aniso:
                        with torch.no_grad():
                            d_obs = run(cls, so, "eager", sg, cfg, an, mp_true, None)
                        for start, mp0 in (("mp=0", np.zeros_like(mp_true)), ("mp=mp/2", 0.5 * mp_true)):
                            tag = f"born-{e}/so{so}/{cname}/{an}/{sg}/{start}"
                            ref = run(cls, so, "eager", sg, cfg, an, mp0, d_obs)
                            mask = Q.off_source_mask(cfg["shape"], Q.geometry(cfg["shape"], so, cfg["rec_stride"])[0])
                            keys = ("record", "mp")
                            fl_runs = {}
                            for lab in ("eager-cpu", "eager-gpu-pert"):
                                out = run(cls, so, lab, sg, cfg, an, mp0, d_obs)
                                fl_runs[lab] = {k: Q.region_metrics(out[k], ref[k], mask) for k in keys}
                                rows.append({"case": tag, "mode": lab, "vs": "eager-gpu", "metrics": fl_runs[lab]})
                            floor = Q.floors(fl_runs["eager-cpu"], fl_runs["eager-gpu-pert"])
                            rows.append({"case": tag, "mode": "floor", "floor": floor})
                            print(f"\n== {tag}   floor: record {Q.judged_floor(floor['record']):.1e}"
                                  f"  mp (off the source) {Q.judged_floor(floor['mp']):.1e}", flush=True)
                            results = {}
                            for label in args.modes:
                                if label == "eager-cpu":
                                    continue
                                try:
                                    out = run(cls, so, label, sg, cfg, an, mp0, d_obs)
                                except Exception as exc:
                                    print(f"{label:16s} ERROR {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                                    rows.append({"case": tag, "mode": label, "error": str(exc)})
                                    continue
                                results[label] = out
                                ms = {k: Q.region_metrics(out[k], ref[k], mask) for k in keys}
                                ok = all(Q.floor_pass(ms[k], floor[k]) for k in keys)
                                cells = "   ".join(
                                    f"{k} {Q.judged(ms[k])['rel_l2']:9.2e}"
                                    f"({Q.judged(ms[k])['rel_l2'] / max(Q.judged_floor(floor[k]), 1e-30):5.1f}x)"
                                    for k in keys)
                                verdict = "PASS" if ok else "FAIL"
                                print(f"{label:16s} {cells}   {verdict}", flush=True)
                                rows.append({"case": tag, "mode": label, "vs": "eager-gpu", "matrix_pass": ok,
                                             "metrics": ms})
                            if "c-full" in results:
                                for label, out in results.items():
                                    if label.startswith("c-bs"):
                                        m = Q.region_metrics(out["mp"], results["c-full"]["mp"], mask)
                                        print(f"  {label} vs c-full: mp rel_l2 off the source "
                                              f"{Q.judged(m)['rel_l2']:.2e} (all {m['all']['rel_l2']:.2e})", flush=True)
                                        rows.append({"case": tag, "mode": label, "vs": "c-full",
                                                     "metrics": {"mp": m}})
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
