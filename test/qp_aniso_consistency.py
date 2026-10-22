#!/usr/bin/env python3
"""eager vs impl='c' consistency of the pure-qP VTI / TTI equations.

``AcousticVTI`` / ``AcousticTTI`` (Liang et al. 2023, Geophysics 88(1) WA335)
run on the compiled ``qp_vti2d`` / ``qp_tti2d`` core entries in full and
boundary-saving mode (gpu / cpu / disk storage, fp32; int8 is refused).  The reference is the
uncompiled eager step on the GPU.

The floor of the comparison is the case's own conditioning, the larger of
eager on the CPU (other summation order) and eager on the GPU with the models
times (1 + 1e-7 noise): the qP step is nonlinear in p and some cases move by
1e-3 under a 1e-7 model change, far more than eager cpu / gpu shows
(qp_bs_diag.py).  The model gradients are judged off the source (more than
R_SRC cells away): the gradient's norm sits almost entirely in the source
singularity, where S_n (homogeneous of degree 0 in p) turns the round-off of
boundary saving's reconstruction into O(1) direction changes; there the
global rel_l2 measures only that neighbourhood.

    PYTHONPATH=src python test/qp_aniso_consistency.py --out qp.json

Every row: record and each gradient (wavelet, vp, epsilon, delta[, theta]) as
rel_l2 = |c - ref| / |ref|, cos, and peak = max|c - ref| / max|ref|, on the
whole array ("all") and, for the model gradients, off the source
("off_source"); fp32 modes pass when rel_l2 <= 3 x floor + 1e-6 (off the source
for model gradients).
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

EQUATIONS = {"vti": AcousticVTI, "tti": AcousticTTI}

# (label, impl, memory factory)
MODES = {
    "eager-cpu": ("eager", None),
    "c-full": ("c", lambda: Full()),
    "c-bs-gpu-fp32": ("c", lambda: BoundarySaving(storage="gpu")),
    "c-bs-cpu-fp32": ("c", lambda: BoundarySaving(storage="cpu", transfer_interval=4)),
    "c-bs-disk-fp32": ("c", lambda: BoundarySaving(storage="disk", transfer_interval=4)),
}

CONFIGS = {
    # the canonical gradient-test geometry of gradient_cases.CUDA_SUITE_CONFIG
    "canonical": dict(shape=(48, 56), abcn=30, nt=120, dt=0.0015, freq=10.0, delay=0.06, dh=10.0,
                      rec_stride=6),
    # longer, so the wave crosses the box, reflects and sits in the CPML a while
    "long": dict(shape=(72, 90), abcn=20, nt=420, dt=0.001, freq=12.0, delay=0.05, dh=10.0,
                 rec_stride=4),
}

ANISO = {
    "eps>delta": dict(eps=(0.20, 0.08), delta=(0.06, 0.04)),
    "eps<delta": dict(eps=(0.08, 0.04), delta=(0.20, 0.06)),
}


R_SRC = 10       # cells around a source left out of the model-gradient criterion
PERT = 1e-7      # relative model perturbation of the conditioning floor
FLOOR_FACTOR, FLOOR_SLACK = 3.0, 1e-6


def perturb(arrays, seed=0):
    rng = np.random.default_rng(seed)
    return [(a * (1.0 + PERT * rng.standard_normal(a.shape))).astype(np.float32) for a in arrays]


def off_source_mask(shape, src, r=R_SRC):
    nz, nx = shape
    z = np.arange(nz)[:, None]
    x = np.arange(nx)[None, :]
    m = np.ones(shape, bool)
    for sx, sz in np.asarray(src).reshape(-1, 2):
        m &= (z - sz) ** 2 + (x - sx) ** 2 > r * r
    return torch.from_numpy(m)


def region_metrics(c, r, mask):
    """``metrics`` on the whole array and, for model-shaped arrays, off the source."""
    out = {"all": metrics(c, r)}
    if mask is not None and tuple(c.shape) == tuple(mask.shape):
        out["off_source"] = metrics(c[mask], r[mask])
    return out


def judged(m):
    return m["off_source"] if "off_source" in m else m["all"]


def floors(per_key_a, per_key_b):
    """Per key and region, the larger rel_l2 of the two floor runs."""
    return {k: {reg: max(per_key_a[k][reg]["rel_l2"], per_key_b[k][reg]["rel_l2"]) for reg in per_key_a[k]}
            for k in per_key_a}


def judged_floor(floor):
    return floor["off_source"] if "off_source" in floor else floor["all"]


def floor_pass(m, floor):
    """rel_l2 within FLOOR_FACTOR x the case's floor (off the source for model gradients);
    a zero reference (the Born record at mp = 0) must be matched exactly."""
    if judged(m)["ref_norm"] == 0.0:
        return judged(m)["norm"] == 0.0
    return judged(m)["rel_l2"] <= FLOOR_FACTOR * judged_floor(floor) + FLOOR_SLACK


def ricker(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float64) * dt - delay
    a = (np.pi * fm * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)


def model_arrays(names, shape, aniso):
    """Smooth but non-uniform models: a constant model hides coefficient-placement
    errors in a transposed stencil (reference_canonical_gradient_test_model)."""
    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
    ramp = 0.6 * z + 0.4 * x
    bump = np.exp(-(((z - 0.55) / 0.18) ** 2 + ((x - 0.45) / 0.22) ** 2)).astype(np.float32)
    out = []
    for n in names:
        if n == "vp":
            out.append(2000.0 + 500.0 * ramp + 300.0 * bump)
        elif n == "epsilon":
            e0, de = ANISO[aniso]["eps"]
            out.append(e0 + de * (ramp - 0.5) + 0.5 * de * bump)
        elif n == "delta":
            d0, dd = ANISO[aniso]["delta"]
            out.append(d0 + dd * (0.5 - x + 0.3 * z) + 0.3 * dd * bump)
        elif n == "theta":
            out.append(0.35 + 0.45 * x - 0.2 * z + 0.15 * bump)
        else:
            raise KeyError(n)
    return [np.ascontiguousarray(a, dtype=np.float32) for a in out]


def geometry(shape, so, rec_stride):
    nz, nx = shape
    M = so // 2
    src = np.array([[nx // 2, max(1, nz // 4)]], dtype=np.int32)          # (x, z)
    rx = np.arange(max(2, M), nx - max(2, M), rec_stride, dtype=np.int32)
    rec = np.stack([rx, np.full_like(rx, max(1, M))], axis=-1)[None]    # (1, nrec, 2)
    return src, rec


S_GRADIENT = "adjoint"


def build(eq_cls, so, label, dev, cfg):
    impl, mem = MODES[label] if label in MODES else ("eager", None)
    eq = eq_cls(spatial_order=so, device=dev, backend="torch", s_gradient=S_GRADIENT)
    kw = dict(shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev=dev,
              nt=cfg["nt"], B=1)
    if impl == "eager":
        return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False),
                         memory=Full(), **kw)
    return PropTorch(eq, impl="c", memory=mem(), **kw)


def run(eq_cls, so, label, cfg, aniso):
    dev = torch.device("cpu" if label == "eager-cpu" else "cuda")
    prop = build(eq_cls, so, "eager" if label.startswith("eager") else label, dev, cfg)
    names = eq_cls(spatial_order=so).models
    wav = torch.tensor(ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=dev,
                       requires_grad=True)
    arrays = model_arrays(names, cfg["shape"], aniso)
    if label == "eager-gpu-pert":
        arrays = perturb(arrays)
    ms = [torch.tensor(a, device=dev, requires_grad=True) for a in arrays]
    src, rec = geometry(cfg["shape"], so, cfg["rec_stride"])
    t0 = time.perf_counter()
    d = prop(wav, src, rec, models=ms)
    loss = d.pow(2).sum()
    loss.backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    el = time.perf_counter() - t0
    out = {"record": d.detach().double().cpu(), "wavelet": wav.grad.detach().double().cpu()}
    for n, m in zip(names, ms):
        out[n] = m.grad.detach().double().cpu()
    return out, el


def metrics(c, r):
    c = c.reshape(-1)
    r = r.reshape(-1)
    dn = float(torch.linalg.vector_norm(c - r))
    rn = float(torch.linalg.vector_norm(r))
    cn = float(torch.linalg.vector_norm(c))
    cos = float(torch.dot(c, r) / (cn * rn)) if rn > 0 and cn > 0 else float("nan")
    pk = float(r.abs().max())
    return {"rel_l2": dn / rn if rn > 0 else float("nan"), "cos": cos,
            "peak": float((c - r).abs().max()) / pk if pk > 0 else float("nan"),
            "ref_norm": rn, "norm": cn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", nargs="+", default=["vti", "tti"])
    ap.add_argument("--orders", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--configs", nargs="+", default=["canonical", "long"])
    ap.add_argument("--aniso", nargs="+", default=["eps>delta", "eps<delta"])
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--s-gradient", nargs="+", default=["adjoint"], choices=["adjoint", "frozen", "exact"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    global S_GRADIENT

    rows = []
    for sg in args.s_gradient:
      S_GRADIENT = sg
      for cname in args.configs:
          cfg = CONFIGS[cname]
          for e in args.eq:
              for so in args.orders:
                  for an in args.aniso:
                      tag = f"{e}/so{so}/{cname}/{an}/{sg}"
                      ref, t_ref = run(EQUATIONS[e], so, "eager-gpu", cfg, an)
                      mask = off_source_mask(cfg["shape"], geometry(cfg["shape"], so, cfg["rec_stride"])[0])
                      keys = list(ref)
                      fl_runs = {}
                      for lab in ("eager-cpu", "eager-gpu-pert"):
                          out, _ = run(EQUATIONS[e], so, lab, cfg, an)
                          fl_runs[lab] = {k: region_metrics(out[k], ref[k], mask) for k in keys}
                      floor = floors(fl_runs["eager-cpu"], fl_runs["eager-gpu-pert"])
                      print(f"\n== {tag}  (eager-gpu reference {t_ref:.2f}s; rel_l2 off the source "
                            f"for model gradients, (x floor))", flush=True)
                      print(f"{'mode':16s} " + " ".join(f"{k:>18s}" for k in keys) + "   matrix  time")
                      print(f"{'floor':16s} " + " ".join(f"{judged_floor(floor[k]):9.1e}{'':9s}" for k in keys))
                      for lab, ms in fl_runs.items():
                          rows.append({"case": tag, "mode": lab, "vs": "eager-gpu", "metrics": ms})
                      rows.append({"case": tag, "mode": "floor", "floor": floor})
                      results = {}
                      for label in args.modes:
                          if label == "eager-cpu":
                              continue
                          try:
                              out, el = run(EQUATIONS[e], so, label, cfg, an)
                          except Exception as exc:   # report and keep going
                              print(f"{label:16s} ERROR {type(exc).__name__}: {exc}", flush=True)
                              rows.append({"case": tag, "mode": label, "error": f"{type(exc).__name__}: {exc}"})
                              continue
                          results[label] = out
                          ms = {k: region_metrics(out[k], ref[k], mask) for k in keys}
                          ok = all(floor_pass(ms[k], floor[k]) for k in keys)
                          cells = " ".join(f"{judged(ms[k])['rel_l2']:9.1e}({judged(ms[k])['rel_l2'] / max(judged_floor(floor[k]), 1e-30):5.1f}x)"
                                           for k in keys)
                          verdict = "PASS" if ok else "FAIL"
                          print(f"{label:16s} {cells}   {verdict:4s}  {el:5.2f}s", flush=True)
                          rows.append({"case": tag, "mode": label, "vs": "eager-gpu", "matrix_pass": ok,
                                       "time_s": el, "metrics": ms})
                      # boundary saving against the compiled full mode: the reconstruction alone
                      if "c-full" in results:
                          for label, out in results.items():
                              if not label.startswith("c-bs"):
                                  continue
                              ms = {k: region_metrics(out[k], results["c-full"][k], mask) for k in keys}
                              each = ", ".join("%s %.1e" % (k, judged(ms[k])["rel_l2"]) for k in keys)
                              print(f"  {label} vs c-full (off the source): {each}", flush=True)
                              rows.append({"case": tag, "mode": label, "vs": "c-full", "metrics": ms})
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
