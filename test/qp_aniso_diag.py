#!/usr/bin/env python3
"""Diagnostics for the qP VTI/TTI c-vs-eager gradient gap.

* an fp64 eager reference (the uncompiled eager step, run in float64 by
  patching the propagator's dtype casts) -- the "true" discrete gradient;
* eps == delta, where S_n == 0 and the operator is linear: isolates the
  CPML / stencil adjoint from the S_n linearisation;
* a central finite-difference check of <grad, dm> along a smooth dm.
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
from sweep.propagator.options import EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

Q.ANISO["eps=delta"] = dict(eps=(0.15, 0.06), delta=(0.15, 0.06))


def models_np(names, cfg, aniso):
    arrs = Q.model_arrays(names, cfg["shape"], aniso)
    if aniso == "eps=delta":
        i_e, i_d = names.index("epsilon"), names.index("delta")
        arrs[i_d] = arrs[i_e].copy()
    return arrs


def eager64(eq_cls, so, cfg, s_gradient=None):
    dev = torch.device("cuda")
    eq = eq_cls(spatial_order=so, device=dev, backend="torch", s_gradient=s_gradient or Q.S_GRADIENT)
    prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(),
                     shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev=dev,
                     nt=cfg["nt"], B=1)
    f32, f64 = torch.float32, torch.float64
    impl = getattr(prop, "_backend_impl", prop)    # PropTorch delegates to its backend object
    orig_as = impl._as_device_tensor
    impl._as_device_tensor = lambda v, dtype: orig_as(v, dtype=f64 if dtype == f32 else dtype)
    orig_cache = impl._get_cached_tensor
    impl._get_cached_tensor = lambda kind, shape, device, dtype: orig_cache(
        kind, shape, device=device, dtype=f64 if dtype == f32 else dtype)
    orig_abc = impl.init_abc

    def abc(**kw):
        out = orig_abc(**kw)
        e = impl.equation
        e.b = type(e.b)(t.double() if torch.is_tensor(t) else t for t in e.b)
        return out
    impl.init_abc = abc
    e = impl.equation
    e.laplace_kernels = tuple(k.double() for k in e.laplace_kernels)
    e.grad_kernels = {a: k.double() for a, k in e.grad_kernels.items()}
    return prop


def run_prop(prop, eq_cls, so, cfg, aniso, dtype, dev, dm=None, h=0.0, need_grad=True):
    names = eq_cls(spatial_order=so).models
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=dev,
                       dtype=dtype, requires_grad=need_grad)
    arrs = models_np(names, cfg, aniso)
    ms = []
    for n, a in zip(names, arrs):
        t = torch.tensor(a, device=dev, dtype=dtype)
        if dm is not None and n in dm:
            t = t + h * torch.tensor(dm[n], device=dev, dtype=dtype)
        ms.append(t.requires_grad_(need_grad))
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    d = prop(wav, src, rec, models=ms)
    loss = d.double().pow(2).sum()
    if not need_grad:
        return float(loss)
    loss.backward()
    out = {"record": d.detach().double().cpu(), "wavelet": wav.grad.detach().double().cpu()}
    for n, m in zip(names, ms):
        out[n] = m.grad.detach().double().cpu()
    return out, float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", nargs="+", default=["vti", "tti"])
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--config", default="canonical")
    ap.add_argument("--aniso", nargs="+", default=["eps=delta", "eps>delta"])
    ap.add_argument("--modes", nargs="+", default=["eager-cpu", "c-full", "c-bs-gpu-fp32"])
    ap.add_argument("--fd", action="store_true")
    ap.add_argument("--s-gradient", default="adjoint", choices=["adjoint", "frozen", "exact"])
    args = ap.parse_args()
    Q.S_GRADIENT = args.s_gradient
    cfg = Q.CONFIGS[args.config]
    cuda = torch.device("cuda")
    for e in args.eq:
        cls = Q.EQUATIONS[e]
        names = cls(spatial_order=args.so).models
        for an in args.aniso:
            print(f"\n=== {e} so{args.so} {args.config} {an} s_gradient={args.s_gradient}", flush=True)
            ref64, L64 = run_prop(eager64(cls, args.so, cfg), cls, args.so, cfg, an, torch.float64, cuda)
            ref32, _ = run_prop(Q.build(cls, args.so, "eager", cuda, cfg), cls, args.so, cfg, an,
                                torch.float32, cuda)
            keys = list(ref64)
            hdr = " ".join(f"{k:>22s}" for k in keys)
            print(f"{'vs eager-fp64':16s} {hdr}")

            def show(label, out):
                cells = " ".join(f"{Q.metrics(out[k], ref64[k])['rel_l2']:9.2e}/{Q.metrics(out[k], ref64[k])['cos']:.6f}"
                                 for k in keys)
                print(f"{label:16s} {cells}", flush=True)
            show("eager-gpu-fp32", ref32)
            results = {"eager-gpu-fp32": ref32}
            for label in args.modes:
                dev = torch.device("cpu") if label == "eager-cpu" else cuda
                prop = Q.build(cls, args.so, "eager" if label == "eager-cpu" else label, dev, cfg)
                try:
                    out, _ = run_prop(prop, cls, args.so, cfg, an, torch.float32, dev)
                except Exception as exc:
                    print(f"{label:16s} ERROR {type(exc).__name__}: {exc}", flush=True)
                    continue
                results[label] = out
                show(label, out)
            if args.fd:
                # directional derivative along a smooth bump in vp (and eps / delta)
                nz, nx = cfg["shape"]
                z = np.linspace(0, 1, nz)[:, None]
                x = np.linspace(0, 1, nx)[None, :]
                bump = np.exp(-(((z - 0.6) / 0.15) ** 2 + ((x - 0.5) / 0.2) ** 2))
                for pname, scale in (("vp", 20.0), ("epsilon", 0.01), ("delta", 0.01)):
                    dm = {pname: (scale * bump).astype(np.float64)}
                    for h in (1.0, 0.25):
                        lp = run_prop(eager64(cls, args.so, cfg), cls, args.so, cfg, an, torch.float64, cuda,
                                      dm=dm, h=h, need_grad=False)
                        lm = run_prop(eager64(cls, args.so, cfg), cls, args.so, cfg, an, torch.float64, cuda,
                                      dm=dm, h=-h, need_grad=False)
                        fd = (lp - lm) / (2 * h)
                        msg = f"  FD d{pname} h={h}: fd={fd:.6e}"
                        other = "exact" if args.s_gradient != "exact" else "adjoint"
                        if h == 1.0:
                            ref_o, _ = run_prop(eager64(cls, args.so, cfg, other), cls, args.so, cfg, an,
                                                torch.float64, cuda)
                        g = float((ref_o[pname] * torch.tensor(dm[pname])).sum())
                        msg += f"  eager64-{other} {g:.6e} ({(g - fd) / abs(fd):+.1e})"
                        for label, out in [("eager64", ref64)] + list(results.items()):
                            g = float((out[pname] * torch.tensor(dm[pname])).sum())
                            msg += f"  {label} {g:.6e} ({(g - fd) / abs(fd):+.1e})"
                        print(msg, flush=True)


if __name__ == "__main__":
    main()
