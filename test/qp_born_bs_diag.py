#!/usr/bin/env python3
"""Why does Born boundary saving (fp32) miss full storage in 'frozen' mode?

matrix_born_v2.json: with s_gradient='frozen' and mp = mp_true / 2, the mp
gradient of c-bs-* sits 4-23 x the case's floor away from c-full, off the
source; 'adjoint' and mp = 0 pass.  In 'frozen' the scattered adjoint is
linearised about the SCATTERED state, whose S_n (homogeneous of degree 0)
boundary saving rebuilds by reverse stepping; where the scattered field is weak
the rebuild's round-off changes its direction.

Per case, for anisotropy (as matrix / eps = delta, where S_n == 0), s_gradient
(frozen / adjoint) and starting reflectivity (0, mp/2, 2 mp), the off-source
rel_l2 of the mp gradient of:

    bs        c-bs-gpu (fp32)          against c-full
    pert-mp   c-full, mp x (1 + 1e-7)  against c-full   (conditioning in mp)
    pert-bg   c-full, background x (1 + 1e-7) against c-full
    eager     eager-cpu against eager-gpu

    PYTHONPATH=src python test/qp_born_bs_diag.py
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
import qp_born_consistency as B  # noqa: E402
from sweep.propagator.options import BoundarySaving, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

CASES = ["tti/so8/canonical/eps>delta", "tti/so4/canonical/eps>delta", "vti/so4/canonical/eps<delta",
         "vti/so8/long/eps>delta"]


def background(names, shape, aniso, pert):
    ms = Q.model_arrays(names, shape, "eps>delta" if aniso == "eps=delta" else aniso)
    if aniso == "eps=delta":
        ms[names.index("delta")] = ms[names.index("epsilon")].copy()
    return Q.perturb(ms, seed=1) if pert else ms


def run(cls, so, cfg, aniso, sg, label, mp0, d_obs, pert_bg=False, pert_mp=False):
    dev = torch.device("cpu" if label == "eager-cpu" else "cuda")
    eq = cls(spatial_order=so, device=dev, backend="torch", s_gradient=sg)
    kw = dict(shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"], dt=cfg["dt"], dev=dev, nt=cfg["nt"], B=1)
    if label.startswith("eager"):
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=BoundarySaving(storage="gpu") if label == "c-bs" else Full(), **kw)
    names = eq.models
    bg = [torch.tensor(a, device=dev) for a in background(names[:-1], cfg["shape"], aniso, pert_bg)]
    mp_np = Q.perturb([mp0], seed=2)[0] if pert_mp else mp0
    mp = torch.tensor(mp_np, device=dev, requires_grad=d_obs is not None)
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=dev)
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    d = prop(wav, src, rec, models=bg + [mp])
    if d_obs is None:
        return d.detach()
    (0.5 * (d - d_obs.to(dev)).pow(2).sum()).backward()
    return mp.grad.detach().double().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=CASES)
    args = ap.parse_args()
    print(f"{'case':44s} {'start':8s} {'bs':>9s} {'pert-mp':>9s} {'pert-bg':>9s} {'eager':>9s}   (mp gradient, "
          f"rel_l2 off the source, r > {Q.R_SRC})", flush=True)
    for case in args.cases:
        e, so, cname, an0 = case.split("/")
        so = int(so[2:])
        cls, cfg = B.EQUATIONS[e], Q.CONFIGS[cname]
        mp_true = B.reflectivity(cfg["shape"])
        mask = Q.off_source_mask(cfg["shape"], Q.geometry(cfg["shape"], so, cfg["rec_stride"])[0])
        for an in (an0, "eps=delta"):
            for sg in ("frozen", "adjoint"):
                with torch.no_grad():
                    d_obs = run(cls, so, cfg, an, sg, "eager-gpu", mp_true, None)
                for start, mp0 in (("mp=0", np.zeros_like(mp_true)), ("mp=mp/2", 0.5 * mp_true),
                                   ("mp=2mp", 2.0 * mp_true)):
                    full = run(cls, so, cfg, an, sg, "c-full", mp0, d_obs)
                    bs = run(cls, so, cfg, an, sg, "c-bs", mp0, d_obs)
                    pmp = run(cls, so, cfg, an, sg, "c-full", mp0, d_obs, pert_mp=True)
                    pbg = run(cls, so, cfg, an, sg, "c-full", mp0, d_obs, pert_bg=True)
                    eg = run(cls, so, cfg, an, sg, "eager-gpu", mp0, d_obs)
                    ec = run(cls, so, cfg, an, sg, "eager-cpu", mp0, d_obs)

                    def off(a, b):
                        return Q.metrics(a[mask], b[mask])["rel_l2"]

                    print(f"{e}/so{so}/{cname}/{an}/{sg:8s} {start:8s} {off(bs, full):9.1e} {off(pmp, full):9.1e} "
                          f"{off(pbg, full):9.1e} {off(ec, eg):9.1e}", flush=True)


if __name__ == "__main__":
    main()
