#!/usr/bin/env python3
"""s_gradient='adjoint': does a scale-relative floor on the ADJOINT field's S_n
shrink the mode's fp32 floor?

The adjoint recursion takes S_n from the adjoint field's own gradient, so it is
nonlinear in lambda and inherits the forward's sensitivity where lambda is weak.
Here the adjoint's S_n gets ``den + (kappa * max|grad lambda|)^4`` (the max per
step, detached); the forward is untouched.  Printed per kappa: eager fp32 (gpu),
eager fp32 (cpu) and eager fp64 against each other.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))

import qp_aniso_consistency as Q  # noqa: E402
import qp_aniso_diag as D  # noqa: E402
from sweep.equations import qP_tti, qP_vti  # noqa: E402

KAPPA = 0.0
_orig_vti, _orig_tti = qP_vti.AcousticVTI._adjoint_fns, qP_tti.AcousticTTI._adjoint_fns


def _floored(fns_of):
    def _adjoint_fns(self, dt, h, b):
        rhs_fn, s_fn, lap_fn = fns_of(self, dt, h, b)

        def s_floor(field, models):
            if KAPPA <= 0.0:
                return s_fn(field, models)
            gx = self.gradient(field, h, -1, kernels=self.grad_kernels)
            gz = self.gradient(field, h, -2, kernels=self.grad_kernels)
            g4 = (KAPPA * torch.sqrt(gx * gx + gz * gz).amax()) ** 4
            eps, delta = models[1], models[2]
            if len(models) == 4:
                c, s = torch.cos(models[3]), torch.sin(models[3])
                gx, gz = gx * c - gz * s, gx * s + gz * c
            num = -2 * (eps - delta) * gx**2 * gz**2
            den = (1 + 2 * eps) * gx**4 + gz**4 + 2 * (1 + delta) * gx**2 * gz**2
            return num / (den + g4 + 1e-26)
        return rhs_fn, s_floor, lap_fn
    return _adjoint_fns


qP_vti.AcousticVTI._adjoint_fns = _floored(_orig_vti)
qP_tti.AcousticTTI._adjoint_fns = _floored(_orig_tti)


def main():
    global KAPPA
    ap = argparse.ArgumentParser()
    ap.add_argument("--eq", nargs="+", default=["vti", "tti"])
    ap.add_argument("--config", default="canonical")
    ap.add_argument("--kappas", nargs="+", type=float, default=[0.0, 1e-4, 1e-3, 1e-2])
    args = ap.parse_args()
    cfg = Q.CONFIGS[args.config]
    Q.S_GRADIENT = "adjoint"
    cuda, cpu = torch.device("cuda"), torch.device("cpu")
    for e in args.eq:
        cls, so, an = Q.EQUATIONS[e], 4, "eps>delta"
        for kappa in args.kappas:
            KAPPA = kappa
            r64, _ = D.run_prop(D.eager64(cls, so, cfg, "adjoint"), cls, so, cfg, an, torch.float64, cuda)
            g32, _ = D.run_prop(Q.build(cls, so, "eager", cuda, cfg), cls, so, cfg, an, torch.float32, cuda)
            c32, _ = D.run_prop(Q.build(cls, so, "eager", cpu, cfg), cls, so, cfg, an, torch.float32, cpu)
            keys = [k for k in r64 if k != "record"]
            line = " | ".join(f"{k}: gpu-64 {Q.metrics(g32[k], r64[k])['rel_l2']:.1e} cpu-gpu "
                              f"{Q.metrics(c32[k], g32[k])['rel_l2']:.1e}" for k in keys)
            print(f"{e} kappa={kappa:g}: {line}", flush=True)


if __name__ == "__main__":
    main()
