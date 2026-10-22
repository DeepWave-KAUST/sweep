#!/usr/bin/env python3
"""Does a scale-relative S_n regularisation make the exact qP gradient well conditioned?

Replaces the absolute ``den + 1e-26`` of qP_vti.rhs_cpml by
``den + (kappa * max|grad p|)^4`` (the max detached, per step) and measures, for
s_gradient='exact':
  * eager fp32 vs eager fp64 gradients (the conditioning),
  * the record change against the 1e-26 scheme (how much the physics moves),
  * the FD check of the fp64 gradient.
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
from sweep.equations import qP_vti  # noqa: E402

_ORIG = qP_vti.rhs_cpml
KAPPA = None


def rhs_relative(u_now, psix, psiz, zetax, zetaz, vp, epsilon, delta, dt, h, b, pml,
                 lap_x, lap_z, grad_op, grad_kernels=None, s_frozen=False):
    az, bz, dbzdz, ax, bx, dbxdx = pml
    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)
    dpdz = (1 + bz) * dudz + az * psiz
    dpdx = (1 + bx) * dudx + ax * psix
    tmpz = ((1 + bz) * lap_z + dbzdz * dudz) + grad_op(az * psiz, h, -2, kernels=grad_kernels)
    tmpx = ((1 + bx) * lap_x + dbxdx * dudx) + grad_op(ax * psix, h, -1, kernels=grad_kernels)
    lap_z_pml = (1 + bz) * tmpz + az * zetaz
    lap_x_pml = (1 + bx) * tmpx + ax * zetax
    px, pz = (dpdx.detach(), dpdz.detach()) if s_frozen else (dpdx, dpdz)
    numerator = -2 * (epsilon - delta) * px**2 * pz**2
    denominator = (1 + 2 * epsilon) * px**4 + pz**4 + 2 * (1 + delta) * px**2 * pz**2
    gmax = torch.sqrt(dpdx.detach()**2 + dpdz.detach()**2).amax()
    sk = numerator / (denominator + (KAPPA * gmax) ** 4 + 1e-26)
    vp2dt2 = vp**2 * dt**2
    rhs = vp2dt2 * (((1 + 2 * epsilon) + sk) * lap_x_pml + (1 + sk) * lap_z_pml)
    psixn = bx * dudx + ax * psix
    psizn = bz * dudz + az * psiz
    zetaxn = bx * tmpx + ax * zetax
    zetazn = bz * tmpz + az * zetaz
    return rhs, (psixn, psizn, zetaxn, zetazn)


def main():
    global KAPPA
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="canonical")
    ap.add_argument("--kappas", nargs="+", type=float, default=[1e-6, 1e-5, 1e-4, 1e-3])
    args = ap.parse_args()
    cfg = Q.CONFIGS[args.config]
    cls, so, an = Q.EQUATIONS["vti"], 4, "eps>delta"
    cuda = torch.device("cuda")
    Q.S_GRADIENT = "exact"
    qP_vti.rhs_cpml = _ORIG
    base64, _ = D.run_prop(D.eager64(cls, so, cfg, "exact"), cls, so, cfg, an, torch.float64, cuda)
    for kappa in [None] + args.kappas:
        qP_vti.rhs_cpml = _ORIG if kappa is None else rhs_relative
        KAPPA = kappa
        r64, _ = D.run_prop(D.eager64(cls, so, cfg, "exact"), cls, so, cfg, an, torch.float64, cuda)
        r32, _ = D.run_prop(Q.build(cls, so, "eager", cuda, cfg), cls, so, cfg, an, torch.float32, cuda)
        rec = Q.metrics(r64["record"], base64["record"])["rel_l2"]
        cells = " ".join(f"{k} {Q.metrics(r32[k], r64[k])['rel_l2']:.2e}" for k in ("wavelet", "vp", "epsilon", "delta"))
        msg = f"kappa={kappa}: record vs 1e-26 scheme {rec:.2e} | fp32 vs fp64: {cells}"
        nz, nx = cfg["shape"]
        z = np.linspace(0, 1, nz)[:, None]
        x = np.linspace(0, 1, nx)[None, :]
        bump = np.exp(-(((z - 0.6) / 0.15) ** 2 + ((x - 0.5) / 0.2) ** 2))
        dm = {"vp": (20.0 * bump)}
        for h in (1.0, 0.25):
            lp = D.run_prop(D.eager64(cls, so, cfg, "exact"), cls, so, cfg, an, torch.float64, cuda,
                            dm=dm, h=h, need_grad=False)
            lm = D.run_prop(D.eager64(cls, so, cfg, "exact"), cls, so, cfg, an, torch.float64, cuda,
                            dm=dm, h=-h, need_grad=False)
            fd = (lp - lm) / (2 * h)
            g = float((r64["vp"] * torch.tensor(dm["vp"])).sum())
            msg += f" | FD dvp h={h}: {(g - fd) / abs(fd):+.1e}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
