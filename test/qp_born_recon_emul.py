#!/usr/bin/env python3
"""Emulate boundary saving's rebuild of the Born scattered field from the
full-mode history of impl='c' (AcousticVTIBorn).

Full mode keeps, per step, the scattered state of step it + 1 and the
background update r0 of step it (u_allt, qp_born_forward).  From the last two
scattered fields this rebuilds sc^{n-1} = 2 sc^n - sc^{n+1} + r1(sc^n) + mp r0^n
on the deep interior (qp_born_reverse's cells) with the frame taken from the
history -- boundary saving's recursion, but with the EXACT r0, so it isolates
the scattered field's own reverse round-off.  Printed per step: the
reconstruction's rel_l2 against the history, and that of its S_n field (what
s_gradient='frozen' linearises about).  Step 0 checks that the emulated
forward residual is at fp32 round-off (the stencil matches the core).

    PYTHONPATH=src python test/qp_born_recon_emul.py
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
from sweep.equations import AcousticVTIBorn  # noqa: E402
from sweep.propagator.options import Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402


def wrapper_node(t):
    seen, stack = set(), [t.grad_fn]
    while stack:
        n = stack.pop()
        if n is None or n in seen:
            continue
        seen.add(n)
        if type(n).__name__.startswith("WrapperBackward"):
            return n
        stack.extend(f for f, _ in n.next_functions)
    raise RuntimeError("no WrapperBackward node")


def history(so, cfg, aniso, mp0):
    dev = torch.device("cuda")
    eq = AcousticVTIBorn(spatial_order=so, device=dev, backend="torch", s_gradient="frozen")
    prop = PropTorch(eq, impl="c", memory=Full(), shape=cfg["shape"], abcn=cfg["abcn"], dh=cfg["dh"],
                     dt=cfg["dt"], dev=dev, nt=cfg["nt"], B=1)
    bg = Q.model_arrays(eq.models[:-1], cfg["shape"], aniso)
    mp = torch.tensor(mp0, device=dev, requires_grad=True)
    wav = torch.tensor(Q.ricker(cfg["nt"], cfg["dt"], cfg["freq"], cfg["delay"]), device=dev)
    src, rec = Q.geometry(cfg["shape"], so, cfg["rec_stride"])
    d = prop(wav, src, rec, models=[torch.tensor(a, device=dev) for a in bg] + [mp])
    saved = wrapper_node(d).saved_tensors
    u_allt, lap, grad = saved[0], saved[6], saved[7]
    return u_allt[:, :, 0].float(), lap.float(), grad.float(), bg


def stencil(u, lc, gc, M, dh):
    """Lx, Lz, Dux, Duz on the whole grid (np.roll; only the interior is used)."""
    Lx = -lc[0] * u
    Lz = -lc[0] * u
    Dx = torch.zeros_like(u)
    Dz = torch.zeros_like(u)
    for k in range(1, M + 1):
        xp, xm = torch.roll(u, -k, 1), torch.roll(u, k, 1)
        zp, zm = torch.roll(u, -k, 0), torch.roll(u, k, 0)
        Lx = Lx + lc[k] * (xp + xm)
        Lz = Lz + lc[k] * (zp + zm)
        Dx = Dx + gc[k] * (xp - xm)
        Dz = Dz + gc[k] * (zp - zm)
    return Lx / (dh * dh), Lz / (dh * dh), Dx / dh, Dz / dh


def s_op(X, Z, eps, dl):
    X2, Z2 = X * X, Z * Z
    num = -2.0 * (eps - dl) * X2 * Z2
    den = (1.0 + 2.0 * eps) * (X2 * X2) + Z2 * Z2 + 2.0 * (1.0 + dl) * X2 * Z2
    return num / (den + 1e-26)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--config", default="canonical")
    ap.add_argument("--aniso", default="eps<delta")
    args = ap.parse_args()
    cfg, so, M = Q.CONFIGS[args.config], args.so, args.so // 2
    mp_true = B.reflectivity(cfg["shape"])
    for start, mp0 in (("mp/2", 0.5 * mp_true), ("2mp", 2.0 * mp_true)):
        H, lc, gc, bg = history(so, cfg, args.aniso, mp0)
        nt, _, Pz, Px = H.shape
        nz, nx = cfg["shape"]
        oz, ox = (Pz - nz) // 2, (Px - nx) // 2
        pad = lambda a: torch.tensor(np.pad(a, ((oz, Pz - nz - oz), (ox, Px - nx - ox)), mode="edge"),
                                     device=H.device)
        vp, eps, dl = (pad(a) for a in bg)
        mpp = pad(mp0.astype(np.float32))
        W = (vp * vp) * (cfg["dt"] * cfg["dt"])
        frame = cfg["abcn"] + M + 1
        lo_z, hi_z, lo_x, hi_x = M + frame, Pz - M - frame, M + frame, Px - M - frame
        inner = torch.zeros(Pz, Px, dtype=torch.bool, device=H.device)
        inner[lo_z:hi_z, lo_x:hi_x] = True

        sc = lambda n: H[n - 1, 0] if n >= 1 else torch.zeros_like(H[0, 0])   # sc^n
        r0 = lambda n: H[n, 5]                                                  # r0^n

        def r1(u):
            Lx, Lz, Dx, Dz = stencil(u, lc, gc, M, cfg["dh"])
            S = s_op(Dx, Dz, eps, dl)
            return W * ((1.0 + 2.0 * eps + S) * Lx + (1.0 + S) * Lz), S

        res = []
        for n in range(1, nt - 1):          # forward residual of the emulated stencil
            pred = 2 * sc(n) - sc(n - 1) + r1(sc(n))[0] + mpp * r0(n)
            res.append(float((pred - sc(n + 1))[inner].norm() / max(float(sc(n + 1)[inner].norm()), 1e-30)))
        print(f"\n== vti/so{so}/{args.config}/{args.aniso} mp={start}: emulated forward residual "
              f"median {np.median(res):.1e} max {max(res):.1e}", flush=True)

        nxt, now = sc(nt), sc(nt - 1)
        marks = sorted({nt - 2, (3 * nt) // 4, nt // 2, nt // 4, nt // 8, 8, 2})
        for n in range(nt - 1, 0, -1):
            r, _ = r1(now)
            prev = torch.where(inner, 2 * now - nxt + r + mpp * r0(n), sc(n - 1))
            nxt, now = now, prev
            if n - 1 in marks:
                ref = sc(n - 1)
                e = float((now - ref)[inner].norm() / max(float(ref[inner].norm()), 1e-30))
                S_rec = r1(now)[1]
                S_ref = r1(ref)[1]
                eS = float((S_rec - S_ref)[inner].norm() / max(float(S_ref[inner].norm()), 1e-30))
                big = float(((S_rec - S_ref).abs() > 0.1 * S_ref.abs().max())[inner].float().mean())
                print(f"  step {n - 1:4d}: sc rel {e:.1e}   S_n rel {eS:.1e}   cells with |dS| > 0.1 max|S|: "
                      f"{100 * big:.1f}%", flush=True)


if __name__ == "__main__":
    main()
