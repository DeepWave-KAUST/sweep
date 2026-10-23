#!/usr/bin/env python3
"""Finite-difference check of the Acoustic vp gradient, eager against impl='c'.

acoustic2nd (the impl='c' forward) takes its bare-Laplacian fast path from
pad + halo, so the M cells next to an absorbing band lose the dbdx and
d(a psi) terms; the fused adjoint takes its fast path only from pad + 2 halo,
i.e. it transposes the full expression there.  If so, the c gradient is not
the adjoint of the c forward near the bands.

For J(m) = 0.5 |d(m) - d_obs|^2 and a direction dm, compare <g, dm> with the
central difference (J(m + e dm) - J(m - e dm)) / (2 e) over several e, for
  edge      dm on columns / rows 1 .. M-1 next to every band (inside the cells
            the fast path changes; the outer ring is left out, see setup),
  interior  a Gaussian in the middle,
  smooth    a smooth random field over the whole box.
eager is the floor (fp32 + truncation).  d_obs is each implementation's own
record of a fixed perturbed model, so it is a smooth function of m for both.

    PYTHONPATH=src python test/acoustic_fd_gradcheck.py --so 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "test"))

import qp_aniso_consistency as Q  # noqa: E402
from sweep.equations import Acoustic  # noqa: E402
from sweep.propagator.options import BoundarySaving, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

NZ, NX, ABCN, NT, DT, DH, FM = 80, 120, 20, 800, 1e-3, 10.0, 15.0
EPS = (4.0, 2.0, 1.0, 0.5)      # m/s, dm has max |dm| = 1


def prop(so, label):
    eq = Acoustic(spatial_order=so, device="cuda", backend="torch")
    kw = dict(shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, dev="cuda", nt=NT, B=2)
    if label == "eager":
        return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    return PropTorch(eq, impl="c", memory=BoundarySaving(storage="gpu") if label == "c-bs" else Full(), **kw)


def setup(so):
    z = np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, NX, dtype=np.float32)[None, :]
    vp = (2200.0 + 800.0 * z + 100.0 * x).astype(np.float32)
    true = vp + 150.0 * np.exp(-(((z - 0.6) / 0.1) ** 2 + ((x - 0.5) / 0.15) ** 2)).astype(np.float32)
    M = so // 2
    src = np.array([[NX // 2, M + 1], [M + 3, M + 1]], np.int32)
    rx = np.arange(M, NX - M, 2, dtype=np.int32)
    rec = np.repeat(np.stack([rx, np.full_like(rx, M)], -1)[None], 2, 0)
    wav = torch.tensor(Q.ricker(NT, DT, FM, 0.08), device="cuda")
    # The outermost row / column is edge-padded into the absorbing layer and its
    # gradient is cropped (EdgePadding.backward), so a perturbation there moves
    # the layer too and no gradient sees it: perturb rows / columns 1 .. M-1
    # only -- still inside the fast-path band of acoustic2nd -- and keep every
    # direction zero on the outer ring.
    ring = np.ones((NZ, NX), np.float32)
    ring[0] = ring[-1] = 0.0
    ring[:, 0] = ring[:, -1] = 0.0
    edge = np.zeros((NZ, NX), np.float32)
    edge[:, 1:M] = 1.0
    edge[:, NX - M:NX - 1] = 1.0
    edge[NZ - M:NZ - 1, :] = 1.0
    edge[1:M, :] = 1.0
    edge *= ring
    interior = np.exp(-(((z - 0.5) / 0.08) ** 2 + ((x - 0.5) / 0.08) ** 2)).astype(np.float32)
    rng = np.random.default_rng(3)
    smooth = gaussian_filter(rng.standard_normal((NZ, NX)), 4).astype(np.float32)
    smooth = smooth * ring
    smooth /= np.abs(smooth).max()
    return vp, true, src, rec, wav, {"edge": edge, "interior": interior, "smooth": smooth}


def J(p, wav, src, rec, m, d_obs):
    with torch.no_grad():
        d = p(wav, src, rec, models=[torch.tensor(m, device="cuda")])
    return 0.5 * float(((d.double() - d_obs.double()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=int, default=4)
    args = ap.parse_args()
    vp, true, src, rec, wav, dirs = setup(args.so)
    print(f"Acoustic so {args.so}, {NZ}x{NX}, abcn {ABCN}, nt {NT}; rel err = |<g,dm> - FD| / |FD|", flush=True)
    for label in ("eager", "c-full", "c-bs"):
        p = prop(args.so, label)
        pobs = prop(args.so, "eager" if label == "eager" else "c-full")
        with torch.no_grad():
            d_obs = pobs(wav, src, rec, models=[torch.tensor(true, device="cuda")])
        m = torch.tensor(vp, device="cuda", requires_grad=True)
        d = p(wav, src, rec, models=[m])
        (0.5 * (d - d_obs).pow(2).sum()).backward()
        g = m.grad.double().cpu().numpy()
        for name, dm in dirs.items():
            gd = float((g * dm).sum())
            errs = []
            for e in EPS:
                fd = (J(p, wav, src, rec, vp + e * dm, d_obs) - J(p, wav, src, rec, vp - e * dm, d_obs)) / (2 * e)
                errs.append(abs(gd - fd) / abs(fd))
            print(f"  {label:7s} {name:9s} <g,dm> {gd:+.6e}   rel err per eps {EPS}: "
                  + " ".join(f"{v:.1e}" for v in errs) + f"   (min {min(errs):.1e})", flush=True)


if __name__ == "__main__":
    main()
