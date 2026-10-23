#!/usr/bin/env python3
"""Which impl='c' behaviour makes its CPML return more late energy than eager?

qp_pml_check.py: after the direct wave has left an 80 x 80 box, impl='c' sends
back ~2x (qP) and ~6x (Acoustic) the energy eager does.  The CPML expressions
are the same; two places where the kernels do something eager does not:

  * halo: the c kernels never update the outer M cells (they stay 0); eager's
    zero-padded stencils update them like any other cell;
  * fast path (Acoustic only): acoustic2nd takes the bare-Laplacian branch for
    every cell outside the x AND z pad bands (pad + halo from the edge), so the
    M cells next to a band lose the dbdx term (np.gradient spreads it one cell
    out of the band) and the d(a psi) taps that reach into the band.  qp2d's
    in_cx / in_cz extend the band by M and keep them.

Eager is run as is, with the halo zeroed every step, and (Acoustic) with the
fast path as well; each is compared with impl='c' on receivers covering the
box.  The variant that matches impl='c' to round-off is what impl='c' does.

    PYTHONPATH=src python test/c_pml_emul.py --so 8
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
from sweep.equations import Acoustic, AcousticVTI  # noqa: E402
from sweep.propagator.options import EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

N, BIG, ABCN, NT, DT, DH, FM = 80, 400, 20, 700, 1e-3, 10.0, 15.0


def variant(base, halo0, fast):
    """``base`` with impl='c''s untouched halo and / or acoustic2nd's fast path."""

    class V(base):
        def func(self, wavefields, models, dt, h, b, **kw):
            out = list(base.func(self, wavefields, models, dt, h, b, **kw))
            M = self.so // 2
            nz, nx = out[0].shape[-2:]
            if fast:
                pad = ABCN
                iz = torch.arange(nz, device=out[0].device)[:, None]
                ix = torch.arange(nx, device=out[0].device)[None, :]
                in_x = (ix < pad + M) | (ix >= nx - (pad + M))
                in_z = (iz < pad + M) | (iz >= nz - (pad + M))
                interior = ~(in_x | in_z)
                hz, hx = self._spacings_2d(h)
                lz, lx = self.separable_d2_2d(wavefields[0], self.laplace_kernels, hz, hx)
                vp = models[0]
                bare = 2 * wavefields[0] - wavefields[1] + vp ** 2 * dt ** 2 * (lx + lz)
                out[0] = torch.where(interior, bare, out[0])
            if halo0:
                keep = torch.zeros((nz, nx), dtype=torch.bool, device=out[0].device)
                keep[M:nz - M, M:nx - M] = True
                for k in (0, 2, 3, 4, 5):
                    out[k] = torch.where(keep, out[k], torch.zeros_like(out[k]))
            return tuple(out)

    V.__name__ = base.__name__
    return V


def run(cls, so, impl, n):
    extra = {} if issubclass(cls, Acoustic) else {"s_gradient": "frozen"}
    eq = cls(spatial_order=so, device="cuda", backend="torch", **extra)
    kw = dict(shape=(n, n), abcn=ABCN, dh=DH, dt=DT, dev="cuda", nt=NT, B=1)
    if impl == "eager":
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=Full(), **kw)
    vals = {"vp": 2500.0, "epsilon": 0.0, "delta": 0.0}
    ms = [torch.full((n, n), vals[k], device="cuda") for k in eq.models]
    off = (n - N) // 2
    src = np.array([[n // 2, n // 2]], np.int32)
    zz, xx = np.meshgrid(np.arange(N) + off, np.arange(N) + off, indexing="ij")
    rec = np.stack([xx.ravel(), zz.ravel()], -1).astype(np.int32)[None]
    wav = torch.tensor(Q.ricker(NT, DT, FM, 0.08), device="cuda")
    with torch.no_grad():
        d = prop(wav, src, rec, models=ms)
    return d.double().cpu().numpy()[0].reshape(NT, N, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=int, default=8)
    args = ap.parse_args()
    so = args.so
    wins = [(0, 220), (220, 370), (370, NT)]
    print(f"so {so}, {N}x{N}, abcn {ABCN}, nt {NT}, windows {wins}", flush=True)
    for base in (Acoustic, AcousticVTI):
        ref = run(base, so, "eager", BIG)
        c = run(base, so, "c", N)
        variants = {"eager": base, "eager+halo0": variant(base, True, False)}
        if base is Acoustic:
            variants["eager+fast"] = variant(base, False, True)
            variants["eager+halo0+fast"] = variant(base, True, True)
        tot = float(np.sqrt((ref ** 2).sum()))
        print(f"\n== {base.__name__} (iso): c boundary return / total per window: " + "  ".join(
            f"{float(np.sqrt(((c - ref)[a:b] ** 2).sum())) / tot:.2e}" for a, b in wins), flush=True)
        for name, cls in variants.items():
            e = run(cls, so, "eager", N)
            cmp = []
            for a, b in wins:
                num = float(np.sqrt(((c - e)[a:b] ** 2).sum()))
                den = float(np.sqrt((e[a:b] ** 2).sum()))
                cmp.append(f"{num / den:.1e}")
            ret = "  ".join(f"{float(np.sqrt(((e - ref)[a:b] ** 2).sum())) / tot:.2e}" for a, b in wins)
            print(f"  {name:18s} c vs it, per window: {'  '.join(cmp)}   | its boundary return: {ret}", flush=True)


if __name__ == "__main__":
    main()
