#!/usr/bin/env python3
"""Coverage of the qP impl='c' paths the main matrix does not reach.

* several shots in one call (B > 1, per-shot sources),
* dz != dx (an axis-order slip is invisible with a scalar dh),
* spatial orders 2, 6 and 10 (10 runs the runtime-order instantiation),
* the Born pair with B > 1.

Each row: c (full / bs-gpu-fp32) against eager, s_gradient='frozen' (the
linear adjoint, so the comparison is to rounding), records and gradients.
"""
from __future__ import annotations

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
from sweep.equations import AcousticTTI, AcousticTTIBorn, AcousticVTI, AcousticVTIBorn  # noqa: E402
from sweep.propagator.options import BoundarySaving, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

SHAPE, ABCN, NT, DT = (44, 60), 18, 220, 0.001


def geometry(nshot):
    nz, nx = SHAPE
    xs = np.linspace(12, nx - 13, nshot).astype(np.int32)
    src = np.stack([xs, np.full_like(xs, 8 + np.arange(nshot))], -1).astype(np.int32)
    rx = np.arange(3, nx - 3, 3, dtype=np.int32)
    rec = np.stack([rx, np.full_like(rx, 3)], -1)[None].repeat(nshot, 0)
    return src, rec


def run(cls, so, dh, impl, mem, nshot, born=False):
    eq = cls(spatial_order=so, device="cuda", backend="torch", s_gradient="frozen")
    kw = dict(shape=SHAPE, abcn=ABCN, dh=dh, dt=DT, dev="cuda", nt=NT, B=nshot)
    if impl == "eager":
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=mem, **kw)
    names = eq.models
    arrs = Q.model_arrays([n for n in names if n != "mp"], SHAPE, "eps>delta")
    if born:
        mp = np.zeros(SHAPE, np.float32)
        mp[28] = 0.2
        arrs.append(mp)
    ms = [torch.tensor(a, device="cuda", requires_grad=(n == "mp") if born else True) for n, a in zip(names, arrs)]
    wav = torch.tensor(Q.ricker(NT, DT, 15.0, 0.05), device="cuda", requires_grad=not born)
    src, rec = geometry(nshot)
    d = prop(wav, src, rec, models=ms)
    d.pow(2).sum().backward()
    out = {"record": d.detach().double().cpu()}
    if not born:
        out["wavelet"] = wav.grad.detach().double().cpu()
    for n, m in zip(names, ms):
        if m.requires_grad:
            out[n] = m.grad.detach().double().cpu()
    return out


def main():
    cases = []
    for cls in (AcousticVTI, AcousticTTI):
        cases += [(cls, 4, 10.0, 3, False), (cls, 4, (8.0, 12.0), 1, False), (cls, 4, (12.0, 8.0), 2, False)]
        cases += [(cls, so, 10.0, 1, False) for so in (2, 6, 10)]
    cases += [(AcousticVTIBorn, 4, 10.0, 2, True), (AcousticTTIBorn, 4, (8.0, 12.0), 2, True)]
    worst = 0.0
    for cls, so, dh, nshot, born in cases:
        ref = run(cls, so, dh, "eager", None, nshot, born)
        for label, mem in (("c-full", Full()), ("c-bs-gpu-fp32", BoundarySaving(storage="gpu"))):
            try:
                out = run(cls, so, dh, "c", mem, nshot, born)
            except Exception as exc:
                print(f"{cls.__name__:18s} so{so:<2d} dh={dh} B={nshot} {label:14s} ERROR {type(exc).__name__}: {str(exc)[:200]}",
                      flush=True)
                worst = float("inf")
                continue
            ms = {k: Q.metrics(out[k], ref[k]) for k in ref}
            worst = max(worst, max(m["rel_l2"] for m in ms.values()))
            cells = " ".join(f"{k} {m['rel_l2']:.1e}" for k, m in ms.items())
            print(f"{cls.__name__:18s} so{so:<2d} dh={str(dh):12s} B={nshot} {label:14s} {cells}", flush=True)
    print(f"worst rel_l2 {worst:.2e}")


if __name__ == "__main__":
    main()
