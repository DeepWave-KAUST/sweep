#!/usr/bin/env python3
"""impl='c' against an eager twin of its forward, for the second-order acoustic
family (Acoustic, Acoustic3D, AcousticLSRTM(3D), AcousticVRZ(3D), ViscoAcoustic).

The c kernels never update the outer M (halo) cells; eager's zero-padded
stencils do.  ``eager+halo0`` (eager with the halo zeroed after every step)
is the eager twin of a c forward that evaluates the full CPML expression
everywhere -- which is what the acoustic2d fast-path fix makes c do
(c_pml_emul.py).  If the c record matches the twin, the c gradients must match
autograd through the twin: that is the check that the c backward is the exact
transpose of the c forward (gradient consistency), for every model and the
wavelet, in every memory mode.

Also, where impl='c' returns the wavelet gradient, the dot test of the linear
map w -> d: |<w, g_w> - <d, r>| / |<d, r>| for L = <d, r>; and for the Born
equations the same for mp (d is affine in mp).

Geometry: a source near a corner and receivers along all four edges, a random
residual r, so the waves and the adjoint work the absorbing layers hard.

    PYTHONPATH=src python test/c_consistency_suite.py --so 4 --eqs Acoustic AcousticVRZ
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
from sweep import equations as E  # noqa: E402
from sweep.propagator.options import BoundarySaving, Ckpt, EagerOptions, Full  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from c_twins import (halo0, product_rule_acoustic3d, product_rule_lsrtm2d, product_rule_lsrtm3d,  # noqa: E402
                     product_rule_vrz2d, product_rule_vrz3d)

DT, DH, FM, ABCN = 1e-3, 10.0, 15.0, 12
SHAPES = {2: (60, 80), 3: (28, 32, 36)}
NTS = {2: 500, 3: 260}
MODES = {"c-full": lambda: Full(), "c-bs": lambda: BoundarySaving(storage="gpu"),
         "c-ckpt": lambda: Ckpt(mode="chunk", chunks=10),
         "c-rckpt": lambda: Ckpt(mode="recursive", count=7)}


def models_for(names, shape):
    idx = np.indices(shape).astype(np.float32)
    s = [i / max(n - 1, 1) for i, n in zip(idx, shape)]
    vp = 2200.0 + 700.0 * s[0] + 200.0 * s[-1] + 150.0 * np.sin(6.0 * s[-1] + 3.0 * s[0])
    rng = np.random.default_rng(7)
    # mp zero on the outer ring: that ring is edge-padded into the absorbing
    # layer and its gradient cropped (EdgePadding.backward), so a nonzero ring
    # would move the layer where no gradient looks and break the mp dot test
    ring = np.zeros(shape, bool)
    ring[tuple(slice(1, n - 1) for n in shape)] = True
    vals = {"vp": vp, "z": vp * (1800.0 + 300.0 * s[0]), "mp": 0.05 * rng.standard_normal(shape) * ring,
            "Q": 60.0 + 40.0 * s[0], "omega": np.full(shape, 2 * np.pi * FM)}
    return [np.ascontiguousarray(vals[n], dtype=np.float32) for n in names]


def geometry(shape, so):
    nd = len(shape)
    M = so // 2
    if nd == 2:
        nz, nx = shape
        src = np.array([[M + 4, M + 4]], np.int64)
        pts = [(x, M + 1) for x in range(M + 1, nx - M - 1, 3)] + [(x, nz - M - 2) for x in range(M + 1, nx - M - 1, 3)]
        pts += [(M + 1, z) for z in range(M + 1, nz - M - 1, 3)] + [(nx - M - 2, z) for z in range(M + 1, nz - M - 1, 3)]
    else:
        nz, ny, nx = shape
        src = np.array([[M + 4, M + 4, M + 4]], np.int64)
        pts = [(x, y, M + 1) for x in range(M + 1, nx - M - 1, 4) for y in (M + 1, ny // 2, ny - M - 2)]
        pts += [(M + 1, y, z) for y in range(M + 1, ny - M - 1, 4) for z in (nz // 2, nz - M - 2)]
        pts += [(nx - M - 2, y, z) for y in range(M + 1, ny - M - 1, 4) for z in (nz // 2,)]
    return src, np.array(pts, np.int64)[None]


def run(cls, so, label, grads):
    nd = 3 if cls.__name__.endswith("3D") else 2
    shape, nt = SHAPES[nd], NTS[nd]
    eq = (twin(cls) if label == "eager+halo0" else cls)(spatial_order=so, device="cuda", backend="torch")
    kw = dict(shape=shape, abcn=ABCN, dh=DH, dt=DT, dev="cuda", nt=nt, B=1)
    if label.startswith("eager"):
        prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False), memory=Full(), **kw)
    else:
        prop = PropTorch(eq, impl="c", memory=MODES[label](), **kw)
    names = eq.models
    ms = [torch.tensor(a, device="cuda", requires_grad=n in grads) for n, a in zip(names, models_for(names, shape))]
    wav = torch.tensor(Q.ricker(nt, DT, FM, 0.06), device="cuda", requires_grad="wavelet" in grads)
    src, rec = geometry(shape, so)
    d = prop(wav, src, rec, models=ms)
    r = torch.tensor(np.random.default_rng(11).standard_normal(tuple(d.shape)).astype(np.float32), device="cuda")
    L = (d.double() * r.double()).sum()
    L.backward()
    out = {"record": d.detach().double().cpu().numpy(), "L": float(L)}
    if "wavelet" in grads:
        out["wavelet"] = wav.grad.double().cpu().numpy()
        out["dot_w"] = abs(float((wav.detach().double() * wav.grad.double()).sum()) - float(L)) / abs(float(L))
    for n, m in zip(names, ms):
        if m.requires_grad:
            out[n] = m.grad.double().cpu().numpy()
    if "mp" in grads:   # d is affine in mp: <mp, g> = L(mp) - L(0)
        with torch.no_grad():
            ms0 = [torch.zeros_like(m) if n == "mp" else m.detach() for n, m in zip(names, ms)]
            L0 = float((prop(wav.detach(), src, rec, models=ms0).double() * r.double()).sum())
        mp = ms[names.index("mp")]
        out["dot_mp"] = abs(float((mp.detach().double() * mp.grad.double()).sum()) - (float(L) - L0)) / abs(float(L) - L0)
    return out


# The eager twin of each c forward: the halo kept at 0, and the CPML in the
# kernel's own form (c_twins.py).
TWINS = {"Acoustic3D": product_rule_acoustic3d, "AcousticVRZ": product_rule_vrz2d,
         "AcousticVRZ3D": product_rule_vrz3d, "AcousticLSRTM": product_rule_lsrtm2d,
         "AcousticLSRTM3D": product_rule_lsrtm3d}


def twin(cls):
    return halo0(TWINS.get(cls.__name__, lambda c: c)(cls))


# "<class>+vp": the same class with vp requiring grad too (the LSRTM RWI path);
# "<class>+w": with the wavelet requiring grad (d is linear in it: dot_w applies)
GRADS = {"Acoustic": ["wavelet", "vp"], "Acoustic3D": ["wavelet", "vp"],
         "AcousticLSRTM": ["mp"], "AcousticLSRTM+vp": ["mp", "vp"], "AcousticLSRTM+w": ["mp", "wavelet"],
         "AcousticLSRTM3D": ["mp"], "AcousticLSRTM3D+vp": ["mp", "vp"], "AcousticLSRTM3D+w": ["mp", "wavelet"],
         "AcousticVRZ": ["vp", "z"], "AcousticVRZ3D": ["vp", "z"],
         "AcousticVRZ+w": ["vp", "z", "wavelet"], "AcousticVRZ3D+w": ["vp", "z", "wavelet"],
         "ViscoAcoustic": ["wavelet", "vp"]}


def rel(a, b):
    nb = np.linalg.norm(b)
    return np.linalg.norm(a - b) / nb if nb > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--eqs", nargs="+", default=list(GRADS))
    args = ap.parse_args()
    print(f"so {args.so}; vs eager+halo0: rel_l2 of record and gradients; dot = |<x,g_x> - dL|/|dL|", flush=True)
    for name in args.eqs:
        cls = getattr(E, name.split("+")[0])
        grads = GRADS[name]
        ref = run(cls, args.so, "eager+halo0", grads)
        keys = [k for k in ref if k not in ("L", "dot_w", "dot_mp")]
        dots = "  ".join(f"{k} {ref[k]:.1e}" for k in ("dot_w", "dot_mp") if k in ref)
        print(f"\n== {name}   (eager+halo0 dot: {dots or '-'})", flush=True)
        for label in MODES:
            try:
                out = run(cls, args.so, label, grads)
            except Exception as exc:   # refused modes / grads: report, keep going
                print(f"  {label:7s} ERROR {type(exc).__name__}: {str(exc)[:140]}", flush=True)
                continue
            cells = "  ".join(f"{k} {rel(out[k], ref[k]):.1e}" for k in keys)
            dots = "  ".join(f"{k} {out[k]:.1e}" for k in ("dot_w", "dot_mp") if k in out)
            print(f"  {label:7s} {cells}   | {dots}", flush=True)


if __name__ == "__main__":
    main()
