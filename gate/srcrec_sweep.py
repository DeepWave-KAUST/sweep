"""Bit-exact gradient sweep over SOURCE x RECEIVER type combinations.

tier B of the bit gate covers equation x memory mode, but every one of its
configs uses each equation's DEFAULT source and receiver kinds. That leaves the
source/receiver dimension untested, and this repository has been bitten there
twice: a body-force source that dropped a rho gradient term (found only by
sweeping source kinds) and an elastic/DAS/anisotropic receiver adjoint defect
(PR #62).

Records the forward record AND every model gradient for each combination and
compares them with torch.equal, the same criterion the gates use.

    PYTHONPATH=<tree>/src python srcrec_sweep.py --save  base.pt
    PYTHONPATH=<tree>/src python srcrec_sweep.py --compare base.pt

Models are deliberately NON-UNIFORM: a constant model hides whole classes of
gradient defect because every spatial term collapses to the same value.
"""
from __future__ import annotations

import argparse
import itertools
import sys


# canonical small gradient-test grids for this repo
SHAPE_2D = (48, 56)
SHAPE_3D = (24, 20, 24)
NT_2D, NT_3D = 90, 60


def equations():
    import sweep.equations as E
    out = []
    for name, shape, nt in (
        ("Elastic", SHAPE_2D, NT_2D),
        ("Elastic3D", SHAPE_3D, NT_3D),
        ("DASMu", SHAPE_2D, NT_2D),
        ("DASMu3D", SHAPE_3D, NT_3D),
        ("ElasticTTISG", SHAPE_2D, NT_2D),
    ):
        cls = getattr(E, name, None)
        if cls is not None:
            out.append((name, cls, shape, nt))
    return out


def combos_for(eq):
    """Single-kind sweeps on each axis, plus cross-family mixes.

    One source kind at a time against the default receivers isolates the
    injection path and its gradient; one receiver kind at a time against the
    default sources isolates the recording path and its adjoint. The mixes then
    cover pairings neither pure sweep reaches (velocity source with a stress
    receiver, stress source with a strain receiver, and so on)."""
    srcs = [s.name for s in eq.available_source_fields()]
    recs = [s.name for s in eq.available_receiver_fields()]
    dsrc = list(eq.default_source_fields)
    drec = list(eq.default_receiver_fields)

    out = []
    for s in srcs:
        out.append(([s], drec))
    for r in recs:
        out.append((dsrc, [r]))

    def family(n):
        return "v" if n.startswith("v") else ("e" if n.startswith("e") else "s")

    fam_s = {}
    fam_r = {}
    for s in srcs:
        fam_s.setdefault(family(s), s)
    for r in recs:
        fam_r.setdefault(family(r), r)
    for fs, fr in itertools.product(sorted(fam_s), sorted(fam_r)):
        if fs == fr:
            continue                       # already covered by the pure sweeps
        out.append(([fam_s[fs]], [fam_r[fr]]))

    out.append((dsrc, drec))               # the default, i.e. what tier B runs
    if len(srcs) >= 2 and len(recs) >= 2:  # multi-component on both axes
        out.append((srcs[:2], recs[:2]))
    if len(recs) >= 3:                     # a mixed-family receiver LIST
        mixed = [recs[0]] + [r for r in recs if family(r) != family(recs[0])][:1]
        if len(mixed) == 2:
            out.append((dsrc, mixed))

    seen, uniq = set(), []
    for s, r in out:
        k = (tuple(s), tuple(r))
        if k not in seen:
            seen.add(k)
            uniq.append((list(s), list(r)))
    return uniq


def run_one(cls, shape, nt, src_t, rec_t, free_surface, colocate=False):
    import numpy as np
    import torch
    from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
    from sweep.propagator.torch import PropTorch

    dev = "cuda"
    ndim = len(shape)
    eq = cls(spatial_order=4, device=dev, backend="torch")
    co = CUDAOptions(memory=MemoryOptions(strategy="boundary",
                                          boundary=BoundaryOptions(storage="gpu")))
    prop = PropTorch(eq, backend="torch", impl="c", shape=shape, dev=dev,
                     dh=10.0, dt=6e-4, abcn=8, free_surface=free_surface,
                     nt=nt, B=1, source_type=list(src_t),
                     receiver_type=list(rec_t), cuda_options=co)

    nz, nx = shape[0], shape[-1]
    # non-uniform in EVERY axis: a z-only ramp leaves lateral derivative terms
    # multiplying zero, which hides exactly the defects this sweep is for.
    idx = np.indices(shape).astype(np.float32)
    bump = sum((i - s / 2) ** 2 / s for i, s in zip(idx, shape))
    vp = (2000.0 + 8.0 * idx[0] + 40.0 * np.sin(bump)).astype(np.float32)
    vs = (vp / 1.73 + 20.0 * np.cos(bump)).astype(np.float32)
    rho = (1800.0 + 4.0 * idx[-1] + 30.0 * np.sin(2 * bump)).astype(np.float32)
    models = [torch.tensor(vp, device=dev), torch.tensor(vs, device=dev),
              torch.tensor(rho, device=dev)]
    if "TTI" in cls.__name__:
        # 8 parameters: vp, vs, rho then Thomsen-like anisotropy and two tilt
        # angles. Filling these by cloning rho (order 1e3) is not a slightly
        # wrong model, it is an unstable one -- it produced NaN on BOTH trees.
        for base in (0.12, 0.06, 0.05, 0.30, 0.20):
            models.append(torch.tensor(
                (base * (1.0 + 0.15 * np.sin(bump))).astype(np.float32), device=dev))
    models = [m.clone().requires_grad_(True) for m in models]

    if ndim == 2:
        src = np.array([[nx // 2, nz // 3]], dtype=np.int64)
        rxx = np.arange(4, nx - 4, 5, dtype=np.int64)
        rec = np.stack([rxx, np.full(rxx.size, 4, dtype=np.int64)], -1)[None]
        if colocate:
            # a receiver ON the source cell: the source-side and receiver-side
            # rho corrections then land on the same cell, which is the geometry
            # a double-count fix had to get right.
            rec = np.concatenate([np.array([[[nx // 2, nz // 3]]],
                                           dtype=np.int64), rec], axis=1)
    else:
        ny = shape[1]
        # OFF the y midplane on purpose: with iy == ny//2 a transposed y/z index
        # in the source/receiver address arithmetic is invisible, because the
        # two coordinates are then interchangeable.
        sy = ny // 2 + 3
        src = np.array([[nx // 2, sy, nz // 3]], dtype=np.int64)
        rxx = np.arange(4, nx - 4, 4, dtype=np.int64)
        rec = np.stack([rxx, np.full(rxx.size, ny // 2 - 2, dtype=np.int64),
                        np.full(rxx.size, 4, dtype=np.int64)], -1)[None]
        if colocate:
            rec = np.concatenate([np.array([[[nx // 2, sy, nz // 3]]],
                                           dtype=np.int64), rec], axis=1)

    t = np.arange(nt, dtype=np.float32) * 6e-4 - 0.02
    wav = torch.tensor((1e3 * np.exp(-8000.0 * t * t)
                        * (1 - 16000.0 * t * t)).astype(np.float32), device=dev)

    syn = prop(wav, src, rec, models=models)
    syn.pow(2).mean().backward()
    out = {"record": syn.detach().cpu()}
    for i, m in enumerate(models):
        out[f"grad{i}"] = (m.grad.detach().cpu() if m.grad is not None
                           else torch.zeros(0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--free-surface", action="store_true")
    a = ap.parse_args()

    import torch
    import sweep
    print(f"# sweep={sweep.__file__}", flush=True)

    res, n = {}, 0
    for name, cls, shape, nt in equations():
        eq = cls(spatial_order=4, device="cpu", backend="torch")
        for src_t, rec_t in combos_for(eq):
            key = f"{name}|src={'+'.join(src_t)}|rec={'+'.join(rec_t)}"
            key += "|fs" if a.free_surface else "|nofs"
            try:
                for kk, vv in run_one(cls, shape, nt, src_t, rec_t,
                                      a.free_surface).items():
                    res[f"{key}|{kk}"] = vv
                n += 1
                print(f"  ok  {key}", flush=True)
            except Exception as exc:
                res[f"{key}|ERROR"] = f"{type(exc).__name__}: {exc}"[:160]
                print(f"  ERR {key}: {res[f'{key}|ERROR']}", flush=True)
    print(f"# {n} combinations ran", flush=True)

    if a.save:
        torch.save(res, a.save)
        print(f"# saved -> {a.save}")
    if a.compare:
        base = torch.load(a.compare, weights_only=False)
        npass = nfail = nerr = nnan = 0
        for k in sorted(base):
            if k.endswith("|ERROR"):
                same = res.get(k) == base[k]
                nerr += 1
                if not same:
                    print(f"ERROR-TEXT DIFFERS {k}\n  base={base[k]}\n  now ={res.get(k)}")
                continue
            if k not in res:
                nfail += 1
                print(f"MISSING {k}")
            elif torch.equal(base[k], res[k]):
                npass += 1
            elif (torch.isnan(base[k]).equal(torch.isnan(res[k]))
                  and torch.equal(base[k].nan_to_num(0.0), res[k].nan_to_num(0.0))):
                # identical including where both are NaN. torch.equal alone says
                # False for NaN==NaN, which would report a non-difference as a
                # failure; the NaN itself is a separate problem, counted as such.
                nnan += 1
            else:
                nfail += 1
                d = (base[k].double() - res[k].double()).abs().max().item()
                print(f"FAIL {k}  max|d|={d:.3e}")
        print(f"BITWISE PASS {npass}  FAIL {nfail}  "
              f"(+{nerr} error-text, +{nnan} identical-but-contain-NaN)")
        sys.exit(1 if nfail else 0)


if __name__ == "__main__":
    main()
