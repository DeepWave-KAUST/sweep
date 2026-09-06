"""Targeted source/receiver cells, chosen from a code-path map rather than swept.

The 116-combination sweep in srcrec_sweep.py is broad but blind. A read of the
injection and adjoint paths says the cross product collapses hard -- with cpmls
boundaries the SOURCE side has exactly two classes per equation (BODY FORCE,
which alone carries a rho gradient term, and STRESS, which gets no adjoint
correction at all because the tape stores velocities only), and the RECEIVER
side has three (velocity: raw residual + rho correction; stress: NEGATED
residual, no rho; strain: raw residual, no rho, and the kernel supplies its own
minus). What the blind sweep therefore misses is not more single kinds but the
handful of cells where two paths meet:

  1. Two body-force components at ONE cell. The rho correction is launched once
     per velocity source field, each accumulating into the same grad slot with
     atomicAdd. Nothing anywhere runs two components together.
  2. A receiver ON the source cell, so the source-side and receiver-side rho
     corrections land on the same cell.
  3. Receiver lists that CROSS a sign boundary. Receiver gradients are linearly
     decomposable, so a mixed list adds information only against indexing and
     sign-classification defects -- which makes velocity+stress+strain the only
     mixes worth running, and pairs like [sxx,szz] worthless.
  4. A y-DISPLACED 3-D source. The broad sweep puts the source at the y
     midplane, where a transposed y/z index is invisible.
  5. Per-edge free surface (Elastic 2-D on impl='c' only), where sxz and szz are
     annihilated by different predicates on different faces.

Deliberately NOT run: any (strain source, *) pair. exx/ezz/exz declare
supports_source=True but are write-only accumulators nothing reads, so the model
gradient is structurally zero and the cell proves nothing.
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "/home/wangs0j/sweep-local")
from srcrec_sweep import SHAPE_2D, SHAPE_3D, NT_2D, NT_3D, run_one   # noqa: E402


# (label, equation, shape, nt, sources, receivers, free_surface, rec_on_source)
CELLS = [
    ("el2d-two-body-force", "Elastic", SHAPE_2D, NT_2D,
     ["vx", "vz"], ["vz", "sxx"], False, False),
    ("el2d-two-bf-colocated", "Elastic", SHAPE_2D, NT_2D,
     ["vx", "vz"], ["vz", "sxx"], False, True),
    ("el3d-three-body-force", "Elastic3D", SHAPE_3D, NT_3D,
     ["vx", "vy", "vz"], ["vz", "szz"], False, True),
    ("el2d-sxz-fs", "Elastic", SHAPE_2D, NT_2D,
     ["sxz"], ["sxx", "szz"], True, False),
    ("el2d-vz-fs-colocated", "Elastic", SHAPE_2D, NT_2D,
     ["vz"], ["vz", "szz"], True, True),
    ("el3d-sxy-fs", "Elastic3D", SHAPE_3D, NT_3D,
     ["sxy"], ["vz", "szz"], True, False),
    ("dasmu-three-family-rec", "DASMu", SHAPE_2D, NT_2D,
     ["vz"], ["vz", "szz", "exx"], False, True),
    ("dasmu-shear-pair", "DASMu", SHAPE_2D, NT_2D,
     ["vx", "vz"], ["exz", "sxz"], False, False),
    ("dasmu-mixed-fs", "DASMu", SHAPE_2D, NT_2D,
     ["sxx", "szz"], ["szz", "exx", "exz"], True, False),
    ("dasmu3d-three-family", "DASMu3D", SHAPE_3D, NT_3D,
     ["vz"], ["vz", "sxz", "ezz"], False, True),
    ("dasmu3d-ydisp-bf", "DASMu3D", SHAPE_3D, NT_3D,
     ["vy"], ["exx", "eyy", "ezz"], False, False),
    ("el3d-ydisp-bf", "Elastic3D", SHAPE_3D, NT_3D,
     ["vy"], ["vx", "vy", "vz"], False, False),
    ("tti2d-vy-sh", "ElasticTTISG", SHAPE_2D, NT_2D,
     ["vy"], ["vy", "sxx"], False, False),
    ("el2d-rec-permutation-a", "Elastic", SHAPE_2D, NT_2D,
     ["vz"], ["vz", "sxx"], False, False),
    ("el2d-rec-permutation-b", "Elastic", SHAPE_2D, NT_2D,
     ["vz"], ["sxx", "vz"], False, False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    a = ap.parse_args()

    import torch
    import sweep
    import sweep.equations as E
    import srcrec_sweep as S
    print(f"# sweep={sweep.__file__}", flush=True)

    orig_run = S.run_one

    res = {}
    for (label, eqname, shape, nt, src_t, rec_t, fs, colocate) in CELLS:
        cls = getattr(E, eqname, None)
        if cls is None:
            continue
        key = f"{label}|{eqname}|src={'+'.join(src_t)}|rec={'+'.join(rec_t)}"
        key += "|fs" if fs else "|nofs"
        key += "|colocated" if colocate else ""
        try:
            out = orig_run(cls, shape, nt, src_t, rec_t, fs,
                           colocate=colocate) if _accepts_colocate(orig_run) \
                else orig_run(cls, shape, nt, src_t, rec_t, fs)
            for kk, vv in out.items():
                res[f"{key}|{kk}"] = vv
            print(f"  ok  {key}", flush=True)
        except Exception as exc:
            res[f"{key}|ERROR"] = f"{type(exc).__name__}: {exc}"[:160]
            print(f"  ERR {key}: {res[f'{key}|ERROR']}", flush=True)

    if a.save:
        torch.save(res, a.save)
        print(f"# saved -> {a.save}")
    if a.compare:
        base = torch.load(a.compare, weights_only=False)
        npass = nfail = nerr = 0
        for k in sorted(base):
            if k.endswith("|ERROR"):
                nerr += 1
                if res.get(k) != base[k]:
                    print(f"ERROR-TEXT DIFFERS {k}")
                continue
            if k not in res:
                nfail += 1
                print(f"MISSING {k}")
            elif torch.equal(base[k], res[k]):
                npass += 1
            else:
                nfail += 1
                d = (base[k].double() - res[k].double()).abs().max().item()
                print(f"FAIL {k}  max|d|={d:.3e}")
        print(f"BITWISE PASS {npass}  FAIL {nfail}  (+{nerr} error-text only)")
        sys.exit(1 if nfail else 0)


def _accepts_colocate(fn):
    import inspect
    return "colocate" in inspect.signature(fn).parameters


if __name__ == "__main__":
    main()
