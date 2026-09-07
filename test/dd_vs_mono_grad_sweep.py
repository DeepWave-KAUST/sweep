"""Does domain decomposition give the same gradient as one card, for every
equation DD admits, across process-grid shapes?

Nothing in the tree answered that. ``gate/ddgate.py`` has py/px axes but its
stored baseline is twelve ``1x1`` entries -- one rank, no cut, no halo exchange
-- so every "GATE dd1 12/12" in this repository's history verified a
single-domain run wearing a ModelParallel wrapper. ``dd_pad_grad_check.py`` and
``dd_corner_check.py`` DO compare DD against a single-domain reference and DO
take --py/--px, but both hard-code Acoustic3D.

This is those two generalised over the admissible set. It is deliberately at the
PUBLIC level -- build ModelParallel, call it, call .backward() -- rather than the
protocol level of dd_nccl_backward_check.py, because the question is what a user
gets, not whether the exchange protocol is self-consistent.

KNOWN, EXPLAINED, NOT A DEFECT -- AcousticVRZ3D is the one equation here that
is not bit-exact against one card, by construction. Its DD backward must use the
SPLIT gradient (materialise c/e, exchange, then take the divergence) because the
fused nested-stencil kernel's accessor zeroes cut-side taps; the single-GPU path
meanwhile picks AUTO, which is the FUSED kernel on Ada. Two arithmetic
decompositions of the same expression, so the gradients differ by ~1 ULP per
cell (rel 3.3e-09 vp / 1.3e-08 z here, deterministic and reproducible to every
digit across processes). Run with SWEEP_VRZ_GRAD_SPLIT=1 -- which forces the
single-GPU path onto the same split kernels -- and it is bit-exact.

Launch (one process per tile):
    torchrun --standalone --nproc-per-node=4 test/dd_vs_mono_grad_sweep.py \
        --equation Elastic3D --py 2 --px 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "test")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sweep.equations import (Acoustic, Acoustic3D, AcousticVRZ3D,  # noqa: E402
                             Elastic, Elastic3D)
from sweep.parallel import MeshTopology, pad_to_mesh                # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel              # noqa: E402
from sweep.propagator.torch import PropTorch                        # noqa: E402

DH, DT = 10.0, 6e-4

# name -> (class, ndim, source_type, receiver_type, model base values)
EQUATIONS = {
    "Acoustic":      (Acoustic,      2, ["h1"], ["h1"], {"vp": 1800.0}),
    "Acoustic3D":    (Acoustic3D,    3, ["h1"], ["h1"], {"vp": 1800.0}),
    "AcousticVRZ3D": (AcousticVRZ3D, 3, ["h1"], ["h1"], {"vp": 1800.0, "z": 2000.0}),
    "Elastic":       (Elastic,       2, None,   None,   {"vp": 2600.0, "vs": 1500.0, "rho": 2200.0}),
    "Elastic3D":     (Elastic3D,     3, None,   None,   {"vp": 2600.0, "vs": 1500.0, "rho": 2200.0}),
}


def _model(base, shape, ndim):
    """A heterogeneous model -- a homogeneous one can hide an adjoint defect."""
    z = np.arange(shape[0], dtype=np.float32).reshape((-1,) + (1,) * (ndim - 1))
    m = base + 9.0 * np.maximum(z - 6.0, 0.0)
    m = np.broadcast_to(m, shape).astype(np.float32).copy()
    m[: shape[0] // 8] = base * 0.86
    lo = shape[0] // 2
    m[lo:lo + max(2, shape[0] // 12)] += base * 0.12
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equation", required=True, choices=sorted(EQUATIONS))
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--px", type=int, default=1)
    ap.add_argument("--shot-groups", type=int, default=1,
                    dest="shot_groups", help="shot-parallel groups; world = "
                    "shot_groups*py*px. One shot per group. EVERY test and "
                    "example in this tree passes shot_groups=1, so the "
                    "cross-group gradient sum has no single-card reference "
                    "anywhere.")
    ap.add_argument("--shots", type=int, default=0,
                    help="total distinct shots (default: one per shot group). "
                    "Must be a multiple of shot_groups; group g runs shots "
                    "g, g+G, g+2G, ... SEQUENTIALLY through one ModelParallel, "
                    "which is what an FWI epoch does and what nothing in the "
                    "tree compares against one card.")
    ap.add_argument("--same-source", action="store_true", dest="same_source",
                    help="put every shot at the SAME position -- a control that "
                    "separates 'more shots' from 'shots in different places'.")
    ap.add_argument("--fs", type=int, default=0)
    ap.add_argument("--nt", type=int, default=1000)
    ap.add_argument("--abcn", type=int, default=12)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--dump", default="", help="save this rank's ref+dd grads to "
                    "PATH.pt; two such runs in two PROCESSES measure the "
                    "run-to-run floor, which same-process repeats cannot")
    ap.add_argument("--n", type=int, default=0, help="per-axis physical size; 0 = per-ndim default")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    tile = args.py * args.px
    assert tile * args.shot_groups == world, (
        f"shot_groups*py*px = {args.shot_groups}*{args.py}*{args.px} must equal "
        f"world size {world}")

    cls, ndim, st, rt, bases = EQUATIONS[args.equation]
    n = args.n or (140 if ndim == 2 else 56)
    # deliberately NOT divisible by the mesh, so pad_to_mesh is exercised
    shape = (n, n + 3) if ndim == 2 else (n, n + 3, n + 5)
    mesh = MeshTopology(py=args.py, px=args.px, shot_groups=args.shot_groups,
                        world_size=world, rank=rank)

    models_np = [_model(bases[s.name], shape, ndim) for s in cls.MODEL_SPECS]
    probe = pad_to_mesh(torch.zeros(shape), mesh)
    padded_shape = tuple(probe.shape)

    t = np.arange(args.nt, dtype=np.float32) * DT - 0.035
    a = np.pi * 12.0 * t
    wav = torch.as_tensor(((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32),
                          device=dev)

    # One shot per shot group by default, at DIFFERENT positions -- identical
    # shots would make the cross-group sum indistinguishable from "one shot
    # times G".
    nz, G = shape[0], args.shot_groups
    nshot = args.shots or G
    assert nshot % G == 0, f"--shots {nshot} must be a multiple of shot_groups {G}"
    S = nshot
    if ndim == 2:
        nx = shape[1]
        srcs = [np.array([[[nx // 4 + (0 if args.same_source
                                    else g * (nx // (2 * S + 2))), nz // 4]]],
                         dtype=np.int64) for g in range(S)]
        rec = np.array([[[ix, 3] for ix in range(3, nx - 3, 4)]], dtype=np.int64)
    else:
        ny, nx = shape[1], shape[2]
        srcs = [np.array([[[nx // 4 + (0 if args.same_source
                                    else g * (nx // (2 * S + 2))),
                            ny // 3, nz // 4]]],
                         dtype=np.int64) for g in range(S)]
        rec = np.array([[[ix, iy, 3]
                         for iy in range(3, ny - 3, 5)
                         for ix in range(3, nx - 3, 5)]], dtype=np.int64)

    def build():
        eq = cls(spatial_order=args.so, device=dev, backend="torch")
        kw = {}
        if st is not None:
            kw.update(source_type=st, receiver_type=rt)
        return PropTorch(eq, backend="torch", impl="c", shape=padded_shape,
                         dh=DH, dt=DT, nt=args.nt, abcn=args.abcn, dev=dev,
                         free_surface=bool(args.fs), **kw)

    if rank == 0:
        print(f"{args.equation}  physical={shape}  padded={padded_shape}  "
              f"mesh=py{args.py}xpx{args.px}xsg{args.shot_groups}  nshot={S}  "
              f"world={world}  fs={args.fs}  "
              f"nt={args.nt}  models={[s.name for s in cls.MODEL_SPECS]}", flush=True)

    # ---- single domain: every shot, gradient accumulated over shots --------
    ref_leaves = [torch.tensor(m, device=dev, requires_grad=True) for m in models_np]
    mono = build()
    r_ref = None
    my_shots = list(range(mesh.shot_group, S, G))     # this group's shots
    for g, sg_src in enumerate(srcs):
        r = mono(wav, sg_src, rec, models=[pad_to_mesh(x, mesh) for x in ref_leaves])
        (r.double() ** 2).sum().backward()
        if g == my_shots[0]:
            # clone: a later shot through the same propagator may reuse the
            # record buffer, and the comparison must not read shot G-1's data.
            r_ref = r.detach().clone()

    # ---- domain decomposed, same physical leaves ---------------------------
    dd_leaves = [torch.tensor(m, device=dev, requires_grad=True) for m in models_np]
    ddp = ModelParallel(build(), mesh)
    r_dd = None
    for g in my_shots:                 # sequential, ONE ModelParallel -- an epoch
        r = ddp(wav, srcs[g], rec,
                models=[pad_to_mesh(x, mesh) for x in dd_leaves])
        (r.double() ** 2).sum().backward()
        if r_dd is None:
            r_dd = r.detach().clone()
    # _run_adjoint has ALREADY summed each tile's gradient across the shot
    # groups (its shot_pg all_reduce), so every rank now holds the shot-summed
    # gradient of ITS OWN tile and zeros elsewhere. Assembling the global
    # gradient therefore reduces over the TILE ranks of one shot group
    # (model_pg) -- a world-wide all_reduce would count each tile once per shot
    # group and inflate the gradient by exactly G.
    pg = getattr(getattr(ddp, "mesh", None), "model_pg", None)
    if world > 1:
        for leaf in dd_leaves:
            dist.all_reduce(leaf.grad, group=pg)

    # ---- compare -----------------------------------------------------------
    lines, bad = [], 0
    for spec, gref, gdd in zip(cls.MODEL_SPECS, ref_leaves, dd_leaves):
        a_, b_ = gref.grad, gdd.grad
        shape_ok = tuple(a_.shape) == shape and tuple(b_.shape) == shape
        bit = bool(torch.equal(a_, b_))
        scale = float(a_.abs().max())
        d = float((a_ - b_).abs().max())
        rel = d / max(scale, 1e-30)
        # A gradient at the fp32 floor makes every ratio below meaningless --
        # say so instead of reporting a green that means nothing.
        # 1e-20 would let a dead gradient through; require it to be alive
        # relative to the record that produced it.
        alive = scale > 1e-12
        lines.append(f"    grad[{spec.name:4s}] bit={bit!s:5s} rel={rel:.3e} "
                     f"|ref|max={scale:.3e}{'' if alive else '   <-- AT THE FP32 FLOOR, ratio meaningless'}")
        bad += (not bit) or (not shape_ok) or (not alive)

    # Locate the receiver axis by SIZE rather than by guessing an orientation:
    # the elastic record carries a receiver-FIELD axis too (vx, vz), and a
    # transpose heuristic indexes that one instead -- which is what made the
    # first version of this script die with "index 2 out of bounds for axis 0
    # with size 2" on Elastic and never reach the comparison.
    # The two paths hand back DIFFERENT record layouts -- PropTorch returns
    # (B, nt, nrec, nfield), ModelParallel returns the raw CUDA layout, which is
    # (B, nrec, nt) for the acoustic family and (nfield, B, nrec, nt) for the
    # staggered one.  Same elements, so (r**2).sum() -- and therefore the
    # gradient -- is layout-independent; only this comparison has to care.
    # Canonicalise both to (nfield, B, nrec, nt) rather than guess an
    # orientation: guessing is what made the first version index the elastic
    # record's FIELD axis and die with "index 2 out of bounds for axis 0".
    nrec_total, nt = rec.shape[1], args.nt

    def canon(a, tag):
        a = a.detach().cpu().numpy()
        if a.ndim == 4 and a.shape[0] == 1 and a.shape[1] == nt:
            return np.transpose(a, (3, 0, 2, 1))          # PropTorch (B,nt,R,F)
        if a.ndim == 3:
            return a[None]                                 # DD acoustic (B,R,nt)
        if a.ndim == 4 and a.shape[-1] == nt:
            return a                                       # DD staggered (F,B,R,nt)
        raise AssertionError(f"unrecognised {tag} record layout {a.shape}")

    R4, D4 = canon(r_ref, "ref"), canon(r_dd, "dd")
    own = getattr(ddp, "_own_rec_idx", None)
    own = (np.arange(nrec_total) if not own
           else np.asarray(own, dtype=np.int64).ravel())
    # Self-evidence that the run really decomposed: a green from a mesh that
    # silently collapsed to one tile would prove nothing. Every rank prints its
    # tile origin/extent and how many of the global receivers it owns.
    print(f"    [rank {rank}] sg={mesh.shot_group} tile=(yi{mesh.yi},xi{mesh.xi}) "
          f"x0={getattr(ddp, 'x0', 0)} nxp={getattr(ddp, 'nxp', '-')} "
          f"y0={getattr(ddp, 'y0', 0)} nyp={getattr(ddp, 'nyp', '-')} "
          f"owns {len(own)}/{nrec_total} receivers  src={srcs[mesh.shot_group].ravel().tolist()}",
          flush=True)

    if D4.shape[2] != len(own) or R4.shape[2] != nrec_total:
        rbit, rrel = False, float("nan")
        rnote = (f"   <-- RECEIVER AXIS MISMATCH ref{R4.shape} dd{D4.shape} "
                 f"nrec={nrec_total} own={len(own)}")
    else:
        rsel = R4[:, :, own, :]
        rbit = bool(np.array_equal(D4, rsel))
        rrel = float(np.abs(D4 - rsel).max()) / max(float(np.abs(rsel).max()), 1e-30)
        rnote = "" if rbit else f"  rel={rrel:.3e}"
    bad += not rbit
    # An absolute amplitude threshold is the wrong gate: the elastic record is
    # a VELOCITY driven by a stress source into a 5.7e6 Rayl medium, so it is
    # legitimately ~1e-5 where the acoustic PRESSURE record is ~1e2. What has to
    # be excluded is the trap from the VTI probe -- a record that is zero, or
    # that carries only the source's own footprint because the wave never
    # crossed the spread. So test it scale-free: energy must have reached the
    # LATE part of the window and the FAR half of the spread.
    rec_peak = float(np.abs(R4).max())
    late = float(np.abs(R4[:, :, :, int(0.6 * nt):]).max())
    far = float(np.abs(R4[:, :, nrec_total // 2:, :]).max())
    live = rec_peak > 0.0 and late > 1e-3 * rec_peak and far > 1e-3 * rec_peak
    if not live:
        bad += 1
        rnote += (f"   <-- DEAD ACQUISITION peak={rec_peak:.3e} "
                  f"late={late / max(rec_peak, 1e-30):.1e} "
                  f"far={far / max(rec_peak, 1e-30):.1e} of peak: "
                  f"the wave never crossed the spread, this proves nothing")
    if rank == 0:
        print(f"    record     bit={rbit!s:5s} |rec|max={rec_peak:.3e} "
              f"late={late / max(rec_peak, 1e-30):.2f} "
              f"far={far / max(rec_peak, 1e-30):.2f}{rnote}", flush=True)
        for l in lines:
            print(l, flush=True)
        if (bad and args.equation == "AcousticVRZ3D"
                and os.environ.get("SWEEP_VRZ_GRAD_SPLIT", "0") == "0"):
            print("    NOTE: VRZ3D single-GPU picks the FUSED gradient kernel and "
                  "DD must use the SPLIT one\n"
                  "          (the fused accessor zeroes cut-side taps). Re-run with "
                  "SWEEP_VRZ_GRAD_SPLIT=1\n"
                  "          to put both on the same kernel; that is bit-exact.",
                  flush=True)
        print(f"    -> {'PASS' if bad == 0 else 'FAIL'}", flush=True)

    if args.dump:
        torch.save({"ref": [g.grad.detach().cpu() for g in ref_leaves],
                    "dd": [g.grad.detach().cpu() for g in dd_leaves],
                    "rec": r_ref.detach().cpu(),
                    "names": [sp.name for sp in cls.MODEL_SPECS]},
                   f"{args.dump}.r{rank}.pt")

    fail = torch.tensor([bad], device=dev)
    dist.all_reduce(fail)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if int(fail.item()) > 0 else 0)


if __name__ == "__main__":
    main()
