"""NCCL multi-GPU elastic DD BACKWARD (gradients) vs single-domain reference.

torchrun --standalone --nproc-per-node=<P> test/dd_nccl_elastic_backward_check.py \
    [--ndim 2] [--free-surface]

Each rank owns one x-tile and runs the full elastic DD pipeline (the
protocol proven bitwise on one GPU in test_dd_elastic_backward_two_tile.py,
here split across ranks with NCCL halo exchanges replacing the manual
``_swap``):

  forward    boundary saving on; E1 half-step protocol — phase 1 (velocity)
             then exchange velocity M-wide, phase 2 (stress + save tail)
             then exchange stress M-wide.
  backward   phased descending per-step segments with cut_face_mask; after
             phase 1 exchange adjoint-velocity + recon-stress, after phase 2
             exchange adjoint-stress + recon-velocity (each M wide).

Rank 0 also runs the single-domain reference (monolithic backward_bs replay
with Python-bound grads_out) and compares the assembled grad_vp / grad_vs /
grad_rho over each tile's owned region.  Grading: PASS (bitwise) /
PASS_TOL (rel <= 1e-5) / FAIL.

Geometry is the fixed correctness geometry from the single-process test
(NX must be divisible by the rank count); the bench script covers scaling.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT), str(REPO_ROOT / "test")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sweep.parallel import MeshTopology, ModelParallelMesh  # noqa: E402
from sweep.parallel.fast_halo import FastHaloSet  # noqa: E402

from test_dd_elastic_backward_two_tile import (  # noqa: E402
    ABCN, DT, M, NPHYS, NV, PAD, REC_GX2, REC_GX3, REC_Y3, REC_Z2, REC_Z3,
    SRC3, SRC_Z2, WAVELET_SCALE, X_LO_BIT, X_HI_BIT,
    NX2, NX3, NY3, NZ2, NZ3, NT2, NT3,
    TileState, build_reference_2d, build_reference_3d, capture_both,
    global_models, make_prop, ricker,
)

REL_TOL = 1e-5
import argparse  # noqa: E402


# Harness defaults for the type axes; every case sets BOTH dims explicitly so
# no case inherits the previous one's binding.
_DEFAULT_TYPES = {2: (["sxx", "szz"], ["vx", "vz"]),
                  3: (["sxx", "syy", "szz"], ["vx", "vy", "vz"])}


def run_case(case, rank, world, dev):
    """One isolated DD-vs-monolithic comparison.

    Isolation contract: everything a case touches is constructed here and
    dies here -- tiles, runners, the reference, and the FastHaloSets (their
    data_ptr-keyed exchanger caches are instance-level, so a fresh instance
    per case is what makes allocator address reuse harmless).  The only
    state shared between cases is the NCCL process group, which is
    stateless between collectives.  The matrix driver additionally re-runs
    the first case at the end and asserts bitwise-identical gradients: any
    cross-case leakage fails that check.
    """
    import test_dd_elastic_backward_two_tile as H
    ndim, fs = case.ndim, case.free_surface
    if not hasattr(run_case, "_hdefaults"):
        run_case._hdefaults = (list(H.REC_GX2), list(H.REC_GX3), tuple(H.SRC3))
    H.REC_GX2 = list(run_case._hdefaults[0])
    H.REC_GX3 = list(run_case._hdefaults[1])
    H.SRC3 = tuple(run_case._hdefaults[2])
    for d in (2, 3):
        H.SRC_TYPE[d] = list(_DEFAULT_TYPES[d][0])
        H.REC_TYPE[d] = list(_DEFAULT_TYPES[d][1])
    if case.source_type:
        H.SRC_TYPE[ndim] = list(case.source_type)
    if case.receiver_type:
        H.REC_TYPE[ndim] = list(case.receiver_type)
    label = f"src={H.SRC_TYPE[ndim]} rec={H.REC_TYPE[ndim]}"

    if ndim == 2:
        NZ, NX, NT = NZ2, NX2, NT2
        rec_gx = list(REC_GX2)
        src_g = (NX // 2 + case.src_dx, SRC_Z2)
        shape_full = (NZ, NX)
    else:
        NZ, NX, NT = NZ3, NX3, NT3
        rec_gx = list(REC_GX3)
        src_g = (SRC3[0] + case.src_dx,) + tuple(SRC3[1:])
        shape_full = (NZ, NY3, NX)
    assert NX % world == 0, f"NX={NX} not divisible by world={world}"
    if case.rec_on_cut:
        cuts = [i * (NX // world) for i in range(1, world)]
        rec_gx = sorted(set(rec_gx) | set(cuts))
    label += f" src_x={src_g[0]} cuts={[i * (NX // world) for i in range(1, world)]}"
    # The reference must solve the SAME problem as the tiles: hand it this
    # case's receiver list (rec-on-cut extends it) and, in 3-D, the shifted
    # source (build_reference_3d reads H.SRC3; the 2-D builder takes src_x).
    if ndim == 2:
        H.REC_GX2 = list(rec_gx)
    else:
        H.REC_GX3 = list(rec_gx)
        H.SRC3 = tuple(src_g)
    nxp = NX // world

    topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, world)) if world > 1 else None

    # ---------------- reference forward record -> shared residual ----------
    if rank == 0:
        # A failure here (e.g. the library rejecting a receiver type) must
        # not strand rank 1 inside the broadcast: ship the error as the
        # payload so every rank raises the same thing.
        try:
            if ndim == 2:
                ref = build_reference_2d(src_g[0], fs)
            else:
                ref = build_reference_3d(fs)
            residual_raw = ref.fwd_record_raw.clone()
            shape_obj = [list(residual_raw.shape)]
        except Exception as e:
            shape_obj = [("__ERROR__", type(e).__name__, str(e)[:300])]
    else:
        shape_obj = [None]
    dist.broadcast_object_list(shape_obj, src=0)
    if isinstance(shape_obj[0], tuple) and shape_obj[0][0] == "__ERROR__":
        raise RuntimeError(f"{shape_obj[0][1]}: {shape_obj[0][2]}")
    if rank != 0:
        residual_raw = torch.zeros(shape_obj[0], device=dev)
    dist.broadcast(residual_raw, src=0)

    # ---------------- per-rank tile setup ----------------
    models_np = global_models(ndim)
    # edge-pad to runtime extent; z-top pad = M under free surface (no top PML)
    full_pad = ABCN + M
    top_pad = M if fs else full_pad

    def grp(m):
        if ndim == 2:
            pw = ((top_pad, full_pad), (full_pad, full_pad))
        else:
            pw = ((top_pad, full_pad), (full_pad, full_pad), (full_pad, full_pad))
        return np.pad(m, pw, mode="edge")

    models_padded = [torch.tensor(grp(m), device=dev) for m in models_np]
    x0 = rank * nxp
    if ndim == 2:
        tile_models = [m[:, x0:x0 + nxp].copy() for m in models_np]
    else:
        tile_models = [m[:, :, x0:x0 + nxp].copy() for m in models_np]

    sx = src_g[0]
    owns_src = x0 <= sx < x0 + nxp
    if owns_src:
        loc = [sx - x0] + list(src_g[1:])
        t_src = np.array([[loc]], dtype=np.int32)
        t_wav = ricker(NT, DT, scale=WAVELET_SCALE)
    else:
        dummy = [1, 1] if ndim == 2 else [1, 1, 1]
        t_src = np.array([[dummy]], dtype=np.int32)
        t_wav = np.zeros(NT, dtype=np.float32)

    own_rec = [gx for gx in rec_gx if x0 <= gx < x0 + nxp]
    own_idx = [rec_gx.index(gx) for gx in own_rec]
    if ndim == 2:
        t_rec = np.array([[[gx - x0, REC_Z2] for gx in own_rec]], dtype=np.int32)
    else:
        t_rec = np.array(
            [[[gx - x0, REC_Y3, REC_Z3] for gx in own_rec]], dtype=np.int32
        )

    shape_tile = (NZ, nxp) if ndim == 2 else (NZ, NY3, nxp)
    prop = make_prop(ndim, shape_tile, fs, topo=topo)
    cap = capture_both(prop)
    models_t = [torch.tensor(a, device=dev, requires_grad=True)
                for a in tile_models]
    syn = prop(t_wav, t_src, t_rec, models=models_t)
    rec = syn[0] if isinstance(syn, (tuple, list)) else syn
    rec.sum().backward()
    tile = TileState(cap, ndim)
    tile.cut_face_mask = ((X_LO_BIT if rank > 0 else 0)
                          | (X_HI_BIT if rank < world - 1 else 0))
    tile.x_off = x0

    # global edge-padded material in the pad columns (true neighbour values);
    # runtime models carry leading (B, channel) dims so reshape the bare
    # spatial slice (numel matches) before copy.
    #
    # P-series cut-aware pad: this tile's physical region no longer starts at
    # the symmetric PAD — it starts at (padding[0] + M) in the tile buffer (a
    # cut face has 0 PML, only the M halo).  ``models_padded`` is the GLOBAL
    # model symmetric-padded by PAD per side, so global phys index g lives at
    # column PAD+g.  Align the source so the tile's physical cell (lo = the
    # padding[0]+M offset) maps to global phys x0 (= PAD+x0).  Source start is
    # therefore (PAD + x0) - (padding[0] + M); copy the tile buffer's REAL
    # x-width read off the buffer (.size(-1)), not nxp+2*PAD.  (Old
    # ``mp[x0 : x0 + width]`` assumed a symmetric pad and mis-mapped every
    # tile; e.g. rank0 fed global cols [0:w] instead of the correct edge-
    # aligned window, and the cut tile pulled the wrong neighbour material.)
    src_start = (PAD + x0) - (prop._backend_impl.padding[0] + M)
    for mt in (tile.fp.models, tile.bp.models):
        for m, mp in zip(mt, models_padded):
            sl = mp[..., src_start:src_start + m.size(-1)]
            m.copy_(sl.reshape(m.shape))
    # elastic2d AND elastic3d forward/recon in_pml are cut-aware — set the
    # forward cut_face_mask for BOTH dims (was 3D-only; the 2D omission left the
    # recon forward on the wrong PML branch -> per-tile gradient drift).
    tile.fp.cut_face_mask = tile.cut_face_mask

    # this tile's receivers' traces of the shared synthetic residual
    adj = torch.zeros_like(tile.bp.adjoint_source)
    for j, gi in enumerate(own_idx):
        adj[:, :, j] = residual_raw[:, :, gi]
    tile.bp.adjoint_source = adj

    # Physical-region crop in THIS tile's runtime buffers.  Under the P-series
    # cut-aware compact pad a cut (neighbour-facing) face carries 0 PML width,
    # only the stencil halo M, so the interior no longer starts at the
    # symmetric PAD.  Read the real per-rank x_lo pad off the prop: the
    # physical region starts at padding[0] + M (was the wrong ``lo = PAD``,
    # which mis-cropped every cut tile — e.g. rank1 whose x_lo pad is 0).
    # The halo view [lo-M:hi+M] and the owned-grad crop [lo:hi] inherit this.
    lo = prop._backend_impl.padding[0] + M
    hi = lo + nxp

    def exchange(fast, lists_slots):
        for L, slots in lists_slots:
            for f in slots:
                fast.exchange(L[f][..., lo - M: hi + M])

    # ---------------- DD forward (half-step + NCCL v/s exchange) ----------
    fwd_fast = FastHaloSet(mesh, M, ("x",)) if world > 1 else None
    fr = tile.fwd_runner()
    with torch.no_grad():
        for it in range(NT):
            fr.run_phase(it + 1, 1)
            if world > 1:
                exchange(fwd_fast, [(fr.L, range(NV[ndim]))])
            fr.run_phase(it + 1, 2)
            if world > 1:
                exchange(fwd_fast, [(fr.L, range(NV[ndim], NPHYS[ndim]))])
    tile.bp.boundary_gpu = list(tile.fp.boundary_gpu)
    tile.bp.u_last_two = tile.fp.last_two

    # ---------------- DD backward (phased + NCCL 4-group exchange) ---------
    tile.zero_backward_state()
    tile.bp.cut_face_mask = tile.cut_face_mask
    bwd_fast = FastHaloSet(mesh, M, ("x",)) if world > 1 else None
    br = tile.bwd_runner()
    with torch.no_grad():
        for it in range(NT - 1, 0, -1):   # elastic BS floor: it == 1
            # injections as their own sub-phase, then ship the ph2 fields
            # they may have written (adj stress / recon velocity) BEFORE the
            # phase-1 kernels read them across the cut
            br.run_phase(it + 1, it, 3)
            if world > 1:
                exchange(bwd_fast, [(br.L_adj, range(NV[ndim], NPHYS[ndim])),
                                    (br.L_recon, range(NV[ndim]))])
            br.run_phase(it + 1, it, 1)
            if world > 1:
                exchange(bwd_fast, [(br.L_adj, range(NV[ndim])),
                                    (br.L_recon, range(NV[ndim], NPHYS[ndim]))])
            br.run_phase(it + 1, it, 2)
            if world > 1:
                exchange(bwd_fast, [(br.L_adj, range(NV[ndim], NPHYS[ndim])),
                                    (br.L_recon, range(NV[ndim]))])
    assert br.k_adj == NT - 1

    # ---------------- gather owned grad slices on rank 0 ----------------
    payload = (own_idx, [g[..., lo:hi].cpu() for g in tile.gbufs])
    gathered = [None] * world
    if world > 1:
        dist.gather_object(payload, gathered if rank == 0 else None, dst=0)
    else:
        gathered = [payload]

    if rank == 0:
        # reference monolithic backward replay -> ref.gbufs
        ref.bp.adjoint_source = residual_raw
        ref.zero_backward_state()
        ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
        ref.bp.step_phase, ref.bp.cut_face_mask = 0, 0
        ref.bp.adjoint_wavefields = ref.L_adj
        ref.bp.forward_wavefields = ref.recon
        with torch.no_grad():
            ref.bwd_func(ref.bp)
        assert any(t.abs().max() > 0 for t in ref.gbufs), "reference grads zero"

        def grade(name, got, want):
            bit = torch.equal(got, want)
            mad = (got - want).abs().max().item()
            scale = want.abs().max().item() + 1e-30
            rel = mad / scale
            print(f"[rank0] {name}: bitexact={bit} max|d|={mad:.3e} rel={rel:.3e}")
            return 2 if bit else (1 if rel < REL_TOL else 0)

        worst = 2
        names = ["grad_vp", "grad_vs", "grad_rho"]
        for r, (idxs, gslices) in enumerate(gathered):
            for k, name in enumerate(names):
                want = ref.gbufs[k][..., PAD + r * nxp: PAD + r * nxp + nxp].cpu()
                worst = min(worst, grade(f"tile{r} {name}", gslices[k], want))
        verdict = {2: "PASS", 1: "PASS_TOL", 0: "FAIL"}[worst]
        print(f"DD_NCCL_ELASTIC_BACKWARD_CHECK: {verdict}   [{label}]")
        blob = torch.cat([g.flatten() for _, gs in gathered for g in gs]).clone()
    else:
        verdict, blob = None, None
    if world > 1:
        vobj = [verdict]
        dist.broadcast_object_list(vobj, src=0)
        verdict = vobj[0]
    return verdict, blob



def _matrix_cases():
    """The regression matrix (world=2). Mirrors the sbatch sweep, plus the
    repeat-first isolation probe at the end."""
    from types import SimpleNamespace as C

    def mk(ndim, st=None, rt=None, fs=False, dx=0, roc=False, tag=""):
        return C(ndim=ndim, source_type=st, receiver_type=rt,
                 free_surface=fs, src_dx=dx, rec_on_cut=roc,
                 expect_reject=False,
                 tag=tag or f"{ndim}D src={st or 'def'} rec={rt or 'def'}"
                            f"{' fs' if fs else ''}"
                            f"{f' dx={dx}' if dx else ''}"
                            f"{' rec-on-cut' if roc else ''}")

    cases = []
    t2s = ["vx", "vz", "sxx", "szz", "sxz"]     # all five are valid SOURCES
    t2r = ["vx", "vz", "sxx", "szz"]            # sxz is not a valid receiver
    for st in t2s:
        for rt in t2r:
            cases.append(mk(2, [st], [rt]))
    # the library must REFUSE shear-stress receivers (half-half staggered
    # node) loudly -- pin that instead of feeding the matrix invalid cases
    rej2 = mk(2, None, ["sxz"], tag="2D rec=sxz (expect library reject)")
    rej2.expect_reject = True
    cases.append(rej2)
    cases += [mk(2, None, ["sxx", "szz"]), mk(2, None, ["vz", "sxx"]),
              mk(2, ["vx", "vz"], None), mk(2, ["vz", "sxx"], None)]
    for dx in (0, 1, 2, 3, -1, -2, -3, 8):
        cases += [mk(2, None, None, dx=dx), mk(2, ["vz"], None, dx=dx)]
    cases += [mk(2, roc=True), mk(2, ["vz"], None, roc=True),
              mk(2, None, ["sxx", "szz"], roc=True)]
    cases += [mk(2, fs=True), mk(2, ["vz"], None, fs=True),
              mk(2, None, ["sxx", "szz"], fs=True)]
    t3s = ["vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz"]
    t3r = ["vx", "vy", "vz", "sxx", "syy", "szz"]
    cases += [mk(3, [st], None) for st in t3s]
    cases += [mk(3, None, [rt]) for rt in t3r]
    rej3 = mk(3, None, ["sxy"], tag="3D rec=sxy (expect library reject)")
    rej3.expect_reject = True
    cases.append(rej3)
    cases += [mk(3, roc=True), mk(3, dx=1), mk(3, dx=8)]
    first = cases[0]
    cases.append(mk(first.ndim, first.source_type, first.receiver_type,
                    tag="REPEAT-FIRST (isolation probe)"))
    return cases


def main():
    import argparse
    import gc
    import time
    from types import SimpleNamespace

    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--free-surface", action="store_true")
    ap.add_argument("--source-type", type=str, default=None)
    ap.add_argument("--receiver-type", type=str, default=None)
    ap.add_argument("--src-dx", type=int, default=0)
    ap.add_argument("--rec-on-cut", action="store_true")
    ap.add_argument("--matrix", action="store_true",
                    help="run the whole regression matrix in-process: one "
                         "NCCL bring-up, everything else per-case fresh")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    dev_idx = local_rank % max(1, torch.cuda.device_count())
    torch.cuda.set_device(dev_idx)
    dev = torch.device(f"cuda:{dev_idx}")

    if not args.matrix:
        case = SimpleNamespace(
            ndim=args.ndim, free_surface=args.free_surface,
            source_type=args.source_type.split(",") if args.source_type else None,
            receiver_type=(args.receiver_type.split(",")
                           if args.receiver_type else None),
            src_dx=args.src_dx, rec_on_cut=args.rec_on_cut, tag="single")
        verdict, _ = run_case(case, rank, world, dev)
        if world > 1:
            dist.destroy_process_group()
        sys.exit(0 if verdict != "FAIL" else 1)

    assert world == 2, "--matrix is defined for exactly 2 ranks"
    cases = _matrix_cases()
    results, blob0, t_all = [], None, time.time()
    for i, case in enumerate(cases):
        t0 = time.time()
        try:
            verdict, blob = run_case(case, rank, world, dev)
            if case.expect_reject:
                verdict = "FAIL(SHOULD-HAVE-REJECTED)"
        except Exception as e:  # deterministic failures raise on both ranks
            if case.expect_reject and "not valid for receiver_type" in str(e):
                verdict, blob = "PASS", None
            else:
                verdict, blob = f"ERROR({type(e).__name__})", None
        if i == 0:
            blob0 = blob
        if case.tag.startswith("REPEAT-FIRST") and rank == 0:
            iso = (blob is not None and blob0 is not None
                   and torch.equal(blob, blob0))
            print(f"[matrix] isolation probe: "
                  f"{'BITWISE-IDENTICAL' if iso else 'CONTAMINATED'}")
            if not iso:
                results.append(("isolation-probe", "FAIL"))
        results.append((case.tag, verdict))
        if rank == 0:
            print(f"[matrix] {i + 1:3d}/{len(cases)}  {case.tag:44s} "
                  f"{verdict}  ({time.time() - t0:4.1f}s)", flush=True)
        del blob
        gc.collect()
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()

    bad = [(t, v) for t, v in results if v not in ("PASS",)]
    if rank == 0:
        print(f"\n[matrix] {len(results)} cases in "
              f"{(time.time() - t_all) / 60:.1f} min; "
              f"{'ALL PASS' if not bad else f'{len(bad)} NOT PASS:'}")
        for t, v in bad:
            print(f"  {v:10s} {t}")
    if world > 1:
        dist.destroy_process_group()
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
