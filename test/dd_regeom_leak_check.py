"""Variable-geometry re-capture leak + correctness check for ModelParallel.

The encoded-OBN FWI loop re-seeds the shared shots every iteration, so each
tile's owned (#sources, #receivers) changes iter-to-iter. Previously the runner
forced ``solver._captured = False`` every iter -> a full re-capture that
reallocated every model-sized buffer + the boundary ring and LEAKED ~one
adjoint-wavefield set per iter (memory_allocated climbed ~monotonically ->
NCCL-timeout / OOM). ModelParallel._set_geometry now reuses those buffers and
reallocates only the two small count-sized buffers (record ~ nrec,
grad-wavelet ~ nsrc).

Three isolated checks (memory phases run ONE solver each so the shared CUDA
pool can't cross-contaminate the readings):

  Phase A  leak(fixed)   : one persistent solver, in-place re-geometry every
                           iter -> memory_allocated must be FLAT.
  Phase B  leak(control) : one persistent solver, FORCE re-capture every iter
                           (the old behaviour) -> memory_allocated GROWS
                           (~one buffer set / iter) = the leak the fix removes.
  Phase C  correctness   : the in-place gradient must be BIT-EXACT to a forced
                           re-capture (proves reuse == fresh capture) and close
                           to single-domain PropTorch.

torchrun --standalone --nproc-per-node=2 test/dd_regeom_leak_check.py --px 2
torchrun --standalone --nproc-per-node=4 test/dd_regeom_leak_check.py --px 2 --py 2
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.09):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def cos_rel(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    cos = (a @ b) / (a.norm() * b.norm() + 1e-300)
    rel = (a - b).norm() / (b.norm() + 1e-300)
    return float(cos), float(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--py", type=int, default=1)
    # Grid chosen so every tile interior stays wider than the PML even at 2x2
    # (ny/py, nx/px > abcn): a tile thinner than its PML is a degenerate config
    # the DD reconstruction was never meant for and adds a spurious cut error.
    ap.add_argument("--nz", type=int, default=40)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--nt", type=int, default=140)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--abcn", type=int, default=15)
    ap.add_argument("--so", type=int, default=4)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li); dev = torch.device(f"cuda:{li}")
    px, py = args.px, args.py
    assert px * py == world, f"px*py={px*py} != world={world}"
    nz, ny, nx, nt = args.nz, args.ny, args.nx, args.nt
    gshape = (nz, ny, nx)

    z = np.linspace(0, 1, nz, dtype=np.float32)[:, None, None]
    init_vp = np.broadcast_to(1800.0 + 600.0 * z, gshape).astype(np.float32).copy()
    init_vp[nz // 3:2 * nz // 3, ny // 3:2 * ny // 3, nx // 3:2 * nx // 3] += 250.0

    base = ricker(nt, DT)
    # x/y cut columns/rows to AVOID for correctness (a source/receiver exactly on
    # a tile boundary makes the halo reconstruction look like a numeric failure).
    xcuts = {k * nx // px for k in range(1, px)}
    ycuts = {k * ny // py for k in range(1, py)}

    def geom(i, avoid_cuts):
        """Different per-tile (#src,#rec) each iter -> exercises both realloc
        branches of _set_geometry (record ~ nrec, grad-wavelet ~ nsrc)."""
        rng = np.random.RandomState(100 + i)
        B_i = int(5 + (i % 4))                       # 5..8 encoded sources
        nrec_i = int(30 + 6 * (i % 5))               # 30..54 receivers

        def _shift(v, cuts, hi):
            if avoid_cuts:
                while v in cuts or v <= 1 or v >= hi - 1:
                    v += 1
            return int(v)

        xs = np.linspace(4, nx - 5, B_i).astype(np.int64)
        ys = np.where(np.arange(B_i) % 2 == 0, ny // 4, 3 * ny // 4).astype(np.int64)
        src = np.array([[[_shift(int(xs[k]), xcuts, nx),
                          _shift(int(ys[k]), ycuts, ny), 3] for k in range(B_i)]],
                       dtype=np.int64)
        signs = rng.choice([-1.0, 1.0], size=B_i).astype(np.float32)
        wav = (base[None, :] * signs[:, None]).astype(np.float32)   # (B_i, nt)
        rxs = np.linspace(2, nx - 3, nrec_i).astype(np.int64)
        rys = (3 + (np.arange(nrec_i) * 3) % (ny - 6)).astype(np.int64)
        recv = np.array([[[_shift(int(rxs[k]), xcuts, nx),
                           _shift(int(rys[k]), ycuts, ny), 2]
                          for k in range(nrec_i)]], dtype=np.int64)
        return src, recv, wav, B_i, nrec_i

    mesh = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)

    def build_prop():
        eq = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=gshape, dh=10.0, dt=DT,
                         nt=nt, abcn=args.abcn, source_type=["h1"],
                         receiver_type=["h1"], dev=dev, free_surface=False)

    def dd_grad(solver, wav, src, recv, force_recapture):
        vp = torch.tensor(init_vp, device=dev, requires_grad=True)
        wav_t = torch.as_tensor(wav, device=dev)
        if force_recapture:
            solver._captured = False
        rc = solver(wav_t, src, recv, models=[vp])
        loss = (rc.double() ** 2).sum()
        dist.all_reduce(loss)
        loss.backward()
        dist.all_reduce(vp.grad)
        g = vp.grad.detach().clone()
        return float(loss), g

    def leak_phase(force_recapture):
        """One isolated solver; measure live memory after each varying-count
        iter (all refs dropped) -> a per-iter buffer leak shows as growth."""
        solver = ModelParallel(build_prop(), mesh)
        mem = []
        for i in range(args.iters):
            src, recv, wav, _, _ = geom(i, avoid_cuts=False)
            _l, g = dd_grad(solver, wav, src, recv, force_recapture)
            del g
            torch.cuda.synchronize(dev)
            mem.append(torch.cuda.memory_allocated(dev) / 2**20)
        return solver, mem

    # -- Phase A: fixed (in-place re-geometry) must be FLAT --------------------
    solver_fixed, mem_fixed = leak_phase(force_recapture=False)
    # -- Phase B: control (force re-capture) reproduces the leak ---------------
    solver_ctrl, mem_ctrl = leak_phase(force_recapture=True)

    # -- Phase C: correctness (bit-exact vs re-capture, close to single-domain) -
    prop_ref = build_prop()
    rows = []
    for i in range(args.iters):
        src, recv, wav, B_i, nrec_i = geom(i, avoid_cuts=True)
        vp_ref = torch.tensor(init_vp, device=dev, requires_grad=True)
        wav_t = torch.as_tensor(wav, device=dev)
        rc_ref = prop_ref(wav_t, src, recv, models=[vp_ref])
        loss_ref = (rc_ref.double() ** 2).sum()
        loss_ref.backward()
        g_ref = vp_ref.grad.detach().clone()

        l_fx, g_fx = dd_grad(solver_fixed, wav, src, recv, force_recapture=False)
        _l_rc, g_rc = dd_grad(solver_ctrl, wav, src, recv, force_recapture=True)
        cos, rel = cos_rel(g_fx, g_ref)
        lrel = abs(l_fx - float(loss_ref)) / (abs(float(loss_ref)) + 1e-300)
        bit_vs_recap = bool(torch.equal(g_fx, g_rc))
        rows.append((i, B_i, nrec_i, l_fx, lrel, cos, rel, bit_vs_recap))
        del g_ref, g_fx, g_rc, vp_ref, rc_ref, loss_ref

    if rank == 0:
        print(f"\n=== DD variable-geometry re-capture leak+correctness  "
              f"world={world} px={px} py={py} global={gshape} nt={nt} "
              f"abcn={args.abcn} ===", flush=True)

        # correctness: bit-exact vs re-capture is the hard gate; vs single-domain
        # cos/rel is a sanity band (DD physics, unaffected by this change).
        corr_ok = True
        for (i, B_i, nrec_i, l_fx, lrel, cos, rel, bit) in rows:
            ok = bit and (cos > 0.999 and rel < 5e-3)
            corr_ok = corr_ok and ok
            print(f"  iter {i} B={B_i} nrec={nrec_i}: loss={l_fx:.6e} lrel={lrel:.1e} "
                  f"| vs single-domain cos={cos:.8f} rel_l2={rel:.2e} "
                  f"| bit==recapture:{bit} -> {'PASS' if ok else 'FAIL'}", flush=True)

        # leak: fixed flat, control grows. Compare per-iter growth from iter 1
        # (iter 0 allocates the reusable buffers on first capture).
        gf = max(mem_fixed[1:]) - min(mem_fixed[1:])
        gc = mem_ctrl[-1] - mem_ctrl[1]
        print(f"\n  memory_allocated MB per iter (isolated solver, refs dropped):",
              flush=True)
        print(f"    fixed  (in-place _set_geometry): {[f'{m:.0f}' for m in mem_fixed]}",
              flush=True)
        print(f"    control(force re-capture)      : {[f'{m:.0f}' for m in mem_ctrl]}",
              flush=True)
        print(f"  fixed spread (iter>=1) = {gf:.1f} MB (should be ~0)  |  "
              f"control growth = {gc:.1f} MB (the per-iter leak the fix removes)",
              flush=True)
        leak_free = gf < 5.0 and gc > 3.0 * max(gf, 1.0)  # fixed flat AND control clearly leaks
        print(f"\n  RESULT: correctness={'PASS' if corr_ok else 'FAIL'}  "
              f"leak_free={'PASS' if leak_free else 'FAIL'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
