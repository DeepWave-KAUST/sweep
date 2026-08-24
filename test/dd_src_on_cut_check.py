"""Source / receivers sitting ON a tile cut: DD parity vs single domain.

The DD forward injects the source in phase 2, after phase 1 has exchanged the
cut strips, so a source inside a strip would cross the cut one step late.
``ModelParallel._src_away_from_cuts`` guards this by dropping comm/compute
overlap whenever a source is within M of an x-cut. That guard is easy to break
silently: if it stops firing, results stay plausible and only differ in the
last bits near the cut; if overlap stops being taken at all, the guard becomes
dead code and every placement "passes" without the risky path ever running.

So this walks the source across the boundary -- exactly on the cut, 1 / M /
M+1 cells off, on the y-cut, and on the cut intersection -- and asserts
bit-exactness against the single-domain answer each time, while also reporting
whether overlap actually engaged. The receiver grid is asserted to land on the
cut lines, so every case is a source AND receivers on the interface.

  py=1 px=N : x-cuts only -> overlap is eligible, the guard is exercised
  py=2 px=2 : cut_mask carries y bits -> overlap is off by design
              (phased forward v1 is x-face only), the serial path is exercised

Launch: torchrun --standalone --nproc-per-node=4 test/dd_src_on_cut_check.py
        torchrun --standalone --nproc-per-node=4 test/dd_src_on_cut_check.py --py 1 --px 4
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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Acoustic3D                    # noqa: E402
from sweep.parallel import MeshTopology                   # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel    # noqa: E402
from sweep.propagator.torch import PropTorch              # noqa: E402

DT = 0.0012
DH = 15.0


def ricker(nt, dt, fm=6.0, delay=0.2):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", type=int, default=2)
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--fs", type=int, default=1)
    ap.add_argument("--nt", type=int, default=700)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--nz", type=int, default=80)
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--nx", type=int, default=96)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert args.py * args.px == world, f"py*px must equal world size {world}"

    nz, ny, nx, nt = args.nz, args.ny, args.nx, args.nt
    gshape = (nz, ny, nx)
    M = args.so // 2
    cx, cy = nx // args.px, ny // args.py
    has_y_cut = args.py > 1

    z = np.arange(nz, dtype=np.float32)[:, None, None]
    vp = 1800.0 + 14.0 * np.maximum(z - 10.0, 0.0)
    vp = np.broadcast_to(vp, gshape).astype(np.float32).copy()
    vp[:10] = 1500.0
    vp[40:46] += 500.0
    vp = np.minimum(vp, 3050.0)

    wav = torch.as_tensor(ricker(nt, DT), device=dev)
    rec = np.array([[[ix, iy, 0]
                     for iy in range(2, ny - 2, 2)
                     for ix in range(2, nx - 2, 2)]], dtype=np.int64)
    # the point of the test: receivers must sit ON the cut, not near it
    assert cx in range(2, nx - 2, 2), "receiver grid must land on the x-cut"
    if has_y_cut:
        assert cy in range(2, ny - 2, 2), "receiver grid must land on the y-cut"

    cases = [
        ("on x-cut",       [cx,         ny // 4, 18]),
        ("x-cut - 1",      [cx - 1,     ny // 4, 18]),
        ("x-cut + 1",      [cx + 1,     ny // 4, 18]),
        ("x-cut + M",      [cx + M,     ny // 4, 18]),
        ("x-cut + M+1",    [cx + M + 1, ny // 4, 18]),
        ("far from cuts",  [nx // 4,    ny // 4, 18]),
    ]
    if has_y_cut:
        cases += [("on y-cut",     [nx // 4, cy, 18]),
                  ("on both cuts", [cx,      cy, 18])]

    def build_prop():
        eq = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=gshape, dh=DH,
                         dt=DT, nt=nt, abcn=args.abcn, source_type=["h1"],
                         receiver_type=["h1"], dev=dev,
                         free_surface=bool(args.fs))

    mesh = MeshTopology(py=args.py, px=args.px, shot_groups=1,
                        world_size=world, rank=rank)

    if rank == 0:
        print(f"grid={gshape} so={args.so} M={M} abcn={args.abcn} "
              f"mesh=py{args.py}xpx{args.px} x-cut@{cx}"
              f"{f' y-cut@{cy}' if has_y_cut else ''}")
        print(f"{'case':16s} {'overlap':>8s} {'grad bit':>9s} "
              f"{'rec bit':>8s} {'rel':>10s}")

    bad, overlap_seen = 0, False
    for name, s in cases:
        src = np.array([[s]], dtype=np.int64)

        vp_ref = torch.tensor(vp, device=dev, requires_grad=True)
        r_ref = build_prop()(wav, src, rec, models=[vp_ref])
        (r_ref.double() ** 2).sum().backward()

        ddp = ModelParallel(build_prop(), mesh)
        vp_dd = torch.tensor(vp, device=dev, requires_grad=True)
        r_dd = ddp(wav, src, rec, models=[vp_dd])
        (r_dd.double() ** 2).sum().backward()
        dist.all_reduce(vp_dd.grad)

        overlap = bool(getattr(ddp, "_overlap_ok", False)) and \
            bool(ddp._src_away_from_cuts(torch.as_tensor(src)))
        overlap_seen |= overlap

        gbit = bool(torch.equal(vp_dd.grad, vp_ref.grad))
        rel = float((vp_dd.grad - vp_ref.grad).norm()
                    / (vp_ref.grad.norm() + 1e-30))

        R = r_ref.detach().cpu().numpy().squeeze()
        D = r_dd.detach().cpu().numpy().squeeze()
        own = getattr(ddp, "_own_rec_idx", None)
        own = (np.arange(rec.shape[1]) if own is None
               else np.asarray(own, dtype=np.int64).ravel())
        if R.shape[0] != rec.shape[1]:
            R = R.T
        if D.shape[0] != len(own):
            D = D.T
        rbit = bool(np.array_equal(D, R[own]))

        if rank == 0:
            print(f"{name:16s} {str(overlap):>8s} {str(gbit):>9s} "
                  f"{str(rbit):>8s} {rel:10.2e}")
        if not (gbit and rbit):
            bad += 1
        del ddp

    # A green run in which overlap never engaged has not tested the risky path.
    # On an x-only mesh the "x-cut + M+1" placement must take it.
    if rank == 0 and not has_y_cut and not overlap_seen:
        print("ERROR: overlap never engaged on an x-only mesh -- the guarded "
              "path was not exercised, so these PASSes are vacuous")
        bad += 1

    ok = bad == 0
    if rank == 0:
        print(f"-> {'PASS' if ok else 'FAIL'} "
              f"({bad} bad case(s), overlap_exercised={overlap_seen})")

    fail = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(fail)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
