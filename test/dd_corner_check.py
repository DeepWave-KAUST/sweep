"""2-axis DD (PYxPX) cut/PML-corner parity vs single domain (bit-exact).

Regression for the multi-axis halo-view bug: the old ``_halo_view`` cropped
the exchange view to owned±M on EVERY cut axis, so on a 2-axis mesh the
y-exchange strips lost the x-side global-PML pad columns (and vice versa).
Those pad cells are live — the in_pml branch updates them each step and
their stencil reads the cut halo — so the un-exchanged halo cells stayed
zero: a hard reflecting wall at each (cut face x perpendicular PML band)
intersection. Waves crossing a cut inside the perpendicular PML band
scattered off it (seen in production: velocity-anomaly artifacts at tile-cut x
model-edge corners). Single-axis splits never crop the perpendicular axis,
which is why 2x1 / 1x2 stayed bit-exact while 2x2 diverged.

The geometry here makes the corners LOUD: one source 1 cell off the y-cut,
one near the x-hi edge, receivers edge-to-edge on the surface, nt long
enough for the wavefront to cross both cuts inside the PML bands and
reverberate.

Launch: torchrun --nproc-per-node=4 test/dd_corner_check.py [--fs 0]
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
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", type=int, default=2)
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--fs", type=int, default=1)
    ap.add_argument("--nt", type=int, default=1200)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--so", type=int, default=4)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert args.py * args.px == world, (
        f"this check needs nproc-per-node={args.py * args.px} "
        f"(py={args.py} x px={args.px}), got world={world}. It exists to guard a "
        f"TWO-AXIS mesh — the corner bug only appears where a cut face crosses "
        f"the perpendicular PML band — so a 2-rank launch cannot exercise it at "
        f"all. Launch with 4 ranks, or pass --py/--px matching your rank count.")
    nz, ny, nx, nt = 64, 96, 80, args.nt
    gshape = (nz, ny, nx)

    z = np.arange(nz, dtype=np.float32)[:, None, None]
    vp = 1800.0 + 14.0 * np.maximum(z - 12.0, 0.0)
    vp = np.broadcast_to(vp, gshape).astype(np.float32).copy()
    vp[:12] = 1500.0
    vp[44:50] += 600.0
    vp = np.minimum(vp, 3050.0)

    # mid-tile / 1 cell off the y-cut / near the x-hi edge
    src = np.array([[[37, 27, 20], [60, 47, 20], [75, 70, 20]]], dtype=np.int64)
    wav = torch.as_tensor(ricker(nt, DT), device=dev)
    rec = np.array([[[ix, iy, 0]
                     for iy in range(2, ny - 2, 2)
                     for ix in range(2, nx - 2, 2)]], dtype=np.int64)
    nrec = rec.shape[1]

    def build_prop():
        eq = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=gshape, dh=DH,
                         dt=DT, nt=nt, abcn=args.abcn, source_type=["h1"],
                         receiver_type=["h1"], dev=dev,
                         free_surface=bool(args.fs))

    vp_ref = torch.tensor(vp, device=dev, requires_grad=True)
    rec_ref = build_prop()(wav, src, rec, models=[vp_ref])
    (rec_ref.double() ** 2).sum().backward()

    mesh = MeshTopology(py=args.py, px=args.px, shot_groups=1,
                        world_size=world, rank=rank)
    ddp = ModelParallel(build_prop(), mesh)
    vp_dd = torch.tensor(vp, device=dev, requires_grad=True)
    rec_dd = ddp(wav, src, rec, models=[vp_dd])
    own = getattr(ddp, "_own_rec_idx", None)
    own = (np.arange(nrec) if own is None
           else np.asarray(own, dtype=np.int64).ravel())
    (rec_dd.double() ** 2).sum().backward()
    dist.all_reduce(vp_dd.grad)

    R = rec_ref.detach().cpu().numpy().squeeze()
    D = rec_dd.detach().cpu().numpy().squeeze()
    if R.shape[0] != nrec:          # -> (nrec, nt); single and DD raw
        R = R.T                     # layouts differ, normalise separately
    if D.shape[0] != len(own):
        D = D.T
    rec_bit = bool(np.array_equal(D, R[own]))
    grad_bit = bool(torch.equal(vp_dd.grad, vp_ref.grad))
    dmax = float(np.abs(D - R[own]).max()) / (float(np.abs(R).max()) + 1e-30)
    gr = float((vp_dd.grad - vp_ref.grad).norm()
               / (vp_ref.grad.norm() + 1e-30))
    ok = rec_bit and grad_bit
    print(f"[rank{rank}] rec_bit={rec_bit} (max_rel={dmax:.2e}) "
          f"grad_bit={grad_bit} (rel={gr:.2e}) -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    fail = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(fail)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
