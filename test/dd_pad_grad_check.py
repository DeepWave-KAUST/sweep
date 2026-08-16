"""Physical-grid leaf + differentiable pad_to_mesh: DD parity end to end.

The recommended way to run DD on a non-divisible grid is to keep the
UNPADDED model as the optimisation variable and apply
:func:`sweep.parallel.pad_to_mesh` inside the closure. This checks the
whole loop on a deliberately non-divisible grid (ny = nx = 95):

* the acquisition is untouched — source / receiver indices are the same
  integers in the single-domain and DD runs, no shifting anywhere;
* ``vp.grad`` lands on the PHYSICAL grid on both sides (autograd routes
  the pad-region gradient back through the replicate adjoint);
* DD gradient and records are bit-exact against the single-domain run of
  the same padded problem.

Launch: torchrun --standalone --nproc-per-node=4 test/dd_pad_grad_check.py
        torchrun --standalone --nproc-per-node=4 test/dd_pad_grad_check.py --py 1 --px 4
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
from sweep.parallel import MeshTopology, pad_to_mesh      # noqa: E402
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
    ap.add_argument("--ny", type=int, default=95)   # deliberately NOT divisible
    ap.add_argument("--nx", type=int, default=95)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert args.py * args.px == world, f"py*px must equal world size {world}"

    nz, ny, nx, nt = args.nz, args.ny, args.nx, args.nt
    assert ny % args.py or nx % args.px, \
        "grid divides evenly -- this test wants the padded path"

    z = np.arange(nz, dtype=np.float32)[:, None, None]
    vp = 1800.0 + 14.0 * np.maximum(z - 10.0, 0.0)
    vp = np.broadcast_to(vp, (nz, ny, nx)).astype(np.float32).copy()
    vp[:10] = 1500.0
    vp[40:46] += 500.0
    vp = np.minimum(vp, 3050.0)

    mesh = MeshTopology(py=args.py, px=args.px, shot_groups=1,
                        world_size=world, rank=rank)
    padded_shape = pad_to_mesh(vp, mesh).shape
    dy, dx = padded_shape[1] - ny, padded_shape[2] - nx

    wav = torch.as_tensor(ricker(nt, DT), device=dev)
    # acquisition on the PHYSICAL grid -- identical integers on both sides
    src = np.array([[[nx // 3, ny // 3, 18]]], dtype=np.int64)
    rec = np.array([[[ix, iy, 0]
                     for iy in range(2, ny - 2, 2)
                     for ix in range(2, nx - 2, 2)]], dtype=np.int64)

    def build_prop():
        eq = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=padded_shape,
                         dh=DH, dt=DT, nt=nt, abcn=args.abcn,
                         source_type=["h1"], receiver_type=["h1"], dev=dev,
                         free_surface=bool(args.fs))

    if rank == 0:
        print(f"physical=({nz},{ny},{nx}) padded={tuple(padded_shape)} "
              f"(dy={dy}, dx={dx}) mesh=py{args.py}xpx{args.px} "
              f"so={args.so} abcn={args.abcn}")

    # ---- single domain: physical leaf, pad in the closure ----------------
    vp_ref = torch.tensor(vp, device=dev, requires_grad=True)
    r_ref = build_prop()(wav, src, rec, models=[pad_to_mesh(vp_ref, mesh)])
    (r_ref.double() ** 2).sum().backward()

    # ---- DD: same physical leaf pattern ----------------------------------
    ddp = ModelParallel(build_prop(), mesh)
    vp_dd = torch.tensor(vp, device=dev, requires_grad=True)
    r_dd = ddp(wav, src, rec, models=[pad_to_mesh(vp_dd, mesh)])
    (r_dd.double() ** 2).sum().backward()
    dist.all_reduce(vp_dd.grad)

    # 1) both gradients live on the physical grid
    shape_ok = (tuple(vp_ref.grad.shape) == (nz, ny, nx)
                and tuple(vp_dd.grad.shape) == (nz, ny, nx))

    # 2) gradient bit-exact
    gbit = bool(torch.equal(vp_dd.grad, vp_ref.grad))
    rel = float((vp_dd.grad - vp_ref.grad).norm()
                / (vp_ref.grad.norm() + 1e-30))

    # 3) records bit-exact on this rank's owned receivers
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

    # 4) the pad actually did something and the edge picked up the summed
    #    adjoint: the last physical row/col gradient must be finite, nonzero
    edge_ok = bool(torch.isfinite(vp_dd.grad).all()) and \
        (dy == 0 or float(vp_dd.grad[:, -1, :].abs().sum()) > 0.0) and \
        (dx == 0 or float(vp_dd.grad[:, :, -1].abs().sum()) > 0.0)

    ok = shape_ok and gbit and rbit and edge_ok
    if rank == 0:
        print(f"grad on physical grid: {shape_ok}")
        print(f"grad bit-exact:        {gbit}   (rel={rel:.2e})")
        print(f"records bit-exact:     {rbit}")
        print(f"edge adjoint sane:     {edge_ok}")
        print(f"-> {'PASS' if ok else 'FAIL'}")

    fail = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(fail)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
