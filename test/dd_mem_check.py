"""Per-GPU memory balance check for ModelParallel (real DD run).

torchrun --standalone --nproc-per-node=<P> test/dd_mem_check.py \
    --family acoustic --ndim 3 --px 8 --nz 192 --nyp 192 --nxp 192 --nt 400

Each rank generates ONLY its own tile model (no global on any card), runs the
DD forward + gradient (boundary saving), and reports its peak GPU memory.
Rank 0 gathers all ranks' peaks + tile coords + whether they own the source /
how many receivers, and prints the spread so we can see if any card is a
hotspot or the ring/edge effects matter.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic3D, Elastic3D, Acoustic, Elastic  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DT = 0.0005


def ricker(nt, dt, fm=10.0, delay=0.04, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return (scale * (1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="acoustic", choices=("acoustic", "elastic"))
    ap.add_argument("--ndim", type=int, default=3, choices=(2, 3))
    ap.add_argument("--px", type=int, default=0)
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--nyp", type=int, default=192, help="tile y (3-D)")
    ap.add_argument("--nxp", type=int, default=192, help="tile x")
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--free-surface", action="store_true")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li); dev = torch.device(f"cuda:{li}")
    px = args.px or world; py = args.py
    assert px * py == world, f"px*py={px*py} != world={world}"
    fam, ndim, nt = args.family, args.ndim, args.nt

    nz, nyp, nxp = args.nz, args.nyp, args.nxp
    nx = nxp * px
    if ndim == 3:
        ny = nyp * py
        global_shape = (nz, ny, nx)
    else:
        global_shape = (nz, nx)

    topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
    if fam == "acoustic":
        eqc = Acoustic3D if ndim == 3 else Acoustic
        st, rt = ["h1"], ["h1"]
        nmodel = 1
    else:
        eqc = Elastic3D if ndim == 3 else Elastic
        st = ["sxx", "syy", "szz"] if ndim == 3 else ["sxx", "szz"]
        rt = ["vx", "vy", "vz"] if ndim == 3 else ["vx", "vz"]
        nmodel = 3
    prop = PropTorch(eqc(spatial_order=args.so, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=global_shape, dh=10.0, dt=DT,
                     nt=nt, abcn=args.abcn, source_type=st, receiver_type=rt, dev=dev,
                     free_surface=args.free_surface)
    ddp = ModelParallel(prop, topo)

    # this rank's OWN tile model (uniform) — no global model on any card
    ls = ddp.local_shape
    if fam == "acoustic":
        models = [np.full(ls, 2500.0, dtype=np.float32)]
    else:
        models = [np.full(ls, 3000.0, dtype=np.float32),
                  np.full(ls, 1730.0, dtype=np.float32),
                  np.full(ls, 2200.0, dtype=np.float32)]

    # realistic geometry: one source at centre; receivers spread on the surface
    if ndim == 3:
        src = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, iy, 2]
                         for iy in range(2, ny - 2, max(1, ny // 16))
                         for ix in range(2, nx - 2, max(1, nx // 16))]], dtype=np.int32)
    else:
        src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, 2] for ix in range(2, nx - 2, max(1, nx // 64))]],
                       dtype=np.int32)
    wav = ricker(nt, DT, scale=1e6 if fam == "elastic" else 1.0)

    models_t = [torch.tensor(m, device=dev, requires_grad=True) for m in models]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    rec_tile = ddp.forward(wav, src, rec, models=models_t)   # autograd forward
    rec_tile.backward(gradient=rec_tile.detach())            # adjoint = record
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30

    info = (rank, tuple(topo.coord), bool(ddp._owns_src),
            len(ddp._own_rec_idx), float(peak))
    gathered = [None] * world
    dist.gather_object(info, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        peaks = [g[4] for g in gathered]
        lo, hi = min(peaks), max(peaks)
        print(f"\n=== {fam} {ndim}D  px={px} py={py}  tile={ls}  nt={nt} "
              f"fs={args.free_surface} ===")
        for r, coord, owns_src, nrec, pk in sorted(gathered):
            edge = "src" if owns_src else "   "
            print(f"  rank{r} coord{coord} {edge} nrec={nrec:<4d} "
                  f"peak={pk:.3f} GB")
        print(f"  --> min={lo:.3f} max={hi:.3f} GB  imbalance={100*(hi-lo)/lo:.1f}%")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
