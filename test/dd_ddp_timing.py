"""Time ModelParallel.forward end-to-end for a given decomposition.

Uses the PRODUCTION DD path (correct multi-axis / corner halo exchange and the
acoustic comm/compute overlap), so unlike dd_nccl_bench it is correct for 2-D
(px>1 AND py>1) decompositions. Reports per-step wall time for the SAME global
problem under different px/py — the strong-scaling axis comparison.

  torchrun --nproc-per-node=8 test/dd_ddp_timing.py --px 8 --py 1 \
      --nz 256 --ny 256 --nx 1024 --nt 200
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DT = 0.0005


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=0, help="x tiles (0 => world)")
    ap.add_argument("--py", type=int, default=1, help="y tiles")
    ap.add_argument("--nz", type=int, default=256)
    ap.add_argument("--ny", type=int, default=256)
    ap.add_argument("--nx", type=int, default=1024, help="GLOBAL x")
    ap.add_argument("--nt", type=int, default=200)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")

    px = args.px or world
    py = args.py
    assert px * py == world, f"px*py={px * py} != world={world}"
    nz, ny, nx, nt = args.nz, args.ny, args.nx, args.nt
    shape = (nz, ny, nx)

    vp = np.full(shape, 2500.0, dtype=np.float32)
    src = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
    rec = np.array([[[ix, ny // 2, 2] for ix in range(2, nx - 2, 5)]], dtype=np.int32)
    wav = ricker(nt, DT)

    topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
    prop = PropTorch(Acoustic3D(spatial_order=args.so, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=shape, dh=10.0, dt=DT, nt=nt,
                     abcn=args.abcn, source_type=["h1"], receiver_type=["h1"], dev=dev,
                     free_surface=False)
    ddp = ModelParallel(prop, topo)

    def one():
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ddp.forward(wav, src, rec, models=[vp])
        torch.cuda.synchronize()
        dist.barrier()
        return time.perf_counter() - t0

    # warm-up (capture + clock ramp): >= 1.6 s wall
    tw = time.perf_counter()
    one()
    while time.perf_counter() - tw < 1.6:
        one()

    times = [one() for _ in range(args.repeats)]
    best = min(times)
    cv = float(np.std(times) / np.mean(times))
    peak = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        print(f"DDP_TIMING global={shape} px={px} py={py} nt={nt} "
              f"best={best:.3f}s per_step={best / nt * 1e3:.3f}ms cv={cv:.3f} "
              f"peak_mem={peak:.2f}GB times={['%.3f' % t for t in times]}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
