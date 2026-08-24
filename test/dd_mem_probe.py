"""Per-GPU + host memory probe for ModelParallel DD — boundary-storage aware.

Extends dd_mem_check with explicit MemoryOptions so we can probe the actual
iFWI config (boundary saving, storage gpu|cpu, dtype fp32|fp16|bf16|int8) and
report per-card peak ALLOCATED + RESERVED GPU plus host RSS (for cpu boundary).

torchrun --standalone --nproc-per-node=8 test/dd_mem_probe.py \
    --ndim 3 --px 4 --py 2 --nz 602 --nyp 857 --nxp 200 --nt 5500 \
    --boundary-storage cpu --boundary-dtype fp32
"""
from __future__ import annotations
import argparse, os, sys, resource
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic3D, Acoustic  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator.options import MemoryOptions, BoundaryOptions  # noqa: E402

DT = 0.0005


def ricker(nt, dt, fm=10.0, delay=0.04):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=3)
    ap.add_argument("--px", type=int, default=0)
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--nz", type=int, default=192)
    ap.add_argument("--nyp", type=int, default=192)
    ap.add_argument("--nxp", type=int, default=192)
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--free-surface", action="store_true")
    ap.add_argument("--boundary-storage", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--boundary-dtype", default="fp32",
                    choices=("fp32", "fp16", "bf16", "int8"))
    ap.add_argument("--timing", action="store_true",
                    help="warmup fwd+bwd, then time a 2nd forward -> ms/step")
    ap.add_argument("--no-backward", action="store_true",
                    help="skip the (mem-heavy) backward; for pure strong-scaling timing")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li); dev = torch.device(f"cuda:{li}")
    px = args.px or world; py = args.py
    assert px * py == world, f"px*py={px*py} != world={world}"
    ndim, nt = args.ndim, args.nt

    nz, nyp, nxp = args.nz, args.nyp, args.nxp
    nx = nxp * px
    if ndim == 3:
        ny = nyp * py
        global_shape = (nz, ny, nx)
    else:
        global_shape = (nz, nx)

    topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
    eqc = Acoustic3D if ndim == 3 else Acoustic
    st, rt = ["h1"], ["h1"]
    mem = MemoryOptions(strategy="boundary",
                        boundary=BoundaryOptions(storage=args.boundary_storage,
                                                 storage_dtype=args.boundary_dtype))
    prop = PropTorch(eqc(spatial_order=args.so, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=global_shape, dh=10.0, dt=DT,
                     nt=nt, abcn=args.abcn, source_type=st, receiver_type=rt, dev=dev,
                     free_surface=args.free_surface, memory=mem)
    ddp = ModelParallel(prop, topo)

    ls = ddp.local_shape
    models = [np.full(ls, 2500.0, dtype=np.float32)]
    if ndim == 3:
        src = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, iy, 2]
                         for iy in range(2, ny - 2, max(1, ny // 16))
                         for ix in range(2, nx - 2, max(1, nx // 16))]], dtype=np.int32)
    else:
        src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, 2] for ix in range(2, nx - 2, max(1, nx // 64))]],
                       dtype=np.int32)
    wav = ricker(nt, DT)

    models_t = [torch.tensor(m, device=dev, requires_grad=True) for m in models]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ok, err, ms_step = True, "", float("nan")
    try:
        rec_tile = ddp.forward(wav, src, rec, models=models_t)
        if not args.no_backward:
            rec_tile.backward(gradient=rec_tile.detach())
        torch.cuda.synchronize()
        if args.timing:
            # warmup done above; time a 2nd forward (no grad) over nt steps.
            import time as _time
            with torch.no_grad():
                torch.cuda.synchronize(); _t0 = _time.perf_counter()
                _ = ddp.forward(wav, src, rec, models=[m.detach() for m in models_t])
                torch.cuda.synchronize(); _t1 = _time.perf_counter()
            ms_step = (_t1 - _t0) / nt * 1e3
    except RuntimeError as e:
        ok = False; err = str(e)[:120]
    alloc = torch.cuda.max_memory_allocated() / 2**30
    reserv = torch.cuda.max_memory_reserved() / 2**30
    host_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # GB (ru_maxrss in KB on linux)

    info = (rank, tuple(topo.coord), bool(ddp._owns_src), float(alloc),
            float(reserv), float(host_rss), ok, err, float(ms_step))
    gathered = [None] * world
    dist.gather_object(info, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        allocs = [g[3] for g in gathered]; reservs = [g[4] for g in gathered]
        hosts = [g[5] for g in gathered]; allok = all(g[6] for g in gathered)
        msteps = [g[8] for g in gathered if g[8] == g[8]]  # drop nan
        print(f"\n=== acoustic {ndim}D  px={px} py={py}  tile={ls}  nt={nt} "
              f"abcn={args.abcn} so={args.so}  boundary={args.boundary_storage}/"
              f"{args.boundary_dtype}  global={global_shape}  world={world} ===")
        for r, coord, owns_src, al, rv, hs, o, e, mst in sorted(gathered):
            edge = "src" if owns_src else "   "
            tag = "" if o else f"  OOM/ERR: {e}"
            print(f"  rank{r} coord{coord} {edge} "
                  f"alloc={al:6.2f} reserved={rv:6.2f} hostRSS={hs:6.1f} GB{tag}")
        line = (f"  --> GPU alloc max={max(allocs):.2f}  reserved max={max(reservs):.2f} GB/card | "
                f"host RSS sum~{sum(hosts):.0f} GB | "
                f"{'ALL-OK' if allok else 'SOME-FAILED'}")
        if msteps:
            line += f" | per-step={max(msteps):.2f} ms (world={world})"
        print(line)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
