"""A/B micro-bench: elastic DD forward with batched halo (FastHaloGroup) vs
per-field exchange (the pre-aggregation behavior), back-to-back on the same hot
GPUs to isolate the comm-aggregation win.

torchrun --standalone --nproc-per-node=<P> test/dd_agg_bench.py
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Elastic  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1e6):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg ** 2) * np.exp(-(arg ** 2))).astype(np.float32)


def per_field_group(self, halo, tensors):
    """OLD behavior: one exchange (batch_isend_irecv + wait) per field."""
    if halo is not None:
        for t in tensors:
            halo.exchange(self._halo_view(t))


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")

    so, abcn = 4, 10
    for (nz, nxp, nt) in [(64, 64, 300), (128, 128, 300)]:
        nx = nxp * world
        shape = (nz, nx)
        g = np.linspace(0, 1, nz * nx, dtype=np.float32).reshape(shape)
        models = [2200.0 + 400.0 * g, 1200.0 + 200.0 * g, 2000.0 + 100.0 * g]
        wav = ricker(nt, DT)
        src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 4)]], dtype=np.int32)
        topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
        prop = PropTorch(Elastic(spatial_order=so, device=dev, backend="torch"),
                         backend="torch", impl="c", shape=shape, dh=10.0, dt=DT, nt=nt,
                         abcn=abcn, source_type=["sxx", "szz"],
                         receiver_type=["vx", "vz"], dev=dev)
        ddp = ModelParallel(prop, topo)

        def timeit(reps):
            torch.cuda.synchronize(); dist.barrier(); t0 = time.time()
            for _ in range(reps):
                ddp.forward(wav, src, rec, models=None)
            torch.cuda.synchronize(); dist.barrier()
            return (time.time() - t0) / reps * 1e3  # ms/forward

        ddp.forward(wav, src, rec, models=models)        # capture
        for _ in range(4):                               # warm up clocks
            ddp.forward(wav, src, rec, models=None)
        torch.cuda.synchronize(); dist.barrier()

        new_ms = timeit(8)                               # batched FastHaloGroup
        ddp._exchange_group = types.MethodType(per_field_group, ddp)
        old_ms = timeit(8)                               # per-field exchange
        ddp._exchange_group = types.MethodType(ModelParallel._exchange_group, ddp)

        if rank == 0:
            nphys = ddp._nphys
            print(f"[rank0] elastic2d tile({nz}x{nxp}) px{world} nt{nt} nphys={nphys}: "
                  f"per-field={old_ms:.2f} ms  batched={new_ms:.2f} ms  "
                  f"speedup={old_ms / new_ms:.3f}x")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
