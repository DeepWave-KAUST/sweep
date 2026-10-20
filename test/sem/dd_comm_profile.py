"""How much of a DD step is interface communication? (torchrun launcher)

    torchrun --nproc_per_node=P test/sem/dd_comm_profile.py [--nz .. --nx ..]

Overlapping comm with compute is only worth building if comm is a real share
of the step, and that share shrinks as the strip grows (comm is O(nz) per
cut, compute is O(nz*nx_local)). So this measures, per rank:

  * total wall per forward+backward iteration,
  * the exchange time alone (same calls, timed with CUDA events),

at a couple of problem sizes, and prints the ratio. Run it before writing
any overlap machinery -- if comm is a few percent at production sizes, the
FD-style iphase split buys nothing here and the honest answer is to say so.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import torch.distributed as dist

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem import dd as ddmod
from sweep.sem.dd import SEMModelParallel, StripTopology
from sweep.sem.equations import AcousticPotentialSEM2D

NO, DH, F0 = 4, 25.0, 8.0
DT = 5e-4


def main():
    ap = argparse.ArgumentParser()
    # the models; the 3-element layer outside them (sides and bottom) makes
    # the meshes 49x97, 97x193, 193x385 -- 24, 48, 96 elements wide
    ap.add_argument("--sizes", default="37x73,85x169,181x361")
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    for size in args.sizes.split(","):
        nz, nx = (int(v) for v in size.split("x"))
        prop = S.PropSEM(AcousticPotentialSEM2D, shape=(nz, nx), dh=DH, dt=DT,
                         nt=args.nt, n_order=NO, free_surface=("top",),
                         abcn=3 * NO, memory=BoundarySaving(), f_dominant=F0,
                         dev="cuda", impl="c")
        torch.manual_seed(0)
        vp = (2000.0 + 300.0 * torch.rand(nz, nx)).cuda().requires_grad_(True)
        t = np.arange(args.nt) * DT
        a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
        wav = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                           device="cuda")
        eq, _ = prop._resolve_model_batch([vp.detach()])
        nxg = eq.mesh.nex * NO + 1
        src = 2 * NO * nxg + nxg // 2
        rec = (torch.arange(4, nxg - 4, 4) + 1 * nxg).cuda()

        # instrument the exchange: accumulate its own CUDA-event time
        comm_ms = [0.0]
        orig_many = ddmod._DDExchange.many
        orig_call = ddmod._DDExchange.__call__
        orig_block = ddmod._DDExchange.block

        def timed(fn):
            def wrap(self, *a, **k):
                s0, s1 = (torch.cuda.Event(enable_timing=True),
                          torch.cuda.Event(enable_timing=True))
                s0.record()
                out = fn(self, *a, **k)
                s1.record()
                s1.synchronize()
                comm_ms[0] += s0.elapsed_time(s1)
                return out
            return wrap

        ddmod._DDExchange.many = timed(orig_many)
        ddmod._DDExchange.__call__ = timed(orig_call)
        ddmod._DDExchange.block = timed(orig_block)

        ddp = SEMModelParallel(prop, StripTopology(world, rank=rank))

        def one():
            out = ddp(wav, src, rec, [vp])        # boundary saving: the prop's
            (out ** 2).sum().backward()
            vp.grad = None

        one()                                     # warm up
        comm_ms[0] = 0.0
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            one()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / args.iters * 1e3
        comm = comm_ms[0] / args.iters

        ddmod._DDExchange.many = orig_many
        ddmod._DDExchange.__call__ = orig_call
        ddmod._DDExchange.block = orig_block

        stats = torch.tensor([wall, comm], device="cuda")
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        if rank == 0:
            w, c = stats.tolist()
            print(f"[dd-comm] {nz}x{nx} px={world} nt={args.nt}: "
                  f"wall {w:8.1f} ms/iter | exchange {c:7.1f} ms "
                  f"({100 * c / w:4.1f}%)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
