"""What does ONE interface exchange actually cost? (torchrun launcher)

    torchrun --nproc_per_node=P test/sem/dd_p2p_floor.py

The DD profile says each exchange costs ~270 us for a ~400-byte interface
column -- 30x NCCL's NVLink latency floor. Before choosing a fix, separate
the causes:

  bare      : batch_isend_irecv + wait, nothing else      (torch P2P floor)
  packed    : + the pack/unpack index kernels DD does     (our overhead)
  allreduce : one small all_reduce instead of the P2P pair (alternative)
  async     : issue the P2P, do fake compute, then wait   (overlap headroom)

If `bare` is already ~270 us, the calling mechanism is the wall and overlap
is the only lever. If `bare` is ~15 us, the wall is what DD wraps around it.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import torch
import torch.distributed as dist

N = 101            # interface column length (nz of the 49x97 case)
K = 6              # fields per exchange (acoustic forward)
REP = 300


def timeit(fn, rep=REP):
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e6      # us


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    peer = 1 - rank if world == 2 else (rank + 1) % world
    buf = torch.randn(K * N, device="cuda")
    recv = torch.empty_like(buf)
    field = torch.randn(20000, device="cuda")
    idx = torch.arange(N, device="cuda") * 7

    def bare():
        for req in dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, buf, peer),
                 dist.P2POp(dist.irecv, recv, peer)]):
            req.wait()

    def packed():
        s = torch.empty(K * N, device="cuda")
        for k in range(K):
            s[k * N:(k + 1) * N] = field[idx]
        r = torch.empty_like(s)
        for req in dist.batch_isend_irecv(
                [dist.P2POp(dist.isend, s, peer),
                 dist.P2POp(dist.irecv, r, peer)]):
            req.wait()
        tot = s + r
        for k in range(K):
            field.index_copy_(0, idx, tot[k * N:(k + 1) * N])

    def allreduce():
        dist.all_reduce(buf)

    big = torch.randn(2000, 2000, device="cuda")

    def async_overlap():
        reqs = dist.batch_isend_irecv(
            [dist.P2POp(dist.isend, buf, peer),
             dist.P2POp(dist.irecv, recv, peer)])
        big.mul_(1.0000001)          # stand-in for interior compute
        for r in reqs:
            r.wait()

    def compute_only():
        big.mul_(1.0000001)

    res = {nm: timeit(fn) for nm, fn in
           (("bare", bare), ("packed", packed), ("allreduce", allreduce),
            ("compute_only", compute_only), ("async+compute", async_overlap))}
    stats = torch.tensor(list(res.values()), device="cuda")
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[p2p-floor] world={world}  {K}x{N} floats "
              f"({4 * K * N} B)", flush=True)
        for nm, v in zip(res, stats.tolist()):
            print(f"[p2p-floor]   {nm:14s} {v:8.1f} us", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
