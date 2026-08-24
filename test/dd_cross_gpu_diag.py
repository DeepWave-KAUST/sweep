"""Cross-GPU determinism + exchange-cost diagnosis (torchrun, N ranks).

Part 1: every rank runs the IDENTICAL single-domain problem on its own GPU;
rank 0 compares outputs bitwise across ranks. If this fails, same-model
GPUs do not reproduce bitwise and the DD acceptance bar must be ulp-level
for cross-GPU comparisons (single-GPU manual harness stays bitwise).

Part 2: micro-benchmark of the per-step halo exchange on the production
view size, decomposed: full exchange_halos() vs bare batch_isend_irecv vs
contiguous staging only.

torchrun --standalone --nproc-per-node=4 test/dd_cross_gpu_diag.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sweep.equations import Acoustic  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallelMesh, exchange_halos  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBindingRunner,
    acoustic_psi_pairs,
)

NZ, NX, NT, DT, SO, ABCN = 48, 112, 120, 0.0015, 4, 8
M = SO // 2


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def capture(prop):
    cap = {}
    impl = prop._backend_impl
    orig = impl.forward_func

    def wrapper(p):
        out = orig(p)
        cap["params"], cap["raw_out"] = p, out
        return out

    impl.forward_func = wrapper
    cap["func"] = orig
    return cap


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    # ---------- Part 1: identical problem on every GPU ----------
    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp = np.broadcast_to(vp, (NZ, NX)).copy()
    vp[NZ // 3:NZ // 2, NX // 3:2 * NX // 3] += 180.0
    wavelet = ricker(NT, DT)
    eq = Acoustic(spatial_order=SO, device=dev, backend="torch")
    prop = PropTorch(
        eq, backend="torch", impl="c", shape=(NZ, NX), dev=dev, dh=10.0,
        dt=DT, source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
        free_surface=False, pml_type="cpmlr", nt=NT, B=1, use_ckpt=False,
        boundary_saving_config={"enabled": False},
    )
    cap = capture(prop)
    src = np.array([[[NX // 2, NZ // 4]]], dtype=np.int32)
    rec = np.array([[[gx, 2] for gx in range(2, NX - 2, 6)]], dtype=np.int32)
    with torch.no_grad():
        prop(wavelet, src, rec, models=[torch.tensor(vp, device=dev)])
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields) or [torch.zeros_like(p.models[0]) for _ in range(9)]
    for t in L:
        t.zero_()
    record = torch.zeros_like(cap["raw_out"][2])
    p.record_out = record
    r = SteppedBindingRunner(func, p, L, acoustic_psi_pairs(2))
    with torch.no_grad():
        r.run_to(NT)
    torch.cuda.synchronize()

    payload = (record.cpu(), r.u_now.cpu())
    gathered = [None] * world
    dist.gather_object(payload, gathered if rank == 0 else None, dst=0)
    if rank == 0:
        rec0, u0 = gathered[0]
        for k in range(1, world):
            reck, uk = gathered[k]
            br, bu = torch.equal(reck, rec0), torch.equal(uk, u0)
            mr = (reck - rec0).abs().max().item()
            mu = (uk - u0).abs().max().item()
            print(f"[diag] rank{k} vs rank0 (identical problem): "
                  f"record bit={br} max|d|={mr:.3e}; u_now bit={bu} max|d|={mu:.3e}")
        print("[diag] CROSS_GPU_DETERMINISM:",
              "BITWISE" if all(torch.equal(g[0], rec0) and torch.equal(g[1], u0)
                               for g in gathered[1:]) else "ULP_DIFFERS")

    # ---------- Part 2: exchange micro-benchmark ----------
    topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, world))
    nxp = NX // world
    pad = ABCN + M
    view = r.u_now[..., pad - M: pad + nxp + M]

    def timeit(fn, iters):
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        return (time.perf_counter() - t0) / iters * 1e6

    # (a) full exchange_halos
    us_full = timeit(lambda: exchange_halos([view], mesh, M, ("x",)), 500)
    # (b) contiguous staging only
    us_stage = timeit(lambda: (view[..., M:2 * M].contiguous(),
                               view[..., -2 * M:-M].contiguous()), 500)
    # (c) bare paired sendrecv of the staged buffer
    sbuf = view[..., M:2 * M].contiguous()
    rbuf = torch.empty_like(sbuf)
    left = topo.neighbour_rank("x", -1)
    right = topo.neighbour_rank("x", +1)

    def bare():
        ops = []
        if left is not None:
            ops += [dist.P2POp(dist.isend, sbuf, left, group=mesh.model_pg),
                    dist.P2POp(dist.irecv, rbuf, left, group=mesh.model_pg)]
        if right is not None:
            ops += [dist.P2POp(dist.isend, sbuf, right, group=mesh.model_pg),
                    dist.P2POp(dist.irecv, rbuf, right, group=mesh.model_pg)]
        for q in dist.batch_isend_irecv(ops):
            q.wait()

    us_bare = timeit(bare, 500)

    # (d) FastHaloSet (preallocated, no autograd)
    from sweep.parallel.fast_halo import FastHaloSet
    fset = FastHaloSet(mesh, M, ("x",))
    fset.exchange(view)  # build cache
    us_fast = timeit(lambda: fset.exchange(view), 500)
    if rank == 0:
        print(f"[diag] exchange micro-bench (view {tuple(view.shape)}): "
              f"exchange_halos={us_full:.1f}us fast_halo={us_fast:.1f}us "
              f"bare_p2p={us_bare:.1f}us staging={us_stage:.1f}us")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
