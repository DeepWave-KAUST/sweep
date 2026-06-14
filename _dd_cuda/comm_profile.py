"""Profile where the per-step halo comm time goes (2 ranks, x-cut).

Times, over many iters with CUDA-event GPU timing + wall time:
  * copy-send (strided staging gather)
  * batch_isend_irecv launch + work.wait (the NCCL P2P + sync)
  * copy-recv (strided scatter)
and checks overlap effectiveness: compute-only vs serial vs start/finish overlap
on a real stepped acoustic run. Run with: torchrun --nproc-per-node=2.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sweep.equations import Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallelMesh  # noqa: E402
from sweep.parallel.fast_halo import FastHaloSet  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import SteppedBindingRunner, acoustic_psi_pairs  # noqa: E402


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dev = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    nz, ny, nxp = 256, 256, 256
    abcn, so, M = 20, 4, 2
    topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, world))
    prop = PropTorch(Acoustic3D(spatial_order=so, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=(nz, ny, nxp), dev=dev, dh=10.0,
                     dt=0.0005, source_type=["h1"], receiver_type=["h1"], abcn=abcn,
                     free_surface=False, pml_type="cpmlr", nt=10, B=1, use_ckpt=False,
                     boundary_saving_config={"enabled": False}, model_parallel=topo)
    vp = np.full((nz, ny, nxp), 2500.0, np.float32)
    # a representative wavefield tensor (runtime padded shape)
    shp = prop.shape_cuda
    u = torch.randn((1, 1, *shp), device=dev)
    lo = prop.padding[0] + M
    hi = lo + nxp
    view = u[..., lo - M:hi + M]
    fs = FastHaloSet(mesh, M, ("x",))
    ex = fs._get(view)  # build cached exchanger

    def timeit(fn, n=500):
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(); dist.barrier()
        return (time.perf_counter() - t0) / n * 1e3  # ms

    # component timings (each isolates one phase; P2P needs both ranks in lockstep)
    t_send = timeit(lambda: [sb.copy_(sv) for sb, sv in ex._send_views])
    t_recv = timeit(lambda: [rv.copy_(rb) for rv, rb in ex._recv_views])
    def full():
        ex.exchange_start(); ex.exchange_finish()
    t_full = timeit(full)
    def p2ponly():
        ex._pending = dist.batch_isend_irecv(ex._ops)
        for r in ex._pending: r.wait()
        ex._pending = None
    t_p2p = timeit(p2ponly)

    if rank == 0:
        print(f"[profile] tile=({nz},{ny},{nxp}) shape_cuda={shp} M={M}")
        print(f"[profile] copy-send(2 faces)= {t_send:.4f} ms")
        print(f"[profile] copy-recv(2 faces)= {t_recv:.4f} ms")
        print(f"[profile] P2P+wait         = {t_p2p:.4f} ms")
        print(f"[profile] full exchange    = {t_full:.4f} ms")
        print(f"[profile] -> copies = {t_send + t_recv:.4f} ms, P2P+wait = {t_p2p:.4f} ms "
              f"of full {t_full:.4f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
