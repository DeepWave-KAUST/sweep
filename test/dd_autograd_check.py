"""Ergonomics check for the autograd-transparent DD forward:
 (1) the record carries a grad_fn,
 (2) a plain loss.backward() populates the model tile's .grad,
 (3) the per-tile-leaf mode and the replicated-global-leaf mode agree (the
     global leaf's .grad, sliced to this tile, equals the tile-leaf .grad).
Correctness of the gradient vs a single-domain reference is covered by
dd_api_check.py (now also on the autograd path).

torchrun --standalone --nproc-per-node=2 test/dd_autograd_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic                  # noqa: E402
from sweep.parallel import DDPropagator, MeshTopology  # noqa: E402

DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.08):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")

    so, abcn, nt = 4, 10, 80
    nz, nxp = 48, 28
    nx = nxp * world
    grid = np.linspace(0, 1, nz * nx, dtype=np.float32).reshape(nz, nx)
    true_g = 1800.0 + 600.0 * grid
    init_g = np.broadcast_to(1800.0 + 600.0 * np.linspace(0, 1, nz, np.float32)[:, None],
                             (nz, nx)).copy()
    wav = ricker(nt, DT)
    src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
    rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 4)]], dtype=np.int32)

    topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    ddp = DDPropagator(Acoustic(spatial_order=so, device=dev, backend="torch"),
                       (nz, nx), dh=12.0, dt=DT, nt=nt, abcn=abcn, spatial_order=so,
                       source_type=["h1"], receiver_type=["h1"],
                       model_parallel=topo, dev=dev)
    x0 = ddp.x0

    obs = ddp.forward(wav, src, rec, models=[true_g]).clone()    # no-grad obs

    # (A) per-tile leaf -> loss.backward() -> vp_tile.grad
    vp_tile = torch.tensor(init_g[:, x0:x0 + nxp], device=dev, requires_grad=True)
    syn = ddp.forward(wav, src, rec, models=[vp_tile])
    has_graph = bool(syn.requires_grad and syn.grad_fn is not None)
    (0.5 * (syn - obs).pow(2).sum()).backward()
    g_tile = vp_tile.grad.detach().reshape(nz, nxp).clone()

    # (B) replicated global leaf -> grad is global-shaped, this tile filled
    vp_glob = torch.tensor(init_g, device=dev, requires_grad=True)
    syn2 = ddp.forward(wav, src, rec, models=[vp_glob])
    (0.5 * (syn2 - obs).pow(2).sum()).backward()
    glob_ok = (tuple(vp_glob.grad.shape) == (nz, nx))
    g_from_glob = vp_glob.grad[:, x0:x0 + nxp].detach().reshape(nz, nxp)

    modes_match = torch.equal(g_tile, g_from_glob)
    nonzero = g_tile.abs().max().item() > 0

    if rank == 0:
        print(f"[rank0] record has grad_fn: {has_graph}")
        print(f"[rank0] tile .grad nonzero: {nonzero}  |g|max={g_tile.abs().max().item():.3e}")
        print(f"[rank0] global-leaf grad shape==global: {glob_ok}; "
              f"tile==global-tile: {modes_match}")
        ok = bool(has_graph and nonzero and glob_ok and modes_match)
        print("AUTOGRAD_CHECK:", "PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
