"""Validate E6: shot-parallel gradient all_reduce across shot groups.

torchrun --standalone --nproc-per-node=4 test/dd_shotpar_check.py

Layout: world=4 = shot_groups(2) x px(2), py=1. shot_group 0 (ranks 0,1)
runs shot A; shot_group 1 (ranks 2,3) runs shot B; within each group the model
is x-decomposed over 2 tiles. DDPropagator.gradient() all_reduces each tile's
gradient across the shot process group, so after it rank with (xi) holds
grad_tile(A) + grad_tile(B). Assemble over xi on rank 0 and compare to the
single-domain reference grad_A + grad_B (dd_api_check.reference per shot).
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
sys.path.insert(0, str(REPO / "test"))

from sweep.equations import Acoustic  # noqa: E402
from sweep.parallel import DDPropagator, MeshTopology  # noqa: E402
from dd_api_check import reference, ricker  # noqa: E402

DT = 0.0015
REL_TOL = 1e-5


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4, "this check expects nproc-per-node=4 (2 shot_groups x px2)"
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")

    so, abcn, nt = 4, 10, 60
    nz, nxp, px, py, sgN = 48, 28, 2, 1, 2
    nx = nxp * px
    shape = (nz, nx)
    grid = np.linspace(0, 1, nz * nx, dtype=np.float32).reshape(shape)
    vp = 1800.0 + 600.0 * grid
    wav = ricker(nt, DT)
    rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 5)]], dtype=np.int32)
    srcA = np.array([[[nx // 4, nz // 4]]], dtype=np.int32)
    srcB = np.array([[[3 * nx // 4, nz // 4]]], dtype=np.int32)

    shot_group = rank // (px * py)
    src = srcA if shot_group == 0 else srcB

    topo = MeshTopology(py=py, px=px, shot_groups=sgN, world_size=world, rank=rank)
    ddp = DDPropagator(Acoustic(spatial_order=so, device=dev, backend="torch"),
                       shape, dh=10.0, dt=DT, nt=nt, abcn=abcn, spatial_order=so,
                       source_type=["h1"], receiver_type=["h1"],
                       model_parallel=topo, dev=dev)
    rec_tile = ddp.forward(wav, src, rec, models=[vp])
    grads_tile = ddp.gradient(rec_tile)            # E6 all_reduces across shot_pg

    payload = (ddp.x0, ddp.nxp, [g.cpu() for g in grads_tile])
    gathered = [None] * world
    dist.gather_object(payload, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        M = so // 2
        pad = abcn + M
        # single-domain references: grad for shot A and shot B (adj = own record)
        _, gA = reference("acoustic", 2, shape, so, abcn, False, nt, dev, [vp], srcA, rec, wav)
        _, gB = reference("acoustic", 2, shape, so, abcn, False, nt, dev, [vp], srcB, rec, wav)
        ref = [(a + b).cpu() for a, b in zip(gA, gB)]    # shot-summed reference

        # assemble the DD shot-summed gradient from the xi tiles (use ranks of
        # shot_group 0; each already holds A+B after all_reduce)
        worst = 2
        for k in range(len(ref)):
            full = torch.zeros_like(ref[k][..., :nz, :])  # (.., nz, nx) interior
            for r in range(px):                            # ranks 0,1 = sg0 tiles
                x0, nxp_r, g_list = gathered[r]
                full[..., :, x0:x0 + nxp_r] = g_list[k]
            want = ref[k][..., pad:pad + nz, pad:pad + nx]
            bit = torch.equal(full, want)
            mad = (full - want).abs().max().item()
            rel = mad / (want.abs().max().item() + 1e-30)
            print(f"[rank0] grad[{k}] shot-summed bit={bit} max|d|={mad:.3e} rel={rel:.3e}")
            worst = min(worst, 2 if bit else (1 if rel < REL_TOL else 0))
        print("SHOTPAR_CHECK:", {2: "PASS", 1: "PASS_TOL", 0: "FAIL"}[worst])
        if worst == 0:
            sys.exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
