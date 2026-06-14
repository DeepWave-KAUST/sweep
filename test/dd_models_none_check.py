"""Validate DDPropagator.forward(models=None) model-reuse fast path.

torchrun --standalone --nproc-per-node=<P> test/dd_models_none_check.py

Claim: after a forward with models=[vp], the runtime model buffers hold the
edge-padded + NCCL-halo-exchanged model, so a later forward(models=None) for a
DIFFERENT shot must be BITWISE identical to passing models=[vp] again (which
re-pads + re-exchanges the same vp). Also checks the fail-loud guard.
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

from sweep.equations import Acoustic  # noqa: E402
from sweep.parallel import DDPropagator, MeshTopology  # noqa: E402

DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg ** 2) * np.exp(-(arg ** 2))).astype(np.float32)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")

    so, abcn, nt = 4, 10, 60
    nz, nxp = 48, 28
    nx = nxp * world
    shape = (nz, nx)
    grid = np.linspace(0, 1, nz * nx, dtype=np.float32).reshape(shape)
    vp = 1800.0 + 600.0 * grid
    wav = ricker(nt, DT)
    rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 5)]], dtype=np.int32)
    srcA = np.array([[[nx // 3, nz // 4]]], dtype=np.int32)
    srcB = np.array([[[2 * nx // 3, nz // 4]]], dtype=np.int32)

    def mk():
        return DDPropagator(Acoustic(spatial_order=so, device=dev, backend="torch"),
                            shape, dh=10.0, dt=DT, nt=nt, abcn=abcn, spatial_order=so,
                            source_type=["h1"], receiver_type=["h1"],
                            model_parallel=MeshTopology(py=1, px=world, shot_groups=1,
                                                        world_size=world, rank=rank),
                            dev=dev)

    # Path 1: reuse — set model on shot A, reuse for shot B
    d1 = mk()
    d1.forward(wav, srcA, rec, models=[vp])
    recB_reuse = d1.forward(wav, srcB, rec, models=None).detach().clone()
    gB_reuse = [g.clone() for g in d1.gradient(recB_reuse)]

    # Path 2: explicit — pass the same vp again for shot B
    d2 = mk()
    d2.forward(wav, srcA, rec, models=[vp])
    recB_expl = d2.forward(wav, srcB, rec, models=[vp]).detach().clone()
    gB_expl = [g.clone() for g in d2.gradient(recB_expl)]

    rec_bit = torch.equal(recB_reuse, recB_expl)
    g_bit = all(torch.equal(a, b) for a, b in zip(gB_reuse, gB_expl))
    rec_d = (recB_reuse - recB_expl).abs().max().item()
    g_d = max((a - b).abs().max().item() for a, b in zip(gB_reuse, gB_expl)) if gB_reuse else 0.0

    # Fail-loud guard: models=None before any forward must raise
    guard_ok = False
    try:
        mk().forward(wav, srcA, rec, models=None)
    except RuntimeError:
        guard_ok = True

    if rank == 0:
        print(f"[rank0] reuse-vs-explicit record bit={rec_bit} max|d|={rec_d:.3e}")
        print(f"[rank0] reuse-vs-explicit grad   bit={g_bit} max|d|={g_d:.3e}")
        print(f"[rank0] models=None pre-capture guard raised: {guard_ok}")
        ok = rec_bit and g_bit and guard_ok
        print("MODELS_NONE_CHECK:", "PASS" if ok else "FAIL")
        if not ok:
            sys.exit(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
