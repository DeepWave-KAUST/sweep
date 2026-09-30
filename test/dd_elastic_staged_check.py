"""Elastic 3-D domain decomposition with the boundary ring staged to the host.

The staggered skeleton refused cpu/disk staging for every DD and stepped
backward_bs ("unsupported in v1"), which capped the largest elastic 3-D tile at
whatever boundary ring fits in VRAM. Two things actually blocked it: the
skeleton built its own copy stream per call instead of taking the Python-owned
BoundarySession (so under DD, where each time step is a separate entry, the
stream and its events were destroyed and rebuilt every step and no transfer
could stay in flight), and it primed the TAIL chunk on every segment instead of
its own.

    torchrun --standalone --nproc-per-node=2 dd_elastic_staged_check.py

Gate: the staged gradient must be BIT-EXACT against gpu-direct, per model, on
every rank. Timings are reported but are not the gate.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from sweep.equations import Elastic3D
from sweep.parallel import MeshTopology, ModelParallel
from sweep.propagator.torch import PropTorch

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count()))
dev = torch.device(f"cuda:{torch.cuda.current_device()}")

PY_, PX = int(os.environ.get("MESH_PY", 1)), int(os.environ.get("MESH_PX", world))
assert PY_ * PX == world, (PY_, PX, world)
nz, ny = int(os.environ.get("NZ", 96)), int(os.environ.get("NY", 96))
nx = int(os.environ.get("NX_PER", 48)) * PX
shape, nt = (nz, ny, nx), int(os.environ.get("NT", 240))
dh, dt, abcn, order = 10.0, 4e-4, 8, 4

zr = np.linspace(0, 1, nz, dtype=np.float32).reshape(nz, 1, 1)
vp = np.broadcast_to(3000.0 + 600.0 * zr, shape).astype(np.float32).copy()
vs = (vp / 1.73).astype(np.float32)
rho = (1800.0 + 300.0 * np.broadcast_to(zr, shape)).astype(np.float32)
t = np.arange(nt) * dt - 0.02
a = np.pi * 12.0 * t
wav = ((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e6).astype(np.float32)
src = np.array([[[nx // 2, ny // 2, 4]]], np.int32)
gx, gy = np.meshgrid(np.arange(2, nx, 6), np.arange(2, ny, 6), indexing="xy")
rec = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 3, np.int32)], -1)[None].astype(np.int32)

P = (lambda *a_: print(*a_, flush=True)) if rank == 0 else (lambda *a_: None)
P(f"mesh {PY_}x{PX}  shape {shape}  nt {nt}  world {world}")


def run(storage, interval=1, ring=1):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = {"enabled": True, "storage": storage, "storage_dtype": "fp32"}
    if storage != "gpu":
        cfg.update({"transfer_interval": interval, "ring_buffers": ring,
                    "pinned_memory": True})
    prop = PropTorch(Elastic3D(spatial_order=order, device=dev), backend="torch",
                     impl="c", shape=shape, dh=dh, dt=dt, nt=nt, abcn=abcn, B=1,
                     dev=dev, pml_type="cpmls",
                     source_type=["sxx", "syy", "szz"],
                     receiver_type=["vx", "vy", "vz"],
                     boundary_saving_config=cfg)
    ddp = ModelParallel(prop, MeshTopology(py=PY_, px=PX, shot_groups=1,
                                           world_size=world, rank=rank))
    obs = ddp(wav, src, rec, models=[vp, vs, rho]).detach().clone()
    ms = [torch.tensor(m, device=dev, requires_grad=True) for m in (vp, vs, rho)]
    torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
    (0.5 * (ddp(wav, src, rec, models=ms) - obs).pow(2).sum()).backward()
    torch.cuda.synchronize(); dist.barrier(); el = time.perf_counter() - t0
    return [m.grad.detach().clone() for m in ms], torch.cuda.max_memory_allocated() / 1e9, el


def agree(ga, gb):
    """All-rank verdict: one tile off is a failure."""
    bit = torch.tensor([1.0 if all(torch.equal(a_, b_) for a_, b_ in zip(ga, gb)) else 0.0], device=dev)
    mad = torch.tensor([max(float((a_ - b_).abs().max()) for a_, b_ in zip(ga, gb))], device=dev)
    dist.all_reduce(bit, op=dist.ReduceOp.MIN); dist.all_reduce(mad, op=dist.ReduceOp.MAX)
    return bool(bit.item()), float(mad.item())


gref, pref, tref = run("gpu")
P(f"\n{'config':>26} {'sec':>8} {'vs gpu':>8} {'peak GB':>9} {'bitexact':>9}")
P(f"{'gpu (baseline)':>26} {tref:>8.2f} {1.0:>8.2f} {pref:>9.2f} {'-':>9}")

fail = []
for interval, ring in ((1, 1), (8, 2), (32, 4)):
    try:
        g, pk, el = run("cpu", interval, ring)
    except Exception as exc:
        fail.append(f"cpu {interval}/{ring}: {type(exc).__name__}: {str(exc)[:140]}")
        P(f"{'cpu interval=%d ring=%d' % (interval, ring):>26}   RAISED  {str(exc)[:60]}")
        continue
    bit, mad = agree(gref, g)
    P(f"{'cpu interval=%d ring=%d' % (interval, ring):>26} {el:>8.2f} {el/tref:>7.2f}x {pk:>9.2f} {str(bit):>9}")
    if not bit:
        fail.append(f"cpu {interval}/{ring}: not bit-exact vs gpu, max|d|={mad:.3e}")

if rank == 0:
    P("\n=== gate ===")
    for f in fail:
        P("  FAIL " + f)
    P("  all bit-exact" if not fail else f"  {len(fail)} gate(s) failed")
dist.barrier()
sys.exit(1 if fail else 0)
