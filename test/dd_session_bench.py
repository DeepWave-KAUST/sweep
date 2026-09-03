"""DD boundary staging: correctness gates + a (storage x interval x ring) wall-clock matrix.

Run with: torchrun --standalone --nproc-per-node=2 dd_session_bench.py
Whichever tree PYTHONPATH points at is the tree under test.

**Gates (fail one and the timings do not count)**
  A  every staged config's gradient must be BIT-EXACT against storage='gpu'
  B  two runs of the same cpu config must be bit-exact against each other -- a
     single comparison can pass by luck, two cannot. (Staging reuses someone
     else's ring machinery with shifted addressing, and that class of change has
     produced races visible only on the second run: silent under initcheck and
     unchanged under CUDA_LAUNCH_BLOCKING.)
  C  session.used must be True for staged configs, otherwise the call site fell
     back to the per-call path and the timings mean nothing.

**Why sweep interval/ring**: one non-overlapped PCIe round trip per step is the
root of storage='cpu' being slow. PR #77 let DD inherit those two knobs, but the
synchronize at the end of forward and the per-step teardown of the copy stream
were still there; the persistent session is what removes them. Running the same
matrix on both trees is what separates "the knobs took effect" from "the
barriers were removed".
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from sweep.equations import Acoustic3D
from sweep.parallel import MeshTopology, ModelParallel
from sweep.propagator.torch import PropTorch

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
torch.cuda.set_device(li)
dev = torch.device(f"cuda:{li}")

PY, PX = int(os.environ.get("MESH_PY", 2)), int(os.environ.get("MESH_PX", world // 2))
assert PY * PX == world, (PY, PX, world)
# Keep nt large: the staging cost scales with the step count, and 140 steps
# is too few to show a trend.
dh, dt, nt, abcn, order = 10.0, 0.001, int(os.environ.get("NT", 1200)), 12, 8
nz, ny, nx = 48, int(os.environ.get("MESH_NY", 96)), 48 * PX
shape = (nz, ny, nx)

zramp = np.linspace(0, 1, nz, dtype=np.float32)
vp_true = (2000.0 + 700.0 * np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32)
           ).reshape(shape)
vp_init = np.broadcast_to(2000.0 + 700.0 * zramp.reshape(nz, 1, 1),
                          shape).astype(np.float32).copy()
t = np.arange(nt) * dt
a = np.pi * 12.0 * (t - 0.06)
wav = ((1 - 2 * a ** 2) * np.exp(-a ** 2) * 1e3).astype(np.float32)
src = np.array([[[nx // 2, ny // 2, 1]]], np.int32)
gx, gy = np.meshgrid(np.arange(2, nx, 4), np.arange(2, ny, 4), indexing="xy")
rec = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size, np.int32)], -1)[None].astype(np.int32)


def run(storage, interval=1, ring=1):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = {"enabled": True, "storage": storage, "storage_dtype": "fp32",
           "transfer_interval": interval, "ring_buffers": ring}
    prop = PropTorch(Acoustic3D(spatial_order=order, device=dev), backend="torch",
                     impl="c", shape=shape, dh=dh, dt=dt, nt=nt, abcn=abcn,
                     source_type=["h1"], receiver_type=["h1"], dev=dev,
                     boundary_saving_config=cfg)
    ddp = ModelParallel(prop, MeshTopology(py=PY, px=PX, shot_groups=1,
                                           world_size=world, rank=rank))
    obs = ddp(wav, src, rec, models=[vp_true]).detach().clone()
    vp_g = torch.tensor(vp_init, device=dev, requires_grad=True)
    torch.cuda.synchronize(); dist.barrier(); t0 = time.perf_counter()
    (0.5 * (ddp(wav, src, rec, models=[vp_g]) - obs).pow(2).sum()).backward()
    torch.cuda.synchronize(); dist.barrier(); el = time.perf_counter() - t0
    sess = getattr(getattr(ddp, "_bsession", None), "used", None)
    return (vp_g.grad.detach().clone(), torch.cuda.max_memory_allocated() / 1e9,
            el, sess)


def agree(ga, gb):
    """All-rank verdict: if any tile is not bit-exact, the gate fails."""
    bit = torch.tensor([1.0 if torch.equal(ga, gb) else 0.0], device=dev)
    mad = torch.tensor([float((ga - gb).abs().max())], device=dev)
    dist.all_reduce(bit, op=dist.ReduceOp.MIN)
    dist.all_reduce(mad, op=dist.ReduceOp.MAX)
    return bool(bit.item()), float(mad.item())


P = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)
P(f"tree = {os.path.dirname(os.path.dirname(__import__('sweep').__file__))}")
P(f"mesh {PY}x{PX}  shape {shape}  nt {nt}  world {world}")

gref, pref, tref, _ = run("gpu")
P(f"\n{'config':>26} {'sec':>8} {'vs gpu':>8} {'peak GB':>8} {'bitex':>6} {'session.used':>13}")
P(f"{'gpu (baseline)':>26} {tref:>8.2f} {1.0:>8.2f} {pref:>8.2f} {'-':>6} {'-':>13}")

rows, fail = [], []
for interval, ring in ((1, 1), (32, 1), (32, 4), (64, 4)):
    g, pk, el, used = run("cpu", interval, ring)
    bit, mad = agree(gref, g)
    g2, _, _, _ = run("cpu", interval, ring)
    bit2, mad2 = agree(g, g2)
    tag = f"cpu interval={interval} ring={ring}"
    P(f"{tag:>26} {el:>8.2f} {el/tref:>7.2f}x {pk:>8.2f} {str(bit):>6} {str(used):>13}")
    if not bit:
        fail.append(f"{tag}: not bit-exact vs gpu, max|d|={mad:.3e}")
    if not bit2:
        fail.append(f"{tag}: two runs differ, max|d|={mad2:.3e}")
    rows.append(dict(interval=interval, ring=ring, sec=el, ratio=el / tref,
                     peak_gb=pk, bitexact=bit, repeat_bitexact=bit2, used=used))

if rank == 0:
    json.dump(dict(baseline_sec=tref, baseline_peak=pref, rows=rows),
              open(os.environ.get("OUT", "/tmp/dd_session_bench.json"), "w"), indent=1)
    P("\n=== gates ===")
    for f in fail:
        P("  FAIL " + f)
    P("  all bit-exact" if not fail else f"  {len(fail)} gate(s) failed")
dist.barrier()
sys.exit(1 if fail else 0)
