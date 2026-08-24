"""DD boundary CPU-staging equivalence check (acoustic 2-D and 3-D).

    torchrun --standalone --nproc-per-node=<P> test/dd_offload_check.py
    env: DIM(2|3) MESH_PY MESH_PX MESH_NY BND_DTYPE(fp32|fp16|bf16|int8) TAIL

CPU staging only changes WHERE the boundary lives and WHEN it is copied, never
the arithmetic, so the DD gradient must be **bit-identical** to the gpu-direct
one in fp32 (the low-precision rings keep their own quantisation floor).
Four gates:

  A  storage='cpu' vs storage='gpu'                      bitwise
  B  the same with BoundaryOptions.tail_steps set        bitwise
  C  storage='cpu' run twice                             bitwise
  D  storage='disk' under DD                             refused, loudly

C is not redundant.  Staging reuses another path's ring machinery and shifts
the addressing under it, and that shape of change has already shipped a bug
that only a repeat could see: a borrowed kernel whose aux row stride collapsed
raced between threads while ``compute-sanitizer initcheck`` stayed silent and
``CUDA_LAUNCH_BLOCKING=1`` changed nothing.  A single comparison can pass by
luck; two runs cannot.
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "SWEEP_SRC", os.path.join(os.path.dirname(__file__), "..", "src")))

import numpy as np
import torch
import torch.distributed as dist

from sweep.equations import Acoustic, Acoustic3D
from sweep.parallel import MeshTopology, ModelParallel
from sweep.propagator.torch import PropTorch

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
torch.cuda.set_device(li)
dev = torch.device(f"cuda:{li}")

DIM = int(os.environ.get("DIM", 3))
PY = 1 if DIM == 2 else int(os.environ.get("MESH_PY", 2))
PX = int(os.environ.get("MESH_PX", world // PY))
NY = int(os.environ.get("MESH_NY", 48))
BND = os.environ.get("BND_DTYPE", "fp32")
TAIL = int(os.environ.get("TAIL", 60))
assert PY * PX == world, f"PY*PX={PY*PX} != world={world}"

dh, dt, nt, abcn, order = 10.0, 0.001, 140, 12, 8
nz, nx = 40, 40 * PX
ny = NY
shape = (nz, nx) if DIM == 2 else (nz, ny, nx)

zramp = np.linspace(0, 1, nz, dtype=np.float32)
vp_true = (2000.0 + 700.0 * np.linspace(0, 1, int(np.prod(shape)), dtype=np.float32)
           ).reshape(shape)
vp_init = np.broadcast_to(2000.0 + 700.0 * zramp.reshape((nz,) + (1,) * (DIM - 1)),
                          shape).astype(np.float32).copy()

t = np.arange(nt) * dt
a = np.pi * 12.0 * (t - 0.06)
wav = ((1 - 2 * a**2) * np.exp(-a**2) * 1e3).astype(np.float32)

if DIM == 2:
    src = np.array([[[nx // 2, 1]]], np.int32)
    rx = np.arange(2, nx, 4)
    rec = np.stack([rx, np.ones(rx.size, np.int32)], -1)[None].astype(np.int32)
else:
    src = np.array([[[nx // 2, ny // 2, 1]]], np.int32)
    gx, gy = np.meshgrid(np.arange(2, nx, 4), np.arange(2, ny, 4), indexing="xy")
    rec = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size, np.int32)], -1)[None].astype(np.int32)

EQ = Acoustic if DIM == 2 else Acoustic3D


def run(storage, tail=None):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = {"enabled": True, "storage": storage, "storage_dtype": BND}
    if tail:
        cfg["tail_steps"] = tail
    prop = PropTorch(EQ(spatial_order=order, device=dev), backend="torch", impl="c",
                     shape=shape, dh=dh, dt=dt, nt=nt, abcn=abcn,
                     source_type=["h1"], receiver_type=["h1"], dev=dev,
                     boundary_saving_config=cfg)
    ddp = ModelParallel(prop, MeshTopology(py=PY, px=PX, shot_groups=1,
                                           world_size=world, rank=rank))
    obs = ddp(wav, src, rec, models=[vp_true]).detach().clone()
    vp_g = torch.tensor(vp_init, device=dev, requires_grad=True)
    (0.5 * (ddp(wav, src, rec, models=[vp_g]) - obs).pow(2).sum()).backward()
    torch.cuda.synchronize()
    return vp_g.grad.detach().clone(), torch.cuda.max_memory_allocated() / 1e9


def compare(label, ga, gb, ref_peak=None, peak=None):
    """All-rank verdict: bitwise on every tile, worst deviation anywhere."""
    bit = torch.equal(ga, gb)
    mad = float((ga - gb).abs().max())
    flag = torch.tensor([1.0 if bit else 0.0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    mt = torch.tensor([mad], device=dev)
    dist.all_reduce(mt, op=dist.ReduceOp.MAX)
    scale = torch.tensor([float(gb.abs().max())], device=dev)
    dist.all_reduce(scale, op=dist.ReduceOp.MAX)
    ok = bool(flag.item() == 1.0) if BND == "fp32" else float(mt) <= 1e-2 * float(scale)
    if rank == 0:
        extra = ""
        if peak is not None and ref_peak is not None:
            extra = f"   peak GPU {ref_peak:.3f} -> {peak:.3f} GB (saved {ref_peak - peak:.3f})"
        print(f"  {label:34s} bitwise={str(bool(flag.item() == 1.0)):5s} "
              f"worst max|d|={float(mt):.3e}  {'PASS' if ok else 'FAIL'}{extra}")
    return ok


g_gpu, peak_gpu = run("gpu")
g_cpu, peak_cpu = run("cpu")
g_gpu_t, _ = run("gpu", tail=TAIL)
g_cpu_t, peak_cpu_t = run("cpu", tail=TAIL)
g_cpu_2, _ = run("cpu")

gsum = torch.tensor([float(g_gpu.abs().sum())], device=dev)
dist.all_reduce(gsum)

if rank == 0:
    print(f"=== DD cpu-staging check  {DIM}-D  py={PY} px={PX} "
          f"shape={shape} dtype={BND} tail={TAIL}/{nt} ===")
    print(f"  gradient |sum| over all ranks = {float(gsum):.6e}  (nonzero => a real gradient)")

def disk_is_refused():
    """Gate D: disk staging is NOT wired for DD, so it must fail loudly.

    A silent fall-through here would be worse than the missing feature: the
    ring would be driven with the wrong chunk and the gradient would be quietly
    wrong.  The check passes only if the error names the unsupported knob."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            prop = PropTorch(EQ(spatial_order=order, device=dev), backend="torch",
                             impl="c", shape=shape, dh=dh, dt=dt, nt=nt, abcn=abcn,
                             source_type=["h1"], receiver_type=["h1"], dev=dev,
                             boundary_saving_config={"enabled": True, "storage": "disk",
                                                     "disk_dir": td,
                                                     "storage_dtype": "fp32"})
            ddp = ModelParallel(prop, MeshTopology(py=PY, px=PX, shot_groups=1,
                                                   world_size=world, rank=rank))
            vp_g = torch.tensor(vp_init, device=dev, requires_grad=True)
            ddp(wav, src, rec, models=[vp_g]).pow(2).sum().backward()
        torch.cuda.synchronize()
        return False, "no error raised"
    except Exception as e:
        msg = str(e)
        return ("disk" in msg.lower()), msg.strip().split("\n")[0][:90]


ok = True
ok &= compare("A  cpu vs gpu", g_cpu, g_gpu, peak_gpu, peak_cpu)
ok &= compare(f"B  cpu vs gpu, tail={TAIL}", g_cpu_t, g_gpu_t, peak_gpu, peak_cpu_t)
ok &= compare("C  cpu run twice", g_cpu_2, g_cpu)
d_ok, d_msg = disk_is_refused()
dflag = torch.tensor([1.0 if d_ok else 0.0], device=dev)
dist.all_reduce(dflag, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"  {'D  disk refused loudly':34s} {'PASS' if dflag.item() == 1.0 else 'FAIL'}"
          f"   [{d_msg}]")
ok &= bool(dflag.item() == 1.0)
ok &= float(gsum) != 0.0

if rank == 0:
    print("DD_OFFLOAD_CHECK:", "PASS" if ok else "FAIL")
dist.barrier()
if not ok:
    dist.destroy_process_group()
    sys.exit(1)
dist.destroy_process_group()
