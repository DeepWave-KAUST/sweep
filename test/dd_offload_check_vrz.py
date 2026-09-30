"""DD + host-staged boundary parity for AcousticVRZ3D, and proof the fast path is live.

`storage='cpu'` used to disqualify the persistent backward runner outright, so
VRZ fell back to the per-call binding -- and its coupling schedule enters that
binding 3 x nt times per iteration, rebuilding 1/z, the CPML binding and the
boundary saver on every entry.  Every persistent runner now accepts host-staged
boundaries, and acoustic_vrz3d ships one (Vrz3dBackwardBsRunner), which means
the staged path executes DIFFERENT code than before: the boundary runtime now
persists across stepped segments instead of being re-armed per entry.

Three things have to hold, and the first is the one a green tick could otherwise
hide (lesson_optimisation_flag_silently_inactive):

  0  the runner is ACTUALLY admitted on the staged path (probe, not inference)
  A  storage='cpu' == storage='gpu'   bitwise, on vp.grad AND z.grad
  C  storage='cpu' run twice          bitwise (the staged path is deterministic)

Launch: torchrun --standalone --nproc-per-node=N test/dd_offload_check_vrz.py
Env: MESH_PY, MESH_PX, NT, DUMP (npy prefix for a cross-tree comparison).
"""
from __future__ import annotations
import os, sys
from pathlib import Path

os.environ.setdefault("SWEEP_VRZ_GRAD_SPLIT", "1")
import numpy as np, torch, torch.distributed as dist

# SWEEP_SRC lets one copy of this file drive ANOTHER tree's sweep -- the A/B
# arm has to import the production tree without that tree being written to.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(os.environ.get("SWEEP_SRC") or (REPO_ROOT / "src"))
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from sweep.equations import AcousticVRZ3D                 # noqa: E402
from sweep.parallel import MeshTopology                   # noqa: E402
from sweep.parallel import dd_propagator as _ddp_mod      # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel    # noqa: E402
from sweep.propagator.torch import PropTorch              # noqa: E402

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
torch.cuda.set_device(li)
dev = torch.device(f"cuda:{li}")

PY = int(os.environ.get("MESH_PY", 2)); PX = int(os.environ.get("MESH_PX", world // PY))
assert PY * PX == world, f"PY*PX={PY*PX} != world={world}"
NT = int(os.environ.get("NT", 600))
DUMP = os.environ.get("DUMP")

# ---- probe: did the staged path really get the persistent runner? -----------
ADMITTED = []
_Orig = _ddp_mod.SteppedBackwardRunner


class _Probe(_Orig):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        ADMITTED.append(self._cr is not None)


_ddp_mod.SteppedBackwardRunner = _Probe

DT, DH, ABCN, SO = 0.0012, 15.0, 20, 4
nz, ny, nx = 64, 96, 80
gshape = (nz, ny, nx)
zc = np.arange(nz, dtype=np.float32)[:, None, None]
vp = 1800.0 + 14.0 * np.maximum(zc - 12.0, 0.0)
vp = np.broadcast_to(vp, gshape).astype(np.float32).copy()
vp[:12] = 1500.0; vp[44:50] += 600.0; vp = np.minimum(vp, 3050.0)
rho = np.where(vp <= 1505.0, 1.0, 0.31 * np.clip(vp, 1.0, None) ** 0.25)
zimp = (rho * vp / 1000.0).astype(np.float32)

t = np.arange(NT, dtype=np.float32) * DT - 0.2
a = np.pi * 6.0 * t
wav = torch.as_tensor(((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32), device=dev)
src = np.array([[[37, 27, 20], [60, 47, 20], [75, 70, 20]]], dtype=np.int64)
rec = np.array([[[ix, iy, 0] for iy in range(2, ny - 2, 2)
                 for ix in range(2, nx - 2, 2)]], dtype=np.int64)


def run(storage, interval=None):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = {"enabled": True, "storage": storage, "storage_dtype": "int8"}
    if interval is not None:
        cfg["transfer_interval"] = interval
    n0 = len(ADMITTED)
    prop = PropTorch(AcousticVRZ3D(spatial_order=SO, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=gshape, dh=DH, dt=DT, nt=NT,
                     abcn=ABCN, source_type=["h1"], receiver_type=["h1"], dev=dev,
                     free_surface=True, pml_type="cpmlr", boundary_saving_config=cfg)
    ddp = ModelParallel(prop, MeshTopology(py=PY, px=PX, shot_groups=1,
                                           world_size=world, rank=rank))
    vpg = torch.tensor(vp, device=dev, requires_grad=True)
    zg = torch.tensor(zimp, device=dev, requires_grad=True)
    r = ddp(wav, src, rec, models=[vpg, zg])
    (r.double() ** 2).sum().backward()
    dist.all_reduce(vpg.grad); dist.all_reduce(zg.grad)
    torch.cuda.synchronize()
    return (vpg.grad.detach().clone(), zg.grad.detach().clone(),
            torch.cuda.max_memory_allocated() / 2 ** 30,
            ADMITTED[n0:])


def compare(label, a, b):
    """All-rank verdict: bitwise on every tile, worst deviation anywhere."""
    bit = all(torch.equal(x, y) for x, y in zip(a, b))
    mad = max(float((x - y).abs().max()) for x, y in zip(a, b))
    flag = torch.tensor([1.0 if bit else 0.0], device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    mt = torch.tensor([mad], device=dev); dist.all_reduce(mt, op=dist.ReduceOp.MAX)
    ok = bool(flag.item() == 1.0)
    if rank == 0:
        print(f"  {label:34s} bitwise={str(ok):5s} worst max|d|={float(mt):.3e}  "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


# Repeat BOTH storages: a bitwise criterion is only meaningful once the same
# code run twice reproduces itself (lesson_bitexact_criterion_must_verify_
# attainable).  If the staged path has a run-to-run floor of its own, the honest
# question is not "is cpu bit-equal to gpu" but "is the cpu-vs-gpu difference
# inside the floor the paths already have" (lesson_floor_relative_criterion_
# needs_pooled_floor).
g_gpu = run("gpu")
g_gpu2 = run("gpu")
g_cpu = run("cpu", interval=1)
g_cpu2 = run("cpu", interval=1)

if rank == 0:
    print(f"=== DD VRZ host-staged boundary  py={PY} px={PX} nt={NT} ===", flush=True)
    print(f"  runner admitted: gpu={g_gpu[3]}  cpu={g_cpu[3]}", flush=True)
    print(f"  peak GPU  gpu={g_gpu[2]:.3f} GiB  cpu={g_cpu[2]:.3f} GiB", flush=True)

# 0. the point of the change: the staged path must be on the runner.
staged_fast = all(g_cpu[3]) and len(g_cpu[3]) > 0
sf = torch.tensor([1.0 if staged_fast else 0.0], device=dev)
dist.all_reduce(sf, op=dist.ReduceOp.MIN)
if rank == 0:
    print(f"  {'0  staged path uses the runner':34s} "
          f"{'PASS' if sf.item() == 1.0 else 'FAIL -- the guard still refuses it'}", flush=True)

def worst(a, b):
    m = max(float((x - y).abs().max()) for x, y in zip(a, b))
    s = max(float(y.abs().max()) for y in b)
    mt = torch.tensor([m], device=dev); dist.all_reduce(mt, op=dist.ReduceOp.MAX)
    st = torch.tensor([s], device=dev); dist.all_reduce(st, op=dist.ReduceOp.MAX)
    return float(mt) / (float(st) + 1e-30)


floor_gpu = worst(g_gpu2[:2], g_gpu[:2])
floor_cpu = worst(g_cpu2[:2], g_cpu[:2])
cross = worst(g_cpu[:2], g_gpu[:2])
floor = max(floor_gpu, floor_cpu)
if rank == 0:
    print(f"  {'gpu run-to-run floor':34s} rel={floor_gpu:.3e}", flush=True)
    print(f"  {'cpu run-to-run floor':34s} rel={floor_cpu:.3e}", flush=True)
    print(f"  {'cpu vs gpu':34s} rel={cross:.3e}", flush=True)

ok = bool(sf.item() == 1.0)
if floor == 0.0:
    # Both paths reproduce themselves exactly -- then bitwise is attainable and
    # is the criterion.
    ok &= compare("A  cpu vs gpu (vp.grad, z.grad)", g_cpu[:2], g_gpu[:2])
    ok &= compare("C  cpu run twice", g_cpu2[:2], g_cpu[:2])
else:
    within = cross <= 4.0 * floor
    if rank == 0:
        print(f"  {'A  cpu vs gpu within 4x floor':34s} "
              f"{cross:.3e} vs 4x{floor:.3e}  {'PASS' if within else 'FAIL'}", flush=True)
    ok &= within

if DUMP and rank == 0:
    np.save(f"{DUMP}_cpu_vp.npy", g_cpu[0].cpu().numpy())
    np.save(f"{DUMP}_cpu_z.npy", g_cpu[1].cpu().numpy())
    print(f"  dumped {DUMP}_cpu_{{vp,z}}.npy", flush=True)

if rank == 0:
    print(f"VRZ-OFFLOAD {'PASS' if ok else 'FAIL'}", flush=True)
fail = torch.tensor([0 if ok else 1], device=dev)
dist.all_reduce(fail); dist.barrier(); dist.destroy_process_group()
sys.exit(int(fail.item() > 0))
