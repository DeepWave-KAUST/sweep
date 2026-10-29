# Parallelism: shots, source encoding, domain decomposition, JAX

Contents: 1. Which kind · 2. Shot parallelism (torchrun / DDP pattern) · 3. Source encoding ·
4. Domain decomposition (`ModelParallel`) · 5. JAX

## 1. Which kind

| Situation | Use |
|---|---|
| Model fits on one GPU, many shots | batch shots per call; across GPUs split the shot list per rank and `all_reduce` the gradient |
| Many shots, want fewer simulations | source encoding: one super-shot `(1, nsrc, ndim)` |
| Model does not fit on one GPU | `sweep.parallel.ModelParallel` (x strips in 2-D, x/y tiles in 3-D), `impl='c'` |
| Both | `MeshTopology(py, px, shot_groups=...)` |

## 2. Shot parallelism (torchrun / DDP pattern)

Every rank builds the **same** `PropTorch` on its own GPU, runs its slice of shots, and
all-reduces the model gradient. The script below also runs as a single process (no `torchrun`);
launch with `torchrun --standalone --nproc-per-node=4 script.py` for 4 GPUs.

```python
import os, numpy as np, torch, torch.distributed as dist
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

distributed = "RANK" in os.environ                       # set by torchrun
if distributed:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
else:
    rank, world = 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nz, nx, dt, nt = 40, 64, 1e-3, 300
solver = PropTorch(Acoustic(device=device), shape=(nz, nx), dh=10.0, dt=dt, abcn=20,
                   device=device, B=2)
w = ricker(np.arange(nt) * dt - 0.07, f=15.0).astype(np.float32)
all_src = np.array([[x, 2] for x in range(8, 64, 8)])        # 7 shots
rec1 = [[ix, 2] for ix in range(nx)]
vp_true = torch.full((nz, nx), 2000.0, device=device)
vp_true[20:] = 2400.0
vp = torch.full((nz, nx), 2000.0, device=device, requires_grad=True)

mine = all_src[rank::world]                                 # this rank's shots
local_loss = torch.zeros((), device=device, dtype=torch.float64)
for i in range(0, len(mine), 2):                             # chunks of <= B shots
    src = mine[i:i + 2]
    rec = np.array([rec1] * len(src))
    with torch.no_grad():
        obs = solver(w, src, rec, models=[vp_true])          # in practice: load observed data
    loss = 0.5 * (solver(w, src, rec, models=[vp]) - obs).double().pow(2).sum()
    loss.backward()
    local_loss += loss.detach()
if distributed:
    dist.all_reduce(vp.grad)                                 # sum gradients over ranks
    dist.all_reduce(local_loss)
    dist.destroy_process_group()
print(rank, float(local_loss), tuple(vp.grad.shape))
assert torch.isfinite(vp.grad).all()
```

- Give every rank identical constructor arguments; only `device` differs.
- Generate or load observed data once (rank 0) and broadcast, rather than recomputing per rank.
- Draw random shot subsets on one rank and `dist.broadcast` the indices so ranks agree.
- Accumulate misfits in float64 when comparing with single-GPU runs.

## 3. Source encoding

`sources` of shape `(1, nsrc, ndim)` fires all `nsrc` sources in one simulation (wavelet `(nt,)`
or `(nsrc, nt)`, e.g. with random +-1 polarities per source); `receivers` is `(1, nrec, ndim)`.
The record is the sum of the individual shots (linearity; see the check in `propagators.md`).
For FWI, encode the observed data with the same weights and draw new codes every iteration
(crosstalk is suppressed only on average).

## 4. Domain decomposition (`ModelParallel`)

<!-- skill-test: skip -->
```python
# one process per GPU: torchrun --standalone --nproc-per-node=4 script.py
from sweep.parallel import MeshTopology, ModelParallel, pad_to_mesh
topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)  # 2-D: py=1
prop = PropTorch(eq, shape=(nz, nx), dh=dh, dt=dt, abcn=abcn, impl="c", device=dev)  # GLOBAL spec
ddp = ModelParallel(prop, topo)
vp = vp_init.clone().requires_grad_(True)                         # full physical model, every rank
src = np.array([[[ix, iz]]])                                      # (1, nsrc, ndim): 3-D only
rec_tile = ddp(wavelet, src, rec, models=[pad_to_mesh(vp, px=world)])
loss = 0.5 * (rec_tile - obs_tile).double().pow(2).sum()          # obs_tile: this rank's receivers
loss.backward()
dist.all_reduce(vp.grad)                                          # global-leaf style
full_record = ddp.gather_record(rec_tile)                         # on the group's root rank, else None
```

Verified constraints (single-rank run of this pattern on one GPU worked; refusals raise):

| Item | Rule |
|---|---|
| Equations | `Acoustic` (2-D), `Acoustic3D`, `AcousticVRZ3D`, `AcousticBorn`, `AcousticBorn3D`, `Elastic`, `Elastic3D` |
| Backend / memory | `impl='c'` with boundary saving only (`Full()`/`Ckpt()` refused); `storage` `"gpu"` or `"cpu"`, not `"disk"` |
| Geometry | `sources` must be 3-D `(1, nsrc, ndim)`; a 2-D `(nshots, ndim)` array raises `ValueError` |
| Grid | `pad_to_mesh(model, px=...)` pads split axes to a tile multiple (a slightly different problem than unpadded) |
| Records | each rank returns only the receivers its tile owns (`ddp.own_receiver_indices`) |
| Gradients | models only; a wavelet requiring grad is refused |
| Boundaries | free surface top only; no `topography=` |
| Reuse | `models=None` reuses the last prepared model (skips a collective) when the model has not changed, e.g. observed-data loops; keep passing the leaf for gradient calls |

Launch across nodes: `torchrun --nnodes=N --nproc-per-node=G --rdzv-backend=c10d
--rdzv-endpoint=HOST:29500 --rdzv-id=job script.py` with `py*px == N*G`. Examples:
`examples/multi-gpu/torch/dd_fwi_*.py` in the repository.

## 5. JAX

`PropJax` (see `propagators.md`) composes with `jax.grad`, `jax.pmap` / sharding; the
repository's `examples/multi-gpu/jax/fwi_marmousi_pmap.py` shards shots over devices with
`jax.pmap` and `jax.lax.psum` of the gradient. Install `sweep-solver[jax]`.
