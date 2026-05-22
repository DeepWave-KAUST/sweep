# Multi-GPU · multi-process DDP

When you need to scale across **multiple processes** — multiple GPUs on one node, or across nodes via Slurm / mpirun — the SWEEP pattern is the standard PyTorch
DistributedDataParallel (DDP) launcher with `torchrun`. Each rank holds its
own `PropTorch`, owns a slice of the global shot list, and exchanges
gradients through NCCL.

This page documents the production script
[`examples/multi-gpu/torch/fwi_marmousi_dist.py`](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/multi-gpu/torch/fwi_marmousi_dist.py) — it is **not** a notebook because notebooks run inside one Python kernel, while DDP requires *N* independent interpreters. See the [single-process variant](../notebooks/12_multi_gpu.ipynb) for the in-kernel demo.

## Launch

Prepare the Marmousi `.npy` files first (one-time, same as the 2-D notebook):

```bash
python3 examples/models/marmousi/download_marmousi.py --extract
python3 examples/models/marmousi/extract_model_segy.py
python3 examples/models/marmousi/convert_segy_to_npy.py
python3 examples/models/marmousi/prepare_fwi_models.py \
  --input examples/models/marmousi/npy/vp_1p25m.npy \
  --source-dh 1.25 --target-dh 12.5 \
  --radii 16,16 --passes 3
```

Then on a 2-GPU box:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/multi-gpu/torch/fwi_marmousi_dist.py \
  --backend torch --impl c --device cuda
```

For a multi-node SLURM job, the launch is the same idea but with `--nnodes`,
`--node_rank`, `--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT`.

The script writes per-epoch progress figures to
`multi_gpu_acoustic_fwi_cuda/` (from rank 0 only); the other ranks stay
silent except for one-line status prints.

## How it works

The script is structured as five small steps; the SWEEP-specific bits are
identical to single-process FWI — only the launcher and the gradient
reduction change.

### 1. Initialise the process group

```python
import torch
import torch.distributed as dist

def setup_distributed():
    rank       = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl',
                            rank=rank, world_size=world_size)
    return rank, world_size, torch.device('cuda', local_rank)
```

`torchrun` populates `RANK / WORLD_SIZE / LOCAL_RANK` automatically; each
process pins itself to one GPU and joins the NCCL group.

### 2. Build the solver (each rank, identical args)

```python
solver = PropTorch(
    Acoustic(spatial_order=cfg['spatial_order'], device=device),
    shape=shape, dev=device, dh=cfg['dh'], dt=cfg['dt'],
    source_type=['h1'], receiver_type=['h1'],
    pml_type='cpmlr', impl='c',
    cuda_options=CUDAOptions(
        memory=MemoryOptions(strategy='boundary',
                             boundary=BoundaryOptions(storage='gpu')),
    ),
)
```

Every rank constructs the **same** solver with the same shape / dh / dt;
the only thing that differs is `device`.

### 3. Generate observed data once on rank 0, broadcast

```python
if rank == 0:
    obs = chunked_observed_data(solver, wave, sources, receivers,
                                models=[true_vp],
                                forward_batchsize=cfg['forward_batchsize'])
else:
    obs = None
obs = broadcast_observed_data(rank, device, obs)
```

The forward pass through the true model is identical work for every rank —
do it once and `dist.broadcast` the result, instead of paying for it on
every GPU.

### 4. Per-step shot split + gradient all-reduce

```python
shot_idx = torch.from_numpy(np.random.choice(nshots, size=batchsize,
                                              replace=False).astype(np.int64))
dist.broadcast(shot_idx, src=0)                # same draw on every rank
local_idx = split_indices_for_rank(shot_idx, rank, world_size)

# local forward + backward on this rank's shots
for s in local_idx:
    syn = solver(wavelet, sources[s:s+1], receivers[s:s+1], models=[inv_vp])
    (0.5 * (syn - obs_torch[s:s+1].to(device)).pow(2).sum()).backward()

# sum gradients across ranks → identical inv_vp.grad on every rank
dist.all_reduce(inv_vp.grad, op=dist.ReduceOp.SUM)

optimizer.step()      # same parameters → same update on every rank
```

The key invariant: **identical RNG state + identical reduced gradient ⇒ every
rank stays in sync**. There is no need to broadcast `inv_vp` after the
step — every rank computed the same update independently.

### 5. Tear-down

```python
dist.barrier(device_ids=[device.index])
dist.destroy_process_group()
```

## Speedup expectations

Best case (`impl='c'`, big enough shot batch to amortise the all-reduce):

| GPUs | wall-clock relative to 1 GPU |
|---|---|
| 2 | ~1.85× |
| 4 | ~3.5× |
| 8 | ~6.5× |

The diminishing returns come from the all-reduce (one synchronous step per
optimizer iteration) and from the work that is *not* per-shot — observed
data preparation, figure writing, optimizer step. For very small jobs
(few shots, low spatial order) the per-step kernel time can become
comparable to the NCCL handshake and you may want to stay on
[single-process multi-GPU](../notebooks/12_multi_gpu.ipynb).

## See also

- [`examples/multi-gpu/jax/fwi_marmousi_pmap.py`](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/multi-gpu/jax/fwi_marmousi_pmap.py) — the same idea with `jax.pmap` (single-process, but truly multi-device thanks to XLA SPMD).
- [Single-process multi-GPU notebook](../notebooks/12_multi_gpu.ipynb) — when you don't want to deal with launchers.
