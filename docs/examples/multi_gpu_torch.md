# Torch Distributed Multi-GPU FWI

Source file:

- `examples/multi-gpu/torch/fwi_marmousi_dist.py`

This script is a data-parallel version of the 2D acoustic Marmousi Torch FWI
example. It uses `torch.distributed` with one process per GPU.

Each rank keeps its own copy of the velocity model on one GPU. At each
iteration:

1. rank 0 samples a global shot batch and broadcasts the shot indices
2. each rank receives a subset of those shots
3. each rank computes its local synthetic data and local loss contribution
4. model gradients are summed with `torch.distributed.all_reduce`
5. every rank applies the same optimizer step

The model is a plain inversion tensor, so the script does not wrap a module in
`DistributedDataParallel`. The synchronized state is the gradient, loss
summary, shot indices, and optional illumination panels used for visualization.

## How the Split Works

`torchrun` starts one Python process per GPU. Each process receives environment
variables such as `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. The example maps
`LOCAL_RANK` to a CUDA device and initializes NCCL:

```python
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{local_rank}")
```

The global inversion model is replicated on every rank:

```python
inv_vp = torch.from_numpy(init_model).to(device).requires_grad_(True)
optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)
```

At each epoch, rank 0 chooses the global shot batch and broadcasts the selected
shot indices:

```python
if rank == 0:
    shot_idx = sample_global_batch(...)
else:
    shot_idx = torch.empty(batchsize, dtype=torch.int64, device=device)
dist.broadcast(shot_idx, src=0)
```

Then the selected shot indices are split by rank. For example, with
`WORLD_SIZE=4` and `batchsize=8`, each rank receives two shots:

```python
local_idx = torch.tensor_split(shot_idx, world_size)[rank]
syn = solver(wave, sources[local_idx], receivers[local_idx], models=[inv_vp])
```

Each rank computes only its local residual and backpropagates a loss normalized
by the global selected-shot batch size. After local backward, the model gradient
is summed across GPUs:

```python
local_loss.backward()
dist.all_reduce(inv_vp.grad, op=dist.ReduceOp.SUM)
optimizer.step()
```

Because every rank starts from the same `inv_vp`, receives the same reduced
gradient, and uses the same optimizer settings, all ranks keep identical model
copies after each optimizer step.

## Run

Use `torchrun` from the repository root. For two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/multi-gpu/torch/fwi_marmousi_dist.py --backend torch --impl c --device cuda
```

The script also accepts `--backend torch --impl eager`, which still uses one distributed
process per CUDA device but runs each local propagation through the eager
`PropTorch` implementation.

## Outputs

Only rank 0 writes figures. Outputs are saved under:

- `examples/multi-gpu/torch/multi_gpu_acoustic_fwi_cuda/`
- `examples/multi-gpu/torch/multi_gpu_acoustic_fwi_eager/`

Saved files include:

- `ricker.png`
- `observed_data.png`
- `loss.png`
- `epoch_XXXX.png`

The progress panel includes the true model, inverted model, gradient, source
illumination, and receiver illumination. Illumination is shown only for
inspection; it is not used by the optimizer.

## Notes

- Launch with one process per GPU.
- The default c CUDA path uses boundary saving through `PropTorch(..., backend="torch", impl="c")`.
- Rank 0 generates observed data once, then broadcasts it to the other ranks.
- The loss is normalized by the global selected-shot batch, so ranks with
  different local shot counts still contribute the correct gradient scale.
- If GPU memory is tight, reduce `--batchsize` or use fewer receivers in the
  shared Marmousi configuration.
