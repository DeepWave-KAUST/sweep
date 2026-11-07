# SEM · multi-GPU domain decomposition

A SEM model larger than one GPU holds runs across several with `sweep.sem.dd`.
`SEMModelParallel(prop, StripTopology(px, rank))` cuts the mesh into `px` strips
along x, one per rank, and runs the `impl='c'` kernels on each strip. After every
step's assembly the nodes that two neighbouring strips share are summed across them
(NCCL), so the run is the single-GPU one up to the order of the floating-point sums.
This page documents the script
[`examples/multi-gpu/torch/sem_dd_acoustic3d.py`](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/multi-gpu/torch/sem_dd_acoustic3d.py).
It is not a notebook, because each rank is a process of its own, started by
`torchrun`.

Five equations run this way, all with their C-PML: `AcousticPotentialSEM2D`,
`ElasticSEM2DDeformed`, `AcousticElasticSEM2D`, `AcousticSEM3D` and
`AcousticSEM3DDeformed`. How the strips are built and what they exchange is in
[Domain decomposition](../user-guide/sem.md#domain-decomposition-multi-gpu-phase-1)
of the SEM user guide.

## Launch

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/multi-gpu/torch/sem_dd_acoustic3d.py --check
```

The script records one shot in a 3-D model with a velocity anomaly, takes the
gradient of the misfit at the background model, and prints the time and the peak
GPU memory of each rank. Its options are the model (`--shape NZ NY NX`, every
`n - 1` a multiple of 4, and `--dh`), the time axis (`--nt`, `--dt`), the wavelet's
peak frequency (`--f0`), what the backward keeps (`--memory rim|boundary|full`) and
`--check`, which compares the run with the same shot on one GPU.

## How it works

### 1. The same propagator on every rank

```python
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank % torch.cuda.device_count())

prop = PropSEM(AcousticSEM3D, shape=(nz, ny, nx), dh=10.0, dt=5e-4, nt=2000,
               n_order=4, abcn=16, free_surface=True, f_dominant=10.0,
               dev="cuda", impl="c", memory=BoundarySaving(), bs_band_mode="rim")
ddp = SEMModelParallel(prop, StripTopology(world, rank=rank))
```

`PropSEM` carries the configuration, from which each rank builds its strip. The
strips are equal: the number of element columns of the mesh, the model's and the
C-PML's, must be a multiple of `px` (here 64 + 2 × 4 = 72). The gradient memory is
the propagator's, and each strip keeps only its own part of it: with boundary
saving, the ring of its own share of the band.

### 2. Sources and receivers as global node ids

The call takes one shot, with the source and the receivers as global node ids of
the SEM mesh. On the flat 3-D mesh the node grid is the model grid with the C-PML
added, numbered row-major with z slowest, and a node coincides with a grid point at
the element corners, where every index is a multiple of `n_order`:

```python
def node_id(prop, ix, iy, iz):
    nz, ny, nx = prop.shape
    nyg, nxg = ny + 2 * prop.abcn, nx + 2 * prop.abcn
    z0 = 0 if "top" in prop.free_surface else prop.abcn
    return ((iz + z0) * nyg + iy + prop.abcn) * nxg + ix + prop.abcn
```

### 3. The forward and the gradient

```python
pred = ddp(wavelet, src, rec, [vp])           # (nt, nrec): this rank's receivers
loss = 0.5 * (pred - obs).pow(2).sum()        # adds up over the ranks
loss.backward()
dist.all_reduce(vp.grad)                      # this rank's nodes: the sum is the gradient
record = ddp.gather_record(pred.detach())     # the whole record, on rank 0
```

Each rank's record holds the receivers on its strip, and its gradient the nodes of
its strip, so sums over the ranks give the global misfit and the global gradient. A
loss that is not a sum over the receivers, a misfit normalized by the record's
energy for example, needs its normalizer reduced across the ranks first.

### 4. The check

With `--check`, rank 0 frees its strip and runs the same shot through `PropSEM` on
its own GPU, with the source and the receivers as the grid points of the same
element corners. It compares the records and the gradients of the record's energy.
The energy's gradient is dominated by the direct wave; the misfit's comes from a
small residual, and the atomic sums of the CUDA kernels move it by about a percent
from one run to the next, on one GPU as on several.

## Results

Measured on four A100 80GB PCIe GPUs with the script's defaults (`AcousticSEM3D`, 2000
steps of 0.5 ms, boundary saving of the band's first element layer), for the misfit's
forward and backward:

| model | GPUs | forward + backward (s) | peak memory per rank (GiB) |
|---|---|---|---|
| 97 × 129 × 257 (3.2 M points) | 1 | 5.2 | 12.3 |
| | 2 | 3.1 | 6.2 |
| | 4 | 3.1 | 2.7 to 3.6 |
| 193 × 257 × 513 (25.4 M points) | 1 | 28.8 | 49.8 |
| | 2 | 15.2 | 25.1 |
| | 4 | 8.5 | 10.7 to 14.9 |

The memory per rank falls with the number of strips; the two edge strips, which also hold
the C-PML on their outer side, carry more than the inner ones. The time falls as long as each
strip has enough work: on the larger model four GPUs run 3.4 times as fast as one, while on
the smaller one the exchange after every step holds four GPUs at the time of two. The ring of
the whole band (`--memory boundary`) takes 55.6 GiB on one GPU for the smaller model, against
12.3 GiB for its first element layer. For every split the misfit is the same to the printed
digits and the gradient's norm within 0.6%, the spread of the misfit's gradient from run to
run.

`--check` on the smaller model:

```
one GPU against 2: record 6.8e-06, gradient of the record's energy 1.7e-07 (two runs on one GPU: 4.5e-07)
one GPU against 4: record 6.7e-06, gradient of the record's energy 1.3e-07 (two runs on one GPU: 1.7e-07)
```

The records agree to single-precision round-off, and the gradients differ from the
single-GPU one by less than two runs on one GPU differ from each other.

## Notes

- **One shot per call.** Loop over the shots and sum their gradients, as for any
  `PropSEM` call.
- **Not on the strips:** wavefield snapshots (`return_wavefield`), illumination
  (`compute_illumination`) and per-shot model batches; they run on one GPU.
- **Memory.** The time history, the ring of boundary saving or the full tape, grows
  with `nt` and is split between the strips; the two edge strips also hold the
  C-PML on their outer side, and carry more of it.
