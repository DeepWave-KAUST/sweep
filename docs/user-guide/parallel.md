# Domain decomposition

When a model does **not** fit on one GPU, `ModelParallel` splits it into
tiles — one GPU per tile — and runs the same physics with per-timestep halo
exchanges over NCCL. It is the model-parallel counterpart to shot-parallel
DDP (which replicates the whole model per rank and splits the shot list):

|                       | what is split          | each rank holds       | use it when                      |
|-----------------------|------------------------|-----------------------|----------------------------------|
| shot parallel (DDP)   | the shot list          | the whole model       | the model fits on one GPU        |
| **model parallel (DD)** | **the model, into tiles** | **one tile, all shots** | **the model does not fit**     |

Both compose: `MeshTopology(py, px, shot_groups=...)` describes a
`shot_groups × (py × px)` rank grid, where ranks that share a tile
coordinate run different shots and their gradients are all-reduced
automatically after the backward.

Hands-on companions: notebook
[25 · Domain decomposition](../notebooks/25_domain_decomposition.ipynb),
notebook [26 · Overthrust 3-D](../notebooks/26_dd_overthrust_3d.ipynb), and
the runnable FWI scripts under `examples/multi-gpu/torch/dd_fwi_*.py`.

## Two classes

```python
from sweep.parallel import MeshTopology, ModelParallel, pad_to_mesh

topo = MeshTopology(py=1, px=4, shot_groups=1,
                    world_size=world, rank=rank)      # 2-D: py must be 1
prop = PropTorch(eq, shape=(nz, nx), dh=dh, dt=dt, nt=nt, abcn=abcn,
                 impl="c", dev=dev, ...)              # the GLOBAL problem spec
ddp  = ModelParallel(prop, topo)                      # wrap; tiles are automatic
```

`ModelParallel` reads the global problem off the wrapped single-domain
`PropTorch` and derives everything per rank: the tile slice, the
**cut-aware padding** (a cut face carries only the stencil halo, no PML —
absorbing boundaries live only on true domain edges), global→tile source
and receiver remapping, and the per-tile boundary-saving ring. The
gradient-memory configuration (`storage`, `storage_dtype`,
`BoundaryOptions.tail_steps`) is inherited from the wrapped prop's memory
config, so every rank is consistent by construction — see
[Boundary storage under DD](#boundary-storage-under-dd) for which values the
DD backward actually accepts.

Launch with one process per GPU:

```bash
torchrun --standalone --nproc-per-node=4 your_script.py
```

## Forward and gradients — plain autograd

The forward returns this rank's differentiable tile record; `backward()`
produces model gradients exactly like the single-domain path:

```python
vp_tile = torch.tensor(vp_global[..., ddp.x0:ddp.x0 + ddp.nxp],
                       device=dev, requires_grad=True)      # tile leaf
rec_tile = ddp(wavelet, src_global, rec_global, models=[vp_tile])
loss = misfit(rec_tile, obs_tile)
loss.backward()                       # vp_tile.grad = this tile's gradient
full_rec = ddp.gather_record(rec_tile)   # rank 0 assembles; others get None
```

Two equivalent leaf styles are in use:

- **Tile leaf** (above): each rank keeps only its slice; gradients stay
  per-tile. Least memory, no gradient collective.
- **Global leaf** (the `dd_fwi_*` example scripts): every rank holds the
  full physical model, passes `models=[pad_to_mesh(vp, px=px)]`, and adds
  one `dist.all_reduce(vp.grad)` after the backward. A single-GPU script
  becomes multi-GPU with two marked lines.

Each rank's record carries only the receivers its tile owns —
`ddp.own_receiver_indices` gives the global indices, and ownership is a
partition, so per-rank misfits over those traces sum to the global misfit.
Accumulate the misfit scalar in `float64` if you compare against a
single-GPU run: the per-rank partial sums add in a different order than
one GPU's single sum, which shows up at fp32 rounding otherwise.

## The two explicit performance switches

**`models=None` reuses the prepared model.** Passing `models=` triggers the
per-call model setup: tile slicing, edge padding, and a model-halo NCCL
collective. When the model has not changed since the previous call — every
shot of an observed-data loop, a line search — pass `models=None` to skip
it. This is explicit on purpose: the propagator never guesses whether an
in-place optimizer step changed the model. Autograd calls must keep passing
the leaf tensors (the graph attaches to what you pass *this* call), so
inversion loops re-pass `models=[...]` every shot; the per-step wavefield
halo exchange always runs regardless.

**`torch.no_grad()` keeps the capture forward-only.** The first
gradient-capable call allocates the adjoint wavefields and the nt-scaled
boundary ring; a forward under `no_grad` with non-grad models does not.
Generate observed data and run QC forwards under `no_grad`, and keep a
separate `ModelParallel` instance for pure-forward work — an instance that
has once produced a gradient keeps its adjoint machinery for its lifetime.

## Correctness

On fp32 gpu-direct boundaries the DD gradient is **bit-identical** to the
single-domain gradient (`test/test_dd_backward_two_tile*.py`,
`test/dd_api_check.py`), including with boundary tail truncation
(`test/test_dd_tail_two_tile.py`). `pad_to_mesh` pads the split axes up to
the tile multiple; a padded run is a (slightly) different discrete problem
than an unpadded one, so compare like against like — the example scripts'
`--check` mode does exactly that.

## Scope and limits

- Equations: those with stepped compiled kernels — `Acoustic` (2-D),
  `Acoustic3D`, `AcousticVRZ3D`, `Elastic` (2-D) and `Elastic3D`. Everything
  else is refused at construction with an error that names the equation,
  including the **2-D** `AcousticVRZ` (only its 3-D sibling is stepped),
  `AcousticVTI`/`AcousticVTI1st`, `AcousticTTI`, `ElasticTTI`, `ElasticVRR`
  and `ViscoAcoustic`.
- Cuts: x strips in 2-D (`py=1`); x/y tile grids in 3-D.
- Free surface: top face only under DD (a cut face can never carry one).
- Boundary storage and dtype: see the table below.
- `BoundaryOptions.tail_steps`: Acoustic 2-D/3-D (see
  [Propagators](propagators.md#boundary-tail-truncation-boundaryoptionstail_steps));
  it composes with cpu staging.
- Not routed through DD: `rtm()` — use the gradient path (notebook
  [08](../notebooks/08_rtm_acoustic_marmousi.ipynb) shows how). Encoded
  supershots (a `(nsrc, nt)` wavelet) *are* supported: each tile keeps the
  rows of the sources it owns (`test/dd_encoded_check.py`).
- `SWEEP_DD_DISABLE_OVERLAP=1` forces the serial step-then-exchange path —
  the bit-exact reference for the comm/compute-overlap forward and a
  production escape hatch.

## Boundary storage under DD

The per-tile ring inherits `storage` / `storage_dtype` from the wrapped prop,
but not every value is wired for the DD backward — which Python drives one
step per kernel call, unlike the monolithic loop the staged paths were built
for. What is refused, is refused loudly at the first backward:

| `BoundaryOptions.storage` | Acoustic 2-D / 3-D | Elastic 2-D / 3-D |
| --- | --- | --- |
| `"gpu"` (default, gpu-direct) | yes | yes |
| `"cpu"` (pinned-host staging) | **yes** | no — raises |
| `"disk"` | no — raises | no — raises |

`storage="cpu"` also needs a **real cut**: a single-tile `ModelParallel`
(`world_size=1`) refuses it, because that path reaches a reconstruction
indexing that is not exercised by any multi-tile run. Use gpu-direct there, or
a plain `PropTorch` backward, which supports cpu and disk staging as usual.

Every `storage_dtype` works with either storage. On fp32 and bf16 the
cpu-staged gradient is **bit-identical** to the gpu-direct one; fp16 and int8
differ only within their own run-to-run quantisation floor, i.e. by no more
than two runs of the *same* configuration differ from each other
(`test/dd_offload_check.py` checks exactly that, on both counts).

Staging trades PCIe traffic for GPU memory and the copies are synchronous, so
it is the escape hatch for a tile whose ring does not fit, not a default.

API details: [sweep.parallel reference](../api/parallel.md).
