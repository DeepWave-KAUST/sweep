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

The gradient is shot-summed, but the *record* is not: each shot group holds a
different shot, so `gather_record` assembles per group, on that group's root
rank (`shot_group * py * px`). With the default `shot_groups=1` that root is
rank 0 and every other rank gets `None`.

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
                 impl="c", device=dev, ...)           # the GLOBAL problem spec
ddp  = ModelParallel(prop, topo)                      # wrap; tiles are automatic
```

`ModelParallel` reads the global problem off the wrapped single-domain
`PropTorch` and derives everything per rank: the tile slice, the
**cut-aware padding** (a cut face carries only the stencil halo, no PML —
absorbing boundaries live only on true domain edges), global→tile source
and receiver remapping, and the per-tile boundary-saving ring. The
gradient-memory configuration (`storage`, `storage_dtype`,
`BoundarySaving.tail_steps`) is inherited from the wrapped prop's memory
config, so every rank is consistent by construction — see
[Boundary storage under DD](#boundary-storage-under-dd) for which values the
DD backward actually accepts.

Launch with one process per GPU:

```bash
torchrun --standalone --nproc-per-node=4 your_script.py
```

`--standalone` means *this machine only*. To spread the same tiles over several
machines, drop it and give the ranks a meeting point instead — every node then
runs an identical command and negotiates its own numbering:

```bash
torchrun --nnodes=3 --nproc-per-node=4 \
  --rdzv-backend=c10d --rdzv-endpoint=<first-node>:29500 --rdzv-id=dd1 \
  your_script.py                      # py * px must now be 3 * 4 = 12
```

Nothing in your script changes: `ModelParallel` never picks a device and NCCL
does not care whether a tile's neighbour is on this machine or the next one.
The cross-node gradient is bit-identical to the single-domain one, same as
within a node. A ready SLURM batch script and the pitfalls (one task per *node*,
not per GPU; ask for partial nodes) are in
[`examples/multi-gpu/torch/README.md`](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/multi-gpu/torch/README.md#across-nodes).

## Forward and gradients — plain autograd

The forward returns this rank's differentiable tile record; `backward()`
produces model gradients exactly like the single-domain path:

```python
vp_tile = torch.tensor(vp_global[..., ddp.x0:ddp.x0 + ddp.nxp],
                       device=dev, requires_grad=True)      # tile leaf
rec_tile = ddp(wavelet, src_global, rec_global, models=[vp_tile])
loss = misfit(rec_tile, obs_tile)
loss.backward()                       # vp_tile.grad = this tile's gradient
full_rec = ddp.gather_record(rec_tile)   # the shot group's root assembles
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
(`test/test_dd_tail_two_tile.py`). `AcousticVRZ3D` runs the same gradient
kernels on one card and under DD at every order. `AcousticBorn` and
`AcousticBorn3D` are bit-identical too, record and both gradients
(`test/dd_corner_check_born2d.py`, `test/dd_corner_check_born3d.py`).
`pad_to_mesh` pads the split axes up to the tile multiple; a padded run is a
(slightly) different discrete problem than an unpadded one, so compare like
against like — the example scripts' `--check` mode does exactly that.

## The eager propagator

The same call decomposes an `impl="eager"` propagator:

```python
prop = PropTorch(eq, shape=(nz, nx), dh=dh, dt=dt, nt=nt, abcn=abcn,
                 impl="eager", device=dev, memory=Ckpt(chunks=100))
ddp  = ModelParallel(prop, topo, exchange_steps=8)      # x strips (2-D), x/y tiles (3-D)
syn  = ddp(wavelet, sources, receivers, models=[vp])     # the receivers this tile owns
loss = 0.5 * (syn - obs[:, :, ddp.own_receivers]).pow(2).sum()
loss.backward()                   # vp.grad: this rank's tile; wavelet.grad: complete
```

There are no decomposed kernels here. Each rank runs the single-domain eager
rollout on its tile -- its owned cells of the grid plus `H` ghost cells on
each cut side (x strips in 2-D, an x/y tile grid in 3-D, as the stepped
class cuts) -- and every `exchange_steps` steps the ghosts are replaced by
the neighbours' cells: the x faces, the y faces and the corners in one
exchange, one message per neighbour. That exchange is an ordinary differentiable op (the
transpose of a copy is its backward), so the gradients come from autograd as
on one card, the wavelet's included, on the full tape or under chunk
checkpointing as the wrapped propagator does. The call and its semantics are
the stepped class's: `models` are either the global models replicated on
every rank (the gradient is then global-shaped with this rank's tile filled,
zero elsewhere; all-reduce to assemble) or this rank's tile of them
(`nx // px` columns; its gradient is complete), and a rank's part is the
derivative of the global misfit -- the sum over ranks of the misfits over the
receivers they own -- with respect to its tile. The wavelet's gradient, which
the stepped class does not return, is complete on every rank.
`ddp.own_receiver_indices` are the receivers this rank's record carries;
`ddp.gather_record(syn)` assembles the record on rank 0.

What a tile exchanges is probed, not declared. At the first call the reach
of one step along x is measured from the step's dependency structure, as is
the set of state tensors whose ghost columns feed an owned column within
`exchange_steps` steps (at `exchange_steps=1` the acoustic equation exchanges
its current pressure only, the staggered elastic one 7 of its 15 tensors), and
the halo width `H` follows. Then a self-check runs a few steps on the tiles
and on the global grid and compares. Grid-shaped tensors the equation holds --
the CPML profiles, a topography's rows per column, a profile of your own -- are
cut per tile. A new equation therefore needs no DD code: it is admitted when
its step is local in x, its backward is autograd's (or at least linear in
the cotangent), and every grid-shaped value it uses is a tensor whose last axis
is the grid's; otherwise the first call refuses it with the reason (a numpy
profile, a staggered `nx - 1` array padded at use, a row mean, a non-local
operator such as `ViscoAcoustic`'s and `PseudoelasticP`'s). The qP equations
(`AcousticVTI`, `AcousticTTI` and their Born versions) are refused in their
default `s_gradient='adjoint'`, whose hand-written backward takes S_n from the
adjoint field and so is not additive across tiles; `s_gradient='frozen'` and
`'exact'` are admitted. Every other FD equation in sweep is admitted (checked
against the single domain, record and gradients).

`exchange_steps` trades host time for halo width: `H = exchange_steps * R`
columns per cut side are computed twice, and the exchange's cost (a few host
ops and the messages) is paid once per `exchange_steps` steps. It must be a
multiple of the compiled call's steps (`EagerOptions.steps_per_call`); the
default is two calls in 2-D and one in 3-D, where the step is GPU-bound and a
longer interval only widens the halo on two axes and adds fields to exchange. Weak-scaling efficiency on V100 PCIe GPUs (one single-GPU
problem per tile, gradient, compiled steps):

| per-GPU tile | 2 GPUs | 4 GPUs |
|---|---|---|
| `Acoustic3D` 128³, `cpml_strips=False`, x strips / 2x2 grid | | 101% / 102% |
| `Acoustic3D` 176³, x strips / 2x2 grid | | 104% / 109% |
| `Acoustic` 1000², 4 shots, `exchange_steps=8` | 90% | 86% |
| `Acoustic` 1000², 1 shot, `exchange_steps=8` / `16` | 87% / 90% | 83% / 88% |
| `Elastic` 800², 1 shot, `exchange_steps=8` / `12` | 86% / 89% | 81% / 83% |

Above 100% in 3-D because the single-domain baseline carries a PML on every
face while a tile carries one on its outer faces only, with a 4-cell ghost on
a cut face (at `abcn=16`, a 128³ tile's array is 152x152x168 against the
single domain's 168³). Per computed cell the decomposed run costs 21% more
than the single domain at 128³ per GPU and 7% at 176³ on the 2x2 grid (9% and
4% on x strips): the ghost cells computed twice and the exchange. Each GPU
holds its tile's wavefields only; the self-check allocates a few global-grid
fields per state tensor once, at the first call.

`balance=True` cuts so that every tile computes the same number of cells. By
default the physical axes are split evenly, as the stepped class splits them;
an edge tile then also carries the absorbing pad on its outer face, an inner
tile a halo on both cut faces, and the slowest tile sets the pace. Balanced,
the edge tiles own fewer physical columns. The cut depends on the halo, which
the first call's admission measures, so `balance=True` takes the global
models (tile models would have to be cut before it is known), and
`ddp.local_extent()` gives this rank's physical tile after the first call.
It changes which rank owns which cells, not the result. It only matters with
three or more tiles along an axis (two tiles are both edge tiles). The
balanced widths are rounded to multiples of 4 cells (the compiled 3-D acoustic
step is ~10% slower on some other widths). Off by default: the step time of a
tile is not set by its width alone (exchange and synchronisation take a large
share), and the gain is small. Four V100s, four tiles along x, 8th order:

| Case | Absorbing pad | Even split (ms/step) | Balanced (ms/step) | Ratio |
|---|---|---|---|---|
| 3-D acoustic 128×128×512 | 16 | 1.70 | 1.75 | 0.97 |
| 3-D acoustic 128×128×512 | 32 | 2.58 | 2.46 | 1.05 |
| 3-D acoustic 176×176×704 | 32 | 5.24 | 5.07 | 1.03 |
| 3-D elastic 96×96×384 | 24 | 8.37 | 8.36 | 1.00 |
| 2-D acoustic 1000×4000 | 50 | 0.418 | 0.415 | 1.01 |

It pays only with a thick absorbing pad.

Refused: boundary saving (`memory=BoundarySaving(...)`; the tiles run the
full tape or checkpointing), the CPML strips layout (`cpml_strips=True`, the
3-D default: build the propagator with `cpml_strips=False`), registered custom
gradients, a free surface on a face other than the top, `return_wavefield`,
`halo="ipc"` and `overlap=` (the stepped drivers' switches; refused with a
message rather than ignored; the stepped class refuses `balance=` likewise), and receiver spreads that differ between
shots across a cut.

## Scope and limits

The stepped `impl="c"` tiles; the eager propagator's are in the section above.

- Equations: those with stepped compiled kernels — `Acoustic` (2-D),
  `Acoustic3D`, `AcousticVRZ3D`, `AcousticBorn` (2-D), `AcousticBorn3D`,
  `Elastic` (2-D) and `Elastic3D`. The LSRTM pair carries two coupled fields
  (background and scattered), and DD rotates and halo-exchanges both; its
  backward returns the reflectivity `mp` gradient and, when `vp` requires
  grad, the RWI `vp` gradient, and its forward always takes the serial
  step-then-exchange path (no comm/compute overlap). Everything
  else is refused at construction with an error that names the equation,
  including the **2-D** `AcousticVRZ` (stepped, but its backward has no
  coupling-exchange phases),
  `AcousticVTI`/`AcousticVTI1st`, `AcousticTTI`, `ElasticTTI`, `ElasticVRR`
  and `ViscoAcoustic`.
- Cuts: x strips in 2-D (`py=1`); x/y tile grids in 3-D. At spatial orders
  2-8 an `AcousticVRZ3D` tile cut on both sides of an axis needs at least
  `spatial_order` cells along it (refused at the first backward).
- Free surface: top face only under DD (a cut face can never carry one).
- Topography: not supported. `ModelParallel` refuses a propagator built with
  `topography=` (`NotImplementedError`): the tiles carry no surface, and
  boundary saving gives a wrong gradient under a per-column surface anyway.
- Gradient memory: boundary saving only. Each tile reconstructs its forward
  wavefield from saved boundaries, so a wrapped prop built with
  `memory=Full()` or `memory=Ckpt(...)` is refused at construction.
- One pending backward per instance. The DD backward reads the boundary ring
  and geometry an instance keeps for its last forward, so a forward —
  `torch.no_grad()` included — before the previous differentiable forward's
  backward has run raises `RuntimeError`. Call `backward()` after each
  forward (gradients accumulate), or free the earlier output.
- Gradients: the models' only. A `wavelet` that requires grad is refused
  (`NotImplementedError`) before the forward runs, for every equation: the
  domain-decomposed backward returns no source-wavelet gradient. Take that
  gradient from a single-domain `PropTorch` run.
- Boundary storage and dtype: see the table below.
- `BoundarySaving.tail_steps`: Acoustic 2-D/3-D (see
  [Propagators](propagators.md#boundary-tail-truncation-boundarysavingtail_steps));
  it composes with cpu staging.
- RTM: take the image through the gradient path (notebook
  [08](../notebooks/08_rtm_acoustic_marmousi.ipynb) shows how); there is no
  separate `rtm()` entry. Encoded supershots (a `(nsrc, nt)` wavelet) *are*
  supported: each tile keeps the rows of the sources it owns
  (`test/dd_encoded_check.py`).
- `ModelParallel(prop, topo, overlap=False)` takes the serial
  step-then-exchange path instead of the comm/compute-overlap forward — the
  bit-exact reference for the overlap and a production escape hatch. (It
  used to be `SWEEP_DD_DISABLE_OVERLAP=1` in the environment; a set variable
  is an error now.)
- `ModelParallel(prop, topo, halo="ipc")` sends the halo strips by CUDA
  peer-to-peer through torch symmetric memory instead of NCCL
  `batch_isend_irecv` (the default, `halo="nccl"`): each sender copies its
  strip straight into the neighbour's receive buffer and signals it on the
  stream, so no `torch.distributed` call is made per step. Results are
  bit-identical. It needs CUDA tiles, a torch with
  `torch.distributed._symmetric_memory` (2.9 tested) and every rank of a tile
  group on one node with peer access between the GPUs; otherwise the first
  DD run raises instead of falling back. Every rank must pass the same value.
  On two A100s the per-step cost of 2-D problems dropped by 40-60 %; 3-D
  problems are compute-bound and gain a few per cent at most.

## Boundary storage under DD

The per-tile ring inherits `storage` / `storage_dtype` from the wrapped prop,
but not every value is wired for the DD backward — which Python drives one
step per kernel call, unlike the monolithic loop the staged paths were built
for. What is refused, is refused loudly at the first backward:

| `BoundarySaving.storage` | Acoustic 2-D / 3-D, VRZ 3-D | Elastic 2-D / 3-D | LSRTM 2-D / 3-D |
| --- | --- | --- | --- |
| `"gpu"` (default, gpu-direct) | yes | yes | yes |
| `"cpu"` (pinned-host staging) | **yes** | **yes** — it used to raise | **yes** — the 3-D one used to race |
| `"disk"` | no — raises | no — raises | no — raises at construction |

Those three columns are the whole DD-admissible set: admission is *declared* by
the equation's `cuda_layout` (see `check_dd_admission`), and nothing else
declares it today. The refusal was lifted in the shared staggered driver, so the
rest of that family (DAS-mu, elastic TTI SG, elastic VR) is no longer blocked by
*this* check — but each still has to declare DD admission before any of it is
reachable under `ModelParallel`.

For the acoustic and VRZ equations, `storage="cpu"` also needs a **real cut**:
a single-tile `ModelParallel` (`world_size=1`) refuses it, because that path
reaches a reconstruction indexing that is not exercised by any multi-tile run.
Use gpu-direct there, or a plain `PropTorch` backward, which supports cpu and
disk staging as usual.

Every `storage_dtype` works with either storage, and with every one of them
(fp32, bf16, fp16, int8) the cpu-staged gradient is **bit-identical** to the
gpu-direct one and to a repeat of itself (`test/dd_offload_check.py` checks
both; `EQ=lsrtm` runs it for the LSRTM pair).

Staging trades PCIe traffic for GPU memory. The DD backward is driven one step
per kernel call from Python, so the boundary runtime — a copy stream beside the
compute stream, events either way, ring slots, and a prefetch of the next chunk
issued while the current one is still being consumed — used to be built and
destroyed inside every call, and a prefetch never survived to the step it was
meant for. It is now owned by a **`BoundarySession` held on the Python side**
for the whole reverse loop, so the stream and its ring events outlive the call
that created them. The LSRTM pair's backward takes the same session: built per
call, it left the next step's prefetch on a stream nothing waited on, and in
3-D that copy could land after a later step had refilled the ring slot, so the
cpu-staged gradient changed from run to run.

What it costs, measured on 2×V100 with a 3-D elastic tile
(`test/dd_elastic_staged_check.py`), every arm bit-exact against gpu-direct:

| ring config | backward | vs gpu-direct | peak boundary memory |
| --- | --- | --- | --- |
| gpu-direct (baseline) | 0.85 s | 1.00× | 1.03 GB |
| cpu, `transfer_interval=1`, `ring_buffers=1` | 2.28 s | 2.68× | **0.29 GB** |
| cpu, `transfer_interval=8`, `ring_buffers=2` | 3.19 s | 3.75× | 0.37 GB |
| cpu, `transfer_interval=32`, `ring_buffers=4` | 8.84 s | 10.38× | 0.81 GB |

So it is still an escape hatch rather than a default — 3.5× less boundary memory
for 2.68× the backward — but it is a working one, and **the knobs matter more
than the switch**: `32`/`4` is the worst point measured here, not the best.
The inherited cpu default (`transfer_interval=64`, `ring_buffers=1`, pinned)
was not measured. Start at `1`/`1` and raise only if profiling asks.

API details: [sweep.parallel reference](../api/parallel.md).
