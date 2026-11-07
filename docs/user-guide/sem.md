# SEM propagator

`sweep.sem.PropSEM` is the spectral-element counterpart of `PropTorch`. It
keeps the FD constructor and call — the same options with the same meaning
(`shape`, `abcn`, `free_surface`, `topography`, `memory`, `impl`, `device`),
the same geometry conventions, `models=[...]` input, `source_type` /
`receiver_type` selection, and a `(nshots, nt, nrec, ncomp)` seismogram — so
switching a script between the FD and SEM solvers is a construction-line
change, not a rewrite.

```python
import torch
from sweep.sem import PropSEM
from sweep.sem.equations import AcousticPotentialSEM2D

solver = PropSEM(
    AcousticPotentialSEM2D,
    shape=(141, 681), dh=25.0, dt=0.002,
    n_order=4,                    # GLL polynomial degree per element
    abcn=40,                      # C-PML in grid cells (ten elements), outside the model
    free_surface=True,            # the top; or edge names, as on PropTorch
    f_dominant=8.0, device="cuda",
)
seis = solver(wavelet, sources, receivers, models=[vp])   # (nshots, nt, nrec, 1)
```

One forward and one gradient, end to end, is
[Hello · SEM](../notebooks/43_sem_forward_backward.ipynb).

The options PropTorch has mean what they mean there:

- `shape` is the model, and `abcn` grid cells of C-PML are added outside it on
  every side without a free surface; the model is edge-extended into them and
  its gradient stops at the model edge (see [The model edge](#the-model-edge)).
  On a half-space of 10 m elements under a 15 Hz force, the default ten-element
  layer reflects 1.5e-4 of the record (relative L2) on the velocity-stress
  classes and 1.6e-4 on the displacement ones, whose C-PML is SPECFEM2D's (the
  records agree to 7e-7); SPECFEM2D's Stacey condition reflects 5.7e-2
  (`test/sem/specfem_compare/run_compare_absorb.py`), and the SEM has no Stacey
  condition;
- `free_surface` — `True` (the top), edge names (`"top"`, `["top", "left"]`),
  a per-edge mask or an `{edge: bool}` dict; default `False`. On the
  velocity-stress solids (elastic and Biot) a C-PML side between two free
  faces is refused -- free top, left and right over an absorbing bottom, or
  free top and bottom with absorbing sides: the solid between the free faces
  is a plate, whose backward-propagating Lamb modes a PML amplifies, and the
  run grows without bound after a few seconds. Leave one face of the pair
  absorbing;
- `topography` — the surface row of every column, `(nx,)` in 2-D and
  `(ny, nx)` in 3-D, fractional rows allowed: a free surface on top, which the
  curved mesh follows (it passes through the rows at the element nodes);
- `memory` — `Full()`, `BoundarySaving(storage_dtype=...)` or
  `Ckpt(chunks=...)` from `sweep.propagator` (see
  [Boundary saving](#boundary-saving)). Without it PropSEM takes PropTorch's
  default strategy: on `impl='c'` boundary saving where its kernels have it,
  checkpoints in chunks of 64 steps for the velocity-stress classes, else the full
  tape; on eager checkpointing in chunks of 8 steps; `memory_strategy` reports
  `'full'`, `'boundary'` or `'ckpt'`;
- `impl` — `None` runs the CUDA kernels where the equation has them and the
  device is CUDA, eager otherwise; `'eager'` / `'c'` choose;
- `eager_options` — `EagerOptions(...)` from `sweep.propagator` (or its dict),
  for `impl='eager'`: the step is compiled with `torch.compile` unless
  `use_compile=False`, with its `compile_mode`, `compile_dynamic`,
  `compile_backend` and `compile_fullgraph` (see
  [Speed and memory](#speed-and-memory-on-impleager));
- `nt` is optional — the wavelet's length; checked when given — and `device`
  may be spelled `dev`.

Two SEM-specific constraints to know up front:

- every grid axis must tile into elements: `(n - 1) % n_order == 0` for each
  entry of `shape` — this includes the width of any cropped sub-model you
  build (see [local windows](#local-window-fwi) below);
- the absorbing layer is whole elements: `abcn` is a multiple of `n_order`
  (the default, `None`, is ten elements).

Equations are discovered automatically — `sweep.sem.equations.describe()`
prints the full table (mesh family, C-PML support, fields, receiver
components) for every equation class. `sweep list equations` lists them
next to the finite-difference ones, `sweep show <Equation>` prints one's
fields, source and receiver types and the `models=` it takes (see
[CLI](cli.md)), and the API is in the [SEM reference](../api/sem/index.md).

## Curved interfaces

A grid model is interpolated to the GLL nodes, so a material step that does not fall on
element boundaries is smeared over the elements it crosses. With `interfaces`, the mesh follows
internal interfaces instead, as SPECFEM2D's internal mesher follows those of its
`interfaces.dat`.

- **The interfaces.** Each is given like `topography`: its row in every column, `(nx,)` in
  2-D and `(ny, nx)` in 3-D, fractional rows allowed, top-down.
- **The layers.** The surface (or the topography), the interfaces and the model's last row
  bound the layers, and each layer gets whole element rows. `interface_elements` sets them,
  top-down and summing to `(nz - 1) / n_order`; by default the layers share them by their mean
  thicknesses. In every column of control nodes, a layer's rows are graded linearly between its
  two interfaces.
- **The absorbing pads** join the layers next to them, and the interfaces stay flat across the
  lateral pads.

Interfaces need the curved meshes: the Quad9 classes in 2-D and the Hex27 ones (the
`...Deformed` classes) in 3-D. `element_layer` gives each element's layer, and
`layered_model(values)` gives a GLL-native model of one value per layer, which steps exactly at
the interfaces:

```python
from sweep.sem import AcousticElasticSEM2D, PropSEM

seafloor = (600.0 - 150.0 * np.exp(-((x - 1800.0) / 300.0) ** 2)) / dh   # rows, z down
solver = PropSEM(AcousticElasticSEM2D, shape=(nz, nx), dh=dh, dt=dt, free_surface=True,
                 interfaces=[seafloor], interface_elements=[60, 240],
                 source_type=["chi"], receiver_type=["p", "vx", "vz"])
models = [solver.layered_model(v)
          for v in ([1500.0, 3000.0], [0.0, 1500.0], [1020.0, 2400.0])]     # vp, vs, rho
seis = solver(wavelet, sources, receivers, models=models)
```

The values may be tensors that require gradients. Against SPECFEM2D's mesh of the same
interfaces, see [Against SPECFEM2D](#against-specfem2d).

In 3-D the mesh is `build_layered_hex27`'s, and a sea floor is the case that needs it:
`BiotVSSEM3DDeformed` models the water and a dry rock in one mesh (its fluid and dry elastic
regions, coupled as SPECFEM3D couples its acoustic and elastic domains), but each element has
to be wholly one medium, so a curved sea floor has to run along element faces:

```python
from sweep.equations.biot_params import biot_elastic, biot_fluid
from sweep.sem import BiotVSSEM3DDeformed, PropSEM

solver = PropSEM(BiotVSSEM3DDeformed, shape=(nz, ny, nx), dh=dh, dt=dt, free_surface=True,
                 interfaces=[seafloor], interface_elements=[3, 7],      # seafloor: (ny, nx) rows
                 source_type=["p"], receiver_type=["vx", "vy", "vz", "p"])
water, rock = biot_fluid(1020.0 * 1500.0 ** 2, 1020.0), biot_elastic(2400.0, 1300.0, 2000.0)
models = [solver.layered_model(c) for c in zip(water, rock)]
seis = solver(wavelet, sources, receivers, models=models)
```

The control nodes of the Hex27 elements, mid-nodes included, lie on the interfaces at their
columns, so the faces between the layers follow the interfaces' curvature. SPECFEM3D's
internal mesher takes only the corners from its interfaces and puts the mid-nodes of its
27-node elements at their averages, which makes its elements straight-sided. On that straight-sided mesh of a curved interface
(20 m of relief over 320 m) `ElasticVSSEM3DDeformed` agrees with SPECFEM3D to 3e-14; on its own
mesh, whose faces follow the interface, the records differ from SPECFEM3D's by 9e-3, which is
the difference between the two meshes (a 10 m element's straight face departs from the curve
by up to 0.1 m there). See [Against SPECFEM3D](#against-specfem3d). Notebook 61 shows a
seamount in 3-D.

## Speed and memory on `impl='eager'`

`impl='c'` advances a time step with a few fused CUDA kernels. `impl='eager'`
runs the same step as PyTorch operations, about 110–260 kernels per step for the
C-PML equations, so on a GPU its time at small and medium sizes goes to the host
work of issuing them, and at large sizes to the memory traffic between them.
Three things speed it up, and they combine:

- **Compiling the step.** By default the step is compiled with `torch.compile`,
  as on PropTorch; `eager_options=EagerOptions(use_compile=False)` runs it as it
  is, and the other `compile_*` options of `EagerOptions` pass through. Inductor
  fuses the pointwise work, and when it compiles for CUDA the GLL
  reference-derivative contractions are written out as multiply-adds that it
  fuses with the gather before them and the pointwise work after them. The 3-D
  deformed classes with a C-PML keep them as batched matrix products: there the
  inverse Jacobian makes every physical derivative read all three reference
  directions, and the fused form, one large kernel, ran 1.4 to 6 times slower.
  The first
  call compiles, and so does the first call with a new grid size, shot count,
  dtype or set of models that require grad; each says so on stderr before it
  starts, and later calls reuse it. With gradients, the first steps of a rollout
  run uncompiled until the fields' `requires_grad` flags stop changing, as on
  PropTorch, so that a rollout compiles one graph. On the CPU, Inductor
  builds C++ with `torch._inductor.config.cpp.cxx` (seeded from `$CXX`), which
  must accept `-std=c++20` (g++ 10 or newer); when the compiler fails, the call
  warns once and runs the step uncompiled. The vmapped multi-shot batch of
  boundary saving runs uncompiled, and the CUDA-graph modes (`reduce-overhead`,
  `max-autotune`) go with `Full()` only.
- **`memory=Ckpt(chunks=K)`** — eager's default, with `K = 8` — checkpoints a
  single shot, or each shot of a per-shot model batch: the forward pass keeps one state every `K` steps, and
  the backward pass recomputes each chunk under autograd. Peak memory is `nt/K`
  states plus one chunk's tape. On CUDA each chunk is captured once per call as
  a CUDA graph and replayed `nt/K` times, so the time loop issues no kernels
  from the host. First derivatives only (`create_graph=True` raises; second
  derivatives need `Full()`).
- **The C-PML on the absorbing band only.** The elastic C-PML equations
  (`ElasticSEM2DDeformed`, `ElasticVSSEM2DDeformed`, `ElasticVTISEM2DDeformed`,
  `ElasticTTISEM2DDeformed`, `ElasticVSSEM3D`, `ElasticVSSEM3DDeformed`) run the bare
  operators on the interior elements and the C-PML terms on the band elements, as the CUDA
  kernels do, on meshes with at least 40000 interior elements (an 801 × 1601
  grid has 80000). On smaller meshes the extra element pass costs about as much
  as it saves, or more, so they keep the full-mesh step;
  `equation_options={"band_split": True}` (or `False`) overrides the choice.

Measured on an RTX 6000 Ada (48 GB) for one shot, order 4, fp32, a free surface
on top and the C-PML on the other sides, `nt` from 64 to 640 — forward / forward +
backward in ms per time step, and in parentheses the peak GPU memory of the
forward + backward call in GB:

| ms per step (peak GB) | 2-D elastic, 201 × 401 | 2-D elastic, 801 × 1601 | 3-D elastic VS, 49³ |
| --- | --- | --- | --- |
| `impl='c'`, `Full()` | 0.025 / 0.060 (3.2) | 0.24 / 0.89 (20.4) | 0.30 / 0.83 (20.9) |
| `impl='c'`, `BoundarySaving()` | 0.025 / 0.12 (1.2) | 0.24 / 1.39 (3.4) | — |
| eager, `use_compile=False`, `Full()` | 1.92 / 7.98 (8.3) | 2.65 / 11.33 (37.8) | 2.38 / 9.37 (35.7) |
| eager, `Full()` | 0.59 / 1.85 (6.8) | 0.67 / 2.65 (21.7) | 0.66 / 2.41 (16.3) |
| eager, `use_compile=False`, `Ckpt(chunks=8)` | 0.41 / 1.47 (1.3) | 1.48 / 5.65 (3.9) | 0.80 / 3.36 (7.9) |
| eager, `Ckpt(chunks=8)` (the defaults) | 0.072 / 0.33 (1.2) | 0.48 / 2.17 (3.5) | 0.42 / 1.53 (7.7) |

On a Hex27 mesh, `ElasticVSSEM3DDeformed` runs the same CUDA time loop as
`ElasticVSSEM3D`; its four element kernels apply the full 3 × 3 inverse Jacobian at
every GLL node instead of the Hex8 kernels' one scale factor per axis. On an A100
(order 4, C-PML on five sides, 10³ and 16³ elements, 400 steps) that took 1.15–1.3×
the Hex8 time on the same number of elements, at the same peak memory.

The velocity-stress classes on `impl='c'` (2-D and 3-D, isotropic, VTI / TTI and
visco-elastic) have no boundary saving; their adjoint needs, per step, the velocities
and the stretched velocity gradients, several times the state. `memory=Full()` keeps
them for the whole record. `memory=Ckpt(chunks=K)`, PropSEM's default there with
`K = 64`, keeps the state at the start of every `K`-step chunk instead and reruns each
chunk in the backward pass (one more forward pass in all), with the same records and
gradients up to the kernels' run-to-run rounding (`test/sem/test_sem_c_vs_ckpt.py`).
On an A100, one forward and backward call with the default options:

| | `impl='eager'` (compiled, `Ckpt(8)`) | `impl='c'`, `Full()` | `impl='c'`, `Ckpt(64)` |
| --- | --- | --- | --- |
| `ElasticVSSEM2DDeformed`, 401 × 801, 2000 steps | 2.28 s, 7.3 GB | 0.81 s, 23.3 GB | 1.16 s, 1.8 GB |
| `ElasticVSSEM3DDeformed`, 65³, 1000 steps | 9.84 s, 12.5 GB | 0.91 s, 36.4 GB | 1.32 s, 4.4 GB |

Where the equation has CUDA kernels (the `impl="c"` column of the
[equation table](equations.md#sem-family-spectral-elements)), `impl='c'` is the
faster backend: in the table the fastest eager configuration takes 1.4–2.9× as
long per forward step, and 1.5–2.8× as long per forward + backward step as
`impl='c'` with `BoundarySaving()` (1.8–5.5× against `Full()`). `impl='eager'`
runs everything else: the equations without CUDA kernels
(`AcousticSEM2DDeformed` and `ElasticSHSEM2DDeformed`), the CPU, fp64,
[wavefield snapshots](#wavefield-snapshots) and second derivatives
(`create_graph=True` with `Full()`). On a GPU, its defaults — the compiled
step and `Ckpt(chunks=8)` — are its fastest configuration for single shots, at
about the memory of `impl='c'` with `BoundarySaving()`. Longer chunks cost more
there: the CUDA graphs are rebuilt per call, at a cost that grows with the
chunk (2-D elastic at 201 × 401, nt = 1000, one forward + backward call on an
A100: 0.51 / 0.61 / 0.81 s for chunks of 8 / 32 / 100 steps, 1.24 s with
`Full()`). On large 3-D meshes the step is bound by the GPU rather than by
issuing kernels, the graphs save little, and the recomputation can make `Ckpt`
slower than compiled `Full()` (1.35× for 3-D acoustic at 81³ on an H100), which
is the faster choice there where its tape fits in memory.

## Shot batching on `impl='c'`

A multi-shot call — `sources` of shape `(nshots, ndim)` — executes as **one
batched CUDA kernel sequence** (`gridDim.y = B`) instead of a per-shot Python
loop. The SEM CUDA backend is launch-bound (many small kernels per time step),
so folding `B` shots into each launch is the main throughput lever:

| B | forward (vs loop) | forward + backward (vs loop) |
| --- | --- | --- |
| 4 | 1.8× | 1.7× |
| 8 | **2.2×** | **1.7× (save-all) / 2.4× (boundary saving)** |
| 16 | **2.4×** | 2.2× (boundary saving) |

(RTX 6000 Ada, 153×401 grid, order 4, nt=1000, ms/shot; the forward keeps
improving with B, the backward saturates around **B ≈ 8** once the grid fills
the SM occupancy.)

The batch engages when all of these hold — otherwise the call falls back to
the per-shot loop, which is equally correct, just unamortised:

- the equation's CUDA path is batch-capable (today: the three 2-D C-PML
  paths — acoustic-potential, the coupled fluid-solid `AcousticElasticSEM2D`
  whose ~15–20 launches per step made its per-shot loop the most launch-bound
  of all, and the standalone elastic `ElasticSEM2DDeformed`);
- every shot has the **same receiver rows** (a fixed spread, or a streamer in
  window-local coordinates);
- `memory` is `None` or `'boundary'`;
- the wavelet is `(nt,)` shared or `(nshots, nt)` per-shot.

Gradient semantics are unchanged from the loop: a model tensor shared by the
whole batch accumulates the **sum of the per-shot gradients** — exactly what
autograd gives a tensor reused across a loop — and this is pinned by tests
against the per-shot reference at the CUDA-atomics noise floor.

The fluid-solid and elastic batches carry every receiver component (and the
derived `p`/`vx`/`vz`) through one kernel sequence, and always share **one
model** across the batch — their kernels have no per-shot-material axis, so a
per-shot model stack declines the batch and runs the per-shot loop instead
(only the acoustic-potential path takes per-shot materials, `Bm == B`). Measured
on an OBN-FWI-sized problem (2584 elements, order 4, nt=1800, B=8, boundary
saving, RTX 6000 Ada), one forward+backward iteration:

| mode | ms/iter | vs loop |
| --- | --- | --- |
| per-shot loop | 3804 | — |
| **batched** | **1173** | **3.2×** |
| batched + CUDA graph | 1052 | 3.6× |

(The standalone elastic batch measures **2.2×** on the same problem size —
fewer launches per step than the coupled system, so less dispatch to
amortise.)

The optional CUDA-graph layer on top captures the whole batched loop once and
replays it (`SEMGraphGradRunner(prop, memory='boundary', batch=props)` — one
graph covers every shot). After batching it only adds ~1.1×: the batch has
already amortised most of the launch dispatch. It stays opt-in — a graph pins
a private memory pool and freezes geometry/nt at capture.

## Per-shot models

`models=[vp]` accepts either a shared model or a **per-shot batch** with a
leading shot axis:

```python
vp_b = torch.stack([vp_for_shot_0, vp_for_shot_1, ...])   # (B, nz, nx)
seis = solver(wavelet, sources, receivers, models=[vp_b]) # shot b uses vp_b[b]
```

- Dropping the batch axis broadcasts one model over all shots (the
  "same model, different source positions" case) — on `impl='c'` this reads
  the single model with stride 0 rather than copying it per shot.
- Entries mix freely: `models=[vp_b, rho]` gives each shot its own `vp` and a
  shared `rho`.
- The number of shots and the model batch must agree (`B == nshots`).
- Gradients stay **per shot**: `vp_b.grad[b]` receives only shot `b`'s
  contribution (zero cross-shot leakage, pinned by tests).

Two batch-wide conventions make `impl='c'` (one batched kernel sequence) and
`impl='eager'` (a per-shot loop) return the same numbers:

- **one C-PML profile for the whole batch**, built with the batch-max vp —
  the damping profile is per-mesh in the batched kernels, so eager must not
  give each shot its own damping strength;
- a broadcast model's gradient is the sum over shots (above).

On `impl='eager'`, per-shot models run as a loop over shots (the vmapped
fast path applies to the shared-model batch only); results match `impl='c'`
at the run-to-run noise floor either way.

## Boundary saving

`memory=BoundarySaving()` runs the backward pass by **reconstructing** the wavefield
instead of keeping the full autograd tape: the forward stores only the C-PML
**band ring** per step (the absorbing band is the one region the reverse time
march cannot recompute) plus the final field levels, and the backward marches
the interior back exactly. Peak memory goes from O(nt · n_global) to
O(nt · |band|) — the seismograms are unchanged and the model gradient is
pinned to the full tape at the run-to-run noise floor by the test suite.

```python
from sweep.propagator import BoundarySaving

solver = PropSEM(..., memory=BoundarySaving())                       # fp32 ring
solver = PropSEM(..., memory=BoundarySaving(storage_dtype="int8"))   # 4× smaller ring
```

Coverage differs by backend — the [equation table](equations.md#sem-family-spectral-elements)
carries the per-equation column:

| Backend | Coverage |
| --- | --- |
| `impl='eager'` | every equation but the dissipative ones (visco-elastic, visco-acoustic, Biot), which refuse it: stepping their attenuation or drag back amplifies the round-off |
| `impl='c'` | the five C-PML paths: `AcousticPotentialSEM2D`, `ElasticSEM2DDeformed`, `AcousticElasticSEM2D` (each requiring their C-PML), `AcousticSEM3D`, `AcousticSEM3DDeformed` |

Elsewhere `impl=None` runs the equation on eager, and `impl='c'` raises
`NotImplementedError` when the `PropSEM` is built — the kernels once accepted
it and silently ran the full tape, which looked like it worked.

Knobs:

- `BoundarySaving(storage_dtype='bf16' | 'int8')` — low-precision rings,
  wired on **every** C-PML path (acoustic 2-D/3-D, elastic, fluid-solid, and
  their decomposed twins). What it costs differs sharply by path — see **Ring
  precision** — and the constructor's `bs_store_dtype_elem` sets a separate
  dtype for the ELEMENT rings (`impl='c'`). In the
  shot batch the int8 quantisation scale is **per (field, step, shot)**, so one
  loud shot cannot degrade a quiet shot's reconstruction. `'fp16'`, which
  PropTorch takes, is refused: a plain fp16 ring flushes the SEM's small field
  values to zero (2-D elastic gradients came out 15% off), where `'bf16'` keeps
  fp32's range.
- `bs_band_mode='full' | 'rim'` (constructor) — what the ring covers: the whole
  band (default) or only its first element layer, the one touching the model.
  Both give the full-tape gradient over the whole model. See **Rim mode**.

### One index set per field

The acoustic paths (2-D potential, 3-D, and their DD strips, `impl='c'` and
`'eager'`) do not ring every saved field on the whole band. A C-PML memory
field `psi_x` / `zeta_x` is updated as `b_x · (…) + a_x · psi_x` from a zero
start, so it is **exactly zero** wherever `b_x == 0` — on the whole band except
the two x-slabs — and storing it there stores zeros. Each memory field
therefore rings only the band nodes its own coefficient is non-zero on
(`band_idx_x/y/z` in the consts); the wavefield itself keeps the whole band.
The union of the three slabs is about 2.4× any one of them, so the seven 3-D
rings shrink from `7 · |band|` to about half (measured: 244,699 → 122,077
node-slots on a 33³ grid; DD rank memory for forward + backward 921 → 581
B per global node at 161³). In `'rim'` mode they cover the band's first
element layer only, like the wavefield ring. Nothing changes in
what is reconstructed — `test_sem_bs_ring_split.py` checks on the stored
bytes that the dropped entries were all zero, and the eager path is bitwise
identical to the old layout.

### The memory fields live on their slabs

The same fact shrinks the fields themselves, not just their rings. On the
3-D acoustic C path (`AcousticSEM3D`, `AcousticSEM3DDeformed`, single domain
and every DD strip) the fifteen per-node arrays that carry `psi_d`, `zeta_d`,
their double buffers and `a_d·psi_d` used to be `n_global` long each — 60
bytes per node holding zeros on ~93% of a production grid. Each is now
stored only on the index **slab** of its axis, the planes on which some node
has `b_d ≠ 0`; the node numbering `g = (iz·ny + iy)·nx + ix` gives the slot
arithmetically from one short table per axis, so there is no per-node map.
The kernels' floating-point work is unchanged (the SASS of both layouts has
the same multiset of fp32 arithmetic instructions; the additions are the
integer divisions of the slot arithmetic), and what they no longer store is
exactly the `+0.0` the full arrays held, checked bitwise on the full-length
tapes with the wave inside the layers. Measured on one RTX 6000 Ada rank:

| forward-only | full layout | slabs |
|---|---|---|
| 193³ nodes (`nex=48`) | 180.2 B/node | **131.0** B/node |
| 257³ nodes (`nex=64`) | 177.2 B/node | **125.9** B/node |
| forward + backward, boundary saving, 161³ | 580.6 B/node | 554.3 B/node |

The saving grows with the grid (the slabs are a fixed number of planes); at
the 1000³ scale it is ~58 B/node. The backward's six restores and three
products follow the same layout. A save-all tape (no boundary saving,
gradient requested) keeps the full layout, since the tape copies whole
arrays — production runs boundary saving, where the slabs apply.

The backward's twelve adjoint carries (`lpsi_d`, `lzeta_d` and their double
buffers, 48 B/node) live on the slabs too. They are a different case: off
the layer they are **not** zero — the adjoint update accumulates
`a_d·(λ + lap)` there with `a_d = 1` — but they are dead, consumed only as
`b_d·λ` with `b_d = +0.0`. Dropping them changes no value; it can change the
sign of an exact-zero product, which nothing downstream can see. The check
is bit for bit where that is attainable: one reverse step is run on the same
input in both layouts (the full copy carrying random values off the slabs)
and the node kernels' outputs compared bitwise, differing at most in ±0.0.
The assembled adjoint sums `lapx/y/z` stay full-length: they cross the DD
cuts. The 2-D acoustic path keeps its full-length carries, but no longer
allocates three scratch arrays (`apsi_x`, `apsi_z`, `w_sum`, 12 B/node) that
its C entries never read or wrote — proven by filling them with NaN.

### One driver for 3-D acoustic

On CUDA the 3-D acoustic C-PML families (`AcousticSEM3D`,
`AcousticSEM3DDeformed`) have a single driver: the strip driver of
`sweep.sem.dd`. `PropSEM.forward` runs it with one strip, placing sources
and receivers as point stamps (Lagrange weights over one element, since
grid positions are not GLL nodes); `SEMModelParallel` runs the same driver
with `px` strips, and the low-level `SEMCudaPropagator.forward` (a built
equation, no `PropSEM`) reaches it through `SEMModelParallel.for_equation`
and `run`. The fused single-domain driver it replaced built the
global equation first and was measured slower and heavier (129³, nt=60:
backward −15% to −22%, peak memory −43% to −52% with the strip driver).
Only at very small grids does the Python-stepped loop show its ~40 µs per
step. Per-shot model batches (a leading batch axis on the model) still take
the equation path.

### Where the CUDA drivers live

Each equation family's autograd Functions sit in one module of
`sweep/sem/drivers/` (`acoustic2d`, `acoustic3d`, `elastic2d`,
`fluid_solid`, `vs2d` for the 2-D velocity-stress family, and `bare` for the
paths without a C-PML and the 3-D velocity-stress classes, whose chunked
checkpoints both velocity-stress modules take from `vs_chunks`) — the fused
single-domain Function (the C entry runs the time loop) next to the strip
Function `SEMModelParallel` steps (Python runs the stepped entries with one
exchange per assembly). A module names
its struct field families once and fills the forward and backward input
structs through `sweep/sem/cdrive.py`: `fill` copies the constant tables,
`Alloc` hands out the per-step buffers (families a C entry re-zeroes
together are the rows of one buffer, so one memset covers them),
`check_filled` refuses to call a C entry with a field left unset (an unset
tensor is a null pointer on the other side), and the forward's saves reach
the backward by name (`take` / `give`) rather than as a positional tuple.
`prop_c` and `dd` re-export the Functions, so `SEMCudaPropagator`,
`SEMModelParallel`, the shot-batch path and the graph runner see no change.
The constant tables a Function reads come from one builder per family: on
the fused side `SEMCudaPropagator._consts_*` on top of `_ConstsBase` (the
receiver gather, the source patterns, the ibool and geometry casts and the
scalar tail, gathered once per call), on the strip side
`SEMModelParallel._consts_*` on top of `_Strip` — each a single dict
literal with the source of every C-side value beside its key.

The kernels are entries of the same torch-free core as the FD equations
(`libsweep_core.so`): kind 3 of the C API (`core/capi.h`), called through
`sweep_sem_call` with the step index the DD phase entries take. Their input
structs are generated: `utils/gen_input_core.py` turns the torch-typed specs
`cuda/equations/sem2d/sem_types.h` and `sem3d/sem3d_types.h` into the twins the
drivers take (`core/sem_input_core.h`), their ctypes mirror and the layout
probe. On the Python side an input (`sweep._C.SemForwardInput` and the rest,
`sweep/backend/c/sem.py`) writes each assignment straight into its twin, so a
call hands the core a pointer — the C-PML paths call the DD phase entries
several times a step, and re-packing a struct of up to ~230 fields per call
would cost more than the step. Add a field to a spec (never rename or remove
one) and re-run the generator; `test/test_input_core_generated.py` fails on a
stale twin.

### The 3-D save-all backward keeps one tape

Without boundary saving the 3-D acoustic C path used to tape seven
fields per step (`u`, `psi_d`, `zeta_d`; 28 B per node per step) and, at
every reverse step, run the forward's assembly again — `a·psi`, nine
memsets, the element kernel, and under DD an exchange of the nine assembled
fields — only to recompute `w_sum^k`. In 3-D the material gradient enters
through `w_sum` alone and the adjoint element kernel reads no forward
field, so `w_sum^k` is the only forward quantity the backward needs. The
forward now saves it (one `(nt, n_global)` tape, 4 B/node/step) and the
backward rebuilds nothing: the reverse step is the adjoint node algebra,
the adjoint element kernel and `combine`, with one DD exchange per step
instead of two. The memory-field slabs apply in this mode too. What the
old backward recomputed was the forward's `w_sum` up to the assembly's
run-to-run floor; the tape hands it the forward's own bits, and given the
same `w_sum^k` the reverse step is bit-identical (checked kernel by
kernel). The old seven-tape path stays behind `prop_c._WSUM_TAPE = False`.

This moves the line between the two memory modes: save-all now costs
`4·nt·n_global` bytes, while boundary saving keeps its rings (a fraction
of that) plus per-node reconstruction buffers. On small grids the one-tape
save-all is the smaller of the two (measured: 335 MB vs 585 MB on a 33×49×129
DD test grid); boundary saving wins once `nt·n_global` no longer fits,
which is every production run.

### The 2-D save-all backward keeps two tapes

The same trade one dimension down. The 2-D acoustic C-PML save-all backward
used to tape five fields per step (`chi`, `psi_d`, `zeta_d`; 20 B per node
per step) and, at every reverse step, re-run the forward's assembly on the
taped state only to recompute `w_sum^k` — whose consumers are the combine
kernel and `grad_inv_Mg`, which every driver discards. Unlike 3-D, the chi
tape has to stay: the adjoint element kernel images `inv_rho` from `chi^k`.
So the forward now stores `w_sum^n` as well and drops the four memory-field
tapes (8 B per node per step), and the reverse step loses one element
kernel, six memsets and, under DD, the exchange of the six assembled
fields — one exchange per reverse step instead of two. `grad_inv_Mg` is
then never accumulated (it needs the assembly) and stays zero. Boundary
saving is untouched: its reverse march needs `w_sum^k` rebuilt on the fly.
The five-tape path stays behind `prop_c._WSUM_TAPE_2D = False`.

### Rim mode

`bs_band_mode='rim'` rings only the band's **first element layer**, the one
that touches the model, instead of the whole band: one element deep instead of
`abcn` cells. That layer is all the model gradient reads. The reverse march
rebuilds the model's own wavefield once the layer's inner edge (the model's
edge line) is restored, and the model's edge nodes share their material with
the layer's elements, whose stiffness and C-PML memories enter the edge nodes'
update. Deeper band elements hold no model node and are cut from the gradient
(the band is an edge extension of the model, see `_interp_to_gll`), so their
wavefield is never needed. Rim and full give the same gradient over the
**whole model**, its edge included. It is honoured on every path: the
single-domain acoustic/elastic/coupled propagators and every DD strip.

Measured on one shot 300 m from the left edge of a 97×193 grid (2-D, `abcn=20`,
4-sided band) and a 41×41×65 grid (3-D, `abcn=12`), against `Full()` over the
whole model (fluid-solid: element-local models, over the elements inside the
model):

| path | peak MB: `Full()` / full band / rim (old rim) | `1 - cos`, rim (old rim) |
|---|---|---|
| 2-D acoustic potential, `impl='c'` | 755 / 553 / 114 (50) | 1e-11 (0.35) |
| 2-D acoustic potential, eager | 403 / 555 / 116 (52) | 1e-11 (0.82) |
| 2-D acoustic potential, DD px=2 | 766 / 556 / 121 (56) | 5e-11 (0.35) |
| 2-D elastic, `impl='c'` | 4155 / 1773 / 1525 (1471) | 5e-13 (2e-5) |
| 2-D elastic, eager | 4370 / 7140 / 6893 (6838) | 3e-12 (0.11) |
| 2-D elastic, DD px=2 | 4198 / 1784 / 1532 (1477) | 3e-12 (4e-3) |
| fluid-solid, `impl='c'` | 5107 / 1892 / 1235 (1177) | 2e-11 (7e-8) |
| fluid-solid, DD px=2 | 5135 / 1927 / 1263 (1213) | 1e-11 (4e-8) |
| 3-D `AcousticSEM3D`, `impl='c'` | 2103 / 5875 / 1330 (312) | 2e-16 (2e-5) |
| 3-D `AcousticSEM3D`, eager | 2363 / 6123 / 1561 (544) | 9e-15 (2e-4) |
| 3-D `AcousticSEM3D`, DD px=2 | 2199 / 6012 / 1418 (388) | 2e-15 (5e-5) |
| 3-D `AcousticSEM3DDeformed`, `impl='c'` | 2459 / 6291 / 1723 (656) | 8e-16 (2e-5) |
| 3-D `AcousticSEM3DDeformed`, eager | 4316 / 2065 / 758 (551) | 8e-13 (5e-4) |

The full band is the same at round-off on every row. The rim stores about
twice what the old one did (the band's first element layer and its C-PML
memories, where the old one stored four node lines and no memory) and still
4-5x less than the full band on the acoustic paths. On these small test grids
the 3-D full band is most of the mesh, so boundary saving there costs more
than the tape; that is unrelated to rim.

An element belongs to the band when `b_x`, `b_y`, `b_z` (or the elastic
`node_in_pml`) is non-zero at one of its *interior* GLL nodes. Interior nodes
only, because on the band's inner face the profile is zero only up to the
coordinate round-off (measured `b_z = -3.8e-16` on a top edge line). The
derivative profiles `dbx_dx` … come from `np.gradient` and are non-zero on
the model's edge line outright. The first rim used them: it counted the edge
line as band, zeroed it in the reverse march and stored the model's outermost
element instead of the band's first. That element's gradient was wrong (2-D acoustic
`1 - cos` 0.35 uncropped, exact two elements in), and the fluid-solid "2e-5
plateau" this page used to report was the same error.

**Element rings are exempt** and always cover the whole band. The elastic C-PML
memory fields live only inside it, and zeroing them corrupts the adjoint
entering the *interior*, not just the band gradient — dropping them takes even
full-band node mode to cos 0.755. So rim shrinks the node rings only: a big win
where the rings are node-only (acoustic), a small one on elastic and coupled,
where the element rings are the bulk of the bytes.

A model given element-local (GLL-native) covers the band too, and those
entries are not reproduced beyond the first layer; crop the band before
comparing such a gradient.

Asking for `'rim'` on a mesh whose C-PML band covers every node raises — there
is no model node for a first layer to touch, and boundary saving would silently
ring nothing.
Small 3-D test grids hit this routinely (measured: band 5746 of 5746 nodes).

### In the shot batch

Boundary saving rides the batched acoustic and fluid-solid paths: the rings
become `(nt, B, nband)` instead of the save-all histories `(nt, B, n_global)`
— 10–20× less history memory at the same wall-clock, which is what keeps
large batches affordable in forward + backward. (On the fluid-solid path this
is also what makes the batch *usable* for FWI at all: the eight-shot save-all
tape is tens of GB.)

### Coupled fluid-solid

`AcousticElasticSEM2D` stores both domains' rings (the acoustic chi/psi/zeta
node ring, the elastic u ring + six element-local history rings) **plus a
full-state re-anchor checkpoint every ~50 steps** — the coupled reverse map
can amplify roundoff, and the checkpoints bound any drift to one segment.
With the rings the measured anchor-free drift is already flat in nt; the
anchors stay on as insurance. Gradient pinned to the save-all tape and to the
eager full tape at the noise floor (`test/sem/test_sem_fs_bs.py`).

## Local-window FWI

The FD "local model window" pattern — propagate every shot in a cropped
sub-model around its own aperture instead of the full model — needs no
dedicated machinery on SEM: it is per-shot models plus tensor slicing, and
autograd scatter-adds overlapping windows back onto the full model.

```python
# One window shape W for all shots (a streamer is translation-invariant),
# rounded UP to the SEM element-tiling constraint:
W += (-(W - 1)) % n_order

solver = PropSEM(AcousticPotentialSEM2D, shape=(nz, W),         # window-sized
                 memory=BoundarySaving(), ...)

def batched_syn(vp_full, wavelet):
    crops = torch.stack([vp_full[:, x0[s]:x0[s] + W] for s in range(nshot)])
    return solver(wavelet, loc_sources, loc_receivers,   # window-local coords
                  models=[crops])

loss = misfit(batched_syn(vp_full, wavelet), obs)
loss.backward()          # overlapping windows scatter-add into vp_full.grad
```

Because the window-local geometry is identical for every shot, the receiver
rows match and the call lands on the batched kernel path — the batch differs
only in its material crops. Practical notes:

- pick the padding so no window clips at the model edge; a clipped window
  shifts its local geometry and drops the batch back to the per-shot loop
  (still correct, just slower);
- the window's side boundaries are artificial truncations that carry the
  absorbing layer — size the padding accordingly, exactly as in the FD
  version;
- the window width `W` must satisfy `(W - 1) % n_order == 0`.

A complete Marmousi inversion in this style is
[notebook 44](../notebooks/44_sem_batched_local_window_fwi.ipynb) — the SEM
twin of the FD
[notebook 23](../notebooks/23_batched_local_window_fwi.ipynb), same recipe on
both solvers.

## FD-aligned receiver components

Field data arrives as pressure and particle velocity, so PropSEM converts
internally -- ``receiver_type=['p', 'vx', 'vz']`` works exactly like the FD
side, on every equation and both backends:

```python
solver = PropSEM(AcousticElasticSEM2D, ..., receiver_type=["vx", "vz"])  # OBS
hydro  = PropSEM(AcousticElasticSEM2D, ..., receiver_type=["p"])   # hydrophone
```

Per family: ``p = -d2(chi)/dt2`` on the potential-form acoustics (SPECFEM
normalization), ``vx/vy/vz = d(u)/dt`` on the displacement elastics, and pure
name aliases where no transform is needed (the scalar-wave acoustics' field is
already the pressure-like wavefield; the velocity-stress family already
evolves velocities). The conversion is a matched-stencil time difference of
the recorded traces, applied on the way out -- shot batching, boundary saving
and gradients all flow through it unchanged. Derived names are receivers
only: as a ``source_type``, ``p`` is refused, and on the elastics ``vx`` /
``vz`` name a force (see [Source types](#source-types)).

On the potential forms (``AcousticPotentialSEM2D`` and the water of
``AcousticElasticSEM2D``) the wavelet is the **pressure** source, as on the
pressure form and ``PropTorch``: the records are those of chi driven by minus
the wavelet's double time integral, so the same wavelet gives the same ``p``
waveform on ``AcousticSEM2DDeformed`` and on the potential form, and in the
fluid-solid class ``p`` agrees with ``vx``/``vz``. The amplitude still
differs by ``kappa = rho*c^2`` at the source (the potential form's normalization).
PropSEM gets these records from the run the wavelet itself drives -- the
equations are linear and time-invariant, so that run records ``p`` sample for
sample and ``chi``/``u``/``v`` are its running integrals -- which keeps float32
from differencing a double integral; only ``return_wavefield=True`` runs the
integrated wavelet, so that the snapshots are chi and u. A wavelet with a
non-zero mean (or first moment) leaves chi and u a static (or ramping) offset
that ``p`` and ``v`` do not see. ``SEMPropagator`` and ``SEMModelParallel``
below PropSEM still take the wavelet as the chi source. Without a
``receiver_type`` PropSEM records ``p`` on the potential form and ``p``,
``vx``, ``vz`` on the fluid-solid class (PropTorch's defaults); ``chi`` is a
near-DC tail, a poor record to fit -- ask for it by name.
### Strain, rotation and pressure

A receiver can also record a spatial derivative of the fields, the seismograms
SPECFEM2D's `seismotype` 4 and 5 give and that DAS and rotational sensors
record. It takes the gradient of its element's basis at its point
(`sweep.sem.receivers`), so it runs on both backends, in shot batches and
under every memory mode, and the gradient reaches the model through it:

| `receiver_type` | on | records |
| --- | --- | --- |
| `exx`, `ezz`, `exz` (3-D also `eyy`, `exy`, `eyz`) | the elastic classes | the strain `(d_i u_j + d_j u_i) / 2` |
| `rot_y` (3-D `rot_x`, `rot_y`, `rot_z`) | the elastic classes | the rotation `curl(u) / 2`, in the solver's axes (z down); in 2-D SPECFEM2D's curl seismogram |
| the same with `_t` | the elastic classes | their rates (DAS records the strain rate `exx_t` along a horizontal fibre) |
| `p` | the isotropic elastic classes | the pressure `-kappa div u`, `kappa = lambda + 2 mu / 3` at the receiver, as SPECFEM takes it (plane strain in 2-D) |
| `ux`, `uz`; `vx`, `vz`; `ax`, `az` | `AcousticPotentialSEM2D` | the particle displacement `grad(chi) / rho`, its velocity and acceleration |

On the velocity-stress classes, whose fields are the velocity, the strain is
the running sum of its rate times `dt` -- the displacement their leapfrog
integrates. On `AcousticElasticSEM2D` every receiver takes the medium its
element holds: `vx`, `vz` (and `ux`, `ax`, ...) are the fluid's particle
motion in the water and the displacement's in the rock, `p` is `-d2(chi)/dt2`
in the water and `-kappa div u` in the rock, and the strain and rotation are
zero in the water. The anisotropic classes have no `p`.

The field's gradient jumps across element edges, and receivers on a regular
line sit on them every few channels: a receiver on an edge takes every element
that meets there by its share, as a moment source is spread over them
(`EdgeStamp`), in its own medium on a fluid-solid form.

Against SPECFEM2D, which records the same quantities, on the water over the
rock (`test/sem/specfem_compare/run_compare_box.py`): inside an element the
particle velocity at the hydrophones and the pressure at the geophones agree
to round-off (4e-12..5e-12 on the curved sea floor). SPECFEM2D takes one
element at an edge, so a receiver there differs by the gradient's jump: 1e-4
to 2e-4 (median) on the flat sea floor, whose receivers all sit on edges.

## Source types

On the elastic equations `source_type` takes `PropTorch`'s names as well as
the equation's own source fields, and means the same source:

| `source_type` | source | the wavelet is |
| --- | --- | --- |
| `["vz"]`, `["vx"]` (or the field's own name, `v_z` / `u_z`) | a body force along that axis | the force |
| `["vx", "vz"]` | a force at 45°, not an explosion | the force |
| `["sxx", "szz"]` | an explosion: moment tensor `Mxx = Mzz = 1` | the moment rate |
| `["sxz"]` | a shear moment, `Mxz = Mzx = 1` | the moment rate |
| 3-D: `["sxx", "syy", "szz"]`, `["sxy"]`, ... | the explosion, the shear moments (`syy`, `sxy`, `syz` too) | the moment rate |

```python
solver = PropSEM(ElasticVSSEM2DDeformed, ..., source_type=["sxx", "szz"],
                 receiver_type=["vz"])
```

FD injects a source by adding the wavelet to the named fields every step, so
on a stress field the wavelet is the moment *rate*. A SEM elastic equation has
no stress source field -- its loads act on `u` / `v` -- so `PropSEM` stamps
the moment tensor as a weak-form load and drives it with the wavelet's
running integral. With that, the same `source_type` and wavelet give FD's
waveform (`test/sem/test_sem_source_type_fd.py` runs both and checks it);
without it the SEM record would be one time derivative off. The amplitude
scales still differ between the backends.

The default stays a unit vertical force (FD's elastic default is the
explosion). A force and a stress cannot be mixed in one call (they take the
wavelet differently), and a per-call `source_vector` only applies to a
force. Both backends carry every source
type: the kernels only ever see the assembled per-field patterns.

A stress source on an element edge -- every `n_order`-th grid index lands on
one -- is stamped in every element that meets there, each taking the share of
a small disk it holds (half a side on an edge, the corner angles at a
corner): the moment tensor differentiates the basis, whose gradient jumps
across the edge, and stamped in one element the explosion came out 1-2%
lopsided. In 3-D the disk is a ball: a face gives each side half, an edge or a
corner of an axis-aligned mesh a quarter or an eighth, and a curved mesh the
solid angles there. A force needs none of this; its stamp has no derivative.
`test/sem/test_sem_moment3d.py` checks the 3-D stamp against differences of
the point load, the shares, the mirror symmetry of an explosion on a corner and
Hex27 against Hex8.

## Plane-wave incidence

`sources` may be a `PlaneWave`: a plane P, SV or SH wave from below, or a Rayleigh wave along
the surface, SPECFEM2D's `initialfield`. P, SV and Rayleigh waves run on
`ElasticSEM2DDeformed` and `ViscoElasticSEM2DDeformed`, SH waves on `ElasticSHSEM2DDeformed`,
on `impl='eager'`:

```python
from sweep.sem import ElasticSEM2DDeformed, PlaneWave, PropSEM

prop = PropSEM(ElasticSEM2DDeformed, shape=(201, 401), dh=10.0, dt=1e-3, free_surface=True,
               impl="eager", receiver_type=["u_x", "u_z"])
wave = PlaneWave("SV", 25.0)                # 25 degrees from the vertical, towards +x
seis = prop(wavelet, wave, receivers, models=[vp, vs, rho])
free = prop.free_field(wavelet, wave, receivers[0], [vp, vs, rho])   # the plane wave alone
```

| `PlaneWave` | the wave | the wavelet is | SPECFEM2D `source_type` |
| --- | --- | --- | --- |
| `("P", a)` | a plane P wave `a` degrees from the vertical, towards +x for `a > 0` | the incident displacement along the propagation direction | 1 |
| `("SV", a)` | a plane SV wave; past the critical angle its reflected P wave is evanescent | the incident displacement along `(cos a, sin a)` (x, z down), its mirror image for `a < 0` | 2 |
| `("SH", a)` | a plane SH wave, reflected whole by the free surface | the incident `u_y` | 2 with `P_SV = .false.` |
| `("Rayleigh", 90)`, `("Rayleigh", -90)` | a Rayleigh wave along the surface, towards +x or -x | the horizontal displacement on the surface | 3 |
| any, under an absorbing top | the whole-space wave, without the surface's reflections | as above | 4, 5 |

The wave enters through a band of elements one element thick around a box: the domain
reduction method (Bielak et al., 2003). Inside the box the run carries the whole field, the
plane wave of the background and what the medium in the box scatters. From the band out it
carries only the scattered field, which the C-PML absorbs. With `u0` the plane wave, `b` the
box's boundary nodes and `e` the band's other nodes, the band's elements are loaded with
`P_b = -K_be u0_e` and `P_e = K_eb u0_b`, `K` their stiffness; the mass is diagonal, so no
mass term enters. SPECFEM2D starts its run with the plane wave in the mesh and feeds it
through its Stacey edges instead (Bielak's conditions, which its code ties to Stacey).

Inside the box the medium is free: a topography, layers, a basin, attenuation. Outside it --
the band, any margin and the absorbing layer -- it must be the uniform elastic background
under a flat free surface, the plane wave's half-space; anything else is refused, as is a band
that reaches into the absorbing layer.

- **The plane wave.** With a free top it carries the half-space's reflected and converted
  waves; under an absorbing top, the incident wave alone. It is computed in the frequency
  domain from the wavelet's samples, so a delay is the band-limited one. One traction-free
  solve gives the reflections at any angle: past the critical angle the reflected P wave is
  evanescent, and a Rayleigh wave is two evanescent waves, each decaying with depth at a rate
  set by the frequency. Below the critical angle it is SPECFEM2D's formulas to 2e-14
  (`test/sem/test_sem_plane_wave.py`); past it, and as a Rayleigh wave, it solves the wave
  equation with a free surface, and its Rayleigh speed is SPECFEM2D's to 1e-15.
- **`origin`** (model grid units, x first) is where the wavelet is the incident displacement.
  The default is the band's node the wave reaches first, so a wavelet that starts at rest
  starts the run at rest. A run must start at rest: a wave already on the band at the first
  step (more than 1e-3 of its peak) is refused. A Rayleigh wave's vertical motion, like an
  evanescent wave's, is the wavelet's Hilbert transform, whose tails run ahead of the pulse;
  a Ricker needs about 4.5 periods of delay there instead of 1.5.
- **`box`** is `((x0, z0), (x1, z1))` in model grid units, on element edges. The default is one
  element in from the model's edges, from the top down when the top is free. With a
  topography the element rows follow the surface, and `z0`, `z1` count them (`n_order` grid
  rows each).
- **The records.** Inside the box they carry the whole field, in the band a mix, outside it the
  scattered field alone. `free_field(wavelet, wave, points, models)` gives the plane wave at any
  points on the records' times: in a flat half-space the records inside the box are that, to
  the run's own error.
- **Gradients.** The loads are a fixed source computed from the background, with no gradient
  to the model through them; the model inside the box takes its gradient as with any source.
  The loads are linear in the wavelet and carry its gradient.
- **Cost.** The loads are a time series per band node and component, `2 n_band nt` values
  (half that in SH): 160 MB in fp64 for 5000 band nodes over 2000 steps. Every eager memory mode runs them (`Full()`,
  `Ckpt()`, `BoundarySaving()`). `impl='c'` takes one wavelet per shot and refuses them.

The run's error inside the box is the leapfrog's error in time. In a flat half-space (an SV
wave at 20 degrees, 10 Hz, 20 m elements: 8 points per wavelength at 25 Hz) it is 1.4e-3 of the
peak with a 0.4 ms step and 4.1e-4 with 0.2 ms, the leak outside the band falling from 1.6e-4
to 3.9e-5; halving the elements then moves it to 3.5e-4. Against SPECFEM2D:

`test/sem/specfem_compare/run_compare_planewave.py` runs the cases of
`make_specfem_planewave_cases.py`: the rock vp 2, vs 1, rho 1 (no dimensions, as SPECFEM2D's
own plane-wave examples), elements of 0.1, a Ricker of f0 = 1, dt = 2^-9, SPECFEM2D 8.1.0 in
double precision with its Stacey edges and Bielak's conditions on the sides and bottom. Ours
starts earlier, at rest, and the records are aligned on the plane wave's time. Relative L2 over
all receivers and components:

| Case (domain) | ours vs the plane wave | SPECFEM2D vs the plane wave | ours vs SPECFEM2D |
| --- | --- | --- | --- |
| P at 30° (8 x 4) | 2.0e-4 | 2.9e-3 | 3.0e-3 |
| SV at 20° (8 x 4) | 3.7e-4 | 2.3e-3 | 2.3e-3 |
| SV at 40°, past the critical angle (8 x 4) | 4.2e-4 | 2.0e-3 | 2.0e-3 |
| Rayleigh (8 x 4) | 4.0e-4 | 2.5e-3 | 2.7e-3 |
| SH at 30° (8 x 4) | 3.9e-4 | 1.0e-1 | 1.0e-1 |
| P at 20° in a whole space (8 x 4, Stacey on all four sides) | 1.6e-4 | 4.7e-3 | 4.8e-3 |
| P at 30° (24 x 12) | 2.4e-4 | 2.5e-4 | 1.3e-4 |
| SV at 20° (24 x 12) | 4.6e-4 | 2.4e-4 | 2.4e-4 |
| SH at 30° (24 x 12) | 4.8e-4 | 2.7e-4 | 2.3e-4 |
| SV at 20° under a soft basin (24 x 12) | — | — | 2.5e-4 |

- SPECFEM2D starts with the exact plane wave in its mesh; ours is the one the mesh carried in
  from the band. Each run is off the plane wave by its own discretization error.
- In the small domain SPECFEM2D's Stacey edges, 1 to 3 from the receivers, add 2e-3 to 5e-3
  of their own. In the large one they are far from the receivers, and the two runs agree to
  1.3e-4 to 2.5e-4, under the basin (vs 0.5, 2 wide and 0.4 deep) as well.
- SPECFEM2D's amplitude conventions, which the script also fits: below the critical angle
  the incident displacement at its reference point is the Ricker `R(t - t0)` (fitted 0.999996
  to 1.0002). Past it and for the Rayleigh wave it is
  `-0.5 R(t - 1.2 / f0)`, from its frequency-domain field (`paco_beyond_critical`; fitted
  -0.50007 and -0.50028). In SH it is `cos(a) R`, the x component of its SV polarization
  (0.866027 at 30°).
- In SH, SPECFEM2D's Stacey edges take no Bielak terms: the SH branch of
  `compute_stacey_elastic.f90` damps `rho vs v_y` alone, so the incident wave is not fed in
  there and the records near the edges miss the plane wave by up to 20% of the peak.
- SPECFEM2D computes its frequency-domain field point by point, at 32768 frequencies in the
  large domain, which made those cases too slow to run (under 500 of 460000 points in ten
  minutes). The past-critical and Rayleigh cases are compared in the small domain only.

Notebook 58 runs an SV wave in the flat half-space, past the critical angle, a Rayleigh wave,
a hill and a canyon, and a soft basin.

## Anisotropic media

`ElasticVTISEM2DDeformed` is the velocity-stress elastic SEM in a VTI medium, one with a
vertical symmetry axis: its models are `vp`, `vs` (the vertical velocities), `rho` and
Thomsen's `epsilon` and `delta` (`sweep.sem.thomsen_to_vti_cij`). `ElasticTTISEM2DDeformed`
tilts the axis by a sixth model, `theta` (radians), with the closed-form Bond rotation of the
FD `ElasticTTI2nd`, in its axes (z down) and with its sign of `theta`
(`sweep.sem.thomsen_to_tti_cij`). The tilt adds the couplings `c15` and `c35` of the normal
stresses to the shear strain; `theta = 0` is the VTI class, and all six models are
differentiable.

```python
from sweep.sem import ElasticTTISEM2DDeformed, PropSEM

solver = PropSEM(ElasticTTISEM2DDeformed, shape=(nz, nx), dh=2.5, dt=2e-4,
                 free_surface=True, source_type=["vz"], receiver_type=["vx", "vz"])
seis = solver(wavelet, sources, receivers, models=[vp, vs, rho, epsilon, delta, theta])
```

The fastest wave is not `vp`: along the plane of isotropy the P velocity is
`vp sqrt(1 + 2 epsilon)`, and the time step has to allow for it.

Both classes agree with SPECFEM2D's anisotropic elastic medium to 1e-13 (see
[Against SPECFEM2D](#against-specfem2d)). That medium is SPECFEM2D's material type 2,
`c11 c13 c15 c33 c35 c55`; in its z-up axes `c15` and `c35` change sign. The cases are a VTI
shale (`epsilon = 0.25`, `delta = 0.10`) with a vertical and a 30° axis, each with a point
force and with an explosion. They need the box with free sides, because SPECFEM2D's C-PML
and Stacey conditions use isotropic moduli in anisotropic elements. Against SPECFEM2D in its
default single precision the records agree to 3-5e-5, the round-off of a 6000-step fp32 run.

### Anisotropy in 3-D

`ElasticVTISEM3D` and `ElasticTTISEM3D` (Hex8), with `ElasticVTISEM3DDeformed` and
`ElasticTTISEM3DDeformed` on the Hex27 mesh of a topography, are the 3-D velocity-stress SEM
in a transversely isotropic medium. The VTI models are `vp`, `vs` (the vertical velocities),
`rho` and Thomsen's `epsilon`, `delta` and `gamma`; the TTI classes add the tilt `theta` of
the symmetry axis from the vertical and its azimuth `phi` from x toward y (radians). The
parameters and the Bond rotation are those of the FD `ElasticTTISG3D`
(`sweep.sem.constitutive.thomsen_to_cij3d`), so at `phi = 0` the tilt is that of
`ElasticTTISEM2DDeformed`. As in 2-D only the stress update changes: the stiffnesses (21 for
the TTI classes, 9 for VTI) act on the six strain rates, and the time loop, the C-PML, the
free surface, the sources and receivers and the gradients are those of `ElasticVSSEM3D`.

```python
from sweep.sem import ElasticTTISEM3DDeformed, PropSEM

solver = PropSEM(ElasticTTISEM3DDeformed, shape=(nz, ny, nx), dh=10.0, dt=5e-4,
                 topography=topo, free_surface=True, source_type=["vz"],
                 receiver_type=["vx", "vy", "vz"])
seis = solver(wavelet, sources, receivers,
              models=[vp, vs, rho, epsilon, delta, gamma, theta, phi])
```

Where `|theta|` is below 1e-7 the stiffnesses are exactly the VTI ones, and a TTI medium
without tilt runs the VTI step bit for bit. The gradient in `theta` there is still the
rotation's, so an inversion that starts from a vertical axis can update the tilt; the FD
class switches to the VTI entries there, which removes that gradient. The time step has to
allow for the fastest wave, `vp sqrt(1 + 2 epsilon)` along the plane of isotropy for a
positive `epsilon`.

`test/sem/test_sem_tti3d.py` checks the isotropic limit on both meshes and the TTI class
without tilt against the VTI class, bit for bit. It also checks an axis tilted by 90 degrees
toward x and toward y against the VTI problem turned by the same rotation, which the cubic
mesh maps onto itself; the records agree to round-off, which checks all 21 stiffnesses in the
step. Further checks: the stiffnesses against `ElasticTTISG3D`'s, the gradient in the tilt at
zero tilt against central differences, boundary saving against the full tape in a box
without absorbing layers, a flat Hex27 mesh against Hex8, and a tilted medium in the C-PML
over 2 s. The C-PML stretches the coordinates as for an isotropic medium; in a strongly
anisotropic medium such a layer can become unstable, as any perfectly matched layer can, so
long runs in such media need checking. Against SPECFEM3D's anisotropic medium, given the same
21 stiffnesses, the records agree to 5e-14 for a vertical and for a tilted axis (see
[Against SPECFEM3D](#against-specfem3d)). Notebook 60 shows the wavefronts and shear-wave
splitting.

All four classes run on `impl='c'`: the CUDA kernels of `ElasticVSSEM3D` (and of
`ElasticVSSEM3DDeformed` on Hex27) with the nine or 21 stiffnesses in the stress update, and
the stiffness gradients go on to the Thomsen parameters, the tilt and the azimuth through the
same `thomsen_to_cij3d` as on eager. The records and the gradients in every model agree with
eager to 2e-7 to 2e-6 in relative L2, with the C-PML, on Hex8 and on a curved Hex27
(`test/sem/test_sem_c_tti3d.py`). On an A100 (16³ elements, order 4, 400 steps, forward and
backward) the VTI classes took 1.1× and the TTI classes 1.4× the time of the isotropic
kernels on the same mesh.

## Visco-elastic attenuation

`ViscoElasticVSSEM2DDeformed` adds attenuation to `ElasticVSSEM2DDeformed` with the
rheology of the FD `sweep.equations.ViscoElastic`, which is also SPECFEM2D's: `n_sls`
standard linear solids whose relaxation times are fixed over the band `f_band`, with
strengths fitted for a constant `Qp` and `Qs` over that band. The default band, a factor 12
around `f_ref`, is that of SPECFEM2D's original fit; its current fit spans `f_ref / 10` to
`10 f_ref`. The models are `vp`, `vs`, `rho`, `Qp` and `Qs`, where `vp` and `vs` are the
phase velocities at `f_ref`:

```python
from sweep.sem import PropSEM, ViscoElasticVSSEM2DDeformed

solver = PropSEM(ViscoElasticVSSEM2DDeformed, shape=(nz, nx), dh=5.0, dt=5e-4,
                 free_surface=True, f_dominant=12.0, source_type=["sxx", "szz"],
                 receiver_type=["vx", "vz"], equation_options={"f_ref": 12.0, "n_sls": 3})
seis = solver(wavelet, sources, receivers, models=[vp, vs, rho, qp, qs])
```

The memory variables are element-local, as the stresses are, and they are updated as in the
FD class at every GLL node. The explicit stress update runs at the unrelaxed velocities,
which exceed `vp` and `vs` for a finite Q (by about 8% at Q = 10 over the default band), so
the time step should be taken from those (`unrelaxed_velocities()` of the built equation).
All five models are differentiable. Q must exceed the lower bound of the fit (about 1.8 for
three mechanisms over the default band), and element arrays of `Q = inf` give the elastic
step bit for bit. The C-PML terms run on every element, without the band split. Boundary
saving is refused because the attenuation is dissipative; use `Full()` or
`Ckpt(chunks=...)`.

Two conventions of SPECFEM2D's `ATTENUATION_VISCOELASTIC` differ from the defaults, and
both are options:

- **Velocities at `f_ref`.** By default `vp` and `vs` are the phase velocities at `f_ref`.
  With `equation_options={"velocities": "real_modulus"}`, `rho vp^2` and `rho vs^2` are
  instead the real parts of the moduli at `f_ref`, which is how SPECFEM2D's
  `READ_VELOCITIES_AT_f0` reads them. Its phase velocity at `f_ref` is then higher than
  `vp`, by about `3 / (8 Q^2)`, or 4e-4 at Q = 30.
- **The constant-Q fit.** By default the least squares samples the band at 200 frequencies.
  With `n_fit = 2 n_sls - 1` it samples SPECFEM2D's frequencies instead: `2 n_sls - 1` of
  them, log-spaced, ends included.

`test/sem/test_sem_visco_elastic.py` checks the elastic limit, the checkpointed and vmapped
rollouts, the gradients in `vp`, `Qp` and `Qs` against central differences, and the two
options against SPECFEM2D's formulas. Against SPECFEM2D itself (Q = 30, three mechanisms over
its band `f_ref / 10 .. 10 f_ref`; see [Against SPECFEM2D](#against-specfem2d)), the records
differ by 0.9% in relative L2 with the defaults. The difference is almost entirely a time
shift from the velocity reading: the visco/elastic spectral ratios of the two codes agree to
0.1% in amplitude, and differ in phase linearly with frequency. With both options the
records agree to 2e-4 at a time step of 0.2 ms. What is left is the time scheme of the memory
variables: SPECFEM2D convolves its strain memories recursively at the displacement's step,
while ours advance by the trapezoidal rule at the half step. Both are second order in `dt`,
and halving the step divides the difference by four (4.2e-5 at 0.1 ms).
`test/sem/specfem_compare/run_compare_visco.py`, an earlier comparison on a half-space with
Stacey sides, gives 1.2% for the visco-elastic records and 1.7% for the elastic ones; there the
absorbing conditions of the two codes differ as well.

### Visco-elastic in 3-D

`ViscoElasticVSSEM3D` (Hex8) and `ViscoElasticVSSEM3DDeformed` (Hex27, under a topography)
add the same rheology to `ElasticVSSEM3D` and `ElasticVSSEM3DDeformed`: six memory variables
per mechanism (`xx`, `yy`, `zz`, `xy`, `xz`, `yz`), the same models, options and refusals.
With the C-PML of the velocity-stress classes they are the 3-D classes that combine
attenuation, a topography and absorbing boundaries:

```python
from sweep.sem import PropSEM, ViscoElasticVSSEM3DDeformed

solver = PropSEM(ViscoElasticVSSEM3DDeformed, shape=(nz, ny, nx), dh=10.0, dt=5e-4,
                 topography=topo, free_surface=True, f_dominant=10.0,
                 source_type=["vz"], receiver_type=["vx", "vy", "vz"],
                 equation_options={"f_ref": 10.0})
seis = solver(wavelet, sources, receivers, models=[vp, vs, rho, qp, qs])
```

A vertical point force in a whole space (a C-PML on all six sides, Q = 25 and 15, 5 m GLL
spacing, a 12 Hz Ricker) matches the visco-elastic Green's function -- the elastic one, near
field included, with the GSLS complex moduli of the built equation in the frequency domain --
to 0.7e-3 to 1.2e-3 in relative L2 at 60 to 113 m from the source, where the attenuation
changes the records by 12% to 47%. On the same grid the elastic class is 1.2e-3 to 1.7e-3
from its own Green's function. `test/sem/test_sem_visco_elastic3d.py` runs a coarser version
of that check, the elastic limit on both meshes, the checkpointed and vmapped rollouts and the
gradients in `vp`, `Qp` and `Qs`. Against SPECFEM3D the records agree to 3.6e-3
on its own mechanisms, which is the time discretization of the memory variables (see
[Against SPECFEM3D](#against-specfem3d)).

Both classes run on `impl='c'`: the CUDA kernels of the elastic classes with the unrelaxed
moduli, the memories advancing in the stress kernel and their adjoint in its transpose (the
adjoint needs no history of the memories, only the saved strain rates). The records and the
gradients in all five models agree with eager to 2e-7 to 4e-6 in relative L2, with the C-PML,
on Hex8 and on a curved Hex27, and `Q = inf` gives the elastic kernels' records
(`test/sem/test_sem_c_visco3d.py`). With three mechanisms a forward and backward call took
1.55–1.6× the elastic time on an A100 (16³ elements, 400 steps). Boundary saving stays
refused on both backends.

### SPECFEM2D's displacement scheme

`ViscoElasticSEM2DDeformed` is the displacement class `ElasticSEM2DDeformed` with SPECFEM2D's
`ATTENUATION_VISCOELASTIC` as its Newmark path runs it, and it agrees with SPECFEM2D to
round-off. Its mechanisms relax the bulk modulus `kappa = lambda + mu` and the shear modulus,
as SPECFEM2D's do, so the models are `vp`, `vs`, `rho`, `Qk` (SPECFEM2D's `Qkappa`) and `Qmu`:

```python
from sweep.sem import PropSEM, ViscoElasticSEM2DDeformed

solver = PropSEM(ViscoElasticSEM2DDeformed, shape=(nz, nx), dh=5.0, dt=5e-4,
                 free_surface=True, f_dominant=15.0, receiver_type=["vx", "vz"],
                 equation_options={"f_ref": 15.0, "n_sls": 3})
seis = solver(wavelet, sources, receivers, models=[vp, vs, rho, qk, qmu])
```

Each mechanism keeps three memories at every element GLL node. At every step each one takes
`e = A^2 e + B (s + A s_old)`, with `A = exp(-dt / (2 tau_sigma))`, from a strain `s` and its
value at the last step: the dilatation `theta` for `e1`, `dux/dx - theta / 2` for `e11` and
the shear strain for `e13`. The stress gains `(lambda + mu)_U sum(e1) + 2 mu_U sum(e11)` (xx),
`(lambda + mu)_U sum(e1) - 2 mu_U sum(e11)` (zz) and `mu_U sum(e13)` (xz).

The defaults are SPECFEM2D's:

- the band `f_ref / 10 .. 10 f_ref`;
- the fit at `2 n_sls - 1` frequencies;
- `vp` and `vs` read as the unrelaxed velocities (`velocities="unrelaxed"`), which is how
  SPECFEM2D reads them without `READ_VELOCITIES_AT_f0`.

`"real_modulus"` is that flag: `rho vs^2` and `rho (vp^2 - vs^2)` are then the real parts of
the shear and the bulk modulus at `f_ref`. The phase reading of the velocity-stress class is not
offered. Read as unrelaxed, `vp` and `vs` are the speeds at infinite frequency, so a finite Q
makes the waves of the band slower than those of the elastic class.

The absorbing layer attenuates too, as SPECFEM2D's visco-elastic C-PML does. There the
stress is the C-PML's non-symmetric stress with the complex moduli,
`sigma_xx = (kappa + mu)(w) (s_z / s_x) dux/dx + (kappa - mu)(w) duz/dz` and so on, and the
memories take the stretched strains: those of `sigma_xx` the strains with `dux/dx` stretched,
those of `sigma_zz` with `duz/dz` stretched, and the two shear stresses one set each. Each
memory therefore has a second copy on the band elements. SPECFEM2D expands the product of the
stretching and the relaxation in partial fractions, one recursion per pole; here the
relaxation runs on the stretched strain. That is the same operator to second order in `dt`,
and it has no two poles to meet: SPECFEM2D's coefficients grow without bound where a
relaxation frequency equals a C-PML `alpha` or `beta`, and both vary across the layer. The
memories load the layer's outer edge as well, so the step pins that edge again after adding
them; unpinned, the outer-edge recursion integrates without damping and the run grows from
round-off after a second or two.

Without attenuation in the layer the medium's modulus would jump at its edge, by the
dispersion over the band (a few percent at Q = 30), and the layer would reflect the jump. On a
200 m x 400 m box with a free top, a 100 m layer, `Qkappa = 30` and `Qmu = 20`, the
reflection is 5.1e-2 of the record with the layer elastic and 4.7e-4 with it attenuating; the
elastic medium's C-PML gives 1.0e-3. Run for 8 s, the energy falls steadily to 4e-19 of its
peak.

Against SPECFEM2D's visco-elastic C-PML, on the absorbing-layer case of the elastic comparison
with `Qkappa = 30` and `Qmu = 20` (a 2000 m x 1000 m interior, a 10-element layer, 6000 steps
in double precision; `test/sem/specfem_compare/run_compare_absorb_visco.py`), the records agree
to 2.5e-7 of the reference's norm, and the two layers reflect alike: 9.3e-5 of the record over
every receiver, 2.4e-3 at most on a line 50 m above the layer. With the layer elastic the
reflection was 5.3e-3 over every receiver and up to 0.11 on a line inside the left layer. Our
run of the larger box agrees with SPECFEM2D's to 9e-14.

All five models are differentiable, and element arrays of `Q = inf` give
`ElasticSEM2DDeformed` bit for bit. Boundary saving is refused because the attenuation is
dissipative, and the class runs on eager only.
`test/sem/test_sem_visco_elastic_displacement.py` checks the elastic limit, the decay with
distance, that `Qmu` takes the shear wave down while `Qk` leaves it alone, the attenuating
layer (its reflection, the pinned outer edge, the band split), the gradients and the two
readings.

### The nonlinear fit

`equation_options={"fit": "nonlinear"}` is SPECFEM2D's `USE_SOLVOPT`: the relaxation
frequencies are fitted with the strengths, for each Q. Both fits minimise the residual
`Q Im M(w) - Re M(w)` at frequencies log-spaced over the band; the linear one keeps the
relaxation times log-spaced over the band and solves for the strengths, the nonlinear one
moves the relaxation times too and keeps both positive
(`sweep.equations.gsls_nonlinear.GSLSNonlinearFit`). The largest error of `Q(f)` over
SPECFEM2D's band:

| Mechanisms, Q | linear, `2 n_sls - 1` frequencies | linear, 200 | nonlinear, `4 n_sls` |
| --- | --- | --- | --- |
| 3, Q = 30 | 5.1% | 4.9% | 3.3% |
| 3, Q = 10 | 5.8% | 5.1% | 3.5% |
| 3, Q = 3 | 8.5% | 6.7% | 5.0% |
| 4, Q = 30 | 3.4% | 5.2% | 0.7% |
| 4, Q = 10 | 4.0% | 6.1% | 0.8% |
| 4, Q = 3 | 7.0% | 10.1% | 1.0% |

The dense linear fit minimises the mean square, not the largest error. Below a Q of 1.5 (three
mechanisms) or 1.8 (four) the linear fit needs a negative strength and is refused; the
nonlinear fit keeps every strength positive down to `Q = 1`. Each node then has its own
relaxation times, the bulk ones from `Qkappa` and the shear ones from `Qmu`. SPECFEM2D runs
SolvOpt from the linear fit at every GLL point and stops at about four significant digits;
here Levenberg-Marquardt runs to full precision at the Chebyshev nodes of `1 / Q` in
`[0, 1]`, once per number of mechanisms, band ratio and number of frequencies (a fraction of a
second), and a model's Q maps to its mechanisms through the Chebyshev series. Q stays a
differentiable model, and `Q = inf` gives zero strengths. With one mechanism the optimum jumps
between two branches as Q falls, so `fit="nonlinear"` needs `n_sls >= 2`. Against
SPECFEM2D's own SolvOpt (`test/sem/specfem_compare/specfem_solvopt_driver.f90` links its
`attenuation_model`), the mechanisms agree to 2.5e-4 for two to four mechanisms and Q from 2
to 10^4, our residual a little lower; `test/sem/test_sem_gsls_nonlinear.py` checks that, the
flatter `Q(f)`, the table against direct fits and the derivative in Q. Against SPECFEM2D run
with `USE_SOLVOPT` (`Qkappa = 30`, `Qmu = 10`, a force in the free box of
[Against SPECFEM2D](#against-specfem2d)) the records differ by 1.8e-5, which is that
difference of the mechanisms; with SPECFEM2D's own mechanisms fed in they agree to 6e-14.
`ViscoAcousticPotentialSEM2D` takes the same option; the velocity-stress classes keep one set
of relaxation times for P and S and take no `fit`.

## SH waves

`ElasticSHSEM2DDeformed` solves for the out-of-plane displacement `u_y` of an elastic medium
that does not vary along y, SPECFEM2D's `P_SV = .false.`:
`rho u_y'' = d/dx (mu du_y/dx) + d/dz (mu du_y/dz) + f_y`. This is the scalar wave equation of
the potential form, with its mass weight `1 / kappa` read as `rho` and its stiffness weight
`1 / rho` as `mu`, so the class is `AcousticPotentialSEM2D` built with `vp = vs` and the density
`1 / (rho vs^2)`. It shares that class's operators, its C-PML and its boundary saving, which is
exact. Its surface is traction-free: a side without an absorbing layer is free, and there are
no Dirichlet nodes. The models are `vs` and `rho`. The source is a force along y, and the
receivers record `u_y` or its velocity `vy`:

```python
from sweep.sem import ElasticSHSEM2DDeformed, PropSEM

solver = PropSEM(ElasticSHSEM2DDeformed, shape=(nz, nx), dh=5.0, dt=5e-4, free_surface=True,
                 receiver_type=["vy"])
seis = solver(wavelet, sources, receivers, models=[vs, rho])
```

It runs on eager, flat or under a `topography`, and takes `interfaces`.

## Visco-acoustic attenuation

`ViscoAcousticPotentialSEM2D` is `AcousticPotentialSEM2D` with SPECFEM2D's
`ATTENUATION_VISCOACOUSTIC`: its weak form and its Newmark update of the memory variables.
`n_sls` standard linear solids relax the bulk modulus, with relaxation times fixed over
`f_band` and strengths fitted for a constant `Qkappa`. The memories are node fields, as
SPECFEM2D's are. At every step, with `d = M_geo^-1 (-K chi)` (the stiffness over the
geometric mass), each memory takes `e = A^2 e + B (d + A d_prev)`, with
`A = exp(-dt / (2 tau_sigma))`, and the potential's load gains `M_geo sum(e)`. The models are
`vp` and `Qk`, and the density is `density=`, as for the potential form:

```python
from sweep.sem import PropSEM, ViscoAcousticPotentialSEM2D

solver = PropSEM(ViscoAcousticPotentialSEM2D, shape=(nz, nx), dh=5.0, dt=5e-4,
                 free_surface=True, density=1020.0, receiver_type=["p"],
                 equation_options={"f_ref": 15.0, "n_sls": 3})
seis = solver(wavelet, sources, receivers, models=[vp, qk])
```

`equation_options` take the visco-elastic class's `f_ref`, `f_band`, `n_sls`, `n_fit`,
`velocities` and `fit` (`"nonlinear"`, SPECFEM2D's `USE_SOLVOPT`, gives every node its own
relaxation times, read from the same element copy as its strengths; see
[The nonlinear fit](#the-nonlinear-fit)). The default reading of `vp` is the phase velocity
at `f_ref`. `"unrelaxed"`
reads it as the infinite-frequency velocity, which is how SPECFEM2D reads its input without
`READ_VELOCITIES_AT_f0`; with that flag SPECFEM2D does not rebuild its acoustic mass matrix.
All models are differentiable, and element arrays of `Qk = inf` give the potential step bit for
bit. Boundary saving is refused because the attenuation is dissipative, and the class runs on
eager only.

Against SPECFEM2D, with its mechanisms, its fit (`n_fit = 2 n_sls - 1`) and its reading of the
velocity, the records differ by 2.8% (Q = 30, 0.2 ms steps, see
[Against SPECFEM2D](#against-specfem2d)). The difference is in SPECFEM2D 8.1.0:
`update_memory_var_acous_weak_form` sets `dot_e1_old = dot_e1` inside its loop over the
mechanisms, so every mechanism after the first convolves the current `d` in place of the last
step's. That makes those mechanisms first order in `dt` instead of second. With that one line
reproduced (`SpecfemMemoryLine` in the comparison script) the records agree to 8e-13.

## Poroelastic waves (Biot)

`BiotVSSEM2DDeformed` solves the equations of the FD `sweep.equations.Biot` —
solid velocity, Darcy flux, total stress and pore pressure, the low-frequency
Biot theory — on the spectral-element mesh, with the same eight coefficients
`[H, mu, C, M, rho, rho_f, m, b]` (build them with
`sweep.equations.biot.biot_from_rock`) and the same source and receiver names:

```python
from sweep.equations.biot import biot_from_rock
from sweep.sem import BiotVSSEM2DDeformed, PropSEM

models = biot_from_rock(k_s=36e9, rho_s=2650.0, k_f=2.25e9, rho_f=1000.0, eta=1e-3,
                        k_d=k_d, mu=mu, phi=phi, kappa=1e-13, tortuosity=2.0)
solver = PropSEM(BiotVSSEM2DDeformed, shape=(nz, nx), dh=2.0, dt=1e-4,
                 free_surface=True, f_dominant=30.0,
                 source_type=["sxx", "szz"], receiver_type=["vx", "vz", "p"])
seis = solver(wavelet, sources, receivers, models=models)   # (nshots, nt, nrec, 3)
```

| `source_type` | source | the wavelet is |
| --- | --- | --- |
| `["sxx", "szz"]` (default), `["sxz"]` | total stress, as on the elastics | the moment rate |
| `["fx"]`, `["fz"]` | a point force on the solid and the fluid alike (SPECFEM2D's poroelastic force) | the force |
| `["p"]` | pore pressure; the total stress does not move | its rate |
| `["vx"]`, `["vz"]` (`v_x`, `v_z`) | solid velocity; the flux does not move | the velocity source |
| `["qx"]`, `["qz"]` (`q_x`, `q_z`) | Darcy flux; the solid does not move | the flux source |

As on `PropTorch`, a stress source is a moment tensor that `PropSEM` drives
with the wavelet's integral, while the others take the wavelet itself; the two
kinds do not mix in one call (run two and add the records). The force loads
the total momentum and the relative flow by the same amount, which moves the
solid and the fluid in proportion; the velocity and flux sources are injected
into their fields after the update, so they do not mix with a force either.
Receivers take `vx`, `vz`, `p`, `qx`, `qz`, `sxx`, `szz` and `sxz` (default
`["vx", "vz"]`); the flux and the stresses are read from the receiver's
element.

The solid is discretised as the velocity-stress elastic equation is:
velocity on the continuous nodes, total stress in the elements, momentum in
weak form with a lumped mass. The pore fluid is the pressure form of the
acoustic SEM: `p` lives on the continuous nodes with its equation in weak
form, and the Darcy flux in the elements, driven by the gradient of `p`. On
its own (`rho_f = C = 0`) the fluid is exactly the acoustic pressure SEM.
The other placement -- the flux continuous and `p` in the elements, as in
SPECFEM2D's poroelastic solver -- leaves the relative flow, which has no
shear stiffness, on a continuous field. A point source then excites
grid-scale modes along the grid lines through it. On those lines the pore
pressure came out 4 to 14 times the analytic one and grew like `1 / dh`
under refinement, while the solid velocity was off by a few percent.

Where an element node is shared, the solid momentum (one value) and the
flux copies of the elements there are coupled through the fluid's inertia.
Per node they leave `A dq/dt + W_b q = r` over the copies. This is
integrated exactly over the step (one small matrix exponential per node,
computed once per run), so any permeability is stable. The time step is
that of the inviscid fast P wave
(`sweep.equations.biot.biot_fast_p_velocity`), which is also the speed the
C-PML is designed for. A `free_surface` side is open-pore: traction-free
with `p = 0` on its nodes. A `topography` gives a curved one. Without an
absorbing layer the boundary is traction-free and closed to flow. When the
surface's drainage layer `sqrt(2 D / omega)` is thinner than the mesh (the
Darcy regime), its first nodes are fully drained. This is about 1% on
surface records on 8 m elements, which is the FD `Biot`'s figure on 2 m
cells. The undrained limit (`b` → ∞) agrees with `ElasticVSSEM2DDeformed`
(`vp = sqrt(H / rho)`, `vs = sqrt(mu / rho)`) to the discretisation: the
volumetric `alpha^2 M` part of the modulus there passes through the
continuous `p`.

### Anisotropic permeability and viscous attenuation

`BiotAnisoPermSEM2DDeformed` takes the drag as a symmetric tensor
`eta kappa^-1`: the models end in `b_xx, b_xz, b_zz` instead of `b`, which
`sweep.equations.biot_params.biot_drag_tensor(eta, k_xx, k_xz, k_zz)` gives
from a permeability tensor (x right, z down -- flip the sign of `k_xz` from a
z-up convention such as SPECFEM2D's). The flux copies at a node then advance
with both components coupled, still exactly.

`equation_options={"viscous_q": (Q0, f0)}` turns on SPECFEM2D's viscous
attenuation (`ATTENUATION_PORO_FLUID_PART`): the drag becomes a standard
linear solid, `b (1 + i w tau_e) / (1 + i w tau_s)` with
`tau_e, tau_s = (sqrt(Q0^2 + 1) ± 1) / (2 pi f0 Q0)` -- the plain drag at low
frequency, `tau_e / tau_s` times it at high frequency, `1 / Q0` at `f0`. It
costs a memory variable per flux component. `equation_options` passes any of
an equation's own `build` keywords; a name the equation does not take is
refused.

### Frame attenuation

`ViscoBiotVSSEM2DDeformed` attenuates the frame with the GSLS of
[`ViscoElasticVSSEM2DDeformed`](#visco-elastic-attenuation): the models end in `Qp` and
`Qs`, which act on the drained P modulus `H - C^2 / M` and on the shear modulus `mu`;
`f_ref`, `f_band` and `n_sls` come through `equation_options`, as for the visco-elastic
class, and `H` and `mu` are taken as the moduli at `f_ref`. The fluid's share of `H`
does not relax, and `viscous_q` attenuates the drag as before. The memories enter the frame
part of the total stress only. In a dry solid (`M = 0`) the class gives the records of
`ViscoElasticVSSEM2DDeformed` to round-off, and element arrays of `Q = inf` give those of
`BiotVSSEM2DDeformed` bit for bit (`test/sem/test_sem_visco_biot.py`, with the gradients
in `Qp` and `Qs` against central differences).

`test/sem/specfem_compare/run_compare_visco_biot.py` compares the frame attenuation with
SPECFEM2D's on its `poroelastic_acoustic` example with `ATTENUATION_VISCOELASTIC`
(SPECFEM2D attenuates a porous frame only next to a fluid region). SPECFEM2D attenuates the
shear modulus only, so there `Qp` makes the drained P modulus relax as `mu` does.
SPECFEM2D 8.1.0 updates its poroelastic memory variables four times per step and drops the
previous step's term from their recursive convolution, so its frame Q is about 115 where
50 is requested; its fluid routine also has `mu` for `lambda` in the attenuated stress.
With those three fixed and SPECFEM2D's own mechanisms, the attenuation changes the records
by 7.1% in both codes, and its effect agrees to 0.5% in relative L2. Our fit over the same
band gives an effect 8% larger, because SPECFEM2D's poroelastic routine scales its
memories by the relaxed shear modulus where its elastic routine uses the unrelaxed one,
which raises its Q by about 10%.

### Fluid and dry elastic regions

An element without a solid frame (`rho = 0`) is a fluid, one without a pore
fluid (`M = 0`) a dry elastic solid, so a sea floor, a reservoir under a
cap rock or a porous layer on a basement is one model.
`sweep.equations.biot_params.biot_fluid(k_f, rho_f)` and
`biot_elastic(vp, vs, rho)` give their coefficients; the ones a medium does
not have are zero, which is checked, and get no gradient, so an inversion
keeps the media as they are. Each element must be wholly one medium,
so give these models on the GLL nodes (`(E, ngll, ngll)` arrays, e.g. from
the centre depth of each element in `solver.mesh.z_e` -- the mesh includes
the absorbing layer) with the interfaces on element edges. In a fluid, `q` is the fluid velocity (the `qx` / `qz`
receivers) and `p` the pressure; `v` and the stresses are zero there, as `p`
and `q` are in a dry solid.

The interfaces are those of SPECFEM2D. Between a solid (porous or dry) and a
fluid, the fluid's pressure acts on the solid and the solid's normal
velocity enters the fluid's continuity equation, by two boundary integrals
that are exact transposes of each other; with the continuous `p` this is
the open-pore condition at a porous one (pressure, traction and the fluid's
volume flux `(v + q)·n` continuous). Between a porous and a dry solid
nothing is added: the velocity and the traction are continuous through the
shared nodes and the weak form, and the pores are sealed, `q·n = 0`, which
is the natural condition of the pressure equation. (SPECFEM2D sets the whole
relative displacement to zero on those nodes.) A source loads only the fields
of the element it is in, so one that would load nothing there -- a stress or
solid-velocity source in a fluid, a pore pressure or flux in a dry solid --
is refused, and a force in a fluid moves the fluid alone; on an interface
each field takes its load from the side that has it. An
interface may cross the side or bottom absorbing bands but not a corner
region.

`sweep.equations.biot_params` also maps the coefficients to the parameter
families of Morency, Luo & Tromp (2009) that SPECFEM2D writes its kernels
in -- `"primary"`, `"density"` (density-normalised) and `"wavespeed"` -- and
turns a coefficient gradient into a family's kernels by the exact chain rule
(`biot_family_kernels`).

### On CUDA

`BiotVSSEM2DDeformed`, `BiotAnisoPermSEM2DDeformed` and `ViscoBiotVSSEM2DDeformed` (its GSLS
frame memories in the stress kernel, as for the visco-elastic classes) run on `impl='c'` (two
core entries,
`sem2d_biot_forward` / `_backward`): the same step as eager in one kernel per sub-step --
the injected pressure and its pin, the strong pressure gradient, the stress divergence, the
flux copies of every node advanced together with the drag's exact step (the eager class's
`drag_operators`, whose matrix exponentials autograd differentiates), the frame stress and the
weak flux divergence, the fluid-solid interface, the viscous memory and the C-PML on every
derivative -- and the transposes of each. Records of all eight receiver fields and the
gradients in every coefficient agree with eager to the two backends' run-to-run rounding,
for every kind of source, with the C-PML or an open-pore surface, through water, porous and
dry regions, with the viscous memory and with a drag tensor
(`test/sem/test_sem_c_biot2d.py`). The adjoint checkpoints in chunks, PropSEM's default there,
as for the velocity-stress classes. On an A100, a layered porous rock on 201 × 401 for 2000
steps (order 4, forward and backward) took 1.58 s and 6.1 GB, against 7.00 s and 5.7 GB with
eager's defaults. `BiotVSSEM3DDeformed` runs the same step with y (`sem3d_biot_forward` /
`_backward`); its flux copies go in groups of their number (1 inside an element, 2 on a face,
4 on an edge, 8 at a corner), one launch per group, with the eager class's blocks of the drag
step. It agrees with eager the same way (`test/sem/test_sem_c_biot3d.py`, the fluid-solid faces
included); on 41³ for 600 steps it took 1.04 s and 2.4 GB against 4.20 s and 4.5 GB.

### Validation

`test/sem/test_biot_sem.py` checks the analytic line source (solid velocity
and pore pressure) from inviscid to the Darcy limit, with the viscous
attenuation against the frequency-dependent drag, the open-pore half-space
against wavenumber integration, the pressure-release image of a pore fluid on
its own, the pore pressure on the source's grid line, the FD records, the
undrained limit, the drag tensor's symmetries, the parameter families and
the gradient (also through the drag tensor, the viscous memory and three
media). A dry solid is `ElasticVSSEM2DDeformed` and water on a dry solid is
`AcousticElasticSEM2D`, both to round-off; across water, porous rock and
shale the Green's functions are reciprocal to round-off.
`test/sem/specfem_compare/` runs SPECFEM2D (inputs from
`make_specfem_poro_cases.py`): `run_compare_poro.py` its
`poroelastic_semi_infinite_homogeneous` example against its reference
seismograms (4e-4); `run_compare_poro_features.py` the point force, the
anisotropic permeability and the viscous drag with and without attenuation
(relative L2 0.02-0.2%; the attenuation moves the records by 0.3-1%, and
ours by the same to 3-15% of that); `run_compare_poro_coupling.py` its
`poroelastic_acoustic` example and porous-elastic and water-porous-elastic
layers (0.01-0.08% in the source band; the example 0.07-0.85% on its own
mesh, where its inviscid slow wave has about one element per wavelength,
and 0.003-0.013% with both on elements half as large). SPECFEM2D computes
no pressure in poroelastic media, so these compare velocities. Across a
porosity jump (porous rock over a more porous sand, `pp_layers`), the two
codes differ by 2-3% in the waves transmitted below the interface, on 5 m and
on 2.5 m elements alike, whereas each layer alone agrees to 0.01-0.07%. The FD
`Biot` on a 1.25 m grid, an independent discretisation of the same open-pore
conditions, agrees with the SEM to 0.3% there. The difference is the same at
normal incidence and at 75° (a line of receivers below the interface), so it
does not come from SPECFEM2D's continuity of the tangential relative
displacement, which normal incidence leaves inactive. It comes with the jump in
porosity and tortuosity alone. A jump in the frame moduli and the permeability,
with the same porosity, tortuosity and densities on both sides, agrees to
0.02-0.08%. A jump in porosity and tortuosity alone gives 0.8-1.5%.
SPECFEM2D eliminates the solid-fluid inertia element by element, with each
element's own `phi / T` and `rho_f / rho_bar` (its masses
`rho_bar - phi rho_f / T` and `m - rho_f^2 / rho_bar`), and then assembles. On
an interface node shared by elements whose ratios differ, that is not the
assembled system.

Boundary saving is refused on both backends because the viscous drag is
dissipative everywhere, so the field cannot be stepped back; use `Full()` or
`Ckpt(chunks=...)`.

### In 3-D

`BiotVSSEM3DDeformed` is the 3-D class, on the Hex27 mesh of
`ElasticVSSEM3DDeformed` (flat, or under a topography). It has the same eight
coefficients, the same placement (`v` and `p` continuous, `q` and the
stresses element-local) and the same exact drag at every node, with the
open-pore surface, the C-PML and `viscous_q`:

```python
from sweep.sem import BiotVSSEM3DDeformed, PropSEM

solver = PropSEM(BiotVSSEM3DDeformed, shape=(nz, ny, nx), dh=4.0, dt=2e-4,
                 free_surface=True, f_dominant=30.0, source_type=["p"],
                 receiver_type=["vx", "vy", "vz", "p"])
seis = solver(wavelet, sources, receivers, models=[H, mu, C, M, rho, rho_f, m, b])
```

The flux copies at a node, one per element that shares it (1, 2, 4 or 8), are
grouped by their number, and the drag's exponentials are stored per group:
about 2.7 kB per element in float32 at `n_order=4`. A compiled step costs about
2.5 times one of `ElasticVSSEM3DDeformed` on the same mesh and 1.4 times its
memory (7.4 against 2.9 ms on 27k elements, float32, RTX 6000 Ada).

The sources are the 2-D class's with `y`: the moment tensor `sxx` ... `syz` on
the total stress (`["sxx", "syy", "szz"]`, the explosion, is the default), the
forces `fx`, `fy` and `fz` on both equations, the velocities and fluxes `vx`
... `qz`, and the pore-pressure injection `p`. Fluid and dry elastic elements
work as in 2-D, the fluid-solid faces through `FluidSolidInterface3D` (the
same two integrals on the faces of the Hex27 elements, exact transposes). An
interface may cross a C-PML band that does not stretch its normal direction
-- a sea floor crosses the x-y corner columns of the layer -- or one that
stretches only that direction, but not a region that stretches both, the 3-D
form of SPECFEM2D's corner rule. What the 2-D classes have and this one does
not: an anisotropic drag, and an attenuating frame
(`ViscoBiotVSSEM2DDeformed` is 2-D).

With fluid and dry elastic regions it agrees with SPECFEM3D's acoustic-elastic coupling to
round-off (see [Against SPECFEM3D](#against-specfem3d)).

`test/sem/test_biot_sem3d.py` checks that a dry solid is
`ElasticVSSEM3DDeformed` to round-off and the undrained limit to 3e-3 on 40 m
elements (the continuous pore pressure averages div v at the nodes, as in 2-D). It also
checks the drag's operators node by node in a medium that changes from
element to element, the flux divergence as the transpose of the pressure
gradient on a curved mesh, the three axes against each other, the gradient in
`mu`, `M`, `m` and `b` against central differences (with `viscous_q` and the
open-pore surface), shots and checkpointing. On CUDA it compares a point
pore-pressure injection in Dai's medium with the analytic solution (the fast
and the slow P wave): the solid velocity and the pressure agree to 0.3-0.8%,
inviscid and with a propagating but attenuated slow wave.

## Wavefield snapshots

`return_wavefield=True` (eager) returns `(seis, snapshots)` with
`PropTorch`'s `snapshot_times` / `snapshot_interval` rule (default: every
step) and its layout: the snapshots are on the grid of the model and its
absorbing layer, each grid point taking the SEM field through its element's
basis (zero above a topography):

```python
seis, snaps = solver(wavelet, sources, receivers, models=[vp],
                     return_wavefield=True, snapshot_times=[400, 800])
# snaps: (nsnap, nfield, nshots, 1, nz + pads, nx + pads), detached, on the host
solver.snapshot_fields               # the fields, e.g. ['v_z']
```

The fields are the ones the receivers record from (`v_z` behind
`receiver_type=["vz"]`, `chi` behind `p` on the potential form); the shots
run one by one. A receiver records through the same basis, so its trace is the
snapshot at its grid point. `impl='c'` keeps no intermediate state and refuses
snapshots, as do an explicit `BoundarySaving()` and `Ckpt()`; the default
strategy gives way to the full tape for them.

## Illumination

`compute_illumination = True` gives PropTorch's pair after a call and its backward:

```python
solver.compute_illumination = True
syn = solver(wavelet, sources, receivers, models=[vp])
(syn * obs).sum().backward()
image = vp.grad / (solver.source_illumination + eps)
solver.receiver_illumination           # after the backward
```

- **`source_illumination`** is `sum_t |a|^2` at every GLL node, summed over the
  shots. `a` is the second time derivative of the field the gradient is built
  from: `chi_tt` on the potential forms, whose chi in PropSEM's run is the
  pressure the wavelet drives (the wavelet is the pressure source, see above),
  so `a = p_tt` as on the pressure form; `u_tt` on the displacement forms
  and `v_t` on the velocity-stress forms. It is the pseudo-Hessian of Shin et al.
  (2001) and SPECFEM2D's first approximate Hessian (`APPROXIMATE_HESS_KL`,
  `b_accel . b_accel`).
- **`receiver_illumination`** is `sum_t |lambda|^2`, where the adjoint wavefield
  `lambda = M^-1 dL/df` is the field's gradient mapped back through the mass the
  step inverts.

Both come out shaped like the call's models, the way the gradient does. Each
element's GLL nodes take their node's value times `w |J|`, the factor every
model value's gradient carries, and a grid model takes the transpose of its
interpolation. So `grad / illumination` divides out the quadrature and the
interpolation.

On `AcousticElasticSEM2D` the fluid's and the solid's fields add on the
interface nodes. Their units differ there, so compensate each domain apart. The
switch does not change the gradient.

It runs on `impl='eager'` (on the full tape, shot by shot: an explicit
`BoundarySaving()` or `Ckpt()` is refused, the default strategy gives way) and on
`impl='c'` for the three 2-D C-PML paths: `AcousticPotentialSEM2D`,
`ElasticSEM2DDeformed` and `AcousticElasticSEM2D`. On `impl='c'` it also covers the
shot batch and boundary saving, and the eager and CUDA values agree to 1e-5. The sums are
kept in fp64 and returned as fp64: the receiver term reaches 1e43 on notebook 48's model,
out of fp32's range.

Record the field itself when you want the receiver term. A derived receiver such as
`p = -chi_tt` differentiates the residual twice in the adjoint, which weights it by
`omega^4` toward frequencies the mesh does not resolve. The receiver illumination then
follows that part. In notebook 48 recording `p` instead of `chi` steps it by 3x at the
first reflector and puts the mesh's lines into it; the source term and the image are
unaffected.
`impl='c'` without a C-PML, the 3-D paths and domain decomposition refuse it, as
do the Biot classes, whose solid and fluid velocities share a 2x2 mass
(`test/sem/test_sem_illumination.py`).

## The model edge

The absorbing layer reads the model's edge values, as FD's edge padding
does, and like FD the gradient stops at the model edge: the layer's material
is a constant of the model. Before, the clamped interpolation
was differentiated, and the whole C-PML band's sensitivity landed on the
outermost ring of cells -- the strip around the edge of every SEM gradient.
On the notebook-36 model the ring carried 24% of the gradient's L2; it now
carries 4.3%, against FD's 4.2%. A node on the model edge counts as inside, on
both elements that share it, which is what keeps the eager and `impl='c'`
gradients equal.

## A sea floor through the absorbing layer

On `AcousticElasticSEM2D` a sea floor runs through the side bands of the C-PML, where
each domain's absorbing form carries its stretching into the interface terms. The solid's
form (SPECFEM2D's) is scaled by `s_x s_z`, so its load from the water is
`(s_z*p n_x, s_x*p n_z)`. The fluid's double-convolution form is not scaled, and its load
from the rock is `(1/s_x)*u_x n_x + (1/s_z)*u_z n_z`. On a horizontal floor through a side
band that is `s_x*p` on the solid and the plain term on the fluid. The interface points
inside a band carry the four convolution memories (`sweep.sem.coupling.InterfaceCPML`), on
both backends, in the shot batch, under boundary saving and on domain decomposition.

With the plain terms the floor reflected several times what the layer does on the rock
alone. With 200 m of water over rock, a 10 Hz source 20 m above the floor, 20 m elements and
a ten-element layer, against a box with no boundary within reach (relative L2 per receiver,
the largest):

| Receivers | Plain terms | Stretched | The rock alone |
| --- | --- | --- | --- |
| 10 m above the floor | 9.0e-3 | 1.3e-3 | 2.4e-3 |
| 10 m below the floor | 7.8e-3 | 2.2e-3 | 2.3e-3 |
| 300 m below the floor | 3.9e-2 | 3.8e-3 | 3.8e-3 |

SPECFEM2D stretches both loads by `s_x` (`compute_coupling_viscoelastic_ac`,
`compute_coupling_acoustic_el`). That is right for its fluid, whose form it scales too, but
on ours the `s_x` on the fluid load brings the largest error back to 1.3e-2.
`PropSEM` extends the interfaces flat across the lateral pads, so its floor is horizontal
in the bands. A sloped one, through the low-level `AcousticElasticSEM2D.build`, takes all
four terms. A band corner on the interface is still refused, as in SPECFEM2D.

## Domain decomposition (multi-GPU, phase 1)

`sweep.sem.dd` splits the model domain into **x strips** across ranks on
the **impl='c' backend** — the SEM mirror of the FD `sweep.parallel` stack,
sharing its API shape (`SEMModelParallel(prop, StripTopology(px))` with the
propagator as config carrier, global coordinates in, `gather_record` out):

```python
import torch.distributed as dist
from sweep.sem.dd import SEMModelParallel, StripTopology

dist.init_process_group("nccl")          # torchrun --nproc_per_node=P
ddp = SEMModelParallel(prop, StripTopology(px=world, rank=rank))
rec = ddp(wavelet, src_node, rec_nodes, [vp])   # this rank's columns
full = ddp.gather_record(rec)                   # assembled on rank 0
```

A cut needs an interface-column **sum** after each step's assembly, so DD
runs the time loop in Python over split-phase C entry points (the FD
stepped pattern): forward = `dd_grads` → exchange six fields → `dd_step`;
backward = `dd_bwd_pre` → exchange 6 → `dd_bwd_mid` → exchange 3 →
`dd_bwd_end`. The exchange is self-adjoint, and **four** DD-specific dedup
rules make the gradients match the fused single-domain adjoint:

1. per-node material gradients are ownership-masked on cut columns — the
   adjoint exchange leaves them COMPLETE on both sides, so summing along
   the strip chain would double them;
2. the wavelet gradient (a global scalar reduction) reads an
   ownership-masked source pattern;
3. a receiver sitting exactly ON a cut is **registered** on both strips —
   otherwise the non-owning side never gets its adjoint injection — but
   **owned** by one, and the duplicate row's `grad_syn` is taken from the
   owner;
4. the source pattern is **not** exchanged at setup: it is a slice of a
   global field and is already complete on both sides, so summing it
   doubles a source that lands on a cut.

Rules 3 and 4 are not hypothetical: skipping them cost 94% on the vp
gradient and 100% on the forward respectively. All four are pinned by
tests against the fused driver, including geometries that deliberately put
the source and receivers on the cut (the px=1 case isolates the stepped
split itself).

Setup follows a strict *slice, don't rebuild* rule: masses, `kappa_node`
and the C-PML profile come from ONE global build, column-sliced per rank —
a per-strip profile rebuild would grow absorbing bands at every internal
cut (the mirror of FD's "PML hole"), and the forward asserts the cut
columns are PML-free (against the GLOBAL profile's peak, since a strip
lying wholly outside the band has no scale of its own).

### Flat 3-D acoustic: nothing global on a rank

For `AcousticSEM3D` with one shared model the rule above is kept in
*value* but not in *construction*: a rank builds neither the global
equation nor the global mesh. It reads the propagator's `Hex8Layout` — the
reference element, the element size and the three 1-D node-coordinate axes,
O(nx+ny+nz) — and produces its own strip's material, mass, geometry
factors, free-surface pins, C-PML block and source pattern from that. Every
entry is bit-identical to slicing a global build (`test_sem_dd_consts_contract.py`
holds the comparison at `== 0.0`), because the same floats are computed by
the same operations: the axes ARE the mesh's coordinate lines
(`x_global` is a meshgrid of `x_axis`), the strip's connectivity composed
with its column map IS the global row-major numbering (checked, with a
negative control, in `test_sem_hex8_layout.py`), and the mass is the same
repeated float32 addition the global `index_add_` performs.

`PropSEM.mesh` is therefore built on first access rather than in the
constructor (same object, same numbers, later — topography meshes stay
eager so an inverted element is still refused at construction). A rank that
only runs DD never triggers it; the fused path does, on its first forward.
Assigning `prop.mesh = other` is honoured by both paths: a mesh built by
`build_hex8_mesh` carries the layout it was tiled from, and the DD strips
describe that one.

What that buys is the size ceiling. Per rank, the global mesh was
50.7 B per *global* node (`ibool` 15.1, `x_e/y_e/z_e` 22.8,
`x/y/z_global` 12) and did not shrink with the number of GPUs; with the
compact source it was the last such term. Measured forward-only on one
GPU: 349 B/node with the mesh forced into existence, 298 without — the
same 51 B/node. The px-invariant residue is now the caller's model tensor
(4 B/node): per rank the account is about 4.5 + 267/px bytes per global
node, measured at px = 1, 4, 8 and 9 (the strip term was 298 before the
flat geometry weights stopped being tiled per element — the 3-D kernels
read those per thread, so `m` floats replace `E·m`). In practice
(forward only, nt=40)
eight 32 GB V100s take a 897³ model, eight 80 GB A100s a 1217³ one
(1025³ at 42 GB per rank), and nine A100s 1225³.

### Topography (`AcousticSEM3DDeformed`)

Covered, on the same strips and the same stepped kernels. Only the geometry
the strip consts carry changes: the flat kernels read `w_jac` per THREAD —
valid because every element's slice is identical on a flat mesh, which is
what makes an element subset legal there — while the deformed ones read
`w_J` and the nine inverse-Jacobian metrics per ELEMENT-NODE. So a strip
slices all ten from the global build by element id and rebuilds the strip
mesh for its **ibool numbering only**.

That rebuild-vs-slice split is the load-bearing part, so it is asserted
rather than documented: `_assert_strip_ibool` checks that the rebuilt
connectivity reproduces the global one restricted to the strip, i.e. that
global node `gidx[j]` really is the strip's node `j`. A different element
family, or a mesh whose numbering depended on its coordinates, would
misalign every sliced field at once and otherwise do it silently.

Modes and bars: without `dist` one process steps every strip (the
correctness harness, one GPU); under `torchrun` each rank owns one strip
via batched `isend/irecv` (NCCL needs one GPU per rank). Unlike FD's
copy-semantics halos, SEM assembles by **summing**, so the cross-device bar
is tolerance-based (fp32 atomics floor), never bitwise.

**Boundary saving composes with DD** — the strips take the propagator's
`memory`:

```python
prop = PropSEM(..., memory=BoundarySaving())
rec = SEMModelParallel(prop, StripTopology(px))(wavelet, src_node, rec_nodes, [vp])
```

Each strip stores only its OWN C-PML band ring, so the two memory levers
multiply: per rank O(nt · |local band|) instead of O(nt · n_global). The
composition needs **no extra interface exchange**, but not for the reason
that first suggests itself. A cut is *not* generally outside the ring — the
rim reaches it through the other axes (the bottom band in 2-D, the y and
bottom bands in 3-D), and on small grids the band covers nearly everything.
What makes it sound is that the profile is sliced from ONE global build, so
the two strips sharing a cut compute **identical band membership** for it:
every shared node is either restored on both sides from rings holding the
same (exchanged) forward values, or recomputed on both sides by the same
pointwise march fed by the already-exchanged assembly — never one of each.
A per-strip profile rebuild is what would break this, which is the same
defect the sliced-profile assertion catches.

The **fluid-solid** path is the one exception to "no extra exchange". Each
strip rings both domains (the fluid band on nodes, the solid band on nodes
*and* elements, since its history buffers are element-local) and re-anchors
on full-state checkpoints every `bs_frame_checkpoint_every` steps, because
the coupled reverse march accumulates error the other paths do not — see
**Reverse-march stability** below. Its reverse march has to
recompute the solid acceleration, and that is an **assembly** — so boundary
saving adds one 4-field exchange per reverse step there, the same set the
forward exchanges after `dd_mid`. Measured on a 24×12-element grid at px=2
(bands 57% fluid / 46% solid nodes / 44% solid elements, so the march
genuinely runs): every material gradient and the wavelet come out at
cos 1.000000 and L2 3–12e-06 against the save-all tape, matching the fused
single-domain driver's own 3–11e-06, and peak memory drops 109 MB → 60 MB.

**Reverse-march stability.** Boundary saving reconstructs the forward state
by marching it backwards, so how that march behaves decides whether anchors
are needed. Measured (eager path, worst of the ringed fields, error normalised
by the record's **peak** field):

| equation | rate | error at 200 steps | at 600 steps |
|---|---|---|---|
| acoustic | ×0.9995 /step | 3e-7 | — |
| elastic | ×1.0001 /step | 3e-8 | 1.6e-6 (flat from ~200) |
| fluid-solid, no anchors | ×1.009 /step | 2.4e-5 | 1.8e-5 (plateau) |
| fluid-solid, anchors | — | 9.8e-11 | 4.2e-11 |

So only the coupled path needs the checkpoints, and it needs them for a
*plateau* at ~2e-5, not for a runaway. Two earlier claims in this codebase did
not survive re-measurement and are gone: that the coupled map compounds at
×1.15/step and "eventually overflows", and that elastic BS degrades as the
marched interior grows. The elastic boundary-saved gradient is cos 1.0000000
against the full tape at nt=400 for every band fraction from 100% down to
36.7%, on both `eager` and `c`.

Both retractions trace to the same two measurement traps, which are worth
knowing before trusting any curve of this shape:

- **normalising by the instantaneous ‖truth‖** — the reverse march ends where
  the source has not rung yet, so the denominator vanishes and manufactures a
  clean exponential. Doing this reports ×1.04–1.09 /step for the acoustic,
  elastic *and* coupled equations alike, including the acoustic one, whose
  reconstruction is exact.
- **rebuilding random models between the reference run and the test run**
  without a fixed seed — that compares the gradients of two different media,
  which lands at cos 0.3–0.9 and drifts with configuration, looking exactly
  like accuracy falling off with some parameter.

**Ring precision.** `BoundarySaving(storage_dtype=...)` picks how the ring is
stored — `'fp32'`, `'bf16'` (2× smaller) or `'int8'` (4×) — on the strips as
on the single domain:

```python
prop = PropSEM(..., memory=BoundarySaving(storage_dtype="int8"))
```

The decomposition adds nothing here: DD's degradation matches the fused
single-domain driver's to three digits on the same problem, which is what
the tests assert. What *does* decide whether int8 is usable is one property
of the setup — **is the source cell inside the band?** "Band" here is the
saved set, not the absorbing layer: every node of every element that touches
a node with a non-zero C-PML coefficient, i.e. the rim widened by one
element. int8 keeps a single amax per (field, step) over that whole set, so
a source inside it makes the amax the source's near-field spike and
quantises every other node toward zero (the band's amax/median runs 1e6 to
1e9 there). Measured on the 2-D acoustic path, same code, same bars:

| source | bf16 | int8 |
|---|---|---|
| inside the band (small test grid, band 90%) | 1.3e+01 | 1.3e+03 |
| outside the band (24×12 elements, band 57%) | 6.2e-06 | 3.5e-06 |

The regimes line up conveniently: a band big enough to swallow the source
is a band that is most of the domain, and there the ring is barely smaller
than the tape — so the configurations where int8 fails are the ones where
boundary saving was not worth enabling anyway (band 100% → 0% saved → int8
L2 ~1e0; band 57% → 43% saved → 3.5e-06). Low-precision rings are for
production geometries, where the rim is a rim and sources sit in the
interior; on a domain small enough for the band to reach the source, keep
fp32 — and expect little memory back either way.

**The coupled fluid-solid ring takes it too**, and takes it well: three ring
families (five fluid node rings + two solid node rings on one amax table, six
solid ELEMENT rings on their own) all reach the fp32 ring's *own*
reconstruction error — measured on a 24×12-element grid, fp32-vs-tape
1.4–8.3e-06 against bf16 1.2–8.3e-06 and int8 1.5–9.6e-06. The march seeds
and re-anchor checkpoints stay fp32 by design; they are what the march
restarts from.

**The elastic ring is coarser.** It is two rings — `u` on band nodes and six
element-local history buffers on band ELEMENTS — and int8 keeps one amax per
(field, step) for the first and per (field, step, *element*) for the second.
Even so int8 lands about 8× above bf16 on the gradient (measured on a
32×16-element model with a 5-element C-PML rim added outside it, band 59% of
the nodes, source outside it: bf16 L2 0.9–2.6e-04 at cos 1.000000, int8
1.0–2.0e-03 at cos 0.999998–0.999999).
**Prefer `bfloat16` on the elastic path**, or better, split the two ring
families — see below.

Where that cost sits is worth knowing, because it is not spread evenly.
Quantising one ring family at a time to int8, the other kept fp32, gives:

| family quantised | vp | vs | rho |
|---|---|---|---|
| the two `u` NODE rings | 2.0e-03 | 1.4e-03 | 1.0e-03 |
| all six ELEMENT rings | 2.3e-04 | 5.5e-05 | 1.1e-04 |
| everything | 2.0e-03 | 1.4e-03 | 1.0e-03 |

These are model gradients. Most of the `u` rings' error never reaches them:
it lands on the gradient of the absorbing layer, which sits outside the
model and is not returned. Over the whole mesh, layer included, all-int8
sits ~120× above element-only int8 (2.9e-02 / 4.7e-02 / 1.7e-02 against
3.9e-04 / 1.6e-04 / 2.9e-04), not ~9×.

So the whole cost is the `u` rings — which are only **18%** of the ring
memory, the six element rings being the other 82%. That is what
`bs_store_dtype_elem` is for: it overrides the store dtype for the ELEMENT
rings only, so the bulk can go to int8 while `u` stays exact.

```python
prop = PropSEM(..., impl="c", memory=BoundarySaving(storage_dtype="bf16"),
               bs_store_dtype_elem="int8")
```

Measured on the same grid (peak of the whole propagation, and the worst of
the three material gradients against the fp32 ring):

| store | peak | gradient |
|---|---|---|
| save-all tape | 681 MB | — |
| fp32 ring | 411 MB | — |
| all bf16 | 224 MB | 2.6e-04 |
| all int8 | 143 MB | 2.0e-03 |
| nodes fp32, elements int8 | 194 MB | 2.3e-04 |
| **nodes bf16, elements int8** | **160 MB** | **3.4e-04** |

The last row is the one to use on this path: 12% more memory than all-int8
for 6× the accuracy, and 29% less than all-bf16 at 1.3× its error. The
flag is refused on paths that ring nodes only, rather than silently doing
nothing. Read the split as `u` being inside the
march recursion (`march_fin` injects it, `march_asm` reads it back to
rebuild the histories and the acceleration) while the element histories only
feed the imaging transpose. That reading is not established: the coupled
path has the same `u` rings and shows none of the cost, and neither the
checkpoints (disabling them changes nothing) nor march length (the ratio is
flat in nt) account for the difference. The elastic and fluid-solid
paths are fp32-only regardless — `SemElasticCpml*Input` carries no
`band_kind`/`band_scale` — and say so rather than silently storing fp32.

One 3-D caveat worth knowing before reaching for it: a C-PML rim is a shell
on five of six faces, so the band is a much larger fraction of the volume
than in 2-D. On a 25×49×129 grid with a 1-element rim the band is still 65%
of the nodes, and the measured saving (151 MB of a predicted 156 MB, tape
616 MB → ring 466 MB) tracks that fraction. The 3-D memory win is real but
only becomes large on production-sized domains.

Verified on 2–3× H100 (NCCL): forward and gradients match the fused
single-domain driver at its own run-to-run floor (grad rel ~1.5e-6, and the
cut columns are the *cleanest* part of the field, 1.7e-8 at px=3). Under
boundary saving the DD gradient stays as close to fused-BS as DD-save-all
is to fused-save-all — the decomposition adds nothing on top of the
reconstruction's own approximation.

Coverage (all C-PML, single shot):

| path | forward exchange | backward exchange | boundary saving |
|---|---|---|---|
| `AcousticPotentialSEM2D` | 6 fields | 6 + 3 | yes |
| `ElasticSEM2DDeformed` | 2 | 4 (+2 for the BS march) | yes (fp32/bf16/int8) |
| fluid-solid (`AcousticElasticSEM2D`) | 7 + 4 (two assemblies) | 2 + 6 + 3 + 4 (+4 under BS) | yes (fp32/bf16/int8) |
| `AcousticSEM3D` | 9 | 9 + 4 | yes |

Every ring takes `storage_dtype` (fp32/bf16/int8); what it costs differs
sharply by path. See **Ring precision**.

The field count per exchange is just the number of assembled quantities in
that step, so it tracks the equation, not the mesh: the elastic step has
only ONE assembly per time step, and its C-PML coefficients are
element-local, so internal cut elements carry no PML by construction — the
pure-elastic path is the only one exempt from the sliced-profile assertion,
which every path carrying a node-level `b_x` (including the fluid half of
fluid-solid, and 3-D) still runs. The fluid-solid
path needs no special machinery where the seabed meets a vertical cut — the
exchange moves the whole cut column and the interface node is simply one of
its entries — but interface facets must be assigned by their **element's**
column, since assigning by the node's own column double-counts the shared
one.

**3-D** decomposes the same way with the cut being a whole **YZ plane**:
the global numbering `((iz*ny_g + iy)*nx_g + ix)` puts x fastest, so a cut
plane is still a strided index set and the 2-D geometry generalises by
widening the row axis to `nz_g * ny_g`. Same four dedup rules, same
sliced-profile assertion, same bars (measured: forward and both gradients
5–6e-7 against a fused floor of 2.5e-7–1.1e-6 at px=1/2/3).

Comm/compute overlap was **measured and rejected**, not skipped: after
batching the interface indexing into whole-parent `block()` exchanges
(1.9–4.3× cumulative, which also flipped 4-GPU scaling from slower- to
faster-than-2), the residual ~113 µs per exchange is local kernels and call
overhead against a bare-P2P floor of 36 µs — an async microbenchmark came
out *slower* than serial. The next lever, if one is needed, is fusing
gather+add+scatter into a single kernel, not FD-style phase splitting.

Distributed checks (multi-GPU box):
`torchrun --nproc_per_node=P test/sem/dd_api_check_sem.py` and
`test/sem/dd_grad_compare.py` (saves single-vs-multi gradients for
plotting).

## Against SPECFEM2D

`test/sem/specfem_compare/run_compare_box.py` runs the 2-D classes against SPECFEM2D 8.1.0
built in double precision. The cases come from `make_specfem_box_cases.py`:

- a homogeneous medium in a 3000 m box of 10 m elements with all four sides free;
- a 15 Hz Ricker inside an element;
- receivers on a 500 m ring around the source and 10 m below the top edge;
- 6000 steps of 0.2 ms, the edge echoes included.

The box has no absorbing layer, so the two codes solve the same problem. Our step runs in
fp64, and no shift or scale is fitted. The records agree as follows (relative L2 over all
receivers):

| Class | Case | Agreement |
| --- | --- | --- |
| `ElasticVSSEM2DDeformed`, `ElasticSEM2DDeformed`, `ElasticSEM2D` | rock, a force and an explosion | 1e-13 |
| `ElasticVSSEM2DDeformed`, `ElasticSEM2DDeformed` | an explosion under a 100 m Gaussian hill | 3e-12 |
| `ElasticVTISEM2DDeformed`, `ElasticTTISEM2DDeformed` | a VTI shale with a vertical and a 30° axis, a force and an explosion | 1e-13 |
| `AcousticPotentialSEM2D`, `AcousticSEM2D` | water, a pressure source, the pressure | 1e-12 |
| `AcousticElasticSEM2D` | 600 m of water over the rock: hydrophones and geophones | 2e-12 |
| `AcousticElasticSEM2D`, with `interfaces` | the same over a curved sea floor (a 150 m seamount) | 4e-12 |
| `ElasticVSSEM2DDeformed`, `ElasticSEM2DDeformed`, with `interfaces` | the rock over a stiffer one across a sinusoidal interface | 2e-12 |
| `ElasticSHSEM2DDeformed` | SH, a force along y, flat and under the hill | 1e-13, 2e-12 |
| `ViscoElasticVSSEM2DDeformed` | the rock with Q = 30, SPECFEM2D's conventions | 2e-4 |
| `ViscoElasticSEM2DDeformed` | the rock with Qkappa = Qmu = 30 read at 15 Hz, an explosion; with Qkappa = 60, Qmu = 20 read as unrelaxed, a force | 7e-14 |
| `ViscoElasticSEM2DDeformed`, `fit="nonlinear"` | Qkappa = 30, Qmu = 10 with SPECFEM2D's `USE_SOLVOPT`, a force | 2e-5; 6e-14 on SPECFEM2D's own mechanisms |
| `ViscoAcousticPotentialSEM2D` | the water with Qkappa = 30, SPECFEM2D's conventions | 3e-2; 8e-13 with SPECFEM2D's memory-update line |
| `ElasticSEM2DDeformed`, `ElasticSHSEM2DDeformed`, a `PlaneWave` | plane P, SV and SH waves and an SV wave under a soft basin, SPECFEM2D's `initialfield` far from its Stacey edges | 1.3e-4 to 2.5e-4, each run's own error (see [Plane-wave incidence](#plane-wave-incidence)) |

Matching the two codes takes these conventions on our side:

- **The velocity record.** SPECFEM2D's Newmark velocity at a step is the mean of the two
  half-step velocities around it. On the velocity-stress classes a linear interpolation of
  ours reproduces it. On the displacement classes it is the central difference that
  `PropSEM` takes.
- **The explosion's wavelet.** `PropSEM`'s stress source loads the running sum of the
  wavelet, so the wavelet is the backward difference of SPECFEM2D's moment samples.
- **The acoustic source.** SPECFEM2D's pressure source adds `h stf / kappa` to the
  potential's load, so our `chi` source takes `stf / kappa`.
- **`AcousticSEM2D`.** Its field is SPECFEM2D's potential for the wavelet `stf`, and the
  pressure is the trace's second difference. Driving the field with the second difference of
  the source instead would start it with a step, because SPECFEM2D's Ricker is not zero at
  its first sample.

The remaining differences are round-off, except for two classes:

- **Under the hill**, the receivers next to the curved surface reach 5e-11. That comes from
  locating them in curved elements: our Newton tolerance is about 4e-11 m there, where
  SPECFEM2D locates them to 1e-12 m.
- **The pressure** is a second difference of the potential, which amplifies round-off.
- **On the curved interfaces**, two receivers differ by 2e-7 and 4e-6. SPECFEM2D locates
  them in the element of their closest GLL point, 1% and 0.3% outside it
  (`|gamma| = 1.010`, `1.003` in its log), and extrapolates that element's basis; ours
  interpolates in the element that holds the point. The table leaves those two out.
- **The velocity-stress visco-elastic records** keep the time scheme of the memory variables
  (see [Visco-elastic attenuation](#visco-elastic-attenuation)). The displacement class runs
  SPECFEM2D's scheme and agrees to round-off.
- **The visco-acoustic records** keep a line of SPECFEM2D's that we do not reproduce (see
  [Visco-acoustic attenuation](#visco-acoustic-attenuation)).

The Biot classes discretize the poroelastic equations differently from SPECFEM2D (see
[Validation](#validation)). The 3-D classes are compared with SPECFEM3D in the next section.

## Against SPECFEM3D

`test/sem/specfem_compare/run_compare_box3d.py` runs the 3-D classes against SPECFEM3D
(Cartesian) 4.1.1 built in double precision. The cases come from
`make_specfem3d_box_cases.py`:

- a homogeneous rock (vp 2000, vs 1000 m/s, rho 2000 kg/m³) in a 320 m cube of 10 m Hex8
  elements from SPECFEM3D's internal mesher, with all six sides free; the same rock under
  80 m of water; the rock transversely isotropic; or the rock over a stiffer one across a
  curved interface;
- a vertical point force inside an element, a 25 Hz Ricker that SPECFEM3D reads as an
  external source time function;
- ten stations inside elements, on element faces and edges, and on the free surface;
- 1600 steps of 0.25 ms, the echoes from the six sides included.

As against SPECFEM2D, our step runs in fp64 and no shift or scale is fitted. The records agree
as follows (relative L2 over all traces, each component scaled by its peak):

| Class | Case | Agreement |
| --- | --- | --- |
| `ElasticVSSEM3D`, `ElasticVSSEM3DDeformed` (flat), `ElasticSEM3D` | the elastic rock | 5e-14, 5e-14, 8e-14 |
| `ViscoElasticVSSEM3D` | Qkappa = Qmu = 30, SPECFEM3D's band (2.5 to 250 Hz, three mechanisms), the velocities read at 25 Hz (`velocities="real_modulus"`) | 3.8e-3 |
| `ViscoElasticVSSEM3D` on SPECFEM3D's mechanisms | the same; Qkappa = 60, Qmu = 20 | 3.6e-3, 4.4e-3; 1.3e-14 with SPECFEM3D's memory update |
| `ElasticVTISEM3D`, `ElasticVTISEM3DDeformed` (flat) | the rock transversely isotropic (`epsilon = 0.25`, `delta = 0.1`, `gamma = 0.15`), a vertical axis | 5e-14 |
| `ElasticTTISEM3D`, `ElasticTTISEM3DDeformed` (flat) | the same medium, its axis tilted by 30° toward the azimuth 40° | 5e-14 |
| `ElasticVSSEM3DDeformed`, with `interfaces` | the rock over a stiffer one (vp 3000, vs 1700 m/s) across an interface 160 m deep with 20 m of relief | 3e-14 on SPECFEM3D's straight-sided elements; 9e-3 on ours, whose faces follow the interface |
| `BiotVSSEM3DDeformed`, fluid and dry elastic regions | 80 m of water over the rock, the force in the rock: the pressure and the fluid velocity in the water, the velocity in the rock | 2e-12; 2e-3 to 5e-3 for the fluid velocity across the element faces a station sits on |

On SPECFEM3D's mechanisms our class takes its relaxation times, its strengths (its simplex
fit of the strain relaxation times, read from its `attenuation.bin`) and its unrelaxed moduli.
Its bulk and shear defects of a mechanism, `kappa_U c_kappa` and `mu_U c_mu`, are our P and
shear ones, `kappa_U c_kappa + 4/3 mu_U c_mu` and `mu_U c_mu`, so separate Qkappa and Qmu
are reproduced exactly.

What separates those records from SPECFEM3D's is the time discretization of the memory
variables. SPECFEM3D advances them on the strain one step ahead, extrapolated from the last
two displacements (`2 u^n - u^(n-1)`), which overstates that strain by a factor
`1 + (omega dt)^2`. At 25 Hz and dt = 0.25 ms its discrete complex modulus differs from the
GSLS modulus by `-1.2e-4 + 5.5e-5 i` relative to its magnitude; ours differs by 3e-6, since
the trapezoidal rule only warps the frequency. With SPECFEM3D's update in place of ours (the
script's `SpecfemScheme`) the records of both cases agree to 1.3e-14, and the difference
between the two updates falls by a factor of four each time dt is halved. Near 25 Hz,
SPECFEM3D's error slows the waves by about as much as the leapfrog speeds them up, so against
a reference run at dt/9 its records are the closer of the two: 7.6e-4 against our 1.7e-3 in
a small box, where the elastic leapfrog alone is 2.0e-3 from its reference.

Our own constant-Q fit over the same band gives Q = 28.6 at 25 Hz, where SPECFEM3D's gives
28.5; both stay within 5% of 30 across the band. With that fit and our memory update the
records differ from SPECFEM3D's by 3.8e-3.

The water is SPECFEM3D's acoustic domain: its potential `chi`, the pressure `-chi''` and the
fluid velocity `grad(chi') / rho`, coupled to the displacement of the rock, with a
pressure-free sea surface and rigid sides. The fluid region of `BiotVSSEM3DDeformed` carries the
pressure on the nodes and the fluid velocity per element. Eliminating the velocity leaves the
pressure with the potential's stiffness, and SPECFEM3D's `-chi''` obeys the same leapfrog,
coupled through the same solid velocity at the half step and the same pressure at the step,
so the two are one scheme. The pressure, the velocity in the rock and the fluid velocity at
stations inside elements agree to round-off. A station on an element face is the exception:
the fluid velocity, a gradient of the pressure or potential, jumps across the face.
SPECFEM3D takes it from the element it located the station in, and ours averages the
elements, so the component normal to that face differs by 2e-3 to 5e-3.

SPECFEM3D takes an anisotropic medium from its `model_aniso.f90`, a routine its users edit,
by the anisotropy flag of the material. The case generator writes the 21 stiffnesses of each
case (`sweep_aniso_cij.inc`) for that routine to include; in SPECFEM3D's z-up axes the eight
entries with one index in `yz` or `xz` change sign. The generator computes them with its own
numpy Bond rotation, and the comparison first checks that the classes build the same ones.

## Backend parity

"Same solver call, same arguments → same numbers" is the standing contract
between `impl='eager'` and `impl='c'` — forward and gradients, shared and
per-shot models, with and without boundary saving. Neither backend is
bit-reproducible on CUDA (both scatter with atomics), so the parity suites
(`test/sem/test_sem_backend_parity.py`, `test/sem/test_sem_shot_batch.py`)
measure the run-to-run noise floor in-session and hold the backend difference
to it, rather than to a fixed tolerance.
