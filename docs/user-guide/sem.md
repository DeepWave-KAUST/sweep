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
  its gradient stops at the model edge (see [The model edge](#the-model-edge));
- `free_surface` — `True` (the top), edge names (`"top"`, `["top", "left"]`),
  a per-edge mask or an `{edge: bool}` dict; default `False`;
- `topography` — the surface row of every column, `(nx,)` in 2-D and
  `(ny, nx)` in 3-D, fractional rows allowed: a free surface on top, which the
  curved mesh follows (it passes through the rows at the element nodes);
- `memory` — `Full()` (default), `BoundarySaving(storage_dtype=...)` or
  `Ckpt(chunks=...)` from `sweep.propagator` (see
  [Boundary saving](#boundary-saving));
- `impl` — `None` runs the CUDA kernels where the equation has them and the
  device is CUDA, eager otherwise; `'eager'` / `'c'` choose;
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
components) for every equation class.

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
| `impl='eager'` | every equation (the BS framework hooks any `field_specs`) |
| `impl='c'` | the five C-PML paths: `AcousticPotentialSEM2D`, `ElasticSEM2DDeformed`, `AcousticElasticSEM2D` (each requiring their C-PML), `AcousticSEM3D`, `AcousticSEM3DDeformed` |

Asking for boundary saving anywhere else on `impl='c'` raises
`NotImplementedError` — it used to be accepted and silently run the full
tape, which looked like it worked.

Knobs:

- `BoundarySaving(storage_dtype='bf16' | 'int8')` — low-precision rings,
  wired on **every** C-PML path (acoustic 2-D/3-D, elastic, fluid-solid, and
  their decomposed twins). What it costs differs sharply by path — see **Ring
  precision** — and the constructor's `bs_store_dtype_elem` sets a separate
  dtype for the ELEMENT rings (`impl='c'`). In the
  shot batch the int8 quantisation scale is **per (field, step, shot)**, so one
  loud shot cannot degrade a quiet shot's reconstruction.
- `bs_band_mode='full' | 'rim'` (constructor) — what the ring covers: the whole
  band (default; gradient equals the full tape everywhere) or only its
  interface line. See **Rim mode**.

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
B per global node at 161³). In `'rim'` mode the memory rings are empty
altogether: the interface line lies outside every slab. Nothing changes in
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
`fluid_solid`, and `bare` for the paths without a C-PML) — the fused
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

`bs_band_mode='rim'` rings the band's interface **line** instead of the band
**area** — a perimeter instead of a surface, so it scales with the mesh edge
rather than the C-PML thickness. It is honoured on every path: the
single-domain acoustic/elastic/coupled propagators and every DD strip. (The
strips and the coupled path used to hard-code the "full" convention and ignore
the flag silently, handing back full-band memory with no warning.)

The bargain, measured worst-of-all-models cosine against the full tape:

| path | band nodes full → rim | band in the gradient | band cropped | band + one element cropped |
|---|---|---|---|---|
| acoustic 2-D, px=1 and 2 | 8816 → 816 (10.8×) | 0.9984 | — | 1.0000000 |
| elastic, px=1 and 2 | 6808 → 620 (11.0×) | 0.0466 | 0.99746 | 1.0000000 |
| fluid-solid, px=1 and 2 | 2737 → 540 fluid, 2197 → 145 solid | 0.68 | 0.9838 | 0.999978 |

Measured when the band lay inside `shape`. It now lies outside the model and
its gradient is cut at the model edge, so a model gradient is the "band
cropped" column, and dropping the model's outermost element gives the last.

Three things that bite:

- **Crop the model's outermost element as well.** One element short is not a
  detail — elastic sits at 0.997 with the band alone cropped and at exactly
  1.0 with that element too.
- **Element rings are exempt** and always cover the whole band. The elastic
  C-PML memory fields live only inside it, and zeroing them corrupts the
  adjoint entering the *interior*, not just the band gradient — dropping them
  takes even full-band node mode to cos 0.755. So rim shrinks the node rings
  only: a big win where the rings are node-only (acoustic), a partial one on
  elastic and coupled, where the element rings are the bulk of the bytes.
- **Fluid-solid plateaus** at ~2e-5 instead of converging to 1.0. The other
  paths reach the full-tape gradient exactly. Unexplained; the two domains'
  bands are built separately there, so their rims meet at the seabed.

Asking for `'rim'` on a mesh whose C-PML band covers every node raises — there
is no interface line to store, and boundary saving would silently ring nothing.
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

## Source types

On the elastic equations `source_type` takes `PropTorch`'s names as well as
the equation's own source fields, and means the same source:

| `source_type` | source | the wavelet is |
| --- | --- | --- |
| `["vz"]`, `["vx"]` (or the field's own name, `v_z` / `u_z`) | a body force along that axis | the force |
| `["vx", "vz"]` | a force at 45°, not an explosion | the force |
| `["sxx", "szz"]` | an explosion: moment tensor `Mxx = Mzz = 1` | the moment rate |
| `["sxz"]` | a shear moment, `Mxz = Mzx = 1` | the moment rate |

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
wavelet differently); the stress names are 2-D only for now, and a per-call
`source_vector` only applies to a force. Both backends carry every source
type: the kernels only ever see the assembled per-field patterns.

A stress source on an element edge -- every `n_order`-th grid index lands on
one -- is stamped in every element that meets there, each taking the share of
a small disk it holds (half a side on an edge, the corner angles at a
corner): the moment tensor differentiates the basis, whose gradient jumps
across the edge, and stamped in one element the explosion came out 1-2%
lopsided. A force needs none of this; its stamp has no derivative.

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
snapshots, as does boundary saving.

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
Even so it only reaches a few percent on the gradient (measured on a
32×16-element grid with a 5-element rim, band 82%, source outside it: bf16
L2 2.2–4.7e-04 at cos 1.000000, int8 1.6–4.3e-02 at cos 0.9991–0.9999).
**Prefer `bfloat16` on the elastic path**, or better, split the two ring
families — see below.

Where that cost sits is worth knowing, because it is not spread evenly.
Quantising one ring family at a time (in Python, against the same amax
contract — it reproduces the real int8 number to three digits) gives:

| family quantised | vp | vs | rho |
|---|---|---|---|
| the two `u` NODE rings | 2.5e-02 | 4.3e-02 | 1.6e-02 |
| all six ELEMENT rings | 3.6e-04 | 1.6e-04 | 2.5e-04 |
| everything | 2.5e-02 | 4.3e-02 | 1.6e-02 |

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
| save-all tape | 326 MB | — |
| fp32 ring | 266 MB | — |
| all bf16 | 140 MB | 4.7e-04 |
| all int8 | 89 MB | 4.3e-02 |
| nodes fp32, elements int8 | 123 MB | 3.6e-04 |
| **nodes bf16, elements int8** | **100 MB** | **5.6e-04** |

The last row is the one to use on this path: 12% more memory than all-int8
for 77× the accuracy, and 29% less than all-bf16 at the same accuracy. The
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

## Backend parity

"Same solver call, same arguments → same numbers" is the standing contract
between `impl='eager'` and `impl='c'` — forward and gradients, shared and
per-shot models, with and without boundary saving. Neither backend is
bit-reproducible on CUDA (both scatter with atomics), so the parity suites
(`test/sem/test_sem_backend_parity.py`, `test/sem/test_sem_shot_batch.py`)
measure the run-to-run noise floor in-session and hold the backend difference
to it, rather than to a fixed tolerance.
