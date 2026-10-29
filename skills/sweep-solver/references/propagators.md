# Propagators

Contents: 1. `PropTorch` constructor · 2. `backend` vs `impl` · 3. Call and geometry modes ·
4. Batching and per-shot models · 5. Wavefield snapshots · 6. Option blocks · 7. Attributes
worth reading · 8. `PropJax`

Exact signatures: `api-signatures.md`. `PropSEM` (spectral elements) is in `sem.md`.

## 1. `PropTorch` constructor

`PropTorch(equation, shape, *, ...)` from `sweep.propagator.torch`. Everything is a keyword
except `equation` and `shape`; unknown keywords raise `TypeError`.

| Keyword | Default | Meaning |
|---|---|---|
| `shape` | required | physical model shape `(nz, nx)` / `(nz, ny, nx)` |
| `dh` | `10.0` | m; scalar or per axis `(dz, dx)` / `(dz, dy, dx)` |
| `dt` | `0.002` | s |
| `device` | equation's device | `"cuda"`, `"cuda:1"`, `"cpu"`; `dev=` is a deprecated alias |
| `impl` | `None` (auto) | `"c"`, `"eager"`; see section 2 |
| `source_type`, `receiver_type` | equation defaults | lists of field names, e.g. `["vz"]` |
| `abcn` | `50` | CPML cells added outside `shape` on each absorbing face (scalar; per-face tuple eager-only) |
| `free_surface` | `False` | `True`/`"top"`, face list, dict, or per-face tuple; see `boundaries.md` |
| `topography` | `None` | per-column surface row (int) for the staircase image method; implies a free surface |
| `memory` | backend default | `Full()`, `BoundarySaving(...)`, `Ckpt(...)`; see `gradients.md` |
| `use_ckpt` | `None` | legacy switch: `False` = full storage, `True` = checkpointing |
| `B` | `1` | initial batch capacity of `impl='c'` buffers (see section 4) |
| `allow_growth` | `True` | let `impl='c'` buffers grow with the batch |
| `nt` | `-1` | taken from the wavelet length |
| `boundary_buffer` | `None` | sigma=0 cells between model and CPML; equation picks what it needs |
| `pml_type` | `None` | leave unset |
| `eager_options` / `cuda_options` | `None` | only with the matching explicit `impl` |
| `backend` | equation's | `"torch"` (use `PropJax` for JAX) |

## 2. `backend` vs `impl`

| `impl` | Runs | Device | Notes |
|---|---|---|---|
| `None` | `'c'` if a CUDA core fits, the device is CUDA and the equation has kernels; else `'eager'` | any | read `solver.impl` to see what you got |
| `"c"` | prebuilt CUDA core via ctypes | CUDA only, FP32 | falls back to eager with a `UserWarning` if impossible; default memory: boundary saving |
| `"eager"` | PyTorch ops + autograd (`torch.compile`d step by default) | CPU or CUDA | every equation; default memory: chunked checkpointing |

The first eager call compiles the time step (seconds to minutes, cached on disk per grid size).
For tiny or one-off runs pass `impl="eager", eager_options=EagerOptions(use_compile=False)`.

## 3. Call and geometry modes

`solver(wavelet, sources, receivers, models=[...])` (same as `.forward`). The mode is decided by
the rank of `sources`:

| Mode | `sources` | `wavelet` | `receivers` | Batch axis |
|---|---|---|---|---|
| A1 | `(nshots, ndim)` | `(nt,)` shared | `(nshots, nrec, ndim)` | shots |
| A2 | `(nshots, ndim)` | `(nshots, nt)` per shot | `(nshots, nrec, ndim)` | shots |
| B (encoded) | `(1, nsrc, ndim)` | `(nt,)` or `(nsrc, nt)` | `(1, nrec, ndim)` | 1 super-shot, all sources fire together |

Every shot needs the same `nrec` (pad or split otherwise). Coordinates are grid indices,
horizontal first: `(x, z)` / `(x, y, z)`.

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
shape, dt, nt = (40, 48), 1e-3, 200
solver = PropTorch(Acoustic(device=device), shape=shape, dh=10.0, dt=dt, abcn=15,
                   device=device, B=3)
vp = torch.full(shape, 2000.0, device=device)
w = ricker(np.arange(nt) * dt - 0.05, f=15.0).astype(np.float32)
rec1 = [[ix, 3] for ix in range(0, 48, 4)]                      # 12 receivers, (x, z)

with torch.no_grad():
    a1 = solver(w, np.array([[10, 3], [24, 3], [38, 3]]), np.array([rec1] * 3), models=[vp])
    w3 = np.stack([w, 0.5 * w, -w])                             # A2: one wavelet per shot
    a2 = solver(w3, np.array([[10, 3], [24, 3], [38, 3]]), np.array([rec1] * 3), models=[vp])
    b = solver(w, np.array([[[10, 3], [24, 3], [38, 3]]]), np.array([rec1]), models=[vp])

print(tuple(a1.shape), tuple(a2.shape), tuple(b.shape))  # (3,200,12,1) (3,200,12,1) (1,200,12,1)
# the encoded super-shot equals the sum of the three single shots (linearity)
assert torch.allclose(b[0], a1.sum(0), atol=1e-4 * a1.abs().max().item())
```

## 4. Batching and per-shot models

- One call simulates all shots of the batch in parallel; gradients are summed over shots.
  Memory scales with the batch, so loop over shot chunks and accumulate `.grad` (call
  `backward()` per chunk).
- `impl='c'` preallocates buffers for `B` shots and grows them on demand, **except** that a
  `no_grad` call with more shots than the preceding gradient call fails
  (`start (0) + length (n) exceeds dimension size (m)`). Construct with `B=` your largest batch.
- Per-shot models: pass each model as `(nshots, nz, nx)`. On `impl='c'` only equations with
  `supports_batched_models` (`Acoustic`, `Elastic`, `ViscoAcoustic`) accept it; others raise.

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
shape, dt = (32, 40), 1e-3
solver = PropTorch(Acoustic(device=device), shape=shape, dh=10.0, dt=dt, abcn=10,
                   device=device, B=2)
vp2 = torch.full((2, *shape), 2000.0, device=device)
vp2[1] += 500.0                                       # shot 1 sees a faster medium
w = ricker(np.arange(300) * dt - 0.05, f=15.0).astype(np.float32)
with torch.no_grad():
    d = solver(w, np.array([[5, 5], [5, 5]]), np.array([[[30, 5]]] * 2), models=[vp2])
assert d.shape == (2, 300, 1, 1)
assert d[1].abs().argmax() < d[0].abs().argmax()      # earlier arrival in the faster model
```

## 5. Wavefield snapshots

Only `impl='eager'`, and not with checkpointing (`Ckpt`, the eager default, raises
`ValueError`); `impl='c'` silently returns the record only. Use `memory=Full()`.
Snapshots are `(nsnap, nfields, nshots, 1, *padded_shape)`, fields in `eq.wavefields` order
(index 0 is the main field, e.g. `h1` or `vx`), detached. `solver.crop(...)` removes the CPML pad.

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import Full, EagerOptions
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
shape, dt = (40, 48), 1e-3
eq = Acoustic(device=device)
solver = PropTorch(eq, shape=shape, dh=10.0, dt=dt, abcn=15, device=device,
                   impl="eager", memory=Full(), eager_options=EagerOptions(use_compile=False))
w = ricker(np.arange(200) * dt - 0.05, f=15.0).astype(np.float32)
rec, snaps = solver(w, np.array([[24, 10]]), np.array([[[30, 10]]]),
                    models=[torch.full(shape, 2000.0, device=device)],
                    return_wavefield=True, snapshot_times=[100, 150])   # or snapshot_interval=50
print(eq.wavefields[0], tuple(snaps.shape))           # h1 (2, 6, 1, 1, 70, 78)
p = solver.crop(snaps)[:, 0, 0, 0]                    # (2, 40, 48) on the physical grid
assert p.shape == (2, *shape)
```

## 6. Option blocks (`sweep.propagator.options`)

| Block | Pass as | Fields |
|---|---|---|
| `Full()` | `memory=` | none |
| `BoundarySaving(...)` | `memory=` | `storage` `"gpu"`/`"cpu"`/`"disk"`, `storage_dtype` `"fp32"`/`"fp16"`/`"bf16"`/`"int8"`, `transfer_interval`, `pinned_memory`, `ring_buffers`, `disk_dir`, `disk_async_read`, `tail_steps` (c, Acoustic only) |
| `Ckpt(...)` | `memory=` | `mode` `"chunk"`/`"recursive"` (recursive: c only), `chunks` (steps per chunk, 100), `count`, `storage` `"gpu"`/`"cpu"` |
| `EagerOptions(...)` | `eager_options=` with `impl="eager"` | `use_compile` (True), `compile_mode`, `compile_dynamic`, `compile_backend`, `compile_fullgraph`, `store_last_wavefield`, `cpml_strips` |
| `CUDAOptions(memory=...)` | `cuda_options=` with `impl="c"` | only a memory strategy; prefer `memory=` |

Invalid combinations raise at construction (e.g. `disk_async_read=True` with `storage="cpu"`,
`tail_steps` on eager, recursive checkpointing on eager).

## 7. Attributes worth reading

| Attribute | Meaning |
|---|---|
| `solver.impl` | `'c'` or `'eager'` actually used |
| `solver.memory_strategy` | `'full'`, `'boundary'` or `'ckpt'` actually used |
| `solver.pad` | CPML pad per face, axis-major `(z_lo, z_hi, x_lo, x_hi)`; 0 on free faces |
| `solver.fs_faces` | free-surface flag per face, same order |
| `solver.wavefield_names`, `solver.model_names` | from the equation |
| `solver.source_illumination`, `receiver_illumination` | after a backward with `compute_illumination=True` (see `imaging.md`) |

## 8. `PropJax`

Same constructor and call as `PropTorch` (positional `equation, shape`, keyword rest) plus
`memory=`, `scan_unroll=`. Needs `pip install "sweep-solver[jax]"` and an equation built with
`backend="jax"`; arrays are `jnp` arrays and gradients come from `jax.grad`.

<!-- skill-test: skip -->
```python
import jax, jax.numpy as jnp, numpy as np
from sweep.equations import Acoustic
from sweep.propagator.jax import PropJax
from sweep.signal import ricker

solver = PropJax(Acoustic(spatial_order=4, backend="jax"), shape=(40, 48), dh=10.0, dt=1e-3,
                 abcn=15, source_type=["h1"], receiver_type=["h1"])
w = jnp.asarray(ricker(np.arange(200) * 1e-3 - 0.05, f=15.0), dtype=jnp.float32)
src, rec = np.array([[10, 3]]), np.array([[[30, 3]]])
obs = solver(w, src, rec, models=[jnp.full((40, 48), 2100.0)])
loss = lambda vp: jnp.sum((solver(w, src, rec, models=[vp]) - obs) ** 2)
g = jax.grad(loss)(jnp.full((40, 48), 2000.0))
```
