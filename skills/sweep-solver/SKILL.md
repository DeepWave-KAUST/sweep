---
name: sweep-solver
description: Writes and debugs Python code that uses the sweep-solver package (import name `sweep`), a differentiable seismic wave-equation engine on PyTorch/JAX. Covers forward modelling of shot gathers, FWI gradients through autograd, RTM / LSRTM (Born) imaging, choosing the equation (acoustic, elastic, VTI/TTI, visco, DAS, Biot), the backend (impl='c' compiled CUDA vs impl='eager' vs PropJax), boundaries (CPML, free surface, topography), gradient-memory options (boundary saving, checkpointing), the spectral-element PropSEM, multi-GPU shot and domain decomposition, the `sweep` CLI and `sweep.datasets`. Use it whenever code imports `sweep`, mentions PropTorch / PropSEM / sweepx / sweep-solver, or asks for finite-difference seismic modelling or inversion with this library, including fixing shape errors, NaNs, wrong gradients or slow runs.
---

# sweep-solver

**Mental model.** An *equation* object (`sweep.equations.Acoustic`, `Elastic`, ...) holds the
physics and stencils. A *propagator* (`PropTorch`, or `PropJax` / `PropSEM`) wraps it with the
grid (`shape`, `dh`, `dt`), absorbing boundary (`abcn` CPML cells added *outside* `shape`), free
surface and gradient-memory strategy. Calling the propagator with
`(wavelet, sources, receivers, models=[...])` time-steps the wave equation and returns the
recorded traces `(nshots, nt, nrec, ncomp)`. Model tensors with `requires_grad=True` get exact
adjoint-state gradients from `loss.backward()`, so FWI/LSRTM are ordinary PyTorch optimisation
loops.

## Install

<!-- skill-test: skip -->
```bash
# 1. torch matching the NVIDIA driver FIRST (PyPI's default torch is CUDA 13 and needs driver >= 580)
pip install torch --index-url https://download.pytorch.org/whl/cu126   # driver >= 525; also V100
#   driver >= 580: plain `pip install torch` (cu13). cu128 also works on driver >= 525 but has no V100 kernels.
# 2. the solver (import name: sweep)
pip install sweep-solver          # solver only, Python >= 3.9
# or: pip install sweepx          # solver + sweepx toolkit, Python >= 3.10
# optional extras: "sweep-solver[jax]" (PropJax), "sweep-solver[datasets]" (downloads)
```

- The wheel ships prebuilt CUDA cores for CUDA 12 and 13 and picks the one matching torch's
  CUDA major: `impl='c'` needs no nvcc and no build step. A git clone has no core and builds one
  with nvcc on first use (`python -m sweep.build`).
- No GPU: everything runs with `impl='eager'` (pure PyTorch, CPU or GPU).
- Check an environment: `python scripts/check_env.py`.

## Minimal forward + gradient

```python
import numpy as np
import torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
nz, nx = 40, 64                       # model shape: depth first
dh, dt, nt = 10.0, 1e-3, 300          # metres, seconds, steps

vp_true = torch.full((nz, nx), 2000.0)
vp_true[20:] = 2500.0                 # reflector at row 20
vp_true = vp_true.to(device)

eq = Acoustic(spatial_order=4, device=device)          # same device as the propagator
solver = PropTorch(eq, shape=(nz, nx), dh=dh, dt=dt, abcn=20, device=device)
# impl=None (default): 'c' (compiled CUDA) on a GPU, 'eager' (pure PyTorch) otherwise

t = np.arange(nt) * dt
wavelet = ricker(t - 0.07, f=15.0).astype(np.float32)       # (nt,), shared by all shots
sources = np.array([[16, 2], [48, 2]])                      # (nshots, 2): (x, z) grid indices
receivers = np.array([[[ix, 2] for ix in range(nx)]] * 2)   # (nshots, nrec, 2)

with torch.no_grad():                                       # observed data: no gradient
    obs = solver(wavelet, sources, receivers, models=[vp_true])

vp = torch.full((nz, nx), 2000.0, device=device, requires_grad=True)
pred = solver(wavelet, sources, receivers, models=[vp])     # (nshots, nt, nrec, 1)
loss = 0.5 * (pred - obs).pow(2).sum()
loss.backward()                                             # vp.grad: (nz, nx), physical grid

print(solver.impl, solver.memory_strategy, tuple(pred.shape), tuple(vp.grad.shape))
assert pred.shape == (2, nt, nx, 1) and vp.grad.shape == (nz, nx)
assert torch.isfinite(vp.grad).all() and vp.grad.abs().sum() > 0
```

Swapping physics is a one-line change (`Elastic(...)` + `models=[vp, vs, rho]`); everything
else stays.

## Choose

**Equation by physics** (2-D unless noted; full table with fields and flags in
`references/equations.md`):

| Physics | Class | `models=[...]` order | `impl='c'` | Free surface |
|---|---|---|---|---|
| Constant-density acoustic | `Acoustic` / `Acoustic3D` | `vp` | yes | yes (2-D per-edge) |
| Variable density acoustic | `AcousticVRZ` / `AcousticVRZ3D` (`z` = impedance) | `vp, z` | yes | top |
| | `Acoustic1st` (velocity-pressure; default `backend='jax'`!) | `vp, rho` | no | **not modelled correctly**, see Pitfalls |
| Born / LSRTM | `AcousticBorn` / `AcousticBorn3D` | `vp, mp` | yes | top |
| Elastic Born / LSRTM | `ElasticBorn` (2-D) | `vp, vs, rho, dvp, dvs, drho` | yes | yes (2-D per-edge) |
| Visco-acoustic (const. Q) | `ViscoAcoustic` | `vp, Q, omega` | yes | yes (2-D per-edge) |
| Isotropic elastic | `Elastic` / `Elastic3D` | `vp, vs, rho` | yes | yes (2-D per-edge) |
| Visco-elastic | `ViscoElastic` | `vp, vs, rho, Qp, Qs` | yes | yes |
| VTI / TTI acoustic (pure qP) | `AcousticVTI` / `AcousticTTI` | `vp, epsilon, delta[, theta]` | yes | **refused** |
| VTI acoustic 1st order | `AcousticVTI1st` / `AcousticVTI1st3D` | `vp, epsilon, delta, rho` | yes | refused |
| Elastic TTI | `ElasticTTISG` / `ElasticTTISG3D` | `vp0, vs0, rho, epsilon, delta, gamma, theta, phi` | yes | refused |
| DAS (strain / strain rate) | `DASZhao`, `DASMu` (+`3D`) | `vp, vs, rho` | yes | top |
| Poroelastic (Biot) | `Biot` | `H, mu, C, M, rho, rho_f, m, b` | no | top |
| Irregular topography | `AcousticCurvilinear`, `ElasticCurvilinear`, or `topography=` (staircase), or `sweep.sem` | as base | see `references/boundaries.md` | built in |

**Backend:**

| Need | Use |
|---|---|
| Speed on an NVIDIA GPU, large models, boundary saving to CPU/disk | `PropTorch(..., impl="c")` (CUDA tensors only, FP32, no CPU path) |
| CPU, no GPU, debugging, wavefield snapshots, custom imaging conditions, double backward, any equation | `PropTorch(..., impl="eager")` |
| JAX transforms (`jax.grad`, `pmap`) | `PropJax` with `Equation(..., backend="jax")` |
| Curved-mesh topography, spectral elements | `sweep.sem.PropSEM` |
| Let sweep decide | `impl=None` (default): `'c'` if the GPU, core and equation allow, else `'eager'` |

**Boundaries:** leave `pml_type` unset (each equation knows its CPML). `abcn` (default 50) cells
of CPML are added on each absorbing face. `free_surface=True` / `"top"` / `["top","left"]` /
per-face tuple, fixed at construction. See `references/boundaries.md`.

**Gradient memory** (`memory=` from `sweep.propagator.options`): `Full()` (store everything,
fastest backward), `BoundarySaving(storage="gpu"|"cpu"|"disk", storage_dtype=...)` (default on
`impl='c'`), `Ckpt(chunks=...)` (default on eager). See `references/gradients.md`.

## Conventions

| Item | Convention |
|---|---|
| Model arrays | `(nz, nx)` / `(nz, ny, nx)`, depth first, **physical region only** (CPML padded outside internally). Pass a list in `eq.models` order |
| Source / receiver coordinates | integer **grid indices** in `(x, z)` / `(x, y, z)` order (depth **last**, opposite to arrays). Floats are truncated toward zero |
| `sources` | `(nshots, ndim)` one point source per shot; or `(1, nsrc, ndim)` = one encoded super-shot |
| `receivers` | always 3-D `(nshots, nrec, ndim)`; repeat a fixed spread per shot |
| `wavelet` | `(nt,)` shared, `(nshots, nt)` per shot, or `(nsrc, nt)` when encoded; `nt` = its length |
| Output | `(nshots, nt, nrec, len(receiver_type))`, last axis in `receiver_type` order |
| `dh` | scalar, or per axis in **array order** `(dz, dx)` / `(dz, dy, dx)`; metres |
| `dt` | seconds; must satisfy CFL (`scripts/stability_check.py`). Defaults `dh=10`, `dt=0.002` apply silently if omitted |
| Units | velocities m/s, density kg/m^3, angles (`theta`, `phi`) **radians**, `omega` rad/s |
| Default fields | `eq.default_source_fields` / `default_receiver_fields`; override with `source_type=` / `receiver_type=` |
| Gradients | same shape as the model (physical grid), summed over the shots of a call (per-shot models `(nshots, nz, nx)` get per-shot gradients) |

## Pitfalls

1. **Free surface is construction-only.** `PropTorch(..., free_surface=True)`. Assigning
   `solver.free_surface = True` later raises `AttributeError` (also `abcn`, `pml_type`,
   `topography`). Older versions silently ignored it on `impl='c'`.
2. **Don't pass `pml_type`.** Each equation has one CPML: `cpmlr` for the second-order acoustic
   family (and `ElasticTTI`), `cpmls` for the staggered family (`Elastic*`, `DAS*`,
   `ElasticTTISG*`, `AcousticVTI1st*`, `Biot`, `ViscoElastic`, `Acoustic1st`). A wrong value
   raises `ValueError`.
3. **Equation and propagator on the same device.** `Acoustic()` defaults to `device='cpu'`; with
   `PropTorch(..., device="cuda", impl="eager")` the first call fails with `Input type
   (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`. Build the
   equation with `device=device`.
4. **Coordinates are `(x, z)` but `shape`/`dh`/models are `(z, x)`.** Swapping them gives a
   valid-looking but wrong survey. Out-of-range indices are **not checked**: a source at
   `x >= nx` or `z < 0` silently lands in the CPML. Convert metres with `round(x / dh)`.
5. **`impl='c'` ignores `return_wavefield=True`** (returns the record only). Snapshots need
   `impl='eager'` with `memory=Full()` (or `use_ckpt=False`); eager's default checkpointing raises.
6. **`impl='c'` needs CUDA float32 model tensors** (numpy arrays, CPU tensors and float64 are
   refused); eager converts. Gradients come back on the model's device.
7. **Batch growth bug on `impl='c'`:** after a gradient call with `nshots=B`, a `no_grad` call
   with more shots fails (`start (0) + length ... exceeds dimension size`). Pass
   `B=<max shots per call>` to `PropTorch` or keep the batch size fixed.
8. **No second backward on `impl='c'`** (`retain_graph=True` then `backward()` again raises):
   run the forward again; for Hessian-vector products use eager with `memory=Full()`.
9. **TF32 on Ampere+ with `impl='eager'`:** cuDNN TF32 moves eager gradients by 1e-2..1e-1
   (3-D). Set `torch.backends.cudnn.allow_tf32 = False` (and `torch.backends.cuda.matmul.allow_tf32
   = False`) before comparing eager with `c` or with finite differences.
10. **Outermost model cells get a cropped, not exact, gradient.** The model is edge-replicated
    into the CPML and the backward just crops. Keep finite-difference test perturbations
    (and anything you trust quantitatively) away from the outer ring; see `references/gradients.md`.
11. **Free-surface plane position:** `Acoustic` puts the pressure-release plane ~0.9 cell
    *above* row 0; `Elastic` at ~row 0. A receiver at row 0 of `Acoustic` is not at zero pressure.
12. **`Acoustic1st` has no free-surface implementation.** `free_surface=True` only removes the top
    CPML; the zero halo then reflects at about `spatial_order/2 + 1` cells above row 0
    (measured). Use `Acoustic` / `Elastic` for free-surface work. Also: its default
    `backend` is `'jax'`; pass `backend='torch'` for PropTorch.
13. **Anisotropic equations refuse any free surface** (`NotImplementedError`), including the one
    implied by `topography=`.
14. **Wavelet sign vs deepwave:** for the same wavelet sweep's `Acoustic` record is the
    negative of `deepwave.scalar`'s (correlation -0.997 measured) and amplitudes are scaled
    differently. Negate the wavelet when porting, and calibrate amplitudes.
15. **Stability is not checked** except for `AcousticVTI/TTI` (warning). A too-large `dt` gives
    NaN/inf silently. Run `scripts/stability_check.py`; elastic: use `max(vp)` for stability and
    keep >= ~5 points per minimum S wavelength (rule of thumb).
16. **`eager_options=` with `impl=None` on a GPU raises** (auto resolves to `'c'`). Pass
    `impl="eager"` explicitly when you pass eager options; `cuda_options` likewise needs `impl="c"`.
17. **Unknown keyword arguments raise `TypeError`** (constructor and call). `dev=` is a deprecated
    alias of `device=`; `boundary_saving_config={...}` / `MemoryOptions` are deprecated: use
    `memory=`.
18. **Wavelet / background gradients that `c` does not compute are refused:**
    `AcousticBorn*` and `AcousticVRZ*` on `impl='c'` raise if the wavelet requires grad; detach it
    or use eager.
19. **`ElasticTTI2nd` (second-order displacement) with plain CPML grows at late times**; keep
    the default `mpml_ratio=0.05` (multiaxial PML), do not set it to 0.
20. **`theta`/`phi` are radians.** Degrees are accepted silently and model a different tilt.
21. **`register_gradient(..., mode="gradient")` needs `memory=Full()`** (eager only). Under the
    eager default checkpointing the registration is silently ignored and plain autograd
    gradients come back.
22. **`ModelParallel` wants 3-D `sources` `(1, nsrc, ndim)`**, `impl='c'` and boundary saving;
    see `references/parallel.md`.

## Where to look

| Task | Open |
|---|---|
| Pick an equation, its models, fields, source/receiver names, FS support | `references/equations.md` |
| Constructor/call options, geometry modes, batching, snapshots, options blocks | `references/propagators.md` |
| CPML width, free surface faces, topography, buffer, surface position | `references/boundaries.md` |
| FWI loop, memory strategies, FD gradient checks, custom gradients, illumination | `references/gradients.md` |
| RTM, LSRTM / Born (acoustic and elastic), RWI, ADCIG, pre-0.3.6 names | `references/imaging.md` |
| Spectral elements, curved topography (`PropSEM`) | `references/sem.md` |
| Multi-GPU: shot-parallel DDP, `ModelParallel` domain decomposition, JAX | `references/parallel.md` |
| `sweep list/show`, `sweep datasets`, `sweep.datasets.load` | `references/cli-datasets.md` |
| Exact signatures and the full capability table (generated) | `references/api-signatures.md` |
| Environment / GPU / core check | `scripts/check_env.py` |
| CFL and dispersion for a planned grid | `scripts/stability_check.py` |
| Regenerate signatures after an upgrade | `scripts/gen_api_signatures.py` |

Authoritative live listing on any install: `sweep list equations`, `sweep show <Equation>`.
Docs: https://sweepx.deepwave.group/solver/
