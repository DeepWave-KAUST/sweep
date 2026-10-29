# Gradients, FWI and memory strategies

Contents: 1. How gradients work · 2. FWI loop · 3. Memory strategies · 4. Which gradients each
backend returns · 5. Finite-difference gradient checks (and the outer ring) · 6. Custom
gradients / imaging conditions · 7. Precision

## 1. How gradients work

- Any model tensor passed with `requires_grad=True` gets `.grad` from `loss.backward()`; the
  backward is the discrete adjoint of the forward (exact transpose on `impl='c'` for the
  second-order acoustic family, autograd on eager).
- Gradients have the model's shape (physical grid, CPML cropped) and are summed over the shots
  of the call. Accumulate over shot chunks by calling `backward()` per chunk.
- The wavelet can require grad too (source estimation) on eager and on `impl='c'` `Acoustic`;
  see section 4 for the exceptions.
- Raw gradient magnitudes are tiny (the kernel carries `dt^2/vp` factors); use Adam or normalise
  before a fixed-step update.
- Observed data: generate under `torch.no_grad()` so nothing is stored for a backward.

## 2. FWI loop

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
nz, nx, dh, dt, nt = 40, 64, 10.0, 1e-3, 400
vp_true = torch.full((nz, nx), 2000.0, device=device)
vp_true[18:26, 24:40] = 2300.0                               # an anomaly to recover
solver = PropTorch(Acoustic(spatial_order=4, device=device), shape=(nz, nx), dh=dh, dt=dt,
                   abcn=20, device=device, B=4)
w = ricker(np.arange(nt) * dt - 0.08, f=12.0).astype(np.float32)
src = np.array([[x, 1] for x in (8, 24, 40, 56)])
rec = np.array([[[ix, 1] for ix in range(nx)]] * len(src))
with torch.no_grad():
    obs = solver(w, src, rec, models=[vp_true])

vp = torch.full((nz, nx), 2000.0, device=device, requires_grad=True)
opt = torch.optim.Adam([vp], lr=10.0)                        # lr in m/s per step
losses = []
for it in range(3):
    opt.zero_grad()
    for s0 in range(0, len(src), 2):                          # shot chunks of 2
        pred = solver(w, src[s0:s0 + 2], rec[s0:s0 + 2], models=[vp])
        loss = 0.5 * (pred - obs[s0:s0 + 2]).pow(2).sum()
        loss.backward()                                       # accumulates into vp.grad
    opt.step()
    with torch.no_grad():
        vp.clamp_(1500.0, 3000.0)                             # keep the model physical / stable
    losses.append(float(loss))
print(losses)
assert all(np.isfinite(losses))
```

## 3. Memory strategies

The backward needs the forward wavefield. Choose how it is kept with
`memory=` (`from sweep.propagator.options import Full, BoundarySaving, Ckpt`):

| Strategy | Memory | Time | Default for | Notes |
|---|---|---|---|---|
| `Full()` | whole history (`nt` x grid), largest | fastest backward | `impl='c'` `ViscoAcoustic`, `DASZhao3D` | needed for eager snapshots and eager `register_gradient` |
| `BoundarySaving(storage="gpu")` | CPML-ring strips per step | ~1 extra forward | `impl='c'` (others) | reconstructs the wavefield backwards in time |
| `BoundarySaving(storage="cpu", pinned_memory=True)` / `"disk"` | ring in host RAM / files | slower | | `disk_dir=`, `disk_async_read=True` for disk |
| `BoundarySaving(storage_dtype="bf16"\|"fp16"\|"int8")` | 2x / ~2x / ~4x smaller ring | | | lossy: perturbs the gradient; validate before use |
| `BoundarySaving(tail_steps=K)` | last K steps only | shorter backward | | `impl='c'` `Acoustic`/`Acoustic3D` only, steady-state (frequency-selection) losses only |
| `Ckpt(mode="chunk", chunks=100)` | checkpoints every `chunks` steps | recomputes | eager / PropJax | `storage="cpu"` keeps checkpoints on the host (c) |
| `Ckpt(mode="recursive", count=n)` | fixed `n` checkpoints | most recompute | | `impl='c'` only |

Rules: one strategy per propagator; contradictory requests (`memory=BoundarySaving(),
use_ckpt=True`) raise. `use_ckpt=False` alone means `Full()`. Equations that cannot reverse
their step refuse boundary saving instead of returning a wrong gradient: `ViscoAcoustic`
(both impls), `ViscoElastic` (c), `DASZhao3D` (c), `Biot` (both), the pure-qP
`AcousticVTI/TTI` (eager; on c `int8` is refused). Boundary saving + `topography=` is refused on
`impl='c'` (at the first backward). `solver.memory_strategy` tells you what was chosen.

All strategies give the same gradient up to rounding (fp32 storage):

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import Full, BoundarySaving, Ckpt, EagerOptions
from sweep.signal import ricker

torch.backends.cudnn.allow_tf32 = False
device = "cuda" if torch.cuda.is_available() else "cpu"
backend = {"impl": "c"} if device == "cuda" else {
    "impl": "eager", "eager_options": EagerOptions(use_compile=False)}
shape, dt = (40, 48), 1e-3
w = ricker(np.arange(250) * dt - 0.06, f=12.0).astype(np.float32)
grads = {}
for name, mem in [("full", Full()), ("boundary", BoundarySaving(storage="gpu" if device == "cuda" else "cpu")),
                  ("ckpt", Ckpt(chunks=50))]:
    solver = PropTorch(Acoustic(device=device), shape=shape, dh=10.0, dt=dt, abcn=15,
                       device=device, memory=mem, **backend)
    vp = torch.full(shape, 2000.0, device=device, requires_grad=True)
    solver(w, np.array([[10, 5]]), np.array([[[38, 5]]]), models=[vp]).pow(2).sum().backward()
    grads[solver.memory_strategy] = vp.grad.flatten().double()
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
print({k: round(cos(grads["full"], g), 6) for k, g in grads.items()})
assert all(cos(grads["full"], g) > 0.999 for g in grads.values())
```

## 4. Which gradients each backend returns

| Case | Behaviour |
|---|---|
| eager, any equation | autograd: every model and the wavelet |
| `impl='c'` `Acoustic`, `Elastic`, ... | models and wavelet |
| `impl='c'` `AcousticBorn*`, `AcousticVRZ*` | wavelet gradient **refused** (`NotImplementedError`); detach the wavelet |
| `impl='c'` `AcousticBorn*` with `Ckpt` | `vp` (RWI) gradient refused; `mp` only. Use `Full()`/`BoundarySaving()` |
| `impl='c'` qP Born pair | `mp` gradient only |
| `impl='c'` `ElasticBorn` | `dvp, dvs, drho` gradients only; background and wavelet refused |
| `ModelParallel` (any) | models only; a wavelet requiring grad is refused |
| `impl='c'`, second `backward()` on the same graph | `RuntimeError`; re-run the forward (eager keeps the graph) |

## 5. Finite-difference gradient checks (and the outer ring)

Do it right or the check proves nothing:

- Disable TF32 for eager on Ampere+ (`torch.backends.cudnn.allow_tf32 = False`).
- Sum the loss in float64; the forward is float32, so use perturbations of a few m/s and scan
  the step: agreement should be stable across steps (~1e-3 relative is the fp32 floor here).
- Perturb a smooth bump **inside** the model. The outermost ring of cells gets a cropped (not
  exact) gradient: the model is edge-replicated into the CPML and the backward only crops, so
  the outer row/column's sensitivity to its CPML copies is dropped (it can even have the wrong
  sign, as below). Rows/columns 1 and further in are exact.
- With `abs`-tolerance helpers (`pytest.approx`, `np.allclose`) set `atol=0`: gradients are
  ~1e-3 or smaller and a default absolute tolerance passes anything.

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import Full, EagerOptions
from sweep.signal import ricker

torch.backends.cudnn.allow_tf32 = False
device = "cuda" if torch.cuda.is_available() else "cpu"
backend = {"impl": "c"} if device == "cuda" else {
    "impl": "eager", "eager_options": EagerOptions(use_compile=False)}
nz, nx, dt, nt = 40, 48, 1e-3, 300
solver = PropTorch(Acoustic(device=device), shape=(nz, nx), dh=10.0, dt=dt, abcn=15,
                   device=device, memory=Full(), **backend)
w = ricker(np.arange(nt) * dt - 0.06, f=12.0).astype(np.float32)
src, rec = np.array([[10, 5]]), np.array([[[ix, 5] for ix in range(4, 44, 2)]])
vp0 = torch.full((nz, nx), 2000.0, device=device)
vp0[20:] = 2300.0
with torch.no_grad():
    obs = solver(w, src, rec, models=[vp0 + 100.0])

def loss_of(vp):
    return 0.5 * (solver(w, src, rec, models=[vp]).double() - obs.double()).pow(2).sum()

vp = vp0.clone().requires_grad_(True)
loss_of(vp).backward()

def check(dm, eps):
    adj = float((vp.grad * dm).sum())
    with torch.no_grad():
        fd = (float(loss_of(vp0 + eps * dm)) - float(loss_of(vp0 - eps * dm))) / (2 * eps)
    return adj, fd, abs(fd - adj) / abs(fd)

z, x = torch.meshgrid(torch.arange(nz, device=device), torch.arange(nx, device=device), indexing="ij")
bump = torch.exp(-((z - 25.0) ** 2 + (x - 24.0) ** 2) / (2 * 4.0 ** 2))
rels = [check(bump, eps)[2] for eps in (20.0, 10.0)]
print("interior bump rel. error:", rels)                  # ~1e-3
assert min(rels) < 1e-2

for cell in [(0, 24), (1, 24)]:                            # top row vs one row in
    dm = torch.zeros_like(vp0)
    dm[cell] = 1.0
    adj, fd, rel = check(dm, 20.0)
    print(cell, f"adjoint={adj:.3e} fd={fd:.3e} rel={rel:.1e}")   # (0, 24) disagrees
```

## 6. Custom gradients / imaging conditions (eager only)

`solver.register_gradient(name, fn, mode="gradient"|"imaging", imaging_type=None)` replaces the
autograd gradient of model `name` by `sum_t fn(forward, adjoint, models, dt, dh)` (dicts of
wavefields keyed by field name, on the padded grid). `mode="imaging"` leaves `.grad` alone and
is read with `solver.imaging(wavelet, sources, receivers, models=..., observed=obs)` (returns
`{name: image}`). `solver.clear_gradients()` removes registrations.

- **Requires `memory=Full()` (or `use_ckpt=False`)**: under checkpointing (the eager default)
  a `mode="gradient"` registration is **silently ignored** and the ordinary autograd gradient is
  returned (measured: a registered zero gradient came back non-zero with `Ckpt`, zero with
  `Full()`).
- Not available on `impl='c'` (`AttributeError`).

## 7. Precision

- Everything runs in float32; `impl='c'` refuses float64 models. Accumulate scalar misfits over
  many shots in float64 when comparing runs.
- Eager on Ampere+ GPUs: TF32 convolutions move 3-D gradients by 1e-2..1e-1 relative unless
  `torch.backends.cudnn.allow_tf32 = False` (sweep warns once).
- Boundary-saving `storage_dtype` other than `"fp32"` changes the gradient (int8 most).
- The two backends agree to ~1e-6..1e-4 relative, not bit for bit.
