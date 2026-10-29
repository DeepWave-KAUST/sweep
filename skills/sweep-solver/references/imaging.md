# Imaging: RTM, LSRTM / Born, RWI, ADCIG

Contents: 1. RTM as a gradient · 2. Born modelling and LSRTM (`AcousticBorn`) · 3. Reflection
(RWI) velocity gradient · 4. Anisotropic Born pair · 5. Other imaging tools · 6. Elastic Born
(`ElasticBorn`) · 7. Names before 0.3.6

There is no `solver.rtm()` (removed in 0.3.0; old scripts get `AttributeError`). Images are
gradients of an inner-product or least-squares loss.

## 1. RTM as a gradient

For `Acoustic`, the vp gradient of `sum(pred * d)` is `sum_t (2 dt^2 / vp) * u_tt * lambda`, with
`lambda` the adjoint field driven by the data `d`. Driving the forward with the **twice
time-integrated** wavelet turns `u_tt` into the field of the original wavelet, so
`vp.grad * vp / (2 dt^2)` is the zero-lag cross-correlation image. On `impl='c'`,
`solver.compute_illumination = True` also accumulates the source illumination for compensation
(not available on eager).

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
nz, nx, dh, dt, nt = 40, 64, 10.0, 1e-3, 400
vp_true = torch.full((nz, nx), 2000.0, device=device)
vp_true[25:] = 2400.0                                       # reflector at row 25
vp_mig = torch.full((nz, nx), 2000.0, device=device)        # smooth migration model
solver = PropTorch(Acoustic(device=device), shape=(nz, nx), dh=dh, dt=dt, abcn=20,
                   device=device, B=3)
t = np.arange(nt) * dt
w = ricker(t - 0.08, f=12.0).astype(np.float32)
w_int2 = (np.cumsum(np.cumsum(w)) * dt * dt).astype(np.float32)   # migration source
src = np.array([[x, 2] for x in (12, 32, 52)])
rec = np.array([[[ix, 2] for ix in range(nx)]] * 3)
with torch.no_grad():                                       # reflection data: remove direct wave
    d = solver(w, src, rec, models=[vp_true]) - solver(w, src, rec, models=[vp_mig])

if solver.impl == "c":
    solver.compute_illumination = True
vp = vp_mig.clone().requires_grad_(True)
(solver(w_int2, src, rec, models=[vp]) * d).sum().backward()      # adjoint source = d
image = (vp.grad * vp_mig / (2 * dt * dt)).detach()
if solver.impl == "c":
    illum = solver.source_illumination.detach()             # (nz, nx), summed over shots
    image = image / (illum + 1e-3 * illum.max())
    solver.compute_illumination = False
row = int(image[5:].abs().sum(1).argmax()) + 5
print("strongest image row:", row)                          # near the reflector (row 25)
assert abs(row - 25) <= 4
```

Production RTM: mute the direct wave / near offsets in `d`, loop shot chunks accumulating
`vp.grad` (and `source_illumination`), and use boundary saving (`impl='c'` default).

## 2. Born modelling and LSRTM (`AcousticBorn`, `AcousticBorn3D`)

Two coupled fields: background `h1` in `vp`, scattered `sh1` driven by `mp * vp^2 * lap(h1)`
(so `mp` is the squared-slowness/velocity perturbation ratio, `mp = 2 dv / v` to first
order). Models `[vp, mp]`; sources on `h1`, receivers on `sh1` by default, so the output is the
**Born (scattered) data only** (zero when `mp = 0`). It is linear in `mp`: `-mp.grad` of the
misfit is the migrated image and LSRTM is plain gradient descent on `mp`.

```python
import numpy as np, torch
from sweep.equations import AcousticBorn
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
nz, nx, dt, nt = 40, 64, 1e-3, 400
solver = PropTorch(AcousticBorn(device=device), shape=(nz, nx), dh=10.0, dt=dt, abcn=20,
                   device=device, B=2)
print(solver.source_type, solver.receiver_type)            # ['h1'] ['sh1']
vp = torch.full((nz, nx), 2000.0, device=device)           # background, fixed
mp_true = torch.zeros((nz, nx), device=device)
mp_true[25] = 0.1
w = ricker(np.arange(nt) * dt - 0.08, f=12.0).astype(np.float32)
src = np.array([[20, 2], [44, 2]])
rec = np.array([[[ix, 2] for ix in range(nx)]] * 2)
with torch.no_grad():
    d_obs = solver(w, src, rec, models=[vp, mp_true])        # Born data

mp = torch.zeros((nz, nx), device=device, requires_grad=True)
opt = torch.optim.Adam([mp], lr=0.01)
for it in range(3):                                          # LSRTM iterations
    opt.zero_grad()
    loss = 0.5 * (solver(w, src, rec, models=[vp, mp]) - d_obs).pow(2).sum()
    loss.backward()
    opt.step()
print("image peak row:", int(mp.detach().abs().sum(1).argmax()))   # 25
assert int(mp.detach().abs().sum(1).argmax()) == 25
```

- Keep `vp` without `requires_grad` for LSRTM: a `vp` that requires grad triggers the RWI
  velocity gradient (section 3), which roughly doubles backward time and memory on `impl='c'`.
- `impl='c'` refuses a wavelet that requires grad; detach it.
- Free surface: top only.

## 3. Reflection (RWI) velocity gradient

With `vp.requires_grad_(True)`, `AcousticBorn` on `impl='c'` returns the Wu & Alkhalifah (2015)
reflection tomographic gradient (terms II+III+IV) in `vp.grad`, with `memory=Full()` or
`BoundarySaving()` (checkpointing refuses it). To weight the image-point term III separately:

<!-- skill-test: skip -->
```python
import os
from sweep.propagator import last_grad_split_iii
os.environ["SWEEP_LSRTM_SPLIT_III"] = "1"     # read by every impl='c' LSRTM backward
loss.backward()
grad_vp = vp.grad + beta * last_grad_split_iii()   # vp.grad holds II+IV, III kept aside
```

The transmission term I comes from plain `Acoustic` FWI.

## 4. Anisotropic Born pair

`AcousticVTIBorn` with models `[vp, epsilon, delta, mp]` and
`AcousticTTIBorn` with `[vp, epsilon, delta, theta, mp]`: background and
scattered pure-qP fields, `mp` = velocity reflectivity `2 dv / v`. Eager and `impl='c'` (full
and boundary saving); the compiled backward returns the `mp` gradient only and refuses the
background models' and the wavelet's. No free surface.

## 5. Other imaging tools

| Tool | Where | Notes |
|---|---|---|
| Source / receiver illumination | `solver.compute_illumination = True` before the backward; read `solver.source_illumination`, `solver.receiver_illumination` | `impl='c'` |
| Space-lag ADCIG | `solver.compute_adcig = True`, `solver.adcig_max_lag = L`; read `solver.adcig` | `impl='c'`, boundary saving only, acoustic 2-D/3-D |
| Custom imaging condition | `solver.register_gradient(..., mode="imaging")` + `solver.imaging(...)` | eager, needs `memory=Full()` (see `gradients.md`) |

## 6. Elastic Born (`ElasticBorn`)

2-D first-order elastic Born pair (Li, Guo, Li & Alkhalifah 2019). Models
`[vp, vs, rho, dvp, dvs, drho]`; the perturbations are **absolute** (m/s, kg/m^3), not ratios.
Default source on the background stresses `sxx, szz` (explosion), receivers on the scattered
velocities `svx, svz`, so the output is the Born data only. `pml_type` `cpmls`, per-edge free
surface and topography as `Elastic`. The `dvp, dvs, drho` gradients are the elastic LSRTM images
in every memory mode on both `impl`s; the background (RWI) and wavelet gradients are
`impl='eager'` only and the compiled path refuses them (detach `vp, vs, rho` and the wavelet
on `impl='c'`).

## 7. Names before 0.3.6

0.3.6 renamed `AcousticLSRTM`, `AcousticLSRTM3D`, `AcousticVTILSRTM`, `AcousticTTILSRTM` to
`AcousticBorn`, `AcousticBorn3D`, `AcousticVTIBorn`, `AcousticTTIBorn` with **no aliases**: old
scripts get `ImportError`; rename the class, nothing else changed. The environment variable
`SWEEP_LSRTM_SPLIT_III` kept its name. Code that must also run on sweep <= 0.3.5:

```python
import sweep.equations as E
AcousticBorn = getattr(E, "AcousticBorn", None) or getattr(E, "AcousticLSRTM")
```
