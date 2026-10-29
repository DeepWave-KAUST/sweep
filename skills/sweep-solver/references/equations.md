# Equations

Contents: 1. Discovering what an equation needs · 2. Family tables · 3. Elastic example
(multi-component) · 4. Visco-acoustic example · 5. Anisotropic notes · 6. Facades and aliases

All classes live in `sweep.equations` and are constructed as
`Eq(spatial_order=4, device=..., backend="torch")` (exceptions noted). `spatial_order` is the
even FD accuracy order; `impl='c'` kernels are specialised for 2, 4, 6, 8 (higher orders run a
slower generic path). The full generated capability table (all fields, per-edge flags, boundary
saving, batched models) is in `api-signatures.md`.

## 1. Discovering what an equation needs

Never guess model order or field names; ask the equation:

```python
from sweep.equations import Acoustic, Elastic, DASZhao, equation_classes, torch_binding_supported_equations

eq = Elastic(device="cpu")
print(eq.models)                                    # ['vp', 'vs', 'rho']  -> models=[vp, vs, rho]
print(eq.default_source_fields, eq.default_receiver_fields)   # ['sxx', 'szz'] ['vx', 'vz']
print([f.name for f in eq.available_source_fields()])          # every valid source_type entry
print([f.name for f in eq.available_receiver_fields()])        # every valid receiver_type entry
print(eq.describe_field("vz"))
print(Acoustic.defaults())                          # spatial_order, pml_type, fields, models
print(len(equation_classes()), "registered names")
print(torch_binding_supported_equations()[:5], "...")          # names with impl='c' kernels
assert eq.models == ["vp", "vs", "rho"]
```

CLI equivalent: `sweep list equations`, `sweep show Elastic`. Field aliases are accepted
(`Acoustic`: `h1` = `pressure` = `p`).

## 2. Family tables

`c` = compiled CUDA kernels exist. FS = free surface: `per-edge` means any subset of the four 2-D
faces on eager and c; `top` = top face only; `refused` = raises `NotImplementedError`.

### Acoustic

| Class | Dim | models | src / rec default | c | FS | Notes |
|---|---|---|---|---|---|---|
| `Acoustic` | 2 | `vp` | `h1` / `h1` | yes | per-edge | workhorse; per-shot models `(B,nz,nx)` on c |
| `Acoustic3D` | 3 | `vp` | `h1` / `h1` | yes | top | |
| `AcousticVRZ`, `AcousticVRZ3D` | 2/3 | `vp, z` | `h1` / `h1` | yes | top | variable density, `z` = impedance; c refuses wavelet grad |
| `AcousticVRR` | 2 | `vp, rx, rz` | `h1` / `h1` | no | top | vector-reflectivity form |
| `Acoustic1st` | 2 | `vp, rho` | `p` / `p` (rec also `vx, vz`) | no | not implemented (see SKILL pitfall 12) | default `backend='jax'`; `pml_type` `cpmls` or `spml` |
| `ViscoAcoustic` | 2 | `vp, Q, omega` | `h1` / `h1` | yes | per-edge | Zhu & Harris (2014) constant Q; no boundary saving (c defaults to `Full()`) |
| `AcousticBorn`, `AcousticBorn3D` | 2/3 | `vp, mp` | `h1` / **`sh1`** | yes | top | Born pair; see `imaging.md` |
| `AcousticCurvilinear` | 2 | `vp` | `h1` / `h1` | no | built in | boundary-fitted grid for `topography=` |

### Anisotropic acoustic (all refuse a free surface)

| Class (aliases) | Dim | models | src / rec | c | Notes |
|---|---|---|---|---|---|
| `AcousticVTI` (`AcousticVTILiang`, `AcousticVTIDefault`) | 2 | `vp, epsilon, delta` | `h1` / `h1` | yes (full, BS) | pure qP, Liang et al. 2023; `s_gradient='adjoint'`; warns when `dt` breaks its stability limit; eager BS refused |
| `AcousticTTI` (`AcousticTTILiang`) | 2 | `vp, epsilon, delta, theta` | `h1` / `h1` | yes | `theta` in rad |
| `AcousticVTIBorn` / `AcousticTTIBorn` | 2 | `+ mp` | `h1` / `sh1` | yes, `mp` grad only | Born pair of the above |
| `AcousticVTI1st` (`AcousticVTIDuveneck`), `AcousticVTI1st3D` (`AcousticVTIDuveneck3D`, `AcousticVTIDefault3D`) | 2/3 | `vp, epsilon, delta, rho` | **`sH, sV`** / **`vz`** | yes | Duveneck 2008, staggered; warns if `epsilon < delta`; `recommended_dt(vp_max, epsilon_max, h)` classmethod |
| `AcousticTariq` (`AcousticVTIAlkhalifah`, `AcousticTTIAlkhalifah`) | 2 | `vv, v, eta` | `h1` / `h1` | no | Alkhalifah 2000 eta form |
| `AcousticAniso` | 2/3 | per `method=` | | | factory: `AcousticAniso(method="duveneck"|"liang"|"alkhalifah", symmetry="vti"|"tti", ndim=2)` returns one of the above |

### Elastic

| Class | Dim | models | src / rec default | c | FS | Notes |
|---|---|---|---|---|---|---|
| `Elastic` | 2 | `vp, vs, rho` | `sxx, szz` (explosion) / `vx, vz` | yes | per-edge | Virieux staggered; sources also `vx, vz` (force), `sxz`; receivers also `sxx, szz` |
| `Elastic3D` | 3 | `vp, vs, rho` | `sxx, syy, szz` / `vx, vy, vz` | yes | top | `type(eq).__name__` prints `Elastic` |
| `ElasticBorn` | 2 | `vp, vs, rho, dvp, dvs, drho` | `sxx, szz` / **`svx, svz`** | yes, `d*` grads only | per-edge | Born pair of `Elastic`; absolute perturbations; see `imaging.md` §6 |
| `ViscoElastic` | 2 | `vp, vs, rho, Qp, Qs` | as `Elastic` | yes | per-edge | GSLS, `f_ref=10.0`, `n_sls=3`; `Qp=Qs=inf` equals `Elastic`; c has no boundary saving |
| `ElasticVRR` | 2 | `vp, vs, Rp_x, Rp_z, Rs_x, Rs_z` | `sxx, szz` / `px, pz` | yes | top | vector reflectivity, no density model; helper `compute_vector_reflectivity` |
| `ElasticTTISG`, `ElasticTTISG3D` | 2/3 | `vp0, vs0, rho, epsilon, delta, gamma, theta, phi` | `sxx, szz` / `vx, vz` | yes | refused | standard staggered grid; `spatial_order` default 8 (2-D) |
| `ElasticTTI` | 2 | same 8 | `sxx, szz` / `vx, vz` | no | refused | rotated staggered grid, `pml_type` `cpmlr` |
| `ElasticTTI2nd` | 2 | `vh, vs, rho, epsilon, eta, theta` | `uz` / `ux, uz` | yes | refused | displacement form; plain CPML unstable late, keep `mpml_ratio=0.05` |
| `ElasticCurvilinear` | 2 | `vp, vs, rho` | `sxx, szz` / `vx, vz` | no | built in | Bartana-style boundary-fitted grid, `spatial_order` default 16 |

### DAS (elastic models, strain read-outs)

| Class (aliases) | Dim | rec default | Notes |
|---|---|---|---|
| `DASZhao` (`DASElastic`), `DASZhao3D` (`DASElastic3D`) | 2/3 | `exx_t, ezz_t` (strain rate) | also `das35_t`, `das54x_t`, `das54z_t` helical read-outs; `DASZhao3D` has no c boundary saving |
| `DASMu`, `DASMu3D` | 2/3 | `exx, ezz, exz` (strain) | Mu & Hung 2022 |
| `DAS` | 2/3 | | `nn.Module` facade: `DAS(shape=..., method="zhao"|"mu", gauge_length=..., ...)` builds the propagator itself |

### Poroelastic

`Biot` (2-D, eager only): models `H, mu, C, M, rho, rho_f, m, b` (Pa, kg/m^3, Pa s/m^2), build
them with `sweep.equations.biot.biot_from_rock(k_s, rho_s, k_f, rho_f, eta, k_d, mu, phi, kappa,
tortuosity=1.0)` (differentiable). Sources/receivers: any of `vx, vz, qx, qz, sxx, szz, sxz, p`.
Open-pore top free surface. Boundary saving is refused (dissipative); use `Ckpt`/`Full`.

## 3. Elastic example (multi-component)

The last output axis follows `receiver_type`. An explosion is `["sxx", "szz"]` (default); a
vertical force is `["vz"]`.

```python
import numpy as np, torch
from sweep.equations import Elastic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
nz, nx, dh, dt, nt = 48, 64, 10.0, 1e-3, 300
vp = torch.full((nz, nx), 2500.0, device=device)
vs = vp / 1.8
rho = torch.full((nz, nx), 2200.0, device=device)

solver = PropTorch(Elastic(device=device), shape=(nz, nx), dh=dh, dt=dt, abcn=20,
                   device=device, free_surface=True,
                   source_type=["vz"],                    # vertical force
                   receiver_type=["vx", "vz", "sxx"])     # output last axis = 3
w = ricker(np.arange(nt) * dt - 0.08, f=12.0).astype(np.float32)
src = np.array([[32, 1]])
rec = np.array([[[ix, 0] for ix in range(0, nx, 2)]])
with torch.no_grad():
    d = solver(w, src, rec, models=[vp, vs, rho])
print(tuple(d.shape))                                      # (1, 300, 32, 3)
vx, vz, sxx = d[..., 0], d[..., 1], d[..., 2]
assert d.shape == (1, nt, 32, 3) and torch.isfinite(d).all()
```

## 4. Visco-acoustic example

`Q` and `omega` are model arrays of the model shape (not scalars), and both get gradients.

```python
import math, numpy as np, torch
from sweep.equations import ViscoAcoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
shape, dt, nt = (40, 48), 1e-3, 250
vp = torch.full(shape, 2000.0, device=device)
Q = torch.full(shape, 50.0, device=device)
omega = torch.full(shape, 2 * math.pi * 15.0, device=device)   # reference angular frequency
solver = PropTorch(ViscoAcoustic(device=device), shape=shape, dh=10.0, dt=dt, abcn=15,
                   device=device)
print(solver.memory_strategy)            # impl='c': 'full' (no boundary saving); eager: 'ckpt'
w = ricker(np.arange(nt) * dt - 0.07, f=15.0).astype(np.float32)
Q.requires_grad_(True)
d = solver(w, np.array([[10, 5]]), np.array([[[38, 5]]]), models=[vp, Q, omega])
d.pow(2).sum().backward()
assert Q.grad is not None and Q.grad.shape == shape
```

## 5. Anisotropic notes

- Thomsen parameters are dimensionless arrays; tilt `theta` and azimuth `phi` are **radians**.
- `AcousticVTI`/`AcousticTTI` (pure qP) need a smaller `dt` than isotropic (as a first guess
  use `vp*sqrt(1+2*eps)` as `--vmax` in `scripts/stability_check.py`); `PropTorch` checks the
  paper's bound (Liang et al. 2023, eq. B-9; a safe-side bound for TTI) before every forward and
  warns once with the largest stable `dt`, but does not stop the run.
- `AcousticVTI1st` warns when `epsilon < delta` anywhere (unstable system, Grechka et al. 2004).
- `AcousticVTI1st` sources are `sH`/`sV` (horizontal / vertical stress-like fields), its
  receiver default `vz`; for a pressure-like record use `receiver_type=["sH"]` or `["sV"]`.
- No anisotropic equation takes a free surface or `topography=`.

## 6. Facades and aliases

- Aliases are the same class (`AcousticVTILiang is AcousticVTI`), listed by `sweep list equations`.
- `AcousticAniso(...)` and `DAS(...)` are factories/facades, not `WaveEquation` subclasses;
  `sweep show` refuses them.
- 0.3.6 renamed the `*LSRTM` classes to `*Born` with no aliases (see `imaging.md` §7).
