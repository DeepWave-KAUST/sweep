# Spectral elements: `sweep.sem.PropSEM`

Contents: 1. When to use · 2. Differences from `PropTorch` · 3. Example · 4. Equations ·
5. Sources and receivers · 6. Memory, batching, multi-GPU

## 1. When to use

Irregular free-surface topography that a staircase (`topography=` on `PropTorch`) or the 2-D
curvilinear FD classes cannot represent well, coupled fluid-solid (ocean over
seabed), poroelastic (Biot) on a curved mesh. Quad9 / Hex27 meshes are fitted to the surface,
GLL points, diagonal mass matrix. On a flat-topped box `PropTorch` is the simpler choice.

## 2. Differences from `PropTorch`

`PropSEM` keeps `PropTorch`'s constructor and call (`shape`, `dh`, `dt`, `abcn`, `free_surface`,
`topography`, `memory`, `impl`, `device`, `source_type`, `receiver_type`, `models=[...]`, output
`(nshots, nt, nrec, ncomp)`), with these differences:

| Item | `PropSEM` |
|---|---|
| Equation argument | the **class**, not an instance: `PropSEM(AcousticPotentialSEM2D, shape=..., ...)` |
| Grid | every axis must tile into elements: `(n - 1) % n_order == 0` for each entry of `shape` (also for cropped sub-models) |
| `abcn` | grid cells of C-PML, a multiple of `n_order`; `None` (default) = ten elements |
| `n_order` | GLL polynomial degree per element (default 4) |
| `topography` | surface row per column `(nx,)` / `(ny, nx)`, **fractional rows allowed**; free surface on top implied |
| `memory` | default `Full()`; `BoundarySaving(...)`, `Ckpt(chunks=...)` also accepted |
| `impl` | `None`: CUDA kernels when the class has them and the device is CUDA, else eager |
| Elastic default source | a **unit vertical force** (FD `Elastic` defaults to an explosion) |
| Acoustic potential density | constant `density=` keyword (default 1000.0), not a model |
| Snapshots | `return_wavefield=True` eager only, not with boundary saving; `impl='c'` refuses |
| Amplitude | same waveform as FD for the same `source_type` and wavelet, but amplitude scales differ |

Discretisation helper: `sweep.sem.suggest_discretization(x_max, z_max, v_min=..., v_max=...,
f0=..., n_order=4, ppw=5.0, courant=0.3, record_length=None)` returns element counts and a
stable `dt` (ppw 5 per shortest wavelength, Courant 0.3 on the smallest GLL spacing).

## 3. Example

```python
import numpy as np, torch
from sweep.sem import PropSEM
from sweep.sem.equations import AcousticPotentialSEM2D
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
nz, nx, dh = 41, 61, 10.0                  # (n - 1) % n_order == 0 on every axis
dt, nt = 5e-4, 600
rows = 4.0 + 3.0 * np.sin(np.linspace(0, np.pi, nx))     # surface row per column, fractional ok
solver = PropSEM(AcousticPotentialSEM2D, shape=(nz, nx), dh=dh, dt=dt, n_order=4,
                 abcn=8, topography=rows, f_dominant=15.0, device=device,
                 receiver_type=["p"])              # pressure, derived from the potential
w = ricker(np.arange(nt) * dt - 0.07, f=15.0).astype(np.float32)
vp = torch.full((nz, nx), 2000.0, device=device, requires_grad=True)
seis = solver(w, np.array([[30, 15]]), np.array([[[ix, 12] for ix in range(5, 56, 5)]]),
              models=[vp])
seis.pow(2).sum().backward()
print(tuple(seis.shape), tuple(vp.grad.shape))      # (1, 600, 11, 1) (41, 61)
assert torch.isfinite(vp.grad).all()
```

## 4. Equations (`sweep.sem.equations`)

`sweep.sem.equations.describe()` prints the live table; `sweep list equations` lists them after
the FD classes. `models=` per class: `api-signatures.md`.

| Class | Mesh | Models | c | C-PML | Use |
|---|---|---|---|---|---|
| `AcousticPotentialSEM2D` | quad9 | `vp` | yes (batched, BS) | yes | acoustic FWI workhorse; records `chi`, `p` |
| `AcousticSEM2D` / `AcousticSEM2DDeformed` | quad4 / quad9 | `vp` | no | yes | |
| `AcousticSEM3D` / `AcousticSEM3DDeformed` | hex8 / hex27 | `vp` | yes | yes | 3-D acoustic (with topography for Deformed) |
| `ElasticSEM2DDeformed` | quad9 | `vp, vs, rho` | yes | yes | 2-D elastic with topography |
| `ElasticVSSEM2DDeformed` | quad9 | `vp, vs, rho` | no | yes | velocity-stress |
| `ElasticSEM2D` | quad4 | `vp, vs, rho` | no | **no** (reflecting box, `abcn=0`) | |
| `ElasticSEM3D` / `ElasticSEM3DDeformed` | hex8 / hex27 | `vp, vs, rho` | yes | **no** | |
| `ElasticVSSEM3D` | hex8 | `vp, vs, rho` | yes | yes | 3-D elastic with absorbing boundary, flat |
| `ElasticVSSEM3DDeformed` | hex27 | `vp, vs, rho` | no | yes | 3-D elastic, topography + absorbing |
| `ElasticVTISEM2DDeformed` | quad9 | `vp, vs, rho, epsilon, delta` | no | yes | VTI |
| `AcousticElasticSEM2D` | quad9 | `vp, vs, rho` | yes | yes | coupled fluid-solid (ocean over seabed); records `chi, u_x, u_z` (+ `p, vx, vz`) |
| `BiotVSSEM2DDeformed`, `BiotAnisoPermSEM2DDeformed` | quad9 | Biot set | no | yes | poroelastic; no boundary saving |

## 5. Sources and receivers

- `receiver_type=["p", "vx", "vz"]` work on every SEM equation (converted from the native fields,
  e.g. `p = -d2(chi)/dt2` on the potential form, `v = du/dt` on displacement elastics).
  `p` is receiver-only.
- Elastic `source_type`: `["vz"]`/`["vx"]` = force along that axis (default: unit vertical
  force), `["sxx", "szz"]` = explosion (moment rate), `["sxz"]` = shear moment. Forces and
  stresses cannot be mixed in one call; stress names are 2-D only.

## 6. Memory, batching, multi-GPU

- Multi-shot calls on `impl='c'` run as one batched kernel sequence (2-3x faster than a shot
  loop) when the class supports it (2-D C-PML paths), all shots share the same receiver rows,
  `memory` is `None`/boundary saving, and the wavelet is `(nt,)` or `(nshots, nt)`; otherwise it
  falls back to a per-shot loop (same result).
- Per-shot models `(nshots, nz, nx)`: `AcousticPotentialSEM2D` only.
- Boundary saving storage dtypes and coverage per class: see the user guide
  (`docs/user-guide/sem.md`); Biot classes refuse it.
- Domain decomposition (x strips, `impl='c'`): `sweep.sem.dd` for `AcousticPotentialSEM2D`,
  `ElasticSEM2DDeformed`, `AcousticElasticSEM2D`, `AcousticSEM3D`, `AcousticSEM3DDeformed`.
