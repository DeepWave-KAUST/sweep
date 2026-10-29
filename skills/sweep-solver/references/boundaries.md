# Boundaries: CPML, free surface, topography

Contents: 1. Absorbing layer (CPML) · 2. Free surface forms and support · 3. Where the free
surface actually is · 4. Topography · 5. `boundary_buffer` · 6. What is refused

## 1. Absorbing layer (CPML)

- `abcn` (default 50) CPML cells are **added outside** `shape` on every absorbing face; the
  model is edge-replicated into them. The runtime grid is `nz + 2*abcn` by `nx + 2*abcn`
  (minus `abcn` per free face), so a large `abcn` on a small model dominates cost. The examples
  here use 10-20 cells to stay fast; a thinner layer reflects more.
- `abcn` may be a per-face tuple `(z_lo, z_hi, x_lo, x_hi)` on 2-D `Acoustic`/`Elastic`/
  `ViscoAcoustic` with `impl='eager'` only (`impl='c'` raises `NotImplementedError`).
- `pml_type`: leave it `None`. Each equation declares `default_pml_type` and `supported_pml`;
  anything else raises `ValueError`.

| `pml_type` | Equations |
|---|---|
| `cpmlr` | `Acoustic*`, `AcousticVRZ*`, `AcousticVRR`, `AcousticBorn*`, `ViscoAcoustic`, `AcousticVTI/TTI` (+LSRTM), `AcousticTariq`, `AcousticCurvilinear`, `ElasticTTI` |
| `cpmls` | `Elastic`, `Elastic3D`, `ViscoElastic`, `ElasticVRR`, `ElasticTTISG*`, `ElasticTTI2nd`, `ElasticCurvilinear`, `DAS*`, `AcousticVTI1st*`, `Biot`, `Acoustic1st` (`spml` also allowed) |

- Sources and receivers placed inside the CPML (index `< 0` or `>= n`) are **not rejected**;
  they silently sit in the damping layer.

## 2. Free surface forms and support

Fixed at construction (assigning `solver.free_surface` later raises `AttributeError`). Forms:

| Value | Meaning |
|---|---|
| `False` (default) | absorbing on every face |
| `True`, `"top"` | top face (z = row 0 side) |
| `["top", "left"]`, `{"bottom": True}` | named faces: 2-D `top, bottom, left, right`; 3-D `top, bottom, front, back, left, right` |
| `(True, False, True, False)` | canonical per-face tuple, axis-major `(z_lo, z_hi, x_lo, x_hi)` |

| Equations | Free surface |
|---|---|
| 2-D `Acoustic`, `Elastic`, `ViscoAcoustic`, `ViscoElastic` | any face subset, eager and c, gradients in every memory mode (ViscoAcoustic/ViscoElastic: no boundary saving on c) |
| 3-D equations, `AcousticVRZ*`, `AcousticBorn*`, `AcousticVRR`, `ElasticVRR`, `DAS*`, `Biot` | top only (`True`); face lists raise |
| All anisotropic (`AcousticVTI*`, `AcousticTTI*`, `AcousticTariq`, `ElasticTTI*`) | refused (`NotImplementedError`) |
| `Acoustic1st` | accepted but **not implemented** (see section 3) |
| Under `ModelParallel` | top only |

```python
import torch
from sweep.equations import Acoustic, AcousticVTI
from sweep.propagator.torch import PropTorch

device = "cuda" if torch.cuda.is_available() else "cpu"
s = PropTorch(Acoustic(device=device), shape=(40, 48), dh=10.0, dt=1e-3, abcn=15,
              device=device, free_surface=["top", "left"])
print(s.fs_faces, s.pad)          # (True, False, True, False) (0, 15, 0, 15)
assert s.pad == (0, 15, 0, 15)

try:
    s.free_surface = False         # construction-only
except AttributeError as e:
    print("refused:", str(e)[:60])

try:
    PropTorch(AcousticVTI(device=device), shape=(40, 48), dh=10.0, dt=1e-3, free_surface=True)
except NotImplementedError as e:
    print("anisotropic:", str(e)[:60])
```

## 3. Where the free surface actually is

The free surface is an image/zero-halo condition outside row 0, not a plane through row 0.
Measured from the delay of the surface ghost (reflection polarity -1 in all cases):

| Equation | Effective pressure-release plane |
|---|---|
| `Acoustic` (eager and c, orders 4 and 8) | ~0.9 cell **above** row 0 (the top halo is zeroed) |
| `Elastic` (eager and c) | ~0.1 cell above row 0, i.e. at row 0 |
| `Acoustic1st` | ~`spatial_order/2 + 1` cells above row 0: no free-surface treatment, the top CPML is just removed and the zero-padded halo reflects |

Use this for ghost-delay, datum and source-depth calculations; put shallow receivers on row 0
(elastic) or accept the ~0.9-cell offset (acoustic). The measurement, for `Acoustic`:

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
backend = {"impl": "c"} if device == "cuda" else {
    "impl": "eager", "eager_options": EagerOptions(use_compile=False)}
v, dh, dt, nt, z0 = 2000.0, 10.0, 5e-4, 500, 10
shape = (60, 40)
w = ricker(np.arange(nt) * dt - 0.06, f=15.0).astype(np.float32)
vp = torch.full(shape, v, device=device)

def trace(fs, src_z):
    s = PropTorch(Acoustic(device=device), shape=shape, dh=dh, dt=dt, abcn=20,
                  device=device, free_surface=fs, **backend)
    with torch.no_grad():
        r = s(w, np.array([[20, src_z]]), np.array([[[20, z0]]]), models=[vp])
    return r[0, :, 0, 0].double().cpu().numpy()

ghost = trace(True, z0) - trace(False, z0)          # surface reflection only
image = trace(False, 3 * z0)                        # same path length if the plane were at row 0
up = 16
G = np.fft.irfft(np.fft.rfft(ghost), nt * up)
R = np.fft.irfft(np.fft.rfft(image), nt * up)
lag = (np.argmax(np.correlate(-G, R, "full")) - (nt * up - 1)) / up * dt
print(f"plane is {lag * v / (2 * dh):.2f} cells above row 0")   # ~0.9
assert 0.6 < lag * v / (2 * dh) < 1.2
```

## 4. Topography

| Option | Equations / impl | Notes |
|---|---|---|
| `topography=rows` (staircase image method) | eager: `Acoustic`, `Acoustic3D`, `Elastic`, `Elastic3D`, `ViscoAcoustic`; c: `Acoustic`, `Acoustic3D`, `Elastic`, `ElasticVRR` | `rows`: int array `(nx,)` (2-D) / `(ny, nx)` (3-D), surface row per column; free surface implied (no `free_surface=` needed); c refuses boundary saving with it, use `memory=Full()` or `Ckpt` |
| `AcousticCurvilinear`, `ElasticCurvilinear` + `topography=` | eager only | boundary-fitted grid, fractional rows allowed; sources/receivers stay physical indices |
| `sweep.sem.PropSEM` + `topography=` | eager and c | curved spectral elements; see `sem.md` |

Other equations raise `NotImplementedError` for `topography=`. Cells above the surface should
still hold sensible model values (they are vacuum for the solver). Topography cannot be combined
with per-edge free surfaces, `boundary_buffer`, or `ModelParallel`.

```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import Full
from sweep.signal import ricker

device = "cuda" if torch.cuda.is_available() else "cpu"
nz, nx, dt = 40, 48, 1e-3
rows = (4 + 3 * np.sin(np.linspace(0, np.pi, nx))).astype(np.int64)   # surface row per column
s = PropTorch(Acoustic(device=device), shape=(nz, nx), dh=10.0, dt=dt, abcn=15,
              device=device, topography=rows, memory=Full())
w = ricker(np.arange(200) * dt - 0.05, f=15.0).astype(np.float32)
rec = np.array([[[ix, int(rows[ix]) + 1] for ix in range(0, nx, 4)]])  # just below the surface
with torch.no_grad():
    d = s(w, np.array([[24, int(rows[24]) + 2]]), rec, models=[torch.full((nz, nx), 2000.0, device=device)])
print(s.free_surface, tuple(d.shape))      # True (1, 200, 12, 1)
assert torch.isfinite(d).all()
```

## 5. `boundary_buffer`

`boundary_buffer=None` (default) inserts the sigma=0 cells an equation's boundary-saving
gradient needs between the model and the CPML (e.g. `AcousticVRZ*`); shapes and coordinates do
not change. An explicit smaller value is refused under boundary saving; non-zero buffers cannot
be combined with `topography=`.

## 6. What is refused (raises rather than running the wrong physics)

- Free surface on anisotropic equations, including via `topography=`.
- Per-edge free surface / per-face `abcn` on 3-D or on equations other than 2-D
  `Acoustic`/`Elastic`/`ViscoAcoustic`/`ViscoElastic`; per-face `abcn` on `impl='c'`.
- `topo_method='apm'` (removed); anything except `'auto'`/`'image'`.
- Boundary saving + `topography=` on `impl='c'`.
