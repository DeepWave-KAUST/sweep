# DAS family

Distributed Acoustic Sensing (DAS) data is essentially an axial strain-rate
measurement along an optical fiber. SWEEP provides several closely related
equation classes for forward modeling DAS gathers, differing in the underlying
physics (acoustic vs elastic) and in how the strain-rate output is constructed.

**Unified facade**: `DAS` (formerly named `DASModeler`) selects the right
raw class via the keyword arguments `method=` and `ndim=`, builds a
`PropTorch` internally, and exposes a `forward(...)` that returns the
receiver record.  The raw classes below remain usable directly when you
need finer control:

| Class | Dim | Underlying physics | Models | Compiled binding |
| --- | --- | --- | --- | --- |
| `DASElastic` | 2D | Strain-rate computed from elastic stresses (alias of `DASZhao`) | `['vp', 'vs', 'rho']` | ✓ |
| `DASElastic3D` | 3D | Same as `DASElastic` (alias of `DASZhao3D`) | `['vp', 'vs', 'rho']` | ✓ |
| `DASMu` | 2D | μ least-squares strain-rate formulation | `['vp', 'vs', 'rho']` | ✓ |
| `DASMu3D` | 3D | Same as `DASMu` | `['vp', 'vs', 'rho']` | ✓ |
| `DASZhao` | 2D | Zhao-style two-stage (stress + auxiliary tau) | `['vp', 'vs', 'rho']` | ✓ |
| `DASZhao3D` | 3D | Same as `DASZhao` | `['vp', 'vs', 'rho']` | ✓ |

All live under `sweep.equations`:

```python
from sweep.equations import (
    DAS,                       # unified facade (was DASModeler)
    DASModeler,                # back-compat alias of DAS
    DASElastic, DASElastic3D,
    DASMu, DASMu3D,
    DASZhao, DASZhao3D,
)
```

Implementation: `src/sweep/equations/das.py` (`DASZhao` / `DASZhao3D`,
`DASMu` / `DASMu3D`, plus the `DAS` facade and the back-compat
`DASElastic` / `DASElastic3D` / `DASModeler` aliases).

## Shared model parameters

All classes use the elastic `(vp, vs, rho)` triple even when the DAS receiver
is modelled by an acoustic-like formulation — the strain-rate output is
constructed from the underlying elastic stresses or auxiliary variables.

| Name | Meaning | Units |
| --- | --- | --- |
| `vp` (alias `p_velocity`) | Elastic P-wave velocity | m/s |
| `vs` (alias `s_velocity`) | Elastic S-wave velocity | m/s |
| `rho` (alias `density`) | Density | kg/m³ |

## User-facing fields

Every DAS class exposes the same kinds of fields, with naming consistent across
2D and 3D variants:

- **Strain-rate receiver fields** (receiver-only): `exx_t`, `ezz_t`,
  optionally `eyy_t` in 3D — normal strain-rates along the labelled axis.
- **Helical-fiber receiver fields** (receiver-only): `das35_t` for a 35.3°
  helical fiber, `das54x_t` / `das54z_t` for 54.7° helical fibers with x or z
  core orientation. The helper that computes them from the underlying normal
  strain-rates lives in [`sweep.signal.helical_das_response`](#signal-helpers).
- **Stress sources** (source-only or source + receiver): `sxx`, `szz`,
  optionally `syy` / `sxz` / `syz` / `sxy` depending on the variant.

Use the introspection helpers to discover the exact field set on the class
you're using:

```python
from sweep.equations import DASZhao, DASMu3D

# Note: introspection helpers live on the raw equation classes, not on the
# `DAS` facade (which dispatches by method= and is not a WaveEquation).
print([f.name for f in DASZhao.available_fields(role="receiver")])
print([f.name for f in DASMu3D.available_fields(role="source")])
print(DASZhao.describe_field("das35_t"))
```

## Signal helpers

`sweep.signal` ships two helpers that complement the DAS family:

```python
from sweep.signal import gauge_average, helical_das_response

# Sliding average over a finite gauge length (DAS gauge averaging).
exx_t_gauge = gauge_average(exx_t, gauge_length=10.0, spacing=1.0)

# Project normal strain-rates onto a 35.3° helical fiber response.
das_trace = helical_das_response(
    exx_t, ezz_t, angle=35.3, core_axis="x",
    gauge_length=10.0, spacing=1.0,
)
```

Both helpers accept either `torch.Tensor` or `numpy.ndarray` inputs.

## Compiled-binding notes

All eight DAS classes are torch-binding capable
(`supports_torch_binding()` → `True`) and ship `cuda_layout` metadata.
Backward-mode support — including boundary saving, chunk-mode checkpointing,
and recursive checkpointing — is available for the 2D variants. See
[Propagator Options](../propagators/options.md) for how to wire
`CUDAOptions(memory=...)` blocks into `PropTorch`.

## See also

- Runnable end-to-end DAS example: [DAS Figure 4 reproduction](../../examples/wavefields_das.md).
- Method comparison across DAS variants:
  `examples/wavefields/das/reproduce_layered_das.py`.
