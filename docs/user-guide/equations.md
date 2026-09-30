# Equations

This page summarizes the available equation classes and the model parameters they expect.
The goal is to make it easy to answer three practical questions:

- Which physical system does this equation represent?
- Which model tensors must be provided, and in what order?
- Which dimensions/backends does it support?

## Summary Table

Equations are grouped by physics family. The `impl="c"` column reflects whether
compiled CUDA kernels exist in the shipped core; the eager path (`impl="eager"`)
is supported for every class.

### Acoustic family

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `Acoustic` | `['vp']` | 2D | Second-order acoustic wave equation with CPML | ✅ |
| `Acoustic3D` | `['vp']` | 3D | 3D counterpart of `Acoustic` | ✅ |
| `Acoustic1st` | `['vp', 'rho']` | 2D | First-order velocity-pressure formulation | ❌ |
| `ViscoAcoustic` | `['vp', 'Q', 'omega']` | 2D | Nearly constant-Q attenuation ([Zhu & Harris 2014](https://doi.org/10.1190/geo2013-0245.1)); decoupled dispersion / damping switches | ✅ |
| `AcousticVRZ` | `['vp', 'z']` | 2D | Variable-density acoustic, impedance-style parameterization (``z`` = impedance) | ✅ |
| `AcousticVRZ3D` | `['vp', 'z']` | 3D | 3D counterpart of `AcousticVRZ` | ✅ |
| `AcousticLSRTM` | `['vp', 'mp']` | 2D | LSRTM-oriented variant (``mp`` = reflectivity perturbation) | ✅ |
| `AcousticLSRTM3D` | `['vp', 'mp']` | 3D | 3D counterpart of `AcousticLSRTM` | ✅ |
| `AcousticVRR` | `['vp', 'rx', 'rz']` | 2D | Variable-density acoustic in vector-reflectivity form (``rx``, ``rz`` first-derivative density parameters) | ❌ |
| `AcousticCurvilinear` | `['vp']` | 2D | `Acoustic` on a curvilinear grid for irregular topography (`topography=`); eager only | ❌ |

### Anisotropic acoustic family

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `AcousticAniso` | varies (see facade) | 2D / 3D | **Unified facade** — `method=` picks duveneck / liang / alkhalifah; mirrors the `DAS` facade | varies |
| `AcousticVTI` | `['vp', 'epsilon', 'delta']` | 2D | qP acoustic VTI, Liang K. et al. 2022 — aliases `AcousticVTILiang` / `AcousticVTIDefault` | ❌ |
| `AcousticTTI` | `['vp', 'epsilon', 'delta', 'theta']` | 2D | qP acoustic TTI, Liang K. et al. 2022 — alias `AcousticTTILiang` | ❌ |
| `AcousticTariq` | `['vv', 'v', 'eta']` | 2D | qP η-formulation, Alkhalifah 2000 — aliases `AcousticVTIAlkhalifah` / `AcousticTTIAlkhalifah` | ❌ |
| `AcousticVTI1st` | `['vp', 'epsilon', 'delta', 'rho']` | 2D | 1st-order velocity-stress, Duveneck et al. 2008 — alias `AcousticVTIDuveneck` | ✅ |
| `AcousticVTI1st3D` | `['vp', 'epsilon', 'delta', 'rho']` | 3D | 3D counterpart of `AcousticVTI1st` — aliases `AcousticVTIDuveneck3D` / `AcousticVTIDefault3D` | ✅ |

### Elastic family

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `Elastic` | `['vp', 'vs', 'rho']` | 2D | Velocity-stress elastic propagation (Virieux 1986) | ✅ |
| `Elastic3D` | `['vp', 'vs', 'rho']` | 3D | 3D counterpart of `Elastic` | ✅ |
| `ElasticAPM` | `['vp', 'vs', 'rho']` | 2D | Alias of `Elastic` (Cao & Chen 2018 APM free surface) | ✅ |
| `ElasticCurvilinear` | `['vp', 'vs', 'rho']` | 2D | `Elastic` on a curvilinear grid for irregular topography (`topography=`); eager only | ❌ |
| `ElasticVRR` | `['vp', 'vs', 'Rp_x', 'Rp_z', 'Rs_x', 'Rs_z']` | 2D | Vector-reflectivity elastic (Soares & Sacchi 2025), momentum-stress form, no density model | ✅ |
| `ElasticTTI` | `['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']` | 2D | Rotated-staggered-grid elastic TTI | ❌ |
| `ElasticTTISG` | `['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']` | 2D | Standard-staggered-grid elastic TTI | ✅ |
| `ElasticTTISG3D` | `['vp0', 'vs0', 'rho', 'epsilon', 'delta', 'gamma', 'theta', 'phi']` | 3D | Standard-staggered-grid elastic TTI | ✅ |
| `ElasticTTI2nd` | `['vh', 'vs', 'rho', 'epsilon', 'eta', 'theta']` | 2D | Displacement-based elastic TTI (Oh et al. 2020) | ✅ |

### Distributed Acoustic Sensing (DAS) family

All DAS classes take elastic models — `['vp', 'vs', 'rho']` — and read out
strain or strain-rate at the receivers; the choice of formulation only changes
the receiver-side discretisation.

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `DAS` | `['vp', 'vs', 'rho']` | 2D / 3D | **Unified facade** — `method=` picks zhao / mu (formerly `DASModeler`) | ✅ |
| `DASZhao` / `DASZhao3D` | `['vp', 'vs', 'rho']` | 2D / 3D | Zhao-style DAS formulation | ✅ |
| `DASMu` / `DASMu3D` | `['vp', 'vs', 'rho']` | 2D / 3D | Mu & Hung (2022) velocity-stress formulation with integrated strain | ✅ |
| `DASElastic` / `DASElastic3D` | `['vp', 'vs', 'rho']` | 2D / 3D | Elastic DAS variant (alias of `DASZhao` / `DASZhao3D`) | ✅ |

For the authoritative list of equations exported by your installation,
with their models and compiled-binding support, use the CLI:

```bash
sweep list equations
```

Or from Python:

```python
from sweep.equations import equation_classes, torch_binding_supported_equations

print(sorted(equation_classes().keys()))
print(torch_binding_supported_equations())
```

## Acoustic

### What It Represents

`Acoustic` implements the second-order acoustic wave equation with CPML absorbing boundaries.
In the PyTorch path it is usually the default choice for scalar wave propagation, forward modeling,
and acoustic FWI examples.

Implementation:

- [`Acoustic` API page](../api/equations/acoustic.md)
- Source on GitHub: [`src/sweep/equations/acoustic.py`](https://github.com/DeepWave-KAUST/sweep/blob/dev/src/sweep/equations/acoustic.py)

### Constructor

```python
from sweep.equations import Acoustic

eq = Acoustic(
    spatial_order=4,
    device="cuda",
    backend="torch",
)
```

Main constructor arguments:

- `spatial_order`: finite-difference order. Must be even. Common values are `4` and `8`.
- `device`: target device for operator/kernel tensors, such as `"cpu"` or `"cuda"`.
- `backend`: usually `"torch"` or `"jax"`.
- `dim`: defaults to `2` for `Acoustic`.

### Required Models

`Acoustic.models` returns:

```python
['vp']
```

You must therefore provide one model tensor, in this order:

- `vp`: P-wave velocity model

### Wavefields

`Acoustic.wavefields` returns:

```python
['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz']
```

Roughly:

- `h1`, `h2`: main second-order wavefield states
- `psix`, `psiz`, `zetax`, `zetaz`: CPML auxiliary states

### Dimensions and Variants

- `Acoustic`: 2D acoustic equation
- `Acoustic3D`: 3D counterpart
- `AcousticVRZ`: acoustic variant with impedance-like parameterization
- `AcousticLSRTM`: acoustic LSRTM-oriented variant

### Supported Backends

`Acoustic` is commonly used with:

- `backend="torch"`: pure PyTorch propagation and autograd
- `backend="jax"`: JAX propagation and differentiation

### Torch Binding Support

`Acoustic` ships compiled CUDA kernels in the core (`C_NAME = "acoustic2d"`).

That means:

- equation-level torch binding support: yes
- runtime availability depends on a visible CUDA GPU and a fitting core
  (`sweep.backend.torch.binding.is_available()`), not on importing `sweep._C`,
  which always succeeds

You can inspect this from the CLI:

```bash
sweep list equations
```

Or from Python:

```python
import sweep

print(sweep.backend.torch.binding.is_available())
```

### Notes

- In the current PyTorch implementation, the acoustic path uses separable Laplace operators and,
  on CUDA, fixed-stencil gradient kernels where available.

## Inspecting available sources and receivers

Every equation class carries a structured `FIELD_SPECS` table that lists, for
each wavefield it tracks, whether the user is allowed to **inject a source**
there or **place a receiver** there. The propagator reads this when you pass
`source_type=` / `receiver_type=`; you can query it yourself the same way.

```python
from sweep.equations import Acoustic, Elastic, DASZhao

eq = Acoustic(device="cuda")
eq.default_source_fields            # ['h1']
eq.default_receiver_fields          # ['h1']
eq.available_source_fields()        # [FieldSpec(name='h1', aliases=('pressure','p'), ...)]
eq.describe_field("pressure")       # 'h1: ... Aliases: pressure, p.'  (aliases are accepted too)

Elastic(device="cuda").available_receiver_fields()
# → [vx, vz, sxx, szz]    (FieldSpec list; supports_receiver=True on each)

DASZhao(device="cuda").available_receiver_fields()
# → [exx_t, ezz_t, sxx, szz, das35_t, das54x_t, das54z_t]
#   — DAS exposes helical-fiber strain-rate readouts on top of the elastic basics
```

Each `FieldSpec` carries `name`, `aliases`, `description`, `supports_source`,
`supports_receiver`, and `internal` flags — so an IDE / notebook can autocomplete
through them without having to read the source. A handful of equations also have
`default_source_fields` / `default_receiver_fields` populated, which is what
`PropTorch` falls back to when `source_type=` / `receiver_type=` are omitted on
the propagator.

If the default for an equation doesn't suit your survey (e.g. you want to
inject on `sxx + szz + sxz` in `Elastic` instead of just `sxx + szz`), pass an
explicit list:

```python
solver = PropTorch(
    Elastic(device=dev),
    shape=shape, dh=dh, dt=dt, device=dev,
    source_type=["sxx", "szz", "sxz"],     # explicit override
    receiver_type=["vz"],
)
```

## Adding your own equation

The equation classes above are plug-in units. The propagator owns the time
loop and PML wiring, so a new equation is one Python file (eager) plus an
optional CUDA equation directory and five entries in the core's C API table
(`csrc/core/capi.h`, `csrc/cuda/common/capi.cu`) (`impl="c"`). Registration is a decorator plus an import: `@register_equation()` on the
class claims its public name, and importing the module in
`src/sweep/equations/__init__.py` is what makes the decorator run. Both are
needed before it appears in `sweep list equations` or `equation_classes()`.

Two entry points:

- [Extending: Adding a New Equation](extending.md) — the reference: the
  interface contract, the `CUDALayoutSpec` field table, and the C++
  registration pattern.
- [Add a new equation notebook](../notebooks/18_extending_add_new_equation.ipynb)
  — a runnable walkthrough that builds and runs a toy `MyScalar` against
  `PropTorch` end-to-end.

## See Also

- [API Reference](../api/index.md) — full API for every equation class, including
  [`Acoustic`](../api/equations/acoustic.md),
  [`Elastic`](../api/equations/elastic.md), and the 3D / VRZ / LSRTM variants.
- [Backends](backends.md) — choosing between `torch` and `jax`, and between
  `impl="eager"` and `impl="c"`.
- [Propagators](propagators.md) — wiring an equation into `PropTorch` or
  `PropJax`.
- [Extending](extending.md) — adding a new equation class (reference) and the
  [companion notebook](../notebooks/18_extending_add_new_equation.ipynb)
  (runnable walkthrough).
