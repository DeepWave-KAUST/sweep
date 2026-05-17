# Equations

This page summarizes the available equation classes and the model parameters they expect.
The goal is to make it easy to answer three practical questions:

- Which physical system does this equation represent?
- Which model tensors must be provided, and in what order?
- Which dimensions/backends does it support?

## Summary Table

Equations are grouped by physics family. The `impl="c"` column reflects whether
the compiled C++ / CUDA binding is available; the eager path (`impl="eager"`)
is supported for every class.

### Acoustic family

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `Acoustic` | `['vp']` | 2D | Second-order acoustic wave equation with CPML | ✅ |
| `Acoustic3D` | `['vp']` | 3D | 3D counterpart of `Acoustic` | ✅ |
| `Acoustic1st` | `['vp', 'rho']` | 2D | First-order velocity-pressure formulation | ❌ |
| `AcousticVRZ` | `['vp', 'rho']` | 2D | Variable-density acoustic (impedance-style parameterization) | ✅ |
| `AcousticVRZ3D` | `['vp', 'rho']` | 3D | 3D counterpart of `AcousticVRZ` | ✅ |
| `AcousticLSRTM` | `['vp', 'm']` | 2D | LSRTM-oriented acoustic variant | ✅ |
| `AcousticLSRTM3D` | `['vp', 'm']` | 3D | 3D counterpart of `AcousticLSRTM` | ✅ |

### Anisotropic acoustic family

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `AcousticTTI` | TTI parameters | 2D | qP acoustic TTI (optional dependency) | varies |
| `AcousticVTI` | VTI parameters | 2D | qP acoustic VTI | varies |
| `AcousticTariq` | TTI parameters | 2D | TTI variant from Alkhalifah-style formulation | varies |

### Elastic family

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `Elastic` | `['vp', 'vs', 'rho']` | 2D | Velocity-stress elastic propagation | ✅ |
| `Elastic3D` | `['vp', 'vs', 'rho']` | 3D | 3D counterpart of `Elastic` | ✅ |
| `ElasticTTI` | TTI elastic parameters | 2D | Elastic TTI propagation | varies |
| `ElasticTTISG` | TTI elastic parameters | 2D | Elastic TTI on a staggered grid | ✅ |

### Distributed Acoustic Sensing (DAS) family

| Equation | Models | Dim | Notes | `impl="c"` |
| --- | --- | --- | --- | --- |
| `DAS` / `DAS3D` | acoustic + strain parameters | 2D / 3D | DAS forward modeling | varies |
| `DASElastic` / `DASElastic3D` | elastic + strain parameters | 2D / 3D | Elastic DAS variant | varies |
| `DASMu` / `DASMu3D` | acoustic + strain parameters | 2D / 3D | μ-formulation DAS | ✅ (2D) |
| `DASZhao` / `DASZhao3D` | acoustic + strain parameters | 2D / 3D | Zhao-style DAS formulation | varies |

For the authoritative list of equations exported by your installation,
including constructor signatures and compiled-binding availability, use
the CLI:

```bash
sweep list equations
```

Or from Python:

```python
from sweep.equations import _equation_classes, torch_binding_supported_equations

print(sorted(_equation_classes().keys()))
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

`Acoustic` supports the compiled PyTorch extension binding through `sweep._C`.

That means:

- equation-level torch binding support: yes
- runtime availability still depends on whether your environment can import `sweep._C`

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

## See Also

- [API Reference](../api/index.md) — full API for every equation class, including
  [`Acoustic`](../api/equations/acoustic.md),
  [`Elastic`](../api/equations/elastic.md), and the 3D / VRZ / LSRTM variants.
- [Backends](backends.md) — choosing between `torch` and `jax`, and between
  `impl="eager"` and `impl="c"`.
- [Propagators](propagators.md) — wiring an equation into `PropTorch` or
  `PropJax`.
