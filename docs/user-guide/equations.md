# Equations

This page summarizes the available equation classes and the model parameters they expect.
The goal is to make it easy to answer three practical questions:

- Which physical system does this equation represent?
- Which model tensors must be provided, and in what order?
- Which dimensions/backends does it support?

## Summary Table

| Equation | Models | Notes | PyTorch Binding |
| --- | --- | --- | --- |
| `Acoustic/Acoustic3D` | `['vp']` | Second-order acoustic wave equation | ✅ |
| `Acoustic1st` | `['vp', 'rho']` | First-order acoustic wave equation | ❌ |
| `Elastic/Elastic3D` | `['vp', 'vs', 'rho']` | 2D Elastic wave propagation (Velocity-Stress) | ✅ |

## Acoustic

### What It Represents

`Acoustic` implements the second-order acoustic wave equation with CPML absorbing boundaries.
In the PyTorch path it is usually the default choice for scalar wave propagation, forward modeling,
and acoustic FWI examples.

Implementation:

- [acoustic.py](/home/wangs0j/repo/sweep/src/sweep/equations/acoustic.py:1)

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

`Acoustic` supports the compiled PyTorch CUDA binding through `sweep._C`.

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

## Next

If this template looks good, the next equation sections should follow the same pattern:

1. What it represents
2. Constructor
3. Required models
4. Wavefields
5. Dimensions and variants
6. Supported backends
7. Torch binding support
8. Notes
