# AcousticVRZ

```python
class AcousticVRZ(
    spatial_order=4,
    device="cpu",
    backend="torch",
    dim=2,
)
```

Implementation:

- `src/sweep/equations/acoustic_vrz.py`

2D acoustic equation using the `vp` and `z` parameterization.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. It should be an
  even number.
- `device` (device or `str`, optional): Target device used to place operator
  kernels and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend, typically `"torch"` or
  `"jax"`. If you plan to run with `PropCUDA`, this should still normally be
  `"torch"` rather than `"cuda"`.
- `dim` (`int`, optional): Stored dimensionality. For this class the intended
  value is `2`.

## Models

- `models` (`list[str]`): `["vp", "z"]`

Required models:

- `vp`: velocity parameter
- `z`: impedance-like parameter used by this formulation

## Wavefields

- `wavefields` (`list[str]`): `["h1", "h2", "psix", "psiz", "zetax", "zetaz"]`

## Backend Behavior

- uses separable Laplace operators
- uses gradient kernels in the torch implementation
- supports compiled CUDA binding through `_C()`

## Torch Binding

- `supports_torch_binding()` : `True`
