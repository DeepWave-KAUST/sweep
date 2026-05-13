# Acoustic3D

```python
class Acoustic3D(
    spatial_order=4,
    device="cpu",
    backend="torch",
    dim=3,
)
```

Implementation:

- `src/sweep/equations/acoustic3d.py`

Second-order 3D acoustic wave equation.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. It should be an
  even number.
- `device` (device or `str`, optional): Target device used to place operator
  kernels and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend, typically `"torch"` or
  `"jax"`. For the compiled `c` implementation, this should still normally be
  `"torch"` rather than `"cuda"`.
- `dim` (`int`, optional): Stored dimensionality. For this class the intended
  value is `3`.

## Models

- `models` (`list[str]`): `["vp"]`

Required model:

- `vp`: P-wave velocity model

## Wavefields

- `wavefields` (`list[str]`): `["h1", "h2", "psix", "psiy", "psiz", "zetax", "zetay", "zetaz"]`

This extends the 2D acoustic state with `y`-direction CPML components.

## Backend Behavior

- uses separable 3D Laplace operators
- supports compiled extension binding through `_C()`
- supports RTM binding through `_C_rtm()`

## Torch Binding

- `supports_torch_binding()` : `True`
