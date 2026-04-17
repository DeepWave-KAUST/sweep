# Elastic

```python
class Elastic(
    spatial_order=4,
    device="cpu",
    backend="torch",
)
```

Implementation:

- `src/sweep/equations/elastic.py`

First-order 2D elastic velocity-stress equation.

!!! note

    This formulation uses velocity and stress wavefields together with multiple
    CPML memory variables, so its state is much larger than the acoustic one.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. It should be an
  even number.
- `device` (device or `str`, optional): Target device used to place derivative
  operators and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend, typically `"torch"`. If you
  plan to run with `PropCUDA`, this should still normally be `"torch"` rather
  than `"cuda"`.

## Models

- `models` (`list[str]`): `["vp", "vs", "rho"]`

Required models:

- `vp`: P-wave velocity
- `vs`: S-wave velocity
- `rho`: density

## Wavefields

- `wavefields` (`list[str]`): `["vx", "vz", "sxx", "szz", "sxz", "m_vxx", "m_vxz", "m_vzx", "m_vzz", "m_txxx", "m_txxz", "m_tzzx", "m_tzzz", "m_txzx", "m_txzz"]`

The first five entries are the physical velocity-stress fields. The remaining
entries are CPML memory variables.

## Backend Behavior

- implemented as a first-order equation using `PartialDerivative`
- supports compiled CUDA binding through `_C()`

## Torch Binding

- `supports_torch_binding()` : `True`
