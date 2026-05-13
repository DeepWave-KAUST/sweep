# Elastic3D

```python
class Elastic3D(
    spatial_order=4,
    device="cpu",
    backend="torch",
)
```

Implementation:

- `src/sweep/equations/elastic3d.py`

First-order 3D elastic velocity-stress equation.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. It should be an
  even number.
- `device` (device or `str`, optional): Target device used to place derivative
  operators and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend, typically `"torch"`. For the
  compiled `c` implementation, this should still normally be `"torch"` rather
  than `"cuda"`.

## Models

- `models` (`list[str]`): `["vp", "vs", "rho"]`

Required models:

- `vp`: P-wave velocity
- `vs`: S-wave velocity
- `rho`: density

## Wavefields

- `wavefields` (`list[str]`): `["vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz", "m_vxx", "m_vxy", "m_vxz", "m_vyx", "m_vyy", "m_vyz", "m_vzx", "m_vzy", "m_vzz", "m_sxxx", "m_szzz", "m_sxyx", "m_sxyy", "m_sxzx", "m_sxzz", "m_syyy", "m_syzy", "m_syzz"]`

This extends the 2D elastic formulation with `y`-direction physical and CPML
memory variables.

## Backend Behavior

- implemented as a first-order 3D equation using `PartialDerivative`
- supports compiled extension binding through `_C()`

## Torch Binding

- `supports_torch_binding()` : `True`
