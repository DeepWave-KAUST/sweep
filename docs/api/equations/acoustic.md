# Acoustic

```python
class Acoustic(
    spatial_order=4,
    device="cpu",
    backend="torch",
    dim=2,
)
```

Implementation:

- `src/sweep/equations/acoustic.py`

Second-order 2D acoustic wave equation with CPML auxiliary fields.

!!! note

    This is the most common scalar-wave equation in the codebase and is the
    default choice for many forward modeling and acoustic inversion examples.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. It should be an
  even number such as `4` or `8`.
- `device` (device or `str`, optional): Target device used to place operator
  kernels and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend, typically `"torch"` or
  `"jax"`. If you plan to run with `PropCUDA`, this should still normally be
  `"torch"` rather than `"cuda"`.
- `dim` (`int`, optional): Stored dimensionality. For this class the intended
  value is `2`.

## Models

- `models` (`list[str]`): `["vp"]`

You must provide one model tensor:

- `vp`: P-wave velocity model

## Wavefields

- `wavefields` (`list[str]`): `["h1", "h2", "psix", "psiz", "zetax", "zetaz"]`

The first two are the main second-order wavefield states, and the remaining
entries are CPML auxiliary fields.

## Backend Behavior

- PyTorch path uses separable Laplace operators
- CUDA-backed PyTorch binding is available through `_C()`
- JAX path is also supported

## Torch Binding

- `supports_torch_binding()` : `True`

Compiled entry points exposed by this class are used by `PropCUDA`.
