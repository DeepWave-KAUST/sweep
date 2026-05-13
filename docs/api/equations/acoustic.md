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
  `"jax"`. For the compiled `c` implementation, this should still normally be
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

## Field Metadata

This class defines structured field metadata through `FieldSpec`.

## Model Metadata

This class defines structured model metadata through `ModelSpec`.

Required models:

- `vp` alias: `velocity`
  Meaning: acoustic P-wave velocity model
  Units: `m/s`

User-facing fields:

- `h1` aliases: `pressure`, `p`
  Meaning: primary acoustic pressure-like wavefield

Internal fields:

- `h2`: previous-step pressure-like state
- `psix`, `psiz`, `zetax`, `zetaz`: CPML auxiliary fields

Useful helpers:

- `available_fields()`: returns the user-facing fields, excluding CPML fields
- `available_fields(role="source")`: valid source injection fields
- `available_fields(role="receiver")`: valid receiver sampling fields
- `describe_field("h1")` or `describe_field("pressure")`: returns the field description

Defaults:

- `default_source_fields`: `["h1"]`
- `default_receiver_fields`: `["h1"]`

## Backend Behavior

- PyTorch path uses separable Laplace operators
- compiled PyTorch extension binding is available through `_C()`
- JAX path is also supported

## CUDA Layout

For the compiled C++/CUDA propagator, this equation exposes:

- `cuda_layout`

This groups the CUDA runtime buffer metadata needed by the compiled `c` implementation instead of
storing separate `base_nvar`, `pml_nvar`, and checkpoint-count properties on
the equation class.

## Torch Binding

- `supports_torch_binding()` : `True`

Compiled entry points exposed by this class are used by `PropTorch(..., impl="c")`.
