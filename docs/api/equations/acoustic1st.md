# Acoustic1st

```python
class Acoustic1st(
    spatial_order=4,
    device="cpu",
    backend="jax",
)
```

Implementation:

- `src/sweep/equations/acoustic1st.py`

First-order acoustic equation in pressure-velocity form.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. It should be an
  even number.
- `device` (device or `str`, optional): Target device used to place operator
  kernels and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend. The default in this class is
  `"jax"`. If you plan to run with a torch-based propagator, the equation side
  should still use `"torch"` rather than `"cuda"`.

## Models

- `models` (`list[str]`): `["vp", "rho"]`

Required models:

- `vp`: P-wave velocity model
- `rho`: density model

## Wavefields

The actual wavefields depend on the selected PML type.

- `cpmls` mode: `["p", "vx", "vz", "phix", "phiz", "psix", "psiz"]`
- `spml` mode: `["px", "pz", "vx", "vz"]`

## Backend Behavior

- equation behavior changes through `setup_pml(pml_type)`
- supports `cpmls` and `spml`
- does not expose a compiled torch binding hook

## Torch Binding

- `supports_torch_binding()` : `False`
