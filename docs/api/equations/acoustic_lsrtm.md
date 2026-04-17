# AcousticLSRTM

```python
class AcousticLSRTM(
    spatial_order=4,
    device="cpu",
    backend="torch",
)
```

Implementation:

- `src/sweep/equations/acoustic_lsrtm.py`

Linearized acoustic equation for LSRTM-style modeling with background and
scattered wavefields.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. It should be an
  even number.
- `device` (device or `str`, optional): Target device used to place operator
  kernels and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend. If you plan to run with
  `PropCUDA`, this should still normally be `"torch"` rather than `"cuda"`.

## Models

- `models` (`list[str]`): `["vp", "mp"]`

Required models:

- `vp`: background velocity model
- `mp`: perturbation or reflectivity-like model term used by the scattered
  wavefield update

## Wavefields

- `wavefields` (`list[str]`): `["h1", "h2", "psix", "psiz", "zetax", "zetaz", "sh1", "sh2", "spsix", "spsiz", "szetax", "szetaz"]`

This class carries both background and scattered states.

## Backend Behavior

- implemented as a second-order equation
- currently does not expose a compiled torch binding hook

## Torch Binding

- `supports_torch_binding()` : `False`
