# ElasticTTISG

```python
class ElasticTTISG(
    spatial_order=4,
    device="cpu",
    backend="torch",
)
```

Implementation:

- `src/sweep/equations/elastic_tti_sg.py`

Elastic TTI propagator on a **rotated staggered grid (RSG)**. RSG removes the
parameter-averaging requirement of a conventional staggered grid in anisotropic
media and tends to be more accurate near sharp anisotropy contrasts.

`ElasticTTISG` is the **compiled-binding-enabled** TTI variant: it has both
eager and `impl="c"` paths, including free-surface support.

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. Must be even.
- `device` (device or `str`, optional): Target device for derivative operators
  and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend. Typically `"torch"`.

## Models

Same as [ElasticTTI](elastic_tti.md):

```python
["vp0", "vs0", "rho", "epsilon", "delta", "gamma", "theta", "phi"]
```

## Wavefields

Same structure as [ElasticTTI](elastic_tti.md). User-facing source / receiver
fields:

- particle velocities: `vx`, `vy`, `vz`
- normal stresses: `sxx`, `szz` (source + receiver), plus `syz`, `sxz`, `sxy`
  (source only)

CPML memory variables are internal and not exposed through `available_fields()`
by default.

## Defaults

- `default_source_fields`: `["sxx", "szz"]`
- `default_receiver_fields`: `["vx", "vy", "vz"]`

## Backend behavior

- First-order RSG implementation. The rotated-staggered-grid derivative is
  shared with `sweep.operators.RSGDerivative`.
- Compiled binding supported for both CPU and CUDA. Free-surface boundary
  condition is implemented in the compiled path.

## Torch binding

- `supports_torch_binding()` → `True`
- Runtime buffer metadata for the compiled path is grouped under
  `cuda_layout`.

## See also

- [ElasticTTI](elastic_tti.md) — non-staggered TTI (eager only).
- [Elastic TTI wavefields example](../../examples/elastic_tti_wavefields.md)
  — RSG vs SG, absorbing vs free surface, side-by-side wavefield snapshots.
