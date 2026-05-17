# ElasticTTI

```python
class ElasticTTI(
    spatial_order=4,
    device="cpu",
    backend="torch",
)
```

Implementation:

- `src/sweep/equations/elastic_tti.py`

3D-compatible first-order elastic equation in a **tilted transversely
isotropic (TTI)** medium. The formulation uses Thomsen-style anisotropy
parameters (`epsilon`, `delta`, `gamma`) together with tilt and azimuth angles
(`theta`, `phi`) on top of the usual `(vp0, vs0, rho)` velocity-density triple.

!!! note

    The non-staggered TTI equation is provided through the PyTorch eager
    path only. For the compiled C++ / CUDA path, use
    [`ElasticTTISG`](elastic_tti_sg.md) (rotated staggered grid).

## Parameters

- `spatial_order` (`int`, optional): Finite-difference order. Must be even.
- `device` (device or `str`, optional): Target device for derivative operators
  and backend-specific tensors.
- `backend` (`str`, optional): Numerical backend. Typically `"torch"`.

## Models

`models` (`list[str]`) — required order:

```python
["vp0", "vs0", "rho", "epsilon", "delta", "gamma", "theta", "phi"]
```

| Name | Meaning | Units |
| --- | --- | --- |
| `vp0` | VTI-frame vertical P velocity | m/s |
| `vs0` | VTI-frame vertical S velocity | m/s |
| `rho` (alias `density`) | Density | kg/m³ |
| `epsilon` | Thomsen ε | — |
| `delta` | Thomsen δ | — |
| `gamma` | Thomsen γ | — |
| `theta` | Tilt angle | rad |
| `phi` | Azimuth angle | rad |

## Wavefields

```python
["vx", "vy", "vz",
 "sxx", "szz", "syz", "sxz", "sxy",
 "m_vxx", "m_vxz", "m_vyx", "m_vyz", "m_vzx", "m_vzz",
 "m_txxx", "m_txzz", "m_txyx", "m_tyzz", "m_txzx", "m_tzzz"]
```

User-facing fields (selectable as `source_type` / `receiver_type`):

| Name | Aliases | Source | Receiver | Meaning |
| --- | --- | --- | --- | --- |
| `vx` | `velocity_x` | ✓ | ✓ | Particle velocity in x |
| `vy` | `velocity_y` | ✓ | ✓ | Particle velocity in y |
| `vz` | `velocity_z` | ✓ | ✓ | Particle velocity in z |
| `sxx` | `stress_xx` | ✓ | ✓ | Normal stress xx |
| `szz` | `stress_zz` | ✓ | ✓ | Normal stress zz |
| `syz` | `stress_yz` | ✓ | — | Shear stress yz |
| `sxz` | `stress_xz` | ✓ | — | Shear stress xz |
| `sxy` | `stress_xy` | ✓ | — | Shear stress xy |

The `m_*` entries are CPML memory variables. They are internal and are
filtered out by default in `available_fields()`.

## Defaults

- `default_source_fields`: `["sxx", "szz"]`
- `default_receiver_fields`: `["vx", "vy", "vz"]`

## Backend behavior

- Implemented as a first-order equation with `PartialDerivative` operators.
- **Eager only** in the current release — `supports_torch_binding()` returns
  `False`. Use `ElasticTTISG` for the compiled C++ / CUDA path.

## Torch binding

- `supports_torch_binding()` → `False`

## See also

- [ElasticTTISG](elastic_tti_sg.md) — staggered-grid TTI with compiled binding
  support.
- [Elastic](elastic.md) — isotropic elastic.
- The runnable wavefield comparison: [Elastic TTI wavefields](../../examples/elastic_tti_wavefields.md).
