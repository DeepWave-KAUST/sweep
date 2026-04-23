# AcousticLSRTM3D

```python
class AcousticLSRTM3D(
    spatial_order=4,
    device="cpu",
    backend="torch",
)
```

- Implementation:
  - `src/sweep/equations/acoustic_lsrtm3d.py`

Linearized 3D acoustic equation for LSRTM-style modeling with a background
velocity model `vp` and a reflectivity model `mp`.

## Models

- `vp`: background velocity
- `mp`: reflectivity / perturbation model

## Receiver Field

- `sh1`: scattered pressure-like wavefield

## Backends

- eager PyTorch
- CUDA propagator backend
