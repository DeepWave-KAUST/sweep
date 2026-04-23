# 3D Acoustic LSRTM on Overthrust with Torch

- Script:
  - `examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py`
- Shared utilities:
  - `examples/_shared/configure_overthrust.py`
  - `examples/_shared/fwi3d_overthrust.py`

This example runs 3D acoustic least-squares reverse time migration (LSRTM) on
the Overthrust model with one script that supports both the eager PyTorch
backend and the CUDA propagator backend.

The workflow is:

1. load the true and smoothed Overthrust velocity models
2. generate scattered observed data by subtracting the background response from
   the true-model response
3. build an `AcousticLSRTM3D` solver on either the eager or CUDA backend
4. optimize the reflectivity model `mp` while keeping the background velocity
   fixed

## Run

Eager:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend eager
```

CUDA full-memory:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend cuda --cuda-memory full
```

CUDA boundary saving:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend cuda --cuda-memory bs
```

CUDA checkpointing:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend cuda --cuda-memory ckpt
```

CUDA recursive checkpointing:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend cuda --cuda-memory recursive
```

## Notes

- sources inject into `h1`
- receivers read from `sh1`
- models are passed as `[vp, mp]`
- CUDA supports the same memory strategies used by the 2D LSRTM example:
  full wavefield storage, boundary saving, chunk checkpointing, and recursive
  checkpointing
