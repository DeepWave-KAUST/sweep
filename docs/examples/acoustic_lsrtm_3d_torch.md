# 3D Acoustic LSRTM on Overthrust with Torch

- Script:
  - `examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py`
- Shared utilities:
  - `examples/_shared/configure_overthrust.py`
  - `examples/_shared/fwi3d_overthrust.py`

This example runs 3D acoustic least-squares reverse time migration (LSRTM) on
the Overthrust model with one script that supports both the eager PyTorch
implementation and the compiled c implementation on CUDA.

The workflow is:

1. load the true and smoothed Overthrust velocity models
2. generate scattered observed data by subtracting the background response from
   the true-model response
3. build an `AcousticLSRTM3D` solver with either the eager or  implementation
4. optimize the reflectivity model `mp` while keeping the background velocity
   fixed

## Prepare the Overthrust Model Files

This example reads:

- `examples/models/overthrust/true_3d.npy`
- `examples/models/overthrust/smooth_3d.npy`

Generate them from the official SEG/EAGE Overthrust archive before running the
example:

```bash
python3 examples/models/overthrust/download_3d_overthrust.py --extract
python3 examples/models/overthrust/convert_3d_overthrust_vites_to_npy.py
python3 examples/models/overthrust/make_smooth_model.py \
  --input examples/models/overthrust/true_3d.npy \
  --output examples/models/overthrust/smooth_3d.npy \
  --radii 6,6,6 \
  --passes 3
```

## Run

Prepare the Overthrust `.npy` files listed above if they do not already exist,
then choose an implementation.

Eager:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend torch --impl eager
```

 CUDA full-memory:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend torch --impl c --device cuda --cuda-memory full
```

 CUDA boundary saving:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend torch --impl c --device cuda --cuda-memory bs
```

 CUDA checkpointing:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend torch --impl c --device cuda --cuda-memory ckpt
```

 CUDA recursive checkpointing:

```bash
python3 examples/LSRTM/3d/acoustic/torch/lsrtm_overthrust.py --backend torch --impl c --device cuda --cuda-memory recursive
```

## Notes

- sources inject into `h1`
- receivers read from `sh1`
- models are passed as `[vp, mp]`
- the c CUDA path supports the same memory strategies used by the 2D LSRTM example:
  full wavefield storage, boundary saving, chunk checkpointing, and recursive
  checkpointing
