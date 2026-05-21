# Multi-GPU

> :material-github: **Source on GitHub** &mdash; [`examples/multi-gpu/`](https://github.com/DeepWave-KAUST/sweep/tree/main/examples/multi-gpu) (clone, run, modify)

Source directory:

- [`examples/multi-gpu/`](https://github.com/DeepWave-KAUST/sweep/tree/main/examples/multi-gpu)

The multi-GPU examples currently cover 2D acoustic FWI on the Marmousi model.
Each backend has its own page:

- [Torch Distributed Multi-GPU FWI](multi_gpu_torch.md)
- [JAX pmap Multi-GPU FWI](multi_gpu_jax.md)

Both examples use data parallelism over shots: each GPU receives a subset of
the global shot batch, computes local synthetic data and gradients, then
participates in a backend-specific gradient reduction.

## Prepare the Marmousi Model Files

Both examples read:

- [`examples/models/marmousi/true.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/true.npy)
- [`examples/models/marmousi/smooth.npy`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/models/marmousi/smooth.npy)

Generate them before running either multi-GPU example:

```bash
python3 examples/models/marmousi/download_marmousi.py --extract
python3 examples/models/marmousi/extract_model_segy.py
python3 examples/models/marmousi/convert_segy_to_npy.py
python3 examples/models/marmousi/prepare_fwi_models.py \
  --input examples/models/marmousi/npy/vp_1p25m.npy \
  --source-dh 1.25 \
  --target-dh 12.5 \
  --radii 16,16 \
  --passes 3
```
