# JAX Multi-GPU Examples

This directory contains JAX multi-GPU examples based on `jax.pmap`.

## 2D Acoustic FWI on Marmousi

Script:

- `fwi_marmousi_pmap.py`

Prepare the Marmousi `.npy` files first:

```bash
python3 examples/models/marmousi/download_marmousi.py --extract
python3 examples/models/marmousi/extract_model_segy.py
python3 examples/models/marmousi/convert_segy_to_npy.py
python3 examples/models/marmousi/prepare_fwi_models.py \
  --input examples/models/marmousi/npy/vp_1p25m.npy \
  --source-dh 1.25 \
  --target-dh 25.0 \
  --radii 8,8 \
  --passes 3
```

Run on all local JAX devices:

```bash
python3 examples/multi-gpu/jax/fwi_marmousi_pmap.py
```

The global shot batch is reshaped to `(n_devices, shots_per_device)`. Each
device computes one shard, `jax.lax.psum` sums gradients across devices, and
Optax applies the same update to every replicated model copy.
