# Torch Multi-GPU Examples

This directory contains Torch distributed examples launched with `torchrun`.

## 2D Acoustic FWI on Marmousi

Script:

- `fwi_marmousi_dist.py`

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

Run on two GPUs:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/multi-gpu/torch/fwi_marmousi_dist.py --backend cuda
```

The script splits each global shot batch across ranks, sums model gradients
with `torch.distributed.all_reduce`, and applies the same optimizer step on
every rank. Rank 0 writes figures to `multi_gpu_acoustic_fwi_cuda/`.
