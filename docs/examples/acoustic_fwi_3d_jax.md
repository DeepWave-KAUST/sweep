# 3D Acoustic FWI on Overthrust with JAX

Source file:

- `examples/FWI/3d/acoustic/jax/fwi_overthrust.py`

## What This Example Does

This is the JAX counterpart of
[3D Acoustic FWI · Torch](acoustic_fwi_3d_torch.md). It runs 3D acoustic
full-waveform inversion on the SEG/EAGE Overthrust model with
`PropJax(..., backend="jax")` and JAX-side autograd.

The script supports the same two training modes as the Torch version:

- **mini-batch stochastic FWI** (default) — each iteration draws `batchsize`
  shots and propagates them through `vmap`.
- **source-encoding FWI** (`--use-source-encoding`) — each iteration aggregates
  `batchsize` shots with random ±1 polarity and random time shifts into a
  single encoded super-shot, then runs one solver call per epoch.

JAX-specific notes:

- The propagator is `PropJax`. Set the backend with the standard
  `JAX_PLATFORMS` env var (`"cuda"` for GPU, `"cpu"` for CPU).
- Gradients are taken with `jax.grad` / `jax.value_and_grad`; the optimizer is
  a plain JAX loop (no `torch.optim`).
- Multi-GPU is via `jax.pmap`; see
  [JAX pmap Multi-GPU FWI](multi_gpu_jax.md) for the distributed variant.

## Main Components

The solver is built from:

- `equation`: `Acoustic3D(...)`
- `propagator`: `PropJax(..., backend="jax")`
- `wave`: a Ricker wavelet
- `sources` / `receivers`: same surface acquisition geometry as the Torch
  example
- `models`: the 3D velocity model `vp`

## Prepare the Overthrust Model Files

The same Overthrust `.npy` files are used as in the Torch version:

- `examples/models/overthrust/true_3d.npy`
- `examples/models/overthrust/smooth_3d.npy`

If you have not generated them yet, follow the *Prepare the Overthrust Model
Files* section of the [Torch page](acoustic_fwi_3d_torch.md) — both scripts
read from the same location.

## Run

Activate a JAX-enabled environment (with `jax[cuda12]` for GPU) and from the
repository root:

```bash
JAX_PLATFORMS=cuda python examples/FWI/3d/acoustic/jax/fwi_overthrust.py
```

For source-encoded FWI:

```bash
JAX_PLATFORMS=cuda python examples/FWI/3d/acoustic/jax/fwi_overthrust.py \
    --use-source-encoding
```

## Compare with the Torch Version

The two backends share the equation, propagator interface, geometry, and
training loop structure. They differ in:

| Aspect | Torch | JAX |
| --- | --- | --- |
| Propagator | `PropTorch` | `PropJax` |
| Autograd | `torch.autograd` | `jax.grad` / `jax.value_and_grad` |
| Compiled extension | `impl="c"` via `sweep._C` | not applicable |
| Multi-device | `torch.distributed` (DDP) | `jax.pmap` |

Use the JAX path when your downstream workflow is already JAX-based; use the
Torch path when you need the compiled C++ / CUDA extension via `impl="c"`.

## See Also

- [3D Acoustic FWI · Torch](acoustic_fwi_3d_torch.md)
- [JAX pmap Multi-GPU FWI](multi_gpu_jax.md)
- [Backends](../user-guide/backends.md)
