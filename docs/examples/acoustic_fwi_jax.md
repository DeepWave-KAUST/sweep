# 2D Acoustic FWI on Marmousi with JAX

Source file:

- `examples/FWI/2d/acoustic/jax/fwi_marmousi.py`

## What This Example Does

This example runs acoustic full-waveform inversion with the JAX propagator.

The script:

1. loads a true velocity model and a smooth initial model
2. builds a `PropJax(Acoustic(...))` solver
3. generates observed data from the true model
4. inverts the initial model by matching synthetic and observed gathers

## Main Components

The solver is built from:

- `equation`: `Acoustic(..., backend="jax")`
- `propagator`: `PropJax(...)`
- `wave`: a Ricker wavelet
- `sources`: regularly sampled source coordinates
- `receivers`: regularly sampled receiver coordinates
- `models`: the velocity model `vp`

## Prepare the Marmousi Model Files

This example reads the same prepared Marmousi acoustic files as the Torch FWI
example:

- `examples/models/marmousi/true.npy`
- `examples/models/marmousi/smooth.npy`

Generate them before running the example:

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

## Entry Point

Run the example with:

```bash
python3 examples/FWI/2d/acoustic/jax/fwi_marmousi.py
```

The script keeps its runtime settings in one `CONFIG` dictionary.

## Key Configuration

Important entries include:

- `nt`, `dt`: temporal sampling
- `dh`: spatial sampling
- `spatial_order`: finite-difference order
- `abcn`: absorbing boundary width
- `src_step`, `rec_step`: acquisition sampling in the x direction
- `true_model`, `init_model`: `.npy` files loaded from `examples/models/`
- `epochs`, `batchsize`, `lr`: inversion hyperparameters
- `use_ckpt`: whether JAX chunk rematerialization is enabled

## Solver Setup

The equation is created with the JAX backend:

```python
equation = Acoustic(
    spatial_order=cfg["spatial_order"],
    backend="jax",
)
```

Shared propagator arguments are collected first:

```python
prop_kwargs = dict(
    shape=shape,
    dev=None,
    dh=cfg["dh"],
    dt=cfg["dt"],
    source_type=["h1"],
    receiver_type=["h1"],
    abcn=cfg["abcn"],
    free_surface=cfg["free_surface"],
    use_ckpt=cfg["use_ckpt"],
    pml_type="cpmlr",
)
```

Then the solver is created as:

```python
solver = PropJax(equation, **prop_kwargs)
```

## Geometry

The example uses a fixed-depth surface acquisition:

- sources are placed every `src_step` grid points
- receivers are placed every `rec_step` grid points
- all sources use the same source depth `srcz`
- all receivers use the same receiver depth `recz`

The final array shapes are:

- `sources`: `(nshots, 2)`
- `receivers`: `(nshots, nreceivers, 2)`

## Inversion Workflow

Observed data is generated first from the true model, then the inversion updates
the smooth model with `optax.adam`.

At each iteration, the script:

1. selects a random subset of shots
2. computes synthetic data
3. evaluates the L2 data-misfit loss
4. gets gradients with `jax.value_and_grad`
5. updates the model with `optax`

## Outputs

The script creates an output directory under `examples/` and saves:

- `ricker.png`
- `observed_data.png`
- `loss.png`
- `epoch_XXXX.png`: includes the true model, the current inverted model, and the current gradient

The output directory is:

- `acoustic_fwi_jax`

## Running the Example

Step 1. Prepare the Marmousi `.npy` files listed above if they do not already
exist.

Step 2. Run the JAX Marmousi script from the repository root.

```bash
python3 examples/FWI/2d/acoustic/jax/fwi_marmousi.py
```

Step 3. Check `acoustic_fwi_jax` for the saved wavelet, observed data, loss, and epoch figures.

Notes:

- the script disables JAX preallocation with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `use_ckpt` controls JAX rematerialization for reduced memory usage
