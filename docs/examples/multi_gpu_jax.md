# JAX pmap Multi-GPU FWI

> :material-github: **Source on GitHub** &mdash; [`examples/multi-gpu/jax/`](https://github.com/DeepWave-KAUST/sweep/tree/main/examples/multi-gpu/jax) (clone, run, modify)

Source file:

- [`examples/multi-gpu/jax/fwi_marmousi_pmap.py`](https://github.com/DeepWave-KAUST/sweep/blob/main/examples/multi-gpu/jax/fwi_marmousi_pmap.py)

This script is a data-parallel version of the 2D acoustic Marmousi JAX FWI
example. It uses one host process and `jax.pmap` over all local JAX devices.

At each iteration:

1. the velocity model is replicated across devices
2. the global shot batch is reshaped to `(n_devices, shots_per_device)`
3. each device computes synthetic data and gradients for its local shots
4. `jax.lax.psum` sums loss and gradients across devices
5. Optax applies the same update to every replicated model copy

## How the Split Works

The model is replicated directly to all local devices:

```python
devices = jax.local_devices()
ndevices = len(devices)
mesh = Mesh(np.asarray(devices), ("devices",))
sharding = NamedSharding(mesh, P("devices"))
params = jax.device_put(np.stack([init_model] * ndevices), sharding)
```

At each iteration, the selected global shot batch is split on the host, then
each shard is placed directly on its target device:

```python
shot_idx = shot_idx.reshape(ndevices, shots_per_device)
sources_batch = sources[shot_idx]
receivers_batch = receivers[shot_idx]
obs_batch = obs[shot_idx]
sources_shards = jax.device_put(sources_batch, sharding)
receivers_shards = jax.device_put(receivers_batch, sharding)
obs_shards = jax.device_put(obs_batch, sharding)
```

Inside the pmapped step, each device receives only its local source, receiver,
and observed-data shard:

```python
syn = solver(
    wavelet,
    sources=sources_shard,
    receivers=receivers_shard,
    models=[params_shard],
)
```

The local gradients are converted to a global mean gradient with `lax.psum`:

```python
loss_sum, grad_sum = jax.value_and_grad(local_loss_sum)(params_shard)
global_numel = jax.lax.psum(local_numel, axis_name="devices")
grad = jax.lax.psum(grad_sum, axis_name="devices") / global_numel
```

Because `pmap` requires equal shard shapes, the global `--batchsize` must be
divisible by the number of local JAX devices. With `batchsize=8`, the split is:

- 2 devices: 4 shots per device
- 4 devices: 2 shots per device
- 8 devices: 1 shot per device

Avoid creating full training arrays with `jnp.asarray(obs)` before `pmap`.
That puts the complete array on JAX's default device, usually GPU 0, and then
`pmap` creates additional per-device copies. The example keeps the full
observed data on the host as a NumPy array and only transfers the selected
per-device shot shards for the current iteration.

## Run

Run the JAX pmap example on all local JAX devices:

```bash
python3 examples/multi-gpu/jax/fwi_marmousi_pmap.py
```

## Outputs

Outputs are saved under:

- [`examples/multi-gpu/jax/multi_gpu_acoustic_fwi_jax/`](https://github.com/DeepWave-KAUST/sweep/tree/main/examples/multi-gpu/jax/multi_gpu_acoustic_fwi_jax)

Saved files include:

- `ricker.png`
- `observed_data.png`
- `loss.png`
- `epoch_XXXX.png`

The progress panel includes the true model, inverted model, and gradient.
