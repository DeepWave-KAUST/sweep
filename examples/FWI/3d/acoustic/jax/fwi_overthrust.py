import os
from pathlib import Path
import argparse
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def find_examples_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if candidate.name == "examples":
            return candidate
    raise RuntimeError("Could not locate the examples directory.")


EXAMPLES_DIR = find_examples_root()
sys.path.insert(0, str(EXAMPLES_DIR / "_shared"))
from bootstrap import configure_example_imports

IMPORT_MODE = configure_example_imports(EXAMPLES_DIR)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm

import configure_overthrust as shared_config
from fwi3d_overthrust import (
    build_geometry,
    build_wavelet,
    load_models,
    save_observed_figure,
    save_progress_figure,
    save_wavelet_figure,
)
from sweep.equations import Acoustic3D
from sweep.propagator.jax import PropJax
from sweep.signal import ricker


CONFIG = shared_config.get_config("fwi_3d_acoustic_jax")


def build_solver(shape, cfg):
    equation = Acoustic3D(
        spatial_order=cfg["spatial_order"],
        backend="jax",
    )

    return PropJax(
        equation,
        shape=shape,
        dev=None,
        dh=cfg["dh"],
        dt=cfg["dt"],
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=cfg["abcn"],
        free_surface=cfg["free_surface"],
        use_ckpt=cfg["use_ckpt"],
        ckpt_chunks=cfg["ckpt_chunks"],
        pml_type="cpmlr",
    )


def forward_batched(solver, wave, sources, receivers, models, shot_batchsize):
    nshots = sources.shape[0]
    shot_batchsize = max(1, min(int(shot_batchsize), nshots))
    obs_batches = []

    for start in tqdm.trange(0, nshots, shot_batchsize, desc="Forward shots", unit="batch"):
        stop = min(start + shot_batchsize, nshots)
        batch_obs = solver(
            wave,
            sources[start:stop],
            receivers[start:stop],
            models=models,
        )
        obs_batches.append(np.asarray(batch_obs))

    return np.concatenate(obs_batches, axis=0)


def parse_args():
    parser = argparse.ArgumentParser(description="3D acoustic FWI on the Overthrust model for the JAX propagator.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    return parser.parse_args()


def main():
    parse_args()
    cfg = CONFIG.copy()
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)

    true_model, init_model = load_models(EXAMPLES_DIR, cfg)
    shape = true_model.shape
    solver = build_solver(shape, cfg)

    _, wave = build_wavelet(cfg, ricker)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("prepared model shape:", shape)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    true_vp = jnp.array(true_model)
    forward_batchsize = min(cfg["forward_batchsize"], sources.shape[0])
    if forward_batchsize >= sources.shape[0]:
        obs = np.asarray(solver(wave, sources, receivers, models=[true_vp]))
    else:
        obs = forward_batched(
            solver,
            wave,
            sources,
            receivers,
            models=[true_vp],
            shot_batchsize=forward_batchsize,
        )
    save_observed_figure(obs, output_dir)

    wave_jax = jnp.asarray(wave)
    sources_jax = jnp.asarray(sources)
    receivers_jax = jnp.asarray(receivers)
    obs_jax = jnp.asarray(obs)

    optimizer = optax.adam(cfg["lr"], eps=1e-22)
    vp = jnp.array(init_model)
    opt_state = optimizer.init(vp)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    @jax.jit
    def update_fn(params, grads, state):
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    @jax.jit
    def fwi_step(params, shot_idx, wavelet, sources_all, receivers_all, obs_all):
        def loss_fn(current_params, current_shots, current_wavelet, all_sources, all_receivers, all_obs):
            syn = solver(
                current_wavelet,
                sources=all_sources[current_shots],
                receivers=all_receivers[current_shots],
                models=[current_params],
            )
            return jnp.mean((syn - all_obs[current_shots]) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params, shot_idx, wavelet, sources_all, receivers_all, obs_all)
        return loss, grads

    for epoch in tqdm.trange(cfg["epochs"]):
        shot_idx = np.random.choice(nshots, size=batchsize, replace=False).astype(np.int32)
        loss, grads = fwi_step(vp, shot_idx, wave_jax, sources_jax, receivers_jax, obs_jax)
        vp, opt_state = update_fn(vp, grads, opt_state)

        loss_value = float(loss)
        losses.append(loss_value)
        print(f"Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            save_progress_figure(
                true_model,
                np.asarray(vp),
                np.asarray(grads),
                losses,
                epoch,
                cfg,
                output_dir,
            )


if __name__ == "__main__":
    main()
