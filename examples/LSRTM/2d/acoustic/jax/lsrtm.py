import sys, tqdm, os, optax
from pathlib import Path


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
sys.path.insert(0, str(EXAMPLES_DIR / "_shared"))

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import jax.random as random
from sweep.propagator.jax import PropJax
from sweep.equations import Acoustic, AcousticLSRTM
from sweep.signal import ricker

import numpy as np
import matplotlib.pyplot as plt
from configure_marmousi import *

save_path = Path(__file__).resolve().parent / 'acoustic_lsrtm_jax'
save_path.mkdir(parents=True, exist_ok=True)

np.random.seed(0)
key = random.PRNGKey(0)

# Overwrite configures
fm = 10
spatial_order = 8
batchsize = 8

t = np.arange(0, nt*dt, dt)
true_model = np.load(EXAMPLES_DIR / true_path)
smooth_model = np.load(EXAMPLES_DIR / smooth_path)
shape = true_model.shape

nz, nx = shape

wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig(save_path / 'ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = PropJax(Acoustic(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=None,
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            abcn=20,
            pml_type='cpmlr' ,
            free_surface=False)

# Set the true model
model.set_parameters([jnp.array(true_model)])

# Geometry
src_x = jnp.arange(0,nx, src_step).reshape(-1, 1)
src_z = jnp.ones_like(src_x)*srcz
sources = jnp.concatenate([src_x, src_z], axis=1) # (nshots, 2)
rec_x = jnp.arange(0, nx, rec_step).reshape(-1, 1)
rec_z = jnp.ones_like(rec_x)*recz
receivers = jnp.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)

# The following two lines are needed to ensure that the sources and receivers are JAX arrays
# and not NumPy arrays, which is required for JAX to work correctly.

obs = model.forward(wave, 
                    sources, 
                    receivers)

# Direct wave
model.set_parameters([jnp.array(np.ones_like(smooth_model)*1500)])
direct = model.forward(wave, 
                       sources, 
                       receivers)
obs = obs-direct
vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(save_path / 'acoustic_obs.png', dpi=300, bbox_inches='tight')

########## LSRTM inversion ##########
lsrtm = PropJax(AcousticLSRTM(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=None,
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['sh1'],
            abcn=20,
            pml_type='cpmlr' ,            
            free_surface=False)

# Set the model
lsrtm.set_parameters([jnp.array(smooth_model), # smoothed velocity model 
                      jnp.array(np.zeros_like(smooth_model))] # reflectivity (zero initial)
                      )

# criteria = MSE()
opt = optax.adam(0.01, eps=1e-22)
@jax.jit
def update_fn(param, grads, state):
    updates, state = opt.update(grads, state)
    param = optax.apply_updates(param, updates)
    return param, state
opt_state = opt.init(lsrtm.mp)

@jax.jit
def lsrtm_step(vp, ref, rand_shots):
    # @jax.jit
    def loss_fn(vp, ref, shot_nums):
        # Forward modeling
        syn = lsrtm(wave,
                    sources=sources[shot_nums], 
                    receivers=receivers[shot_nums], 
                    models=[vp, ref])
        _obs = obs[shot_nums]
        _loss_ = jnp.mean(1-optax.losses.cosine_similarity(syn.reshape(_obs.shape), _obs, axis=1, epsilon=1e-8))

        return _loss_, (syn, _obs)
    # Compute the gradient
    (loss, data), gradients = jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True)(vp, ref, rand_shots)
    return loss, gradients, data

LOSS = []
for epoch in tqdm.trange(epochs):

    key, subkey = random.split(key)
    rand_shots = random.randint(subkey, (batchsize,), 0, sources.shape[0])

    loss, grads, (syn, _obs) = lsrtm_step(lsrtm.vp, lsrtm.mp, rand_shots)
    grad_ref = grads[-1]
    grad_ref = grad_ref/grad_ref.max()
    lsrtm.mp, opt_state = update_fn(lsrtm.mp, grad_ref, opt_state)

    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    LOSS.append(loss)
    # Save the model
    if epoch % show_every == 0:
        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Velocity Model')
        inverted_ref = lsrtm.mp
        vmin, vmax = np.percentile(inverted_ref, [2, 98])
        ax[1].imshow(inverted_ref, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Reflectivity')
        vmin,vmax=np.percentile(grads[-1], [2, 98])
        print('Gradient:', vmin, vmax)
        ax[2].imshow(grads[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        plt.tight_layout()
        plt.savefig(save_path / f'epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(save_path / 'loss.png', dpi=300, bbox_inches='tight')
