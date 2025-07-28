import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, tqdm, jax, optax

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../src')
from geophyai.rnn import RNNJax
from geophyai.equations import Acoustic
from geophyai.signal import ricker
from functools import partial
import matplotlib.pyplot as plt
from configure import *
from scipy.ndimage import gaussian_filter1d

save_path = 'acoustic3d_fwi_l2_jax'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

spatial_order = 4
nt = 2001
t = np.arange(0, nt*dt, dt)
dt = 0.001
dh = 20.0
nz, nx, ny = 256, 512, 512
batchsize = 1

true_model = np.ones((nz, ny, nx), dtype=np.float32) * 1500
true_model[nz//2:, :, :] = 2000

smooth_model = true_model.copy()
smooth_model = gaussian_filter1d(smooth_model, sigma=11, axis=0)

# true_model = np.load(true_path)
# smooth_model = np.load(smooth_path)
# shape = true_model.shape

shape = true_model.shape

wave = jnp.array(ricker(t-delay, f=fm))
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = RNNJax(Acoustic(spatial_order=spatial_order, backend='jax', dim=3), 
                shape=shape, 
                dev=None,
                dh=dh,
                dt=dt,
                source_type=['h1'],
                receiver_type=['h1'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=True, 
                ckpt_chunks = 500)

# Set the true model
model.set_parameters([jnp.array(true_model)])

# Geometry
sources = np.array([[nx//2, ny//2, 0]], dtype=np.int32).reshape(1, 3)  # (Number of shots, dimension)
xx, yy = np.meshgrid(np.arange(0, nx, 5), np.arange(0, ny, 5), indexing='xy')  # shape: (ny, nx)
zz = np.full_like(xx, fill_value=srcz)  # shape: (ny, nx)
receivers = np.stack([xx, yy, zz], axis=-1).reshape(1, xx.size, 3)  # (n, 3) with z, y, x
print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)

obs = model(wave, 
            sources, 
            receivers)

vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/acoustic_obs.png', dpi=300, bbox_inches='tight')

########## Inversion ##########
opt = optax.adam(lr, eps=1e-22)
@jax.jit
def update_fn(param, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    param = optax.apply_updates(param, updates)
    return param, opt_state

# Set the model
model.set_parameters([jnp.array(smooth_model)])
opt_state = opt.init(model.vp)

sources = jnp.array(sources)
receivers = jnp.array(receivers)

@jax.jit
def fwi_step(params, rand_shots):
    # @jax.jit
    def loss_fn(params, shot_nums):
        # Forward modeling
        syn = model(wave,
                    sources=sources[shot_nums], 
                    receivers=receivers[shot_nums], 
                    models=[params])
        _obs = obs[shot_nums]

        _loss_ = jnp.mean((syn-_obs)**2)

        return _loss_, (syn, _obs)
    # Compute the gradient
    (loss, data), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, rand_shots)
    return loss, gradients

LOSS = []
for epoch in tqdm.trange(epochs):

    key, subkey = random.split(key)
    rand_shots = random.randint(subkey, (batchsize,), 0, sources.shape[0])

    loss, grads = fwi_step(model.vp, rand_shots)
    model.vp, opt_state = update_fn(model.vp, grads, opt_state)

    print(f'Epoch: {epoch}, Loss: {loss}')
    LOSS.append(loss)
    # Save the model
    if epoch % show_every == 0:
        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model[:, ny//2, :], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Model')
        ax[1].imshow(model.vp[:, ny//2, :], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Model')
        vmin,vmax=np.percentile(grads[:, ny//2, :], [2, 98])
        ax[2].imshow(grads[:, ny//2, :], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')
        plt.close()