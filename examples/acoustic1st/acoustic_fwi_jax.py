import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, tqdm, jax, optax

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('/ibex/user/wangs0j/aiseismic/src')
from geophyai.rnn import RNNJax
from geophyai.equations import Acoustic1st as Acoustic
from geophyai.signal import ricker
from functools import partial
import matplotlib.pyplot as plt
from configure import *

save_path = 'acoustic_fwi_l2_jax'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

t = np.arange(0, nt*dt, dt)
true_vp = np.load(true_path)
true_rho = 0.31*true_vp**0.25*1000
smooth_vp = np.load(smooth_path)
smooth_rho = 0.31*smooth_vp**0.25*1000
shape = true_vp.shape

nz, nx = shape

wave = jnp.array(ricker(t-delay, f=fm))
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = RNNJax(Acoustic(spatial_order=spatial_order, backend='jax'), 
                shape=shape, 
                dev=None,
                dh=dh,
                dt=dt,
                source_type=['p'],
                receiver_type=['p'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=False)

# Set the true model
model.set_parameters([jnp.array(true_vp), jnp.array(true_rho)])

# Geometry
src_x = np.arange(0,nx, src_step*10).reshape(-1, 1)
src_z = np.ones_like(src_x)*srcz
sources = np.concatenate([src_x, src_z], axis=1) # (nshots, 2)
rec_x = np.arange(0, nx, rec_step).reshape(-1, 1)
rec_z = np.ones_like(rec_x)*recz
receivers = np.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

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
exit()
########## Inversion ##########
opt = optax.adam(lr, eps=1e-22)
@jax.jit
def update_fn(param, grads, opt_state):
    updates, opt_state = opt.update(grads, opt_state)
    param = optax.apply_updates(param, updates)
    return param, opt_state

# Set the model
model.set_parameters([jnp.array(smooth_vp), jnp.array(smooth_rho)])
opt_state = opt.init([model.vp, model.rho])

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
                    models=params)
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

    loss, grads = fwi_step([model.vp, model.rho], rand_shots)
    params, opt_state = update_fn([model.vp, model.rho], grads, opt_state)
    model.vp, model.rho = params
    print(f'Epoch: {epoch}, Loss: {loss}')
    LOSS.append(loss)
    # Save the model
    if epoch % show_every == 0:
        vmin, vmax = true_vp.min(), true_vp.max()
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0,0].imshow(true_vp, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0,0].set_title('True Vp')
        ax[0,1].imshow(model.vp, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0,1].set_title('Inverted Vp')
        vmin,vmax=np.percentile(grads, [2, 98])
        ax[0,2].imshow(grads[0], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0,2].set_title('Vp Gradient')
        ax[1,0].imshow(true_rho, vmin=true_rho.min(), vmax=true_rho.max(), cmap='seismic', aspect='auto', extent=extent)
        ax[1,0].set_title('True Rho')
        ax[1,1].imshow(model.rho, vmin=true_rho.min(), vmax=true_rho.max(), cmap='seismic', aspect='auto', extent=extent)
        ax[1,1].set_title('Inverted Rho')
        vmin,vmax=np.percentile(grads, [2, 98])
        ax[1,2].imshow(grads[1], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[1,2].set_title('Rho Gradient')
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')
        plt.close()