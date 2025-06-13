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
from geophyai.equations.operator_jax import laplace
from functools import partial
import matplotlib.pyplot as plt
from configure import *

save_path = 'acoustic_fwi_l2_jax_adj'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

t = np.arange(0, nt*dt, dt)
true_model = np.load(true_path)
smooth_model = np.load(smooth_path)
shape = true_model.shape

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
                source_type=['h1'],
                receiver_type=['h1'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=False)

# Set the true model
model.set_parameters([jnp.array(true_model)])

# Geometry
src_x = np.arange(0,nx, src_step).reshape(-1, 1)
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

def fwi_step(vp, rand_shots):
    # Forward modeling
    syn, fwf = model(wave,
                     sources=sources[rand_shots], 
                     receivers=receivers[rand_shots], 
                     return_wavefield=True,
                     models=[vp])
    
    def loss_fn(_syn, _obs):
        # Compute the loss
        return jnp.mean((_syn - _obs)**2)
    
    loss, adj = jax.value_and_grad(loss_fn)(syn, obs[rand_shots]) # (nshots, nrec, nt)

    # Adjoint modeling adj (ns, nrec, nt, 1)-> (ns, nrec, nt)
    non, bwf = model(adj.transpose(0,2,3,1).squeeze(), # (ns, nrec, nt), must be nt last and time reversed
                     sources=receivers[rand_shots],
                     receivers=receivers[rand_shots],
                     models=[vp],
                     return_wavefield=True,
                     adj=True)

    kernel = model.equation.kernel
    # Compute the laplace of the forward wavefield
    def body_fn(tstep, d):
        return d.at[tstep].set(laplace(d[tstep], model.dh, kernel)*model.models_padded[0]**2)

    fwf = jax.lax.fori_loop(0, nt, body_fn, fwf) # (ns, nrec, nz, nx)
    grad = jnp.sum(fwf*bwf[::-1]/model.models_padded[0]**3, axis=(0))#/np.prod(adj.shape[1:])

    return loss, model.crop(grad), adj

LOSS = []
for epoch in tqdm.trange(epochs):

    key, subkey = random.split(key)
    rand_shots = random.randint(subkey, (batchsize,), 0, sources.shape[0])

    loss, grads, adj = fwi_step(model.vp, rand_shots)

    grads = jnp.sum(grads, axis=(0,1)) # (ns, 1, nz, nx) -> (nz, nx)
    model.vp, opt_state = update_fn(model.vp, grads, opt_state)

    print(f'Epoch: {epoch}, Loss: {loss}')
    LOSS.append(loss)
    # Save the model
    if epoch % show_every == 0:
        np.save(f'{save_path}/grad_e{epoch}.npy', grads)
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        vmin, vmax = np.percentile(adj[0], [2, 98])
        _ax = ax.imshow(adj[0].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
        plt.colorbar(_ax, ax=ax)
        ax.set_title('Adjoint Wavefield')
        plt.tight_layout()
        fig.savefig(f'{save_path}/adj_wavefield_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Model')
        ax[1].imshow(model.vp, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Model')
        vmin,vmax=np.percentile(grads, [2, 98])
        _ax = ax[2].imshow(grads, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        plt.colorbar(_ax, ax=ax[2])
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')
        plt.close()