import sys, tqdm, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
sys.path.append('../src')
import optax, jax
import jax.numpy as jnp
from functools import partial

from sweep.propagator.jax import PropJax
from sweep.equations import Elastic
from sweep.signal import ricker
import numpy as np
import matplotlib.pyplot as plt
from configure_ot import *
np.random.seed(0)


def jax_record_to_standard(record):
    return np.array(record)

save_path = 'elastic_jax_ot'
if not os.path.exists(save_path):
    os.makedirs(save_path)

t = np.arange(0, int(nt)*dt, dt)

vp_true = np.load(true_path)#[::2, ::2]
vs_true = vp_true/1.732
rho_true = np.ones_like(vp_true)*1000

vp_smooth = np.load(smooth_path)#[::2, ::2]

shape = vp_true.shape
extent = [0, shape[1]*dh, shape[0]*dh, 0]

nz, nx = shape

wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = PropJax(Elastic(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=None, 
            abcn=abcn, 
            dh=dh,
            dt=dt,
            source_type=['sxx', 'szz'],
            receiver_type=['vx', 'vz'],
            free_surface=False, 
            pml_type='cpmls',
            use_ckpt=False)

# Set the true model, the order of the parameters should be 
# the same as the model names in func <geophyai.equations.elastic.models>
model.set_parameters([jnp.array(vp_true), 
                      jnp.array(vs_true), 
                      jnp.array(rho_true)])

# Geometry
src_x = jnp.arange(0,nx, src_step*2).reshape(-1, 1)
src_z = jnp.ones_like(src_x)*srcz
sources = jnp.concatenate([src_x, src_z], axis=1) # (nshots, 2)
rec_x = jnp.arange(0, nx, rec_step*2).reshape(-1, 1)
rec_z = jnp.ones_like(rec_x)*recz
receivers = jnp.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)


obs = model.forward(wave, 
                    sources, 
                    receivers)

# vmin, vmax = np.percentile(obs[-1][...,0], [2, 98])
# plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'{save_path}/elastic_vz.png', dpi=300, bbox_inches='tight')
# plt.close()

# ########## Inversion ##########
opts = [optax.adam(lr, eps=1e-22), optax.adam(lr/1.73, eps=1e-22), optax.adam(0., eps=1e-22)]
@partial(jax.jit, static_argnums=(3,))
def update_fn(param, grads, opt_state, opt):
    updates, opt_state = opt.update(grads, opt_state)
    param = optax.apply_updates(param, updates)
    return param, opt_state

# Set the model
init_vp = jnp.array(vp_smooth)
init_vs = jnp.array(vp_smooth/1.732)
# init_vp = init_vp.at[:2, :].set(jnp.array(vp_true[:2, :]))
# init_vs = init_vs.at[:2, :].set(jnp.array(vs_true[:2, :]))
model.set_parameters([init_vp, 
                      init_vs, 
                      jnp.array(rho_true)])
opt_states = [opt.init(param) for param, opt in zip(model.parameters(), opts)]

shot_rng = np.random.RandomState(0)
shot_schedule = [
    jnp.array(shot_rng.choice(int(sources.shape[0]), size=batchsize, replace=False), dtype=jnp.int32)
    for _ in range(epochs)
]

@jax.jit
def fwi_step(vp, vs, rho, rand_shots):
    # @jax.jit
    def loss_fn(vp, vs, rho, shot_nums):
        # Forward modeling
        syn = model(wave,
                    sources=sources[shot_nums], 
                    receivers=receivers[shot_nums],
                    # receivers=receivers[0:1], 
                    # source_encoding=True,
                    models=[vp, vs, rho])
        # _obs = jnp.sum(obs[shot_nums], axis=0)
        _obs = obs[shot_nums]
        _loss_ = jnp.sum((syn-_obs)**2)

        return _loss_, (syn, _obs)
    # Compute the gradient
    (loss, data), gradients = jax.value_and_grad(loss_fn, argnums=(0,1,2), has_aux=True)(vp, vs, rho, rand_shots)
    return loss, gradients

LOSS = []
for epoch in tqdm.trange(epochs):

    rand_shots = np.array([0])#shot_schedule[epoch]

    loss, grads = fwi_step(model.vp, model.vs, model.rho, rand_shots)
    model.vp, opt_states[0] = update_fn(model.vp, grads[0], opt_states[0], opts[0])
    model.vs, opt_states[1] = update_fn(model.vs, grads[1], opt_states[1], opts[1])
    model.rho, opt_states[2] = update_fn(model.rho, grads[2], opt_states[2], opts[2])

    # model.vp = model.vp.at[:2, :].set(jnp.array(vp_true[:2, :]))
    # model.vs = model.vs.at[:2, :].set(jnp.array(vs_true[:2, :]))

    print(f'Epoch: {epoch}, Loss: {loss}')
    LOSS.append(loss)


    # Save the model
    if epoch % show_every == 0:
        vmin_vp, vmax_vp = vp_true.min(), vp_true.max()
        vmin_vs, vmax_vs = vs_true.min(), vs_true.max()
        fig, axes = plt.subplots(3, 2, figsize=(8, 9))
        show_data = [vp_true, vs_true, 
                     model.vp, model.vs, 
                     grads[0], grads[1]]
        titles = ['True Vp', 'True Vs', 'Inverted Vp', 'Inverted Vs', 'Gradient Vp', 'Gradient Vs']
        for ax, data, title in zip(axes.ravel(), show_data, titles):
            if 'vp' in title.lower():
                vmin, vmax = vmin_vp, vmax_vp
            else:
                vmin, vmax = vmin_vs, vmax_vs
            if 'gradient' in title.lower():
                vmin, vmax = np.percentile(data, [2, 98])
            plt.colorbar(ax.imshow(data, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto'))
            ax.set_title(title)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(LOSS, label='Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss History')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/loss_history.png', dpi=300, bbox_inches='tight')
        plt.close()

        
