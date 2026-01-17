import sys, tqdm, os, jax, optax
import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../src')
from sweep.propagator.jax import PropJax
from sweep.equations import Acoustic
from sweep.signal import ricker
from functools import partial
import matplotlib.pyplot as plt
from configure import *

save_path = 'acoustic_fwi_l2_jax'
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
model = PropJax(Acoustic(spatial_order=spatial_order, backend='jax'), 
                shape=shape, 
                dev=None,
                dh=dh,
                dt=dt,
                source_type=['h1'],
                receiver_type=['h1'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=False,
                pml_type='cpmlr')

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
receivers = jnp.array(receivers)[0:1]

@jax.jit
def fwi_step(key, params, rand_shots):

    def loss_fn(params, shot_nums):
        nshots = shot_nums.shape[0]

        subkeys = jax.random.split(key, nshots * 2)
        keys_p, keys_tau = subkeys[:nshots], subkeys[nshots:]

        def process_shot(carry, inputs):
            num, key_p, key_tau = inputs
            # Random polarity
            p = jax.random.randint(key_p, shape=(), minval=0, maxval=2)  # 0或1
            # Random time shift
            time_shifts = jnp.arange(0, 0.2 * nt, dtype=jnp.int32)
            tau_s = jax.random.choice(key_tau, time_shifts)
            mask = jnp.arange(nt) < tau_s

            rolled_wave = (-1)**p * jnp.roll(wave, tau_s, axis=0)
            rolled_wave = rolled_wave * (~mask)
            rolled_obs = (-1)**p * jnp.roll(obs[num], tau_s, axis=0)
            rolled_obs = rolled_obs * (~mask.reshape(-1, 1, 1))

            return carry, (rolled_wave, rolled_obs)

        _, (super_wave, super_obs) = jax.lax.scan(
            process_shot,
            None,
            (shot_nums, keys_p, keys_tau)
        )

        super_wave = jnp.stack(super_wave)
        super_obs = jnp.stack(super_obs)

        syn = model(super_wave,
                    sources=sources[shot_nums],
                    receivers=receivers,
                    models=[params],
                    source_encoding=True)
        _obs = jnp.sum(super_obs, axis=0)
        _loss_ = jnp.mean((syn - _obs)**2)
    
        # # Forward modeling
        # syn = model(wave,
        #             sources=sources[shot_nums], 
        #             receivers=receivers, 
        #             models=[params], 
        #             source_encoding=True)
        
        # _obs = jnp.sum(obs[shot_nums], axis=0)
        # _loss_ = jnp.mean((syn-_obs)**2)

        return _loss_, (syn, _obs)
    # Compute the gradient
    (loss, data), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params, rand_shots)
    return loss, gradients, data

LOSS = []
for epoch in tqdm.trange(epochs):

    key, subkey = random.split(key)
    rand_shots = random.permutation(subkey, sources.shape[0])[:batchsize]

    loss, grads, (syn, obs) = fwi_step(subkey, model.vp, rand_shots)
    model.vp, opt_state = update_fn(model.vp, grads, opt_state)

    print(f'Epoch: {epoch}, Loss: {loss}')
    LOSS.append(loss)
    # Save the model
    if epoch % show_every == 0:

        fig, axes = plt.subplots(1, 2, figsize=(6, 8))
        vmin, vmax = np.percentile(obs.squeeze(), [2, 98])
        axes[0].imshow(obs.squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
        axes[0].set_title('Observed Data')
        axes[1].imshow(syn.squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
        axes[1].set_title('Synthetic Data')
        plt.tight_layout()
        plt.savefig(f'{save_path}/data_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Model')
        ax[1].imshow(model.vp, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Model')
        vmin,vmax=np.percentile(grads, [2, 98])
        ax[2].imshow(grads, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')
        plt.close()