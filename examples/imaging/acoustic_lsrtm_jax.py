import sys, tqdm, os, optax
sys.path.append('../../src')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import jax.random as random
from geophyai.rnn import RNNJax as RNN
from geophyai.equations import Acoustic, AcousticLSRTM
from geophyai.signal import ricker
from geophyai.loss import CosineSimilarity, MSE
from geophyai.interpolation import resize
import numpy as np
import matplotlib.pyplot as plt
from configure import *
from scipy.ndimage import gaussian_filter

save_path = 'acoustic_lsrtm_jax'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

# Overwrite configures
fm = 20
spatial_order = 6
batchsize = 8
epochs = 31
ori_dh = 1.25 # m
target_dh = 12.5 # m
dt = 0.001
delay = 0.128
nt = 4000
free_surface = True

t = np.arange(0, nt*dt, dt)
true_model = np.load(true_path)[:,::2]  # Load the true model and downsample by 2 in x direction
ori_nz, ori_nx = true_model.shape

new_shape = (int(ori_nz*ori_dh/target_dh), int(ori_nx*ori_dh/target_dh))

# Interpolate the true model to the target resolution
if ori_dh != target_dh:
    true_model = resize(true_model, (ori_nz, ori_nx), new_shape)
    dh = target_dh

smooth_model = gaussian_filter(true_model, sigma=11)
true_ref = 2*(smooth_model-true_model)/smooth_model
nz, nx = true_model.shape
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
vmin, vmax = true_model.min(), true_model.max()
extent = [0, nx*dh, nz*dh, 0]
kwargs = {'vmin': vmin, 'vmax': vmax, 'cmap': 'seismic', 'aspect': 'auto', 'extent': extent}
axes[0].imshow(true_model, **kwargs)
axes[0].set_title('True Velocity Model')
axes[1].imshow(smooth_model, **kwargs)
axes[1].set_title('Migration Model')
plt.tight_layout()
plt.savefig(f'{save_path}/velocity_models.png', dpi=300, bbox_inches='tight')
plt.close()

shape = true_model.shape

nz, nx = shape

wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = RNN(Acoustic(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=None,
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            abcn=abcn, 
            free_surface=free_surface)

# Set the true model
model.set_parameters([jnp.array(true_model)])

# Geometry
src_x = jnp.arange(0,nx, src_step*4).reshape(-1, 1)
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
plt.savefig(f'{save_path}/acoustic_obs.png', dpi=300, bbox_inches='tight')

########## LSRTM inversion ##########
lsrtm = RNN(AcousticLSRTM(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=None,
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['sh1'],
            abcn=abcn, 
            free_surface=free_surface)

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
opt_state = opt.init(lsrtm.ref)

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

    loss, grads, (syn, _obs) = lsrtm_step(lsrtm.vp, lsrtm.ref, rand_shots)
    grad_ref = grads[-1]
    grad_ref = grad_ref/jnp.abs(grad_ref).max()
    lsrtm.ref, opt_state = update_fn(lsrtm.ref, grad_ref, opt_state)

    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    LOSS.append(loss)
    # Save the model
    if epoch % show_every == 0:
        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Velocity Model')
        inverted_ref = lsrtm.ref
        vmin, vmax = np.percentile(inverted_ref, [2, 98])
        ax[1].imshow(inverted_ref, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Reflectivity')
        vmin,vmax=np.percentile(grads[-1], [2, 98])
        ax[2].plot(inverted_ref[:, int(nx/2)], label='Inverted Reflectivity', c='black')
        ax[2].plot(true_ref[:, int(nx/2)], label='True Reflectivity', c='red')
        ax[2].set_title('Reflectivity Profile at x={}'.format(int(nx/2*dh)))
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')