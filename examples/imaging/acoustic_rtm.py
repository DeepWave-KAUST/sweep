import sys, tqdm, os, optax
sys.path.append('../../src')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
import jax.random as random
from geophyai.rnn import RNNJax as RNN
from geophyai.equations import Acoustic
from geophyai.signal import ricker
from geophyai.loss import CosineSimilarity, MSE
from geophyai.interpolation import resize
import numpy as np
import matplotlib.pyplot as plt
from configure import *
from scipy.ndimage import gaussian_filter

save_path = 'acoustic_rtm_jax'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

# Overwrite configures
fm = 20
spatial_order = 6
batchsize = 8
epochs = 1
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
solver = RNN(Acoustic(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=None,
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            abcn=abcn, 
            free_surface=free_surface)

# Extract forward & adjoint wavefields
def fwd_base(*args):
    return solver.equation.func(*args[:-2])

def fwd(*args):
    wavefields, vjp_fun = jax.vjp(solver.equation.func, *args[:-2])
    return wavefields, args[0:1]+(vjp_fun,)

def bwd(res, g):
    """Backward step of the wave equation, used for vjp."""
    vjp_fun = res[-1]
    grads = vjp_fun(g)
    # Source-side illumination
    sill = jnp.sum(res[0]**2, axis=0).squeeze()
    rill = jnp.sum(grads[0]**2, axis=0).squeeze()
    return grads + (sill,rill)

step_fn = jax.custom_vjp(fwd_base)
step_fn.defvjp(fwd, bwd)

# Set the true model
solver.set_parameters([jnp.array(true_model)])

# Geometry
src_x = jnp.arange(0,nx, src_step*4).reshape(-1, 1)
src_z = jnp.ones_like(src_x)*srcz
sources = jnp.concatenate([src_x, src_z], axis=1) # (nshots, 2)
rec_x = jnp.arange(0, nx, rec_step).reshape(-1, 1)
rec_z = jnp.ones_like(rec_x)*recz
receivers = jnp.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

# Observed data
obs = solver(wave, sources, receivers)

# Set the model
solver.set_parameters([jnp.array(smooth_model)])

sources = jnp.array(sources)
receivers = jnp.array(receivers)

tmp1 = solver.pad(jnp.zeros_like(solver.vp))
tmp2 = solver.pad(jnp.zeros_like(solver.vp))

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)

# Set the model
solver.set_parameters([jnp.array(smooth_model)] )

@jax.jit
def fwi_step(params, rand_shots):
    # @jax.jit
    def loss_fn(params, sill, rill, shot_nums):
        # Forward modeling
        syn = solver(wave,
                    sources=sources[shot_nums], 
                    receivers=receivers[shot_nums], 
                    models=[params], 
                    wave_equation=step_fn, 
                    aux_args=[sill, rill],
                    )
        # step_fn=step_fn
        _obs = obs[shot_nums]

        _loss_ = jnp.mean((syn-_obs)**2)

        return _loss_, (syn, _obs)
    # Compute the gradient
    (loss, data), gradients = jax.value_and_grad(loss_fn, (0, 1, 2), has_aux=True)(params, tmp1, tmp2, rand_shots)
    return loss, gradients

LOSS = []
for epoch in tqdm.trange(epochs):

    rand_shots = jnp.arange(sources.shape[0])

    loss, grads = fwi_step(solver.vp, rand_shots)
    grad_vp = grads[0]
    sill = solver.crop(grads[1])
    rill = solver.crop(grads[2])

    print(f'Epoch: {epoch}, Loss: {loss.item()}')
    LOSS.append(loss)
    # Save the model
    if epoch % show_every == 0:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        vmin, vmax = np.percentile(grad_vp, [2, 98])
        ax[0].imshow(grad_vp, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        ax[0].set_title('Original Gradient')

        # Use illumination
        grad_sill = grad_vp / jnp.sqrt(sill)
        vmin, vmax = np.percentile(grad_sill, [2, 98])
        ax[1].imshow(grad_sill, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        ax[1].set_title('Gradient (Source-side)')

        # Use illumination
        grad_srill = grad_vp / jnp.sqrt(sill*rill)
        vmin,vmax=np.percentile(grad_srill, [2, 98])
        ax[2].imshow(grad_srill, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        ax[2].set_title('Gradient (Source & Receiver-side)')
        plt.tight_layout()
        plt.savefig(f'{save_path}/gradient_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    break