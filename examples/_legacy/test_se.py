import sys, tqdm, os, jax, optax
import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../src')
from geophyai.rnn import RNNJax
from geophyai.equations import Acoustic
from geophyai.signal import ricker
from functools import partial
import matplotlib.pyplot as plt
from configure_marmousi import *

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
model = RNNJax(Acoustic(spatial_order=spatial_order, backend='jax'), 
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
src_x = np.arange(0,nx, src_step).reshape(-1, 1)
src_z = np.ones_like(src_x)*srcz
sources = np.concatenate([src_x, src_z], axis=1) # (nshots, 2)
rec_x = np.arange(0, nx, rec_step).reshape(-1, 1)
rec_z = np.ones_like(rec_x)*recz
receivers = np.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)

receivers = receivers[0:1]
wave1 = wave.reshape(1, -1)
wave2 = -jnp.roll(wave.reshape(1, -1), 1000)
wave = jnp.concatenate([wave1, wave2], axis=0)
sources = jnp.concatenate([sources[0:1], sources[-1:]], axis=0)

obs = model(wave, # (nshots, nt)
            sources, # (nshots, 2)
            receivers,# (1, nreceivers, 2)
            source_encoding=True)

vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/acoustic_obs_encoding.png', dpi=300, bbox_inches='tight')
plt.close()
