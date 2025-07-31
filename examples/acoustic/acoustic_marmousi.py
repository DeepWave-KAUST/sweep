import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, tqdm, jax, optax

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import Acoustic1st, Acoustic
from geophyai.signal import ricker
from functools import partial
import matplotlib.pyplot as plt
from configure import *

save_path = 'marmousi'
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

a1st = RNNJax(Acoustic1st(spatial_order=spatial_order, backend='jax'), 
                shape=shape, 
                dev=None,
                dh=dh,
                dt=dt,
                source_type=['p'],
                receiver_type=['p'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=False)

a2nd = RNNJax(Acoustic(spatial_order=spatial_order, backend='jax'), 
                shape=shape, 
                dev=None,
                dh=dh,
                dt=dt,
                source_type=['h1'],
                receiver_type=['h1'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=False)
# Set the true a1stb
a1st.set_parameters([jnp.array(true_model), jnp.ones_like(true_model)])
a2nd.set_parameters([jnp.array(true_model)])
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

for solver, title in zip([a1st, a2nd], ['Acoustic 1st', 'Acoustic 2nd']):
    if title == 'Acoustic 1st':
        _wave = jnp.cumsum(wave)
    else:
        _wave = jnp.array(wave)
    obs = solver(_wave, sources[-1:], receivers[-1:])
    vmin, vmax = np.percentile(obs[-1], [2, 98])
    plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title.lower().replace(" ", "_")}_obs.png', dpi=300, bbox_inches='tight')
    plt.close()