import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, tqdm, jax, optax

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import AcousticVF, Acoustic1st
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
true_vp = np.load(true_vp_path)
true_rx = np.load(true_rx_path)
true_rz = np.load(true_rz_path)
true_rho = np.load(true_rho_path)

shape = true_vp.shape

nz, nx = shape

wave = jnp.array(ricker(t-delay, f=fm))
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

eq_kwargs = dict(spatial_order=spatial_order, backend='jax')
solver_kwargs = dict(shape=shape, dev=None, dh=dh, dt=dt, abcn=abcn, free_surface=free_surface, use_habc=use_habc)

avf = RNNJax(AcousticVF(**eq_kwargs), source_type=['h1'], receiver_type=['h1'], **solver_kwargs)
a1st = RNNJax(Acoustic1st(**eq_kwargs), source_type=['p'], receiver_type=['p'], **solver_kwargs)

# Set the true a1stb
avf.set_parameters([jnp.array(true_vp), jnp.array(true_rx), jnp.array(true_rz)])
a1st.set_parameters([jnp.array(true_vp), jnp.array(true_rho)])
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

for solver, title in zip([avf, a1st], ['Acoustic VR', 'Acoustic 1st']):
    if title == 'Acoustic 1st':
        _wave = jnp.cumsum(wave)  # Integrate the wavelet for 1st order
    else:
        _wave = wave
    obs = solver(_wave, sources[-1:], receivers[-1:])
    vmin, vmax = np.percentile(obs[-1], [2, 98])
    plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{save_path}/{title.lower().replace(" ", "_")}_obs.png', dpi=300, bbox_inches='tight')
    plt.close()