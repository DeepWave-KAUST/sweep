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

### Configures
nz, nx = 256, 256
dt = 0.001
nt = 351
dh = 5.0
fm = 20
delay = 0.1
spatial_order = 8
abcn = 50
free_surface = False
# Generate models
vp = np.ones((nz, nx), np.float32) * 2000.0

save_path = 'simple'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

t = np.arange(0, nt*dt, dt)

nz, nx = vp.shape

wave = jnp.array(ricker(t-delay, f=fm))
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

a1st = RNNJax(Acoustic1st(spatial_order=spatial_order, backend='jax'), 
                shape=vp.shape, 
                dev=None,
                dh=dh,
                dt=dt,
                source_type=['p'],
                receiver_type=['p'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=False)

a2nd = RNNJax(Acoustic(spatial_order=spatial_order, backend='jax'), 
                shape=vp.shape, 
                dev=None,
                dh=dh,
                dt=dt,
                source_type=['h1'],
                receiver_type=['h1'],
                abcn=abcn, 
                free_surface=free_surface, 
                use_ckpt=False)
# Set the true a1stb
a1st.set_parameters([jnp.array(vp), jnp.ones_like(vp)])
a2nd.set_parameters([jnp.array(vp)])
# Geometry
sources = np.array([nz//2, nx//2]).reshape(1, 2)
receivers = np.array([nz//2, nx//2]).reshape(1, 1, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)
show_d = []
for solver, title in zip([a1st, a2nd], ['Acoustic 1st', 'Acoustic 2nd']):
    if title == 'Acoustic 1st':
        _wave = jnp.cumsum(wave)
    else:
        _wave = jnp.array(wave)

    _, wavefields = solver(_wave, sources, receivers, return_wavefield=True)
    show_d.append(wavefields[350,0].squeeze()[abcn:-abcn, abcn:-abcn])

d = np.zeros_like(show_d[0])
d[:, 0:nx//2] = show_d[0][:, 0:nx//2]
d[:, nx//2:] = show_d[1][:, nx//2:]
vmin, vmax = np.percentile(d, [1, 99])
fig, ax = plt.subplots(1,1, figsize=(4, 4))
ax.imshow(d, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
ax.axis('off')
ax.vlines(nx//2, 0, nz, colors='red', linestyles='dashed')
ax.text(nx//2 - 25, nz//2, '1st', color='red', fontsize=12, va='center')
ax.text(nx//2 + 5, nz//2, '2nd', color='red', fontsize=12, va='center')
plt.tight_layout()
plt.savefig(f'{save_path}/wavefields.png', dpi=300, bbox_inches='tight')
plt.close()
    