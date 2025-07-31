import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import Acoustic
from geophyai.signal import ricker
from functools import partial
import matplotlib.pyplot as plt

### Configures
nz, nx = 256, 256
dt = 0.001
nt = 501
dh = 5.0
fm = 20
delay = 0.1
spatial_order = 8
abcn = 50

# Generate models
vp = np.ones((nz, nx), np.float32) * 2000.0

save_path = 'acoustic'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

t = np.arange(0, nt*dt, dt)
shape = (nz, nx)

wave = jnp.array(ricker(t-delay, f=fm))
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward solver_all for observed data
solver_kwargs = dict(shape=shape, dev=None, dh=dh, dt=dt, source_type=['h1'], receiver_type=['h1'], abcn=abcn, free_surface=False)
eq_kwargs = dict(spatial_order=spatial_order, backend='jax', )
solver_pml = RNNJax(Acoustic(**eq_kwargs), **solver_kwargs)
solver_habc = RNNJax(Acoustic(**eq_kwargs), use_habc=True, **solver_kwargs)
# Set the true solver_all

# Geometry
sources = np.array([nz//2, nx//2]).reshape(1, 2)
receivers = np.array([nz//2, nx//2]).reshape(1, 1, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)


fig, axes = plt.subplots(1,2, figsize=(8, 4))
axes = axes.flatten()
DATA = []
for i, (solver, titles) in enumerate(zip([solver_habc, solver_pml], ['HABC', 'PML'])):
    start = time.time()
    _, wavefields = solver.forward(wave, sources, receivers, models=[vp], return_wavefield=True)
    wavefields.block_until_ready()
    end = time.time()
    print(f"Time taken for {titles} solver: {end - start:.2f} seconds")
    # Plot the data
    show_d = wavefields[500,0].squeeze()[abcn:-abcn, abcn:-abcn]
    vmin, vmax = np.percentile(show_d, [2, 98])
    axes[i].imshow(show_d, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')
    axes[i].set_title(f'Wavefield at t={500*dt:.3f}s ({titles})')
plt.tight_layout()
plt.savefig(f'{save_path}/wavefields.png', dpi=300, bbox_inches='tight')
plt.close()