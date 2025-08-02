import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import Elastic
from geophyai.signal import ricker
from functools import partial
import matplotlib.pyplot as plt

### Configures
nz, nx = 256, 256
dt = 0.001
nt = 601
dh = 5.0
fm = 20
delay = 0.1
spatial_order = 8
abcn = 50
show_time = 600
# Generate models
vp = np.ones((nz, nx), np.float32) * 2000.0
vs = vp / 1.732
rho = np.ones((nz, nx), np.float32) * 1000.0

save_path = 'elastic'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

t = np.arange(0, nt*dt, dt)
shape = (nz, nx)

wave = jnp.array(ricker(t-delay, f=fm))


# Forward solver_all for observed data
solver_kwargs = dict(shape=shape, dev=None, dh=dh, dt=dt, source_type=['txx', 'tzz'], receiver_type=['vz'], abcn=abcn, free_surface=False)
eq_kwargs = dict(spatial_order=spatial_order, backend='jax', )
solver_pml = RNNJax(Elastic(**eq_kwargs), **solver_kwargs)
solver_habc = RNNJax(Elastic(**eq_kwargs), use_habc=True, **solver_kwargs)
# Set the true solver_all

# Geometry
sources = np.array([nz//2, nx//2]).reshape(1, 2)
receivers = np.array([nz//2, nx//2]).reshape(1, 1, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)


fig, axes = plt.subplots(2,2, figsize=(8, 8))
DATA = []
for i, (solver, titles) in enumerate(zip([solver_habc, solver_pml], ['HABC', 'PML'])): #, 'PML'
    start = time.time()
    _, wavefields = solver.forward(wave, sources, receivers, models=[vp, vs, rho], return_wavefield=True)
    wavefields.block_until_ready()
    end = time.time()
    print(f"Time taken for {titles} solver: {end - start:.2f} seconds")
    # Plot the data
    vx = wavefields[show_time,0].squeeze()#[abcn:-abcn, abcn:-abcn]
    vz = wavefields[show_time,1].squeeze()#[abcn:-abcn, abcn:-abcn]
    
    vmin, vmax = np.percentile(vx, [0, 100])
    axes[i, 0].imshow(vx, cmap='seismic', vmin=vmin, vmax=vmax)
    vmin, vmax = np.percentile(vz, [0, 100])
    axes[i, 1].imshow(vz, cmap='seismic', vmin=vmin, vmax=vmax)
    axes[i, 0].set_title(f'vx ({titles})')
    axes[i, 1].set_title(f'vz ({titles})')
    # Draw the boundary of the ABC on image
    axes[i, 0].add_patch(plt.Rectangle((abcn, abcn), nx, nz, linewidth=1, edgecolor='black', facecolor='none'))
    axes[i, 1].add_patch(plt.Rectangle((abcn, abcn), nx, nz, linewidth=1, edgecolor='black', facecolor='none'))

plt.tight_layout()
plt.savefig(f'{save_path}/wavefields.png', dpi=300, bbox_inches='tight')
plt.close()