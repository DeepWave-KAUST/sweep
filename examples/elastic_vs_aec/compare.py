import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, time

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import Elastic, AEC, Acoustic1st
from geophyai.signal import ricker
from functools import partial
import matplotlib.pyplot as plt

### Configures
nz, nx = 512, 512
dt = 0.001
nt = 1201
dh = 5.0
fm = 8
delay = 0.1
spatial_order = 8
abcn = 50
# Generate models
vp = np.zeros((nz, nx), np.float32)
# vp[0:nz//2, :] = 1500.0 # water layer
# vp[nz//2:, :] = 2500.0
vp[0:nz//3, :] = 1500.0 # water layer
vp[nz//3:nz*2//3, :] = 2000.0 # sediment layer
vp[nz*2//3:, :] = 2500.0 # basement layer

vs = np.zeros((nz, nx), np.float32)
# vs[0:nz//2, :] = 0.0 # water layer
# vs[nz//2:, :] = 1250.0 # basement layer
vs[0:nz//3, :] = 0.0 # water layer
vs[nz//3:nz*2//3, :] = 1200.0
vs[nz*2//3:, :] = 1450.0 # basement layer

rho = np.zeros((nz, nx), np.float32)
# rho[0:nz//2, :] = 1000.0 # water layer
# rho[nz//2:, :] = 2100.0 # basement layer
rho[0:nz//3, :] = 1000.0 # water layer
rho[nz//3:nz*2//3, :] = 1600.0 # sediment layer
rho[nz*2//3:, :] = 2000.0 #

save_path = 'elastic'
if not os.path.exists(save_path):
    os.makedirs(save_path)

np.random.seed(0)
key = random.PRNGKey(0)

t = np.arange(0, nt*dt, dt)
shape = (nz, nx)

wave = jnp.array(ricker(t-delay, f=fm))

# Forward solver_all for observed data
solver_kwargs = dict(shape=shape, dev=None, dh=dh, dt=dt, abcn=abcn, free_surface=False)
eq_kwargs = dict(spatial_order=spatial_order, backend='jax')
solver_elastic = RNNJax(Elastic(**eq_kwargs), source_type=['txx', 'tzz'], receiver_type=['vx', 'vz'], **solver_kwargs)
solver_aec = RNNJax(AEC(**eq_kwargs), source_type=['p'], receiver_type=['p', 'vx', 'vz'],**solver_kwargs)
solver_acoustic = RNNJax(Acoustic1st(**eq_kwargs), source_type=['p'], receiver_type=['p'], **solver_kwargs)
# Set the true solver_all

# Geometry
sources = np.array([nx//2, 0]).reshape(1, 2)
receivers = np.array([0, nx//2]).reshape(1, 1, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)


# Show AEC
_, wavefields = solver_aec.forward(-wave, sources, receivers, models=[vp, vs, rho], return_wavefield=True)
# If we want to get the same wavefield as the elastic solver, we need to inverse the wavelet
fig, axes = plt.subplots(3,3, figsize=(9, 9))

for idx, show_time in enumerate([800, 1000, 1200]):
    # Plot the data
    p = wavefields[show_time,0].squeeze()[abcn:-abcn, abcn:-abcn]
    vx = wavefields[show_time,1].squeeze()[abcn:-abcn, abcn:-abcn]
    vz = wavefields[show_time,2].squeeze()[abcn:-abcn, abcn:-abcn]

    vmin, vmax = np.percentile(p, [0, 100])
    axes[0, idx].imshow(p, cmap='gray', vmin=vmin, vmax=vmax)
    vmin, vmax = np.percentile(vx, [0, 100])
    axes[1, idx].imshow(vx, cmap='gray', vmin=vmin, vmax=vmax)
    vmin, vmax = np.percentile(vz, [0, 100])
    axes[2, idx].imshow(vz, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, idx].set_title(f'p at time {show_time*dt:.2f}s')
    axes[1, idx].set_title(f'vx at time {show_time*dt:.2f}s')
    axes[2, idx].set_title(f'vz at time {show_time*dt:.2f}s')


for ax in axes.ravel():
    ax.axis('off')
    ax.hlines(nz//3, 0, nx, colors='red', linestyles='dashed')
    ax.hlines(nz*2//3, 0, nx, colors='red', linestyles='dashed')
plt.tight_layout()
plt.savefig(f'{save_path}/aec_wavefields.png', dpi=300, bbox_inches='tight')
plt.close()

del wavefields

# Show Elastic wavefields
fig, axes = plt.subplots(3,3, figsize=(9, 9))
_, wavefields = solver_elastic.forward(wave, sources, receivers, models=[vp, vs, rho], return_wavefield=True)

for idx, show_time in enumerate([800, 1000, 1200]):
    # Plot the data
    txx = wavefields[show_time,2].squeeze()[abcn:-abcn, abcn:-abcn]
    tzz = wavefields[show_time,3].squeeze()[abcn:-abcn, abcn:-abcn]
    p = -(txx + tzz) / 2.0
    vx = wavefields[show_time,0].squeeze()[abcn:-abcn, abcn:-abcn]
    vz = wavefields[show_time,1].squeeze()[abcn:-abcn, abcn:-abcn]

    vmin, vmax = np.percentile(p, [0, 100])
    axes[0, idx].imshow(p, cmap='gray', vmin=vmin, vmax=vmax)
    vmin, vmax = np.percentile(vx, [0, 100])
    axes[1, idx].imshow(vx, cmap='gray', vmin=vmin, vmax=vmax)
    vmin, vmax = np.percentile(vz, [0, 100])
    axes[2, idx].imshow(vz, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, idx].set_title(f'p at time {show_time*dt:.2f}s')
    axes[1, idx].set_title(f'vx at time {show_time*dt:.2f}s')
    axes[2, idx].set_title(f'vz at time {show_time*dt:.2f}s')
for ax in axes.ravel():
    ax.axis('off')
    ax.hlines(nz//3, 0, nx, colors='red', linestyles='dashed')
    ax.hlines(nz*2//3, 0, nx, colors='red', linestyles='dashed')
plt.tight_layout()
plt.savefig(f'{save_path}/elastic_wavefields.png', dpi=300, bbox_inches='tight')
plt.close()

# Show Acoustic wavefields
fig, axes = plt.subplots(3,3, figsize=(9, 9))
_, wavefields = solver_acoustic.forward(-wave, sources, receivers, models=[vp, rho], return_wavefield=True)

for idx, show_time in enumerate([800, 1000, 1200]):
    # Plot the data
    p = wavefields[show_time,0].squeeze()[abcn:-abcn, abcn:-abcn]
    vx = wavefields[show_time,1].squeeze()[abcn:-abcn, abcn:-abcn]
    vz = wavefields[show_time,2].squeeze()[abcn:-abcn, abcn:-abcn]

    vmin, vmax = np.percentile(p, [0, 100])
    axes[0, idx].imshow(p, cmap='gray', vmin=vmin, vmax=vmax)
    vmin, vmax = np.percentile(vx, [0, 100])
    axes[1, idx].imshow(vx, cmap='gray', vmin=vmin, vmax=vmax)
    vmin, vmax = np.percentile(vz, [0, 100])
    axes[2, idx].imshow(vz, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0, idx].set_title(f'p at time {show_time*dt:.2f}s')
    axes[1, idx].set_title(f'vx at time {show_time*dt:.2f}s')
    axes[2, idx].set_title(f'vz at time {show_time*dt:.2f}s')
for ax in axes.ravel():
    ax.axis('off')
    ax.hlines(nz//3, 0, nx, colors='red', linestyles='dashed')
    ax.hlines(nz*2//3, 0, nx, colors='red', linestyles='dashed')
plt.tight_layout()
plt.savefig(f'{save_path}/acoustic_wavefields.png', dpi=300, bbox_inches='tight')
plt.close()