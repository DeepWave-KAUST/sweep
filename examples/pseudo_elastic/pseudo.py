import sys, tqdm, os
import jax.numpy as jnp
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import ElasticP, Elastic, Acoustic1st
from geophyai.signal import ricker
import numpy as np
import matplotlib.pyplot as plt

dh = 5.0
spatial_order = 8

abcn = 50
free_surface = False
use_habc = False
nt = 801
dt = 0.001
delay = 0.128
fm = 8

save_path = 'pseudo_elastic'
if not os.path.exists(save_path):
    os.makedirs(save_path)

nz, nx = 201, 201
vp = np.ones((nz, nx), dtype=np.float32)*500.
vp[:50,:] = 300.
vs = np.ones((nz, nx), dtype=np.float32)*150.
vs[:50,:] = 0.
rho = np.ones((nz, nx), dtype=np.float32)*1800
rho[:50,:] = 1250

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
plt.colorbar(axes[0].imshow(vp, cmap='seismic', aspect='auto'), ax=axes[0])
axes[0].set_title('P-wave velocity (vp)')
plt.colorbar(axes[1].imshow(vs, cmap='seismic', aspect='auto'), ax=axes[1])
axes[1].set_title('S-wave velocity (vs)')
plt.colorbar(axes[2].imshow(rho, cmap='seismic', aspect='auto'), ax=axes[2])
axes[2].set_title('Density (rho)')
plt.tight_layout()
plt.savefig(f'{save_path}/model.png', dpi=300, bbox_inches='tight')
plt.close()

t = np.arange(0, nt*dt, dt)
shape = vp.shape

wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

eq_kwargs = dict(spatial_order=spatial_order, device=None, backend='jax')
solver_kwargs = dict(shape=shape, dev=None, dh=dh, dt=dt, abcn=abcn, free_surface=free_surface)

esolver = RNNJax(Elastic(**eq_kwargs), source_type=['vz'], receiver_type=['vx', 'vz'], **solver_kwargs)
asolver = RNNJax(Acoustic1st(**eq_kwargs), source_type=['vz'], receiver_type=['vx', 'vz'], **solver_kwargs)
ePsolver = RNNJax(ElasticP(**eq_kwargs), source_type=['vz'], receiver_type=['vx', 'vz'], **solver_kwargs)

esolver.set_parameters([jnp.array(m) for m in [vp, vs, rho]])
asolver.set_parameters([jnp.array(m) for m in [vp, rho]])
ePsolver.set_parameters([jnp.array(m) for m in [vp, vs, rho]])
# Geometry
sources = np.array([nz//2, nx//2]).reshape(1, 2)
receivers = np.array([nz//2, nx//2]).reshape(1, 1, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)
titles = ["Elastic", "Acoustic", "Pseudo Elastic"]

# The model is padded for ABCs, so the images have shape (401, 401)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
idx = 0
for solver, title in zip([esolver, asolver, ePsolver], titles):
    if title == 'Acoustic':
        vx_idx, vz_idx = 1, 2
    else:
        vx_idx, vz_idx = 0, 1
    _, wavefields = solver.forward(wave, sources, receivers, return_wavefield=True)
    vx = wavefields[nt-1, vx_idx, ...].squeeze()[abcn:-abcn, abcn:-abcn]
    vz = wavefields[nt-1, vz_idx, ...].squeeze()[abcn:-abcn, abcn:-abcn]
    vmin, vmax = np.percentile(vx, [1, 99])
    axes[0, idx].imshow(vx, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0, idx].set_title(f'{title} vx')
    vmin, vmax = np.percentile(vz, [1, 99])
    axes[1, idx].imshow(vz, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1, idx].set_title(f'{title} vz')
    idx += 1

for ax in axes.ravel():
    ax.axis('off')
    ax.hlines(50, 0, nx-1, colors='red', linestyles='dashed')

plt.tight_layout()
plt.savefig(f'{save_path}/wavefields.png', dpi=300, bbox_inches='tight')
plt.close()