import sys, tqdm, os
import jax.numpy as jnp
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import AcousticTariq
from geophyai.signal import ricker
import numpy as np
import matplotlib.pyplot as plt

dh = 5
spatial_order = 8

abcn = 50
free_surface = False
use_habc = False
nt = 601
dt = 0.001
delay = 0.1
fm = 10

save_path = 'acoustic_anisotropic'
if not os.path.exists(save_path):
    os.makedirs(save_path)

nz, nx = 301, 301

vv = np.ones((nz, nx), np.float32) * 1000
v = np.ones((nz, nx), np.float32) * 1000
eta_a = np.ones((nz, nx), np.float32) * 0.4
eta_b = np.ones((nz, nx), np.float32) * 0.0

models_a = [vv, v, eta_a]
models_b = [vv, v, eta_b]

t = np.arange(0, nt*dt, dt)
shape = (nz, nx)

fm = 20
wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

solver_vti = RNNJax(AcousticTariq(spatial_order=spatial_order, device=None, backend='jax'), 
                    shape=shape, 
                    dev=None, 
                    dh=dh,
                    dt=dt,
                    source_type=['h1'],
                    receiver_type=['h1'],
                    abcn=abcn, 
                    free_surface=free_surface)

# Geometry
sources = np.array([nz//2, nx//2]).reshape(1, 2)
receivers = np.array([nz//2, nx//2]).reshape(1, 1, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)
titles = [r"$vv=1000, v=1000, \eta=0.4$", r"$vv=1000, v=1000, \eta=0.0$"]

# The model is padded for ABCs, so the images have shape (401, 401)
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
for model_group, ax, title in zip([models_a, models_b], axes, titles):
    solver_vti.set_parameters([jnp.array(m) for m in model_group])
    _, wavefields = solver_vti.forward(wave, sources, receivers, return_wavefield=True)
    show_data = wavefields[600, 0, ...].squeeze()[abcn:-abcn, abcn:-abcn]
    vmin, vmax = np.percentile(show_data, [2, 98])
    ax.imshow(show_data, cmap='gray', aspect='auto')
    # ax.set_title(title, fontsize=12)
    ax.text(150, 25, title, color='red', fontsize=12, ha='center', va='center')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_path}/wavefields_tariq.png', dpi=300, bbox_inches='tight')
plt.close()