import sys, tqdm, os
import jax.numpy as jnp
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import AcousticTTI, AcousticVTI
from geophyai.signal import ricker
import numpy as np
import matplotlib.pyplot as plt

dh = 10
spatial_order = 6

abcn = 50
free_surface = False
use_habc = False
nt = 601
dt = 0.001
delay = 0.256
fm = 10

save_path = 'acoustic_anisotropic'
if not os.path.exists(save_path):
    os.makedirs(save_path)

nz, nx = 301, 301
vel = np.ones((nz, nx), dtype=np.float32)*3000.
# Model A
epsilons_a = np.ones_like(vel)*0.3
deltas_a = np.ones_like(vel)*0.3
# Model B
epsilons_b = np.ones_like(vel)*0.3
deltas_b = np.ones_like(vel)*0.1
# Model C
epsilons_c = np.ones_like(vel)*0.1
deltas_c = np.ones_like(vel)*0.3

theta = np.ones_like(vel)*30.  # in degree

models_a = [vel, epsilons_a, deltas_a]
models_b = [vel, epsilons_b, deltas_b]
models_c = [vel, epsilons_c, deltas_c]

t = np.arange(0, nt*dt, dt)
shape = epsilons_a.shape

fm = 20
wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

solver_vti = RNNJax(AcousticVTI(spatial_order=spatial_order, device=None, backend='jax'), 
                    shape=shape, 
                    dev=None, 
                    dh=dh,
                    dt=dt,
                    source_type=['h1'],
                    receiver_type=['h1'],
                    abcn=abcn, 
                    free_surface=free_surface)

solver_tti = RNNJax(AcousticTTI(spatial_order=spatial_order, device=None, backend='jax'), 
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
titles = [r"$\delta= \epsilon$=0.3", 
          r"$\delta=0.1, \epsilon=0.3$",
          r"$\delta=0.3, \epsilon=0.1$"]

# The model is padded for ABCs, so the images have shape (401, 401)
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for model_group, ax, title in zip([models_a, models_b, models_c], axes, titles):
    solver_vti.set_parameters([jnp.array(m) for m in model_group])
    _, wavefields = solver_vti.forward(wave, sources, receivers, return_wavefield=True)
    show_data = wavefields[600, 0, ...].squeeze()[abcn:-abcn, abcn:-abcn]
    vmin, vmax = np.percentile(show_data, [2, 98])
    ax.imshow(show_data, cmap='gray', aspect='auto')
    # ax.set_title(title, fontsize=12)
    ax.text(150, 150, title, color='red', fontsize=12, ha='center', va='center')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_path}/wavefields_vti.png', dpi=300, bbox_inches='tight')
plt.close()
# RUN TTI
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for model_group, ax, title in zip([models_a, models_b, models_c], axes, titles):
    solver_tti.set_parameters([jnp.array(m) for m in model_group+[theta,]])
    _, wavefields = solver_tti.forward(wave, sources, receivers, return_wavefield=True)
    show_data = wavefields[600, 0, ...].squeeze()[abcn:-abcn, abcn:-abcn]
    vmin, vmax = np.percentile(show_data, [2, 98])
    ax.imshow(show_data, cmap='gray', aspect='auto')
    # ax.set_title(title, fontsize=12)
    ax.text(150, 150, title, color='red', fontsize=12, ha='center', va='center')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{save_path}/wavefields_tti.png', dpi=300, bbox_inches='tight')
plt.close()