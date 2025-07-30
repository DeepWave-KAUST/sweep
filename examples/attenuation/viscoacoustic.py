import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys, tqdm, jax, optax

import numpy as np
import jax.numpy as jnp
import jax.random as random
sys.path.append('../../src')
from geophyai.rnn import RNNJax
from geophyai.equations import ViscoAcoustic
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

# Generate models
vp = np.ones((nz, nx), np.float32) * 2000.0
Q = np.ones((nz, nx), np.float32) * 50.0
omega = np.ones_like(vp) * 2 * np.pi * fm

save_path = 'viscoacoustic'
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
solver_none = RNNJax(ViscoAcoustic(**eq_kwargs, phase_shift=False, amplitude_damping=False), **solver_kwargs)
solver_all = RNNJax(ViscoAcoustic(**eq_kwargs, phase_shift=True, amplitude_damping=True), **solver_kwargs)
solver_phase = RNNJax(ViscoAcoustic(**eq_kwargs, phase_shift=True, amplitude_damping=False), **solver_kwargs)
solver_amplitude = RNNJax(ViscoAcoustic(**eq_kwargs, phase_shift=False, amplitude_damping=True), **solver_kwargs)

# Set the true solver_all

# Geometry
sources = np.array([nz//2, nx//2]).reshape(1, 2)
receivers = np.array([nz//2, nx//2]).reshape(1, 1, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)


fig, axes = plt.subplots(2,2, figsize=(16, 16))
axes = axes.flatten()
DATA = []
for i, solver in enumerate([solver_none, solver_phase, solver_amplitude, solver_all]):
    print(f'Solver {i} with phase_shift={solver.equation.phase_shift}, amplitude_damping={solver.equation.amplitude_damping}')
    _, wavefields = solver.forward(wave, sources, receivers, models=[vp, Q, omega], return_wavefield=True)
    # Plot the data
    show_d = wavefields[350,0].squeeze()[abcn:-abcn, abcn:-abcn]
    DATA.append(show_d)
    vmin, vmax = np.percentile(show_d, [1, 99])
    axes[i].imshow(show_d, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    axes[i].set_title(f'Phase Shift: {solver.equation.phase_shift}, Amplitude Damping: {solver.equation.amplitude_damping}')
plt.tight_layout()
plt.savefig(f'{save_path}/wavefields.png', dpi=300, bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(6, 6))
show_d = np.zeros_like(DATA[0])
show_d[0:nz//2, 0:nx//2] = DATA[0][0:nz//2, 0:nx//2] # Top-left: Acoustic
show_d[0:nz//2, -nx//2:] = DATA[1][0:nz//2, -nx//2:] # Bottom-left: Phase Shift
show_d[-nz//2:, 0:nx//2] = DATA[2][-nz//2:, 0:nx//2] # Bottom-right: Amplitude Damping
show_d[-nz//2:, -nx//2:] = DATA[3][-nz//2:, -nx//2:] # Top-right: All
# add text 
positions = {
    "Acoustic": (75, 100),
    "Phase Shift": (170, 100),
    "Amplitude Damping": (75, 150),
    "Phase & Amplitude": (175, 150)
}
for label, (x, y) in positions.items():
    ax.text(x, y, label, color='red', fontsize=12, ha='center', va='center')
ax.axis('off')
vmin, vmax = np.percentile(show_d, [1, 99])
ax.imshow(show_d, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', interpolation='bilinear')
ax.set_title('Wavefields')
plt.savefig(f'{save_path}/combined_wavefields.png', dpi=300, bbox_inches='tight')
plt.close()
