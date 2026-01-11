import sys
sys.path.append('../src')
import numpy as np
import matplotlib.pyplot as plt

import importlib
from geophyai.signal import ricker
from geophyai.rnn import RNN
import torch

pkg_name = 'geophyai.equations'
eq = importlib.import_module(pkg_name)
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set models for testing
dh = 5
spatial_order = 6
abcn = 50
free_surface = False
use_habc = False
nt = 801
dt = 0.001
delay = 0.256
fm = 20
nz, nx = 301, 301

vp = np.ones((nz, nx), dtype=np.float32)*1000.
rho = np.ones((nz, nx), dtype=np.float32)*800.
vs = np.ones((nz, nx), dtype=np.float32)*700.
z = vp * rho
vpz = vp * rho
vsz = vs * rho
rx = -0.5 * z * np.gradient(1/z, dh, axis=0)
rz = -0.5 * z * np.gradient(1/z, dh, axis=1)

# Tariq's model parameters
vv = np.ones((nz, nx), np.float32) * 1000
v = np.ones((nz, nx), np.float32) * 1000
eta_a = np.ones((nz, nx), np.float32) * 0.4

# Model C
epsilon = np.ones_like(vp)*0.1
delta = np.ones_like(vp)*0.3
theta = np.ones_like(vp)*30.  # in degree
Q = np.ones((nz, nx), np.float32) * 50.0
omega = np.ones_like(vp) * 2 * np.pi * fm

model_lib = dict(vp=vp, rho=rho, vs=vs, vv=vv, v=v, eta=eta_a, 
                 epsilon=epsilon, delta=delta, theta=theta, Q=Q, 
                 omega=omega, rx=rx, rz=rz,
                 z=z, vpz=vpz, vsz=vsz)

t = np.arange(0, nt*dt, dt)
shape = vp.shape

wave = ricker(t-delay, f=fm)

sources = np.array([nz//2, nx//2]).reshape(1, 2)
receivers = np.array([nz//2, nx//2]).reshape(1, 1, 2)

backends = ['jax', 'torch']
notpassed = []
for name in eq.__all__:
    if 'rtm' in name.lower():
        continue
    for backend in backends:
        try:
            _eq = getattr(eq, name)(spatial_order=spatial_order, device=dev, backend=backend)
            solver = RNN(_eq, shape=shape, dev=dev, dh=dh,
                         dt=dt, 
                         source_type=_eq.wavefields[0:1], 
                         receiver_type=_eq.wavefields[0:1],
                         abcn=abcn, 
                         free_surface=free_surface)
            params = [model_lib[m] for m in _eq.models]
            if backend == 'torch':
                params = [torch.from_numpy(p).to(solver.dev) for p in params]
            obs, wavefields = solver(wave, sources, receivers, models=params, return_wavefield=True)

            snapshots = [400, 600, 800]
            fig, axes = plt.subplots(1, len(snapshots), figsize=(15, 5))
            for i, snap in enumerate(snapshots):
                if backend == 'torch':
                    d = wavefields[snap][0].squeeze().cpu().numpy()[abcn:-abcn, abcn:-abcn]
                else:
                    d = wavefields[snap][0].squeeze()[abcn:-abcn, abcn:-abcn]
                vmin, vmax = np.percentile(d, [1, 99])
                axes[i].imshow(d, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
                axes[i].set_title(f"{name} {snap*dt:.3f}s")
            plt.tight_layout()
            plt.savefig(f'{name}_{backend}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Processed {name} {backend} successfully.")
        except Exception as e:
            notpassed.append((name, backend, str(e)))
            continue
print("Not passed tests:")
for name, backend, error in notpassed:
    print(f"{name} {backend}: {error}")
