import tqdm, time
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Acoustic3D
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, ny, nx = 128, 128, 128
true_model = np.ones((nz, ny, nx), dtype=np.float32) * 1500.0
def ricker(t, fm):
    pi2 = np.pi * 2
    wave = (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)
    return wave#.to(torch.float32)

dev = torch.device("cuda:0")
nt = 1000
dt = 0.002
delay = 0.2
dh = 10.0
fm = 5.0
spatial_order = 8
abcn = 30

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([10, 25, 1]).reshape(1, 3)
receivers = np.array([40, 25, 1]).reshape(1, 1, 3)
# receivers = np.stack((np.arange(0, nx, 1), 
#                      np.zeros(nx, dtype=np.int32)), axis=1).reshape(1, -1, 2)

# recx, recy = np.meshgrid(np.arange(0, nx, 10), np.arange(0, ny, 10))
# rec_z = np.ones_like(recx)
# receivers = np.concatenate([recx.reshape(-1, 1), recy.reshape(-1, 1), rec_z.reshape(-1, 1)], axis=1)
# receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

vp = torch.from_numpy(true_model).float().to(device).requires_grad_()

kwargs_eq = dict(spatial_order=spatial_order, device=device)
kwargs_modeling = dict(shape=vp.shape, source_type=['h1'], receiver_type=['h1'], abcn=abcn, dh=dh, dt=dt, pml_type='cpmlr', dev=device)

solver_cuda = PropCUDA(Acoustic3D(**kwargs_eq), **kwargs_modeling)

# CUDA WITHOUT BOUNDARY SAVING
for _ in tqdm.trange(10000):
    vp.grad = None
    syn = solver_cuda(wave, sources, receivers, models=[vp], use_boundary_saving=True)
    syn.pow(2).sum().backward()
