import tqdm
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Acoustic
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, nx = 100, 1024
true_model = np.ones((nz, nx), dtype=np.float32) * 1500.0
true_model[nz//2:, :] = 2000.0

def ricker(t, fm):
    pi2 = np.pi * 2
    wave = (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)
    return wave#.to(torch.float32)

dev = torch.device("cuda:0")
nt = 3000
dt = 0.002
delay = 0.2
dh = 10.0
fm = 5.0
spatial_order = 8
abcn = 20
free_surface=False

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([512, 0]).reshape(1, 2)
receivers = np.stack((np.arange(0, nx, 1), 
                     np.ones(nx, dtype=np.int32)*1), axis=1).reshape(1, -1, 2)

vp = torch.from_numpy(true_model).float().to(device).requires_grad_()

prop = [PropCUDA, PropTorch]
pname = ['CUDA', 'PyTorch']
gradients = []
kwargs = dict(shape=vp.shape, source_type=['h1'], receiver_type=['h1'], abcn=abcn, dh=dh, dt=dt, pml_type='cpmlr', dev=device, free_surface=free_surface)
cuda_solver = PropCUDA(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)
# torch_solver = PropTorch(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)

vp.grad = None
solver_kwargs = dict(wavelet=wave, sources=sources, receivers=receivers, models=[vp])

# CUDA WITH BOUNDARY SAVING
for i in tqdm.trange(10000):
    vp.grad = None
    out = cuda_solver(**solver_kwargs, use_boundary_saving=False)
    loss = out.pow(2).sum()
    loss.backward()
