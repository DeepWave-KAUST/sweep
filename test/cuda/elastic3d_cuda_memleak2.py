import tqdm
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda2 import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Elastic3D
from sweep.utils.general import boundary_gpu_memory, bytes_to_gb

import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, ny, nx = 187, 201, 201
true_vp = np.ones((nz, ny, nx), dtype=np.float32) * 1500.0
true_vs = true_vp/1.73
true_rho = np.ones((nz, ny, nx), dtype=np.float32) * 1000.0
# true_vp[nz//2:, :] = 2000.0
def ricker(t, fm):
    pi2 = np.pi * 2
    wave = (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)
    return wave#.to(torch.float32)

dev = torch.device("cuda:0")
nt = 3001
dt = 0.002
delay = 0.2
dh = 10.0
fm = 5.0
spatial_order = 8
abcn = 10
transfer_interval = 50

cpu_mem = boundary_gpu_memory(9, nt, 1, nz, ny, nx, spatial_order//2+1)
print(f"{bytes_to_gb(cpu_mem):.2f} GB")
print('transfer_interval:', transfer_interval)
t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([nx//2, ny//2, 1]).reshape(1, 3)

# receivers = np.stack((np.arange(0, nx, 1), 
#                      np.zeros(nx, dtype=np.int32)), axis=1).reshape(1, -1, 2)

recx, recy = np.meshgrid(np.arange(0, nx, 5), np.arange(0, ny, 5))
rec_z = np.ones_like(recx)
receivers = np.concatenate([recx.reshape(-1, 1), recy.reshape(-1, 1), rec_z.reshape(-1, 1)], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)
print(f"Number of receivers: {receivers.shape[1]}")
vp = torch.from_numpy(true_vp).float().to(device).requires_grad_()
vs = torch.from_numpy(true_vs).float().to(device).requires_grad_()
rho = torch.from_numpy(true_rho).float().to(device).requires_grad_()

prop = [PropCUDA]
pnames = ['CUDA']

eq_kwargs = dict(spatial_order=spatial_order, device=device)
prop_kwargs = dict(shape=vp.shape, 
                   source_type=['sxx', 'syy', 'szz'], 
                   receiver_type=['vx', 'vy', 'vz'], 
                   abcn=abcn, 
                   dh=dh, 
                   dt=dt, 
                   pml_type='cpmls', 
                   nt = nt,
                   B = 1,
                   transfer_interval = transfer_interval,
                   dev=device, 
                   free_surface=False)
solver = PropCUDA(Elastic3D(**eq_kwargs), **prop_kwargs)
# exit()
for i in tqdm.trange(10001):

    vp.grad = None
    vs.grad = None
    rho.grad = None
    out = solver(wave, sources = sources,
                    receivers = receivers,
                    models=[vp, vs, rho], 
                    use_boundary_saving=True, 
                    transfer_interval=transfer_interval)
    record = out.detach().cpu().numpy().squeeze()
    loss = out.pow(2).sum()
    loss.backward()

