import tqdm
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Elastic
import numpy as np
from sweep.scalars import staggered_grid_coes
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, nx = 100, 100
true_vp = np.ones((nz, nx), dtype=np.float32) * 2000.0
# true_vp[nz//2:, :] = 2000.0
true_vs = true_vp /1.73
rho = np.ones((nz, nx), dtype=np.float32) * 1000.0
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
spatial_orders = [8]
sourcesz = [0]
grid = list(product(spatial_orders, sourcesz))
abcn = 20
free_surface = False
use_boundary_saving = False
t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

# sources = np.array([128, 3]).reshape(1, 2)
# print(staggered_grid_coes(spatial_orders//2))
# receivers = np.stack((np.arange(0, nx, 1), 
#                      np.zeros(nx, dtype=np.int32)), axis=1).reshape(1, -1, 2)

# rec_x = np.arange(0, nx, 1).reshape(-1, 1)
# rec_z = np.ones_like(rec_x)*0
# receivers = np.concatenate([rec_x, rec_z], axis=1)
# receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)
receivers = np.array([384, 3]).reshape(1, 1, 2)

vp = torch.from_numpy(true_vp).float().to(device).requires_grad_()
vs = torch.from_numpy(true_vs).float().to(device).requires_grad_()
rho = torch.from_numpy(rho).float().to(device).requires_grad_()

prop = [PropCUDA]
pname = ['CUDA']

for _ in tqdm.trange(10001):
    for so, srcz in grid:

        sources = np.array([10, srcz]).reshape(1, 2)
        receivers = np.array([40, srcz]).reshape(1, 1, 2)

        for name, propagator in zip(pname, prop):
            vp.grad=None
            vs.grad=None
            rho.grad=None
            solver = propagator(Elastic(spatial_order=so, device=device,), 
                            shape=vp.shape, 
                            source_type=['sxx', 'szz'],
                            receiver_type=['vx', 'vz'],
                            abcn=abcn , 
                            dh = dh,
                            dt = dt,
                            pml_type='cpmls',
                            dev=device,
                            free_surface=free_surface,
                            )
            
            out = solver(wave, sources = sources,
                        receivers = receivers,
                        models=[vp, vs, rho], 
                        use_boundary_saving=use_boundary_saving)
            loss = out.pow(2).sum()
            loss.backward()