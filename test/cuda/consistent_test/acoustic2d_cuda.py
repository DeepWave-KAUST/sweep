import tqdm
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Acoustic
import numpy as np
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, nx = 256, 512
true_vp = np.ones((nz, nx), dtype=np.float32) * 2000.0

def ricker(t, fm):
    pi2 = np.pi * 2
    wave = (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)
    return wave#.to(torch.float32)

dev = torch.device("cuda:0")
nt = 1000
dt = 0.001
delay = 0.2
dh = 5.0
fm = 10.0
spatial_order = 8
abcn = 30
free_surface=False

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([128, 3]).reshape(1, 2)

receivers = np.array([192, 3]).reshape(1, 1, 2)
print('free_surface:', free_surface, sources, receivers)
print('Spatial order:', spatial_order)
print('abcn: ', abcn)
vp = torch.from_numpy(true_vp).float().to(device).requires_grad_()

# Candidate parameters
prop = [PropCUDA, PropTorch]
srcz = [0, 1, 2, 3, 4, 11]
free_surface = [False, True]
abcn = [0, 20]
spatial_order = [2, 4, 6, 8, 10]

grid = list(product(srcz, free_surface, abcn, spatial_order))

pname = ['CUDA', 'PyTorch']
gradients = []


vp.grad = None

for srcz, free_surface, abcn, spatial_order in tqdm.tqdm(grid):

    sources = np.array([128, srcz]).reshape(1, 2)
    receivers = np.array([384, srcz]).reshape(1, 1, 2)

    solver_kwargs = dict(wavelet=wave, sources=sources, receivers=receivers, models=[vp])
    kwargs = dict(shape=vp.shape, source_type=['h1'], receiver_type=['h1'], abcn=abcn, dh=dh, dt=dt, pml_type='cpmlr', dev=device, free_surface=free_surface)
    cuda_solver = PropCUDA(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)
    torch_solver = PropTorch(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)

    # CUDA WITH BOUNDARY SAVING
    vp.grad = None
    out = cuda_solver(**solver_kwargs, use_boundary_saving=True)
    rec_cuda_bs = out.detach().cpu().numpy()
    loss = out.pow(2).sum()
    loss.backward()
    grads_cuda_vp_bs = vp.grad.cpu().numpy()
    del loss, out

    # CUDA WITHOUT BOUNDARY SAVING
    vp.grad = None
    out = cuda_solver(**solver_kwargs, use_boundary_saving=False)
    rec_cuda_nobs = out.detach().cpu().numpy()
    loss = out.pow(2).sum()
    loss.backward()
    grads_cuda_vp = vp.grad.cpu().numpy()
    del loss, out
    # PYTORCH AD
    vp.grad = None
    out = torch_solver(**solver_kwargs)
    rec_torch = out.detach().cpu().numpy()
    loss = out.pow(2).sum()
    loss.backward()
    grads_torch_vp = vp.grad.cpu().numpy()
    del loss, out

    del cuda_solver, torch_solver
    print(rec_cuda_nobs.shape,rec_cuda_bs.shape, rec_torch.shape )
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(rec_cuda_nobs.squeeze(), 'red', label='CUDA (no bs)')
    ax.plot(rec_cuda_bs.squeeze(), 'black', label='CUDA (bs)')
    ax.plot(rec_torch.squeeze(), 'blue', label='CUDA (no bs)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'acoustic/rec_src{srcz}_abcn{abcn}_fc{free_surface}_so{spatial_order}.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    vmin, vmax = np.percentile(grads_torch_vp, [0.5, 99.5])
    ax = axes[0,0].imshow(grads_cuda_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[0,0])
    axes[0,0].set_title('Vp (CUDA, No Boundary Saving)')

    ax = axes[0,1].imshow(grads_torch_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[0,1])
    axes[0,1].set_title('Vp (Torch)')

    ax = axes[0,2].imshow(grads_cuda_vp - grads_torch_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[0,2])
    axes[0,2].set_title('Vp (Difference)')

    ax = axes[1,0].imshow(grads_cuda_vp_bs, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[1,0])
    axes[1,0].set_title('Vp (CUDA, With Boundary Saving)')

    ax = axes[1,1].imshow(grads_torch_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[1,1])
    axes[1,1].set_title('Vp (Torch)')

    ax = axes[1,2].imshow(grads_cuda_vp_bs - grads_torch_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[1,2])
    axes[1,2].set_title('Vp (Difference)')

    plt.tight_layout()
    plt.savefig(f'acoustic/a_src{srcz}_abcn{abcn}_fc{free_surface}_so{spatial_order}.png', dpi=300, bbox_inches='tight')
    plt.close()