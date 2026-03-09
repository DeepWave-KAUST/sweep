import tqdm, time
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Acoustic3D
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, ny, nx = 50, 50, 50
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
abcn = 20
free_surface = True

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([10, 25, 0]).reshape(1, 3)
receivers = np.array([40, 25, 0]).reshape(1, 1, 3)
# receivers = np.stack((np.arange(0, nx, 1), 
#                      np.zeros(nx, dtype=np.int32)), axis=1).reshape(1, -1, 2)

# recx, recy = np.meshgrid(np.arange(0, nx, 10), np.arange(0, ny, 10))
# rec_z = np.ones_like(recx)
# receivers = np.concatenate([recx.reshape(-1, 1), recy.reshape(-1, 1), rec_z.reshape(-1, 1)], axis=1)
# receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

vp = torch.from_numpy(true_model).float().to(device).requires_grad_()

kwargs_eq = dict(spatial_order=spatial_order, device=device)
kwargs_modeling = dict(shape=vp.shape, source_type=['h1'], receiver_type=['h1'], abcn=abcn, dh=dh, dt=dt, pml_type='cpmlr', dev=device, free_surface=free_surface)

solver_cuda = PropCUDA(Acoustic3D(**kwargs_eq), **kwargs_modeling)
solver_torch = PropTorch(Acoustic3D(**kwargs_eq), **kwargs_modeling)

# CUDA WITHOUT BOUNDARY SAVING
print('Running CUDA with boundary saving...')
vp.grad = None
syn = solver_cuda(wave, sources, receivers, models=[vp], use_boundary_saving=True)
syn.pow(2).sum().backward()
grad_cuda_bs = vp.grad.cpu().numpy()
grad_cuda_bs /= grad_cuda_bs.max()
# CUDA WITHOUT BOUNDARY SAVING
print('Running CUDA without boundary saving...')
vp.grad = None
syn = solver_cuda(wave, sources, receivers, models=[vp], use_boundary_saving=False)
syn.pow(2).sum().backward()
grad_cuda_nobs = vp.grad.cpu().numpy()
grad_cuda_nobs /= grad_cuda_nobs.max()

# PYTORCH AD
print('Running PyTorch AD...')
vp.grad = None
syn = solver_torch(wave, sources, receivers, models=[vp])
syn.pow(2).sum().backward()
grad_torch = vp.grad.cpu().numpy()
grad_torch /= grad_torch.max()

fig, axes = plt.subplots(3, 3, figsize=(18, 12))

vmin, vmax = np.percentile(grad_torch, [1, 99])
kwargs = dict(vmin=vmin, vmax=vmax, cmap='RdBu_r', aspect='auto')
axes[0, 0].imshow(grad_cuda_nobs[nz//2], **kwargs)
axes[0, 0].set_title('CUDA No Boundary Saving')

axes[0, 1].imshow(grad_cuda_bs[nz//2], **kwargs)
axes[0, 1].set_title('CUDA With Boundary Saving')

axes[0, 2].imshow(grad_torch[nz//2], **kwargs)
axes[0, 2].set_title('PyTorch AD')

axes[1, 0].imshow(grad_cuda_nobs[:, ny//2], **kwargs)
axes[1, 0].set_title('CUDA No Boundary Saving')

axes[1, 1].imshow(grad_cuda_bs[:, ny//2], **kwargs)
axes[1, 1].set_title('CUDA With Boundary Saving')

axes[1, 2].imshow(grad_torch[:, ny//2], **kwargs)
axes[1, 2].set_title('PyTorch AD')

axes[2, 0].imshow(grad_cuda_nobs[:, :, nx//2], **kwargs)
axes[2, 0].set_title('CUDA No Boundary Saving')

axes[2, 1].imshow(grad_cuda_bs[:, :, nx//2], **kwargs)
axes[2, 1].set_title('CUDA With Boundary Saving')

axes[2, 2].imshow(grad_torch[:, :, nx//2], **kwargs)
axes[2, 2].set_title('PyTorch AD')

plt.tight_layout()
plt.savefig('acoustic3d_gradients.png', dpi=300, bbox_inches='tight')
plt.show()