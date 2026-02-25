import tqdm
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Acoustic
import numpy as np

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
fm = 5.0
spatial_order = 2
abcn = 20
free_surface=False

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([128, 128]).reshape(1, 2)

# receivers = np.stack((np.arange(0, nx, 1), 
#                      np.zeros(nx, dtype=np.int32)), axis=1).reshape(1, -1, 2)

# rec_x = np.arange(0, nx, 1).reshape(-1, 1)
# rec_z = np.ones_like(rec_x)*0
# receivers = np.concatenate([rec_x, rec_z], axis=1)
# receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)
receivers = np.array([384, 128]).reshape(1, 1, 2)

vp = torch.from_numpy(true_vp).float().to(device).requires_grad_()

prop = [PropCUDA, PropTorch]
pname = ['CUDA', 'PyTorch']
gradients = []
kwargs = dict(shape=vp.shape, source_type=['h1'], receiver_type=['h1'], abcn=abcn, dh=dh, dt=dt, pml_type='cpmlr', dev=device, free_surface=free_surface)
cuda_solver = PropCUDA(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)
torch_solver = PropTorch(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)

vp.grad = None
solver_kwargs = dict(wavelet=wave, sources=sources, receivers=receivers, models=[vp])
# CUDA WITHOUT BOUNDARY SAVING
out = cuda_solver(**solver_kwargs, use_boundary_saving=False)
loss = out.pow(2).sum()
loss.backward()
gradients.append((vp.grad.cpu().numpy()))
# PYTORCH AD
vp.grad = None
out = torch_solver(**solver_kwargs)
loss = out.pow(2).sum()
loss.backward()
gradients.append((vp.grad.cpu().numpy()))
# CUDA WITH BOUNDARY SAVING
vp.grad = None
out = cuda_solver(**solver_kwargs, use_boundary_saving=True)
loss = out.pow(2).sum()
loss.backward()
gradients.append((vp.grad.cpu().numpy()))

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

grads_cuda_vp = gradients[0]
grads_torch_vp = gradients[1]
grads_cuda_vp_bs = gradients[2]

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
plt.savefig(f'acoustic_grad_src{sources[0,0]}_{sources[0,1]}.png', dpi=300, bbox_inches='tight')
plt.show()

# for i in tqdm.trange(1001):
#     # vp.grad = None
#     out = solver(wave, sources = sources,
#                  receivers = receivers,
#                  models=[vp, vs, rho], 
#                  use_boundary_saving=True)
#     loss = out.pow(2).sum()
#     loss.backward()

#     # fig, ax = plt.subplots(figsize=(10, 6))
#     # record = out[-1].detach().cpu().numpy()#.squeeze()
#     # vmin, vmax = np.percentile(record, [0.5, 99.5])
#     # im = ax.imshow(record.T, cmap='seismic', aspect='auto',
#     #             extent=[0, nx * dh, nt * dt, 0], vmin=vmin, vmax=vmax)
#     # ax.set_xlabel('Time (s)')
#     # ax.set_ylabel('Depth (m)')
#     # ax.set_title('Seismic Wavefield at Source Location')
#     # fig.colorbar(im, ax=ax, label='Amplitude')
#     # plt.savefig(f'record_{ptype}.png', dpi=300)
#     # plt.show()

#     fig, axes = plt.subplots(1, 3, figsize=(18, 4))
#     grads = [vp.grad.cpu().numpy(), vs.grad.cpu().numpy(), np.zeros_like(true_vp)]
#     titles = ['Gradient of Vp', 'Gradient of Vs', 'Gradient of Density']
#     for ax, grad, title in zip(axes, grads, titles):
#         # grad[:2,:] = 0.
#         # if 'Density' in title: grad = -grad
#         vmin, vmax = np.percentile(grad, [0.5, 99.5])
#         print(grad.max(), grad.min())
#         im = ax.imshow(grad, cmap='seismic', aspect='auto',
#                     extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
#         ax.set_xlabel('Distance (m)')
#         ax.set_ylabel('Depth (m)')
#         ax.set_title(title)
#         fig.colorbar(im, ax=ax, label='Gradient')
#     plt.tight_layout()
#     plt.savefig(f'gradient_{ptype}.png', dpi=300)
#     plt.show()

#     for g, name in zip([vp.grad, vs.grad, rho.grad], ['Vp', 'Vs', 'Density']):
#         np.save(f'{name}_grad_{ptype}.npy', g.cpu().numpy())
#     break
