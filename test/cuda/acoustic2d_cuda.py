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
nt = 3000
dt = 0.001
delay = 0.2
dh = 5.0
fm = 10.0
spatial_order = 8
abcn = 30
free_surface=False

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([[1, 0]], dtype=np.int32)

receivers = np.array([
    [[511, 0]],
], dtype=np.int32)
print('free_surface:', free_surface, sources, receivers)
print('Spatial order:', spatial_order)
print('abcn: ', abcn)
vp = torch.from_numpy(true_vp).float().to(device).requires_grad_()

prop = [PropCUDA, PropTorch]
pname = ['CUDA', 'PyTorch']
gradients = []
kwargs = dict(shape=vp.shape, 
              source_type=['h1'], 
              receiver_type=['h1'], abcn=abcn, dh=dh, dt=dt, 
              pml_type='cpmlr', dev=device, free_surface=free_surface, 
              B=1,
              allow_growth=True,
              nt=nt,
              boundary_saving_config = {
                    "enabled": False,
                    "storage": "cpu",
                    "transfer_interval": 99,
                    "pinned_memory": True,
              },
              use_ckpt=True,
              ckpt_mode="recursive",
              ckpt_num=4
              )
cuda_solver = PropCUDA(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)
torch_solver = PropTorch(Acoustic(spatial_order=spatial_order, device=device,), **kwargs)

vp.grad = None
solver_kwargs = dict(wavelet=wave, sources=sources, receivers=receivers, models=[vp])

# CUDA WITH BOUNDARY SAVING
vp.grad = None
out = cuda_solver(**solver_kwargs, use_boundary_saving=True)
print(out.max(), out.min())
loss = out.pow(2).sum()
loss.backward()
grads_cuda_vp_bs = vp.grad.cpu().numpy()
# CUDA WITHOUT BOUNDARY SAVING
vp.grad = None
out = cuda_solver(**solver_kwargs, use_boundary_saving=False)
loss = out.pow(2).sum()
loss.backward()
grads_cuda_vp = vp.grad.cpu().numpy()
# PYTORCH AD
vp.grad = None
out = torch_solver(**solver_kwargs)
loss = out.pow(2).sum()
loss.backward()
grads_torch_vp = vp.grad.cpu().numpy()

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

vmin, vmax = np.percentile(grads_cuda_vp, [0.5, 99.5])
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
plt.savefig(f'acoustic_grad.png', dpi=300, bbox_inches='tight')
plt.show()
