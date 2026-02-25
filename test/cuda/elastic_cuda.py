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

nz, nx = 256, 512
true_vp = np.ones((nz, nx), dtype=np.float32) * 2000.0
# true_vp[nz//2:, :] = 2000.0
true_vs = true_vp /1.73
rho = np.ones((nz, nx), dtype=np.float32) * 1000.0
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
spatial_orders = [8]
sourcesz = [3]
grid = list(product(spatial_orders, sourcesz))
abcn = 20
free_surface=False
use_boundary_saving = True
t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([128, 3]).reshape(1, 2)
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

prop = [PropCUDA, PropTorch]
pname = ['CUDA', 'PyTorch']
for so, srcz in grid:
    gradients = []
    sources = np.array([128, srcz]).reshape(1, 2)
    print(f"Testing spatial order {so} with source depth {srcz}...")
    for name, propagator in zip(pname, prop):
        print(f"Testing {name} implementation...")
        vp.grad=None
        vs.grad=None
        rho.grad=None
        solver = propagator(Elastic(spatial_order=so, device=device,), 
                        shape=vp.shape, 
                        source_type=['vz'],
                        receiver_type=['vz'],
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
        gradients.append((vp.grad.cpu().numpy(), vs.grad.cpu().numpy(), rho.grad.cpu().numpy()))

        # if name == 'PyTorch':
            # np.save('adjoint_wavefields_torch.npy', solver.adjoint_wavefields.cpu().numpy())

        # np.save(f'record_{name}.npy', out.detach().cpu().numpy())
        del solver, loss
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    grads_cuda = gradients[0]
    grads_torch = gradients[1]

    grads_cuda_vp, grads_cuda_vs, grads_cuda_rho = grads_cuda
    grads_torch_vp, grads_torch_vs, grads_torch_rho = grads_torch

    # Pytorch automatic differentiation
    vmin, vmax = np.percentile(grads_cuda_vp, [0.5, 99.5])
    ax = axes[0,0].imshow(grads_cuda_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[0,0])
    axes[0,0].set_title('Vp (CUDA)')

    vmin, vmax = np.percentile(grads_torch_vp, [0.5, 99.5])
    ax = axes[0,1].imshow(grads_torch_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[0,1])
    axes[0,1].set_title('Vp (Torch)')

    ax = axes[0,2].imshow(grads_cuda_vp - grads_torch_vp, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[0,2])
    axes[0,2].set_title('Vp (Difference)')


    vmin, vmax = np.percentile(grads_cuda_vs, [0.5, 99.5])
    ax = axes[1,0].imshow(grads_cuda_vs, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[1,0])
    axes[1,0].set_title('Vs (CUDA)')

    vmin, vmax = np.percentile(grads_torch_vs, [0.5, 99.5])
    ax = axes[1,1].imshow(grads_torch_vs, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[1,1])
    axes[1,1].set_title('Vs (Torch)')

    ax = axes[1,2].imshow(grads_cuda_vs - grads_torch_vs, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    plt.colorbar(ax, ax=axes[1,2])
    axes[1,2].set_title('Vs (Difference)')

    plt.tight_layout()
    plt.savefig(f'grad_srcz{srcz}_so{so}_{"bs" if use_boundary_saving else "no_bs"}.png', dpi=300, bbox_inches='tight')
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
