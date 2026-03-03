import tqdm
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Elastic3D
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, ny, nx = 25, 25,25
true_vp = np.ones((nz, ny, nx), dtype=np.float32) * 1500.0
true_vs = true_vp/1.73
true_rho = np.ones((nz, ny, nx), dtype=np.float32) * 1000.0
# true_vp[nz//2:, :] = 2000.0
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
spatial_order = 2
abcn = 10
t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([nx//2, ny//2, 25]).reshape(1, 3)

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

prop = [PropCUDA, PropTorch]
pnames = ['CUDA', 'PyTorch']

eq_kwargs = dict(spatial_order=spatial_order, device=device)
prop_kwargs = dict(shape=vp.shape, source_type=['vz'], receiver_type=['vz'], abcn=abcn, dh=dh, dt=dt, pml_type='cpmls', dev=device, free_surface=False)

for i in tqdm.trange(1001):

    for Prop, pname in zip(prop, pnames):
        solver = Prop(Elastic3D(**eq_kwargs), **prop_kwargs)
        vp.grad = None
        vs.grad = None
        rho.grad = None
        out = solver(wave, sources = sources,
                        receivers = receivers,
                        models=[vp, vs, rho], 
                        use_boundary_saving=True)
        print(out.max(), out.min())

        fig, ax = plt.subplots(figsize=(10, 6))
        record = out[-1].detach().cpu().numpy().squeeze()
        if pname == 'CUDA': record = record.T
        vmin, vmax = np.percentile(record, [0.5, 99.5])
        im = ax.imshow(record, cmap='seismic', aspect='auto',
                    extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Seismic Wavefield at Source Location')
        fig.colorbar(im, ax=ax, label='Amplitude')
        plt.savefig(f'record_{pname}.png', dpi=300)
        plt.close()

        loss = out.pow(2).sum()
        loss.backward()

        for grad, mname in zip([vp.grad, vs.grad, rho.grad], ['vp', 'vs', 'rho']):
            grad = grad.cpu().numpy()
            fig,ax=plt.subplots(1, 3, figsize=(12, 3))
            vmin,vmax= np.percentile(grad, [0.5, 99.5])
            ax[0].set_title(f'Gradient of {mname}')
            im = ax[0].imshow(grad[nz//2, :, :], cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax[0])
            ax[1].set_title(f'Gradient of {mname} (X slice)')
            im = ax[1].imshow(grad[:, ny//2, :], cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax[1])
            ax[2].set_title(f'Gradient of {mname} (Y slice)')
            im = ax[2].imshow(grad[:, :, nx//2], cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
            fig.colorbar(im, ax=ax[2])
            plt.savefig(f'3d_{pname}_{mname}.png', dpi=300)
            plt.close()
    #     time.sleep(100)
    break

