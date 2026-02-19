import tqdm
import torch
import matplotlib.pyplot as plt
from sweep.propagator.cuda import PropCUDA
from sweep.equations import Acoustic
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, ny, nx = 99, 100, 201
true_model = np.ones((nz, ny, nx), dtype=np.float32) * 1500.0
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

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([nx//2, ny//2, 0]).reshape(1, 3)

# receivers = np.stack((np.arange(0, nx, 1), 
#                      np.zeros(nx, dtype=np.int32)), axis=1).reshape(1, -1, 2)

recx, recy = np.meshgrid(np.arange(0, nx, 10), np.arange(0, ny, 10))
rec_z = np.ones_like(recx)
receivers = np.concatenate([recx.reshape(-1, 1), recy.reshape(-1, 1), rec_z.reshape(-1, 1)], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)
print(sources.shape, receivers.shape)
vp = torch.from_numpy(true_model).float().to(device).requires_grad_()

solver = PropCUDA(Acoustic(spatial_order=spatial_order, device=device,), 
                   shape=vp.shape, 
                   source_type=['h1'],
                   receiver_type=['h1'],
                   abcn=20, 
                   dh = dh,
                   dt = dt,
                   pml_type='cpmlr',
                   dev=device,
                   )

for i in tqdm.trange(1001):
    vp.grad = None
    out = solver(wave, sources = sources,
                 receivers = receivers,
                 models=[vp], 
                 use_boundary_saving=True)
    loss = out.pow(2).sum()
    loss.backward()

    fig, ax = plt.subplots(figsize=(10, 6))
    record = out[-1].detach().cpu().numpy().squeeze()
    vmin, vmax = np.percentile(record, [0.5, 99.5])
    im = ax.imshow(record, cmap='seismic', aspect='auto',
                extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Seismic Wavefield at Source Location')
    fig.colorbar(im, ax=ax, label='Amplitude')
    plt.savefig('record_sweep.png', dpi=300)
    plt.show()

    fig,ax=plt.subplots(figsize=(10, 6))
    grad = vp.grad.cpu().numpy()
    vmin,vmax= np.percentile(grad, [0.5, 99.5])
    im = ax.imshow(grad, cmap='seismic', aspect='auto',
                   extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Gradient of the velocity model')
    fig.colorbar(im, ax=ax, label='Gradient')
    plt.savefig('gradient_sweep.png', dpi=300)
    plt.show()
    np.save('vp_grad_sweep_pytorch.npy', vp.grad.cpu().numpy())
    break

