import tqdm
import torch
import matplotlib.pyplot as plt
from deepwave import Elastic, elastic
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, nx = 256, 256
true_vp = np.ones((nz, nx), dtype=np.float32) * 1500.0
# true_vp[nz//2:, :] = 2000.0
true_vs = true_vp /1.73
rho = np.ones((nz, nx), dtype=np.float32) * 1000.0
def ricker(t, fm):
    pi2 = np.pi * 2
    wave = (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)
    return wave#.to(torch.float32)

dev = torch.device("cuda:0")
nt = 600
dt = 0.001
delay = 0.2
dh = 5.0
fm = 10.0
spatial_order = 2
abcn = 0
free_surface=False

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([128, 128]).reshape(1, 2)


rec_x = np.arange(0, nx-1, 1).reshape(-1, 1)
rec_z = np.ones_like(rec_x)*128
receivers = np.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

vp = torch.from_numpy(true_vp).float().to(device).requires_grad_()
vs = torch.from_numpy(true_vs).float().to(device).requires_grad_()
rho = torch.from_numpy(rho).float().to(device).requires_grad_()

lame_lambda = vp**2 * rho - 2 * vs**2 * rho
lame_mu = vs**2 * rho
buoyancy = 1.0 / rho
for i in tqdm.trange(1001):
    # vp.grad = None
    out = elastic(lame_lambda.T, 
                  lame_mu.T, 
                  buoyancy.T,
                  dh,
                  dt,
                  source_amplitudes_y=(torch.from_numpy(wave.reshape(1, 1, -1).repeat(sources.shape[0], 0)).to(device).float()), 
                #   source_amplitudes_x=torch.from_numpy(wave.reshape(1, 1, -1).repeat(sources.shape[0], 0)).to(device).float(),
                  source_locations_y = torch.from_numpy(np.expand_dims(sources, 1)).to(device).long(),
                  receiver_locations_y = torch.from_numpy(receivers).to(device), 
                  pml_width=20,
                  accuracy=2,
                  )
    vy, vx = out[:2]

    fig, axes = plt.subplots(1, 2, figsize=(8,3))
    vy, vx = vy.detach().cpu().numpy(), vx.detach().cpu().numpy()
    vmin, vmax = np.percentile(vy, [0.5, 99.5])
    im = axes[0].imshow(vy.squeeze().T, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('Vy')
    vmin, vmax = np.percentile(vx, [0.5, 99.5])
    im = axes[1].imshow(vx.squeeze().T, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title('Vx')
    plt.tight_layout()
    plt.savefig('wavefield_deepwave.png', dpi=300)
    plt.show()


    loss = out[-2].pow(2).sum()
    loss.backward()

    record = out[-2].detach().contiguous().cpu().numpy().squeeze()
    print(record.shape)
    fig, ax = plt.subplots(figsize=(10, 6))
    vmin, vmax = np.percentile(record, [0.5, 99.5])
    im = ax.imshow(record.T, cmap='seismic', aspect='auto',
                extent=[0, nx * dh, nt * dt, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Depth (m)')
    ax.set_title('Seismic Wavefield at Source Location')
    fig.colorbar(im, ax=ax, label='Amplitude')
    plt.savefig('record_deepwave.png', dpi=300)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    grads = [vp.grad.cpu().numpy(), vs.grad.cpu().numpy(), rho.grad.cpu().numpy()]
    titles = ['Gradient of Vp', 'Gradient of Vs', 'Gradient of Density']
    for ax, grad, title in zip(axes, grads, titles):
        # grad[:2,:] = 0.
        # if 'Density' in title: grad = -grad
        vmin, vmax = np.percentile(grad, [0.5, 99.5])
        print(grad.max(), grad.min())
        im = ax.imshow(grad, cmap='seismic', aspect='auto',
                    extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Gradient')
    plt.tight_layout()
    plt.savefig('gradient_deepwave.png', dpi=300)
    plt.show()

    for g, name in zip([vp.grad, vs.grad, rho.grad], ['Vp', 'Vs', 'Density']):
        np.save(f'{name}_grad_deepwave.npy', g.cpu().numpy())
    break

    # fig,ax=plt.subplots(figsize=(10, 6))
    # grad = vp.grad.cpu().numpy()
    # vmin,vmax= np.percentile(grad, [0.5, 99.5])
    # im = ax.imshow(grad, cmap='seismic', aspect='auto',
    #                extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
    # ax.set_xlabel('Distance (m)')
    # ax.set_ylabel('Depth (m)')
    # ax.set_title('Gradient of the velocity model')
    # fig.colorbar(im, ax=ax, label='Gradient')
    # plt.savefig('gradient_sweep.png', dpi=300)
    # plt.show()
    # np.save('vp_grad_sweep_pytorch.npy', vp.grad.cpu().numpy())
    # break

