import tqdm
import torch
import matplotlib.pyplot as plt
from deepwave import scalar
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

nz, nx = 100, 1024
true_model = np.ones((nz, nx), dtype=np.float32) * 1500.0
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

t = np.arange(nt) * dt - delay
wave = -ricker(t, fm=fm)

sources = np.array([512, 0]).reshape(1, 2)
receivers = np.stack((np.arange(0, nx, 1), 
                     np.ones(nx, dtype=np.int32)*1), axis=1).reshape(1, -1, 2)
dh = 10.0
spatial_order = 2
vp = torch.from_numpy(true_model.T).to(device).requires_grad_()
for i in tqdm.trange(1001):
    vp.grad = None
    out = scalar(vp, 
                dh, 
                dt, 
                source_amplitudes=torch.from_numpy(wave.reshape(1, 1, -1).repeat(sources.shape[0], 0)).to(device).float(),
                source_locations=torch.from_numpy(np.expand_dims(sources, 1)).to(device).long(),
                receiver_locations=torch.from_numpy(receivers).to(device).long(),
                accuracy=spatial_order, 
                #  python_backend='eager',
                pml_width=[20, 20, 20, 20], 
                pml_freq=fm,)
    loss = out[-1].pow(2).sum()
    loss.backward()

    # np.save('record_dw.npy', out[-1].detach().cpu().numpy().squeeze().T)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # record = out[-1].detach().cpu().numpy().squeeze().T
    # vmin, vmax = np.percentile(record, [0.5, 99.5])
    # im = ax.imshow(record, cmap='seismic', aspect='auto',
    #             extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Depth (m)')
    # ax.set_title('Seismic Wavefield at Source Location')
    # fig.colorbar(im, ax=ax, label='Amplitude')
    # plt.savefig('record_dw.png', dpi=300)
    # plt.show()
    # np.save('grad_dw.npy', vp.grad.cpu().numpy().T)

    # fig,ax=plt.subplots(figsize=(10, 6))
    # grad = vp.grad.cpu().numpy().T
    # vmin,vmax= np.percentile(grad, [0.5, 99.5])
    # im = ax.imshow(grad, cmap='seismic', aspect='auto',
    #                extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
    # ax.set_xlabel('Distance (m)')
    # ax.set_ylabel('Depth (m)')
    # ax.set_title('Gradient of the velocity model')
    # fig.colorbar(im, ax=ax, label='Gradient')
    # plt.savefig('gradient_deepwave.png', dpi=300)
    # plt.show()

    # break

