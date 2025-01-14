import sys, tqdm
import torch
# sys.path.append('../src')
torch.backends.cudnn.benchmark = True
from geophyai.rnn import RNN
from geophyai.equations import Acoustic
from geophyai.signal import ricker
import numpy as np
import matplotlib.pyplot as plt
from configure import *
from scipy.ndimage import laplace

torch.manual_seed(0)
np.random.seed(0)

# Overwrite configures
fm = 10
spatial_order = 10
batchsize = 8

t = np.arange(0, nt*dt, dt)
true_model = np.load(true_path)
smooth_model = np.load(smooth_path)
shape = true_model.shape

nz, nx = shape
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig('ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = RNN(Acoustic(spatial_order=spatial_order, device=dev), 
            shape=shape, 
            dev=dev, 
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            abcn=abcn, 
            free_surface=free_surface)

# Set the true model
model.set_parameters([torch.from_numpy(true_model).to(dev)])

# Geometry
src_x = np.arange(0,nx, src_step).reshape(-1, 1)
src_z = np.ones_like(src_x)*srcz
sources = np.concatenate([src_x, src_z], axis=1) # (nshots, 2)
rec_x = np.arange(0, nx, rec_step).reshape(-1, 1)
rec_z = np.ones_like(rec_x)*recz
receivers = np.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

with torch.no_grad():
    obs = model.forward(wave, 
                        sources, 
                        receivers).cpu().numpy()

end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print(f"Execution time: {elapsed_time:.2f} ms")

# Direct wave
model.set_parameters([torch.from_numpy(np.ones_like(smooth_model)*1500).to(dev)])
with torch.no_grad():
    direct = model.forward(wave, 
                           sources, 
                           receivers).cpu().numpy()

vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig('acoustic_obs.png', dpi=300, bbox_inches='tight')

########## Inversion ##########
# Set the model
model.set_parameters([torch.from_numpy(smooth_model).to(dev)])

opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-22)
wave_for_rtm = np.cumsum(np.cumsum(wave, axis=0), axis=0)
# sill = torch.zeros_like(model.vp)
# rill = torch.zeros_like(model.vp)
for epoch in tqdm.trange(1):
    # sill.zero_()
    # rill.zero_()
    opt.zero_grad()

    steps = int(np.ceil(sources.shape[0]/batchsize))
    for step in tqdm.trange(steps):
        shots = np.arange(step*batchsize, min((step+1)*batchsize, sources.shape[0]))
        # syn = model(wave_for_rtm, sources[shots], receivers[shots], sill=sill, rill=rill)
        syn = model(wave_for_rtm, sources[shots], receivers[shots])
        loss = (syn*torch.from_numpy(obs[shots]-direct[shots]).to(dev)).mean()
        loss.backward()

    # Pseudo-Hessian
    # scale = torch.sqrt(sill*rill)
    # model.vp.grad /= (scale+1e-22)

    opt.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # Save the model
    if epoch % show_every == 0:
        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Model')
        ax[1].imshow(model.vp.cpu().detach().numpy(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Model')
        grad = model.vp.grad.cpu().detach().numpy()
        vmin,vmax=np.percentile(grad, [2, 98])
        ax[2].imshow(grad, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        plt.tight_layout()
        plt.savefig(f'epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
        np.save('gradient.npy', model.vp.grad.cpu().detach().numpy())

rtm=np.load('gradient.npy')
true = np.load('marmousi_true.npy')
smooth = np.load('marmousi_smooth.npy')
rtm = laplace(rtm)
dh = 25
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
extent = [0, rtm.shape[1]*dh, rtm.shape[0]*dh, 0]
vmin, vmax = true.min(), true.max()
axes[0].imshow(true, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
axes[0].set_title('True Model')

axes[1].imshow(smooth, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
axes[1].set_title('Smoothed Model for RTM')

vmin, vmax = np.percentile(rtm, [5, 95])
axes[2].imshow(rtm, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto', extent=extent)
axes[2].set_title('RTM Gradient')
for ax in axes:
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')

plt.tight_layout()
plt.savefig('rtm.png', dpi=300, bbox_inches='tight')
plt.show()