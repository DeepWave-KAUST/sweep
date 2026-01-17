import sys, tqdm
import torch
torch.backends.cudnn.benchmark = True
from sweep.propagator.torch import PropTorch
from sweep.equations import Acoustic
from sweep.signal import ricker
import numpy as np
import matplotlib.pyplot as plt
from configure import *

torch.manual_seed(0)
np.random.seed(0)

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
model = PropTorch(Acoustic(spatial_order=spatial_order, device=dev), 
            shape=shape, 
            dev=dev, 
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            abcn=abcn, 
            free_surface=free_surface,
            use_ckpt=False,
            pml_type='cpmlr')

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

vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig('acoustic_obs.png', dpi=300, bbox_inches='tight')

########## Source Encoding Inversion ##########
receivers = receivers[0:1] # The receivers are the same for all shots with shape (1, nreceivers, 2)
print('Source shape for inversion:', sources.shape)
print('Receiver shape for inversion:', receivers.shape)
# Set the model
model.set_parameters([torch.from_numpy(smooth_model).to(dev)])

opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-22)

for epoch in tqdm.trange(epochs):

    opt.zero_grad()

    rand_shots = np.random.randint(0, sources.shape[0], batchsize)

    coding_syn = model(wave, sources[rand_shots], receivers, source_encoding=True)
    coding_obs = torch.sum(torch.from_numpy(obs[rand_shots]), dim=0).to(dev)
    loss = (coding_syn-coding_obs).pow(2).mean()
    loss.backward()
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
        # grad = model.get_model('vp').grad.cpu().detach().numpy()
        vmin,vmax=np.percentile(grad, [2, 98])
        ax[2].imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[2].set_title('Gradient')
        plt.tight_layout()
        plt.savefig(f'epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()