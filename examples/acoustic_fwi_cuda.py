import sys, tqdm, os
import torch
sys.path.append('../src')
torch.backends.cudnn.benchmark = True
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.equations import Acoustic
from sweep.signal import ricker
import numpy as np
import matplotlib.pyplot as plt
from configure import *

# Overwritre 
abcn = 30
free_surface = False

save_path = 'acoustic_fwi_l2_cuda'
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
solver = PropCUDA(Acoustic(spatial_order=spatial_order, device=dev), 
            shape=shape, 
            dev=dev, 
            dh=dh,
            dt=dt,
            source_type=['h1'],   # for cuda, no need to specify source and receiver type
            receiver_type=['h1'], # for cuda, no need to specify source and receiver type
            abcn=abcn, 
            free_surface=free_surface,
            pml_type='cpmlr')

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
    obs = solver.forward(wave, 
                        sources, 
                        receivers, 
                        models=[torch.from_numpy(true_model).to(dev)]).cpu().numpy()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print(f"Execution time: {elapsed_time:.2f} ms")
print(obs.shape)
vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze().T, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/acoustic_obs_cuda.png', dpi=300, bbox_inches='tight')
########## Inversion ##########
# Set the model

LOSS = []
vp = torch.from_numpy(smooth_model).float().to(dev).requires_grad_()
opt = torch.optim.Adam([vp], lr=lr, eps=1e-22)

for epoch in tqdm.trange(epochs):

    opt.zero_grad()
    rand_shots = np.random.randint(0, sources.shape[0], batchsize)
    rand_shots = np.arange(0, sources.shape[0], 20) # Use all shots
    syn = solver(wave, sources[rand_shots], receivers[rand_shots], models=[vp], use_boundary_saving=False)
    loss = (syn-torch.from_numpy(obs[rand_shots]).to(dev)).pow(2).mean()
    loss.backward()
    LOSS.append(loss.item())
    opt.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # Save the model
    if epoch % show_every == 0:
        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Model')
        ax[1].imshow(vp.cpu().detach().numpy(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Model')
        grad = vp.grad.cpu().detach().numpy()
        # grad = model.get_model('vp').grad.cpu().detach().numpy()
        vmin,vmax=np.percentile(grad, [2, 98])
        ax[2].imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')
        plt.close()