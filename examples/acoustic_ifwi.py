import sys, tqdm
import torch
# sys.path.append('../src')
torch.backends.cudnn.benchmark = True
from geophyai.rnn import RNN
from geophyai.equations import Acoustic
from geophyai.signal import ricker
from geophyai.networks import SineMLP
import numpy as np
import matplotlib.pyplot as plt
from configure import *

torch.manual_seed(1)
np.random.seed(1)

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

vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig('acoustic_obs.png', dpi=300, bbox_inches='tight')
plt.close()
########## Source Encoding Inversion ##########
# The receivers are the same for all shots with shape (1, nreceivers, 2)
print('Source shape for inversion:', sources.shape)
print('Receiver shape for inversion:', receivers.shape)

# Set the model
nnmlp = SineMLP(6, 
                128, 
                1, 
                omega=10,
                use_hash=False, 
                hash_config=None).to(dev)

def grid_init(shape, vmin=0., vmax=1.):
    """Initialize the mesh of the domain

    Args:
        shape (tuple): The shape of the domain.
        vmin (float): The minimum value of the domain.
        vmax (float): The maximum value of the domain.

    Returns:
        Array : The mesh of the domain with shape (*shape, 2).
    """
    coord_axis = [np.linspace(vmin, vmax, d) for d in shape]
    grid = np.stack(np.meshgrid(*coord_axis, indexing='ij'), -1)
    return grid.astype(np.float32)

grid = torch.from_numpy(grid_init(shape, vmin=0., vmax=1.)).to(dev)

opt = torch.optim.Adam(nnmlp.parameters(), lr=0.0001, eps=1e-22)

for epoch in tqdm.trange(epochs):

    opt.zero_grad()

    rand_shots = np.random.randint(0, sources.shape[0], batchsize)
    vp = nnmlp(grid).view(shape)*1000+3000
    # Source Encoding
    # coding_syn = model(wave, sources[rand_shots], receivers, 0., 0., source_encoding=True, models=[vp])
    # coding_obs = torch.sum(torch.from_numpy(obs[rand_shots]), dim=0).to(dev)
    # loss = (coding_syn-coding_obs).pow(2).mean()

    # Conventional FWI
    syn = model(wave, sources[rand_shots], receivers[rand_shots], models=[vp])
    _obs = torch.from_numpy(obs[rand_shots]).to(dev)
    loss = (syn-_obs).pow(2).mean()
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
        # ax[1].imshow(model.vp.cpu().detach().numpy(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        # ax[1].set_title('Inverted Model')
        ax[1].imshow((nnmlp(grid).view(shape)*1000+3000).detach().cpu().numpy(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Model')
        # grad = init.grad.cpu().detach().numpy()
        # # grad = model.get_model('vp').grad.cpu().detach().numpy()
        # vmin,vmax=np.percentile(grad, [5, 95])
        # ax[2].imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        # ax[2].set_title('Gradient')
        plt.tight_layout()
        plt.savefig(f'epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()