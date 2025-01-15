import sys, tqdm, os
import torch
# sys.path.append('../src')
torch.backends.cudnn.benchmark = True
from geophyai.rnn import RNN
from geophyai.equations import Acoustic, AcousticLSRTM
from geophyai.signal import ricker
from geophyai.loss import CosineSimilarity, MSE
import numpy as np
import matplotlib.pyplot as plt
from configure import *

save_path = 'acoustic_lsrtm'
if not os.path.exists(save_path):
    os.makedirs(save_path)

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
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
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

# Model the observed data
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
obs = obs-direct
vmin, vmax = np.percentile(obs[-1], [2, 98])
plt.imshow(obs[-1].squeeze(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/acoustic_obs.png', dpi=300, bbox_inches='tight')

########## LSRTM inversion ##########
lsrtm = RNN(AcousticLSRTM(spatial_order=spatial_order, device=dev), 
            shape=shape, 
            dev=dev, 
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['sh1'],
            abcn=abcn, 
            free_surface=free_surface)

# Set the model
lsrtm.set_parameters([torch.from_numpy(smooth_model).to(dev), # smoothed velocity model 
                      torch.from_numpy(np.zeros_like(smooth_model)).to(dev)] # reflectivity (zero initial)
                      )
criteria = CosineSimilarity(axis=1)
# criteria = MSE()
opt = torch.optim.Adam([{'params': lsrtm.get_parameters('vp'), 'lr': 0}, 
                        {'params': lsrtm.get_parameters('ref'), 'lr': 0.01}, ], 
                        eps=1e-22)
LOSS = []
for epoch in tqdm.trange(epochs):

    opt.zero_grad()

    rand_shots = np.random.randint(0, sources.shape[0], batchsize)

    # Source encoding for acceleration
    # coding_syn = lsrtm(wave, sources[rand_shots], receivers[0:1], source_encoding=True)
    # coding_obs = torch.sum(torch.from_numpy(obs[rand_shots]), dim=0).to(dev)
    # loss = (coding_syn-coding_obs).pow(2).mean()

    loss_temp = 0.
    # Accumulate the gradients, when the graph is too large to be kept in memory
    for step in range(step_per_epoch):
        rand_shots_this_step = rand_shots[batch_per_step*step:batch_per_step*(step+1)]
        syn = lsrtm(wave, sources[rand_shots_this_step], receivers[rand_shots_this_step])
        # loss = (syn-torch.from_numpy(obs[rand_shots_this_step]).to(dev)).pow(2).mean()
        loss = criteria(syn, torch.from_numpy(obs[rand_shots_this_step]).to(dev))
        loss.backward()
        loss_temp += loss.item()
    LOSS.append(loss_temp)
    lsrtm.ref.grad.data /= lsrtm.ref.grad.max()
    opt.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # Save the model
    if epoch % show_every == 0:
        vmin, vmax = true_model.min(), true_model.max()
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        extent = [0, nx*dh, nz*dh, 0]
        ax[0].imshow(true_model, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
        ax[0].set_title('True Velocity Model')
        inverted_ref = lsrtm.ref.cpu().detach().numpy()
        vmin, vmax = np.percentile(inverted_ref, [2, 98])
        ax[1].imshow(inverted_ref, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        ax[1].set_title('Inverted Reflectivity')
        grad = lsrtm.ref.grad.cpu().detach().numpy()
        vmin,vmax=np.percentile(grad, [2, 98])
        print('Gradient:', vmin, vmax)
        ax[2].imshow(grad, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.plot(LOSS, c='black', label='Loss')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')