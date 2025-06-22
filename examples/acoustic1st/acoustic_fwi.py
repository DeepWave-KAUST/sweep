import sys, tqdm, os
import torch
sys.path.append('/ibex/user/wangs0j/aiseismic/src')
torch.backends.cudnn.benchmark = True
from geophyai.rnn import RNN
from geophyai.equations import Acoustic1st as Acoustic
from geophyai.signal import ricker
import numpy as np
import matplotlib.pyplot as plt
from configure import *

save_path = 'acoustic_fwi_l2'
if not os.path.exists(save_path):
    os.makedirs(save_path)

torch.manual_seed(0)
np.random.seed(0)

t = np.arange(0, nt*dt, dt)
true_vp = np.load(true_path)
true_rho = 0.31*true_vp**0.25*1000
smooth_vp = np.load(smooth_path)
smooth_rho = 0.31*smooth_vp**0.25*1000
shape = true_vp.shape

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
            source_type=['p'],
            receiver_type=['p'],
            abcn=abcn, 
            free_surface=free_surface, 
            use_ckpt=False)

# Set the true model
model.set_parameters([torch.from_numpy(true_vp).to(dev), torch.from_numpy(true_rho).to(dev)])

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
plt.savefig(f'{save_path}/acoustic_obs.png', dpi=300, bbox_inches='tight')
exit()
########## Inversion ##########
# Set the model
model.set_parameters([torch.from_numpy(smooth_vp).to(dev), torch.from_numpy(smooth_rho).to(dev)])

opt = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-22)
LOSS = []
for epoch in tqdm.trange(epochs):

    opt.zero_grad()

    rand_shots = np.random.randint(0, sources.shape[0], batchsize)

    syn = model(wave, sources[rand_shots], receivers[rand_shots])
    loss = (syn-torch.from_numpy(obs[rand_shots]).to(dev)).pow(2).mean()
    loss.backward()
    LOSS.append(loss.item())
    opt.step()
    print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # # Save the model
    # if epoch % show_every == 0:
    #     vmin, vmax = true_vp.min(), true_vp.max()
    #     fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    #     extent = [0, nx*dh, nz*dh, 0]
    #     ax[0].imshow(true_vp, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
    #     ax[0].set_title('True Model')
    #     ax[1].imshow(model.vp.cpu().detach().numpy(), vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
    #     ax[1].set_title('Inverted Model')
    #     grad = model.vp.grad.cpu().detach().numpy()
    #     # grad = model.get_model('vp').grad.cpu().detach().numpy()
    #     vmin,vmax=np.percentile(grad, [2, 98])
    #     ax[2].imshow(grad, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto', extent=extent)
    #     plt.tight_layout()
    #     plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    #     plt.close()

    #     fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    #     ax.plot(LOSS, c='black', label='Loss')
    #     ax.legend()
    #     plt.tight_layout()
    #     fig.savefig(f'{save_path}/loss.png', dpi=300, bbox_inches='tight')
    #     plt.close()