from random import random
import sys, tqdm, os
sys.path.append('../src')
import torch
torch.backends.cudnn.benchmark = True
from sweep.propagator.torch import PropTorch
from sweep.propagator.cuda import PropCUDA
from sweep.equations import Elastic
from sweep.signal import ricker

import numpy as np
import matplotlib.pyplot as plt
from configure_ot import *

np.random.seed(0)
torch.manual_seed(0)


def cuda_record_to_standard(record):
    return np.transpose(record, (1, 3, 2, 0))


def standard_record_to_cuda(record):
    return np.transpose(record, (3, 0, 2, 1))

save_path = 'elastic_cuda_ot'
if not os.path.exists(save_path):
    os.makedirs(save_path)

t = np.arange(0, int(nt)*dt, dt)

vp_true = np.load(true_path)#[::2, ::2]
vs_true = vp_true/1.732
rho_true = np.ones_like(vp_true)*1000

vp_smooth = np.load(smooth_path)#[::2, ::2]

shape = vp_true.shape
extent = [0, shape[1]*dh, shape[0]*dh, 0]

nz, nx = shape
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wave = ricker(t-delay, f=fm)# * 1e6
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = PropCUDA(Elastic(spatial_order=spatial_order, device=dev), 
            shape=shape, 
            dev=dev, 
            abcn=abcn, 
            dh=dh,
            dt=dt,
            source_type=['sxx', 'szz'],
            receiver_type=['vx', 'vz'],
            free_surface=False, 
            pml_type='cpmls',
            boundary_saving_config={
                "enabled":  True,
                "storage": "gpu",
                "transfer_interval": 10,
                "pinned_memory": True,
            },
            )

# Set the true model, the order of the parameters should be 
# the same as the model names in func <geophyai.equations.elastic.models>
# model.set_parameters([torch.from_numpy(vp_true).to(dev), 
#                       torch.from_numpy(vs_true).to(dev), 
#                       torch.from_numpy(rho_true).to(dev)])

# Geometry
src_x = np.arange(0,nx, src_step*2).reshape(-1, 1)
src_z = np.ones_like(src_x)*srcz
sources = np.concatenate([src_x, src_z], axis=1) # (nshots, 2)
rec_x = np.arange(0, nx, rec_step*2).reshape(-1, 1)
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
                        receivers, 
                        models=[torch.from_numpy(vp_true).to(dev), 
                                torch.from_numpy(vs_true).to(dev), 
                                torch.from_numpy(rho_true).to(dev)]).cpu().numpy()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)

print(f'Saved observed data to {obs_path}')
print(f"Execution time: {elapsed_time:.2f} ms")
# # print(obs.shape)
# vmin, vmax = np.percentile(obs[:,-1], [2, 98])
# plt.imshow(obs[:,-1].squeeze().T, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
# plt.colorbar()
# plt.tight_layout()
# plt.savefig(f'{save_path}/elastic_vz.png', dpi=300, bbox_inches='tight')
# plt.close()
vmin, vmax = np.percentile(obs[:,-1][0], [2, 98])
plt.imshow(obs[:,-1].squeeze()[0].T, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/elastic_vx.png', dpi=300, bbox_inches='tight')
plt.close()
vmin, vmax = np.percentile(obs[:,-1][1], [2, 98])
plt.imshow(obs[:,-1].squeeze()[1].T, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/elastic_vz.png', dpi=300, bbox_inches='tight')
plt.close()


# ########## Inversion ##########
# Set the model
vp = torch.from_numpy(vp_smooth).float().to(dev)#.requires_grad_()
# vp[0:2] = torch.from_numpy(vp_true[0:2]).float().to(dev) # Fix the top 2 layers for better convergence
vs = torch.from_numpy(vp_smooth/1.732).float().to(dev)#.requires_grad_()
# vs[0:2] = torch.from_numpy(vs_true[0:2]).float().to(dev) # Fix the top 2 layers for better convergence

vp.requires_grad_()
vs.requires_grad_()
rho = (torch.ones_like(vp)*1000).float().to(dev).requires_grad_()

# model = torch.compile(model, backend='tensorrt')
# Set different lr to different parameters
opt = torch.optim.Adam([{'params': [vp], 'lr': lr}, 
                        {'params': [vs], 'lr': lr/1.73}, 
                        {'params': [rho], 'lr': 0}], 
                        eps=1e-22)
# model = torch.compile(model)
shot_rng = np.random.RandomState(0)
shot_schedule = [
    shot_rng.choice(sources.shape[0], size=batchsize, replace=False)
    for _ in range(epochs)
]
for epoch in tqdm.trange(epochs):

    with torch.no_grad():
        vp[:2] = torch.from_numpy(vp_true[:2]).float().to(dev)
        vs[:2] = torch.from_numpy(vs_true[:2]).float().to(dev)

    opt.zero_grad()

    rand_shots = shot_schedule[epoch]

    _syn = model(wave, sources[rand_shots], receivers[rand_shots], models=[vp, vs, rho])
    _obs = torch.from_numpy(obs[:,rand_shots]).to(dev)

    loss = (_obs-_syn).pow(2).sum()
    loss.backward()

    opt.step()

    # Save the model
    if epoch % 10 == 0:

        vmin_vp, vmax_vp = vp_true.min(), vp_true.max()
        vmin_vs, vmax_vs = vs_true.min(), vs_true.max()
        fig, axes = plt.subplots(3, 2, figsize=(8, 9))
        show_data = [vp_true, vs_true, 
                     vp.detach().cpu().numpy(), vs.detach().cpu().numpy(), 
                     vp.grad.detach().cpu().numpy(), vs.grad.detach().cpu().numpy()]
        titles = ['True Vp', 'True Vs', 'Inverted Vp', 'Inverted Vs', 'Gradient Vp', 'Gradient Vs']
        for ax, data, title in zip(axes.ravel(), show_data, titles):
            if 'vp' in title.lower():
                vmin, vmax = vmin_vp, vmax_vp
            else:
                vmin, vmax = vmin_vs, vmax_vs
            if 'gradient' in title.lower():
                vmin, vmax = np.percentile(data, [2, 98])
            plt.colorbar(ax.imshow(data, vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto'))
            ax.set_title(title)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        
