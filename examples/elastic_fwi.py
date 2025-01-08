import sys, tqdm
import torch
torch.backends.cudnn.benchmark = True
from geophyai.rnn import RNN
from geophyai.equations import Elastic
from geophyai.signal import ricker
import numpy as np
import matplotlib.pyplot as plt
from configure import *

t = np.arange(0, int(nt//2)*dt, dt)

vp_true = np.load(true_path)[::2, ::2]
vs_true = vp_true/1.732
rho_true = np.ones_like(vp_true)*1000

vp_smooth = np.load(smooth_path)[::2, ::2]

shape = vp_true.shape
extent = [0, shape[1]*dh, shape[0]*dh, 0]

nz, nx = shape
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig('ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
model = RNN(Elastic(spatial_order=spatial_order, device=dev), 
            shape=shape, 
            dev=dev, 
            abcn=abcn, 
            free_surface=free_surface)

# Set the true model, the order of the parameters should be 
# the same as the model names in func <geophyai.equations.elastic.models>
model.set_parameters([torch.from_numpy(vp_true).to(dev), 
                      torch.from_numpy(vs_true).to(dev), 
                      torch.from_numpy(rho_true).to(dev)])

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

model.geom = dict(h=dh, 
                  dt=dt, 
                  spatial_order=spatial_order, 
                  source_type = ['vz'],
                  receiver_type = ['vx', 'vz'])

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
with torch.no_grad():
    obs = model.forward(wave, 
                        sources, 
                        receivers,
                        0., 0.).cpu().numpy()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
print(f"Execution time: {elapsed_time:.2f} ms")
# print(obs.shape)

vmin, vmax = np.percentile(obs[-1][...,0], [2, 98])
plt.imshow(obs[-1].squeeze()[...,0], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig('elastic_vx.png', dpi=300, bbox_inches='tight')
plt.close()
vmin, vmax = np.percentile(obs[-1][...,1], [2, 98])
plt.imshow(obs[-1].squeeze()[...,1], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig('elastic_vz.png', dpi=300, bbox_inches='tight')
plt.close()


# ########## Inversion ##########
# Set the model
model.set_parameters([torch.from_numpy(vp_smooth).to(dev), 
                      torch.from_numpy(vp_smooth/1.732).to(dev), 
                      torch.from_numpy(rho_true).to(dev)])

# model = torch.compile(model, backend='tensorrt')
# Set different lr to different parameters
opt = torch.optim.Adam([{'params': model.get_parameters('vp'), 'lr': lr}, 
                        {'params': model.get_parameters('vs'), 'lr': lr/1.73}, 
                        {'params': model.get_parameters('rho'), 'lr': 0}], 
                        eps=1e-22)

for epoch in tqdm.trange(epochs):

    opt.zero_grad()

    rand_shots = np.random.randint(0, sources.shape[0], batchsize)
    # Accumulate the gradients, when the graph is too large to be kept in memory
    for step in range(step_per_epoch):
        rand_shots_this_step = rand_shots[batch_per_step*step:batch_per_step*(step+1)]
        syn = model(wave, sources[rand_shots_this_step], receivers[rand_shots_this_step], 0., 0.)
        loss = (syn-torch.from_numpy(obs[rand_shots_this_step]).to(dev)).pow(2).mean()
        loss.backward()
    opt.step()

    # Save the model
    if epoch % show_every == 0:
        vmin_vp, vmax_vp = vp_true.min(), vp_true.max()
        vmin_vs, vmax_vs = vs_true.min(), vs_true.max()
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        show_data = [vp_true, vs_true, model.vp.detach().cpu().numpy(), model.vs.detach().cpu().numpy()]
        titles = ['True Vp', 'True Vs', 'Inverted Vp', 'Inverted Vs']
        for ax, data, title in zip(axes.ravel(), show_data, titles):
            if 'vp' in title.lower():
                vmin, vmax = vmin_vp, vmax_vp
            else:
                vmin, vmax = vmin_vs, vmax_vs
            ax.imshow(data, cmap='seismic', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Z (m)')
        plt.tight_layout()
        plt.savefig(f'epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

        