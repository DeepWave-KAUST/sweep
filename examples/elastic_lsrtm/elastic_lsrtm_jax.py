import sys, tqdm, os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
sys.path.append('../../src')
import torch, optax, jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
torch.backends.cudnn.benchmark = True
from geophyai.rnn import RNNJax
from geophyai.equations import ElasticLSRTM, Elastic
from geophyai.signal import ricker
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
key = random.PRNGKey(0)

save_path = 'elastic_lsrtm_jax'
if not os.path.exists(save_path):
    os.makedirs(save_path)

fm = 6
dh = 20.
dt = 0.001
nt = 4000
spatial_order = 4
delay = 256*dt
abcn = 50
epochs = 101
batchsize = 8
show_every = 10

t = np.arange(0, int(nt)*dt, dt)

vp_true = np.load('/ibex/user/wangs0j/geophyai/examples/elastic_lsrtm/velocity/true_vp.npy')#[::2, ::2]
vs_true = np.load('/ibex/user/wangs0j/geophyai/examples/elastic_lsrtm/velocity/true_vs.npy')#[::2, ::2]
rho_true = np.ones_like(vp_true)*1000

vp_smooth = np.load('/ibex/user/wangs0j/geophyai/examples/elastic_lsrtm/velocity/smooth_vp.npy')#[::2, ::2]
vs_smooth = np.load('/ibex/user/wangs0j/geophyai/examples/elastic_lsrtm/velocity/smooth_vs.npy')#[::2, ::2]

sources = jnp.load('/ibex/user/wangs0j/geophyai/examples/elastic_lsrtm/geometry/sources.npy')
receivers = jnp.load('/ibex/user/wangs0j/geophyai/examples/elastic_lsrtm/geometry/receivers.npy')

shape = vp_true.shape
extent = [0, shape[1]*dh, shape[0]*dh, 0]

nz, nx = shape
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

wave = ricker(t-delay, f=fm)
plt.plot(wave)
plt.savefig(f'{save_path}/ricker.png', dpi=300, bbox_inches='tight')
plt.close()

# Forward model for observed data
solver = RNNJax(Elastic(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=dev, 
            abcn=abcn, 
            dh=dh,
            dt=dt,
            source_type=['vz'],
            receiver_type=['vx', 'vz'],
            free_surface=False, 
            use_ckpt=True)

born = RNNJax(ElasticLSRTM(spatial_order=spatial_order, backend='jax'), 
            shape=shape, 
            dev=dev, 
            abcn=abcn, 
            dh=dh,
            dt=dt,
            source_type=['vz'],
            receiver_type=['vxs', 'vzs'],
            free_surface=False, 
            use_ckpt=True)

# Set the true model, the order of the parameters should be 
# the same as the model names in func <geophyai.equations.elastic.models>

true_mp = (vp_true-vp_smooth)/vp_true
true_ms = (vs_true-vs_smooth)/vs_true

solver.set_parameters([jnp.array(vp_true), 
                      jnp.array(vs_true), 
                      jnp.array(rho_true)])

born.set_parameters([jnp.array(true_mp), 
                     jnp.array(true_ms),
                     jnp.array(vp_smooth), 
                     jnp.array(vs_smooth),
                     jnp.array(rho_true), ])

print("(Number of shots, dimension)", sources.shape)
print("(Number of shots, number of receivers, dimension)", receivers.shape)

obs = solver.forward(wave, 
                    sources, 
                    receivers)
# obs = born.forward(wave,
#                    sources,
#                    receivers,)
# print(obs.shape)

vmin, vmax = np.percentile(obs[-1][...,0], [2, 98])
plt.imshow(obs[-1].squeeze()[...,0], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/elastic_vx.png', dpi=300, bbox_inches='tight')
plt.close()
vmin, vmax = np.percentile(obs[-1][...,1], [2, 98])
plt.imshow(obs[-1].squeeze()[...,1], vmin=vmin, vmax=vmax, cmap='seismic', aspect='auto')
plt.colorbar()
plt.tight_layout()
plt.savefig(f'{save_path}/elastic_vz.png', dpi=300, bbox_inches='tight')
plt.close()

# ########## Inversion ##########
born.set_parameters([jnp.array(np.zeros_like(vp_smooth)), 
                     jnp.array(np.zeros_like(vs_smooth)),
                     jnp.array(vp_smooth), 
                     jnp.array(vs_smooth), 
                     jnp.array(rho_true), ])
opt = optax.adam(0.01, eps=1e-22)
@jax.jit
def update_fn(param, grads, state):
    updates, state = opt.update(grads, state)
    param = optax.apply_updates(param, updates)
    return param, state

mp_state = opt.init(born.mp)
ms_state = opt.init(born.ms)

@jax.jit
def fwi_step(mvp, mvs, vp, vs, rho, rand_shots):
    # @jax.jit
    def loss_fn(mvp, mvs, vp, vs, rho, shot_nums):
        # Forward modeling
        syn = born(wave,
                   sources=sources[shot_nums], 
                   receivers=receivers[0:1], 
                   source_encoding=True,
                   models=[mvp, mvs, vp, vs, rho])
        _obs = jnp.sum(obs[shot_nums], axis=0)
        # _loss_ = jnp.mean(1-optax.losses.cosine_similarity(syn, obs, axis=1, epsilon=1e-8))
        _loss_ = jnp.mean((syn - _obs)**2)

        return _loss_, (syn, _obs)
    # Compute the gradient
    (loss, data), gradients = jax.value_and_grad(loss_fn, argnums=(0,1,2,3,4), has_aux=True)(mvp, mvs, vp, vs, rho, rand_shots)
    return loss, gradients, data


LOSS = []
for epoch in tqdm.trange(epochs):

    rand_shots = np.random.randint(0, sources.shape[0], batchsize)

    key, subkey = random.split(key)
    rand_shots = random.randint(subkey, (batchsize,), 0, sources.shape[0])

    loss, grads, (_syn, _obs) = fwi_step(born.mp, born.ms, born.vp, born.vs, born.rho, rand_shots)
    born.mp, mp_state = update_fn(born.mp, grads[0]/grads[0].max(), mp_state)
    born.ms, ms_state = update_fn(born.ms, grads[1]/grads[1].max(), ms_state)

    print(f'Epoch: {epoch}, Loss: {loss}')
    LOSS.append(loss)


    # Save the model
    if epoch % show_every == 0:

        fig, axes = plt.subplots(2, 2, figsize=(8, 9))
        show_data = [born.mp, born.ms, 
                     grads[0], grads[1]]
        vmin, vmax = np.percentile(true_mp, [2, 98])
        axes[0,0].imshow(born.mp, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        axes[0,0].set_title('Inverted mp')
        vmin, vmax = np.percentile(true_ms, [2, 98])
        axes[0,1].imshow(born.ms, vmin=vmin, vmax=vmax, cmap='gray', aspect='auto', extent=extent)
        axes[0,1].set_title('Inverted ms')
        axes[1,0].plot(born.mp[:,100], label='Inverted mp')
        axes[1,0].plot(true_mp[:,100], label='True mp')
        axes[1,0].set_title('Inverted mp vs True mp')
        axes[1,0].legend()
        axes[1,1].plot(born.ms[:,100], label='Inverted ms')
        axes[1,1].plot(true_ms[:,100], label='True ms')
        axes[1,1].set_title('Inverted ms vs True ms')
        axes[1,1].legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()

    # break

        