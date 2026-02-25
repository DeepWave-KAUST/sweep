import jax, sys
import numpy as np
import jax.numpy as jnp
from jax import lax
sys.path.append('/ibex/user/wangs0j/geophyai/build/lib.linux-x86_64-cpython-312')
import matplotlib.pyplot as plt

from sweep.operators.general import PartialDerivative
import sys
pd = PartialDerivative(spatial_order=2, backend='jax')


def step_elastic(vx, vz, txx, tzz, txz, vp, vs, rho, dt, h, b, vpill, vsill):

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    c = 0.5*dt*b

    vx_x = pd.x_forward(vx)
    vz_z = pd.z_backward(vz)
    vx_z = pd.z_forward(vx)
    vz_x = pd.x_backward(vz)

    y_txx  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = pd.x_backward(y_txx)
    txz_z = pd.z_backward(y_txz)
    tzz_z = pd.z_forward(y_tzz)
    txz_x = pd.x_forward(y_txz)

    y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz

# Pseudo-Hessian-based Gradient Illumination
def step_fwd_elastic(*args):
    """Forward step of the wave equation, used for vjp."""
    wavefields, vjp_fun = jax.vjp(step_elastic, *args)
    return wavefields, args[0:9]+(vjp_fun,)

def step_bwd_elastic(res, g):
    # res: vx, vz, vp, vs, rho, vjp_fun
    vjp_fun = res[-1]
    grads = vjp_fun(g)
    print(jax.make_jaxpr(vjp_fun)(g))
    vx, vz, txx, tzz, txz = res[:5]
    vp, vs, rho, dt = res[5:9]
    VX, VZ, TXX, TZZ, TXZ = grads[:5]

    vx_x = pd.x_forward(vx)
    vz_z = pd.z_backward(vz)
    vx_z = pd.z_forward(vx)
    vz_x = pd.x_backward(vz)

    # The gradients are the same as the A.D.
    grad_lambda = (TXX+TZZ)*(vx_x+vz_z)
    grad_mu = (TXX+TZZ)*(vx_x+vz_z)+(TXX-TZZ)*(vx_x-vz_z)+TXZ*(vx_z+vz_x)

    # ill_lambda = (txx+tzz)*(vx_x+vz_z)
    # ill_mu = (txx+tzz)*(vx_x+vz_z)+(txx-tzz)*(vx_x-vz_z)+txz*(vx_z+vz_x)

    ill_lambda = (vx_x+vz_z)**2
    ill_mu = 2*(vx_x+vz_z)**2+2*(vx_x-vz_z)**2+(vx_z+vz_x)**2

    grad_vp = 2*rho*vp*grad_lambda # The same as the A.D. expect for the amplitude
    grad_vs = -4*rho*vs*grad_lambda + 2*rho*vs*grad_mu # The same as the A.D. expect for the amplitude

    ill_vp = 2*rho*vp*ill_lambda
    # ill_vs = 2*rho*jnp.clip(vs, 0, None)*ill_mu
    ill_vs = 2*rho*vs*ill_mu

    # Reconstruction of the gradient
    grads = grads[:5] + (grad_vp[0,0], grad_vs[0,0]) + grads[7:]
    grads = grads[:-2] + (ill_vp[0,0], ill_vs[0,0])

    return grads


def forward_elastic(wave, vp, vs, rho, b, sources, receivers, dt, h, abcn, vpill=None, vsill=None, source_encoding=False, **kwargs):
    """Forward modeling of the elastic wave equation.

    Args:
        wave (jnp.array): The wavelet with shape (nt,).
        vp (jnp.array): P-wave velocity with shape (nz, nx).
        vs (jnp.array): S-wave velocity with shape (nz, nx).
        rho (jnp.array): Density with shape (nz, nx).
        b (jnp.array): ABC coefficient with shape (nz,nx).
        sources (jnp.array): Source locations with shape (nshots, 2). The second dimension is the x and z coordinates.
        receivers (jnp.array): Receiver locations with shape (nshots, nreceivers, 2). The second dimension is the x and z coordinates.
        dt (float): Time step for wavelet, modeling and recording.
        h (float): Grid spacing.
        spatial_order (int, optional): The spatial order of the finite difference operator. Defaults to 8.
        sillu (jnp.array): Source wavefield illumination.
        rillu (jnp.array): Residual wavefield illumination.

    Returns:
        jnp.array: The recorded data with shape (nshots, nt, nreceivers).
    """
    nt = wave.shape[0]
    nshots, nreceivers, _ = receivers.shape
    batchsize = 1 if source_encoding else nshots
    wavefields = [jnp.zeros((nshots, 1, *vp.shape), jnp.float32) for _ in range(5)]
    rec = jnp.zeros((batchsize, nt, nreceivers, 1), jnp.float32)

    shots = jnp.arange(0, batchsize, 1)

    sources = sources + abcn
    receivers = receivers + abcn
    srcx, srcz = zip(*sources)
    recx, recz = receivers[..., 0].flatten().astype(jnp.int32), receivers[..., 1].flatten().astype(jnp.int32)

    # _step = step_elastic
    _step = jax.custom_vjp(step_elastic)
    _step.defvjp(step_fwd_elastic, step_bwd_elastic)
    wave = wave.astype(jnp.float32)

    idxx = jnp.array([[i]*nreceivers for i in range(batchsize)], jnp.int32).flatten()
    source_mask = jnp.zeros((batchsize, 1, *vp.shape), jnp.float32)
    source_mask = source_mask.at[shots, 0, srcz, srcx].set(1)

    def step_fn(carry, it):
        vx, vz, txx, tzz, txz, rec = carry
        source = wave[it] * source_mask
        vz = vz + source
        vx, vz, txx, tzz, txz = _step(vx, vz, txx, tzz, txz, vp, vs, rho, dt, h, b, vpill, vsill)
        rec = rec.at[:, it, :, 0].set(vz[idxx, 0, recz, recx].reshape(batchsize, nreceivers))
        return (vx, vz, txx, tzz, txz, rec), None

    vx, vz, txx, tzz, txz = wavefields
    initial_carry = (vx, vz, txx, tzz, txz, rec)

    final_carry, _ = lax.scan(step_fn, initial_carry, jnp.arange(nt))
    rec_final = final_carry[-1]

    return rec_final


nz, nx = 100, 512
true_vp = np.ones((nz, nx), dtype=np.float32) * 1500.0
true_vp[nz//2:, :] = 2000.0
true_vs = true_vp /1.73
rho = np.ones((nz, nx), dtype=np.float32) * 1000.0
def ricker(t, fm):
    pi2 = np.pi * 2
    wave = (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)
    return wave#.to(torch.float32)


nt = 1500
dt = 0.001
delay = 0.2
dh = 5.0
fm = 5.0
spatial_order = 4
abcn = 0
free_surface=False

t = np.arange(nt) * dt - delay
wave = ricker(t, fm=fm).astype(np.float32)

sources = np.array([256, 1]).reshape(1, 2)

rec_x = np.arange(0, nx, 1).reshape(-1, 1)
rec_z = np.ones_like(rec_x)*20
receivers = np.concatenate([rec_x, rec_z], axis=1)
receivers = receivers[None, ...].repeat(sources.shape[0], axis=0) # (nshots, nreceivers, 2)

def loss_fn(vp, vs, rho):
    syn = forward_elastic(jnp.array(wave), vp, vs, rho, jnp.ones((nz, nx)), sources, receivers, dt, dh, abcn, vpill=None, vsill=None, source_encoding=False)
    return jnp.sum(syn**2)

grads = jax.grad(loss_fn, argnums=(0,1,2))(jnp.array(true_vp), jnp.array(true_vs), jnp.array(rho))


fig, axes = plt.subplots(1, 3, figsize=(18, 4))
titles = ['Gradient of Vp', 'Gradient of Vs', 'Gradient of Density']
for ax, grad, title in zip(axes, grads, titles):

    # if 'Density' in title: grad = -grad
    vmin, vmax = np.percentile(grad, [0.5, 99.5])
    im = ax.imshow(grad, cmap='seismic', aspect='auto',
                extent=[0, nx * dh, nz * dh, 0], vmin=vmin, vmax=vmax)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Depth (m)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Gradient')
plt.tight_layout()
plt.savefig('gradient_hand_jax.png', dpi=300)
plt.show()