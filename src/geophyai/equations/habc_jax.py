import numpy as np
import jax.numpy as jnp
import jax

def bound_mask(nz, nx, w, batchsize=1, return_idx=False, free_surface=False):

    top = np.ones((w, nx), dtype=np.float32)

    indices = np.tril_indices(w, k=-1)

    top[indices] = 0.0
    top *= np.fliplr(top)
    bottom = np.flipud(top)

    if free_surface:
        top = None

    left = np.ones((nz, w), dtype=np.float32)
    indices = np.triu_indices(w, k=1)
    left[indices] = 0.0
    left *= np.flipud(left)
    right = np.fliplr(left)

    if free_surface:
        left[:w] = 1.
        right[:w] = 1.

    if not return_idx:
        return top, bottom, left, right
    else:
        tm = np.repeat(top[np.newaxis, :, :], batchsize, 0)==1 if not free_surface else None
        bm = np.repeat(bottom[np.newaxis, :, :], batchsize, 0)==1
        lm = np.repeat(left[np.newaxis, :, :], batchsize, 0)==1
        rm = np.repeat(right[np.newaxis, :, :], batchsize, 0)==1
        return tm, bm, lm, rm

def stack(*args):
    return jnp.stack(args, axis=0)

def cutb(d, w=50, n=0):
    return d[..., :w+n, :]

def flipud(d):
    """Expected shape of d is (batch, 1, nz, nx)-wavefield or (1, nz, nx)"""
    # Flip along the z-axis
    return jnp.flip(d, axis=[-2])

def rot90(d, k=1):
    return jnp.rot90(d, k=k, axes=(2, 3))

def habc(y, h1, h2, vel, coes, dt, h, w=50, maskidx=None):

    if vel.ndim == 2:
        vel = vel[jnp.newaxis, jnp.newaxis, ...]  # Add batch and channel dimensions

    otherargs = [dt, h]
    # # Calculate weighted one/two-wave-wavefield
    tbargs = [stack(cutb(array, w), cutb(flipud(array), w)) for array in [y, h1, h2, vel, coes[jnp.newaxis, jnp.newaxis, ...]]]+otherargs

    lrargs = [stack(cutb(rot90(array, -1), w), cutb(rot90(array, 1), w)) for array in [y, h1, h2, vel, coes[jnp.newaxis, jnp.newaxis, ...]]]+otherargs
    
    top, bottom = jnp.split(_habc(*tbargs, w=w), [1], axis=0)
    left, right = jnp.split(_habc(*lrargs, w=w), [1], axis=0)
    tmidx, bmidx, lmidx, rmidx = maskidx
    free_surface = tmidx is None

    # """Rotate"""
    y_top = top.squeeze(0) # (batchsize, 1, nz, nx)
    y_bottom = jnp.flip(bottom.squeeze(0), axis=[2])
    y_left = rot90(left.squeeze(0))
    y_right = rot90(right.squeeze(0), -1)#.squeeze()

    # Top
    if not free_surface:
        updated_slice = jnp.where(tmidx[:, None, ...], y_top, y[:, :, :w, :])
        y = y.at[:, :, :w, :].set(updated_slice)

    # Bottom
    updated_slice = jnp.where(bmidx[:, None, ...], y_bottom, y[:, :, -w:, :])
    y = y.at[:, :, -w:, :].set(updated_slice)

    # Left
    updated_slice = jnp.where(lmidx[:, None, ...], y_left, y[..., :w])
    y = y.at[..., :w].set(updated_slice)

    # Right boundary
    updated_slice = jnp.where(rmidx[:, None, ...], y_right, y[..., -w:])
    y = y.at[..., -w:].set(updated_slice)

    dix, diz = np.diag_indices(w)

    if not free_surface:
        # Top-Left corner
        y = y.at[..., dix, diz].set(0.5*y_top[..., dix, diz] + 0.5*y_left[..., dix, diz])

        # Top-Right corner
        y = y.at[..., dix, -w+diz].set(0.5*y_top[..., dix, -w+diz] + 0.5*y_right[..., dix, -w+diz])

    # Bottom-Left corner
    y = y.at[..., -w+dix, diz].set(0.5*y_bottom[..., dix, diz] + 0.5*y_left[..., -w+dix, diz])

    # Bottom-Right corner
    y = y.at[..., -w+dix, -w+diz].set(0.5*y_bottom[..., dix, -w+diz] + 0.5*y_right[..., -w+dix, -w+diz])

    return y

def _habc(u_next, u_now, u_pre, c, b, dt, dh, w=50):
    cut = w

    lam = 2*c*dt/dh
    mu = c**2*dt**2/(dh**2)

    # Top
    t1 = (2 - lam - mu) * u_now
    t2 = (lam + 2 * mu) * jnp.roll(u_now, shift=-1, axis=3)
    t3 = -mu * jnp.roll(u_now, shift=-2, axis=3)
    t4 = (lam - 1) * u_pre
    t5 = -lam * jnp.roll(u_pre, shift=-1, axis=3)
    u_one = t1 + t2 + t3 + t4 + t5

    hb_next = (u_one*b + (1-b) * u_next)

    return hb_next[:, :, :, :cut, :]

def habc1st(y, h1, vp, vs, coes, dt, h, w=50, maskidx=None):

    if vp.ndim == 2:
        vp = vp[jnp.newaxis, jnp.newaxis, ...]  # Add batch and channel dimensions
    if vs.ndim == 2:
        vs = vs[jnp.newaxis, jnp.newaxis, ...]  # Add batch and channel dimensions

    otherargs = [dt, h]
    # # Calculate weighted one/two-wave-wavefield
    tbargs = [stack(cutb(array, w), cutb(flipud(array), w)) for array in [y, h1, vp, vs, coes[jnp.newaxis, jnp.newaxis, ...]]]+otherargs

    lrargs = [stack(cutb(rot90(array, -1), w), cutb(rot90(array, 1), w)) for array in [y, h1, vp, vs, coes[jnp.newaxis, jnp.newaxis, ...]]]+otherargs
    
    top, bottom = jnp.split(_habc1st(*tbargs, w=w), [1], axis=0)
    left, right = jnp.split(_habc1st(*lrargs, w=w), [1], axis=0)
    tmidx, bmidx, lmidx, rmidx = maskidx
    free_surface = tmidx is None

    # """Rotate"""
    y_top = top.squeeze(0) # (batchsize, 1, nz, nx)
    y_bottom = jnp.flip(bottom.squeeze(0), axis=[2])
    y_left = rot90(left.squeeze(0))
    y_right = rot90(right.squeeze(0), -1)#.squeeze()

    # Top
    if not free_surface:
        updated_slice = jnp.where(tmidx[:, None, ...], y_top, y[:, :, :w, :])
        y = y.at[:, :, :w, :].set(updated_slice)

    # Bottom
    updated_slice = jnp.where(bmidx[:, None, ...], y_bottom, y[:, :, -w:, :])
    y = y.at[:, :, -w:, :].set(updated_slice)

    # Left
    updated_slice = jnp.where(lmidx[:, None, ...], y_left, y[..., :w])
    y = y.at[..., :w].set(updated_slice)

    # Right boundary
    updated_slice = jnp.where(rmidx[:, None, ...], y_right, y[..., -w:])
    y = y.at[..., -w:].set(updated_slice)

    dix, diz = np.diag_indices(w)

    if not free_surface:
        # Top-Left corner
        y = y.at[..., dix, diz].set(0.5*y_top[..., dix, diz] + 0.5*y_left[..., dix, diz])

    #     # Top-Right corner
    #     y = y.at[..., dix, -w+diz].set(0.5*y_top[..., dix, -w+diz] + 0.5*y_right[..., dix, -w+diz])

    # # Bottom-Left corner
    # y = y.at[..., -w+dix, diz].set(0.5*y_bottom[..., dix, diz] + 0.5*y_left[..., -w+dix, diz])

    # # Bottom-Right corner
    # y = y.at[..., -w+dix, -w+diz].set(0.5*y_bottom[..., dix, -w+diz] + 0.5*y_right[..., -w+dix, -w+diz])

    return y

def _habc1st(u_next, u_now, vp, vs, b, dt, dh, w=50):
    cut = w
    beta = (vp+vs)/2*vs

    # Top
    _u_now = jnp.roll(u_now, shift=-1, axis=3)
    u_one = _u_now + dt*vp/(beta*dh) * (_u_now - u_now)

    hb_next = (u_one*b + (1-b) * u_next)

    return hb_next[:, :, :, :cut, :]
