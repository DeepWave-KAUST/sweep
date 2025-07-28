import torch
import numpy as np
from typing import Tuple, Optional

def cutb(d, w:int=50, n:int=0):
    if d.ndim == 4:
        return d[:, :, :w+n, :]
    if d.ndim == 3:
        return d[:, :w+n, :]

def _habc(u_next, u_now, u_pre, c, b, dt, dh, w:int=50):

    cut = w

    lam = 2*c*dt/dh
    mu = c**2*dt**2/(dh**2)

    # Top
    t1 = (2 - lam - mu) * u_now
    t2 = (lam + 2 * mu) * torch.roll(u_now, shifts=-1, dims=3)
    t3 = -mu * torch.roll(u_now, shifts=-2, dims=3)
    t4 = (lam - 1) * u_pre
    t5 = -lam * torch.roll(u_pre, shifts=-1, dims=3)
    u_one = t1 + t2 + t3 + t4 + t5

    hb_next = (u_one*b + (1-b) * u_next)

    return hb_next[:, :, :, :cut, :]
    
def rot90(d, k=1):
    """Expected shape of d is (batch, 1, nz, nx)"""
    return torch.rot90(d, k=k, dims=(2, 3))
    
def flipud(d):
    """Expected shape of d is (batch, 1, nz, nx)-wavefield or (1, nz, nx)"""
    return torch.flip(d, dims=[2])
    
def identity(d):
    return d

def stack(d1:torch.Tensor, d2:torch.Tensor):
    return torch.stack([d1, d2], dim=0)

def habc(y, h1, h2, vel, coes, dt, h, w:int=50, maskidx=None):
    otherargs = [dt, h]
    # Calculate weighted one/two-wave-wavefield
    tbargs = [stack(cutb(array, w), cutb(flipud(array), w)) for array in [y, h1, h2, vel[torch.newaxis, torch.newaxis, ...], coes[torch.newaxis, torch.newaxis, ...]]]+otherargs
    lrargs = [stack(cutb(rot90(array, -1), w), cutb(rot90(array, 1), w)) for array in [y, h1, h2, vel[torch.newaxis, torch.newaxis, ...], coes[torch.newaxis, torch.newaxis, ...]]]+otherargs
    top, bottom = torch.split(_habc(*tbargs, w=w), 1, dim=0)
    left, right = torch.split(_habc(*lrargs, w=w), 1, dim=0)
    tmidx, bmidx, lmidx, rmidx = maskidx

    free_surface = tmidx is None

    """Rotate"""
    y_top = top.squeeze(0)
    y_bottom = torch.flip(bottom.squeeze(0), dims=[2])
    y_left = rot90(left.squeeze(0))
    y_right = rot90(right.squeeze(0), -1)

    # Top
    if not free_surface:
        updated_slice = torch.where(tmidx[:, None, ...], y_top, y[:, :, :w, :])
        y[:,:,:w,:] = updated_slice

    # Bottom
    updated_slice = torch.where(bmidx[:, None, ...], y_bottom, y[:, :, -w:, :])
    y[:,:,-w:,:] = updated_slice

    # Left
    updated_slice = torch.where(lmidx[:, None, ...], y_left, y[..., :w])
    y[..., :w] = updated_slice

    # Right boundary
    updated_slice = torch.where(rmidx[:, None, ...], y_right, y[..., -w:])
    y[..., -w:] = updated_slice

    dix, diz = np.diag_indices(w)

    if not free_surface:
        # Top-Left corner
        y[..., dix, diz] = 0.5*y_top[..., dix, diz]\
                    + 0.5*y_left[..., dix, diz]
        
        # Top-Right corner
        y[..., dix, -w+diz] = 0.5*y_top[..., dix, -w+diz]\
                        + 0.5*y_right[..., dix, -w+diz]
    
    # Bottom-Left corner
    y[..., -w+dix, diz] = 0.5*y_bottom[..., dix, diz]\
                      + 0.5*y_left[..., -w+dix, diz]
    
    # Bottom-Right corner
    y[..., -w+dix, -w+diz] = 0.5*y_bottom[..., dix, -w+diz]\
                         + 0.5*y_right[..., -w+dix, -w+diz]

    return y