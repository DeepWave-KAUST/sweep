import torch
import numpy as np

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
        tm = np.repeat(top[np.newaxis, :, :], 1, 0)==1 if not free_surface else None
        bm = np.repeat(bottom[np.newaxis, :, :], 1, 0)==1
        lm = np.repeat(left[np.newaxis, :, :], 1, 0)==1
        rm = np.repeat(right[np.newaxis, :, :], 1, 0)==1
        return tm, bm, lm, rm

def habc_coefficients_2d(domain_shape, 
                         N=50, 
                         free_surface=False):

    nz, nx = domain_shape

    d = np.zeros(domain_shape, dtype=np.float32)

    d_vals = np.linspace(0.0, N, N)/N
    d_vals = np.flip(d_vals, [0])

    tm, bm, lm, rm = bound_mask(*domain_shape, N, free_surface=free_surface)

    if N > 0:
        # Top
        if not free_surface:
            idx = tm==1
            d[:N,:][idx] = (d_vals[:, np.newaxis].repeat(nx, 1).transpose(0, 1)*tm)[idx]

        # Bottom
        idx = bm==1 # Mask for equation left
        d[-N:,:][idx] = (np.flip(d_vals, [0])[:, np.newaxis].repeat(nx, 1).transpose(0, 1)*bm)[idx]

        # Left
        idx = lm==1

        if not free_surface:
            d[:, :N][idx] = (d_vals[:, np.newaxis].repeat(nz, 1).T*lm)[idx]
        if free_surface:
            lm[:N] = 1.
            d[:, :N][lm==1] = (d_vals[:, np.newaxis].repeat(nz, 1).T*lm)[lm==1]

        # Right boundary
        idx = rm==1
        if not free_surface:
            d[:, -N:][idx] = (np.flip(d_vals[:, np.newaxis], [0]).repeat(nz, 1).T*rm)[idx]
        if free_surface:
            rm[:N] = 1.
            d[:, -N:][rm==1] = (np.flip(d_vals[:, np.newaxis], [0]).repeat(nz, 1).T*rm)[rm==1]
    return d


# Coefficients of PML
def abc_coefficients_2d(domain_shape, N=50, B=100., free_surface=False):
    Nx, Ny = domain_shape

    if N == 0:
        return np.zeros(domain_shape, dtype=np.float32)

    R = 10**(-((np.log10(N)-1)/np.log10(2))-3)
    # d0 = -(order+1)*cp/(2*abs_N)*np.log(R) # Origin
    R = 1e-6
    order = 2
    cp = 1000.
    d0 = (1.5 * cp / N) * np.log10(R**-1)
    d_vals = d0 * np.linspace(0.0, 1.0, N) ** order
    d_vals = d_vals[::-1]  # Flip the array

    d_x = np.zeros((Ny, Nx))
    d_y = np.zeros((Ny, Nx))

    if N > 0:
        d_x[0:N, :] = np.tile(d_vals, (Nx, 1)).T
        d_x[(Ny - N):Ny, :] = np.tile(d_vals[::-1], (Nx, 1)).T
        if not free_surface:
            d_y[:, 0:N] = np.tile(d_vals, (Ny, 1))
        d_y[:, (Nx - N):Nx] = np.tile(d_vals[::-1], (Ny, 1))

    _d = np.sqrt(d_x ** 2 + d_y ** 2).T
    _d = _corners(domain_shape, N, _d, d_x.T, d_y.T, free_surface)

    return _d.astype(np.float32)

def _corners(domain_shape, abs_N, d, dx, dy, free_surface=False):
    Nx, Ny = domain_shape
    for j in range(Ny):
        for i in range(Nx):
            # Left-Top
            if not free_surface:
                if i < abs_N and j< abs_N:
                    if i < j: d[i,j] = dy[i,j]
                    else: d[i,j] = dx[i,j]
            # Left-Bottom
            if i > (Nx-abs_N-1) and j < abs_N:
                if i + j < Nx: d[i,j] = dx[i,j]
                else: d[i,j] = dy[i,j]
            # Right-Bottom
            if i > (Nx-abs_N-1) and j > (Ny-abs_N-1):
                if i - j > Nx-Ny: d[i,j] = dy[i,j]
                else: d[i,j] = dx[i,j]
            # Right-Top
            if not free_surface:
                if i < abs_N and j> (Ny-abs_N-1):
                    if i + j < Ny: d[i,j] = dy[i,j]
                    else: d[i,j] = dx[i,j]

    return d
