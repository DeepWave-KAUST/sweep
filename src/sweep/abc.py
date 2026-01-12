import torch
import numpy as np

# Coefficients of PML
def abc_coefficients_2d(domain_shape, N=50, B=100., free_surface=False):
    Nx, Ny = domain_shape

    if N == 0:
        return np.zeros(domain_shape, dtype=np.float32)
    order = 2
    cp = 1500.

    R = 10**(-((np.log10(N)-1)/np.log10(2))-3)
    d0 = -(order+1)*cp/(2*N)*np.log(R) # Origin
    # R = 1e-6
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

def abc_coefficients_2d_wwh(domain_shape, N=100, B=300., free_surface=False):
    Nx, Ny = domain_shape

    if N == 0:
        return np.zeros(domain_shape, dtype=np.float32)
    
    d_vals = B * (1-np.cos(np.pi/2* ( np.arange(1,N+1)/N ) ) )[::-1]

    d_x = np.zeros((Ny, Nx))
    d_y = np.zeros((Ny, Nx))

    if N > 0:
        d_x[0:N, :] = np.tile(d_vals, (Nx, 1)).T
        d_x[(Ny - N):Ny, :] = np.tile(d_vals[::-1], (Nx, 1)).T
        if not free_surface:
            d_y[:, 0:N] = np.tile(d_vals, (Ny, 1))
        d_y[:, (Nx - N):Nx] = np.tile(d_vals[::-1], (Ny, 1))

    _d = np.sqrt(d_x ** 2 + d_y ** 2).T

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

def abc_coefficients_3d(domain_shape, N=50, B=100., free_surface=False):
    nz, ny, nx = domain_shape

    R = 1e-6
    order = 2
    cp = 1000.
    d0 = (1.5 * cp / N) * np.log10(R**-1)
    d_vals = d0 * np.linspace(0.0, 1.0, N) ** order
    d_vals = d_vals[::-1]  # Flip the array

    b_x = np.zeros((nz, ny, nx))
    b_y = np.zeros((nz, ny, nx))
    b_z = np.zeros((nz, ny, nx))

    # z-direction (top and bottom)
    b_z[0:N, :, :] = d_vals[:, np.newaxis, np.newaxis]
    b_z[(nz - N):nz, :, :] = d_vals[::-1][:, np.newaxis, np.newaxis]

    # y-direction (front and back)
    b_y[:, 0:N, :] = d_vals[np.newaxis, :, np.newaxis]
    b_y[:, (ny - N):ny, :] = d_vals[::-1][np.newaxis, :, np.newaxis]

    # x-direction (left and right)
    b_x[:, :, 0:N] = d_vals[np.newaxis, np.newaxis, :]
    b_x[:, :, (nx - N):nx] = d_vals[::-1][np.newaxis, np.newaxis, :]

    if free_surface:
        b_z[0:N, :, :] = 0.0  # no damping at top surface if free surface

    return np.sqrt(b_x ** 2 + b_y ** 2 + b_z ** 2)