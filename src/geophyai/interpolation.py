import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator

def interp1d(d, ori_dt, new_dt):
    """
    Interpolate a 1D array to a new time step.

    Args:
        d (np.ndarray): The input 1D array to be interpolated.
        ori_dt (float): The original time step.
        new_dt (float): The new time step.

    Returns:
        np.ndarray: The interpolated 1D array.
    """
    ori_t = np.arange(d.size) * ori_dt
    new_t = np.arange(int(np.ceil(ori_t[-1] / new_dt)) + 1) * new_dt

    interpolator = RegularGridInterpolator((ori_t,), d, bounds_error=False, fill_value=0.0)
    return interpolator(new_t).astype(np.float32)

def resize(d, new_shape):
    """
    Resize a 2D array to a new shape using interpolation.

    Args:
        d (np.ndarray): The input 2D array to be resized.
        new_shape (tuple): The desired shape of the array (newnz, newnx).
    Returns:
        np.ndarray: The resized 2D array.
    """

    nz, nx = d.shape
    newnz, newnx = new_shape

    z = np.arange(nz)
    x = np.arange(nx)

    new_z = np.linspace(0, nz - 1, newnz)
    new_x = np.linspace(0, nx - 1, newnx)

    interpolator = RegularGridInterpolator((z, x), d, method='linear')

    new_z_mesh, new_x_mesh = np.meshgrid(new_z, new_x, indexing='ij')
    new_points = np.array([new_z_mesh.ravel(), new_x_mesh.ravel()]).T
    new_d = interpolator(new_points).reshape(newnz, newnx)
    
    return new_d.astype(np.float32)

def resample_2d_batch_general(data, 
                              dt=None, dx=None, 
                              t=None, x=None, 
                              dt_new=None, dx_new=None, 
                              t_new=None, x_new=None):
    """
    General batch 2D resampling function.  
    Depending on parameters provided, automatically:
     - Generate original and/or new coordinate arrays from dt/dx if needed
     - Or use given coordinate arrays t, x, t_new, x_new

    Parameters:
    - data: 4D numpy array, shape (batch_size, nt, nx, nc)
    - dt, dx: floats, original uniform sampling intervals (optional)
    - t, x: 1D arrays, original sampling coordinates (optional)
    - dt_new, dx_new: floats, new uniform sampling intervals (optional)
    - t_new, x_new: 1D arrays, new sampling coordinates (optional)

    Returns:
    - data_new: 4D numpy array, resampled data (batch_size, len(t_new), len(x_new), nc)
    """

    batch_size, nt, nx, nc = data.shape

    # --- Generate original coords if not provided ---
    if t is None:
        if dt is None:
            raise ValueError("Must provide either t or dt")
        t = np.arange(nt) * dt

    if x is None:
        if dx is None:
            raise ValueError("Must provide either x or dx")
        x = np.arange(nx) * dx

    # --- Generate new coords if not provided ---
    if t_new is None:
        if dt_new is None:
            raise ValueError("Must provide either t_new or dt_new")
        t_new_len = int(np.ceil(t[-1] / dt_new)) + 1
        t_new = np.arange(t_new_len) * dt_new

    if x_new is None:
        if dx_new is None:
            raise ValueError("Must provide either x_new or dx_new")
        x_new_len = int(np.ceil(x[-1] / dx_new)) + 1
        x_new = np.arange(x_new_len) * dx_new

    # --- Prepare interpolation grid ---
    T_new, X_new = np.meshgrid(t_new, x_new, indexing='ij')
    points_new = np.vstack((T_new.ravel(), X_new.ravel())).T  # shape (nt_new * nx_new, 2)

    data_new = np.zeros((batch_size, len(t_new), len(x_new), nc), dtype=data.dtype)

    # --- Interpolate each batch and channel ---
    for i in range(batch_size):
        for c in range(nc):
            data_2d = data[i, :, :, c]
            interpolator = RegularGridInterpolator((t, x), data_2d, bounds_error=False, fill_value=0.0)
            sampled_values = interpolator(points_new)
            data_new[i, :, :, c] = sampled_values.reshape(len(t_new), len(x_new))

    return data_new

def resample_single_2d(data_2d, old_t, old_x, new_t, new_x):
    """
    Resample a single 2D array with shape (nt, nx).
    """
    nt, nx = data_2d.shape
    nt_new = new_t.size
    nx_new = new_x.size

    t_idx = (new_t - old_t[0]) / (old_t[1] - old_t[0])
    x_idx = (new_x - old_x[0]) / (old_x[1] - old_x[0])

    grid_t, grid_x = jnp.meshgrid(t_idx, x_idx, indexing='ij')
    coords = jnp.array([grid_t, grid_x]).reshape(2, -1)

    sampled = map_coordinates(data_2d, coords, order=1, mode='constant', cval=0.0)
    return sampled.reshape(nt_new, nx_new)

def resample_4d_batch_jax(data, dt, dx, dt_new, dx_new):
    """
    Resample 4D data (nbatch, nt, nx, nc) along time (nt) and space (nx) dims.

    Args:
        data: jnp.array with shape (nbatch, nt, nx, nc)
        dt, dx: original sampling intervals
        dt_new, dx_new: new sampling intervals

    Returns:
        data_new: jnp.array with shape (nbatch, nt_new, nx_new, nc)
        t_new: 1D array of new time coordinates
        x_new: 1D array of new space coordinates
    """
    nbatch, nt, nx, nc = data.shape
    old_t = jnp.arange(nt) * dt
    old_x = jnp.arange(nx) * dx

    nt_new = int(jnp.ceil(old_t[-1] / dt_new)) + 1
    nx_new = int(jnp.ceil(old_x[-1] / dx_new)) + 1

    new_t = jnp.arange(nt_new) * dt_new
    new_x = jnp.arange(nx_new) * dx_new

    # Vectorize interpolation over channel dimension
    channel_vmap = vmap(resample_single_2d, in_axes=(0, None, None, None, None))

    # Vectorize over batch dimension
    batch_vmap = vmap(channel_vmap, in_axes=(0, None, None, None, None))

    data_new = batch_vmap(data, old_t, old_x, new_t, new_x)
    # data_new shape: (nbatch, nc, nt_new, nx_new), we need to permute to (nbatch, nt_new, nx_new, nc)
    data_new = jnp.transpose(data_new, (0, 2, 3, 1))

    return data_new, new_t, new_x