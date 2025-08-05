import numpy as np
from scipy.interpolate import RegularGridInterpolator

def resize(d, ori_shape, new_shape):

    nz, nx = ori_shape
    newnz, newnx = new_shape

    z = np.arange(nz)
    x = np.arange(nx)

    new_z = np.linspace(0, nz - 1, newnz)
    new_x = np.linspace(0, nx - 1, newnx)

    interpolator = RegularGridInterpolator((z, x), d)

    new_z_mesh, new_x_mesh = np.meshgrid(new_z, new_x, indexing='ij')
    new_points = np.array([new_z_mesh.ravel(), new_x_mesh.ravel()]).T
    new_d = interpolator(new_points).reshape(newnz, newnx)
    
    return new_d