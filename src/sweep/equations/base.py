
import numpy as np
from .utils import to_backend
from .operator import PartialDerivative
from sweep.scalars import generate_convolution_kernel, generate_convolution_kernel3d
from .operator import laplace as lap2d_torch
from .operator_jax import laplace as lap2d_jax
from .operator_jax import laplace1d_sep as lap1d_jax
from .operator_jax import laplace3d_sep as lap3d_jax
from .operator import laplace1d_sep as lap1d_torch


def init_wavenumbers(shape, h):
    kz = np.fft.fftfreq(shape[0], d=h) * 2 * np.pi
    kx = np.fft.fftfreq(shape[1], d=h) * 2 * np.pi
    kzz, kxx = np.meshgrid(kz, kx, indexing='ij')
    k = np.sqrt(kxx**2 + kzz**2)
    return k, kx, kz

class FirstOrderEquation:
    """
    Base class for first-order equations.
    This class can be extended to implement specific first-order equations.
    """

    def __init__(self, spatial_order=4, device='cpu', backend='torch', **kwargs):
        """
        Initialize the first-order equation with an initial condition.

        :param initial_condition: The initial condition for the equation.
        """
        self.so = spatial_order
        self.backend = backend
        self.use_habc = False
        self.pd = PartialDerivative(spatial_order, device, backend)

    def init(self, shape, device='cpu', h=1.0):
        self.k, self.kx, self.kz = [to_backend(d, self.backend, device) for d in init_wavenumbers(shape, h)]

class SecondOrderEquation:
    """
    Base class for second-order equations.
    This class can be extended to implement specific second-order equations.
    """

    def __init__(self, spatial_order=4, device='cpu', backend='torch', **kwargs):
        """
        Initialize the second-order equation with an initial condition.

        :param initial_condition: The initial condition for the equation.
        """
        dim = kwargs.get('dim', 2)
        self.so = spatial_order
        self.backend = backend
        self.device = device
        self.use_habc = False
        self.habc_masks = None
        self.abcn = 50 # only useful for HABC

        kernel_func = {2: generate_convolution_kernel, 3: generate_convolution_kernel}[dim]
        self.kernel = to_backend(kernel_func(spatial_order), backend=backend, device=device)
        self.laplace = {'torch': lap2d_torch, 'jax': lap2d_jax, 'cuda': None}[backend]

        other_kernels = kwargs.get('other_kernels', False)
        self.kf = kernel_func
        if other_kernels:
            self.lkernel_x = to_backend(kernel_func(spatial_order, mode='x', no_center=False, grid='normal'), backend=backend, device=device)
            self.lkernel_z = to_backend(kernel_func(spatial_order, mode='z', no_center=False, grid='normal'), backend=backend, device=device)
            self.gkernel_x = to_backend(kernel_func(spatial_order, derivative_order=1, mode='x', no_center=True, grid='normal', sign=-1), backend=backend, device=device)
            self.gkernel_z = to_backend(kernel_func(spatial_order, derivative_order=1, mode='z', no_center=True, grid='normal', sign=-1), backend=backend, device=device)

    def init_laplace(self, ltype='2dmix', backend='jax'):
        """Overwrting the proporty <laplace>.

        Args:
            ltype (str, optional): Should be '2dmix' or '1dsep'. Defaults to '2dmix'.
        """
        assert ltype in ['2dmix', '1dsep', '3dsep'], "Unsupported laplace type"
        self.laplace = {'jax':   {'2dmix': lap2d_jax,   '1dsep': lap1d_jax, '3dsep': lap3d_jax},
                        'torch': {'2dmix': lap2d_torch, '1dsep': lap1d_torch},
                         }[backend][ltype]
        if ltype in ['1dsep', '3dsep']:
            self.kernel = to_backend(self.kf(self.so, mode='x')[0,0][self.so//2,:], backend=self.backend, device=self.device)


    def init(self, shape, device='cpu', h=1.0):
        self.k, self.kx, self.kz = [to_backend(d, self.backend, device) for d in init_wavenumbers(shape, h)]


    