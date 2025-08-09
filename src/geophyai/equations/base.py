
import numpy as np
from .utils import to_backend
from .operator import PartialDerivative
from geophyai.scalars import generate_convolution_kernel, generate_convolution_kernel3d
from .operator import laplace as laplace_torch
from .operator_jax import laplace as laplace_jax


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
        self.backend = backend
        self.use_habc = False
        self.habc_masks = None

        kernel_func = {2: generate_convolution_kernel, 3: generate_convolution_kernel3d}[dim]
        self.kernel = to_backend(kernel_func(spatial_order), backend=backend, device=device)
        self.laplace = {'torch': laplace_torch, 'jax': laplace_jax}[backend]

        other_kernels = kwargs.get('other_kernels', False)

        if other_kernels:
            self.lkernel_x = to_backend(kernel_func(spatial_order, mode='x', no_center=False, grid='normal'), backend=backend, device=device)
            self.lkernel_z = to_backend(kernel_func(spatial_order, mode='z', no_center=False, grid='normal'), backend=backend, device=device)
            self.gkernel_x = to_backend(kernel_func(spatial_order, derivative_order=1, mode='x', no_center=True, grid='normal', sign=-1), backend=backend, device=device)
            self.gkernel_z = to_backend(kernel_func(spatial_order, derivative_order=1, mode='z', no_center=True, grid='normal', sign=-1), backend=backend, device=device)

    def init(self, shape, device='cpu', h=1.0):
        self.k, self.kx, self.kz = [to_backend(d, self.backend, device) for d in init_wavenumbers(shape, h)]


    