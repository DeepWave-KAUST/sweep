
import numpy as np
from .utils import to_backend
from sweep.operators.general import PartialDerivative
from sweep.scalars import generate_convolution_kernel
from sweep.operators.factory import OperatorBase
from sweep.equations.pml import set_cpml_profiles_s, set_cpml_profiles_r, set_spml_profiles



def init_wavenumbers(shape, h):
    kz = np.fft.fftfreq(shape[0], d=h) * 2 * np.pi
    kx = np.fft.fftfreq(shape[1], d=h) * 2 * np.pi
    kzz, kxx = np.meshgrid(kz, kx, indexing='ij')
    k = np.sqrt(kxx**2 + kzz**2)
    return k, kx, kz

class WaveEquation:

    @classmethod
    def supports_torch_binding(cls):
        """Return True when the equation class exposes a compiled ``_C`` binding hook."""
        binding = getattr(cls, "_C", None)
        return callable(binding)

    def __init__(self, spatial_order=4, device='cpu', backend='jax', **kwargs):
        """
        Initialize the wave equation with an initial condition.

        :param initial_condition: The initial condition for the equation.
        """
        self.so = spatial_order
        self.backend = backend
        self.use_habc = False
        self.device = device
        self.pml_type = kwargs.get('pml_type', 'cpmls')

    def init_abc(self, type='cpml', **kwargs):
        pml_func = {'cpmls': set_cpml_profiles_s, 'cpmlr': set_cpml_profiles_r,'spml': set_spml_profiles}[type]
        self.b = pml_func(**kwargs)
        self.b = to_backend(self.b, self.backend, self.device)

class FirstOrderEquation(WaveEquation, ):
    """
    Base class for first-order equations.
    This class can be extended to implement specific first-order equations.
    """

    def __init__(self, spatial_order=4, device='cpu', backend='torch', ndim=2, **kwargs):
        """
        Initialize the first-order equation with an initial condition.

        :param initial_condition: The initial condition for the equation.
        """
        WaveEquation.__init__(self, spatial_order, device, backend, **kwargs)
        self.so = spatial_order
        self.backend = backend
        self.use_habc = False
        self.pd = PartialDerivative(spatial_order, device, backend, ndim=ndim)
        self.pd.to_backend(to_backend)

    def init(self, shape, device='cpu', h=1.0):
        self.k, self.kx, self.kz = [to_backend(d, self.backend, device) for d in init_wavenumbers(shape, h)]

class SecondOrderEquation(OperatorBase, WaveEquation):
    """
    Base class for second-order equations.
    This class can be extended to implement specific second-order equations.
    """

    def __init__(self, spatial_order=4, device='cpu', backend='torch', **kwargs):
        """
        Initialize the second-order equation with an initial condition.

        :param initial_condition: The initial condition for the equation.
        """
        OperatorBase.__init__(self, backend=backend)
        WaveEquation.__init__(self, spatial_order, device, backend, **kwargs)
        dim = kwargs.get('dim', 2)
        self.so = spatial_order
        self.backend = backend
        self.device = device
        self.use_habc = False
        self.habc_masks = None
        self.abcn = 50 # only useful for HABC
        self.laplace_kernels = None

        kernel_func = {2: generate_convolution_kernel, 3: generate_convolution_kernel}[dim]
        self.kernel = to_backend(kernel_func(spatial_order), backend=backend, device=device)

        other_kernels = kwargs.get('other_kernels', False)
        self.kf = kernel_func
        if other_kernels:
            self.lkernel_x = to_backend(kernel_func(spatial_order, mode='x', no_center=False, grid='normal'), backend=backend, device=device)
            self.lkernel_z = to_backend(kernel_func(spatial_order, mode='z', no_center=False, grid='normal'), backend=backend, device=device)
            self.gkernel_x = to_backend(kernel_func(spatial_order, derivative_order=1, mode='x', no_center=True, grid='normal', sign=-1), backend=backend, device=device)
            self.gkernel_z = to_backend(kernel_func(spatial_order, derivative_order=1, mode='z', no_center=True, grid='normal', sign=-1), backend=backend, device=device)

    def _prepare_separable_laplace_kernels(self):
        if self.backend != 'torch':
            return self.kernel
        if self.kernel.ndim == 1:
            return (
                self.kernel.view(1, 1, -1, 1).contiguous(),
                self.kernel.view(1, 1, 1, -1).contiguous(),
            )
        return self.kernel

    def init_laplace(self, ltype='2dmix', backend='jax'):
        """Overwrting the proporty <laplace>.

        Args:
            ltype (str, optional): Should be '2dmix' or '1dsep'. Defaults to '2dmix'.
        """
        if ltype in ['1dsep', '3dsep']:
            self.kernel = to_backend(self.kf(self.so, mode='x')[0,0][self.so//2,:], backend=self.backend, device=self.device)
            self.laplace_kernels = self._prepare_separable_laplace_kernels()
        else:
            self.laplace_kernels = self.kernel

    def init(self, shape, device='cpu', h=1.0):
        self.k, self.kx, self.kz = [to_backend(d, self.backend, device) for d in init_wavenumbers(shape, h)]


    
