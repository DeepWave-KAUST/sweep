
import numpy as np
from .utils import to_backend
from sweep.operators.general import PartialDerivative
from sweep.scalars import generate_convolution_kernel
from sweep.operators.factory import OperatorBase
from sweep.equations.pml import set_cpml_profiles_s, set_cpml_profiles_r, set_spml_profiles
from .fields import (
    available_role_specs,
    ensure_field_specs,
    ensure_model_specs,
    format_field_specs,
    format_model_specs,
)
from .cuda_layout import CUDALayoutSpec


class hybridmethod:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        def wrapper(*args, **kwargs):
            target = obj if obj is not None else cls
            return self.func(target, *args, **kwargs)

        return wrapper



def init_wavenumbers(shape, h):
    kz = np.fft.fftfreq(shape[0], d=h) * 2 * np.pi
    kx = np.fft.fftfreq(shape[1], d=h) * 2 * np.pi
    kzz, kxx = np.meshgrid(kz, kx, indexing='ij')
    k = np.sqrt(kxx**2 + kzz**2)
    return k, kx, kz

class WaveEquation:

    # Default PML formulation when the propagator is not given one explicitly.
    # Subclasses with stricter requirements (e.g. ElasticTTISG → 'cpmls') override this.
    default_pml_type = "cpmlr"

    @classmethod
    def supports_torch_binding(cls):
        """Return True when the equation class exposes a compiled ``_C`` binding hook."""
        binding = getattr(cls, "_C", None)
        return callable(binding)

    @hybridmethod
    def defaults(target):
        """Resolved defaults this equation will use with no overrides.

        Useful for discovering — without reading source — what ``pml_type``,
        source/receiver fields, ``spatial_order``, etc. you get from
        ``EquationCls()``. Works as a classmethod or on an instance::

            >>> from sweep.equations import Acoustic
            >>> Acoustic.defaults()
            {'class_name': 'Acoustic',
             'spatial_order': 4,
             'backend': 'torch',
             'device': 'cpu',
             'default_pml_type': 'cpmlr',
             'default_source_fields': ['h1'],
             'default_receiver_fields': ['h1'],
             'wavefields': ['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz'],
             'models': ['vp']}

        Class-level introspection instantiates with default args; if that fails
        (e.g. equation requires an optional backend), only class-level
        attributes are returned along with a ``note`` explaining why.
        """
        if isinstance(target, type):
            try:
                instance = target()
            except (ImportError, TypeError):
                try:
                    instance = target(backend='torch')
                except Exception as exc:
                    return {
                        "class_name": target.__name__,
                        "default_pml_type": getattr(target, "default_pml_type", None),
                        "note": f"cannot instantiate without arguments: {exc}",
                    }
        else:
            instance = target

        return {
            "class_name": type(instance).__name__,
            "spatial_order": instance.so,
            "backend": instance.backend,
            "device": instance.device,
            "default_pml_type": instance.default_pml_type,
            "default_source_fields": list(instance.default_source_fields),
            "default_receiver_fields": list(instance.default_receiver_fields),
            "wavefields": list(instance.wavefields),
            "models": list(instance.models),
        }

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

    @property
    def field_specs(self):
        # FieldSpec order is semantically significant. The propagators map
        # source/receiver names to positional wavefield indices, and equation
        # step functions are expected to return tensors in the same order.
        # Reordering field specs therefore changes forward/backward behavior,
        # source injection, receiver sampling, checkpointing, and CUDA bindings.
        return ensure_field_specs(self.wavefields, [])

    @property
    def model_specs(self):
        return ensure_model_specs(self.models, [])

    @classmethod
    def _field_specs_for_query(cls):
        field_specs = getattr(cls, "FIELD_SPECS", None)
        if field_specs is not None:
            return list(field_specs)

        try:
            equation = cls()
        except Exception as exc:
            raise TypeError(
                f"{cls.__name__} does not define class-level FIELD_SPECS, so querying fields "
                "from the class requires instantiation. Instantiate the equation first or "
                "define FIELD_SPECS on the class."
            ) from exc

        return list(equation.field_specs)

    @classmethod
    def _model_specs_for_query(cls):
        model_specs = getattr(cls, "MODEL_SPECS", None)
        if model_specs is not None:
            return ensure_model_specs(cls._model_names_for_query(), list(model_specs))

        try:
            equation = cls()
        except Exception as exc:
            raise TypeError(
                f"{cls.__name__} does not define class-level MODEL_SPECS, so querying models "
                "from the class requires instantiation. Instantiate the equation first or "
                "define MODEL_SPECS on the class."
            ) from exc

        return list(equation.model_specs)

    @classmethod
    def _model_names_for_query(cls):
        model_specs = getattr(cls, "MODEL_SPECS", None)
        if model_specs is not None:
            return [spec.name for spec in model_specs]
        try:
            equation = cls()
        except Exception as exc:
            raise TypeError(
                f"{cls.__name__} does not define class-level MODEL_SPECS, so querying model names "
                "from the class requires instantiation. Instantiate the equation first or "
                "define MODEL_SPECS on the class."
            ) from exc
        return list(equation.models)

    @classmethod
    def _field_specs_from_target(cls, target):
        if isinstance(target, WaveEquation):
            return list(target.field_specs)
        if isinstance(target, type) and issubclass(target, WaveEquation):
            return target._field_specs_for_query()
        raise TypeError("target must be a WaveEquation instance or subclass")

    @classmethod
    def _model_specs_from_target(cls, target):
        if isinstance(target, WaveEquation):
            return list(target.model_specs)
        if isinstance(target, type) and issubclass(target, WaveEquation):
            return target._model_specs_for_query()
        raise TypeError("target must be a WaveEquation instance or subclass")

    @property
    def default_source_fields(self):
        source_specs = available_role_specs(self.field_specs, "source")
        if source_specs:
            return [source_specs[0].name]
        return [self.wavefields[0]]

    @property
    def default_receiver_fields(self):
        receiver_specs = available_role_specs(self.field_specs, "receiver")
        if receiver_specs:
            return [receiver_specs[0].name]
        return [self.wavefields[0]]

    @hybridmethod
    def available_source_fields(target):
        specs = WaveEquation._field_specs_from_target(target)
        return [
            spec for spec in available_role_specs(specs, "source")
            if not spec.internal and not spec.boundary_related
        ]

    @hybridmethod
    def available_receiver_fields(target):
        specs = WaveEquation._field_specs_from_target(target)
        return [
            spec for spec in available_role_specs(specs, "receiver")
            if not spec.internal and not spec.boundary_related
        ]

    @hybridmethod
    def available_fields(target, role=None, include_internal=False, include_boundary=False):
        specs = WaveEquation._field_specs_from_target(target)
        if role is None:
            selected = list(specs)
        elif role in {"source", "receiver"}:
            selected = available_role_specs(specs, role)
        else:
            raise ValueError("role must be one of None, 'source', or 'receiver'.")

        if not include_internal:
            selected = [spec for spec in selected if not spec.internal]
        if not include_boundary:
            selected = [spec for spec in selected if not spec.boundary_related]
        return selected

    @hybridmethod
    def describe_field(target, name):
        specs = WaveEquation._field_specs_from_target(target)
        for spec in specs:
            if spec.name == name or name in spec.aliases:
                alias_text = f" Aliases: {', '.join(spec.aliases)}." if spec.aliases else ""
                return f"{spec.name}: {spec.description}{alias_text}".strip()
        available = format_field_specs(specs)
        raise KeyError(f"Unknown field '{name}'. Available fields:\n{available}")

    @hybridmethod
    def available_models(target):
        return WaveEquation._model_specs_from_target(target)

    @hybridmethod
    def describe_model(target, name):
        specs = WaveEquation._model_specs_from_target(target)
        for spec in specs:
            if spec.name == name or name in spec.aliases:
                alias_text = f" Aliases: {', '.join(spec.aliases)}." if spec.aliases else ""
                unit_text = f" Units: {spec.unit}." if spec.unit else ""
                req_text = "" if spec.required else " Optional."
                return f"{spec.name}: {spec.description}{alias_text}{unit_text}{req_text}".strip()
        available = format_model_specs(specs)
        raise KeyError(f"Unknown model '{name}'. Available models:\n{available}")

    @property
    def cuda_layout(self):
        return None

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
        self.ndim = ndim
        self.use_habc = False
        self.pd = PartialDerivative(spatial_order, device, backend, ndim=ndim)
        self.pd.to_backend(to_backend)

    def init(self, shape, device='cpu', h=1.0):
        self.k, self.kx, self.kz = [to_backend(d, self.backend, device) for d in init_wavenumbers(shape, h)]

    def _axis_spacing(self, h, axis):
        axis = axis if axis >= 0 else self.ndim + axis
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"Axis {axis} is out of bounds for ndim={self.ndim}.")

        if np.isscalar(h):
            return h

        if hasattr(h, "ndim") and getattr(h, "ndim", 0) == 0:
            return h

        return h[axis]

    def _spacings_2d(self, h):
        return self._axis_spacing(h, 0), self._axis_spacing(h, 1)

    def _spacings_3d(self, h):
        return (
            self._axis_spacing(h, 0),
            self._axis_spacing(h, 1),
            self._axis_spacing(h, 2),
        )

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
        self.ndim = dim
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
            if self.ndim == 3:
                return (
                    self.kernel.view(1, 1, -1, 1, 1).contiguous(),
                    self.kernel.view(1, 1, 1, -1, 1).contiguous(),
                    self.kernel.view(1, 1, 1, 1, -1).contiguous(),
                )
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

    def _axis_spacing(self, h, axis):
        axis = axis if axis >= 0 else self.ndim + axis
        if axis < 0 or axis >= self.ndim:
            raise ValueError(f"Axis {axis} is out of bounds for ndim={self.ndim}.")

        if np.isscalar(h):
            return h

        if hasattr(h, "ndim") and getattr(h, "ndim", 0) == 0:
            return h

        return h[axis]

    def _spacings_2d(self, h):
        return self._axis_spacing(h, 0), self._axis_spacing(h, 1)

    def _spacings_3d(self, h):
        return (
            self._axis_spacing(h, 0),
            self._axis_spacing(h, 1),
            self._axis_spacing(h, 2),
        )


    
