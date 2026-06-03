
import numpy as np
from .utils import to_backend
from sweep.operators.general import StaggeredDerivative
from sweep.scalars import generate_convolution_kernel
from sweep.operators.factory import LaplaceGradientOps
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

    # Whether this equation has a parameter-modified (APM, Cao & Chen 2018)
    # path for irregular free-surface topography.  When True, the propagator's
    # ``topography=`` accepts a 2-D air_mask + ``free_surface=False`` and
    # routes through the equation's APM dispatch.  Defaults to False;
    # subclasses with an APM implementation (currently :class:`Elastic`)
    # override.
    supports_apm = False

    # Class-level spec tables. Subclasses that declare these tables drive
    # the ``wavefields`` / ``models`` / ``field_specs`` / ``model_specs``
    # properties below automatically; manual property overrides remain
    # supported for backwards compatibility (the override wins).
    FIELD_SPECS: tuple = ()
    MODEL_SPECS: tuple = ()

    @staticmethod
    def _render_specs_admonition(model_specs, field_specs,
                                 default_source, default_receiver,
                                 default_pml):
        """Render ``!!! info`` admonition blocks from spec tables.

        Returns a markdown string that mirrors the hand-written
        ``Models`` / ``Wavefields`` / ``Defaults`` admonitions used in
        the equation docstrings. The output is unindented so it can be
        appended to any ``cls.__doc__`` and rendered after
        ``inspect.getdoc`` dedents the source.
        """
        parts = []
        if model_specs:
            parts.append('!!! info "Models (constructor input order)"\n')
            for s in model_specs:
                unit = f" ({s.unit})" if s.unit else ""
                parts.append(f"    - ``{s.name}``{unit}: {s.description}")
            parts.append("")
        if field_specs:
            parts.append('!!! info "Wavefields"\n')
            for s in field_specs:
                alias = ""
                if s.aliases:
                    alias = " (aliases: " + ", ".join(f"``{a}``" for a in s.aliases) + ")"
                in_src = s.name in (default_source or [])
                in_rcv = s.name in (default_receiver or [])
                desc = s.description.rstrip(".")
                if s.internal:
                    suffix = " (internal)."
                elif in_src and in_rcv:
                    suffix = "; default source and receiver."
                elif in_src:
                    suffix = "; default source."
                elif in_rcv:
                    suffix = "; default receiver."
                else:
                    suffix = "."
                parts.append(f"    - ``{s.name}``{alias}: {desc}{suffix}")
            parts.append("")
        defaults = []
        if default_source:
            defaults.append(f"    - ``source_type``: ``{list(default_source)!r}``")
        if default_receiver:
            defaults.append(f"    - ``receiver_type``: ``{list(default_receiver)!r}``")
        if default_pml:
            defaults.append(f"    - ``pml_type``: ``{default_pml!r}``")
        if defaults:
            parts.append('!!! info "Defaults"\n')
            parts.extend(defaults)
            parts.append("")
        return "\n".join(parts)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Skip facades / placeholders that declare neither table — they have
        # no specs to render, and instantiating them (e.g. AcousticAniso's
        # __new__ dispatch) would cause surprises.
        if not (cls.FIELD_SPECS or cls.MODEL_SPECS):
            return
        # Dedent the source docstring with cleandoc so the appended
        # admonition section (which is unindented) lines up with the
        # original prose at the same indentation level.
        import inspect as _inspect
        existing_doc = _inspect.cleandoc(cls.__doc__ or "")
        # If a subclass author already hand-wrote the ``!!! info "Models"``
        # admonition into the docstring, leave their text in place — the
        # automatic injection is opt-in by deleting the hand-written block.
        if '!!! info "Models' in existing_doc:
            return
        # Resolve defaults via instantiation when possible; otherwise derive
        # from the SPECS tables.
        try:
            inst = cls()
        except Exception:
            inst = None
        if inst is not None:
            try:
                default_source = list(inst.default_source_fields)
                default_receiver = list(inst.default_receiver_fields)
                default_pml = inst.default_pml_type
            except Exception:
                inst = None
        if inst is None:
            default_source = [s.name for s in cls.FIELD_SPECS if s.supports_source][:1]
            default_receiver = [s.name for s in cls.FIELD_SPECS if s.supports_receiver][:1]
            default_pml = getattr(cls, "default_pml_type", None)
        section = WaveEquation._render_specs_admonition(
            cls.MODEL_SPECS, cls.FIELD_SPECS,
            default_source, default_receiver, default_pml,
        )
        if section:
            cls.__doc__ = existing_doc.rstrip() + "\n\n" + section

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

    # Each of the four properties below resolves in the same order:
    #   1. an instance-level write (``self.wavefields = ...`` etc.) wins,
    #      so legacy equations like ``Acoustic1st`` that set names from
    #      ``__init__`` based on the chosen PML type keep working;
    #   2. the class-level ``FIELD_SPECS`` / ``MODEL_SPECS`` table is used
    #      when no instance write happened — this is the recommended path
    #      for new equations;
    #   3. otherwise a ``NotImplementedError`` (names) or a derivation from
    #      ``self.wavefields`` / ``self.models`` (specs) provides the
    #      pre-spec-tables fallback.
    #
    # FieldSpec order is semantically significant. The propagators map
    # source/receiver names to positional wavefield indices, and equation
    # step functions are expected to return tensors in the same order.
    # Reordering field specs therefore changes forward/backward behavior,
    # source injection, receiver sampling, checkpointing, and CUDA bindings.

    @property
    def field_specs(self):
        override = self.__dict__.get('_sweep_field_specs_override')
        if override is not None:
            return override
        if self.FIELD_SPECS:
            return list(self.FIELD_SPECS)
        return ensure_field_specs(self.wavefields, [])

    @field_specs.setter
    def field_specs(self, value):
        self.__dict__['_sweep_field_specs_override'] = list(value)

    @property
    def model_specs(self):
        override = self.__dict__.get('_sweep_model_specs_override')
        if override is not None:
            return override
        if self.MODEL_SPECS:
            return list(self.MODEL_SPECS)
        return ensure_model_specs(self.models, [])

    @model_specs.setter
    def model_specs(self, value):
        self.__dict__['_sweep_model_specs_override'] = list(value)

    @property
    def wavefields(self):
        override = self.__dict__.get('_sweep_wavefields_override')
        if override is not None:
            return override
        if self.FIELD_SPECS:
            return [spec.name for spec in self.FIELD_SPECS]
        raise NotImplementedError(
            f"{type(self).__name__} must declare ``FIELD_SPECS`` or set "
            "``self.wavefields``."
        )

    @wavefields.setter
    def wavefields(self, value):
        self.__dict__['_sweep_wavefields_override'] = list(value)

    @property
    def models(self):
        override = self.__dict__.get('_sweep_models_override')
        if override is not None:
            return override
        if self.MODEL_SPECS:
            return [spec.name for spec in self.MODEL_SPECS]
        raise NotImplementedError(
            f"{type(self).__name__} must declare ``MODEL_SPECS`` or set "
            "``self.models``."
        )

    @models.setter
    def models(self, value):
        self.__dict__['_sweep_models_override'] = list(value)

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
        self.pd = StaggeredDerivative(spatial_order, device, backend, ndim=ndim)
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

class SecondOrderEquation(LaplaceGradientOps, WaveEquation):
    """
    Base class for second-order equations.
    This class can be extended to implement specific second-order equations.
    """

    def __init__(self, spatial_order=4, device='cpu', backend='torch', **kwargs):
        """
        Initialize the second-order equation with an initial condition.

        :param initial_condition: The initial condition for the equation.
        """
        LaplaceGradientOps.__init__(self, backend=backend)
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

        self.kf = kernel_func
        # Subclasses that need the gradient fast path call
        # ``self.init_grad_kernels()`` after ``__init__``; ``None`` means
        # ``self.gradient(u, h, axis, kernels=None)`` falls back to
        # ``torch.gradient`` (the slow general path).
        self.grad_kernels = None

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

    def init_separable_laplace(self):
        """Build the separable 1-D second-derivative kernel stack.

        Dispatches on ``self.ndim`` (2 or 3) — no explicit mode argument
        needed. The kernel stack is stored on ``self.laplace_kernels``
        and consumed by ``self.separable_d2_2d`` / ``separable_d2_3d``
        (per-axis components) and ``self.laplacian_2d`` /
        ``laplacian_3d`` (isotropic ``∇²u`` shortcut).

        Replaces the old ``init_laplace(ltype='1dsep'|'3dsep')`` entry,
        which required callers to repeat what ``self.ndim`` already
        encoded.
        """
        self.kernel = to_backend(
            self.kf(self.so, mode='x')[0, 0][self.so // 2, :],
            backend=self.backend, device=self.device,
        )
        self.laplace_kernels = self._prepare_separable_laplace_kernels()

    def init_grad_kernels(self):
        """Build 2-D 1st-derivative fixed-stencil kernels for the fast path.

        Stored as ``self.gkernel_x / self.gkernel_z`` and bundled into
        ``self.grad_kernels = {-2: gkernel_z, -1: gkernel_x}``, ready to
        pass through ``self.gradient(u, h, axis, kernels=self.grad_kernels)``
        in the equation's step function. The fast path uses a single
        ``conv2d`` per axis and is ~5-10x faster than the default
        ``torch.gradient`` fallback on CUDA.

        Call after :meth:`init_separable_laplace`. Only meaningful for 2-D
        equations — 3-D variants currently build their own
        ``self.grad_kernels`` dict from a custom 3-D stencil.
        """
        self.gkernel_x = to_backend(
            self.kf(self.so, derivative_order=1, mode='x',
                    no_center=True, grid='normal', sign=-1),
            backend=self.backend, device=self.device,
        )
        self.gkernel_z = to_backend(
            self.kf(self.so, derivative_order=1, mode='z',
                    no_center=True, grid='normal', sign=-1),
            backend=self.backend, device=self.device,
        )
        self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}

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


    
