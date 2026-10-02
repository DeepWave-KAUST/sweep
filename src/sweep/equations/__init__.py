"""Wave equations.

Every equation module decorates its class with ``@register_equation`` (see
``_registry.py``); importing the modules below runs those decorators, and the
whole public name set -- canonical names, author-named aliases, and the
per-symmetry ``*Default*`` routing -- is bound at module scope straight from the
registry. Adding an equation is "write the module, decorate the class"; there is
nothing to edit here and no ``globals()`` scan to keep honest.

Alias sets live next to their classes, so the citation-to-class mapping is
visible where the physics is:

* ``AcousticVTILiang`` / ``AcousticTTILiang`` -- Liang K. et al. (2022),
  2nd-order pseudo-acoustic, 10.1190/geo2022-0292.1
* ``AcousticVTIAlkhalifah`` / ``AcousticTTIAlkhalifah`` -- Tariq Alkhalifah
  (2000), qP eta-formulation, 10.1190/1.1444815 (one class covers VTI and TTI)
* ``AcousticVTIDuveneck``/``3D`` -- Duveneck et al. (2008), 1st-order
  velocity-stress, 10.1190/1.3059320
* ``DASElastic``/``3D`` -- alias of ``DASZhao``/``3D``
* ``ElasticAPM`` -- alias of ``Elastic`` (Cao & Chen 2018 APM free surface;
  the same class, dispatched by the propagator's ``topo_method='apm'``)

The ``*Default*`` names point at the class a user gets when they just want "the
standard" solver for that symmetry: 2-D VTI routes to Liang's 2nd-order scalar
(memory-light, the original SWEEP entry), 3-D VTI to Duveneck's first-order
because it is the only one with a 3-D class.
"""
from ._registry import (
    equation_classes,
    equation_method,
    get_equation,
    list_equations,
    register_equation,
)
from .base import WaveEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec

# Importing each module is what runs its @register_equation decorator.
from . import (
    acoustic,
    acoustic1st,
    acoustic3d,
    acoustic_curvilinear,
    acoustic_lsrtm,
    acoustic_lsrtm3d,
    acoustic_vrr,
    acoustic_vrz,
    acoustic_vti_1st,
    elastic,
    elastic3d,
    elastic_apm,
    elastic_curvilinear,
    elastic_tti,
    elastic_tti_2nd,
    elastic_tti_sg,
    elastic_tti_sg3d,
    elastic_vrr,
    qP_tariq,
    qP_vti,
    visco_acoustic,
    visco_elastic,
)

# Public exports that are NOT WaveEquation subclasses, so they are not in the
# registry: a ``__new__`` dispatch facade and a helper function.
from .acoustic_aniso import AcousticAniso
from .elastic_vrr import compute_vector_reflectivity

# ``das`` imports torch at module top; guard it so a jax-only environment can
# still ``import sweep.equations`` and use the jax-only equations. The four DAS
# equation classes self-register; only the nn.Module facade and the DSP helpers
# are bound explicitly.
try:
    from . import das  # noqa: F401  (registers DASZhao/3D, DASMu/3D + DASElastic aliases)
    from .das import DAS, DASModeler, gauge_average, helical_das_response
except ModuleNotFoundError as _das_import_err:
    if _das_import_err.name != "torch":
        raise
    DAS = DASModeler = gauge_average = helical_das_response = None
    DASZhao = DASZhao3D = DASMu = DASMu3D = None
    DASElastic = DASElastic3D = None

# qP_tti (eta-acoustic TTI) is optional for the same reason.
try:
    from . import qP_tti  # noqa: F401  (registers AcousticTTI + AcousticTTILiang)
except ModuleNotFoundError:
    AcousticTTI = AcousticTTILiang = None

# Bind every registered public name at module scope, so
# ``from sweep.equations import Acoustic`` / ``AcousticVTILiang`` / ``Elastic3D``
# all keep working.
globals().update(equation_classes())


def _equation_classes():
    """Deprecated alias of :func:`equation_classes`.

    Kept only for companion repos (sweep-tasks, sweep-agent) that still import
    it. Nothing inside sweep uses it -- ``test_equation_registry_public_api``
    keeps it that way -- so this can go once those repos are updated.
    """
    return equation_classes()


def supports_torch_binding(equation):
    """Check whether an equation class or exported equation name supports ``sweep._C``."""
    if isinstance(equation, str):
        equation_cls = equation_classes().get(equation)
        if equation_cls is None:
            raise KeyError(f"Unknown equation '{equation}'")
        return equation_cls.supports_torch_binding()

    if isinstance(equation, type) and issubclass(equation, WaveEquation):
        return equation.supports_torch_binding()

    if isinstance(equation, WaveEquation):
        return equation.__class__.supports_torch_binding()

    raise TypeError("equation must be an equation name, equation class, or equation instance")


def torch_binding_supported_equations():
    """Return the exported equation names that support the compiled PyTorch binding."""
    return sorted(
        name for name, equation_cls in equation_classes().items()
        if equation_cls.supports_torch_binding()
    )
