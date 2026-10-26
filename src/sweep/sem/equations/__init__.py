"""SEM equation kernels (weak-form time-step functions).

Every module in this package that defines a class with ``field_specs`` is an
equation, and this package DISCOVERS them rather than listing them. Three
equations had been implemented but never added to the hand-kept export list
(``AcousticPotentialSEM2D``, ``AcousticElasticSEM2D``, ``ElasticSEM3DDeformed``),
so they were unreachable as ``sweep.sem.X`` and callers had to reach into the
private module path. Discovery makes that failure impossible: dropping a new
equation module into this directory exports it.

The listing below is likewise generated (see ``describe()``), so it cannot drift
from the code the way the previous hand-written table did.
"""

from __future__ import annotations

import importlib
import pkgutil

_EQUATIONS: dict[str, type] = {}
_MODULE_OF: dict[str, str] = {}

for _info in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f"{__name__}.{_info.name}")
    for _name, _obj in vars(_mod).items():
        # An equation is a class that declares field_specs and is DEFINED here
        # (not merely imported into the module's namespace).
        if (isinstance(_obj, type) and hasattr(_obj, "field_specs")
                and getattr(_obj, "__module__", None) == _mod.__name__):
            _EQUATIONS[_name] = _obj
            _MODULE_OF[_name] = _info.name
            globals()[_name] = _obj

del _info, _mod, _name, _obj

__all__ = sorted(_EQUATIONS)


def equation_classes() -> dict[str, type]:
    """``{class name: class}`` for every equation in this package."""
    return dict(_EQUATIONS)


def summary(cls) -> dict:
    """What picks an equation, as data -- the columns of :func:`describe`.

    ``mesh`` (``"quad4"``, ``"quad9"``, ``"hex8"`` or ``"hex27"``), ``ndim``,
    ``topography`` (the mesh family admits one), ``cpml`` (``"yes"``,
    ``"split"`` or ``"NO"``, see :func:`describe`), ``fields`` (the state, in
    order), ``sources`` and ``receivers`` (the fields a source drives and the
    receivers sample) and ``derived`` (the receiver components computed from
    those, such as ``p``). ``sweep show`` prints it.
    """
    import inspect

    from ..fields import MESH_NDIM, mesh_kind
    from ..mesh import MESH_BUILDERS

    kind = mesh_kind(cls)
    params = inspect.signature(cls.build).parameters
    # "split" = the fluid-solid form, which takes cpml_ac AND cpml_el
    # rather than one cpml. PropSEM supplies both via the equation's
    # cpml_builders declaration, so "split" is informational, not a gap.
    cpml = ("yes" if "cpml" in params
            else "split" if any(k.startswith("cpml_") for k in params)
            else "NO")
    return {
        "mesh": kind,
        "ndim": MESH_NDIM[kind],
        "topography": MESH_BUILDERS[kind][1] is not None,
        "cpml": cpml,
        "fields": [s.name for s in cls.field_specs],
        "sources": [s.name for s in cls.field_specs if s.supports_source],
        "receivers": [s.name for s in cls.field_specs if s.supports_receiver],
        "derived": list(getattr(cls, "derived_receiver_types", {})),
    }


def describe() -> str:
    """One line per equation: everything needed to pick one.

    The columns exist because none of this is inferable from the class name.
    ``AcousticPotentialSEM2D`` has no "Deformed" in its name yet runs on a
    curved mesh and supports topography; ``ElasticVSSEM3D`` has none either and
    does not. And ``VS`` (velocity-stress) is a different FORMULATION, not a
    different mesh -- read it off the recorded fields: ``u_*`` is the 2nd-order
    displacement scheme, ``v_*`` the 1st-order velocity-stress one.

    ``topo`` says the mesh family admits a ``topography``; ``cpml`` says
    ``build()`` accepts an absorbing profile at all. An equation with
    ``cpml no`` is a reflecting box: PropSEM refuses an ``abcn`` for it.
    """
    head = (f"  {'equation':26s} {'mesh':6s} {'D':>1s} {'topo':>5s} "
            f"{'cpml':>5s} {'flds':>5s}  records")
    rows = [head, "  " + "-" * (len(head) - 2)]
    for name in __all__:
        s = summary(_EQUATIONS[name])
        topo = "yes" if s["topography"] else "no"
        rows.append(f"  {name:26s} {s['mesh']:6s} {s['ndim']:>1d} {topo:>5s} "
                    f"{s['cpml']:>5s} {len(s['fields']):>5d}  "
                    f"{', '.join(s['receivers'])}")
    return "\n".join(rows)
