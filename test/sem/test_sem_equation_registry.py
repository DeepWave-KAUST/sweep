"""Every implemented equation is reachable as ``sweep.sem.<Name>``.

The export list used to be hand-kept in four places (two lists in
``sweep/sem/__init__.py``, two in ``sweep/sem/equations/__init__.py``). Three
equations had been implemented and never added: ``AcousticPotentialSEM2D``,
``AcousticElasticSEM2D`` and ``ElasticSEM3DDeformed``. Nothing failed -- they
were simply invisible, and callers reached into the private module path
(``sweep.sem.equations.acoustic_potential2d``) to get at them.

The package now discovers equations by scanning its own modules. These tests
pin that, because the failure mode of a registry is silence: a missing entry
raises nothing, it just makes a working equation unreachable.
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

import sweep.sem as S
import sweep.sem.equations as EQ


def _defined_equations():
    """Scan the package independently of the package's own discovery."""
    found = {}
    for info in pkgutil.iter_modules(EQ.__path__):
        mod = importlib.import_module(f"sweep.sem.equations.{info.name}")
        for name, obj in vars(mod).items():
            if (isinstance(obj, type) and hasattr(obj, "field_specs")
                    and getattr(obj, "__module__", None) == mod.__name__):
                found[name] = info.name
    return found


def test_every_equation_is_reachable_from_sweep_sem():
    missing = sorted(n for n in _defined_equations() if not hasattr(S, n))
    assert not missing, f"defined but unreachable as sweep.sem.X: {missing}"


def test_every_equation_is_in_all():
    missing = sorted(set(_defined_equations()) - set(S.__all__))
    assert not missing, f"defined but absent from sweep.sem.__all__: {missing}"


def test_discovery_agrees_with_an_independent_scan():
    """The package's own discovery must find exactly what a fresh scan finds --
    otherwise the filter (``__module__`` check) is silently dropping or
    double-counting something."""
    assert set(EQ.equation_classes()) == set(_defined_equations())


def test_no_duplicate_exports():
    """Splicing two lists together is how duplicates appear; they shadow
    silently and make ``__all__`` a poor contract."""
    dupes = sorted({n for n in S.__all__ if S.__all__.count(n) > 1})
    assert not dupes, f"duplicated in sweep.sem.__all__: {dupes}"


def test_the_three_previously_missing_ones_specifically():
    """Named explicitly so a future re-introduction of a hand-kept list cannot
    quietly drop the same three again."""
    for name in ("AcousticPotentialSEM2D", "AcousticElasticSEM2D",
                 "ElasticSEM3DDeformed"):
        assert hasattr(S, name), name
        assert name in S.__all__, name


def test_each_equation_is_attributed_to_the_module_that_defines_it():
    """Three equation modules import another equation class into their own
    namespace (``acoustic_elastic2d`` pulls in ``AcousticPotentialSEM2D`` and
    ``ElasticSEM2DDeformed``; ``elastic_vti_vs_deformed`` subclasses
    ``ElasticVSSEM2DDeformed``). Discovery must therefore filter on
    ``__module__``.

    A set comparison CANNOT catch that filter being dropped -- the leaked class
    is already found via its own module, so the set is unchanged and only the
    attribution moves. This asserts the attribution, which is the part that
    actually breaks.
    """
    for name, cls in EQ.equation_classes().items():
        owner = cls.__module__.rsplit(".", 1)[-1]
        assert EQ._MODULE_OF[name] == owner, (
            f"{name} attributed to {EQ._MODULE_OF[name]}, defined in {owner}")


def test_everything_exported_is_actually_an_equation():
    """The reverse direction: discovery must not export a helper class that
    happens to live in an equation module."""
    for name in EQ.__all__:
        cls = getattr(S, name)
        assert isinstance(cls, type) and hasattr(cls, "field_specs"), name
        assert cls.field_specs, f"{name} declares an empty field_specs"


def test_describe_covers_every_equation():
    text = EQ.describe()
    for name in EQ.__all__:
        assert name in text, f"{name} missing from describe()"


@pytest.mark.parametrize("name", sorted(_defined_equations()))
def test_each_equation_declares_a_usable_field_layout(name):
    """Every field's ``layout`` must be one the state allocator understands --
    a typo would only surface as a crash inside init_state at run time."""
    from sweep.sem.fields import LAYOUTS
    for spec in getattr(S, name).field_specs:
        assert spec.layout in LAYOUTS, f"{name}.{spec.name}: {spec.layout!r}"


def test_describe_reports_the_facts_a_user_picks_on():
    """None of mesh family / topography / C-PML support is inferable from the
    class name -- AcousticPotentialSEM2D has no "Deformed" in its name and does
    support topography; ElasticVSSEM3D has none either and does not. describe()
    is the place those are answerable without reading the source."""
    import inspect

    from sweep.sem.fields import mesh_kind
    from sweep.sem.mesh import MESH_BUILDERS

    text = EQ.describe()
    for name in EQ.__all__:
        cls = getattr(S, name)
        kind = mesh_kind(cls)
        line = next(l for l in text.splitlines() if l.strip().startswith(name))
        assert kind in line, f"{name}: mesh family missing from describe()"
        # topography column must match the builder table
        has_topo = MESH_BUILDERS[kind][1] is not None
        assert (" yes " in line) == has_topo or not has_topo, name
        # a C-PML-less equation must not be reported as having one
        params = inspect.signature(cls.build).parameters
        if "cpml" not in params and not any(k.startswith("cpml_") for k in params):
            assert " NO " in line, f"{name} has no C-PML but describe() hides it"


def test_describe_marks_the_three_equations_without_any_cpml():
    """ElasticSEM2D and the two 3-D displacement elastic equations model no
    absorbing layer in either backend. That is the single most surprising fact
    in the package -- PropSEM refuses an abcn for them."""
    text = EQ.describe()
    for name in ("ElasticSEM2D", "ElasticSEM3D", "ElasticSEM3DDeformed"):
        line = next(l for l in text.splitlines() if l.strip().startswith(name))
        assert " NO " in line, f"{name} should be flagged as having no C-PML"
