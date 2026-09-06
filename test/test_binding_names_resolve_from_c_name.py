"""Compiled bindings resolve from ``C_NAME``, including the persistent runners.

Three equations hand-wrote their own ``_C()`` returning an import tuple and
declared no ``C_NAME``. ``_C()`` worked -- but ``_compiled_runner_factories()``
resolves from ``C_NAME`` alone, so it returned ``(None, None)`` for them and the
propagator silently fell back to the per-call stepped path.

That was not hypothetical: ``elastic_tti_sg3d_forward_runner`` and
``elastic_tti_sg3d_backward_bs_runner`` are compiled and bound, and no Python
could reach them.
"""
import pathlib

import pytest

import sweep.equations as eq
from conftest import requires_binding
from sweep.equations.base import WaveEquation

SRC = pathlib.Path(eq.__file__).resolve().parent

#: Equations whose ``_C`` legitimately raises instead of returning bindings.
_EAGER_ONLY_OVERRIDES = {"AcousticCurvilinear", "ElasticCurvilinear"}


def _registered():
    return sorted(eq.equation_classes().items())


def test_no_equation_hand_writes_a_binding_tuple():
    """A hand-written ``_C`` cannot carry ``C_NAME``'s other consumers with it."""
    offenders = []
    for name, cls in _registered():
        own = cls.__dict__.get("_C")
        if own is None or own is WaveEquation._compiled_funcs:
            continue
        if name in _EAGER_ONLY_OVERRIDES:
            continue
        offenders.append(name)
    assert not offenders, (
        "these declare a hand-written _C() instead of C_NAME, so "
        f"_compiled_runner_factories() cannot see them: {offenders}")


@pytest.mark.parametrize("name", ["ViscoAcoustic", "ElasticTTISG3D", "ElasticTTI2nd"])
def test_the_three_converted_equations_declare_c_name(name):
    cls = eq.equation_classes()[name]
    assert cls.C_NAME, f"{name} must declare C_NAME"
    assert cls._C is WaveEquation._compiled_funcs
    assert cls.supports_torch_binding()


def test_every_binding_equation_can_be_asked_for_runners():
    """The lookup must not raise for any equation, built or not."""
    for name, cls in _registered():
        if not cls.C_NAME:
            continue
        assert isinstance(cls.C_NAME, str) and cls.C_NAME


def test_the_bound_runner_symbols_all_have_an_owning_equation():
    """Every ``*_forward_runner`` compiled in must be reachable via some C_NAME."""
    module = SRC.parent / "csrc" / "bindings" / "module.cpp"
    if not module.exists():
        pytest.skip("csrc not present in this checkout")
    text = module.read_text()
    bound = {line.split('m.def("')[1].split('"')[0]
             for line in text.splitlines() if 'm.def("' in line and '_forward_runner"' in line}
    owned = {f"{cls.C_NAME}_forward_runner"
             for _, cls in _registered() if cls.C_NAME}
    orphans = sorted(bound - owned)
    assert not orphans, (
        "these persistent runners are compiled but no equation's C_NAME "
        f"resolves them, so Python can never reach them: {orphans}")


@requires_binding("elastic_tti_sg3d_forward", "elastic_tti_sg3d_forward_runner")
def test_the_converted_tuple_is_the_same_five_symbols():
    """The conversion must not change which bindings an equation resolves."""
    import sweep._C as _C

    cls = eq.equation_classes()["ElasticTTISG3D"]
    resolved = cls(spatial_order=4, device="cpu")._C()
    assert resolved == (
        _C.elastic_tti_sg3d_forward,
        _C.elastic_tti_sg3d_backward,
        _C.elastic_tti_sg3d_backward_bs,
        _C.elastic_tti_sg3d_backward_ckpt,
        None,
    )


@requires_binding("elastic_tti_sg3d_forward_runner", "elastic_tti_sg3d_backward_bs_runner")
def test_the_persistent_runners_are_now_reachable():
    """This is what the hand-written _C() silently withheld."""
    cls = eq.equation_classes()["ElasticTTISG3D"]
    fwd, bwd = cls(spatial_order=4, device="cpu")._compiled_runner_factories()
    assert fwd is not None and bwd is not None
