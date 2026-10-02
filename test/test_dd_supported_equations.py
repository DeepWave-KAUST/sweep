"""DD admission is DECLARED on cuda_layout, not name-listed.

Two capabilities gate it (see ``check_dd_admission``): ``stepped`` -- the
compiled forward/backward_bs honour it_begin/it_end -- and
``dd_backward_phases`` -- the compiled backward implements the numbered
phases its schedule drives.  The retired ``_DD_EQUATIONS`` whitelist keyed on
class names; the 2-D and 3-D elastic classes are both literally named
``Elastic``, and one entry covering both by coincidence was pinned as
load-bearing here.  Declarations make that coincidence irrelevant.

The 2-D AcousticVRZ is the boundary case this design exists for: STEPPED
since it joined the shared template drivers, yet still refused -- loudly,
with the reason -- because its schedule (the VRZ coupling exchange) needs
backward phases only the 3-D sibling implements.  An equation slipping past
either check would not crash; it would run the full record on every stepped
call, or run phased calls un-phased -- silently wrong, which is worse than
unsupported.
"""
import pathlib
import re

import pytest

from sweep.equations.slot_table import slot_table_of
from sweep.parallel.dd_propagator import check_dd_admission
from sweep.parallel.dd_spec import ACOUSTIC_DD, ELASTIC_DD, VRZ_DD

DEV = "cpu"

ACCEPTED = ["Acoustic", "Acoustic3D", "AcousticVRZ3D", "Elastic", "Elastic3D"]


def _make(name):
    import sweep.equations as eqs
    return getattr(eqs, name)(device=DEV, backend="torch")


def _spec_of(eq):
    table = slot_table_of(eq)
    layout = eq.cuda_layout
    return (ELASTIC_DD if not table.u_blocks
            else VRZ_DD if layout.dd_coupling_nvar else ACOUSTIC_DD)


def test_accepted_equations_declare_their_capabilities():
    for name in ACCEPTED:
        eq = _make(name)
        layout = eq.cuda_layout
        assert layout.stepped, name
        assert layout.slots is not None, name
        check_dd_admission(eq, layout, _spec_of(eq))   # must not raise


def test_vrz2d_is_stepped_but_still_refused():
    """The honest ladder: capability declared, gap named."""
    eq = _make("AcousticVRZ")
    layout = eq.cuda_layout
    assert layout.stepped, "template migration made the 2-D VRZ stepped"
    assert not layout.dd_backward_phases
    assert layout.dd_coupling_nvar > 0, "which is why its schedule needs them"
    with pytest.raises(NotImplementedError, match="dd_backward_phases"):
        check_dd_admission(eq, layout, _spec_of(eq))


def test_unstepped_equations_are_refused_with_the_requirement():
    for name in ("AcousticLSRTM", "DASMu", "ElasticTTISG", "ViscoElastic"):
        eq = _make(name)
        with pytest.raises(NotImplementedError, match="stepped"):
            check_dd_admission(eq, eq.cuda_layout, ACOUSTIC_DD)


def test_subclasses_inherit_the_declaration():
    """cuda_layout is a property: a subclass of a supported equation carries
    its declarations with no extra work (the whitelist needed an MRO walk)."""
    from sweep.equations import Acoustic

    class MyAcoustic(Acoustic):
        pass

    eq = MyAcoustic(device=DEV, backend="torch")
    check_dd_admission(eq, eq.cuda_layout, _spec_of(eq))   # must not raise


def test_no_name_list_survives():
    """Admission must never regress to class-name matching."""
    import sweep.parallel.dd_propagator as m

    src = pathlib.Path(m.__file__).read_text()
    # Match DEFINITIONS, not mentions: the admission docstring may name the
    # retired mechanism when explaining what replaced it.
    assert not re.search(r"^_DD_EQUATIONS\s*=", src, re.M)
    assert not re.search(r"^def _family_of\b", src, re.M)
    assert not re.search(r'["\']elastic["\']\s+in\s+name', src)
    assert not re.search(r'["\']acoustic["\']\s+in\s+name', src)


def test_accepted_equations_all_resolve_to_a_schedule():
    """Every DD-accepted equation must select one of the declared schedules.

    The driver used to branch on family in ~20 places; now it picks a DDSpec
    from what the equation declares -- a fixed-slot layout means the staggered
    first-order protocol, a divergence-form gradient means the coupling
    exchange, otherwise the plain second-order one. If an equation is accepted
    but resolves to nothing, the failure would be an AttributeError deep inside
    the first backward, so check it up front.
    """
    for name in ACCEPTED:
        eq = _make(name)
        spec = _spec_of(eq)
        assert spec.forward.phases, f"{name}: schedule has no forward phases"
        assert spec.backward.phases, f"{name}: schedule has no backward phases"
        # Exactly one phase per loop may advance the buffer-role counters.
        for loop, label in ((spec.forward, "forward"), (spec.backward, "backward")):
            n = sum(1 for ph in loop.phases if ph.advances)
            assert n == 1, f"{name}: {label} loop advances {n} phases"


def test_phase_need_is_derived_from_the_spec_not_hardcoded():
    """The dd_backward_phases requirement follows the schedule's own phase
    numbers: the plain acoustic schedule phases nothing, the elastic and VRZ
    schedules do."""
    def needs(spec):
        bwd = spec.backward
        return any(ph.step_phase is not None
                   for ph in tuple(getattr(bwd, "prologue", ()) or ()) + tuple(bwd.phases))

    assert not needs(ACOUSTIC_DD)
    assert needs(ELASTIC_DD)
    assert needs(VRZ_DD)
