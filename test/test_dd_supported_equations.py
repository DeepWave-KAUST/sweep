"""``ModelParallel`` accepts only equations whose CUDA kernels are stepped.

DD runs the solver one time step at a time and exchanges halos in between, so
the equation's forward AND backward must honour ``p.it_begin`` / ``p.it_end``.
The check used to be a substring match on the class name, which let in every
``Acoustic*`` / ``Elastic*`` variant in the library. That is not a harmless
over-acceptance: an equation without a stepped forward runs
``for (it = 0; it < p.nt; ++it)`` regardless, so a "run one step" call runs the
whole record and DD exchanges halos of a wavefield that already reached nt --
no exception, just a wrong answer.

None of this needs a GPU: the whitelist is consulted before any CUDA work.
"""

import pathlib
import re

import pytest

from sweep.parallel.dd_propagator import _DD_EQUATIONS, _FAMILIES, _family_of

DEV = "cpu"

# (class name in sweep.equations, expected family). These are exactly the
# equations with stepped forward.cu AND backward.cu under csrc/cuda/equations.
SUPPORTED = [
    ("Acoustic", "acoustic"),
    ("Acoustic3D", "acoustic"),
    ("AcousticVRZ3D", "acoustic"),
    ("Elastic", "elastic"),
    ("Elastic3D", "elastic"),
]

# Named acoustic/elastic, so the old substring match accepted every one of
# them. AcousticVRZ is the sharpest: its 3-D sibling IS stepped.
REFUSED = ["AcousticVRZ", "AcousticVTI", "AcousticVTI1st", "AcousticTTI",
           "ElasticTTI", "ElasticVRR", "ViscoAcoustic"]


def _make(name):
    import sweep.equations as eqs
    cls = getattr(eqs, name, None)
    if cls is None:
        pytest.skip(f"{name} is not exported by this build")
    try:
        return cls(device=DEV, backend="torch")
    except Exception as e:                      # equation needs args we lack
        pytest.skip(f"{name} could not be constructed on CPU: {e}")


@pytest.mark.parametrize("name,family", SUPPORTED)
def test_supported_equations_resolve_to_a_known_family(name, family):
    assert _family_of(_make(name)) == family
    assert family in _FAMILIES, "family has no wavefield geometry entry"


@pytest.mark.parametrize("name", REFUSED)
def test_unstepped_equations_are_refused_not_silently_run(name):
    with pytest.raises(NotImplementedError) as e:
        _family_of(_make(name))
    msg = str(e.value)
    assert name in msg, "the error must name the equation that was refused"
    assert "stepped" in msg, "the error must say what the requirement is"
    for supported in sorted(_DD_EQUATIONS):
        assert supported in msg, "the error must list what IS supported"


def test_elastic_2d_and_3d_share_one_class_name():
    """``Elastic3D`` is ``elastic3d.Elastic`` under an alias, so one whitelist
    entry covers both. If that ever changes, the 3-D class silently drops out
    of the whitelist and DD starts refusing 3-D elastic."""
    from sweep.equations import Elastic, Elastic3D

    assert Elastic.__name__ == Elastic3D.__name__ == "Elastic"
    assert Elastic is not Elastic3D, "different modules, same class name"


def test_subclasses_of_a_supported_equation_still_work():
    from sweep.equations import Acoustic

    class MyAcoustic(Acoustic):
        pass

    assert _family_of(MyAcoustic(device=DEV, backend="torch")) == "acoustic"


def test_the_substring_guess_does_not_come_back():
    """The whitelist is only worth something while nothing falls back to a
    name match; a reintroduced substring test would re-accept every variant."""
    import sweep.parallel.dd_propagator as m

    src = pathlib.Path(m.__file__).read_text()
    body = src[src.index("def _family_of"):]
    body = body[:body.index("\nclass ")]
    assert not re.search(r'["\']elastic["\']\s+in\s+name', body)
    assert not re.search(r'["\']acoustic["\']\s+in\s+name', body)
