"""The equation registry has one public spelling, and sweep uses it.

``_equation_classes`` is a deprecated alias kept for companion repos. sweep's own
CLI reached for it, which is how a private name stays alive forever: the library
that deprecated it is also its biggest caller, so the deprecation never bites and
the "public" name is the one nobody exercises.
"""
import pathlib

import pytest

import sweep.equations as eq

SRC = pathlib.Path(eq.__file__).resolve().parent.parent
DEFINITION = SRC / "equations" / "__init__.py"


def test_public_name_is_the_registry():
    public = eq.equation_classes()
    assert isinstance(public, dict) and public, "the registry must not be empty"
    assert "Acoustic" in public


def test_deprecated_alias_still_answers_for_companion_repos():
    assert eq._equation_classes() == eq.equation_classes()


def test_nothing_in_sweep_calls_the_deprecated_alias():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == DEFINITION:
            continue
        if "_equation_classes" in path.read_text():
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, (
        "these modules call the deprecated alias instead of "
        f"equation_classes(): {offenders}"
    )


@pytest.mark.parametrize("name", ["Acoustic", "Elastic"])
def test_registry_entries_are_equation_classes(name):
    cls = eq.equation_classes()[name]
    assert isinstance(cls, type)
    assert hasattr(cls, "MODEL_SPECS")
