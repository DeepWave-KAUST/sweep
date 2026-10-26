"""`sweep show` resolves through the registry and reports failure in its exit code.

Two defects this pins:

* resolution was ``getattr(sweep.equations, name)``, so any module attribute
  answered -- ``sweep show FieldSpec`` printed "FieldSpec is a
  facade/dispatcher", which it is not;
* every path returned ``None``, so ``main()`` returned 0 and a typo'd equation
  name exited successfully. Nothing scripting the CLI could tell.
"""
import sys

import pytest

import sweep.equations as eq
from sweep.cli import list_wavefields


def test_a_registered_equation_succeeds(capsys):
    assert list_wavefields("Acoustic") == 0
    out = capsys.readouterr().out
    assert "=== Acoustic ===" in out
    assert "Needed models" in out


def test_an_unknown_name_fails(capsys):
    assert list_wavefields("NoSuchEquation") == 1
    assert "No such wave equation" in capsys.readouterr().out


def test_a_typo_is_offered_the_real_name(capsys):
    assert list_wavefields("Acoustic3d") == 1
    out = capsys.readouterr().out
    assert "Did you mean" in out and "Acoustic3D" in out


@pytest.mark.parametrize("name", ["FieldSpec", "WaveEquation", "ModelSpec"])
def test_a_non_equation_module_attribute_is_not_called_a_facade(name, capsys):
    """These are importable from sweep.equations but are not equations."""
    if not hasattr(eq, name):
        pytest.skip(f"{name} is not exported by sweep.equations")
    assert list_wavefields(name) == 1
    out = capsys.readouterr().out
    assert "facade" not in out, out
    assert "No such wave equation" in out


@pytest.mark.parametrize("name", ["DAS", "AcousticAniso"])
def test_the_real_facades_are_named_as_facades(name, capsys):
    if not hasattr(eq, name):
        pytest.skip(f"{name} needs an optional dependency")
    assert list_wavefields(name) == 1
    assert "facade" in capsys.readouterr().out


def test_main_propagates_the_failure(monkeypatch):
    from sweep.cli import main
    monkeypatch.setattr(sys, "argv", ["sweep", "show", "NoSuchEquation"])
    assert main() == 1


# --- the spectral-element equations (sweep.sem) -----------------------------
# They are not WaveEquation subclasses and run with PropSEM, so the FD registry
# does not hold them; `sweep list` / `sweep show` read sweep.sem's own.

def test_list_names_both_families(capsys):
    import sweep.sem.equations as sem
    from sweep.cli import list_equations

    list_equations()
    out = capsys.readouterr().out
    assert "Available equations:" in out
    assert "Spectral element (sweep.sem, run with PropSEM)" in out
    rows = {line.split()[0] for line in out.splitlines() if line.startswith("  ")}
    missing = (set(eq.equation_classes()) | set(sem.equation_classes())) - rows
    assert not missing, missing


def test_a_sem_equation_shows_what_propsem_takes(capsys):
    assert list_wavefields("AcousticPotentialSEM2D") == 0
    out = capsys.readouterr().out
    assert "=== AcousticPotentialSEM2D ===" in out
    assert "run with PropSEM" in out
    assert "Mesh: quad9, 2-D, topography: yes" in out
    # rho is PropSEM's density=, not a model
    assert "Needed models: ['vp']" in out
    # p is derived from chi; both are receiver types
    line = next(l for l in out.splitlines() if "Receiver types" in l)
    assert "'chi'" in line and "'p'" in line


def test_a_sem_equation_without_cpml_says_so(capsys):
    assert list_wavefields("ElasticSEM2D") == 0
    assert "C-PML: no" in capsys.readouterr().out


def test_a_typo_is_offered_the_sem_name(capsys):
    assert list_wavefields("AcousticPotentialSEM") == 1
    out = capsys.readouterr().out
    assert "Did you mean" in out and "AcousticPotentialSEM2D" in out
    assert "spectral-element" in out
