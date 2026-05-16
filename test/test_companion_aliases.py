"""Tests for the `sweep.<short>` companion-alias machinery.

These tests verify the import-side wiring in ``sweep/__init__.py``:

- ``sweep.io`` / ``sweep.runner`` / etc. resolve to the installed companion
  distribution (``sweep_io`` / ``sweep_runner`` / …).
- The same alias is registered under ``sys.modules['sweep.<short>']`` so
  ``from sweep.io import X`` and ``import sweep.io`` also work.
- Missing companions raise an ``AttributeError`` that names the ``pip
  install`` command to fix it.
- The ``_CompanionFinder`` meta-path entry is installed.
"""

from __future__ import annotations

import importlib
import sys
from importlib.util import find_spec

import pytest

import sweep
from sweep import _CompanionFinder


def _installed_companions() -> list[tuple[str, str]]:
    # Re-resolve each call so `importlib.reload(sweep)` in another test
    # doesn't leave us holding a stale dict reference.
    return [
        (short, full)
        for short, full in sweep._COMPANION_ALIASES.items()
        if find_spec(full) is not None
    ]


def test_companion_aliases_dict_uses_underscored_names():
    """Each value should be a valid Python identifier (hyphens forbidden)."""
    for short, full in sweep._COMPANION_ALIASES.items():
        assert "-" not in full, f"_COMPANION_ALIASES[{short!r}] = {full!r} has a hyphen"
        assert full.startswith("sweep_"), f"{full!r} should start with 'sweep_'"


def test_companion_finder_is_installed_on_meta_path():
    assert any(isinstance(f, _CompanionFinder) for f in sys.meta_path), (
        "_CompanionFinder must be on sys.meta_path for `import sweep.io` to work"
    )


def test_companion_finder_idempotent_on_reimport():
    """Reimporting sweep shouldn't append a second finder."""
    before = sum(isinstance(f, _CompanionFinder) for f in sys.meta_path)
    importlib.reload(sweep)
    after = sum(isinstance(f, _CompanionFinder) for f in sys.meta_path)
    assert before == after == 1


@pytest.mark.parametrize("short,full", _installed_companions())
def test_attribute_access_returns_installed_companion(short, full):
    # Drop any cached state so we exercise the full attribute path.
    sys.modules.pop(f"sweep.{short}", None)
    if short in sweep.__dict__:
        del sweep.__dict__[short]
    mod = getattr(sweep, short)
    assert mod.__name__ == full
    # The alias must also be registered for `from sweep.<short> import X` to work.
    assert sys.modules[f"sweep.{short}"] is mod


@pytest.mark.parametrize("short,full", _installed_companions())
def test_import_dotted_resolves_via_finder(short, full):
    sys.modules.pop(f"sweep.{short}", None)
    mod = importlib.import_module(f"sweep.{short}")
    assert mod.__name__ == full


def test_missing_companion_raises_helpful_error(monkeypatch):
    """Pretend `sweep_io` is not installed and confirm the message."""
    fake_short = "fake_missing_alias_zzz"
    fake_full = "definitely_not_a_real_package_zzz"
    monkeypatch.setitem(sweep._COMPANION_ALIASES, fake_short, fake_full)
    sys.modules.pop(f"sweep.{fake_short}", None)
    with pytest.raises(AttributeError) as exc:
        getattr(sweep, fake_short)
    msg = str(exc.value)
    assert "companion package" in msg
    assert fake_full in msg
    assert "pip install" in msg


def test_non_alias_name_raises_plain_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        sweep.totally_made_up_name_xyz  # noqa: B018
