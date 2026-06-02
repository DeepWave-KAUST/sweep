"""Top-level package helpers for sweep.

In addition to the wave-equation engine submodules (`equations`, `propagator`,
`operators`, …), this package re-exposes the **companion distributions** under
short namespace aliases::

    import sweep
    sweep.io.SEGYReader(...)               # actually sweep_io.SEGYReader
    from sweep import runner                 # actually sweep_runner
    from sweep.tasks import TaskRunner       # actually sweep_tasks.TaskRunner

This works for any companion that's `pip install`'d alongside sweep. Missing
companions surface a helpful ``AttributeError`` pointing at the right
``pip install`` command. The companion packages keep their real distribution
names (`sweep-io`, `sweep-tasks`, …) — the `sweep.<short>` form is purely a
convenience namespace.
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.util import find_spec
from pathlib import Path


_LAZY_SUBMODULES = {
    "backend",
    "equations",
    "memory",
    "operators",
    "propagator",
    "receivers",
    "signal",
    "sources",
    "utils",
}


# Companion distributions exposed under the `sweep` namespace.
# Key   = the short name you write as `sweep.<key>` / `from sweep import <key>`
# Value = the installed distribution's import name (PyPI name with `-` -> `_`)
_COMPANION_ALIASES: dict[str, str] = {
    "io":      "sweep_io",
    "loss":    "sweep_loss",
    "nn":      "sweep_nn",
    "opt":     "sweep_opt",
    "preproc": "sweep_preproc",
    "runner":  "sweep_runner",
    "tasks":   "sweep_tasks",
    "viz":     "sweep_viz",
    "zoo":     "sweep_zoo",
}


def _extend_package_path_with_build_outputs() -> None:
    """Merge any `build/lib*/sweep` directory into `sweep.__path__`.

    Lets a `python setup.py build_ext --inplace`-style build show up to
    `import sweep._C` without a separate `pip install -e`.
    """
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[1]
    build_dir = repo_root / "build"

    if not build_dir.exists():
        return

    package_path = globals().get("__path__")
    if package_path is None:
        return

    for candidate in sorted(build_dir.glob("lib*/sweep")):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in package_path:
            package_path.append(candidate_str)


_extend_package_path_with_build_outputs()


def is_torch_binding_available() -> bool:
    """Return True when PyTorch and the compiled ``sweep._C`` binding are usable."""
    if find_spec("torch") is None:
        return False

    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        import_module("sweep._C")
    except Exception:
        return False

    return True


# ---------------------------------------------------------------------------
# PEP 562 lazy attribute access — handles:
#   import sweep; sweep.equations          (native lazy submodule)
#   import sweep; sweep.io.SEGYReader      (companion alias)
#   from sweep import runner               (companion alias)
# ---------------------------------------------------------------------------
def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _COMPANION_ALIASES:
        full = _COMPANION_ALIASES[name]
        try:
            module = import_module(full)
        except ImportError as e:
            raise AttributeError(
                f"`sweep.{name}` requires the `{full}` companion package "
                f"(install with `pip install {full.replace('_', '-')}`)."
            ) from e
        # Make `from sweep.<name> import X` also work after first access by
        # populating sys.modules under the alias.
        sys.modules[f"sweep.{name}"] = module
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES | set(_COMPANION_ALIASES))


# ---------------------------------------------------------------------------
# Meta-path finder — handles:
#   import sweep.io                         (resolves to sweep_io)
#   from sweep.io import SEGYReader         (ditto)
#   import sweep.io.prefetch                (ditto, transitively)
#
# Without this, only the PEP-562 attribute paths above work; `import sweep.io`
# would raise ModuleNotFoundError because there's no `sweep/io/` on disk.
# ---------------------------------------------------------------------------
class _CompanionFinder(MetaPathFinder):
    """Resolve `sweep.<short>` and its descendants to the installed companion."""

    _PREFIX = "sweep."

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if not fullname.startswith(self._PREFIX):
            return None
        rest = fullname[len(self._PREFIX):]
        head, _, tail = rest.partition(".")
        if head not in _COMPANION_ALIASES:
            return None
        full = _COMPANION_ALIASES[head]
        target_name = full if not tail else f"{full}.{tail}"
        try:
            return find_spec(target_name)
        except (ImportError, ValueError):
            return None


# Install once. The check makes a second `import sweep` (e.g. after reload)
# a no-op rather than registering duplicate finders.
if not any(isinstance(f, _CompanionFinder) for f in sys.meta_path):
    sys.meta_path.append(_CompanionFinder())


__all__ = [
    "is_torch_binding_available",
    *_LAZY_SUBMODULES,
    *_COMPANION_ALIASES,
]
