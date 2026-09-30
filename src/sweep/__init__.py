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

try:  # distribution name differs from the import name (`sweep`)
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    __version__ = _pkg_version("sweep-solver")
except (ImportError, PackageNotFoundError):  # a source tree with nothing installed
    __version__ = "unknown"

import sys
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import EXTENSION_SUFFIXES
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


def _is_extension_origin(origin: str | None) -> bool:
    """True when a module spec's origin is a compiled extension rather than a
    ``.py`` file — i.e. ``sweep._C`` is a real binary, not the JIT shim."""
    return bool(origin) and origin.endswith(tuple(EXTENSION_SUFFIXES))


def _prebuilt_binding_present() -> bool:
    """True when a compiled ``sweep._C`` extension already exists on disk.

    Two ways to get one: a wheel built with ``SWEEP_BUILD_CUDA=1``, or
    ``setup.py build_ext --inplace`` (whose output `_extend_package_path_with_
    build_outputs` above already puts on ``sweep.__path__``). Either way the
    kernels are compiled, so no CUDA toolkit is needed at run time.

    Resolves the spec without importing, so this never compiles anything.
    """
    try:
        spec = find_spec("sweep._C")
    except Exception:
        return False
    return spec is not None and _is_extension_origin(spec.origin)


def is_torch_binding_available() -> bool:
    """Return True when ``sweep._C`` can be used for ``impl='c'`` — either a
    compiled extension is already on disk, or PyTorch + a CUDA GPU are present
    and the CUDA core is at hand: the prebuilt one the wheel ships for your
    torch's CUDA major and GPU (nothing compiles on first use; the shim is
    ctypes), else an nvcc to build one.  Does NOT load or build anything (see
    ``sweep.backend.c.jit``)."""
    if find_spec("torch") is None:
        return False
    # A pre-built extension answers this on its own: the kernels exist, and
    # asking `can_build()` would demand a device the check does not need.
    # Checking this first is what keeps a `SWEEP_BUILD_CUDA=1` install from
    # silently falling back to eager on a node with no CUDA toolkit loaded.
    if _prebuilt_binding_present():
        return True
    try:
        from sweep.backend.c import jit
        # A shipped core that fits needs no nvcc -- can_build() knows, its
        # can_compile() stops at the shipped core.  What it still needs is a
        # device: the compiled backend serves the CUDA core only, so without
        # one 'auto' must resolve to eager.
        return jit.can_build()[0]
    except Exception:
        return False


def precompile(require_gpu: bool = True) -> bool:
    """Make sure a CUDA core for ``sweep._C`` exists now.

    Nothing compiles on first use: the shim is ctypes and the wheel ships the
    core prebuilt (one per CUDA major, ``lib/cu12/`` and ``lib/cu13/``, each a
    fat binary over the common archs), so with a fitting core this is a no-op —
    e.g. right after ``pip install``.  When no
    shipped core fits -- a torch built for another CUDA major, a GPU outside
    the shipped archs and older than the shipped PTX, an sdist/clone install --
    the core is built here once with nvcc (~2-5 min) and cached, and that is
    the surprise this call moves up front.  Raises a clear error if PyTorch, a
    CUDA GPU, or (for that build) a suitable ``nvcc`` (torch's CUDA major; >= 12.4
    for CUDA 12, >= 12.8 for a Blackwell target, or a CUDA 13 nvcc) is missing::

        python -c "import sweep; sweep.precompile()"

    ``require_gpu=False`` builds **without a visible device**, for CI and for
    warming the cache from a CPU allocation that a later GPU job reuses; when
    the core has to be built, ``TORCH_CUDA_ARCH_LIST`` must then name the
    target architecture::

        TORCH_CUDA_ARCH_LIST=7.0 python -m sweep.build --no-gpu-required

    Compiling needs a core and a target arch, not a card — gating it on a
    device forces every build to occupy a GPU it does not use.
    ``SWEEP_JIT_FULL=1`` (the compiled developer path) also compiles the pybind
    shim here, against your torch, as it always did.

    On the default path the core is also loaded into this process (a dlopen
    and the ABI guard, no CUDA call), so
    ``sweep.backend.torch.binding.is_compiled()`` answers True afterwards.
    """
    import sweep._C as _C

    loader = getattr(_C, "_load", None)
    if loader is None:
        # A prebuilt extension shadows _C.py: the import machinery tries the
        # .so suffixes before .py, so `sweep._C` IS the compiled module and
        # importing it has already loaded it. It has no _load to call, and
        # calling one would be an AttributeError on exactly that install
        # (SWEEP_BUILD_CUDA=1 pip install, the documented developer path).
        return True
    from sweep.backend.c import jit
    if jit.jit_full():
        loader(compile_only=not require_gpu)
        return True
    # The ctypes shim needs only the core.  The device check is what
    # require_gpu relaxes: a local build then targets TORCH_CUDA_ARCH_LIST.
    ok, why = jit.can_compile() if not require_gpu else jit.can_build()
    if not ok:
        raise RuntimeError(
            f"sweep's compiled backend (impl='c') is unavailable: {why}. "
            "Use impl='eager' for a pure-Python (slower) CPU/GPU path.")
    jit.core_path()
    # Load it too: is_compiled() reads the loader's cache
    # (sweep.backend.c.loader._lib), and a core that fits on paper but fails
    # the ABI guard is better found now.
    from sweep.backend.c import loader as core_loader
    core_loader.core_lib()
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
