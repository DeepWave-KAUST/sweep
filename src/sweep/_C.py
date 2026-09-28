"""Lazy entry point for sweep's CUDA backend (``impl='c'``).

``import sweep._C`` is instant.  The **first attribute access** (the first real
use of ``impl='c'``) loads the prebuilt core through ctypes
(``sweep.backend.c``; no compile) and exposes its functions here, so
``is_torch_binding_available()`` / plain imports never do any work and
eager/JAX-only users never touch the core at all.  ``SWEEP_JIT_FULL=1``
selects the compiled developer path instead: the pybind shim, compiled against
your torch on first use and cached (see ``sweep/backend/c/jit.py``), where the
CPU engine lives.  ``sweep.precompile()`` does the one-time part up front.
"""

from .backend.c import jit as _jit

_ready = False


def _load(compile_only: bool = False):
    """Expose the backend's functions on this module.  Idempotent -- used by
    both ``__getattr__`` (first use) and ``sweep.precompile()`` (up-front).
    Default: the ctypes layer's namespace over the core ``jit.core_path()``
    resolves (``compile_only`` is moot: nothing compiles).  ``SWEEP_JIT_FULL``:
    the compiled module, ``compile_only`` warming its cache on a machine with
    no GPU (see ``jit.can_compile``)."""
    global _ready
    if _ready:
        return
    _ns = globals()
    if _jit.jit_full():
        mod = _jit.load(compile_only=compile_only)
        for _k in dir(mod):
            if not _k.startswith("__"):
                _ns[_k] = getattr(mod, _k)
    else:
        from .backend import c as _backend_c
        _ns.update(_backend_c.namespace())
    _ready = True


def __getattr__(name):
    _load()                                    # load-on-first-use (cached after)
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'sweep._C' has no attribute {name!r}")
