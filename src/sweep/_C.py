"""Lazy JIT entry point for sweep's compiled CUDA/C++ backend.

``import sweep._C`` is instant. The extension is compiled against your torch on
the **first attribute access** (i.e. the first real use of ``impl='c'``), then
cached — so ``is_torch_binding_available()`` / plain imports never trigger a
surprise ~3 min compile, and eager/JAX-only users never compile at all. Call
``sweep.precompile()`` to run that compile up front. See ``sweep/_jit.py``.
"""

from . import _jit

_ready = False


def _load():
    """Run the one-time JIT compile (cached) and expose the backend's functions
    on this module. Idempotent — used by both ``__getattr__`` (first use) and
    ``sweep.precompile()`` (up-front)."""
    global _ready
    if _ready:
        return
    mod = _jit.load()
    _ns = globals()
    for _k in dir(mod):
        if not _k.startswith("__"):
            _ns[_k] = getattr(mod, _k)
    _ready = True


def __getattr__(name):
    _load()                                    # compile-on-first-use (cached after)
    try:
        return globals()[name]
    except KeyError:
        raise AttributeError(f"module 'sweep._C' has no attribute {name!r}")
