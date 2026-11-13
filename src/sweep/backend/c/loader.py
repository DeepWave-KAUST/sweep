"""Loading a core (``libsweep_core.so``, csrc/core/capi.h) through ctypes.

Two cores share one C API and one ctypes mirror: the CUDA core (nvcc,
``jit.core_path``) and the host core (the same sources built by g++,
``hostcore.core_path``).  :func:`core_lib` loads the one asked for by kind --
``"cuda"`` or ``"cpu"``, the ``device.type`` of the tensors a call carries --
at most once per process: resolve the path, seat cuFFT (CUDA only), dlopen,
declare every C signature (:func:`_declare`), run the ABI guard
(:func:`_check_abi`) that refuses a core whose struct layout is not the
generated mirror's (:mod:`.abi`), and ask the core which device it is
(:func:`_check_device`) rather than trusting the path.  The loaded libraries
are cached in ``_libs``; ``_lib`` keeps naming the first one loaded for the
callers that only ask "is anything loaded"
(``sweep.backend.torch.binding.is_compiled``).  Only :func:`core_lib` touches
the filesystem; the module imports without torch and without a GPU.
"""
from __future__ import annotations

import ctypes
import threading
from ctypes import POINTER, c_char_p, c_int, c_int64, c_size_t, c_void_p

from . import abi as _abi

_ERR_CAP = 4096          # the message buffer every fallible C call fills

KINDS = ("cuda", "cpu")

_libs: dict = {}
_lib = None              # the first core loaded (what is_compiled() reads)
# Guards the loads (core_lib) and the namespace build (sweep.backend.c
# namespace): both happen on a propagator's first use, which several threads
# may reach at once (per-GPU worker threads, a DataLoader worker).  Re-entrant
# because namespace() holds it while it calls core_lib().
_load_lock = threading.RLock()


def _declare(lib, path: str = "<core>") -> None:
    """argtypes/restypes for every C function, so ctypes never guesses a width
    (a size_t/int64 return read as int truncates silently on the wrong guess).
    A core missing one of them predates this layer's ABI: refused by name."""
    def sig(name, argtypes, restype):
        try:
            f = getattr(lib, name)
        except AttributeError:
            raise RuntimeError(
                f"{path} has no {name}: the core predates this layer's C API "
                f"(ABI {_abi.ABI_VERSION}); rebuild it from this tree") from None
        f.argtypes = argtypes
        f.restype = restype

    sig("sweep_core_abi_version", [], c_int)
    sig("sweep_core_device", [], c_char_p)
    sig("sweep_entry_count", [], c_int)
    sig("sweep_entry_name", [c_int], c_char_p)
    sig("sweep_entry_kind", [c_int], c_int)
    sig("sweep_entry_available", [c_int], c_int)
    sig("sweep_call", [c_int, c_void_p, c_void_p, c_char_p, c_int], c_int)
    sig("sweep_sem_call", [c_int, c_void_p, c_int64, c_char_p, c_int], c_int)
    sig("sweep_entry_struct", [c_int], c_char_p)
    for side in ("forward", "backward"):
        sig(f"sweep_{side}_runner_create", [c_int, c_void_p, POINTER(c_void_p), c_char_p, c_int], c_int)
        sig(f"sweep_{side}_runner_run", [c_void_p, c_int, c_int, c_int, c_void_p, c_char_p, c_int], c_int)
        sig(f"sweep_{side}_runner_device_index", [c_void_p], c_int)
        sig(f"sweep_{side}_runner_destroy", [c_void_p], None)
    sig("sweep_session_create", [], c_void_p)
    sig("sweep_session_finish", [c_void_p, c_char_p, c_int], c_int)
    sig("sweep_session_used", [c_void_p], c_int)
    sig("sweep_session_destroy", [c_void_p], None)
    sig("sweep_set_stream", [c_void_p], None)
    sig("sweep_get_stream", [], c_void_p)
    sig("sweep_visco_fft_workspace_bytes", [c_int64, c_int64, c_int64], c_size_t)
    sig("sweep_layout_count", [], c_int64)
    sig("sweep_layout_probe", [c_int64], c_int64)


def _probe_name(i: int) -> str:
    """What LAYOUT_PROBE[i] measures, for an error message."""
    name, member = _abi.LAYOUT_PROBE[i]
    if name in getattr(_abi, "ENUMS", {}):
        return f"enumerator {name}::{member}"
    if member is None:
        return f"sizeof({name})"
    return f"offsetof({name}, {member})"


def _check_abi(lib, path: str) -> None:
    """The ABI guard.  The version number catches a deliberate break; the
    layout probe catches the silent one -- a field added to a core struct
    without regenerating the mirror would otherwise shift every later field
    and the core would read garbage from a struct that looked right, and a
    reordered BoundaryDtype would mis-tag every boundary buffer.  The mirror's
    side of every probe entry is ``abi.ctypes_layout()``, so an entry kind
    this module never heard of (the enum probes) still compares."""
    got = lib.sweep_core_abi_version()
    if got != _abi.ABI_VERSION:
        raise RuntimeError(
            f"sweep core ABI {got} != the shim's ABI {_abi.ABI_VERSION} ({path}); "
            "the core and sweep.backend.c.abi were built from different trees")
    probe = _abi.LAYOUT_PROBE
    expect = list(_abi.ctypes_layout())
    n = int(lib.sweep_layout_count())
    if n != len(probe) or len(expect) != len(probe):
        raise RuntimeError(
            f"sweep core layout probe has {n} entries, the shim's mirror has "
            f"{len(probe)} (ctypes_layout answers {len(expect)}) ({path}); regenerate "
            "sweep.backend.c.abi or rebuild the core from the same tree")
    for i in range(len(probe)):
        val = int(lib.sweep_layout_probe(i))
        if val != expect[i]:
            name, member = probe[i]
            raise RuntimeError(
                f"sweep core/shim layout mismatch at probe {i} ({name}, {member}): "
                f"{_probe_name(i)} expected {expect[i]}, got {val} ({path}); regenerate "
                "the mirror or rebuild the core from the same tree")


def _check_device(lib, kind: str, path: str) -> None:
    """The core says which device it computes on; a mismatch means a core was
    filed in the wrong drawer (or SWEEP_CORE points at the other one)."""
    got = (lib.sweep_core_device() or b"").decode()
    if got != kind:
        raise RuntimeError(f"{path} is a {got!r} core, loaded as the {kind!r} core")


def _entry_rows(lib):
    return [(lib.sweep_entry_name(i), lib.sweep_entry_kind(i)) for i in range(lib.sweep_entry_count())]


def _check_same_table(a, b, path: str) -> None:
    """The two cores are built from one capi.h, so their entry tables are one
    table; the namespace assigns ids once and uses them against either library.
    A mismatch means the cores come from different trees."""
    if _entry_rows(a) != _entry_rows(b):
        raise RuntimeError(f"{path}: its entry table differs from the core already loaded; "
                           "the two cores were built from different trees")


def default_kind() -> str:
    """The core a device-less call (the entry table, a session with no
    device) is served by: the CUDA core when torch can see a GPU, else the
    host core."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def kind_for_device(device) -> str:
    """``"cuda"`` / ``"cpu"`` for a torch device (or device type string); any
    other device type is refused -- no core serves it."""
    dtype = getattr(device, "type", None) or str(device).split(":", 1)[0]
    if dtype not in KINDS:
        raise RuntimeError(f"no compiled core serves tensors on a {device!r} device "
                           "(the cores run on CUDA and on the host)")
    return dtype


def _resolve(kind: str) -> str:
    if kind == "cuda":
        from . import jit
        return str(jit.core_path())
    from . import hostcore
    return str(hostcore.core_path())


def _dlopen(kind: str, path: str):
    """CUDA: RTLD_GLOBAL, so the SWEEP_JIT_FULL pybind shim (linked
    -lsweep_core) and a ``ctypes.CDLL(None)`` lookup bind to this copy.
    Host: RTLD_LOCAL -- it exports the same ``sweep_*`` names, and a second
    global copy would shadow the first for every by-name resolver; ctypes
    itself resolves through the handle, so the local scope costs nothing."""
    if kind == "cuda":
        from . import jit
        # cuFFT: the core's RUNPATH reaches the pip nvidia-cufft wheel only
        # from an installed layout (a custom SWEEP_CORE, a moved
        # site-packages do not).  Seat the wheel's copy first (silent when
        # there is none); a toolkit is looked for only if the load then fails
        # on libcufft, so the common path never probes for nvcc.
        jit._preload_cufft(None)
        try:
            return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            if "libcufft" not in str(exc):
                raise
            jit._preload_cufft(jit._find_cuda_home())
            return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)


def core_lib(kind: str = "cuda"):
    """The loaded core of ``kind`` (cached, one per process and kind).

    Thread-safe: the first load (dlopen, ``_declare``, the ABI guard) runs
    under ``_load_lock`` and later callers get the cached ``ctypes.CDLL``,
    which is safe to share across threads -- ctypes releases the GIL around
    every core call, and the only per-thread state the core keeps (its launch
    stream, ``sweep_set_stream``) is thread-local on the C side.
    """
    global _lib
    if kind not in KINDS:
        raise ValueError(f"core kind must be one of {KINDS}, got {kind!r}")
    lib = _libs.get(kind)
    if lib is not None:
        return lib
    with _load_lock:
        lib = _libs.get(kind)
        if lib is not None:
            return lib
        path = _resolve(kind)
        lib = _dlopen(kind, path)
        _declare(lib, path)
        _check_abi(lib, path)
        _check_device(lib, kind, path)
        for other in _libs.values():
            _check_same_table(other, lib, path)
        _libs[kind] = lib
        if _lib is None:
            _lib = lib
        return lib


def core_lib_for(device):
    """The core serving tensors on ``device`` (a torch device)."""
    return core_lib(kind_for_device(device))


def _ready_without_build(kind: str) -> bool:
    """A core of ``kind`` is at hand as a file (shipped, or cached and current):
    loading it compiles nothing."""
    try:
        if kind == "cuda":
            from . import jit
            return jit._shipped_core()[0] is not None or jit._cached_local_core() is not None
        from . import hostcore
        return hostcore.shipped_core()[0] is not None or hostcore.cached_local_core() is not None
    except Exception:
        return False


def any_core_lib():
    """Whichever core loads: what the entry table is read from when no tensor
    names a device.  Both cores export one table, so the choice only decides
    what gets loaded (or built) first -- a core already loaded, then one that
    is a file on disk (no build), then the default kind.  Without this a CPU
    run on a GPU box with nvcc would build the CUDA core it never calls."""
    first = default_kind()
    by_default = (first,) + tuple(k for k in KINDS if k != first)
    order = list(loaded_kinds())
    order += [k for k in by_default if k not in order and _ready_without_build(k)]
    order += [k for k in by_default if k not in order]
    errors = []
    for kind in order:
        try:
            return core_lib(kind)
        except (RuntimeError, OSError) as exc:
            errors.append(f"{kind}: {exc}")
    raise RuntimeError("no compiled core can be loaded -- " + "; ".join(errors))


def loaded_kinds() -> tuple:
    return tuple(k for k in KINDS if k in _libs)


def _raise_core(err, what: str, rc: int):
    """Raise the message a failed C call left in ``err`` (a
    ``create_string_buffer(_ERR_CAP)``)."""
    msg = err.value.decode("utf-8", "replace")
    raise RuntimeError(msg or f"{what} failed (rc={rc}) without a message")
