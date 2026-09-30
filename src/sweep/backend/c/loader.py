"""Loading ``libsweep_core.so`` (csrc/core/capi.h) through ctypes.

One load per process: :func:`core_lib` resolves the core (:mod:`.jit`
``core_path``), seats cuFFT, dlopens it, declares every C signature
(:func:`_declare`) and runs the ABI guard (:func:`_check_abi`) that refuses a
core whose struct layout is not the generated mirror's (:mod:`.abi`).  The
loaded library is cached in the module attribute ``_lib``, which
``sweep.backend.torch.binding.is_compiled`` reads to answer "already loaded"
without loading anything.  Only :func:`core_lib` touches the filesystem; the
module imports without torch and without a GPU.
"""
from __future__ import annotations

import ctypes
import threading
from ctypes import POINTER, c_char_p, c_int, c_int64, c_size_t, c_void_p

from . import abi as _abi

_ERR_CAP = 4096          # the message buffer every fallible C call fills

_lib = None
# Guards the first load (core_lib) and the namespace build (sweep.backend.c
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
    sig("sweep_entry_count", [], c_int)
    sig("sweep_entry_name", [c_int], c_char_p)
    sig("sweep_entry_kind", [c_int], c_int)
    sig("sweep_call", [c_int, c_void_p, c_void_p, c_char_p, c_int], c_int)
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


def core_lib():
    """The loaded core (cached, one per process).

    Thread-safe: the first load (dlopen, ``_declare``, the ABI guard) runs
    under ``_load_lock`` and later callers get the cached ``ctypes.CDLL``,
    which is safe to share across threads -- ctypes releases the GIL around
    every core call, and the only per-thread state the core keeps (its launch
    stream, ``sweep_set_stream``) is thread-local on the C side.

    Opened ``RTLD_GLOBAL``.  That is not what makes cuFFT or the driver
    resolve: the core links cudart statically under ``-fvisibility=hidden``
    (``nm -D`` on the shipped core exports no ``cuda*`` symbol at all, and its
    NEEDED list is the cuFFT of its CUDA major -- ``libcufft.so.11`` for the
    cu12 core, ``libcufft.so.12`` for cu13 -- plus the system libraries, no
    libcudart, no libcuda), so cuFFT comes through NEEDED/RUNPATH (or the
    ``_preload_cufft`` seat) and the driver through cudart's own dlopen of
    ``libcuda.so.1``, both inside the core's own scope whichever mode it is
    opened with.  What ``RTLD_GLOBAL`` does is put the core's ``sweep_*``
    exports in the process-global lookup scope, so anything loaded after it
    that references them by name -- the ``SWEEP_JIT_FULL`` pybind shim
    (linked ``-lsweep_core``) in the same process, a ``ctypes.CDLL(None)``
    lookup, a profiler's hook -- binds to this one copy.
    """
    global _lib
    if _lib is not None:
        return _lib
    with _load_lock:
        if _lib is not None:
            return _lib
        from . import jit
        path = str(jit.core_path())
        # cuFFT: the core's RUNPATH reaches the pip nvidia-cufft wheel only
        # from an installed layout (a custom SWEEP_CORE, a moved
        # site-packages do not).  Seat the wheel's copy first (silent when
        # there is none); a toolkit is looked for only if the load then fails
        # on libcufft, so the common path never probes for nvcc.
        jit._preload_cufft(None)
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            if "libcufft" not in str(exc):
                raise
            jit._preload_cufft(jit._find_cuda_home())
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        _declare(lib, path)
        _check_abi(lib, path)
        _lib = lib
        return lib


def _raise_core(err, what: str, rc: int):
    """Raise the message a failed C call left in ``err`` (a
    ``create_string_buffer(_ERR_CAP)``)."""
    msg = err.value.decode("utf-8", "replace")
    raise RuntimeError(msg or f"{what} failed (rc={rc}) without a message")
