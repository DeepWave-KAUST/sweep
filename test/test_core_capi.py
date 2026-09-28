"""The C boundary (core/capi.h) of the prebuilt ``libsweep_core.so``: its
symbols load through ctypes (``sweep._capi.core_lib()``, no compiled module
file exists on the default path any more), the entry table names exactly the
functions ``sweep._C``'s namespace binds (same names, every kind right), the
session and the stream round trip, and the visco FFT query answers.  The
dispatcher itself is exercised by the ctypes shim (test_capi_shim.py)."""
import ctypes
import pytest

torch = pytest.importorskip("torch")

OPS = ("forward", "apm_forward", "backward", "backward_bs", "backward_ckpt", "backward_recursive_ckpt", "rtm")


def _lib():
    """The core at hand, or a skip -- never a build.  Only a core that needs
    no compiling counts (the shipped one that fits, ``SWEEP_CORE``, or a cached
    local build whose source stamp is current); one that is there but cannot
    be loaded (the ABI guard: a core older than the shim's mirror, a missing
    cuFFT) skips with the reason rather than failing on the binary."""
    from sweep import _capi, _jit
    if _jit._shipped_core()[0] is None and _jit._cached_local_core() is None:
        pytest.skip(f"no core at hand without compiling: {_jit._shipped_core()[1]}")
    try:
        _jit.core_path()
        lib = _capi.core_lib()
    except (RuntimeError, OSError) as exc:
        pytest.skip(f"the core at hand cannot be loaded: {exc}")
    import sweep._C as mod                 # the pure-Python namespace over the core
    return mod, lib


def _bound(mod):
    """The ``<equation>_<op>`` names ``sweep._C`` exposes.  The namespace is
    filled on first attribute access, so touch one before listing."""
    getattr(mod, "acoustic2d_forward")
    return sorted(a for a in dir(mod) if any(a.endswith("_" + op) for op in OPS) and not a.startswith("_"))


def test_abi_version_and_entry_table_match_the_module():
    from sweep import _core_abi
    mod, lib = _lib()
    assert lib.sweep_core_abi_version() == _core_abi.ABI_VERSION
    n = lib.sweep_entry_count()
    lib.sweep_entry_name.restype = ctypes.c_char_p
    names = [lib.sweep_entry_name(i).decode() for i in range(n)]
    assert lib.sweep_entry_name(n) is None and lib.sweep_entry_kind(n) == -1
    bound = _bound(mod)
    assert sorted(names) == bound, set(names) ^ set(bound)
    for i, nm in enumerate(names):
        op = next(op for op in OPS if nm.endswith("_" + op))
        assert lib.sweep_entry_kind(i) == {"forward": 0, "apm_forward": 0, "rtm": 2}.get(op, 1), nm


def test_session_stream_and_fft_query_round_trip():
    mod, lib = _lib()
    lib.sweep_session_create.restype = ctypes.c_void_p
    lib.sweep_get_stream.restype = ctypes.c_void_p
    lib.sweep_visco_fft_workspace_bytes.restype = ctypes.c_size_t
    lib.sweep_visco_fft_workspace_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    s = lib.sweep_session_create()
    err = ctypes.create_string_buffer(256)
    assert lib.sweep_session_used(ctypes.c_void_p(s)) == 0
    assert lib.sweep_session_finish(ctypes.c_void_p(s), err, 256) == 0
    lib.sweep_session_destroy(ctypes.c_void_p(s))
    lib.sweep_set_stream(ctypes.c_void_p(0x1234))
    assert lib.sweep_get_stream() == 0x1234
    lib.sweep_set_stream(ctypes.c_void_p(0))
    assert (lib.sweep_get_stream() or 0) == 0
    if torch.cuda.is_available():
        # cuFFT's work area for a small 2-D C2C plan is a library decision:
        # cuFFT 11 (CUDA 12) asks for a few KB, cuFFT 12 (CUDA 13) answers 0
        # on an A100.  Both are legitimate; the Python side sizes the pool
        # slot as max(1, ceil(bytes / 4)).  The query must answer, not throw,
        # and answer the same twice.
        n = lib.sweep_visco_fft_workspace_bytes(1, 64, 64)
        assert n >= 0 and n == lib.sweep_visco_fft_workspace_bytes(1, 64, 64)


def test_call_with_a_bad_entry_returns_a_message_not_an_exception():
    mod, lib = _lib()
    err = ctypes.create_string_buffer(256)
    rc = lib.sweep_call(10_000, None, None, err, 256)
    assert rc != 0 and b"no entry 10000" in err.value
    h = ctypes.c_void_p()
    rc = lib.sweep_forward_runner_create(10_000, None, ctypes.byref(h), err, 256)
    assert rc != 0 and b"no stepped forward runner" in err.value
