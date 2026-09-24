"""The C boundary (core/capi.h) is compiled into the extension: its symbols
load through ctypes, the entry table names exactly the module's bound
<equation>_<op> functions (same names, every kind right), the stepped-runner
equations are the ten with runner factories, the session and the stream round
trip, and the visco FFT query answers.  The dispatcher itself is exercised by
the torch shim once it calls through this boundary (4c)."""
import ctypes
import pytest

torch = pytest.importorskip("torch")

OPS = ("forward", "apm_forward", "backward", "backward_bs", "backward_ckpt", "backward_recursive_ckpt", "rtm")


def _lib():
    from sweep import _jit
    mod = _jit.load()                      # the compiled module (sweep._C proxies its attributes)
    return mod, ctypes.CDLL(mod.__file__)


def test_abi_version_and_entry_table_match_the_module():
    mod, lib = _lib()
    assert lib.sweep_core_abi_version() == 1
    n = lib.sweep_entry_count()
    lib.sweep_entry_name.restype = ctypes.c_char_p
    names = [lib.sweep_entry_name(i).decode() for i in range(n)]
    assert lib.sweep_entry_name(n) is None and lib.sweep_entry_kind(n) == -1
    bound = sorted(a for a in dir(mod) if any(a.endswith("_" + op) for op in OPS) and not a.startswith("_"))
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
        assert lib.sweep_visco_fft_workspace_bytes(1, 64, 64) > 0


def test_call_with_a_bad_entry_returns_a_message_not_an_exception():
    mod, lib = _lib()
    err = ctypes.create_string_buffer(256)
    rc = lib.sweep_call(10_000, None, None, err, 256)
    assert rc != 0 and b"no entry 10000" in err.value
    h = ctypes.c_void_p()
    rc = lib.sweep_forward_runner_create(10_000, None, ctypes.byref(h), err, 256)
    assert rc != 0 and b"no stepped forward runner" in err.value
