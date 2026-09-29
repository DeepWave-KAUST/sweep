"""The pure-Python ctypes layer (``sweep.backend.c``: ``loader`` / ``adapt`` /
``entries`` / ``runners`` over the generated ``abi`` mirror) that replaces the
compiled pybind shim: after ``pip install`` nothing compiles, the prebuilt
``libsweep_core.so`` is dlopen'd and every torch tensor is described to it
through the generated ctypes mirrors.

What is pinned here, and why each check exists:

* the ABI guard -- the mirror is generated from the headers, the core exports
  sizeof/offsetof of every struct it was compiled with (and the BoundaryDtype
  enumerator values); the two must agree entry by entry or a struct is read
  at the wrong offsets (silently, on a GPU).  Pinned against a fake core, so
  a version or layout mismatch is proven to be refused by name whatever
  binary is on the box;
* the loader -- cuFFT is seated from the pip wheel first and a toolkit is
  looked for only when the load fails on libcufft; one load per process
  under a lock;
* ``adapt()`` -- the one place a tensor becomes a ``Buf`` (the twin of
  ``cuda/common/buf_torch.h``'s ``buf_of``): pointer, sizes/strides verbatim
  (a DD cut face has a nonzero stride on a 0-size dim), the storage tag, the
  host copies of the three tensors the drivers read on the host, and the
  keepalive that makes those copies outlive the call;
* input validation -- a plain list for a host int field works, a single
  tensor where a list is due is refused by field name, CPU models are refused
  before the core dereferences them;
* the boundary session -- close() destroys the handle the core handed out,
  once; a closed session raises rather than passing NULL;
* the entries and the runners return nothing -- the drivers allocate no
  output, so what the core wrote is in the tensors the caller bound on the
  params object, and a failed call is the core's message, not a crash;
* the error path -- a bad entry / a bad runner is a message, not a crash;
* the stream scope and the runner round trip (GPU): the core launches on
  torch's current stream and puts it back; the persistent runner is the same
  code as the monolithic entry, so the two records are bit-identical.

Nothing here compiles (an autouse fixture makes a compile attempt a failure);
a test that needs the real core skips when none can be had without compiling
(``_core_or_skip``).  Python 3.9-compatible on purpose: this is what the
wheel's floor runs.
"""
from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import requires_binding  # noqa: E402

# The core struct ``adapt()`` is asked for, by the key ``FIELDS`` uses.
FWD = "ForwardInputCore"
BWD = "BackwardInputCore"


def _jit_full():
    """The compiled pybind shim (SWEEP_JIT_FULL=1) replaces this ctypes layer:
    tests of ctypes-only behaviour have nothing to test there."""
    try:
        from sweep.backend.c import jit
        return jit.jit_full()
    except Exception:
        return False


_ctypes_only = pytest.mark.skipif(_jit_full(), reason="SWEEP_JIT_FULL=1 routes sweep._C "
                                  "through the pybind shim; this tests the ctypes layer")


def _mods():
    """``adapt()`` and the generated mirror, imported lazily so a missing one
    fails the test that needs it (a red, which is wanted) instead of erroring
    the module.  The loader, the entries and the runners are imported the same
    way, in the tests that reach them."""
    from sweep.backend.c import abi
    from sweep.backend.c.adapt import adapt
    return adapt, abi


class _CompileRefused(AssertionError):
    """Raised in place of a build by the ``_no_compile`` fixture."""


# --------------------------------------------------------------------------- #
# nothing here may compile
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _no_compile(monkeypatch):
    """The point of the shim: ``pip install`` then use, no toolchain.  Any
    reach for torch's JIT builder or the core build from these tests is a bug
    in the loader, not a slow test -- unless the caller opted into the old
    compiled path with SWEEP_JIT_FULL=1, which these tests do not cover.

    A locally built core that is already cached with nothing to rebuild is not
    a compile: ``jit._build_core`` is let through when ninja has no work (so
    ``core_path()`` can resolve it) and refused otherwise."""
    if os.environ.get("SWEEP_JIT_FULL", "").lower() in ("1", "true", "yes", "on"):
        yield
        return
    from torch.utils import cpp_extension
    from sweep.backend.c import jit

    def refuse(*_a, **_k):
        raise _CompileRefused("a compile was attempted; the ctypes shim must not build anything")

    monkeypatch.setattr(cpp_extension, "load", refuse)
    monkeypatch.setattr(cpp_extension, "load_inline", refuse)
    if hasattr(jit, "_build_core"):
        real_build_core = jit._build_core
        will_build = getattr(jit, "_core_will_build", None)

        def build_core(build_dir, *a, **k):
            if will_build is None or will_build(Path(build_dir)):
                refuse()
            return real_build_core(build_dir, *a, **k)

        monkeypatch.setattr(jit, "_build_core", build_core)
    yield


def _core_or_skip():
    """The loaded core, or a pytest skip.

    These tests never compile, so a core is at hand only when the shipped one
    fits this torch/GPU (``jit._shipped_core``) or a locally built one is
    cached with nothing left to rebuild; ``jit.core_path()`` resolves both
    and is the one call that would otherwise build -- which ``_no_compile``
    turns into ``_CompileRefused``, here a skip.  A core that IS there but is
    refused by the ABI/layout guard is a red, not a skip: that is the guard
    doing its job on a stale binary."""
    from sweep.backend.c import jit, loader
    if loader._lib is not None:
        return loader._lib
    so, why = jit._shipped_core()
    if so is None:
        try:
            jit.core_path()
        except (RuntimeError, OSError, _CompileRefused) as exc:
            pytest.skip(f"no CUDA core without compiling: {why}; {exc}")
    return loader.core_lib()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ptr_addr(p):
    """The address a ctypes pointer / array / byref points at (None for NULL)."""
    if p is None:
        return None
    if isinstance(p, ctypes.c_void_p):
        return p.value
    if isinstance(p, int):
        return p
    return ctypes.cast(p, ctypes.c_void_p).value


def _as_int(x):
    return x.value if isinstance(x, ctypes.c_void_p) else int(x)


def _addr(obj):
    """The address of the memory an object owns, whatever the shim chose to
    keep the copies in (a CPU tensor, a numpy array, a ctypes array)."""
    if isinstance(obj, torch.Tensor):
        return obj.data_ptr()
    if isinstance(obj, np.ndarray):
        return int(obj.ctypes.data)
    if isinstance(obj, (ctypes.Array, ctypes.Structure)):
        return ctypes.addressof(obj)
    if isinstance(obj, ctypes._Pointer):
        return _ptr_addr(obj)
    return None


def _addresses(keepalive):
    assert isinstance(keepalive, (list, tuple)), \
        f"adapt() must hand back a keepalive list, got {type(keepalive).__name__}"
    addrs = set()
    stack = list(keepalive)
    while stack:
        obj = stack.pop()
        if isinstance(obj, (list, tuple)):
            stack.extend(obj)
            continue
        a = _addr(obj)
        if a is not None:
            addrs.add(a)
    return addrs


def _sizes(buf):
    return tuple(int(x) for x in list(buf.sizes_)[: int(buf.ndim_)])


def _strides(buf):
    return tuple(int(x) for x in list(buf.strides_)[: int(buf.ndim_)])


def _span_values(span):
    return [span.p[i] for i in range(int(span.n))]


def _buf_values(buf):
    return list((ctypes.c_int32 * int(buf.numel_)).from_address(buf.data_)) if buf.numel_ else []


class _FakeFn:
    """A ctypes-function stand-in: ``_declare`` stamps argtypes/restype on it,
    and it runs ``fn`` when called (or fails, for a name no test expects)."""

    def __init__(self, fn=None):
        self.fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *a, **k):
        if self.fn is None:
            raise AssertionError("a fake core function nobody expected was called")
        return self.fn(*a, **k)


class _FakeCore:
    """Just enough of libsweep_core for the ABI guard and the loader: the
    version, the layout probe, and any other name ``_declare`` asks for."""

    def __init__(self, abi_version, layout):
        self.abi = int(abi_version)
        self.layout = list(layout)
        self.probed = []
        self.sweep_core_abi_version = _FakeFn(lambda: self.abi)
        self.sweep_layout_count = _FakeFn(lambda: len(self.layout))
        self.sweep_layout_probe = _FakeFn(self._probe)

    def _probe(self, i):
        i = int(i)
        self.probed.append(i)
        return self.layout[i] if 0 <= i < len(self.layout) else -1

    def __getattr__(self, name):
        if name.startswith("sweep_"):
            f = _FakeFn()
            setattr(self, name, f)
            return f
        raise AttributeError(name)


class _FakeSessionLib:
    """The three session calls, recording the handle object each got."""

    def __init__(self):
        self.destroyed = []
        self.finished = []
        self.used_with = []

    def sweep_session_finish(self, h, err, cap):
        self.finished.append(h)
        return 0

    def sweep_session_used(self, h):
        self.used_with.append(h)
        return 1

    def sweep_session_destroy(self, h):
        self.destroyed.append(h)


def _fake_session(lib, addr=0x1234):
    from sweep.backend.c.runners import BoundarySession
    s = BoundarySession.__new__(BoundarySession)
    s._lib = lib
    s.handle = ctypes.c_void_p(addr)
    return s


# --------------------------------------------------------------------------- #
# 1. ABI guard
# --------------------------------------------------------------------------- #
class TestAbiGuard:
    def test_core_loads_and_the_abi_version_matches(self):
        from sweep.backend.c import loader
        _, abi = _mods()
        lib = _core_or_skip()
        assert isinstance(lib, ctypes.CDLL)
        assert lib.sweep_core_abi_version() == abi.ABI_VERSION
        assert loader.core_lib() is lib, "core_lib() must hand back the one loaded library"

    def test_every_layout_probe_entry_matches_the_mirror(self):
        _, abi = _mods()
        lib = _core_or_skip()
        lib.sweep_layout_count.restype = ctypes.c_int64
        lib.sweep_layout_probe.restype = ctypes.c_int64
        lib.sweep_layout_probe.argtypes = [ctypes.c_int64]
        probe = list(abi.LAYOUT_PROBE)
        want = list(abi.ctypes_layout())
        assert probe and len(want) == len(probe)
        assert lib.sweep_layout_count() == len(probe)
        bad = []
        for i, ((sname, field), expect) in enumerate(zip(probe, want)):
            got = lib.sweep_layout_probe(i)
            if got != expect:
                bad.append(f"{i}: {sname}.{field or 'sizeof'}: core {got} != mirror {expect}")
        assert not bad, "\n".join(bad)
        assert lib.sweep_layout_probe(len(probe)) == -1

    def test_fields_table_names_the_mirror_fields_in_order(self):
        _, abi = _mods()
        for key in (FWD, BWD):
            table = [name for name, _kind, _default in abi.FIELDS[key]]
            mirror = [name for name, _t in getattr(abi, key)._fields_]
            assert table == mirror, key

    def test_constants(self):
        _, abi = _mods()
        assert (abi.FP32, abi.FP16, abi.BF16, abi.INT8) == (0, 1, 2, 3)
        assert abi.HOST_INT_SPAN == {"source_field_indices", "receiver_field_indices"}
        assert abi.HOST_BUF == {"checkpoint_steps"}
        assert abi.ENUMS["BoundaryDtype"] == {"FP32": 0, "FP16": 1, "BF16": 2, "INT8": 3}

    def test_mirror_answers_every_probe_entry_and_the_enum_probes_trail(self):
        """The mirror's side of the probe: one answer per LAYOUT_PROBE entry,
        the BoundaryDtype enumerators last (what sizeof/offsetof cannot see)."""
        _, abi = _mods()
        probe = list(abi.LAYOUT_PROBE)
        want = list(abi.ctypes_layout())
        assert len(want) == len(probe)
        tail = [(s, f) for s, f in probe if s in abi.ENUMS]
        assert tail == [("BoundaryDtype", "FP32"), ("BoundaryDtype", "FP16"),
                        ("BoundaryDtype", "BF16"), ("BoundaryDtype", "INT8")]
        assert probe[-4:] == tail and want[-4:] == [0, 1, 2, 3]
        assert want[0] == ctypes.sizeof(abi.Buf) and probe[0] == ("Buf", None)

    # -- the guard against a fake core: what the binary cannot prove --------
    def test_a_matching_core_passes_and_every_entry_is_probed(self):
        from sweep.backend.c import loader
        _, abi = _mods()
        fake = _FakeCore(abi.ABI_VERSION, abi.ctypes_layout())
        loader._check_abi(fake, "<fake>")
        assert fake.probed == list(range(len(abi.LAYOUT_PROBE))), \
            "the guard must compare every LAYOUT_PROBE entry, the enum ones included"

    def test_abi_version_mismatch_names_both_numbers(self, monkeypatch):
        from sweep.backend.c import loader
        _, abi = _mods()
        monkeypatch.setattr(abi, "ABI_VERSION", 7)
        fake = _FakeCore(2, abi.ctypes_layout())
        with pytest.raises(RuntimeError, match=r"core ABI 2 .*shim's ABI 7") as ei:
            loader._check_abi(fake, "<fake>")
        assert "<fake>" in str(ei.value)
        assert fake.probed == [], "a version mismatch is refused before any layout probe"

    def test_probe_count_mismatch_names_both_counts(self):
        from sweep.backend.c import loader
        _, abi = _mods()
        want = list(abi.ctypes_layout())
        fake = _FakeCore(abi.ABI_VERSION, want[:-4])      # a core built before the enum probes
        with pytest.raises(RuntimeError, match=rf"{len(want) - 4} entries.*{len(want)}"):
            loader._check_abi(fake, "<fake>")

    @pytest.mark.parametrize("which", ["first", "middle", "last"])
    def test_layout_mismatch_names_the_entry_expected_and_got(self, which):
        from sweep.backend.c import loader
        _, abi = _mods()
        probe = list(abi.LAYOUT_PROBE)
        want = list(abi.ctypes_layout())
        i = {"first": 0, "middle": probe.index(("ForwardInputCore", "boundary_session")),
             "last": len(probe) - 1}[which]
        bad = list(want)
        bad[i] = want[i] + 8
        fake = _FakeCore(abi.ABI_VERSION, bad)
        name, member = probe[i]
        with pytest.raises(RuntimeError) as ei:
            loader._check_abi(fake, "<fake>")
        msg = str(ei.value)
        assert f"probe {i}" in msg
        assert name in msg and str(member) in msg
        assert f"expected {want[i]}" in msg and f"got {want[i] + 8}" in msg
        assert fake.probed == list(range(i + 1)), "the FIRST mismatch is reported"


# --------------------------------------------------------------------------- #
# 2. the loader
# --------------------------------------------------------------------------- #
class TestLoader:
    def test_load_lock_is_reentrant(self):
        from sweep.backend.c import loader
        assert isinstance(loader._load_lock, type(threading.RLock()))

    def _fake_loader(self, monkeypatch, cdll):
        """``(loader, abi, calls)`` with the core path, the cuFFT seat and the
        toolkit probe faked (``calls`` records the last two) and no core cached."""
        from sweep.backend.c import jit, loader
        _, abi = _mods()
        calls = []
        monkeypatch.setattr(loader, "_lib", None)
        monkeypatch.setattr(jit, "core_path", lambda: Path("/nowhere/libsweep_core.so"))
        monkeypatch.setattr(jit, "_preload_cufft", lambda home: calls.append(("preload", home)))
        monkeypatch.setattr(jit, "_find_cuda_home", lambda: calls.append(("find_cuda_home",)) or "/fake/cuda")
        monkeypatch.setattr(ctypes, "CDLL", cdll)
        return loader, abi, calls

    def test_cufft_is_seated_from_the_wheel_first_and_a_toolkit_only_on_a_libcufft_error(self, monkeypatch):
        attempts = []
        fake = None

        def cdll(path, mode=None, **k):
            attempts.append((str(path), mode))
            if len(attempts) == 1:
                raise OSError("libcufft.so.11: cannot open shared object file: No such file or directory")
            return fake

        loader, abi, calls = self._fake_loader(monkeypatch, cdll)
        fake = _FakeCore(abi.ABI_VERSION, abi.ctypes_layout())
        lib = loader.core_lib()
        assert lib is fake and loader.core_lib() is fake
        assert calls == [("preload", None), ("find_cuda_home",), ("preload", "/fake/cuda")], \
            "the wheel seat first (no toolkit probe), the toolkit only after the libcufft failure"
        assert [m for _, m in attempts] == [ctypes.RTLD_GLOBAL, ctypes.RTLD_GLOBAL]
        assert all(p.endswith("libsweep_core.so") for p, _ in attempts)

    def test_a_load_error_that_is_not_cufft_is_not_retried(self, monkeypatch):
        attempts = []

        def cdll(path, mode=None, **k):
            attempts.append(str(path))
            raise OSError("/nowhere/libsweep_core.so: cannot open shared object file")

        loader, abi, calls = self._fake_loader(monkeypatch, cdll)
        with pytest.raises(OSError, match="cannot open"):
            loader.core_lib()
        assert calls == [("preload", None)], "no toolkit probe for an unrelated load error"
        assert len(attempts) == 1
        assert loader._lib is None, "a failed load leaves nothing cached"

    def test_a_refused_core_is_not_cached(self, monkeypatch):
        loader, abi, calls = self._fake_loader(
            monkeypatch, lambda path, mode=None, **k: _FakeCore(abi.ABI_VERSION + 1, abi.ctypes_layout()))
        with pytest.raises(RuntimeError, match="ABI"):
            loader.core_lib()
        assert loader._lib is None

    def test_concurrent_first_loads_share_one_library(self, monkeypatch):
        from sweep.backend.c import loader
        _core_or_skip()
        real_cdll = ctypes.CDLL
        loads = []

        def counting(path, mode=None, **k):
            if str(path).endswith("libsweep_core.so"):
                loads.append(threading.get_ident())
            return real_cdll(path, mode=mode, **k) if mode is not None else real_cdll(path, **k)

        monkeypatch.setattr(ctypes, "CDLL", counting)
        monkeypatch.setattr(loader, "_lib", None)
        got, errors = [], []
        gate = threading.Barrier(8)

        def worker():
            try:
                gate.wait()
                got.append(loader.core_lib())
            except Exception as exc:      # noqa: BLE001 - reported below
                errors.append(exc)

        ts = [threading.Thread(target=worker) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errors, errors
        assert len(got) == 8 and all(g is got[0] for g in got)
        assert len(loads) == 1, "the lock must make the first load happen once"


# --------------------------------------------------------------------------- #
# 3. adapt() on CPU tensors
# --------------------------------------------------------------------------- #
class TestAdapt:
    def test_defaults_of_the_input_classes(self):
        adapt, abi = _mods()
        for cls, key in ((abi.ForwardInput, FWD), (abi.BackwardInput, BWD)):
            obj = cls()
            for name, _kind, default in abi.FIELDS[key]:
                assert getattr(obj, name) == default, f"{key}.{name}"
        fwd, _ = adapt(FWD, abi.ForwardInput())
        assert (fwd.it_begin, fwd.it_end, fwd.fs_faces, fwd.transfer_interval) == (0, -1, -1, 1)
        assert fwd.boundary_session is None
        assert fwd.models.n == 0 and fwd.source.defined_ is False
        bwd, _ = adapt(BWD, abi.BackwardInput())
        assert (bwd.bw_it_begin, bwd.bw_it_end, bwd.compute_illumination) == (-1, 0, True)
        assert bwd.grads_out.n == 0 and bwd.adcig_out.defined_ is False

    def test_float32_tensor_becomes_a_buf(self):
        adapt, abi = _mods()
        t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        fi = abi.ForwardInput()
        fi.source = t
        core, _ = adapt(FWD, fi)
        b = core.source
        assert b.defined_ is True
        assert b.data_ == t.data_ptr()
        assert b.ndim_ == 3 and b.numel_ == 24 and b.elem_size_ == 4
        assert b.dtype_ == abi.FP32
        assert b.is_cuda_ is False and b.device_ == -1
        assert _sizes(b) == (2, 3, 4) and _strides(b) == (12, 4, 1)
        assert list(b.sizes_)[3:] == [0] * 5 and list(b.strides_)[3:] == [0] * 5

    def test_non_contiguous_view_keeps_torch_strides_verbatim(self):
        adapt, abi = _mods()
        base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        v = base[:, ::2]
        assert not v.is_contiguous()
        fi = abi.ForwardInput()
        fi.source = v
        core, _ = adapt(FWD, fi)
        b = core.source
        assert _sizes(b) == tuple(v.shape) == (2, 2, 4)
        assert _strides(b) == tuple(v.stride()) == (12, 8, 1)
        assert b.numel_ == 16 and b.data_ == v.data_ptr()

    def test_none_is_an_undefined_buf(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.source = None
        core, _ = adapt(FWD, fi)
        b = core.source
        assert b.defined_ is False and b.numel_ == 0 and b.ndim_ == 0
        assert b.data_ is None

    def test_nine_dims_is_refused(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.source = torch.zeros((1,) * 9)
        with pytest.raises(RuntimeError):
            adapt(FWD, fi)

    @pytest.mark.parametrize("dtype, tag, width", [
        (torch.uint8, "INT8", 1),
        (torch.float16, "FP16", 2),
        (torch.bfloat16, "BF16", 2),
        (torch.float32, "FP32", 4),
    ])
    def test_storage_dtype_tags(self, dtype, tag, width):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.source = torch.zeros(3, 5, dtype=dtype)
        core, _ = adapt(FWD, fi)
        assert core.source.dtype_ == getattr(abi, tag)
        assert core.source.elem_size_ == width

    def test_float64_is_refused(self):
        """8 bytes wide, but 'anything else' tags FP32 (4): the width check
        buf_of made is the shim's too."""
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.source = torch.zeros(3, 5, dtype=torch.float64)
        with pytest.raises(RuntimeError):
            adapt(FWD, fi)

    def test_list_of_tensors_becomes_a_buflist(self):
        adapt, abi = _mods()
        ts = [torch.zeros(2, 2), torch.ones(3, 1), torch.zeros(4)]
        fi = abi.ForwardInput()
        fi.models = ts
        core, keep = adapt(FWD, fi)
        assert core.models.n == 3
        for i, t in enumerate(ts):
            assert core.models.p[i].data_ == t.data_ptr()
            assert core.models.p[i].defined_ is True
            assert _sizes(core.models.p[i]) == tuple(t.shape)
        assert _ptr_addr(core.models.p) in _addresses(keep), \
            "the Buf array a BufList points into must be in the keepalive"

    def test_host_int_span_from_an_int64_tensor(self):
        adapt, abi = _mods()
        idx = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
        fi = abi.ForwardInput()
        fi.source_field_indices = idx
        fi.receiver_field_indices = torch.tensor([1], dtype=torch.int64)
        core, keep = adapt(FWD, fi)
        span = core.source_field_indices
        assert span.n == 4
        assert _span_values(span) == [0, 2, 5, 7]
        assert _span_values(core.receiver_field_indices) == [1]
        # the int32 copy outlives the call: it is in the keepalive, and it is
        # a copy (the int64 source is not what the span reads)
        addrs = _addresses(keep)
        assert _ptr_addr(span.p) in addrs
        assert _ptr_addr(span.p) != idx.data_ptr()
        assert _ptr_addr(core.receiver_field_indices.p) in addrs

    def test_host_buf_for_checkpoint_steps(self):
        adapt, abi = _mods()
        steps = torch.tensor([0, 10, 20, 30, 40], dtype=torch.int64)
        fi = abi.ForwardInput()
        fi.checkpoint_steps = steps
        core, keep = adapt(FWD, fi)
        b = core.checkpoint_steps
        assert b.defined_ is True and b.numel_ == 5 and b.elem_size_ == 4
        assert b.is_cuda_ is False
        assert b.data_ in _addresses(keep)
        assert b.data_ != steps.data_ptr()
        assert _buf_values(b) == [0, 10, 20, 30, 40]

    def test_int_lists_float_lists_and_string_lists(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.pad_lo = [10, 0, 20]
        fi.pad_hi = [10, 20]
        fi.spacing = [10.0, 12.5]
        fi.boundary_disk_files = ["/tmp/a.bin", "/tmp/bé.bin"]
        core, keep = adapt(FWD, fi)
        assert _span_values(core.pad_lo) == [10, 0, 20]
        assert _span_values(core.pad_hi) == [10, 20]
        assert _span_values(core.spacing) == [10.0, 12.5]
        assert core.boundary_disk_files.n == 2
        assert [core.boundary_disk_files.p[i] for i in range(2)] == \
            [s.encode() for s in fi.boundary_disk_files]
        addrs = _addresses(keep)
        for span in (core.pad_lo, core.pad_hi, core.spacing, core.boundary_disk_files):
            assert _ptr_addr(span.p) in addrs, "span arrays must be in the keepalive"

    def test_empty_lists_are_empty_spans(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        core, _ = adapt(FWD, fi)
        for span in (core.pad_lo, core.pad_hi, core.spacing, core.boundary_disk_files,
                     core.source_field_indices, core.receiver_field_indices):
            assert span.n == 0

    def test_scalars_are_copied(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.M = 4
        fi.abcn = 20
        fi.nt = 1000
        fi.dt = 0.001
        fi.save_all_wavefields = True
        fi.use_boundary_saving = False
        fi.free_surface = True
        fi.fs_faces = 3
        fi.it_begin = 2
        fi.it_end = 7
        fi.step_phase = 2
        fi.cut_face_mask = 1
        fi.boundary_ring_buffers = 3
        core, _ = adapt(FWD, fi)
        assert (core.M, core.abcn, core.nt) == (4, 20, 1000)
        assert core.dt == pytest.approx(0.001, rel=1e-6)
        assert core.save_all_wavefields is True and core.use_boundary_saving is False
        assert core.free_surface is True and core.fs_faces == 3
        assert (core.it_begin, core.it_end, core.step_phase, core.cut_face_mask) == (2, 7, 2, 1)
        assert core.boundary_ring_buffers == 3

        bi = abi.BackwardInput()
        bi.compute_illumination = False
        bi.bw_it_begin = 50
        bi.bw_it_end = 25
        bi.compute_adcig = True
        bi.adcig_max_lag = 4
        bcore, _ = adapt(BWD, bi)
        assert bcore.compute_illumination is False
        assert (bcore.bw_it_begin, bcore.bw_it_end) == (50, 25)
        assert bcore.compute_adcig is True and bcore.adcig_max_lag == 4

    def test_boundary_session_handle_crosses_as_a_pointer(self):
        from sweep.backend.c.runners import BoundarySession
        adapt, abi = _mods()
        _core_or_skip()
        sess = BoundarySession()
        assert sess.used is False
        sess.finish()                         # nothing outstanding: no error
        h = _as_int(sess.handle)
        assert h, "a session carries a non-null core handle"
        fi = abi.ForwardInput()
        fi.boundary_session = sess
        core, _ = adapt(FWD, fi)
        assert _as_int(core.boundary_session) == h
        fi.boundary_session = None
        core, _ = adapt(FWD, fi)
        assert core.boundary_session is None
        sess.close()

    def test_backward_input_lists_and_outputs_bind(self):
        adapt, abi = _mods()
        bi = abi.BackwardInput()
        g0 = torch.zeros(1, 3, 10)
        g1 = torch.zeros(1, 1, 8, 8)
        illum = [torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8)]
        bi.grads_out = [g0, g1]
        bi.illum_out = illum
        bi.adcig_out = torch.zeros(3, 1, 1, 8, 8)
        core, keep = adapt(BWD, bi)
        assert core.grads_out.n == 2
        assert core.grads_out.p[0].data_ == g0.data_ptr()
        assert core.grads_out.p[1].data_ == g1.data_ptr()
        assert core.illum_out.n == 2 and core.illum_out.p[1].data_ == illum[1].data_ptr()
        assert core.adcig_out.defined_ is True and _sizes(core.adcig_out) == (3, 1, 1, 8, 8)


# --------------------------------------------------------------------------- #
# 4. input validation
# --------------------------------------------------------------------------- #
class TestHostCopyCache:
    """The per-call path copies the host index tensors once per params object
    while they stay the same object at the same version (two device syncs per
    step otherwise); a write or a replacement invalidates the copy."""

    def test_same_object_reuses_the_copy_and_changes_invalidate(self):
        torch = pytest.importorskip("torch")
        import sweep.backend.c.adapt as adapt
        from sweep.backend.c import abi
        p = abi.ForwardInput()
        idx = torch.arange(6, dtype=torch.int64)
        p.source_field_indices = idx
        h1 = adapt._host_int32_cached(p, "source_field_indices", idx)
        h2 = adapt._host_int32_cached(p, "source_field_indices", idx)
        assert h2 is h1 and h1.dtype == torch.int32 and h1.tolist() == list(range(6))
        idx[0] = 41                                             # in-place write bumps _version
        h3 = adapt._host_int32_cached(p, "source_field_indices", idx)
        assert h3 is not h1 and h3[0].item() == 41
        other = torch.arange(6, dtype=torch.int64) + 7         # a new tensor, new identity
        h4 = adapt._host_int32_cached(p, "source_field_indices", other)
        assert h4 is not h3 and h4[0].item() == 7
        assert adapt._host_int32_cached(p, "source_field_indices", other) is h4

    def test_an_object_that_refuses_attributes_still_works(self):
        torch = pytest.importorskip("torch")
        import sweep.backend.c.adapt as adapt

        class Rigid:
            __slots__ = ()

        idx = torch.arange(3, dtype=torch.int64)
        h = adapt._host_int32_cached(Rigid(), "receiver_field_indices", idx)
        assert h.dtype == torch.int32 and h.tolist() == [0, 1, 2]


class TestInputValidation:
    @pytest.mark.parametrize("steps", [[0, 10, 20], (0, 10, 20), np.array([0, 10, 20])])
    def test_checkpoint_steps_from_a_plain_sequence(self, steps):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.checkpoint_steps = steps
        core, keep = adapt(FWD, fi)
        b = core.checkpoint_steps
        assert b.defined_ is True and b.numel_ == 3 and b.elem_size_ == 4 and b.is_cuda_ is False
        assert _buf_values(b) == [0, 10, 20]
        assert b.data_ in _addresses(keep)

    def test_checkpoint_steps_empty_list_is_the_propagators_empty_sentinel(self):
        """The propagator passes ``torch.empty(0, int32)`` for 'no checkpoints':
        a defined, zero-numel Buf.  A plain ``[]`` must mean the same."""
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.checkpoint_steps = []
        core, _ = adapt(FWD, fi)
        assert core.checkpoint_steps.defined_ is True and core.checkpoint_steps.numel_ == 0
        fi.checkpoint_steps = torch.empty(0, dtype=torch.int32)
        core, _ = adapt(FWD, fi)
        assert core.checkpoint_steps.defined_ is True and core.checkpoint_steps.numel_ == 0

    def test_checkpoint_steps_scalar_is_refused(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.checkpoint_steps = 5
        with pytest.raises(RuntimeError, match="checkpoint_steps"):
            adapt(FWD, fi)

    def test_host_int_spans_from_plain_sequences(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.source_field_indices = [0, 2]
        fi.receiver_field_indices = (1,)
        core, keep = adapt(FWD, fi)
        assert _span_values(core.source_field_indices) == [0, 2]
        assert _span_values(core.receiver_field_indices) == [1]
        addrs = _addresses(keep)
        assert _ptr_addr(core.source_field_indices.p) in addrs
        assert _ptr_addr(core.receiver_field_indices.p) in addrs

    @pytest.mark.parametrize("key, field", [(FWD, "models"), (FWD, "wavefields"), (BWD, "grads_out")])
    def test_single_tensor_for_a_buflist_is_refused_by_name(self, key, field):
        adapt, abi = _mods()
        obj = abi.ForwardInput() if key == FWD else abi.BackwardInput()
        setattr(obj, field, torch.zeros(2, 2))
        with pytest.raises(RuntimeError, match=rf"{key}\.{field} takes a list of tensors"):
            adapt(key, obj)

    def test_non_tensor_element_in_a_buflist_is_refused_by_name(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.models = [torch.zeros(2, 2), [1.0, 2.0]]
        with pytest.raises(RuntimeError, match=r"ForwardInputCore\.models\[1\]"):
            adapt(FWD, fi)

    def test_none_entries_in_a_buflist_stay_undefined(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        t = torch.zeros(2, 2)
        fi.models = [t, None]
        core, _ = adapt(FWD, fi)
        assert core.models.n == 2
        assert core.models.p[0].data_ == t.data_ptr()
        assert core.models.p[1].defined_ is False

    def test_cpu_models_are_refused_on_the_call_path_not_in_the_probe(self):
        """Every entry and runner takes the launch stream through
        ``_launch_stream``, which refuses host models before the core can
        dereference them; the bare ``_stream_for_models`` probe keeps
        answering 0 for host models (test_core_stream_scope pins that)."""
        from sweep.backend.c import entries
        _, abi = _mods()
        fi = abi.ForwardInput()
        fi.models = [torch.zeros(1, 1, 8, 8)]
        with pytest.raises(RuntimeError, match=r"acoustic2d_forward: models\[0\] is a cpu tensor.*expected a CUDA tensor"):
            entries._launch_stream(fi, "acoustic2d_forward")
        assert entries._stream_for_models(fi.models) == 0
        fi.models = []
        assert entries._launch_stream(fi, "acoustic2d_forward") == 0

    def test_single_tensor_for_models_is_refused_before_indexing(self):
        from sweep.backend.c import entries
        _, abi = _mods()
        fi = abi.ForwardInput()
        fi.models = torch.zeros(1, 1, 8, 8)
        with pytest.raises(RuntimeError, match="list of tensors"):
            entries._stream_for_models(fi.models)
        with pytest.raises(RuntimeError, match="list of tensors"):
            entries._launch_stream(fi, "x")

    @_ctypes_only
    @requires_binding()
    def test_cpu_models_through_an_entry_are_a_message(self):
        _core_or_skip()
        import sweep._C as _C
        fi = _C.ForwardInput()
        fi.models = [torch.zeros(1, 1, 8, 8)]
        with pytest.raises(RuntimeError, match="expected a CUDA tensor"):
            _C.acoustic2d_forward(fi)


# --------------------------------------------------------------------------- #
# 5. the boundary session
# --------------------------------------------------------------------------- #
class TestBoundarySession:
    def test_close_destroys_the_very_handle_once(self):
        lib = _FakeSessionLib()
        s = _fake_session(lib)
        h = s.handle
        assert s.closed is False
        s.close()
        assert lib.destroyed == [h] and lib.destroyed[0] is h, \
            "the c_void_p instance the core handed out goes back to destroy"
        assert s.handle is None and s.closed is True
        s.close()
        s.__del__()
        assert len(lib.destroyed) == 1

    def test_finish_and_used_after_close_raise(self):
        lib = _FakeSessionLib()
        s = _fake_session(lib)
        s.finish()
        assert s.used is True
        assert lib.finished == [s.handle] and lib.used_with == [s.handle]
        s.close()
        with pytest.raises(RuntimeError, match="BoundarySession is closed"):
            s.finish()
        with pytest.raises(RuntimeError, match="BoundarySession is closed"):
            s.used
        assert len(lib.finished) == 1 and len(lib.used_with) == 1, "nothing reached the core"

    def test_used_is_a_read_only_property_as_on_dev(self):
        """dev's pybind binding exposes ``used`` as a read-only property.  As a
        method, ``if session.used:`` is always true and a caller that records
        it gets a bound method (dd_session_bench failed to json-dump it)."""
        import inspect
        import json
        from sweep.backend.c.runners import BoundarySession
        assert isinstance(inspect.getattr_static(BoundarySession, "used"), property)
        s = _fake_session(_FakeSessionLib())
        assert s.used is True
        assert json.loads(json.dumps({"used": s.used})) == {"used": True}
        with pytest.raises(AttributeError):
            s.used = False

    def test_the_session_dd_builds_answers_used_as_a_bool(self):
        """The object the DD propagator gets from ``sweep._C.BoundarySession``
        -- the ctypes class, or the pybind shim's under SWEEP_JIT_FULL=1."""
        _core_or_skip()
        from sweep import _C
        s = _C.BoundarySession()
        assert s.used is False
        s.finish()
        assert s.used is False

    @pytest.mark.parametrize("key", [FWD, BWD])
    def test_adapt_refuses_a_closed_session(self, key):
        adapt, abi = _mods()
        obj = abi.ForwardInput() if key == FWD else abi.BackwardInput()
        s = _fake_session(_FakeSessionLib(), addr=0xBEEF)
        obj.boundary_session = s
        core, _ = adapt(key, obj)
        assert _as_int(core.boundary_session) == 0xBEEF
        s.close()
        with pytest.raises(RuntimeError, match=rf"{key}\.boundary_session: BoundarySession is closed"):
            adapt(key, obj)
        obj.boundary_session = None
        core, _ = adapt(key, obj)
        assert core.boundary_session is None

    def test_raw_handles_still_cross(self):
        adapt, abi = _mods()
        fi = abi.ForwardInput()
        fi.boundary_session = ctypes.c_void_p(0x40)
        assert _as_int(adapt(FWD, fi)[0].boundary_session) == 0x40
        fi.boundary_session = 0x50
        assert _as_int(adapt(FWD, fi)[0].boundary_session) == 0x50
        fi.boundary_session = 0
        assert adapt(FWD, fi)[0].boundary_session is None

    def test_del_and_close_are_safe_on_a_half_built_session(self):
        from sweep.backend.c.runners import BoundarySession
        s = BoundarySession.__new__(BoundarySession)   # __init__ never ran
        assert s.closed is True
        s.close()
        s.__del__()
        with pytest.raises(RuntimeError, match="closed"):
            s.used

    def test_real_session_lifecycle(self):
        from sweep.backend.c.runners import BoundarySession
        adapt, abi = _mods()
        _core_or_skip()
        s = BoundarySession()
        assert s.used is False
        s.finish()
        s.close()
        assert s.closed and s.handle is None
        for call in (s.finish, lambda: s.used):
            with pytest.raises(RuntimeError, match="BoundarySession is closed"):
                call()
        fi = abi.ForwardInput()
        fi.boundary_session = s
        with pytest.raises(RuntimeError, match="closed"):
            adapt(FWD, fi)
        s.close()


# --------------------------------------------------------------------------- #
# 6. the entries and the runners return nothing (the outputs are the bound
#    tensors); a core error is the core's message
# --------------------------------------------------------------------------- #
class _FakeCallCore(_FakeCore):
    """A fake core whose ``sweep_call`` / runner calls succeed (or fail with
    ``message``) without a GPU: the stream scope and the runner handle are
    stubbed, everything else is ``_FakeCore``."""

    def __init__(self, abi, rc=0, message=b""):
        super().__init__(abi.ABI_VERSION, abi.ctypes_layout())
        self.calls = []
        self.stream = 0

        def call(entry, core_p, out_p, err, cap):
            self.calls.append(("call", int(entry), int(core_p), int(out_p)))
            ctypes.memmove(err, message + b"\0", len(message) + 1)
            return rc

        def runner_create(entry, core_p, h_ref, err, cap):
            self.calls.append(("create", int(entry)))
            h_ref._obj.value = 0xC0FE
            return 0

        def runner_run(h, a, b, phase, out_p, err, cap):
            self.calls.append(("run", int(a), int(b), int(phase), int(out_p)))
            ctypes.memmove(err, message + b"\0", len(message) + 1)
            return rc

        self.sweep_call = _FakeFn(call)
        self.sweep_get_stream = _FakeFn(lambda: self.stream)
        # c_void_p(0).value is None: NULL, i.e. the legacy default stream, is 0 here
        self.sweep_set_stream = _FakeFn(lambda s: setattr(self, "stream", (s.value if isinstance(s, ctypes.c_void_p) else s) or 0))
        self.sweep_forward_runner_create = _FakeFn(runner_create)
        self.sweep_forward_runner_run = _FakeFn(runner_run)
        self.sweep_forward_runner_device_index = _FakeFn(lambda h: -1)
        self.sweep_forward_runner_destroy = _FakeFn(lambda h: self.calls.append(("destroy", h.value)))


def _cpu_input(abi):
    """A ForwardInput with no models: ``_launch_stream`` then needs no device
    (and refuses nothing), and ``adapt`` runs on host tensors alone."""
    fi = abi.ForwardInput()
    fi.record_out = torch.zeros(1, 4, 10)
    fi.nt = 10
    return fi


class TestReturnNothing:
    def test_an_entry_returns_none_and_the_record_is_the_bound_tensor(self):
        from sweep.backend.c import entries
        _, abi = _mods()
        fake = _FakeCallCore(abi)
        fn = entries._make_entry(fake, 3, "acoustic2d_forward", 0)
        fi = _cpu_input(abi)
        rec = fi.record_out
        assert fn(fi) is None, "the entry hands back nothing: the caller reads what it bound"
        assert fi.record_out is rec
        assert [c[:2] for c in fake.calls] == [("call", 3)]
        assert fake.stream == 0, "the stream scope put the previous stream back"
        assert fn.__name__ == "acoustic2d_forward" and "None" in fn.__doc__

    def test_an_entry_error_is_the_cores_message(self):
        from sweep.backend.c import entries
        _, abi = _mods()
        fake = _FakeCallCore(abi, rc=7, message=b"acoustic2d_forward: nt must be positive")
        fn = entries._make_entry(fake, 3, "acoustic2d_forward", 0)
        with pytest.raises(RuntimeError, match="nt must be positive"):
            fn(_cpu_input(abi))
        fake = _FakeCallCore(abi, rc=7)
        fn = entries._make_entry(fake, 3, "acoustic2d_forward", 0)
        with pytest.raises(RuntimeError, match=r"acoustic2d_forward failed \(rc=7\) without a message"):
            fn(_cpu_input(abi))

    def test_an_unknown_entry_kind_is_refused_by_name(self):
        from sweep.backend.c import entries
        _, abi = _mods()
        with pytest.raises(RuntimeError, match="acoustic2d_forward has unknown kind 9"):
            entries._make_entry(_FakeCallCore(abi), 3, "acoustic2d_forward", 9)

    def test_a_runner_run_returns_none_and_close_destroys_once(self, monkeypatch):
        from sweep.backend.c import runners
        _, abi = _mods()
        fake = _FakeCallCore(abi)
        monkeypatch.setattr(runners, "core_lib", lambda: fake)
        r = runners.ForwardRunner(5, _cpu_input(abi))
        assert fake.calls == [("create", 5)]
        assert r.run(0, 4, 0) is None and r.run(4, 8, 2) is None
        assert [c[:4] for c in fake.calls[1:]] == [("run", 0, 4, 0), ("run", 4, 8, 2)]
        assert r.device_index() == -1
        r.close()
        r.close()
        assert fake.calls[-1] == ("destroy", 0xC0FE) and fake.calls.count(("destroy", 0xC0FE)) == 1
        with pytest.raises(RuntimeError, match="forward runner is closed"):
            r.run(0, 1, 0)

    def test_a_runner_error_is_the_cores_message(self, monkeypatch):
        from sweep.backend.c import runners
        _, abi = _mods()
        fake = _FakeCallCore(abi, rc=3, message=b"segment out of range")
        monkeypatch.setattr(runners, "core_lib", lambda: fake)
        r = runners.ForwardRunner(5, _cpu_input(abi))
        with pytest.raises(RuntimeError, match="segment out of range"):
            r.run(0, 99, 0)


# --------------------------------------------------------------------------- #
# 7. errors come back as messages
# --------------------------------------------------------------------------- #
class TestErrors:
    def test_bad_entry_is_a_message(self):
        lib = _core_or_skip()
        err = ctypes.create_string_buffer(4096)
        rc = lib.sweep_call(10_000, None, None, err, 4096)
        assert rc != 0
        assert b"no entry 10000" in err.value, err.value

    def test_bad_runner_is_a_message(self):
        lib = _core_or_skip()
        err = ctypes.create_string_buffer(4096)
        h = ctypes.c_void_p()
        rc = lib.sweep_forward_runner_create(10_000, None, ctypes.byref(h), err, 4096)
        assert rc != 0
        assert b"no stepped forward runner" in err.value, err.value
        assert not h.value


# --------------------------------------------------------------------------- #
# 8. stream scope (GPU)
# --------------------------------------------------------------------------- #
@requires_binding()
def test_core_stream_follows_torch_and_is_restored():
    lib = _core_or_skip()
    import sweep._C as _C
    try:
        from test_core_stream_scope import _forward_input
    except ImportError:                                    # the helper moved: same three lines
        def _forward_input(device):
            fi = _C.ForwardInput()
            fi.models = [torch.zeros(1, 1, 8, 8, device=device)]
            return fi
    lib.sweep_get_stream.restype = ctypes.c_void_p
    dev = torch.device("cuda", torch.cuda.current_device())
    fi = _forward_input(dev)
    before = lib.sweep_get_stream() or 0
    s = torch.cuda.Stream(device=dev)
    with torch.cuda.stream(s):
        seen = _C._core_stream_for(fi)
    assert seen == s.cuda_stream
    assert seen != torch.cuda.default_stream(dev).cuda_stream
    assert (lib.sweep_get_stream() or 0) == before, "the scope must put the previous stream back"
    assert _C._core_stream_for(fi) == torch.cuda.current_stream(dev).cuda_stream
    assert (lib.sweep_get_stream() or 0) == before


# --------------------------------------------------------------------------- #
# 9. runner round trip (GPU)
# --------------------------------------------------------------------------- #
@_ctypes_only
@requires_binding()
def test_forward_runner_matches_the_monolithic_entry_bit_for_bit():
    _core_or_skip()
    import sweep._C as _C
    from conftest import capture
    from test_stepped_forward import build

    prop, wavelet, sources, receivers, models = build(2)
    cap = capture(prop)
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=models)
    p = cap["params"]
    L = list(p.wavefields) or [torch.zeros_like(p.models[0]) for _ in range(9)]
    p.wavefields = L
    nt = int(p.nt)
    raw_record = p.record_out          # what the public run wrote (the entry returns nothing)

    def fresh():
        for t in L:
            t.zero_()
        rec = torch.zeros_like(raw_record)
        p.record_out = rec
        return rec

    rec_runner = fresh()
    runner = _C.acoustic2d_forward_runner(p)
    assert runner.run(0, nt, 0) is None, "run() hands back nothing: the record is the bound record_out"
    got = rec_runner.clone()
    runner.close()
    runner.close()                                          # twice is harmless

    rec_entry = fresh()
    assert _C.acoustic2d_forward(p) is None, "the entry hands back nothing either"
    assert torch.equal(got, rec_entry)
    assert torch.equal(got, raw_record), "and both reproduce the public run"
    assert float(got.abs().max()) > 0.0, "a record with no energy proves nothing"


# --------------------------------------------------------------------------- #
# 10. only the ten stepped equations have a factory
# --------------------------------------------------------------------------- #
def test_non_runner_equations_expose_no_factory():
    _core_or_skip()
    import sweep._C as _C
    assert getattr(_C, "acoustic_lsrtm2d_forward_runner", None) is None
    assert getattr(_C, "acoustic_lsrtm2d_backward_bs_runner", None) is None
    assert callable(getattr(_C, "acoustic2d_forward_runner", None))
    assert callable(getattr(_C, "acoustic2d_backward_bs_runner", None))
