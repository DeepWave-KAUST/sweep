"""Which impl ``PropTorch`` resolves to, per device and per build.

The CUDA core serves CUDA devices in every build: the default ctypes layer
over the prebuilt core, the ``SWEEP_JIT_FULL=1`` pybind shim and an AOT-built
extension all reach the same CUDA core.  On a CPU device the HOST core -- the
same sources built for the CPU (``sweep.backend.c.hostcore``) -- serves the
compiled path through the ctypes layer.  ``impl='auto'`` takes it quietly when
one comes up -- shipped (``lib/cpu`` or ``SWEEP_CORE_CPU``), cached, or built
here once, as a CUDA source install builds its core with nvcc -- the equation
is built into it and nothing the caller passed is eager's; no core and no
compiler leave 'auto' eager quietly, a build or load that fails warns once.  An
explicit ``impl='c'`` falls back to eager with a warning that says why when
there is no host core or the equation is not built into it.  (The
hand-written CPU engine the compiled builds used to carry is gone: its
gradients disagreed with eager and with the CUDA core; the host core cannot,
being the same code.)

No kernel runs here: the build flags, the binding probe and the host-core
probe are monkeypatched, and only the resolver is called.  The host core's
build is met through a fake compiler (a shell script), which fails where a
real one would compile.
"""
import shutil
import subprocess
import sys
import warnings

import pytest

import sweep
from sweep.backend.c import jit
from sweep.propagator import torch as ptorch


class _Eq:
    """Minimal equation: a ``_C`` hook and a ``C_NAME``."""

    def __init__(self, c_name):
        self.C_NAME = c_name

    def _C(self):  # pragma: no cover - never called by the resolver
        raise AssertionError("the resolver must not load kernels")


EQ = _Eq("acoustic2d")
BUILDS = ("default", "jit_full", "aot")


@pytest.fixture
def build(monkeypatch):
    """Pin the build: binding available, no host core, then pick default /
    jit_full / aot.  The host core is pinned OFF so the matrix below is the
    CUDA-only contract whatever the box has built or ships; ``host_core``
    turns it on."""
    monkeypatch.delenv("SWEEP_JIT_FULL", raising=False)
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: True)
    monkeypatch.setattr(ptorch, "_host_core_serves",
                        lambda equation: (False, "no host (CPU) core is at hand (test)",
                                          "get a host core (`python -m sweep.build --cpu`)"))
    monkeypatch.setattr(ptorch, "_host_core_ready", lambda equation: False)

    def _set(kind):
        assert kind in BUILDS
        monkeypatch.setattr(jit, "jit_full", lambda: kind == "jit_full")
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: kind == "aot")
    return _set


def _resolve(impl, device, equation=EQ, eager_only=False):
    """Resolve like PropTorch does; return (impl, [warning messages])."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _, got = ptorch._normalize_backend_impl("torch", impl, equation=equation, device=device,
                                                eager_only=eager_only)
    return got, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


# (impl, device) -> (resolved impl, warned?), the same in every build
MATRIX = {
    (None, "cpu"): ("eager", False),
    ("c", "cpu"): ("eager", True),
    (None, "cuda"): ("c", False),
    ("c", "cuda"): ("c", False),
}


@pytest.mark.parametrize("kind", BUILDS)
@pytest.mark.parametrize("key", list(MATRIX), ids=lambda k: f"{k[0] or 'auto'}-{k[1]}")
def test_matrix(build, kind, key):
    impl, device = key
    want, want_warn = MATRIX[key]
    build(kind)
    got, msgs = _resolve(impl, device)
    assert got == want
    assert bool(msgs) == want_warn, msgs


@pytest.mark.parametrize("kind", BUILDS)
def test_explicit_c_on_cpu_without_a_host_core_says_so(build, kind):
    """The warning names the first thing in the way: on the ctypes layer the
    missing core (and how to get one); under the pybind shim the shim itself,
    since a host core would not be reached through it anyway."""
    build(kind)
    got, msgs = _resolve("c", "cpu")
    assert got == "eager"
    (msg,) = msgs
    if kind == "default":
        assert "no host (CPU) core is at hand" in msg and "sweep.build --cpu" in msg
    else:
        assert "pybind shim" in msg and "SWEEP_JIT_FULL" in msg


@pytest.fixture
def host_core(build, monkeypatch):
    """A host core is at hand and builds every equation but the named ones;
    ``comes_up=False``: it is at hand but did not come up for 'auto' (its
    build or load failed, which the probe reports once)."""
    def _set(kind, unported=("visco_acoustic2d",), comes_up=True):
        build(kind)

        def serves(equation):
            c_name = getattr(equation, "C_NAME", None) if equation is not None else None
            if c_name in unported:
                return (False, f"equation '{type(equation).__name__}' is not built into the host (CPU) core",
                        ptorch._HOST_OTHER_DEVICE_REMEDY)
            return True, "", ""
        monkeypatch.setattr(ptorch, "_host_core_serves", serves)
        monkeypatch.setattr(ptorch, "_host_core_ready", lambda equation: comes_up and serves(equation)[0])
    return _set


@pytest.mark.parametrize("kind", BUILDS)
def test_explicit_c_on_cpu_takes_the_host_core_through_the_ctypes_layer(host_core, kind):
    """The host core is reached through the ctypes layer's per-call core
    choice.  The pybind shim (SWEEP_JIT_FULL=1, an ahead-of-time extension)
    reaches the CUDA core only, so those builds fall back -- with a warning
    that names the shim, not a missing core."""
    host_core(kind)
    assert _resolve("c", "cuda") == ("c", []), "the CUDA path is untouched"
    got, msgs = _resolve("c", "cpu")
    if kind == "default":
        assert (got, msgs) == ("c", [])
    else:
        assert got == "eager"
        (msg,) = msgs
        assert "pybind shim" in msg and "SWEEP_JIT_FULL" in msg
        assert "no host (CPU) core" not in msg


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_equation_without_kernels_blames_the_equation_not_the_device(host_core, device):
    """An equation with no compiled kernels is eager everywhere; the warning
    must say so instead of blaming the device (on the CPU the device branch
    would read 'runs on the CPU ... and this propagator is on cpu')."""
    host_core("default")
    got, msgs = _resolve("c", device, equation=_Eq(None))
    assert got == "eager"
    (msg,) = msgs
    assert "does not expose a compiled kernel" in msg
    assert "Move to a CUDA device" not in msg and "this propagator is on" not in msg


@pytest.mark.parametrize("kind", BUILDS)
def test_auto_on_cpu_takes_the_host_core(host_core, kind):
    """'auto' takes a host core that comes up quietly, through the ctypes
    layer: the pybind shim reaches the CUDA core only, so those builds stay
    eager -- also quietly, nothing was asked for."""
    host_core(kind)
    assert _resolve(None, "cpu") == (("c" if kind == "default" else "eager"), [])
    assert _resolve(None, "cuda") == ("c", []), "the CUDA path is untouched"


def test_auto_on_cpu_stays_eager_when_the_host_core_does_not_come_up(host_core):
    """A host core whose build or load failed (the probe warned once) leaves
    'auto' on eager; an explicit impl='c' still asks for it, and its first
    call raises the failure."""
    host_core("default", comes_up=False)
    assert _resolve(None, "cpu") == ("eager", [])
    assert _resolve("c", "cpu") == ("c", [])


def test_auto_on_cpu_stays_eager_for_what_the_host_core_does_not_run(host_core):
    """An equation the host core does not build, one with no compiled kernels,
    and a caller who passed eager's options (``eager_only``): quietly eager."""
    host_core("default")
    assert _resolve(None, "cpu", equation=_Eq("visco_acoustic2d")) == ("eager", [])
    assert _resolve(None, "cpu", equation=_Eq(None)) == ("eager", [])
    assert _resolve(None, "cpu", eager_only=True) == ("eager", [])
    assert _resolve(None, "cuda", eager_only=True) == ("c", []), "eager_only is the CPU's"
    assert _resolve("c", "cpu", eager_only=True) == ("c", []), "an explicit impl='c' is not overruled"


def _needs_eager(**passed):
    args = {"kwargs": {}, "backend_options": None, "eager_options": None, "memory": None,
            "cuda_options": None}
    args.update(passed)
    return ptorch._cpu_request_needs_eager(**args)


def test_eager_options_keep_auto_on_eager():
    """``eager_options=`` and an eager knob anywhere it is read would raise on
    impl='c' ("eager_options can only be used with impl='eager'") -- the caller
    asked for eager, as on every CPU before the host core."""
    from sweep.propagator.options import EagerOptions
    assert not _needs_eager()
    assert _needs_eager(eager_options=EagerOptions(use_compile=False))
    assert _needs_eager(kwargs={"use_compile": False})
    assert _needs_eager(backend_options={"cpml_strips": True})
    assert not _needs_eager(kwargs={"nt": 10, "use_ckpt": True})


def test_host_staging_keeps_auto_on_eager():
    """The memory requests the host core refuses (host staging: there is no
    second tier on a CPU device) in every spelling that reaches it; what it
    runs -- including what eager does not (recursive ckpt, tail_steps) -- does
    not hold auto back."""
    from sweep.propagator.options import (BoundaryOptions, BoundarySaving, Ckpt, CUDAOptions,
                                          Full, MemoryOptions)
    for request in (BoundarySaving(storage="cpu"), BoundarySaving(storage="disk", disk_dir="/tmp"),
                    Ckpt(storage="cpu"), {"kind": "boundary", "storage": "cpu"}):
        assert _needs_eager(memory=request), request
    assert _needs_eager(cuda_options=CUDAOptions(
        memory=MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="cpu"))))
    assert _needs_eager(kwargs={"boundary_saving_config": {"enabled": True, "storage": "cpu"}})
    for request in (Full(), BoundarySaving(), BoundarySaving(tail_steps=20), Ckpt(),
                    Ckpt(mode="recursive", count=4)):
        assert not _needs_eager(memory=request), request
    assert not _needs_eager(kwargs={"boundary_saving_config": {"enabled": True, "storage": "gpu"}})
    # storage only matters where boundary saving is on
    assert not _needs_eager(kwargs={"use_ckpt": True, "boundary_saving_config": {"storage": "cpu"}})


def test_the_look_ahead_leaves_the_deprecation_warning_to_the_real_read(monkeypatch):
    """Deprecated spellings warn once per process; the look ahead must not use
    that once up, or the caller never hears about the spelling."""
    from sweep.core import arguments
    from sweep.propagator.options import (BoundaryOptions, CUDAOptions, MemoryOptions,
                                          as_memory_strategy)
    monkeypatch.setattr(arguments, "_WARNED_SPELLINGS", set())
    legacy = MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="cpu"))
    nested = {"strategy": "boundary", "boundary": {"storage": "cpu"}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _needs_eager(cuda_options=CUDAOptions(memory=legacy))
        assert _needs_eager(memory=nested)
    with pytest.warns(DeprecationWarning, match="MemoryOptions"):
        as_memory_strategy(legacy)
    with pytest.warns(DeprecationWarning, match="nested"):
        as_memory_strategy(nested)


def test_unreadable_request_is_left_to_raise_where_it_is_read():
    """Not 'needs eager': PropTorch reads the request after the resolution and
    raises there, for either impl, with its own message."""
    from sweep.propagator.options import Full
    assert not _needs_eager(memory="bogus")
    assert not _needs_eager(memory={"kind": "nope"})
    assert not _needs_eager(memory=Full(), kwargs={"use_ckpt": True})      # conflicting requests


def test_unported_equation_on_cpu_falls_back_with_the_reason(host_core):
    """...and with a remedy that helps: building a host core cannot add an
    equation the host core does not build."""
    host_core("default")
    got, msgs = _resolve("c", "cpu", equation=_Eq("visco_acoustic2d"))
    assert got == "eager"
    (msg,) = msgs
    assert "not built into the host (CPU) core" in msg
    assert "sweep.build --cpu" not in msg and "dev='cuda'" in msg


def test_host_core_probe_asks_hostcore(monkeypatch):
    """The real probe: availability and the unported list come from
    sweep.backend.c.hostcore, nothing is loaded or built."""
    from sweep.backend.c import hostcore
    monkeypatch.setattr(hostcore, "available", lambda: False)
    ok, why, remedy = ptorch._host_core_serves(EQ)
    assert not ok and "no host (CPU) core" in why and "sweep.build --cpu" in remedy
    monkeypatch.setattr(hostcore, "available", lambda: True)
    assert ptorch._host_core_serves(EQ) == (True, "", "")
    ok, why, remedy = ptorch._host_core_serves(_Eq("visco_acoustic2d"))
    assert not ok and "not built into the host (CPU) core" in why
    assert "sweep.build --cpu" not in remedy and "dev='cuda'" in remedy


def test_host_core_ready_probe_brings_the_core_up_once(monkeypatch):
    """The real 'auto' probe: the equation built into the host core, and the
    core an explicit impl='c' runs on (``hostcore.core_path``: shipped, cached,
    else built -- with the compiler's output kept in build.log, not on the
    terminal) loads, tried once per process.  No core and no compiler: eager,
    quietly, nothing tried.  A build or a load that fails: reported once."""
    from sweep.backend.c import hostcore, loader
    resolved, loads = [], []

    def core_path(verbose=True):
        resolved.append(verbose)
        return "/cache/core_cpu/libsweep_core.so"
    monkeypatch.setattr(hostcore, "core_path", core_path)
    monkeypatch.setattr(loader, "core_lib", lambda kind="cuda": loads.append(kind))

    monkeypatch.setattr(ptorch, "_HOST_CORE", None)
    monkeypatch.setattr(hostcore, "available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not ptorch._host_core_ready(EQ)
    assert (resolved, loads) == ([], [])

    monkeypatch.setattr(ptorch, "_HOST_CORE", None)
    monkeypatch.setattr(hostcore, "available", lambda: True)
    assert not ptorch._host_core_ready(_Eq("visco_acoustic2d"))
    assert ptorch._host_core_ready(EQ) and ptorch._host_core_ready(EQ)
    assert (resolved, loads) == ([False], ["cpu"]), "resolved quietly and loaded, once"

    def failed_build(verbose=True):
        raise RuntimeError("sweep host core build (libsweep_core.so, g++ 11) failed: x.cu:1:1: error: "
                           "boom (test) (the compiler's output is in /cache/core_cpu/build.log)")
    monkeypatch.setattr(hostcore, "core_path", failed_build)
    monkeypatch.setattr(ptorch, "_HOST_CORE", None)
    with pytest.warns(UserWarning, match=r"unavailable \(.*boom \(test\).*build\.log\).*runs eager on the CPU"):
        assert not ptorch._host_core_ready(EQ)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not ptorch._host_core_ready(EQ), "reported once"

    def broken(kind="cuda"):
        raise OSError("libgomp.so.1: version `GOMP_9.9' not found (test)")
    monkeypatch.setattr(hostcore, "core_path", core_path)
    monkeypatch.setattr(loader, "core_lib", broken)
    monkeypatch.setattr(ptorch, "_HOST_CORE", None)
    with pytest.warns(UserWarning, match="libsweep_core.so does not load .*GOMP_9.9.*runs eager on the CPU"):
        assert not ptorch._host_core_ready(EQ)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not ptorch._host_core_ready(EQ), "reported once"


# --------------------------------------------------------------------------- #
# the host core's build, as 'auto' meets it (a fake compiler; no kernel builds)
# --------------------------------------------------------------------------- #
needs_sh = pytest.mark.skipif(sys.platform == "win32", reason="the fake compiler is a shell script")


def _fake_cxx(bin_dir, version="g++ (GCC) 11.4.0", probe=0):
    """A compiler in ``bin_dir`` that answers ``--version`` with ``version``,
    exits ``probe`` on hostcore's probe (saying why on a failure) and fails
    every other compile."""
    path = bin_dir / "g++"
    path.write_text("#!/bin/sh\n"
                    f'case "$*" in *--version*) echo "{version}"; exit 0;; esac\n'
                    'case "$*" in *probe.cpp*) '
                    f'[ {probe} = 0 ] || echo "probe.cpp:1:10: fatal error: omp.h: No such file (test)" >&2; '
                    f'exit {probe};; esac\n'
                    'echo "x.cu:1:1: error: boom (test)" >&2\nexit 1\n')
    path.chmod(0o755)


@pytest.fixture
def no_core_but_a_fake_compiler(monkeypatch, tmp_path):
    """No host core at hand (none shipped, an empty build drawer) and
    compilers looked for afresh, in a ``bin`` dir that holds ninja (the build
    runs it) and whatever compiler the test puts there."""
    from sweep.backend.c import hostcore
    jit._ensure_ninja_on_path()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if shutil.which("ninja"):
        (bin_dir / "ninja").symlink_to(shutil.which("ninja"))
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("SWEEP_CORE_CPU", raising=False)
    monkeypatch.setattr(hostcore, "shipped_core", lambda: (None, "none shipped (test)"))
    monkeypatch.setattr(hostcore, "_core_so", None)
    monkeypatch.setattr(hostcore, "_cxx_cache", None)
    monkeypatch.setattr(hostcore, "_cxx_rejected", [])
    monkeypatch.setattr(ptorch, "_HOST_CORE", None)
    return hostcore, bin_dir


@needs_sh
def test_a_compiler_that_cannot_build_the_core_is_not_offered(no_core_but_a_fake_compiler):
    """A compiler past the version check that cannot build an OpenMP shared
    object with the host core's flags (Apple clang, clang++ without libomp) is
    turned down by the probe: no core is at hand, so 'auto' is eager quietly
    -- as a CUDA install without nvcc is -- instead of starting a build in
    every process; the reason names the compiler."""
    hostcore, bin_dir = no_core_but_a_fake_compiler
    _fake_cxx(bin_dir, probe=1)
    assert hostcore.find_host_cxx() is None and not hostcore.available()
    ok, why = hostcore.can_build()
    assert not ok and "g++ 11) does not build an OpenMP shared object: probe.cpp:1:10: fatal error" in why
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not ptorch._host_core_ready(EQ)


@needs_sh
def test_too_old_a_compiler_is_named(no_core_but_a_fake_compiler):
    hostcore, bin_dir = no_core_but_a_fake_compiler
    _fake_cxx(bin_dir, version="g++ (GCC) 8.5.0")
    assert hostcore.find_host_cxx() is None
    assert "g++ is g++ 8" in hostcore.can_build()[1]


@needs_sh
def test_a_failed_build_warns_once_and_keeps_the_compiler_output(no_core_but_a_fake_compiler):
    """The build 'auto' starts when no core is at hand keeps the compiler's
    output in build.log beside it, off the terminal; 'auto' warns once with
    the first error line and where the rest is, and runs eager."""
    hostcore, bin_dir = no_core_but_a_fake_compiler
    _fake_cxx(bin_dir)
    assert hostcore.cached_local_core() is None and hostcore.available()
    with pytest.warns(UserWarning, match=r"unavailable \(.*failed: x\.cu:1:1: error: boom \(test\) "
                                         r"\(the compiler's output is in .*build\.log\)\).*runs eager"):
        assert not ptorch._host_core_ready(EQ)
    assert "boom (test)" in (hostcore.build_dir() / "build.log").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not ptorch._host_core_ready(EQ), "reported once, not rebuilt"


def test_a_waiter_takes_only_a_current_core(monkeypatch, tmp_path):
    """A process that waited on another's build takes the drawer's core only
    when its stamp is this tree's -- the holder may have failed, or built
    another worktree's sources into the shared drawer -- and builds it itself
    otherwise."""
    from sweep.backend.c import hostcore
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    monkeypatch.setattr(hostcore, "find_host_cxx", lambda: "/usr/bin/g++")
    monkeypatch.setattr(hostcore, "_cxx_describe", lambda cxx: "g++ 11")
    monkeypatch.setattr(jit, "_ensure_ninja_on_path", lambda: None)

    class Baton(jit._FileBaton):
        """The core lock is someone else's at the first try."""
        tried = set()

        def try_acquire(self):
            if self.lock_file_path.endswith("sweep_core_cpu_lock") and self.lock_file_path not in Baton.tried:
                Baton.tried.add(self.lock_file_path)
                return False
            return super().try_acquire()

        def wait(self):
            pass
    monkeypatch.setattr(jit, "_FileBaton", Baton)
    built = []

    def ninja(cmd, **kw):
        built.append(cmd)
        (hostcore.build_dir() / "libsweep_core.so").write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(hostcore.subprocess, "run", ninja)

    monkeypatch.setattr(hostcore, "_local_core_current", lambda d: False)
    so = hostcore.build_core(verbose=False)
    assert len(built) == 1, "the holder left no current core: built here"

    Baton.tried, built[:] = set(), []
    monkeypatch.setattr(hostcore, "_local_core_current", lambda d: True)
    assert hostcore.build_core(verbose=False) == so
    assert built == [], "the holder's core is current: taken as it is"


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_indexed_cuda_string_is_cuda(build, device):
    """``dev='cuda:0'`` is a CUDA device; a string compare against 'cuda' is not."""
    import torch
    build("default")
    assert _resolve(None, device) == ("c", [])
    assert _resolve(None, torch.device(device)) == ("c", [])


@pytest.mark.parametrize("kind", BUILDS)
def test_other_devices_never_get_the_compiled_path(build, kind):
    """Nothing but CUDA and the host: not mps/xla/meta."""
    build(kind)
    assert _resolve(None, "meta")[0] == "eager"
    got, msgs = _resolve("c", "meta")
    assert got == "eager" and len(msgs) == 1


def test_no_device_keeps_the_compiled_path(build):
    build("default")
    assert _resolve(None, None) == ("c", [])


@pytest.mark.parametrize("kind", BUILDS)
def test_missing_binding_still_demotes_everywhere(build, monkeypatch, kind):
    build(kind)
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: False)
    for device in ("cpu", "cuda"):
        assert _resolve(None, device)[0] == "eager"
        got, msgs = _resolve("c", device)
        assert got == "eager" and len(msgs) == 1
