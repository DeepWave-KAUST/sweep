"""The wheel ships a prebuilt ``libsweep_core.so``; the loader must pick the one
that fits THIS process and refuse the ones that do not.

"Fits" is three checks against the sidecar ``core.json``: the ABI the shim is
compiled against (``SWEEP_CORE_ABI_VERSION`` in core/capi.h), the CUDA major of
the user's torch, and the visible device's arch (an exact SASS entry, a SASS
entry of the same major and a lower minor, or a PTX entry it can JIT forward
from). A core that fits makes nvcc unnecessary -- ``can_compile()`` says so,
provided the CUDA runtime headers the shim includes are around -- while a core
that does not fit falls back to the local nvcc build rather than loading and
crashing.

Everything here is a unit test: no GPU, no compile. torch is pinned per test
because the host may or may not have a card, and that must not decide the
outcome.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import pytest

from sweep import _jit

CAPI = _jit._CSRC / "core" / "capi.h"


def _header_abi() -> int:
    """What core/capi.h defines: the number a core built from this tree
    reports, and what ``build.write_core_sidecar`` records."""
    m = re.search(r"^\s*#define\s+SWEEP_CORE_ABI_VERSION\s+(\d+)", CAPI.read_text(), re.M)
    assert m, "core/capi.h no longer defines SWEEP_CORE_ABI_VERSION"
    return int(m.group(1))


def _shim_abi_now() -> int:
    """What the shim speaks: the generated mirror first, the header as its
    fallback -- the same order ``_jit._shim_abi()`` uses, so a sidecar built
    with this number fits whichever of the two the loader reaches."""
    try:
        from sweep import _core_abi
        return int(_core_abi.ABI_VERSION)
    except Exception:
        return _header_abi()


ABI = _shim_abi_now()


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A fake installed package rooted at tmp: the loader's ``lib/`` is empty,
    its extension cache is empty, torch is cu12 with no device, the driver
    version cannot be told, and nothing in the environment steers it."""
    monkeypatch.setattr(_jit, "_PKG", tmp_path)
    # The lib dir may be a constant fixed at import rather than derived from
    # _PKG per call; rebase it when the loader has one (the contract only
    # promises "<pkg>/lib", not how <pkg> is spelled).
    if isinstance(getattr(_jit, "_LIB", None), Path):
        monkeypatch.setattr(_jit, "_LIB", tmp_path / "lib")
    # capi.h under the fake root too, so the abi parse finds it whichever way
    # the loader spells the path (_CSRC or _PKG/"csrc").
    dst = tmp_path / "csrc" / "core" / "capi.h"
    dst.parent.mkdir(parents=True)
    shutil.copyfile(CAPI, dst)
    monkeypatch.delenv("SWEEP_CORE", raising=False)
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    # The local-core cache: a core the host built earlier under
    # ~/.cache/torch_extensions must not make can_compile()/core_path() say
    # "at hand" here.
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    monkeypatch.setattr(_jit, "_core_so", None)
    # The PTX gate asks the driver; the host's driver must not decide a test
    # that is about archs (TestPtxDriverGate pins it the other ways).
    monkeypatch.setattr(_jit, "_driver_cuda_version", lambda: None)
    # The pip CUDA runtime headers the shim compile needs, pinned present so
    # the host's site-packages does not decide can_compile().
    headers = tmp_path / "nvidia" / "cuda_runtime" / "include"
    headers.mkdir(parents=True)
    (headers / "cuda_runtime_api.h").write_text("")
    (headers / "crt").mkdir()
    (headers / "crt" / "host_defines.h").write_text("")   # a second wheel (nvidia-cuda-nvcc) ships this one
    monkeypatch.setattr(_jit, "_nvidia_pip_includes", lambda: [str(headers)])
    import torch
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def no_device(*_a, **_k):
        raise AssertionError("get_device_capability() called with no device visible")
    monkeypatch.setattr(torch.cuda, "get_device_capability", no_device)
    return tmp_path


def _device(monkeypatch, maj: int, minr: int) -> None:
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (maj, minr))


def _sidecar(abi=ABI, cuda="12.8", archs=("70", "80"), ptx=("80",), payload=b"") -> dict:
    return {"abi": abi, "cuda": cuda, "archs": list(archs), "ptx": list(ptx),
            "sha256": hashlib.sha256(payload).hexdigest(), "flags": ["-O3"]}


def _ship(where: Path, sidecar: dict | None) -> Path:
    """A fake core (bytes, never dlopen'd) plus its sidecar in ``where``."""
    where.mkdir(parents=True, exist_ok=True)
    so = where / "libsweep_core.so"
    so.write_bytes(b"\x7fELF not a real core")
    if sidecar is not None:
        (where / "core.json").write_text(json.dumps(sidecar))
    return so.resolve()


def _chosen(result) -> Path | None:
    path, reason = result
    assert isinstance(reason, str) and reason, "reason must be a human sentence"
    return None if path is None else Path(path).resolve()


@pytest.mark.parametrize("cuda, tag", [("12.8", "cu12"), ("12.4", "cu12"), ("13.0", "cu13")])
def test_core_tag_is_the_cuda_major(cuda, tag):
    assert _jit._core_tag(cuda) == tag


class TestShippedCoreSelection:
    def test_no_lib_dir_means_no_core(self, pkg):
        assert _chosen(_jit._shipped_core()) is None

    def test_fits_without_a_device(self, pkg):
        """Cross-compiling the shim on a CPU node: no device to check the
        archs against, so a core with the right abi and cuda fits."""
        so = _ship(pkg / "lib" / "cu12", _sidecar())
        assert _chosen(_jit._shipped_core()) == so

    def test_rejects_wrong_abi(self, pkg):
        _ship(pkg / "lib" / "cu12", _sidecar(abi=ABI + 1))
        path, reason = _jit._shipped_core()
        assert path is None
        assert "abi" in reason.lower(), reason

    def test_rejects_wrong_cuda_major(self, pkg):
        """A cu13 core filed under cu12 must not load into a cu12 torch."""
        _ship(pkg / "lib" / "cu12", _sidecar(cuda="13.0"))
        path, reason = _jit._shipped_core()
        assert path is None
        assert "cuda" in reason.lower(), reason

    def test_rejects_uncovered_arch(self, pkg, monkeypatch):
        """No SASS of sm_86's major at or below it, and no PTX."""
        _device(monkeypatch, 8, 6)
        _ship(pkg / "lib" / "cu12", _sidecar(archs=("70", "90"), ptx=()))
        path, reason = _jit._shipped_core()
        assert path is None
        assert "86" in reason, reason

    def test_exact_arch_match_fits(self, pkg, monkeypatch):
        _device(monkeypatch, 8, 0)
        so = _ship(pkg / "lib" / "cu12", _sidecar(archs=("70", "80"), ptx=()))
        assert _chosen(_jit._shipped_core()) == so

    def test_same_major_lower_minor_sass_fits(self, pkg, monkeypatch):
        """An sm_80 cubin runs on sm_86: SASS is binary-compatible within a
        major, so the arch list need not spell every minor."""
        _device(monkeypatch, 8, 6)
        so = _ship(pkg / "lib" / "cu12", _sidecar(archs=("80",), ptx=()))
        assert _chosen(_jit._shipped_core()) == so

    def test_same_major_higher_minor_sass_does_not_fit(self, pkg, monkeypatch):
        """The other direction does not hold: sm_86 code cannot run on sm_80."""
        _device(monkeypatch, 8, 0)
        _ship(pkg / "lib" / "cu12", _sidecar(archs=("86",), ptx=()))
        path, reason = _jit._shipped_core()
        assert path is None
        assert "80" in reason, reason

    def test_minor_compatibility_stops_at_the_major(self, pkg, monkeypatch):
        """sm_90 is numerically above sm_86 but a different major: no SASS
        compatibility across majors, only PTX would reach it."""
        _device(monkeypatch, 9, 0)
        _ship(pkg / "lib" / "cu12", _sidecar(archs=("86",), ptx=()))
        assert _chosen(_jit._shipped_core()) is None

    def test_ptx_jits_forward_to_a_newer_device(self, pkg, monkeypatch):
        """sm_90 is not in archs, but compute_80 PTX runs there."""
        _device(monkeypatch, 9, 0)
        so = _ship(pkg / "lib" / "cu12", _sidecar(archs=("70", "80"), ptx=("80",)))
        assert _chosen(_jit._shipped_core()) == so

    def test_ptx_does_not_run_on_an_older_device(self, pkg, monkeypatch):
        """compute_80 PTX cannot be JIT'd for sm_75."""
        _device(monkeypatch, 7, 5)
        _ship(pkg / "lib" / "cu12", _sidecar(archs=("80",), ptx=("80",)))
        assert _chosen(_jit._shipped_core()) is None

    def test_lib_dir_follows_the_torch_cuda_major(self, pkg, monkeypatch):
        import torch
        monkeypatch.setattr(torch.version, "cuda", "13.0")
        _ship(pkg / "lib" / "cu12", _sidecar(cuda="12.8"))          # wrong drawer
        assert _chosen(_jit._shipped_core()) is None
        so = _ship(pkg / "lib" / "cu13", _sidecar(cuda="13.1"))     # same major suffices
        assert _chosen(_jit._shipped_core()) == so


class TestSweepCoreOverride:
    def test_env_beats_the_packaged_core(self, pkg, monkeypatch, tmp_path):
        _ship(pkg / "lib" / "cu12", _sidecar())
        mine = _ship(tmp_path / "custom", _sidecar())
        monkeypatch.setenv("SWEEP_CORE", str(mine))
        assert _chosen(_jit._shipped_core()) == mine

    def test_env_without_a_sidecar_is_taken_on_trust(self, pkg, monkeypatch, tmp_path):
        mine = _ship(tmp_path / "custom", None)
        monkeypatch.setenv("SWEEP_CORE", str(mine))
        path, reason = _jit._shipped_core()
        assert Path(path).resolve() == mine
        assert reason == "SWEEP_CORE without sidecar"

    def test_env_without_a_sidecar_still_needs_a_cuda_torch(self, pkg, monkeypatch, tmp_path):
        """Trust covers the sidecar's checks, not the one that needs no sidecar:
        a CPU-only torch has no CUDA runtime to load any core under."""
        import torch
        monkeypatch.setattr(torch.version, "cuda", None)
        mine = _ship(tmp_path / "custom", None)
        monkeypatch.setenv("SWEEP_CORE", str(mine))
        path, reason = _jit._shipped_core()
        assert path is None
        assert "no CUDA build" in reason, reason

    def test_env_must_be_named_libsweep_core_so(self, pkg, monkeypatch, tmp_path):
        """The shim links -lsweep_core, so any other name would pass here and
        fail at link time with a message that does not mention SWEEP_CORE."""
        d = tmp_path / "custom"
        d.mkdir()
        odd = d / "core.so"
        odd.write_bytes(b"\x7fELF not a real core")
        monkeypatch.setenv("SWEEP_CORE", str(odd))
        path, reason = _jit._shipped_core()
        assert path is None
        assert reason == "SWEEP_CORE must point at a file named libsweep_core.so"

    def test_env_path_is_resolved(self, pkg, monkeypatch, tmp_path):
        """The shim's rpath is the core's directory; a symlink that is later
        repointed would silently swap the loaded core, so the real path wins."""
        real = _ship(tmp_path / "real", _sidecar())
        link_dir = tmp_path / "link"
        link_dir.mkdir()
        link = link_dir / "libsweep_core.so"
        link.symlink_to(real)
        (link_dir / "core.json").write_text(json.dumps(_sidecar(abi=ABI + 1)))   # must be ignored
        monkeypatch.setenv("SWEEP_CORE", str(link))
        path, reason = _jit._shipped_core()
        assert path is not None and Path(path) == real     # already resolved, no .resolve() here

    def test_env_with_a_sidecar_is_still_checked(self, pkg, monkeypatch, tmp_path):
        """An explicit core that declares the wrong abi is refused, not
        silently swapped for the packaged one."""
        _ship(pkg / "lib" / "cu12", _sidecar())
        mine = _ship(tmp_path / "custom", _sidecar(abi=ABI + 1))
        monkeypatch.setenv("SWEEP_CORE", str(mine))
        assert _chosen(_jit._shipped_core()) is None


class TestCanCompile:
    def test_a_fitting_core_needs_no_nvcc(self, pkg, monkeypatch):
        """The point of shipping the core: pip install, no toolkit, impl='c'."""
        _ship(pkg / "lib" / "cu12", _sidecar())
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        ok, why = _jit.can_compile()
        assert ok, why

    def test_a_fitting_core_needs_no_toolkit_and_no_headers(self, pkg, monkeypatch):
        """sweep._C is a ctypes layer now: with a fitting shipped core there is
        nothing to compile, so neither a toolkit nor the pip CUDA headers are
        required (the compiled developer path, SWEEP_JIT_FULL=1, is the only
        one that still wants them)."""
        _ship(pkg / "lib" / "cu12", _sidecar())
        monkeypatch.delenv("SWEEP_JIT_FULL", raising=False)
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        monkeypatch.setattr(_jit, "_nvidia_pip_includes", lambda: [])
        ok, why = _jit.can_compile()
        assert ok, why

    def test_a_toolkit_supplies_the_headers_instead(self, pkg, monkeypatch):
        _ship(pkg / "lib" / "cu12", _sidecar())
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda")
        monkeypatch.setattr(_jit, "_nvidia_pip_includes", lambda: [])
        ok, why = _jit.can_compile()
        assert ok, why

    def test_without_a_core_nvcc_is_still_required(self, pkg, monkeypatch):
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
        ok, why = _jit.can_compile()
        assert not ok
        assert "CUDA toolkit" in why, why

    def test_can_build_still_requires_a_device(self, pkg, monkeypatch):
        _ship(pkg / "lib" / "cu12", _sidecar())
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        ok, why = _jit.can_build()
        assert not ok
        assert "no CUDA GPU is visible" in why


class TestShippedCoreInfo:
    KEYS = {"path", "reason", "tag"}

    def test_reports_the_chosen_core(self, pkg):
        so = _ship(pkg / "lib" / "cu12", _sidecar())
        info = _jit.shipped_core_info()
        assert set(info) == self.KEYS
        assert isinstance(info["path"], str) and Path(info["path"]).resolve() == so
        assert info["tag"] == "cu12"

    def test_reports_why_there_is_none(self, pkg):
        info = _jit.shipped_core_info()
        assert set(info) == self.KEYS
        assert info["path"] is None
        assert isinstance(info["reason"], str) and info["reason"]
        assert info["tag"] == "cu12"


class TestGencodeFlags:
    def test_arch_list_with_ptx(self, pkg, monkeypatch):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0;9.0+PTX")
        assert _jit._gencode_flags() == [
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_90,code=sm_90",
            "-gencode=arch=compute_90,code=compute_90",
        ]

    def test_device_path_targets_the_visible_card(self, pkg, monkeypatch):
        _device(monkeypatch, 8, 9)
        assert _jit._gencode_flags() == ["-gencode=arch=compute_89,code=sm_89"]

    def test_arch_list_wins_over_the_device(self, pkg, monkeypatch):
        """Cross-building for other cards on a machine that has one."""
        _device(monkeypatch, 8, 9)
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
        assert _jit._gencode_flags() == ["-gencode=arch=compute_70,code=sm_70"]


class TestCoreSidecar:
    ARCHS = ["70", "75", "80", "86", "89", "90"]
    FLAGS = ["-O3", "--use_fast_math", "-gencode=arch=compute_90,code=sm_90"]

    def test_schema(self, tmp_path):
        from sweep import build
        payload = b"\x7fELF the shipped core"
        (tmp_path / "libsweep_core.so").write_bytes(payload)

        out = build.write_core_sidecar(tmp_path, self.ARCHS, ["90"], "12.9", self.FLAGS)

        assert Path(out) == tmp_path / "core.json"
        doc = json.loads(Path(out).read_text())
        assert doc == {
            "abi": _header_abi(),          # what a core built from this tree reports
            "cuda": "12.9",
            "archs": self.ARCHS,
            "ptx": ["90"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "flags": self.FLAGS,
        }
        assert isinstance(doc["abi"], int)

    def test_round_trip_through_the_loader(self, pkg, monkeypatch):
        """What --core writes is what _shipped_core() reads: no second schema."""
        from sweep import build
        lib = pkg / "lib" / "cu12"
        so = _ship(lib, None)
        build.write_core_sidecar(lib, self.ARCHS, ["90"], "12.9", self.FLAGS)

        _device(monkeypatch, 8, 6)
        assert _chosen(_jit._shipped_core()) == so
        _device(monkeypatch, 10, 0)                       # only PTX 90 reaches it
        assert _chosen(_jit._shipped_core()) == so


class TestLoadWithAShippedCore:
    """``load()`` must not touch nvcc when the shipped core fits: no core build,
    the shim linked against the shipped directory. The shim compile itself is
    stubbed -- this checks the wiring, not cpp_extension."""

    @pytest.fixture
    def wired(self, pkg, monkeypatch, tmp_path):
        from torch.utils import cpp_extension
        so = _ship(pkg / "lib" / "cu12", _sidecar())
        build_dir = tmp_path / "ext"
        stage = tmp_path / "stage"
        stage.mkdir()
        for name in ("module.cpp", "cpu_binding_stub.cpp"):
            (stage / name).write_text("")
        shim = [str(stage / "module.cpp"), str(stage / "cpu_binding_stub.cpp")]
        # a .cu in the staged list must be dropped, not compiled, as before
        monkeypatch.setattr(_jit, "_stage", lambda _b: (shim + [str(stage / "k.cu")], [str(stage)]))
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        monkeypatch.setattr(_jit, "_nvidia_pip_libs", lambda: [])
        monkeypatch.setattr(_jit, "_module", None)
        monkeypatch.setenv("PATH", os.environ.get("PATH", ""))    # load() prepends; restore after
        monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(cpp_extension, "_get_build_directory", lambda *a, **k: str(build_dir))

        def no_core_build(*_a, **_k):
            raise AssertionError("_build_core() ran although a shipped core fits")
        monkeypatch.setattr(_jit, "_build_core", no_core_build)

        calls = []
        sentinel = object()

        def fake_load(**kw):
            calls.append(kw)
            return sentinel
        monkeypatch.setattr(cpp_extension, "load", fake_load)
        return so, calls, sentinel, shim

    def test_core_path_is_the_shipped_core_and_load_is_the_developer_path(self, wired, monkeypatch):
        """The default path never builds or links anything: core_path() hands
        the shipped library over and load() (the compiled pybind shim) refuses
        unless SWEEP_JIT_FULL=1 selects it."""
        so, calls, sentinel, shim = wired
        monkeypatch.delenv("SWEEP_JIT_FULL", raising=False)
        monkeypatch.setattr(_jit, "_core_so", None, raising=False)
        assert _jit.core_path() == so
        assert calls == []
        with pytest.raises(RuntimeError, match="SWEEP_JIT_FULL"):
            _jit.load(compile_only=True)

    def test_skips_the_core_build_and_links_the_shipped_one(self, wired, monkeypatch):
        """SWEEP_JIT_FULL=1: the compiled developer path still links the
        shipped core instead of building one."""
        so, calls, sentinel, shim = wired
        monkeypatch.setenv("SWEEP_JIT_FULL", "1")
        assert _jit.load(compile_only=True) is sentinel
        assert len(calls) == 1
        kw = calls[0]
        assert kw["name"] == "sweep_C"
        assert sorted(kw["sources"]) == sorted(shim)
        core_dir = str(so.parent)
        ld = kw["extra_ldflags"]
        assert f"-L{core_dir}" in ld
        assert "-lsweep_core" in ld
        assert f"-Wl,-rpath,{core_dir}" in ld


class TestPrecompileLoadsTheCore:
    """``sweep.precompile()`` on the default path must not stop at finding the
    core: ``binding.is_compiled()`` reads the ctypes shim's cache, so the core
    is dlopen'd there and then (and a core that fails the ABI guard is found
    up front, not on the first propagator)."""

    def test_default_path_loads_the_core_it_found(self, pkg, monkeypatch):
        import sweep
        from sweep import _capi
        from sweep.backend.torch import binding
        so = _ship(pkg / "lib" / "cu12", _sidecar())
        monkeypatch.delenv("SWEEP_JIT_FULL", raising=False)
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
        monkeypatch.setattr(_capi, "_lib", None)
        loaded = []

        def fake_core_lib():
            loaded.append(_jit.core_path())
            monkeypatch.setattr(_capi, "_lib", object())     # what the real one caches
            return _capi._lib
        monkeypatch.setattr(_capi, "core_lib", fake_core_lib)

        assert binding.is_compiled() is False
        assert sweep.precompile(require_gpu=False) is True
        assert loaded == [so]
        assert binding.is_compiled() is True


class TestShimAbi:
    """The ABI number the fit check compares a sidecar against is the one the
    ctypes shim's guard uses -- ``sweep._core_abi.ABI_VERSION``, the generated
    mirror -- so the two can never disagree; the header is only the fallback."""

    def test_reads_the_generated_mirror(self):
        from sweep import _core_abi
        assert _jit._shim_abi() == _core_abi.ABI_VERSION

    def test_falls_back_to_the_header_without_the_mirror(self, monkeypatch):
        import sys
        import sweep
        monkeypatch.delattr(sweep, "_core_abi", raising=False)
        monkeypatch.setitem(sys.modules, "sweep._core_abi", None)     # import raises
        assert _jit._shim_abi() == _header_abi()

    def test_none_when_neither_is_reachable(self, monkeypatch, tmp_path):
        import sys
        import sweep
        monkeypatch.delattr(sweep, "_core_abi", raising=False)
        monkeypatch.setitem(sys.modules, "sweep._core_abi", None)
        monkeypatch.setattr(_jit, "_CSRC", tmp_path / "nowhere")
        assert _jit._shim_abi() is None


class TestPtxDriverGate:
    """A PTX-only fit is the driver's JIT, and a driver cannot assemble PTX
    from a toolkit newer than itself: the sidecar's ``cuda`` (major.minor)
    must not exceed the driver's, or the core would load and fail at the
    first kernel.  Not knowing the driver version does not block."""

    @staticmethod
    def _ptx_only(pkg, monkeypatch) -> Path:
        _device(monkeypatch, 12, 0)                    # sm_120: no SASS of major 12 below
        return _ship(pkg / "lib" / "cu12", _sidecar(cuda="12.8", archs=("80", "90"), ptx=("90",)))

    def test_older_driver_is_refused_with_both_versions(self, pkg, monkeypatch):
        self._ptx_only(pkg, monkeypatch)
        monkeypatch.setattr(_jit, "_driver_cuda_version", lambda: (12, 4))
        path, reason = _jit._shipped_core()
        assert path is None
        assert reason == "device sm_120 fits only via PTX and driver CUDA 12.4 < core CUDA 12.8", reason

    @pytest.mark.parametrize("driver", [(12, 8), (12, 9), (13, 0)])
    def test_driver_at_least_the_toolkit_fits(self, pkg, monkeypatch, driver):
        so = self._ptx_only(pkg, monkeypatch)
        monkeypatch.setattr(_jit, "_driver_cuda_version", lambda: driver)
        assert _chosen(_jit._shipped_core()) == so

    def test_unknown_driver_does_not_block(self, pkg, monkeypatch):
        """No libcuda (a CPU node cross-checking a core): today's answer."""
        so = self._ptx_only(pkg, monkeypatch)
        monkeypatch.setattr(_jit, "_driver_cuda_version", lambda: None)
        assert _chosen(_jit._shipped_core()) == so

    def test_a_sass_fit_never_asks_the_driver(self, pkg, monkeypatch):
        _device(monkeypatch, 8, 6)
        so = _ship(pkg / "lib" / "cu12", _sidecar(cuda="12.8", archs=("80",), ptx=("80",)))

        def no_probe():
            raise AssertionError("the driver was probed for a SASS fit")
        monkeypatch.setattr(_jit, "_driver_cuda_version", no_probe)
        assert _chosen(_jit._shipped_core()) == so

    def test_the_sweep_core_override_is_gated_too(self, pkg, monkeypatch, tmp_path):
        _device(monkeypatch, 12, 0)
        mine = _ship(tmp_path / "custom", _sidecar(cuda="12.8", archs=("90",), ptx=("90",)))
        monkeypatch.setenv("SWEEP_CORE", str(mine))
        monkeypatch.setattr(_jit, "_driver_cuda_version", lambda: (12, 4))
        path, reason = _jit._shipped_core()
        assert path is None and "PTX" in reason and "12.4 < core CUDA 12.8" in reason, reason

    def test_driver_int_becomes_major_minor(self, monkeypatch):
        """cuDriverGetVersion's 12040 is CUDA 12.4."""
        for raw, pair in ((12040, (12, 4)), (12080, (12, 8)), (13000, (13, 0)), (None, None)):
            monkeypatch.setattr(_jit, "_driver_version_raw", lambda raw=raw: raw)
            assert _jit._driver_cuda_version() == pair

    def test_sidecar_cuda_spellings(self):
        assert _jit._version_pair("12.8") == (12, 8)
        assert _jit._version_pair("12") == (12, 0)
        assert _jit._version_pair(None) is None


_REAL_BUILD_CORE = _jit._build_core          # captured before any fixture patches it


class TestCachedLocalCore:
    """A local core built earlier is reused without a toolkit exactly when its
    source stamp -- the csrc tree plus the nvcc flags/targets -- is the one this
    process would build with; anything else falls through to the build path."""

    @pytest.fixture
    def cached(self, pkg, monkeypatch, tmp_path):
        """No shipped core; a small csrc of our own (the stamp is over the tree
        and the tests edit it); the target named, no device, no nvcc; and a
        core + stamp in the extension cache as a previous build left them."""
        csrc = tmp_path / "csrc"                       # pkg put core/capi.h here already
        (csrc / "cuda" / "common").mkdir(parents=True)
        (csrc / "cuda" / "common" / "k.cu").write_text("__global__ void k() {}\n")
        monkeypatch.setattr(_jit, "_CSRC", csrc)
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        monkeypatch.setenv("PATH", os.environ.get("PATH", ""))      # the build branch prepends

        def no_build(*_a, **_k):
            raise AssertionError("the build path ran although the cached core is current")
        monkeypatch.setattr(_jit, "_build_core", no_build)
        monkeypatch.setattr(_jit, "_stage", no_build)
        build_dir = _jit._build_dir()
        core_dir = build_dir / "core"
        core_dir.mkdir(parents=True)
        so = core_dir / "libsweep_core.so"
        so.write_bytes(b"\x7fELF not a real core")
        (core_dir / "sources.sha256").write_text(_jit._source_stamp() + "\n")
        return so.resolve(), build_dir, csrc

    def test_current_stamp_reuses_the_core_without_nvcc(self, cached):
        so, build_dir, _ = cached
        assert _jit._local_core_current(build_dir)
        assert _jit._core_will_build(build_dir) is False
        ok, why = _jit.can_compile()
        assert ok, why
        assert _jit.core_path().resolve() == so

    def test_gpu_run_reuses_the_core_a_cpu_build_left(self, cached, monkeypatch):
        """The cluster case: built on a CPU node with TORCH_CUDA_ARCH_LIST=7.0,
        run on a V100 with no arch list and no nvcc -- the same target, so the
        same stamp; a different card is a different core."""
        so, build_dir, _ = cached
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST")
        _device(monkeypatch, 7, 0)
        assert _jit._local_core_current(build_dir)
        ok, why = _jit.can_build()
        assert ok, why
        assert _jit.core_path().resolve() == so
        _device(monkeypatch, 8, 0)
        assert not _jit._local_core_current(build_dir)

    def test_edited_source_falls_through_to_the_build(self, cached, monkeypatch):
        so, build_dir, csrc = cached
        (csrc / "cuda" / "common" / "k.cu").write_text("__global__ void k() { /* edited */ }\n")
        assert not _jit._local_core_current(build_dir)
        assert _jit._core_will_build(build_dir) is True
        ok, why = _jit.can_compile()
        assert not ok and "CUDA toolkit" in why, why
        with pytest.raises(RuntimeError, match="CUDA toolkit"):
            _jit.core_path()

    def test_other_target_falls_through_too(self, cached, monkeypatch):
        so, build_dir, _ = cached
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
        assert not _jit._local_core_current(build_dir)
        with pytest.raises(RuntimeError, match="CUDA toolkit"):
            _jit.core_path()

    def test_missing_stamp_is_not_current(self, cached):
        """A core from before the stamp existed: rebuilt (ninja decides how
        much), never trusted on the strength of the .so alone."""
        so, build_dir, _ = cached
        (build_dir / "core" / "sources.sha256").unlink()
        assert not _jit._local_core_current(build_dir)
        assert _jit._core_will_build(build_dir) is True

    def test_stale_core_with_nvcc_takes_the_build_path(self, cached, monkeypatch, tmp_path):
        so, build_dir, csrc = cached
        (csrc / "cuda" / "common" / "k.cu").write_text("edited\n")
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: str(tmp_path / "cuda"))
        monkeypatch.setattr(_jit, "_stage", lambda _b: ([str(csrc / "cuda/common/k.cu")], [str(csrc)]))
        calls = []

        def fake_build(build_dir_, sources, inc, cuda_home, verbose):
            calls.append((sources, cuda_home))
            return build_dir_ / "core" / "libsweep_core.so"
        monkeypatch.setattr(_jit, "_build_core", fake_build)
        assert _jit.core_path().resolve() == so
        assert len(calls) == 1 and calls[0][1] == str(tmp_path / "cuda")

    def test_stamp_tracks_sources_and_target(self, cached, monkeypatch):
        _, _, csrc = cached
        s0 = _jit._source_stamp()
        assert _jit._source_stamp() == s0
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
        s1 = _jit._source_stamp()
        assert s1 != s0
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
        (csrc / "shared.h").write_text("// a new header\n")
        assert _jit._source_stamp() not in (s0, s1)

    def test_build_core_writes_the_stamp(self, cached, monkeypatch, tmp_path):
        """The stamp is the build's doing: a successful ninja run leaves it
        beside the .so, equal to the stamp of the tree it was built from."""
        import subprocess
        so, build_dir, csrc = cached
        core_dir = build_dir / "core"
        (core_dir / "sources.sha256").unlink()

        def fake_ninja(cmd, **kw):
            assert list(cmd[:2]) == ["ninja", "-C"], cmd
            (core_dir / "libsweep_core.so").write_bytes(b"\x7fELF built")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        monkeypatch.setattr(subprocess, "run", fake_ninja)
        monkeypatch.setattr(_jit, "_ensure_ninja_on_path", lambda: None)
        monkeypatch.setattr(_jit, "_nvidia_pip_libs", lambda: [])

        out = _REAL_BUILD_CORE(build_dir, [str(csrc / "cuda/common/k.cu")], [str(csrc)],
                               str(tmp_path / "cuda"), verbose=False)
        assert out == core_dir / "libsweep_core.so"
        assert (core_dir / "sources.sha256").read_text().strip() == _jit._source_stamp()
        assert _jit._local_core_current(build_dir)

    def test_failed_build_leaves_no_stamp(self, cached, monkeypatch, tmp_path):
        import subprocess
        so, build_dir, csrc = cached
        core_dir = build_dir / "core"
        (core_dir / "sources.sha256").unlink()
        monkeypatch.setattr(subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "nvcc: boom"))
        monkeypatch.setattr(_jit, "_ensure_ninja_on_path", lambda: None)
        monkeypatch.setattr(_jit, "_nvidia_pip_libs", lambda: [])
        with pytest.raises(RuntimeError, match="boom"):
            _REAL_BUILD_CORE(build_dir, [str(csrc / "cuda/common/k.cu")], [str(csrc)],
                             str(tmp_path / "cuda"), verbose=False)
        assert not (core_dir / "sources.sha256").exists()
        assert not _jit._local_core_current(build_dir)
