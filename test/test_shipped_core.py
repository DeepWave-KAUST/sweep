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


def _real_abi() -> int:
    m = re.search(r"^\s*#define\s+SWEEP_CORE_ABI_VERSION\s+(\d+)", CAPI.read_text(), re.M)
    assert m, "core/capi.h no longer defines SWEEP_CORE_ABI_VERSION"
    return int(m.group(1))


ABI = _real_abi()


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    """A fake installed package rooted at tmp: the loader's ``lib/`` is empty,
    torch is cu12 with no device, and nothing in the environment steers it."""
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

    def test_a_fitting_core_still_needs_the_runtime_headers(self, pkg, monkeypatch):
        """No toolkit and no pip nvidia-cuda-runtime: the shim's
        cuda_runtime_api.h include has nowhere to come from. Say so, with the
        fix, instead of letting the compiler's "No such file" say it."""
        _ship(pkg / "lib" / "cu12", _sidecar())
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        monkeypatch.setattr(_jit, "_nvidia_pip_includes", lambda: [])
        ok, why = _jit.can_compile()
        assert not ok
        assert "nvidia-cuda-runtime-cu12" in why, why

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
            "abi": ABI,
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

    def test_skips_the_core_build_and_links_the_shipped_one(self, wired):
        so, calls, sentinel, shim = wired
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
