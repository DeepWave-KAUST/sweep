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
    KEYS = {"path", "reason", "tag", "available"}

    def test_reports_the_chosen_core(self, pkg):
        so = _ship(pkg / "lib" / "cu12", _sidecar())
        info = _jit.shipped_core_info()
        assert set(info) == self.KEYS
        assert isinstance(info["path"], str) and Path(info["path"]).resolve() == so
        assert info["tag"] == "cu12"
        assert info["available"] == ["cu12"]

    def test_reports_why_there_is_none(self, pkg):
        info = _jit.shipped_core_info()
        assert set(info) == self.KEYS
        assert info["path"] is None
        assert isinstance(info["reason"], str) and info["reason"]
        assert info["tag"] == "cu12"
        assert info["available"] == []

    def test_available_lists_every_drawer_holding_a_core(self, pkg, monkeypatch):
        """``available`` is what the install carries, fit or not: under a cu12
        torch the cu13 drawer is listed although it is never picked, and a
        drawer with a sidecar but no .so (a half-copied wheel) is not."""
        _ship(pkg / "lib" / "cu13", _sidecar(**CU13))
        _ship(pkg / "lib" / "cu12", _sidecar(**CU12))
        (pkg / "lib" / "cu11").mkdir()
        (pkg / "lib" / "cu11" / "core.json").write_text(json.dumps(_sidecar(cuda="11.8")))
        info = _jit.shipped_core_info()
        assert info["available"] == ["cu12", "cu13"]
        assert info["tag"] == "cu12" and Path(info["path"]).parent.name == "cu12"
        _torch_cuda(monkeypatch, "13.0")
        info = _jit.shipped_core_info()
        assert info["available"] == ["cu12", "cu13"]
        assert info["tag"] == "cu13" and Path(info["path"]).parent.name == "cu13"

    def test_binding_diagnostics_carry_the_list(self, pkg, monkeypatch):
        """``binding.diagnostics()["shipped_core"]`` is where a user looks: the
        drawer torch points at next to the drawers the install has, so
        "no shipped core for cu13" beside ``available == ["cu12"]`` explains
        itself.  The existing keys stay."""
        import sweep
        from sweep.backend.torch import binding
        _ship(pkg / "lib" / "cu12", _sidecar(**CU12))
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        _torch_cuda(monkeypatch, "13.0")
        d = binding.diagnostics()
        core = d["shipped_core"]
        assert set(core) == self.KEYS
        assert core["available"] == ["cu12"]
        assert core["tag"] == "cu13" and core["path"] is None
        assert core["reason"].startswith("no shipped core for cu13 ("), core["reason"]
        assert {"usable", "reason", "shim", "cuda_home", "already_compiled", "prebuilt"} <= set(d)


def _torch_cuda(monkeypatch, version: str | None) -> None:
    import torch
    monkeypatch.setattr(torch.version, "cuda", version)


# What ``python -m sweep.build --core`` records for the two cores the wheel
# ships: cu12 from nvcc 12.9 (Volta up to Hopper, PTX for Hopper), cu13 from
# nvcc 13 -- which dropped compute capability < 7.5, so no sm_70, and knows
# Blackwell (sm_100, sm_120), with PTX for the newest.
CU12 = dict(cuda="12.9", archs=("70", "75", "80", "86", "89", "90"), ptx=("90",))
CU13 = dict(cuda="13.0", archs=("75", "80", "86", "89", "90", "100", "120"), ptx=("120",))


class TestTwoShippedCores:
    """Two drawers, ``lib/cu12/`` and ``lib/cu13/``; torch's CUDA major picks
    one and the other is never consulted, whatever it would fit -- the CUDA
    runtime a core links must be the one torch brought."""

    @pytest.fixture
    def both(self, pkg) -> dict:
        return {"cu12": _ship(pkg / "lib" / "cu12", _sidecar(**CU12)),
                "cu13": _ship(pkg / "lib" / "cu13", _sidecar(**CU13))}

    def test_torch_cu13_takes_the_cu13_drawer(self, both, monkeypatch):
        _torch_cuda(monkeypatch, "13.0")
        assert _chosen(_jit._shipped_core()) == both["cu13"]
        assert _jit.shipped_core_info()["tag"] == "cu13"

    def test_torch_cu12_takes_the_cu12_drawer(self, both, monkeypatch):
        _torch_cuda(monkeypatch, "12.8")
        assert _chosen(_jit._shipped_core()) == both["cu12"]
        assert _jit.shipped_core_info()["tag"] == "cu12"

    @pytest.mark.parametrize("arch", [(8, 9), (12, 0), (10, 0), (7, 5), (12, 1), (9, 0)])
    def test_cu13_fits_by_sass(self, both, monkeypatch, arch):
        """Ada, Blackwell (sm_120 and sm_100 are SASS entries of the cu13
        list), Turing, Hopper -- and sm_121 through the same-major sm_120
        cubin.  A SASS fit never asks the driver."""
        _torch_cuda(monkeypatch, "13.0")
        _device(monkeypatch, *arch)

        def no_probe():
            raise AssertionError("the driver was probed for a SASS fit")
        monkeypatch.setattr(_jit, "_driver_cuda_version", no_probe)
        assert _chosen(_jit._shipped_core()) == both["cu13"]

    def test_cu13_does_not_reach_volta(self, both, monkeypatch):
        """nvcc 13 cannot emit sm_70, so the cu13 core has no SASS for a V100
        and its PTX (compute_120) is newer than the card: refused, with the
        arch named -- and the cu12 drawer, which would fit, is not consulted."""
        _torch_cuda(monkeypatch, "13.0")
        _device(monkeypatch, 7, 0)
        path, reason = _jit._shipped_core()
        assert path is None
        assert "sm_70" in reason, reason

    @pytest.mark.parametrize("driver, fits", [((12, 8), False), ((12, 9), True),
                                              ((13, 0), True), (None, True)])
    def test_cu12_reaches_blackwell_only_through_ptx_90(self, both, monkeypatch, driver, fits):
        """A cu12 torch on sm_120: no SASS of major 12 in the cu12 list, so
        compute_90 PTX is JIT-ed forward -- by a driver at least as new as the
        nvcc 12.9 that emitted it.  An unknown driver does not block."""
        _torch_cuda(monkeypatch, "12.8")
        _device(monkeypatch, 12, 0)
        monkeypatch.setattr(_jit, "_driver_cuda_version", lambda: driver)
        path, reason = _jit._shipped_core()
        if fits:
            assert path is not None and Path(path).resolve() == both["cu12"], reason
        else:
            assert path is None
            assert "PTX" in reason and "12.8 < core CUDA 12.9" in reason, reason

    def test_only_cu12_shipped_under_torch_cu13_says_so(self, pkg, monkeypatch):
        _ship(pkg / "lib" / "cu12", _sidecar(**CU12))
        _torch_cuda(monkeypatch, "13.0")
        _device(monkeypatch, 8, 9)                     # the cu12 core would fit sm_89
        path, reason = _jit._shipped_core()
        assert path is None
        assert reason.startswith("no shipped core for cu13 ("), reason
        assert "cu13" in reason and reason.endswith(")")

    def test_only_cu13_shipped_under_torch_cu12_is_never_picked(self, pkg, monkeypatch):
        _ship(pkg / "lib" / "cu13", _sidecar(**CU13))
        _torch_cuda(monkeypatch, "12.8")
        _device(monkeypatch, 8, 9)                     # the cu13 core would fit sm_89
        path, reason = _jit._shipped_core()
        assert path is None
        assert reason.startswith("no shipped core for cu12 ("), reason
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        ok, why = _jit.can_compile()                   # ...and nothing else pretends it fits
        assert not ok and "no shipped CUDA core fits (no shipped core for cu12" in why, why

    def test_a_cu12_core_filed_under_cu13_is_refused(self, pkg, monkeypatch):
        """The sidecar, not the drawer name, is what the fit check believes."""
        _ship(pkg / "lib" / "cu13", _sidecar(**CU12))
        _torch_cuda(monkeypatch, "13.0")
        path, reason = _jit._shipped_core()
        assert path is None
        assert reason == "core is CUDA 12.9, torch is CUDA 13.0", reason

    def test_a_cu13_core_filed_under_cu12_is_refused(self, pkg, monkeypatch):
        _ship(pkg / "lib" / "cu12", _sidecar(**CU13))
        _torch_cuda(monkeypatch, "12.8")
        path, reason = _jit._shipped_core()
        assert path is None
        assert reason == "core is CUDA 13.0, torch is CUDA 12.8", reason

    def test_sweep_core_override_is_checked_against_torch_major_too(self, both, monkeypatch, tmp_path):
        mine = _ship(tmp_path / "custom", _sidecar(**CU13))
        monkeypatch.setenv("SWEEP_CORE", str(mine))
        _torch_cuda(monkeypatch, "12.8")
        path, reason = _jit._shipped_core()
        assert path is None and "CUDA 13.0" in reason and "12.8" in reason, reason
        _torch_cuda(monkeypatch, "13.0")
        assert _chosen(_jit._shipped_core()) == mine

    def test_header_message_names_the_wheels_of_torchs_major(self, both, monkeypatch):
        """The compiled shim's header hint: the ``-cu12`` wheels under a cu12
        torch, the unsuffixed CUDA 13 line under a cu13 torch."""
        monkeypatch.setenv("SWEEP_JIT_FULL", "1")
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        monkeypatch.setattr(_jit, "_nvidia_pip_includes", lambda: [])
        _torch_cuda(monkeypatch, "13.0")
        ok, why = _jit.can_compile()
        assert not ok and "pip install nvidia-cuda-runtime nvidia-cuda-nvcc," in why, why
        assert "-cu12" not in why
        _torch_cuda(monkeypatch, "12.8")
        ok, why = _jit.can_compile()
        assert not ok and "pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12," in why, why


class TestPipCudaLayouts:
    """torch's pip CUDA wheels come in two layouts: CUDA 12 is one directory
    per component (``nvidia/cuda_nvcc/bin/nvcc``, ``nvidia/cufft/lib``,
    ``nvidia/cuda_runtime/include``), CUDA 13 one directory per major
    (``nvidia/cu13/{bin,include,lib}``).  The loader must find nvcc, the
    runtime headers and the libs in either, and on a mixed install take the
    nvcc and the cuFFT of the right major."""

    @pytest.fixture
    def site(self, tmp_path, monkeypatch):
        root = tmp_path / "site-packages" / "nvidia"
        root.mkdir(parents=True)
        monkeypatch.setattr(_jit, "_nvidia_pip_roots", lambda: [str(root)])
        monkeypatch.setattr(_jit, "_cuda_home_cache", False)
        for env in ("CUDA_HOME", "CUDA_PATH", "SWEEP_JIT_ALLOW_OLD_CUDA", "TORCH_EXTENSIONS_DIR"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setattr(_jit.shutil, "which", lambda _name: None)     # no nvcc on PATH
        versions: dict[str, tuple[int, int]] = {}
        monkeypatch.setattr(_jit, "_nvcc_version", lambda p: versions.get(str(p)))

        def nvcc(drawer: str, version: tuple[int, int]) -> Path:
            home = root / drawer
            (home / "bin").mkdir(parents=True, exist_ok=True)
            (home / "bin" / "nvcc").write_text("#!/bin/sh\n")
            versions[str(home / "bin" / "nvcc")] = version
            return home
        return root, nvcc

    def test_cu13_layout_nvcc_is_found(self, site, monkeypatch):
        root, nvcc = site
        home = nvcc("cu13", (13, 0))
        _torch_cuda(monkeypatch, "13.0")
        assert _jit._pip_nvcc_homes(13) == [home]
        assert _jit._find_cuda_home() == str(home)

    def test_cu12_layout_nvcc_is_still_found(self, site, monkeypatch):
        root, nvcc = site
        home = nvcc("cuda_nvcc", (12, 9))
        _torch_cuda(monkeypatch, "12.8")
        assert _jit._find_cuda_home() == str(home)

    def test_torchs_major_picks_between_the_layouts(self, site, monkeypatch):
        root, nvcc = site
        cu13 = nvcc("cu13", (13, 0))
        cu12 = nvcc("cuda_nvcc", (12, 9))
        _torch_cuda(monkeypatch, "13.0")
        assert _jit._pip_nvcc_homes(13)[0] == cu13          # asked first, not merely accepted
        assert _jit._find_cuda_home() == str(cu13)
        monkeypatch.setattr(_jit, "_cuda_home_cache", False)
        _torch_cuda(monkeypatch, "12.8")
        assert _jit._pip_nvcc_homes(12)[0] == cu12
        assert _jit._find_cuda_home() == str(cu12)

    def test_the_other_majors_pip_nvcc_is_refused(self, site, monkeypatch):
        """A cu13 nvcc must not build a core for a cu12 torch, nor the reverse:
        the core links the CUDA runtime torch brought."""
        root, nvcc = site
        cu13 = nvcc("cu13", (13, 0))
        _torch_cuda(monkeypatch, "12.8")
        assert _jit._find_cuda_home() is None
        monkeypatch.setattr(_jit, "_cuda_home_cache", False)
        (cu13 / "bin" / "nvcc").unlink()
        nvcc("cuda_nvcc", (12, 9))
        _torch_cuda(monkeypatch, "13.0")
        assert _jit._find_cuda_home() is None

    def test_tag_major(self):
        assert _jit._tag_major("cu13") == 13
        assert _jit._tag_major("cu12") == 12
        assert _jit._tag_major("cuda_nvcc") is None
        assert _jit._tag_major("cufft") is None

    def test_cu13_layout_headers_and_libs(self, site):
        """cuda_runtime_api.h and crt/host_defines.h come from two wheels under
        CUDA 12 but land in the one ``nvidia/cu13/include`` under CUDA 13; the
        lib glob reaches ``nvidia/cu13/lib``; a pip CUDA_HOME of that layout
        has its include dir where a toolkit would."""
        root, _ = site
        inc = root / "cu13" / "include"
        (inc / "crt").mkdir(parents=True)
        (root / "cu13" / "lib").mkdir()
        assert _jit._pip_cuda_runtime_headers() is False
        (inc / "cuda_runtime_api.h").write_text("")
        assert _jit._pip_cuda_runtime_headers() is False      # crt/host_defines.h still missing
        (inc / "crt" / "host_defines.h").write_text("")
        assert _jit._pip_cuda_runtime_headers() is True
        assert _jit._nvidia_pip_includes() == [str(inc)]
        assert _jit._nvidia_pip_libs() == [str(root / "cu13" / "lib")]
        assert _jit._toolkit_includes(str(root / "cu13")) == [str(inc)]

    def test_cu12_layout_headers_span_two_wheels(self, site):
        root, _ = site
        rt = root / "cuda_runtime" / "include"
        rt.mkdir(parents=True)
        (rt / "cuda_runtime_api.h").write_text("")
        assert _jit._pip_cuda_runtime_headers() is False
        crt = root / "cuda_nvcc" / "include" / "crt"
        crt.mkdir(parents=True)
        (crt / "host_defines.h").write_text("")
        assert _jit._pip_cuda_runtime_headers() is True

    @pytest.mark.parametrize("major, soname", [(12, "libcufft.so.11"), (13, "libcufft.so.12"),
                                               (11, None), (None, None)])
    def test_cufft_soname_trails_the_cuda_major_by_one(self, major, soname):
        assert _jit._cufft_soname(major) == soname

    def test_cufft_link_flag_takes_the_toolkits_major(self, site):
        """A mixed install has both cuFFT runtimes; the link line names the
        one of the nvcc doing the build, never the shortest name around."""
        root, nvcc = site
        for drawer, so in (("cufft", "libcufft.so.11"), ("cu13", "libcufft.so.12")):
            (root / drawer / "lib").mkdir(parents=True, exist_ok=True)
            (root / drawer / "lib" / so).write_bytes(b"")
        cu13 = nvcc("cu13", (13, 0))
        cu12 = nvcc("cuda_nvcc", (12, 9))
        assert _jit._cufft_link_flag(str(cu13)) == "-l:libcufft.so.12"
        assert _jit._cufft_link_flag(str(cu12)) == "-l:libcufft.so.11"

    def test_cufft_link_flag_falls_back_to_torchs_major_then_any(self, site, monkeypatch):
        root, _ = site
        (root / "cu13" / "lib").mkdir(parents=True)
        (root / "cu13" / "lib" / "libcufft.so.12").write_bytes(b"")
        _torch_cuda(monkeypatch, "13.0")                       # nvcc unreadable -> torch's major
        assert _jit._cufft_link_flag(str(root / "nowhere")) == "-l:libcufft.so.12"
        _torch_cuda(monkeypatch, None)                         # nothing known -> whatever is there
        assert _jit._cufft_link_flag(str(root / "nowhere")) == "-l:libcufft.so.12"
        (root / "cu13" / "lib" / "libcufft.so").write_bytes(b"")
        assert _jit._cufft_link_flag(str(root / "nowhere")) == "-lcufft"

    def test_preload_cufft_loads_the_soname_of_torchs_major(self, site, monkeypatch):
        import ctypes
        root, _ = site
        for drawer, so in (("cufft", "libcufft.so.11"), ("cu13", "libcufft.so.12")):
            (root / drawer / "lib").mkdir(parents=True, exist_ok=True)
            (root / drawer / "lib" / so).write_bytes(b"")
        loaded = []
        monkeypatch.setattr(ctypes, "CDLL", lambda path, mode=None: loaded.append(path))
        _torch_cuda(monkeypatch, "13.0")
        _jit._preload_cufft(None)
        assert loaded == [str(root / "cu13" / "lib" / "libcufft.so.12")]
        loaded.clear()
        _torch_cuda(monkeypatch, "12.8")
        _jit._preload_cufft(None)
        assert loaded == [str(root / "cufft" / "lib" / "libcufft.so.11")]


class TestCudaThirteenArchFloor:
    """nvcc 13 dropped offline compilation for compute capability < 7.5: a
    local core build for a V100 under a cu13 torch can never succeed, so
    ``can_compile()`` says so before any toolkit is looked for -- and names
    the way out.  Nothing changes under CUDA 12."""

    def test_target_archs_spelling(self, pkg, monkeypatch):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0;9.0+PTX, 12.0")
        assert _jit._target_archs() == [("70", False), ("90", True), ("120", False)]
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST")
        _device(monkeypatch, 8, 9)
        assert _jit._target_archs() == [("89", False)]

    def test_volta_in_the_arch_list_is_refused_under_cu13(self, pkg, monkeypatch):
        _torch_cuda(monkeypatch, "13.0")
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0;8.0")
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda-13")
        ok, why = _jit.can_compile()
        assert not ok
        assert "sm_70" in why and "CUDA 13" in why and "torch built for CUDA 12" in why, why

    def test_a_visible_volta_is_refused_under_cu13(self, pkg, monkeypatch):
        _torch_cuda(monkeypatch, "13.0")
        _device(monkeypatch, 7, 0)
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)   # a toolkit would not help
        ok, why = _jit.can_build()
        assert not ok and "sm_70" in why and "CUDA toolkit" not in why, why

    def test_turing_and_up_pass_the_floor(self, pkg, monkeypatch):
        _torch_cuda(monkeypatch, "13.0")
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX")
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda-13")
        assert _jit._arch_floor_reason() == ""
        ok, why = _jit.can_compile()
        assert ok, why

    def test_cu12_keeps_volta(self, pkg, monkeypatch):
        _torch_cuda(monkeypatch, "12.8")
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda-12")
        assert _jit._arch_floor_reason() == ""
        ok, why = _jit.can_compile()
        assert ok, why

    def test_toolkit_message_names_torchs_major(self, pkg, monkeypatch):
        _torch_cuda(monkeypatch, "13.0")
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        ok, why = _jit.can_compile()
        assert not ok and "CUDA toolkit" in why and "an nvcc of CUDA 13" in why, why


class TestBlackwellNeedsNvcc128:
    """The cu12 core reaches Blackwell only through its compute_90 PTX, which
    a driver older than the nvcc 12.9 that emitted it refuses
    (TestPtxDriverGate); that user is sent to a local build, and an nvcc
    12.4-12.7 -- past the bf16 floor -- cannot emit sm_100/sm_120 (CUDA 12.8
    introduced them).  Refused up front, naming the toolkit and the version,
    rather than mid-build with nvcc's own message."""

    @pytest.fixture
    def blackwell(self, pkg, monkeypatch) -> Path:
        _ship(pkg / "lib" / "cu12", _sidecar(**CU12))
        _torch_cuda(monkeypatch, "12.8")
        _device(monkeypatch, 12, 0)
        monkeypatch.setattr(_jit, "_driver_cuda_version", lambda: (12, 8))   # < the core's 12.9
        path, reason = _jit._shipped_core()
        assert path is None and "PTX" in reason, reason
        return pkg

    @staticmethod
    def _nvcc(monkeypatch, version) -> None:
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda-x")
        monkeypatch.setattr(_jit, "_nvcc_version",
                            lambda p: version if str(p) == "/opt/cuda-x/bin/nvcc" else None)

    def test_an_nvcc_before_12_8_is_refused_by_name(self, blackwell, monkeypatch):
        self._nvcc(monkeypatch, (12, 6))
        ok, why = _jit.can_compile()
        assert not ok
        assert why.startswith("no shipped CUDA core fits (device sm_120 fits only via PTX"), why
        assert "/opt/cuda-x, CUDA 12.6" in why and "sm_120" in why and "12.0 needs CUDA >= 12.8" in why, why

    @pytest.mark.parametrize("version", [(12, 8), (12, 9)])
    def test_12_8_and_up_pass(self, blackwell, monkeypatch, version):
        self._nvcc(monkeypatch, version)
        assert _jit.can_compile() == (True, "ok")

    def test_without_a_toolkit_the_message_names_the_version(self, blackwell, monkeypatch):
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        ok, why = _jit.can_compile()
        assert not ok and "an nvcc of CUDA 12, your torch's CUDA major, and >= 12.8 for sm_120" in why, why

    def test_an_unreadable_nvcc_does_not_block(self, blackwell, monkeypatch):
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda-x")
        monkeypatch.setattr(_jit, "_nvcc_version", lambda p: None)
        assert _jit.can_compile() == (True, "ok")

    def test_a_pre_blackwell_target_is_not_bound(self, pkg, monkeypatch):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.9")
        self._nvcc(monkeypatch, (12, 4))
        assert _jit._nvcc_floor_for_targets() is None
        assert _jit.can_compile() == (True, "ok")

    def test_the_arch_list_is_bound_too(self, pkg, monkeypatch):
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0;10.0+PTX")
        self._nvcc(monkeypatch, (12, 6))
        assert _jit._nvcc_floor_for_targets() == ((12, 8), "100")
        ok, why = _jit.can_compile()
        assert not ok and "sm_100" in why and "10.0 needs CUDA >= 12.8" in why, why

    def test_a_cu13_nvcc_is_past_the_bound(self, pkg, monkeypatch):
        _torch_cuda(monkeypatch, "13.0")
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "12.0")
        self._nvcc(monkeypatch, (13, 0))
        assert _jit.can_compile() == (True, "ok")


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
