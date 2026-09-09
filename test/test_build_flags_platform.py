"""Build flags must be spelled for the platform's compiler — and the POSIX
spellings must not drift while that is being fixed.

The regression this guards: every flag sweep added was a GNU spelling, hardcoded.
``torch.utils.cpp_extension`` hands ``extra_cflags`` / ``extra_ldflags`` to the
host toolchain verbatim (it supplies only ``/std:c++17 /MD /EHsc`` of its own on
MSVC), so on Windows ``-Wno-attributes`` reached cl.exe and ``-L<dir>`` reached
link.exe — where it is not an error but a *silently ignored* option, so the pip
CUDA wheels' import libraries were simply never searched. ``sweep.precompile()``
could not succeed on Windows at all.

The other half is the reverse guarantee. The POSIX lists feed gcc/nvcc codegen,
so touching them moves every bit-exact baseline in the gate suite. They are
pinned literally below, so a Windows fix cannot quietly alter the Linux build.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import build_config                                    # noqa: E402
from sweep import _jit                                 # noqa: E402


# What the Linux/macOS build has always passed. Pinned, not derived.
POSIX_CFLAGS = ["-O3", "-Wno-attributes", "-fopenmp"]
POSIX_CUDA_CFLAGS = ["-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                     "-Xcompiler=-Wno-deprecated-declarations"]

# Spellings only a GNU-style driver accepts. cl.exe / link.exe either reject
# these outright or ignore them, which is the failure mode that cost us the
# CUDA library search path.
GNU_ONLY = ("-O", "-W", "-f", "-L", "-l", "-I", "-std=", "-pthread", "-shared")


def gnu_spellings(flags):
    return [f for f in flags if f.startswith(GNU_ONLY)]


@pytest.fixture
def fake_pip_cuda(monkeypatch, tmp_path):
    """A stand-in for the ``nvidia-*-cu12`` wheels, laid out both ways."""
    base = tmp_path / "nvidia"
    (base / "cuda_runtime" / "lib" / "x64").mkdir(parents=True)
    stub = types.ModuleType("nvidia")
    stub.__path__ = [str(base)]
    monkeypatch.setitem(sys.modules, "nvidia", stub)
    return base / "cuda_runtime" / "lib"


class TestJitFlags:
    """``_jit._compile_flags()`` — the JIT path behind ``sweep.precompile()``."""

    def test_posix_flags_are_pinned(self, monkeypatch, fake_pip_cuda):
        monkeypatch.setattr(_jit, "_WIN", False)
        cflags, cuda_cflags, ldflags = _jit._compile_flags()
        assert cflags == POSIX_CFLAGS
        assert cuda_cflags == POSIX_CUDA_CFLAGS
        assert ldflags == ["-fopenmp", f"-L{fake_pip_cuda}"]

    def test_windows_host_flags_carry_no_gnu_spellings(self, monkeypatch, fake_pip_cuda):
        monkeypatch.setattr(_jit, "_WIN", True)
        cflags, _, _ = _jit._compile_flags()
        assert gnu_spellings(cflags) == []
        assert "/O2" in cflags, "the host compiler must still be told to optimise"

    def test_windows_linker_gets_libpath_not_dash_l(self, monkeypatch, fake_pip_cuda):
        monkeypatch.setattr(_jit, "_WIN", True)
        _, _, ldflags = _jit._compile_flags()
        assert gnu_spellings(ldflags) == []
        assert any(f.startswith("/LIBPATH:") for f in ldflags)

    def test_windows_xcompiler_payloads_are_msvc(self, monkeypatch, fake_pip_cuda):
        """nvcc's own flags stay GNU-ish everywhere, but whatever rides through
        ``-Xcompiler`` lands on cl.exe and must be spelled for it."""
        monkeypatch.setattr(_jit, "_WIN", True)
        _, cuda_cflags, _ = _jit._compile_flags()
        payloads = [f.split("=", 1)[1] for f in cuda_cflags
                    if f.startswith("-Xcompiler=")]
        assert payloads, "the -Xcompiler warning suppression should survive"
        assert gnu_spellings(payloads) == []


class TestPlatformNames:
    """Names torch derives from the platform, which sweep has to agree with."""

    @pytest.mark.parametrize("win,ext", [(True, ".pyd"), (False, ".so")])
    def test_lib_ext(self, monkeypatch, win, ext):
        monkeypatch.setattr(_jit, "_WIN", win)
        assert _jit._lib_ext() == ext

    @pytest.mark.parametrize("win,name", [(True, "nvcc.exe"), (False, "nvcc")])
    def test_nvcc_name(self, monkeypatch, win, name):
        """``_find_cuda_home`` probes ``<home>/bin/<nvcc>`` with ``.exists()``,
        so the bare name never matched a Windows toolkit — CUDA_PATH, which the
        Windows installer always sets, was silently skipped."""
        monkeypatch.setattr(_jit, "_WIN", win)
        assert _jit._nvcc_name() == name

    def test_windows_pip_cuda_libdirs_reach_x64(self, monkeypatch, fake_pip_cuda):
        """The Windows wheels keep import libraries in ``lib/x64``; the plain
        ``lib/`` the POSIX glob finds holds no ``.lib`` for the linker."""
        monkeypatch.setattr(_jit, "_WIN", True)
        assert str(fake_pip_cuda / "x64") in _jit._nvidia_pip_libs()

    def test_posix_pip_cuda_libdirs_stay_flat(self, monkeypatch, fake_pip_cuda):
        monkeypatch.setattr(_jit, "_WIN", False)
        assert _jit._nvidia_pip_libs() == [str(fake_pip_cuda)]


class TestCompileWithoutGpu:
    """Compiling and running are separate capabilities; nvcc needs no GPU."""

    def test_toolchain_ready_ignores_the_gpu(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda")
        ok, why = _jit.toolchain_ready()
        assert ok, why

    def test_can_build_still_demands_a_gpu(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/opt/cuda")
        ok, why = _jit.can_build()
        assert not ok
        assert "TORCH_CUDA_ARCH_LIST" in why, \
            "the refusal has to name the way out, or a build-only box is a dead end"

    def test_missing_nvcc_fails_both(self, monkeypatch):
        import torch
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
        assert not _jit.toolchain_ready()[0]
        assert not _jit.can_build()[0]


class TestAotFlags:
    """``build_config`` — the ``SWEEP_BUILD_CUDA=1 pip install`` path."""

    def test_posix_flags_are_pinned(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        assert build_config.host_compile_flags() == POSIX_CFLAGS
        assert build_config.nvcc_host_flags() == \
            ["-Xcompiler=-Wno-deprecated-declarations"]
        assert build_config.link_flags() == \
            ["-fopenmp", "-Wl,-rpath,$ORIGIN/../torch/lib"]

    def test_windows_flags_carry_no_gnu_spellings(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert gnu_spellings(build_config.host_compile_flags()) == []
        payloads = [f.split("=", 1)[1] for f in build_config.nvcc_host_flags()]
        assert gnu_spellings(payloads) == []

    def test_windows_link_flags_drop_the_rpath(self, monkeypatch):
        """``-Wl,-rpath,$ORIGIN/...`` has no meaning to link.exe; Windows
        resolves the torch DLLs through PATH instead."""
        monkeypatch.setattr(sys, "platform", "win32")
        flags = build_config.link_flags()
        assert not any("rpath" in f for f in flags)
        assert gnu_spellings(flags) == []
