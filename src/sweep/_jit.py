"""Compile sweep's CUDA/C++ backend against the *user's* torch, on first use.

This is why a single ``py3-none`` wheel of sweep works with **any** torch version
and any Python 3: the compiled extension (``sweep._C``) is not shipped pre-built —
it is JIT-compiled at runtime via ``torch.utils.cpp_extension.load()`` against
whatever libtorch is currently imported, then cached. First use of ``impl='c'``
pays a one-time ~2-5 min compile (only for *this* machine's GPU arch); every run
after that loads the cached ``.so`` instantly.

The C++ sources ship inside the wheel under ``sweep/csrc/`` (package data).
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_CSRC = _PKG / "csrc"

_module = None          # cached compiled module (process-local)

# Windows is a *build-flag* platform difference, not a source one: torch's
# cpp_extension already supplies /std:c++17, /MD, /EHsc and the MSVC warning
# suppressions there (COMMON_MSVC_FLAGS), and passes whatever we hand it through
# to cl.exe / link.exe verbatim. So every GNU spelling below must be branched.
_WIN = sys.platform == "win32"


def _lib_ext() -> str:
    """Suffix torch gives the built module (``cpp_extension.LIB_EXT``)."""
    return ".pyd" if _WIN else ".so"


def _nvcc_name() -> str:
    return "nvcc.exe" if _WIN else "nvcc"


# --------------------------------------------------------------------------- #
# CUDA toolkit (nvcc) discovery
# --------------------------------------------------------------------------- #
def _nvidia_pip_includes() -> list[str]:
    """Every ``nvidia/*/include`` dir from the pip CUDA wheels torch pulls in
    (cuda_runtime, cusparse, cublas, cudnn, …) — so nvcc/host cc find the headers
    even when there is no system CUDA toolkit."""
    incs: list[str] = []
    try:
        import nvidia  # namespace package from nvidia-*-cu12 wheels
    except Exception:
        return incs
    for base in getattr(nvidia, "__path__", []):
        for inc in sorted(glob.glob(os.path.join(base, "*", "include"))):
            incs.append(inc)
    return incs


def _torch_cuda_major() -> int | None:
    try:
        import torch
        v = torch.version.cuda            # e.g. "12.8"
        return int(v.split(".")[0]) if v else None
    except Exception:
        return None


def _nvcc_version(nvcc: str):
    import re
    import subprocess
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True,
                             text=True, timeout=20).stdout
        m = re.search(r"release (\d+)\.(\d+)", out)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None


_cuda_home_cache = False   # False = not computed; None/str = computed result


def _find_cuda_home() -> str | None:
    """Return a CUDA_HOME (dir with bin/nvcc) whose CUDA **major matches the
    user's torch**. Priority: explicit CUDA_HOME env, then the pip
    ``nvidia-cuda-nvcc-cu12`` wheel (always cu12, what our dep pulls), then nvcc
    on PATH — each version-checked so an old system nvcc (e.g. CUDA 10.1 in
    /usr/bin) is skipped rather than used and failing mid-compile."""
    global _cuda_home_cache
    if _cuda_home_cache is not False:
        return _cuda_home_cache

    want = _torch_cuda_major()
    allow_old = os.environ.get("SWEEP_JIT_ALLOW_OLD_CUDA", "").strip().lower() \
        in ("1", "true", "yes", "on")

    def match(nvcc: Path) -> bool:
        if not nvcc.exists():
            return False
        v = _nvcc_version(str(nvcc))
        if v is None:
            return False
        maj, minr = v
        if want is not None and maj != want:
            return False
        # Floor: nvcc 12.4. CUDA 12.0-12.5 ship a <cuda/std> bf16 header whose
        # host-device isnan/isinf call __device__-only half intrinsics; torch's
        # build defines (-D__CUDA_NO_BFLOAT16_CONVERSIONS__ ...) plus
        # --expt-relaxed-constexpr neutralize it from 12.4 up (verified: a clean
        # 12.4 toolkit compiles the whole tree). 12.0-12.3 are untested here, so
        # the guard rejects them; SWEEP_JIT_ALLOW_OLD_CUDA=1 tries one anyway.
        return allow_old or not (maj == 12 and minr < 4)

    result = None
    # 1. explicit env (respect user config, but only if it matches torch's CUDA)
    for env in ("CUDA_HOME", "CUDA_PATH"):
        h = os.environ.get(env)
        if h and match(Path(h) / "bin" / _nvcc_name()):
            result = h
            break
    # 2. pip nvidia-cuda-nvcc-cu12 (namespace pkg -> __path__; guaranteed cu12)
    if result is None:
        try:
            import nvidia.cuda_nvcc as _n  # type: ignore
            for base in getattr(_n, "__path__", []):
                if match(Path(base) / "bin" / _nvcc_name()):
                    result = str(Path(base))
                    break
        except Exception:
            pass
    # 3. nvcc on PATH (version-checked -> skips old /usr/bin/nvcc)
    if result is None:
        p = shutil.which("nvcc")
        if p and match(Path(p)):
            result = str(Path(p).resolve().parent.parent)

    _cuda_home_cache = result
    return result


def _nvidia_pip_libs() -> list[str]:
    """``nvidia/*/lib`` dirs so the JIT link step finds libcudart etc. when there
    is no system CUDA toolkit (provided by torch's pip CUDA wheels)."""
    libs: list[str] = []
    try:
        import nvidia
    except Exception:
        return libs
    # The Windows cu12 wheels keep the linker's import libraries one level
    # deeper (``lib/x64``); their plain ``lib/`` holds no .lib at all.
    pats = [("*", "lib", "x64"), ("*", "lib")] if _WIN else [("*", "lib")]
    for base in getattr(nvidia, "__path__", []):
        for pat in pats:
            for lib in sorted(glob.glob(os.path.join(base, *pat))):
                libs.append(lib)
    return libs


def _ensure_ninja_on_path() -> None:
    """torch checks ``ninja --version`` on PATH (not the bundled python pkg)."""
    if shutil.which("ninja"):
        return
    try:
        import ninja  # the pip 'ninja' package exposes BIN_DIR
        bindir = getattr(ninja, "BIN_DIR", None)
        if bindir and os.path.isdir(bindir):
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def toolchain_ready() -> tuple[bool, str]:
    """(usable, reason) — torch and a suitable nvcc are present, i.e. the backend
    can be **compiled** here. Deliberately says nothing about a GPU: nvcc needs
    none to emit code for an arch you name (``TORCH_CUDA_ARCH_LIST``), which is
    what makes a build-only box — a CI runner, a Windows VM, an HPC login node —
    usable for shaking out build problems. Use :func:`can_build` when the result
    also has to *run* here."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "PyTorch is not installed"
    if _find_cuda_home() is None:
        return False, (
            "no suitable CUDA toolkit found (need nvcc >=12.4 matching your "
            "torch's CUDA major — 12.0-12.3 ship a broken <cuda/std> bf16 header). "
            "sweep compiles its GPU backend on first use; provide a recent nvcc "
            "via `module load cuda`, a system CUDA Toolkit, or "
            "`conda install -c nvidia cuda-toolkit`. To try an older toolkit "
            "anyway, set SWEEP_JIT_ALLOW_OLD_CUDA=1)")
    return True, "ok"


def can_build() -> tuple[bool, str]:
    """(usable, reason) — True when torch+CUDA GPU+nvcc are present so the C
    backend can be JIT-compiled *and loaded*. Does NOT compile. Used by
    ``sweep.is_torch_binding_available()`` to avoid a surprise compile."""
    try:
        import torch
    except Exception:
        return False, "PyTorch is not installed"
    if not torch.cuda.is_available():
        return False, (
            "no CUDA GPU is visible (to compile anyway — a build-only VM, a CI "
            "runner, a login node — name the target arch, e.g. "
            "TORCH_CUDA_ARCH_LIST=8.9)")
    return toolchain_ready()


# --------------------------------------------------------------------------- #
# compile flags
# --------------------------------------------------------------------------- #
def _compile_flags() -> tuple[list[str], list[str], list[str]]:
    """``(cflags, cuda_cflags, ldflags)`` to add on top of torch's own.

    --expt-relaxed-constexpr: lets constexpr __host__ funcs call __device__
    ones, which some CUDA toolkits' <cuda/std> bf16 headers (e.g. 12.4's
    nvbf16.h) require to compile. Harmless on toolkits that don't need it.

    The POSIX lists are pinned by ``test_build_flags_platform.py``: they feed
    nvcc/gcc codegen, so editing them moves every bit-exact baseline. Windows
    fixes belong in the ``_WIN`` branch only.
    """
    lib_dirs = _nvidia_pip_libs()
    if _WIN:
        return (
            ["/O2"],
            ["-O3", "--use_fast_math", "--expt-relaxed-constexpr",
             "-Xcompiler=/wd4996",           # MSVC's -Wno-deprecated-declarations
             # nvcc's cudafe pass rewrites the translation unit in a way that
             # makes ``::std`` ambiguous against <valarray> inside torch's
             # compiled_autograd.h -- "error C2872: 'std': ambiguous symbol",
             # which kills every CUDA extension build, sweep's or anyone's.
             # /permissive- is MSVC's conformance mode (proper two-phase name
             # lookup), so this is the standards-correct reading, not a mute.
             "-Xcompiler=/permissive-"],
            [f"/LIBPATH:{d}" for d in lib_dirs],
        )
    return (
        ["-O3", "-Wno-attributes", "-fopenmp"],
        ["-O3", "--use_fast_math", "--expt-relaxed-constexpr",
         "-Xcompiler=-Wno-deprecated-declarations"],
        ["-fopenmp"] + [f"-L{d}" for d in lib_dirs],
    )


# --------------------------------------------------------------------------- #
# source staging (dedupe object basenames)
# --------------------------------------------------------------------------- #
def _sources() -> list[str]:
    """C++/CUDA sources, mirroring build_config.get_sources(). CUDA-only by
    default (fast first compile, what GPU users need); set SWEEP_JIT_FULL=1 to
    also compile the heavy CPU C++ tree."""
    cu = (glob.glob(str(_CSRC / "cuda/common/**/*.cu"), recursive=True)
          + glob.glob(str(_CSRC / "cuda/equations/**/*.cu"), recursive=True))
    binding = [str(_CSRC / "bindings/module.cpp")]
    if os.environ.get("SWEEP_JIT_FULL", "").lower() in ("1", "true", "yes", "on"):
        cpu = [s for s in glob.glob(str(_CSRC / "cpu/**/*.cpp"), recursive=True)
               if not s.endswith("cpu_binding_stub.cpp")]
    else:
        cpu = [str(_CSRC / "cpu/cpu_binding_stub.cpp")]
    return cpu + cu + binding


def _stage(build_dir: Path) -> tuple[list[str], list[str]]:
    """cpp_extension.load() flattens object names by basename; sweep has many
    forward.cu / backward.cu / kernels.cu. Copy csrc into a version-stamped
    staging dir with UNIQUE compiled-source basenames (renamed in place so their
    relative #includes still resolve). Idempotent across runs."""
    try:
        from importlib.metadata import version
        _ver = version("sweep-solver")
    except Exception:
        _ver = "dev"
    stage = build_dir / f"csrc_stage_{_ver}"
    done = stage / ".staged"
    if not done.exists():
        shutil.rmtree(stage, ignore_errors=True)
        shutil.copytree(_CSRC, stage)
        for s in _sources():
            rel = Path(s).resolve().relative_to(_CSRC)
            slug = "_".join(rel.with_suffix("").parts)
            os.replace(stage / rel, stage / rel.parent / (slug + rel.suffix))
        done.write_text("ok")
    staged = []
    for s in _sources():
        rel = Path(s).resolve().relative_to(_CSRC)
        slug = "_".join(rel.with_suffix("").parts)
        staged.append(str(stage / rel.parent / (slug + rel.suffix)))
    inc = [str(stage), str(stage / "bindings"), str(stage / "shared"),
           str(stage / "cuda"), str(stage / "cuda/common"), str(stage / "cuda/equations")]
    return staged, inc


def _will_build(build_dir: Path) -> bool:
    """Whether the next load() will actually *compile* (vs reuse the cached .so).

    A ``sweep_C.so`` can exist yet still be rebuilt — e.g. after the user upgrades
    torch, whose changed headers make ninja re-link — so "the .so exists" is not a
    reliable signal. Ask ninja (``-n`` dry run) whether any target is stale. This
    drives the one-time "compiling…" notice + verbose output, so a genuine rebuild
    is never a silent 2-5 min hang that looks frozen. When we can't tell, assume a
    build so the user always sees *something*."""
    so = build_dir / ("sweep_C" + _lib_ext())
    ninja_file = build_dir / "build.ninja"
    if not so.exists() or not ninja_file.exists():
        return True                       # never built (no .so / no ninja graph yet)
    _ensure_ninja_on_path()
    ninja = shutil.which("ninja")
    if ninja is None:
        return True                       # can't check -> assume yes (never hang silently)
    try:
        import subprocess
        r = subprocess.run([ninja, "-n"], cwd=str(build_dir),
                           capture_output=True, text=True, timeout=30)
        return "no work to do" not in (r.stdout + r.stderr)
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
def _prepare_toolchain() -> str:
    """Point the environment at the nvcc we picked and return its CUDA_HOME."""
    cuda_home = _find_cuda_home()
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    _ensure_ninja_on_path()
    return cuda_home


def _cuda_includes(cuda_home: str) -> list[str]:
    """Use ONLY the selected CUDA toolkit's own headers (version-consistent with
    its nvcc). Do NOT mix in the pip nvidia-*/include dirs: for a torch built
    against an older CUDA (torch 2.5 = cu121 -> 12.1 headers) those clash with a
    newer toolkit and break the <cuda/std> bf16 compile."""
    return [p for p in (os.path.join(cuda_home, "include"),
                        os.path.join(cuda_home, "targets", "x86_64-linux", "include"))
            if os.path.isdir(p)]


_CANARY_CU = r"""// Generated by sweep._jit.canary(). Two files, same toolchain as the real
// backend: nvcc + host compiler + pybind + the cudart link.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__global__ void sweep_canary_kernel(float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __nv_bfloat16 h = __float2bfloat16(2.0f);   // the <cuda/std> bf16 path
        out[i] = __bfloat162float(h) * static_cast<float>(i);
    }
}

torch::Tensor sweep_canary_run(int64_t n) {
    auto out = torch::zeros({n},
        torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const int threads = 128;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    sweep_canary_kernel<<<blocks, threads>>>(out.data_ptr<float>(),
                                             static_cast<int>(n));
    // Real cudart calls, so the linker has to resolve the CUDA runtime rather
    // than quietly producing a module that fails at import.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sweep canary kernel launch failed: ",
                cudaGetErrorString(err));
    return out;
}
"""

_CANARY_CPP = r"""// Generated by sweep._jit.canary(). Host-compiler side of the probe.
#include <torch/extension.h>
#include "wavetypes.h"   // real sweep header, through the host compiler

torch::Tensor sweep_canary_run(int64_t n);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &sweep_canary_run, "sweep build canary");
    // Proof the sweep headers parsed, not just that they were found.
    py::class_<ForwardInput>(m, "ForwardInput").def(py::init<>());
}
"""


def canary(verbose: bool = True):
    """Compile a **two-file** probe with the same toolchain and flags as the real
    backend. Returns the loaded probe module.

    The real backend is 49 translation units of a couple of CPU-minutes each, so
    a full ``precompile()`` is a poor debugging loop. Every build problem we have
    actually hit — a flag the host compiler rejects, a library search path the
    linker ignores, a missing cudart, a header the toolkit can't parse — shows up
    on two files in wall-clock seconds. Iterate here, then run the real
    ``precompile()`` once this is green."""
    from torch.utils import cpp_extension

    arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    ok, why = toolchain_ready() if arch else can_build()
    if not ok:
        raise RuntimeError(f"sweep cannot build here: {why}")

    cuda_home = _prepare_toolchain()
    build_dir = Path(cpp_extension._get_build_directory("sweep_C_canary", verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    cu, cpp = build_dir / "sweep_canary.cu", build_dir / "sweep_canary_binding.cpp"
    for path, text in ((cu, _CANARY_CU), (cpp, _CANARY_CPP)):
        if not path.exists() or path.read_text() != text:
            path.write_text(text)

    cflags, cuda_cflags, ldflags = _compile_flags()
    if verbose:
        print(f"[sweep] canary: 2-file build probe in {build_dir}", file=sys.stderr,
              flush=True)
    return cpp_extension.load(
        name="sweep_C_canary",
        sources=[str(cpp), str(cu)],
        extra_include_paths=[str(_CSRC), str(_CSRC / "shared"),
                             str(_CSRC / "bindings")] + _cuda_includes(cuda_home),
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        build_directory=str(build_dir),
        verbose=verbose,
    )


def load():
    """Compile (first call, cached) and return the ``sweep._C`` module."""
    global _module
    if _module is not None:
        return _module

    import torch
    from torch.utils import cpp_extension

    # An explicit arch is exactly what a visible GPU would have told us, so it
    # also unlocks compiling where none is visible. Without one we still demand a
    # GPU: guessing the arch yields a module that loads and then dies with "no
    # kernel image is available" at the first launch.
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    ok, why = toolchain_ready() if arch else can_build()
    if not ok:
        raise RuntimeError(
            f"sweep's compiled backend (impl='c') is unavailable: {why}. "
            "Use impl='eager' for a pure-Python (slower) CPU/GPU path.")

    cuda_home = _prepare_toolchain()

    build_dir = Path(cpp_extension._get_build_directory("sweep_C", verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    sources, inc = _stage(build_dir)
    # Use ONLY the selected CUDA toolkit's own headers (version-consistent with
    # its nvcc). Do NOT mix in the pip nvidia-*/include dirs: for a torch built
    # against an older CUDA (torch 2.5 = cu121 -> 12.1 headers) those clash with a
    # newer toolkit and break the <cuda/std> bf16 compile.
    inc = inc + _cuda_includes(cuda_home)

    if arch:
        target = f"arch {arch}"
    else:
        cap = torch.cuda.get_device_capability()
        target = f"your GPU (sm_{cap[0]}{cap[1]})"
    building = _will_build(build_dir)
    if building:
        print(f"[sweep] compiling the CUDA backend for {target} — "
              f"one-time, ~2-5 min, then cached at {build_dir} ...",
              file=sys.stderr, flush=True)

    cflags, cuda_cflags, ldflags = _compile_flags()
    _module = cpp_extension.load(
        name="sweep_C",
        sources=sources,
        extra_include_paths=inc,
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_cflags,
        extra_ldflags=ldflags,
        build_directory=str(build_dir),
        verbose=building,
    )
    if building:
        print("[sweep] CUDA backend compiled and cached.", file=sys.stderr, flush=True)
    return _module
