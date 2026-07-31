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
        if h and match(Path(h) / "bin" / "nvcc"):
            result = h
            break
    # 2. pip nvidia-cuda-nvcc-cu12 (namespace pkg -> __path__; guaranteed cu12)
    if result is None:
        try:
            import nvidia.cuda_nvcc as _n  # type: ignore
            for base in getattr(_n, "__path__", []):
                if match(Path(base) / "bin" / "nvcc"):
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
    for base in getattr(nvidia, "__path__", []):
        for lib in sorted(glob.glob(os.path.join(base, "*", "lib"))):
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


def can_build() -> tuple[bool, str]:
    """(usable, reason) — True when torch+CUDA GPU+nvcc are present so the C
    backend can be JIT-compiled. Does NOT compile. Used by
    ``sweep.is_torch_binding_available()`` to avoid a surprise compile."""
    try:
        import torch
    except Exception:
        return False, "PyTorch is not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA GPU is visible"
    if _find_cuda_home() is None:
        return False, (
            "no suitable CUDA toolkit found (need nvcc >=12.4 matching your "
            "torch's CUDA major — 12.0-12.3 ship a broken <cuda/std> bf16 header). "
            "sweep compiles its GPU backend on first use; provide a recent nvcc "
            "via `module load cuda`, a system CUDA Toolkit, or "
            "`conda install -c nvidia cuda-toolkit`. To try an older toolkit "
            "anyway, set SWEEP_JIT_ALLOW_OLD_CUDA=1)")
    return True, "ok"


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
    so = build_dir / "sweep_C.so"
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
def load():
    """Compile (first call, cached) and return the ``sweep._C`` module."""
    global _module
    if _module is not None:
        return _module

    import torch
    from torch.utils import cpp_extension

    ok, why = can_build()
    if not ok:
        raise RuntimeError(
            f"sweep's compiled backend (impl='c') is unavailable: {why}. "
            "Use impl='eager' for a pure-Python (slower) CPU/GPU path.")

    cuda_home = _find_cuda_home()
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    _ensure_ninja_on_path()

    build_dir = Path(cpp_extension._get_build_directory("sweep_C", verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    sources, inc = _stage(build_dir)
    # Use ONLY the selected CUDA toolkit's own headers (version-consistent with
    # its nvcc). Do NOT mix in the pip nvidia-*/include dirs: for a torch built
    # against an older CUDA (torch 2.5 = cu121 -> 12.1 headers) those clash with a
    # newer toolkit and break the <cuda/std> bf16 compile.
    inc = inc + [p for p in (os.path.join(cuda_home, "include"),
                             os.path.join(cuda_home, "targets", "x86_64-linux", "include"))
                 if os.path.isdir(p)]

    cap = torch.cuda.get_device_capability()
    building = _will_build(build_dir)
    if building:
        print(f"[sweep] compiling the CUDA backend for your GPU (sm_{cap[0]}{cap[1]}) — "
              f"one-time, ~2-5 min, then cached at {build_dir} ...",
              file=sys.stderr, flush=True)

    _module = cpp_extension.load(
        name="sweep_C",
        sources=sources,
        extra_include_paths=inc,
        extra_cflags=["-O3", "-Wno-attributes", "-fopenmp"],
        # --expt-relaxed-constexpr: lets constexpr __host__ funcs call __device__
        # ones, which some CUDA toolkits' <cuda/std> bf16 headers (e.g. 12.4's
        # nvbf16.h) require to compile. Harmless on toolkits that don't need it.
        extra_cuda_cflags=["-O3", "--use_fast_math", "--expt-relaxed-constexpr",
                           "-Xcompiler=-Wno-deprecated-declarations"],
        extra_ldflags=["-fopenmp"] + [f"-L{d}" for d in _nvidia_pip_libs()],
        build_directory=str(build_dir),
        verbose=building,
    )
    if building:
        print("[sweep] CUDA backend compiled and cached.", file=sys.stderr, flush=True)
    return _module
