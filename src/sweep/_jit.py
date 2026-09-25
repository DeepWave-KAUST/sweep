"""Build sweep's compiled backend on first use: the torch-free CUDA core
(``libsweep_core.so``, nvcc) and the thin torch shim (``sweep._C``, a plain C++
compile against the *user's* torch, linked to the core).

This is why one wheel of sweep works with **any** torch version and any Python
3: nothing torch-specific ships pre-built.  The core depends on CUDA only, so
the wheel ships it prebuilt under ``sweep/lib/cu<major>/`` (CUDA 12 today, a
fat binary, see ``python -m sweep.build --core``) and first use compiles only
the shim, in about a minute, without nvcc: a C++ compiler plus the CUDA runtime
headers torch's pip wheels bring (``nvidia-cuda-runtime-cu12``) or a toolkit's
include dir.  When no shipped core fits -- a torch built for another CUDA
major, a GPU outside the shipped archs and older than the shipped PTX, an sdist
install -- the core is built here once with nvcc.  Every run after the first
loads the cached libraries instantly.

The C++ sources ship inside the wheel under ``sweep/csrc/`` (package data).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent
_CSRC = _PKG / "csrc"
_LIB = _PKG / "lib"      # shipped cores: lib/<cu tag>/libsweep_core.so + core.json

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


def _pip_cuda_runtime_headers() -> bool:
    """Whether the CUDA runtime headers the shim needs are around without a
    toolkit.  c10/cuda/CUDAStream.h brings cuda_runtime_api.h (the pip
    ``nvidia-cuda-runtime`` wheel), which itself includes crt/host_defines.h
    (the pip ``nvidia-cuda-nvcc`` wheel) -- both must be visible."""
    dirs = _nvidia_pip_includes()
    return all(any(os.path.isfile(os.path.join(d, h)) for d in dirs)
               for h in ("cuda_runtime_api.h", os.path.join("crt", "host_defines.h")))


def _torch_cuda_version() -> str | None:
    """torch's CUDA release, e.g. "12.8"; None for a CPU-only torch."""
    try:
        import torch
        return torch.version.cuda or None
    except Exception:
        return None


def _torch_cuda_major() -> int | None:
    v = _torch_cuda_version()
    return int(v.split(".")[0]) if v else None


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


# --------------------------------------------------------------------------- #
# the shipped core (wheel-bundled libsweep_core.so)
# --------------------------------------------------------------------------- #
def _core_tag(cuda_version: str) -> str:
    """"12.8" -> "cu12": the drawer a shipped core is filed under -- torch's
    CUDA major, the unit the driver ABI and torch's own wheel tags go by."""
    return "cu" + str(cuda_version).strip().split(".")[0]


def _shim_abi() -> int | None:
    """SWEEP_CORE_ABI_VERSION the shim compiles against, read from the header
    that defines it, so the shim and the core it links can never disagree
    silently about the C boundary."""
    try:
        text = (_CSRC / "core" / "capi.h").read_text()
        m = re.search(r"^\s*#define\s+SWEEP_CORE_ABI_VERSION\s+(\d+)", text, re.M)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _device_arch() -> str | None:
    """"<maj><min>" of the visible device (e.g. "90"), None without one."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        maj, minr = torch.cuda.get_device_capability()
        return f"{maj}{minr}"
    except Exception:
        return None


def _sm(arch: str) -> tuple[int, int] | None:
    """"86" -> (8, 6); the last digit is the minor, so "100" -> (10, 0)."""
    try:
        return int(arch[:-1]), int(arch[-1])
    except (TypeError, ValueError):
        return None


def _core_fits(sidecar: dict) -> str:
    """"" when this process can use a core described by ``sidecar``, else why not."""
    abi = _shim_abi()
    if sidecar.get("abi") != abi:
        return f"core ABI {sidecar.get('abi')} != shim ABI {abi}"
    want = _torch_cuda_major()
    if want is None:
        return "torch has no CUDA build"
    have = str(sidecar.get("cuda", "")).split(".")[0]
    if have != str(want):
        return f"core is CUDA {sidecar.get('cuda')}, torch is CUDA {_torch_cuda_version()}"
    arch = _device_arch()
    if arch is None:
        return ""                          # no device: nothing to check the SASS against
    archs = [str(a) for a in sidecar.get("archs", [])]
    if arch in archs:
        return ""
    # SASS is binary-compatible within a major: an sm_80 cubin runs on sm_86.
    dev = _sm(arch)
    for a in archs:
        s = _sm(a)
        if dev and s and s[0] == dev[0] and s[1] <= dev[1]:
            return ""
    # PTX embedded for an older arch is JIT-ed forward by the driver.
    for p in sidecar.get("ptx", []):
        try:
            if int(p) <= int(arch):
                return ""
        except (TypeError, ValueError):
            continue
    return (f"GPU sm_{arch} is not among the core's archs {archs} (nor above one "
            f"of its major) and no PTX is older")


def _shipped_core() -> tuple[Path | None, str]:
    """The shipped ``libsweep_core.so`` this process can link, or (None, why).

    ``SWEEP_CORE=<path>`` wins (a custom core, checked through the ``core.json``
    beside it; accepted blind when there is none, as long as torch has a CUDA
    build to run it under).  Otherwise the wheel's ``lib/<cu tag>/`` for torch's
    CUDA major.  The core does not depend on torch, so a fitting one skips nvcc
    entirely: only the shim is compiled here.
    """
    env = os.environ.get("SWEEP_CORE", "").strip()
    if env:
        # Resolved: the shim's rpath is this directory, and a symlink that is
        # later repointed would silently change what gets loaded.
        so = Path(env).expanduser().resolve()
        if not so.is_file():
            return None, f"SWEEP_CORE={env} is not a file"
        if so.name != "libsweep_core.so":
            # -lsweep_core is what the link line says, so the name is fixed.
            return None, "SWEEP_CORE must point at a file named libsweep_core.so"
        side = so.parent / "core.json"
        if not side.is_file():
            if _torch_cuda_major() is None:
                return None, f"SWEEP_CORE={env}: torch has no CUDA build"
            return so, "SWEEP_CORE without sidecar"
        try:
            sidecar = json.loads(side.read_text())
        except (OSError, ValueError) as exc:
            return None, f"SWEEP_CORE sidecar {side} is unreadable: {exc}"
        why = _core_fits(sidecar)
        return (so, "ok (SWEEP_CORE)") if not why else (None, f"SWEEP_CORE={env}: {why}")

    ver = _torch_cuda_version()
    if ver is None:
        return None, "torch has no CUDA build"
    d = _LIB / _core_tag(ver)
    so = d / "libsweep_core.so"
    if not so.is_file():
        return None, f"no shipped core for {_core_tag(ver)} (no {so})"
    try:
        sidecar = json.loads((d / "core.json").read_text())
    except (OSError, ValueError) as exc:
        return None, f"shipped core {so} has no readable core.json: {exc}"
    why = _core_fits(sidecar)
    return (so, "ok") if not why else (None, why)


def shipped_core_info() -> dict:
    """For diagnostics: which shipped core (if any) this process would use."""
    so, why = _shipped_core()
    ver = _torch_cuda_version()
    return {"path": str(so) if so else None, "reason": why,
            "tag": _core_tag(ver) if ver else ""}


def _cufft_soname(major: int | None) -> str | None:
    # cuFFT's soname trails the CUDA major by one (cu12 -> .so.11, cu13 -> .so.12).
    return f"libcufft.so.{major - 1}" if major in (12, 13) else None


def _preload_cufft(cuda_home: str | None) -> None:
    """Load the cuFFT runtime the core needs into the global namespace before
    the shim is imported, so a core whose rpath does not reach the pip
    ``nvidia-cufft`` wheel (a custom SWEEP_CORE, a moved site-packages) still
    resolves.  Nothing found: fall through to the dynamic loader silently."""
    import ctypes
    dirs = _nvidia_pip_libs()
    if cuda_home:
        dirs += [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]
    soname = _cufft_soname(_torch_cuda_major())
    for d in dirs:
        cands = [os.path.join(d, soname)] if soname else \
            sorted(glob.glob(os.path.join(d, "libcufft.so.*")), key=len)
        for c in cands:
            if os.path.isfile(c):
                try:
                    ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
                    return
                except OSError:
                    continue


def can_compile() -> tuple[bool, str]:
    """(usable, reason) — True when the backend can be COMPILED here.

    With a shipped core that fits (see :func:`_shipped_core`) only the shim is
    compiled: plain C++ against torch, no nvcc, no target arch -- a C++ compiler
    and the CUDA runtime headers, from torch's pip cu12 wheels
    (``nvidia-cuda-runtime-cu12``) or a toolkit's include dir.  Otherwise the
    core is built locally, which needs an nvcc and a target architecture -- but
    **not** a visible device: ``TORCH_CUDA_ARCH_LIST`` names the target
    explicitly, which is how wheels are cross-built, and it is what lets a CI
    job or a CPU-partition allocation warm the cache a later GPU run reuses.
    Gating the compile on a device forces every build to occupy a scarce GPU.

    RUNNING the result still needs a device -- that is :func:`can_build`.
    """
    try:
        import torch
    except Exception:
        return False, "PyTorch is not installed"
    core, core_why = _shipped_core()
    if core is not None:
        if _find_cuda_home() is None and not _pip_cuda_runtime_headers():
            return False, (
                f"a shipped CUDA core fits ({core_why}) but the torch shim needs the "
                "CUDA runtime headers: pip install nvidia-cuda-runtime-cu12 nvidia-cuda-nvcc-cu12, or a "
                "CUDA toolkit")
        return True, "ok"
    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return False, (
            "no CUDA GPU is visible and TORCH_CUDA_ARCH_LIST is unset, so there "
            "is no target architecture to compile for (set e.g. "
            "TORCH_CUDA_ARCH_LIST=8.9 to build for a card this machine has not got)")
    if _find_cuda_home() is None:
        return False, (
            f"no shipped CUDA core fits ({core_why}) and no suitable CUDA toolkit "
            "was found to build one (need nvcc >=12.4 matching your torch's CUDA "
            "major — 12.0-12.3 ship a broken <cuda/std> bf16 header). "
            "Provide a recent nvcc via `module load cuda`, a system CUDA Toolkit, "
            "or `conda install -c nvidia cuda-toolkit`, or point SWEEP_CORE at a "
            "libsweep_core.so built elsewhere. To try an older toolkit anyway, "
            "set SWEEP_JIT_ALLOW_OLD_CUDA=1)")
    return True, "ok"


def can_build() -> tuple[bool, str]:
    """(usable, reason) — True when the C backend can be compiled AND run here.

    Does NOT compile. Used by ``sweep.is_torch_binding_available()`` to avoid a
    surprise compile, so it keeps requiring a visible device: a machine that can
    only cross-compile cannot serve ``impl='c'``.
    """
    try:
        import torch
    except Exception:
        return False, "PyTorch is not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA GPU is visible"
    return can_compile()


# --------------------------------------------------------------------------- #
# source staging (dedupe object basenames)
# --------------------------------------------------------------------------- #
def _cufft_link_flag(cuda_home: str) -> str:
    """``-lcufft`` when the toolkit ships the unversioned ``libcufft.so``; a
    pip-only toolkit (``nvidia-cufft-cu*``) ships only the versioned soname, so
    link that by name (``-l:libcufft.so.11``) instead of failing at link time."""
    import glob as _glob
    for d in [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")] + _nvidia_pip_libs():
        if os.path.exists(os.path.join(d, "libcufft.so")):
            return "-lcufft"
    for d in _nvidia_pip_libs() + [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]:
        sonames = sorted(_glob.glob(os.path.join(d, "libcufft.so.*")), key=len)
        if sonames:
            return f"-l:{os.path.basename(sonames[0])}"
    return "-lcufft"


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


def _staged_name(rel: Path) -> Path:
    """Staged path of a COMPILED source: cpp_extension.load() flattens object
    names by basename and sweep has many forward.cu / backward.cu / kernels.cu,
    so each one is renamed in place (its relative #includes still resolve)."""
    slug = "_".join(rel.with_suffix("").parts)
    return rel.parent / (slug + rel.suffix)


def _stage_plan() -> dict[Path, Path]:
    """Every file under csrc, mapped to where it is staged."""
    renamed = {Path(s).resolve().relative_to(_CSRC) for s in _sources()}
    plan: dict[Path, Path] = {}
    for p in _CSRC.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(_CSRC)
        plan[rel] = _staged_name(rel) if rel in renamed else rel
    return plan


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stage(build_dir: Path) -> tuple[list[str], list[str]]:
    """Mirror csrc into a staging dir with unique compiled-source basenames.

    The sentinel is a MANIFEST of source digests, not a bare "I ran once" flag.
    Keying staleness on the package version alone meant that editing a kernel
    without bumping the version left the previous copy in place: ninja then
    compiled the OLD source and produced a .so that silently did not contain
    the edit, which is why the workflow around this was "delete the extension
    directory after touching csrc". Re-staging only the files whose contents
    changed keeps that from happening AND keeps the build incremental -- ninja
    recompiles the affected translation units and, through its header
    depfiles, whatever includes a changed .cuh.

    The staged copy's mtime is set to now rather than inherited, so a source
    that travels BACKWARDS in time (a `git checkout` of an older revision)
    still invalidates the object built from it.
    """
    try:
        from importlib.metadata import version
        _ver = version("sweep-solver")
    except Exception:
        _ver = "dev"
    stage = build_dir / f"csrc_stage_{_ver}"
    manifest_path = stage / ".staged"
    # torchrun starts one process per GPU and they all import at once, so the
    # staging runs concurrently. It happens BEFORE cpp_extension.load()'s own
    # lock, so nothing else serialises it: two ranks used to race in
    # rmtree+copytree and one lost with FileExistsError on the stage directory
    # (seen on a 2-rank DD benchmark). Torch's own baton is the same mechanism
    # its extension build uses -- the loser waits for the winner to finish
    # rather than staging on top of it.
    from torch.utils.file_baton import FileBaton

    build_dir.mkdir(parents=True, exist_ok=True)
    baton = FileBaton(str(build_dir / "sweep_stage_lock"))
    if not baton.try_acquire():
        baton.wait()
        return _staged_paths(stage)
    try:
        return _stage_locked(stage, manifest_path)
    finally:
        baton.release()


def _staged_paths(stage: Path) -> tuple[list[str], list[str]]:
    staged = [str(stage / _staged_name(Path(s).resolve().relative_to(_CSRC)))
              for s in _sources()]
    inc = [str(stage), str(stage / "bindings"), str(stage / "shared"),
           str(stage / "cuda"), str(stage / "cuda/common"), str(stage / "cuda/equations")]
    return staged, inc


def _stage_locked(stage: Path, manifest_path: Path) -> tuple[list[str], list[str]]:
    try:
        previous = json.loads(manifest_path.read_text())
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, ValueError):
        previous = {}       # first run, or the pre-manifest "ok" sentinel

    plan = _stage_plan()
    manifest: dict[str, str] = {}
    for rel, dst_rel in sorted(plan.items()):
        key = str(dst_rel)
        digest = _digest(_CSRC / rel)
        manifest[key] = digest
        dst = stage / dst_rel
        if previous.get(key) != digest or not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_CSRC / rel, dst)
            os.utime(dst, None)
    # A source that was deleted (or renamed, or dropped by a build-mode switch)
    # must not linger in the stage where it would still compile.
    for key in set(previous) - set(manifest):
        (stage / key).unlink(missing_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    return _staged_paths(stage)


def _core_sources(sources: list[str]) -> list[str]:
    return [x for x in sources if x.endswith(".cu")]


def _shim_sources(sources: list[str]) -> list[str]:
    return [x for x in sources if not x.endswith(".cu")]


def _gencode_flags() -> list[str]:
    """One -gencode per target: the visible device's arch, else the archs
    TORCH_CUDA_ARCH_LIST names ("7.0;8.0+PTX", as torch spells it)."""
    import torch
    if torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        maj, minr = torch.cuda.get_device_capability()
        return [f"-gencode=arch=compute_{maj}{minr},code=sm_{maj}{minr}"]
    flags: list[str] = []
    for a in os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace(",", ";").split(";"):
        a = a.strip()
        if not a:
            continue
        ptx = a.endswith("+PTX")
        a = a[:-4] if ptx else a
        a = a.replace(".", "")
        flags.append(f"-gencode=arch=compute_{a},code=sm_{a}")
        if ptx:
            flags.append(f"-gencode=arch=compute_{a},code=compute_{a}")
    return flags


# What cpp_extension passed for these translation units, minus torch: the same
# defines (they select the fp16/bf16 header code paths) and the same numerics
# flags, so the core compiled here is the core the gate baselines were made with.
_CORE_DEFINES = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                 "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]
_CORE_FLAGS = ["--expt-relaxed-constexpr", "--compiler-options", "'-fPIC'", "-O3", "--use_fast_math",
               "-Xcompiler=-Wno-deprecated-declarations", "-Xcompiler=-fvisibility=hidden", "-std=c++17"]


def _core_ninja(core_dir: Path, sources: list[str], inc: list[str], cuda_home: str) -> str:
    import shlex
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    cufft = _cufft_link_flag(cuda_home)
    if not cufft.startswith("-lcufft"):
        cufft = "-Xlinker " + cufft                     # -l:libcufft.so.11 is the host linker's spelling
    ldflags = " ".join([f"-L{shlex.quote(d)}" for d in _nvidia_pip_libs()] + [cufft])
    cflags = " ".join(_CORE_DEFINES + _gencode_flags() + _CORE_FLAGS + [f"-I{shlex.quote(d)}" for d in inc])
    lines = [f"nvcc = {nvcc}", f"cflags = {cflags}", "",
             "rule nvcc",
             "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cflags -c $in -o $out",
             "  depfile = $out.d", "  deps = gcc", "",
             "rule link", f"  command = $nvcc -shared -o $out $in {ldflags}", ""]
    objs = []
    for src in sources:
        obj = Path(src).stem + ".o"                     # staged names are unique already
        objs.append(obj)
        lines.append(f"build {obj}: nvcc {shlex.quote(src)}")
    lines += ["", f"build libsweep_core.so: link {' '.join(objs)}", "", "default libsweep_core.so", ""]
    return "\n".join(lines)


def _build_core(build_dir: Path, sources: list[str], inc: list[str], cuda_home: str, verbose: bool) -> Path:
    """libsweep_core.so: the torch-free core, nvcc through its own ninja graph
    (incremental; header depfiles).  Serialised with the same baton the staging
    uses, so the ranks of a torchrun job do not build on top of each other."""
    import subprocess
    from torch.utils.file_baton import FileBaton
    core_dir = build_dir / "core"
    core_dir.mkdir(parents=True, exist_ok=True)
    text = _core_ninja(core_dir, sources, inc, cuda_home)
    ninja_file = core_dir / "build.ninja"
    if not ninja_file.exists() or ninja_file.read_text() != text:
        ninja_file.write_text(text)
    _ensure_ninja_on_path()
    baton = FileBaton(str(build_dir / "sweep_core_lock"))
    if not baton.try_acquire():
        baton.wait()
    else:
        try:
            r = subprocess.run(["ninja", "-C", str(core_dir)], capture_output=not verbose, text=True)
            if r.returncode != 0:
                raise RuntimeError("sweep core build (libsweep_core.so) failed:\n"
                                   + (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:])
        finally:
            baton.release()
    so = core_dir / "libsweep_core.so"
    if not so.exists():
        raise RuntimeError(f"sweep core build left no {so}")
    return so


def _ninja_has_work(d: Path) -> bool:
    ninja = shutil.which("ninja")
    if ninja is None:
        return True                       # can't check -> assume yes (never hang silently)
    try:
        import subprocess
        r = subprocess.run([ninja, "-n"], cwd=str(d), capture_output=True, text=True, timeout=30)
        return "no work to do" not in (r.stdout + r.stderr)
    except Exception:
        return True


def _will_build(build_dir: Path, core_shipped: bool = False) -> bool:
    """Whether the next load() will actually *compile* (vs reuse the cached
    libraries): the core's ninja graph or the shim's has work.  A cached .so can
    still be rebuilt -- e.g. after the user upgrades torch, whose changed headers
    make the shim recompile -- so "the .so exists" is not the signal; ninja is.
    This drives the one-time "compiling..." notice + verbose output, so a genuine
    rebuild is never a silent hang that looks frozen.  With a shipped core only
    the shim's graph counts: the local core dir is not touched.  A shipped core
    that merely MOVED (relocated site-packages, SWEEP_CORE repointed) changes
    only the shim's link line, so ninja relinks it silently: no notice, no
    core build."""
    core = build_dir / "core"
    if not core_shipped and (not (core / "libsweep_core.so").exists()
                             or not (core / "build.ninja").exists()):
        return True
    if not (build_dir / "sweep_C.so").exists() or not (build_dir / "build.ninja").exists():
        return True
    _ensure_ninja_on_path()
    return (not core_shipped and _ninja_has_work(core)) or _ninja_has_work(build_dir)


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
def load(compile_only: bool = False):
    """Compile (first call, cached) and return the ``sweep._C`` module.

    Two stages: ``libsweep_core.so`` (the torch-free CUDA core, nvcc) and the
    torch shim ``sweep_C`` (a plain C++ compile against the user's torch, linked
    to the core).  ``compile_only=True`` warms the cache without requiring a
    device, for CI and for pre-building on a CPU allocation;
    ``TORCH_CUDA_ARCH_LIST`` must then name the target arch.  The returned
    module is not usable for kernels on a machine with no GPU -- the point is
    the cached libraries.
    """
    global _module
    if _module is not None:
        return _module
    import torch
    from torch.utils import cpp_extension
    ok, why = can_compile() if compile_only else can_build()
    if not ok:
        raise RuntimeError(
            f"sweep's compiled backend (impl='c') is unavailable: {why}. "
            "Use impl='eager' for a pure-Python (slower) CPU/GPU path.")
    core_so, core_why = _shipped_core()
    cuda_home = _find_cuda_home()          # may be None with a shipped core: no nvcc needed then
    if cuda_home:
        os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    _ensure_ninja_on_path()
    build_dir = Path(cpp_extension._get_build_directory("sweep_C", verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    sources, inc = _stage(build_dir)
    # The shim includes c10/cuda/CUDAStream.h, hence cuda_runtime_api.h.  Use
    # ONLY the selected CUDA toolkit's own headers (version-consistent with its
    # nvcc); do NOT mix in the pip nvidia-*/include dirs next to a toolkit: for
    # a torch built against an older CUDA (torch 2.5 = cu121 -> 12.1 headers)
    # those clash with a newer toolkit and break the <cuda/std> bf16 compile.
    # Without a toolkit (shipped core) the pip headers are the only ones, and
    # they match torch; can_compile() already refused when they are missing.
    if cuda_home:
        inc = inc + [p for p in (os.path.join(cuda_home, "include"),
                                 os.path.join(cuda_home, "targets", "x86_64-linux", "include"))
                     if os.path.isdir(p)]
    else:
        inc = inc + _nvidia_pip_includes()
    building = _will_build(build_dir, core_shipped=core_so is not None)
    if building:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            target = f"your GPU (sm_{cap[0]}{cap[1]})"
        else:
            target = f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}"
        if core_so is not None:
            print(f"[sweep] compiling the torch shim of the CUDA backend against torch "
                  f"{torch.__version__} (prebuilt core {core_so.parent.name}) -- "
                  f"one-time, ~1 min, no nvcc, then cached at {build_dir} ...",
                  file=sys.stderr, flush=True)
        else:
            print(f"[sweep] no prebuilt CUDA core fits ({core_why}); compiling the "
                  f"CUDA backend for {target} -- one-time, ~2-5 min, then cached at "
                  f"{build_dir} ...", file=sys.stderr, flush=True)
    if core_so is None:
        core_so = _build_core(build_dir, _core_sources(sources), inc, cuda_home, verbose=building)
    core_dir = str(core_so.parent)
    _preload_cufft(cuda_home)
    _module = cpp_extension.load(
        name="sweep_C",
        sources=_shim_sources(sources),                 # module.cpp + the CPU binding: C++ only, no nvcc
        extra_include_paths=inc,
        extra_cflags=["-O3", "-Wno-attributes", "-fopenmp"],
        extra_ldflags=["-fopenmp", f"-L{core_dir}", "-lsweep_core", f"-Wl,-rpath,{core_dir}"]
                      + [f"-L{d}" for d in _nvidia_pip_libs()],
        build_directory=str(build_dir),
        verbose=building,
    )
    if building:
        print("[sweep] CUDA backend compiled and cached.", file=sys.stderr, flush=True)
    return _module
