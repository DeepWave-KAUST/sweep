"""Where sweep's CUDA core comes from -- and, for developers, the compiled shim.

One wheel serves **any** torch version and any Python 3, and nothing compiles
after ``pip install``: the core depends on CUDA only, so the wheel ships it
prebuilt (``libsweep_core.so`` under ``sweep/lib/cu<major>/``: two cores, ``cu12``
and ``cu13``, one per CUDA major torch is built for, each a fat binary; see
``python -m sweep.build --core``), and the shim that calls it is pure Python
(``sweep._capi``, ctypes over the core's C API).
:func:`core_path` hands the shim that core; first use is instant.  nvcc enters
only when no shipped core fits -- a torch built for another CUDA major, a GPU
outside the shipped archs and older than the shipped PTX, an sdist install --
and then the core is built here once (staged sources, its own ninja graph) and
cached for every later run: a source stamp beside it (``sources.sha256``, the
tree plus the nvcc flags it was built with) says when that cache is current,
and a current one is reused without a toolkit in sight.

The compiled pybind shim (``sweep_C``, a C++ compile against the *user's*
torch, linked to the core) survives as the developer path behind
``SWEEP_JIT_FULL=1``: the CPU engine lives there.  :func:`load` builds it
through ``torch.utils.cpp_extension``, as it always did.

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

_module = None          # cached compiled module (SWEEP_JIT_FULL; process-local)
_core_so = None         # cached Path of the core this process runs on


def jit_full() -> bool:
    """``SWEEP_JIT_FULL=1``: the compiled pybind shim (the developer path, where
    the CPU engine lives) instead of the ctypes shim."""
    return os.environ.get("SWEEP_JIT_FULL", "").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# CUDA toolkit (nvcc) discovery
# --------------------------------------------------------------------------- #
def _nvidia_pip_roots() -> list[str]:
    """The ``site-packages/nvidia/`` roots the pip CUDA wheels install under
    (``nvidia.__path__``: more than one when several site-packages hold
    wheels); [] without the package.  Two layouts live there, one per CUDA
    major torch's wheels come in: CUDA 12 wheels (``nvidia-*-cu12``) are one
    directory per component -- ``nvidia/cuda_nvcc/bin/nvcc``,
    ``nvidia/cufft/lib``, ``nvidia/cuda_runtime/include`` -- while CUDA 13
    wheels (the unsuffixed ``nvidia-*`` line) share one directory per major,
    ``nvidia/cu13/{bin,include,lib}``."""
    try:
        import nvidia
    except Exception:
        return []
    return [str(p) for p in getattr(nvidia, "__path__", [])]


def _nvidia_pip_includes() -> list[str]:
    """Every ``nvidia/*/include`` dir from the pip CUDA wheels torch pulls in
    (cuda_runtime, cusparse, cublas, cudnn, … under CUDA 12; the one
    ``nvidia/cu13/include`` under CUDA 13 -- the glob covers both layouts) — so
    nvcc/host cc find the headers even when there is no system CUDA toolkit."""
    incs: list[str] = []
    for base in _nvidia_pip_roots():
        for inc in sorted(glob.glob(os.path.join(base, "*", "include"))):
            incs.append(inc)
    return incs


def _pip_cuda_runtime_headers() -> bool:
    """Whether the CUDA runtime headers the shim needs are around without a
    toolkit.  c10/cuda/CUDAStream.h brings cuda_runtime_api.h (the pip
    ``nvidia-cuda-runtime`` wheel), which itself includes crt/host_defines.h
    (the pip ``nvidia-cuda-nvcc`` wheel) -- both must be visible, whichever
    layout they landed in (``nvidia/cuda_runtime/include`` +
    ``nvidia/cuda_nvcc/include`` under CUDA 12, both in ``nvidia/cu13/include``
    under CUDA 13; :func:`_nvidia_pip_includes` lists either)."""
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


def _tag_major(tag: str) -> int | None:
    """"cu13" -> 13: the CUDA major a ``cu<major>`` drawer name spells (the
    wheel's ``lib/`` and the pip CUDA 13 layout use the same spelling); None
    for any other name."""
    m = re.fullmatch(r"cu(\d+)", str(tag))
    return int(m.group(1)) if m else None


def _pip_nvcc_homes(want: int | None) -> list[Path]:
    """Candidate CUDA_HOMEs from the pip nvcc wheels -- directories holding
    ``bin/nvcc`` -- most likely first: the ``nvidia/cu<want>/`` drawer of the
    CUDA 13 layout (its name says its major), then ``nvidia/cuda_nvcc/`` (the
    CUDA 12 layout, whose nvcc has to be asked), then any other ``nvidia/cu*/``
    newest first.  The caller version-checks each; the order only decides
    which nvcc is asked first."""
    homes: list[Path] = []
    for root in _nvidia_pip_roots():
        r = Path(root)
        drawers = [Path(p).parent.parent for p in glob.glob(str(r / "cu[0-9]*" / "bin" / "nvcc"))]
        drawers.sort(key=lambda p: _tag_major(p.name) or 0, reverse=True)
        mine = [p for p in drawers if want is not None and _tag_major(p.name) == want]
        homes += mine
        if (r / "cuda_nvcc" / "bin" / "nvcc").is_file():
            homes.append(r / "cuda_nvcc")
        homes += [p for p in drawers if p not in mine]
    return homes


_cuda_home_cache = False   # False = not computed; None/str = computed result


def _find_cuda_home() -> str | None:
    """Return a CUDA_HOME (dir with bin/nvcc) whose CUDA **major matches the
    user's torch**. Priority: explicit CUDA_HOME env, then the pip nvcc wheel
    (``nvidia-cuda-nvcc-cu12`` at ``nvidia/cuda_nvcc/``, or CUDA 13's
    ``nvidia-cuda-nvcc`` at ``nvidia/cu13/``; see :func:`_pip_nvcc_homes`),
    then nvcc on PATH — each version-checked so an old system nvcc (e.g. CUDA
    10.1 in /usr/bin), or the pip nvcc of the other CUDA major on a mixed
    install, is skipped rather than used and failing mid-compile."""
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
    # 2. the pip nvcc wheel, in either layout (nvidia/cu13/bin/nvcc or
    #    nvidia/cuda_nvcc/bin/nvcc); torch's major is asked first, and match()
    #    refuses the other major's nvcc whichever directory it sits in
    if result is None:
        for home in _pip_nvcc_homes(want):
            if match(home / "bin" / "nvcc"):
                result = str(home)
                break
    # 3. nvcc on PATH (version-checked -> skips old /usr/bin/nvcc)
    if result is None:
        p = shutil.which("nvcc")
        if p and match(Path(p)):
            result = str(Path(p).resolve().parent.parent)

    _cuda_home_cache = result
    return result


def _nvidia_pip_libs() -> list[str]:
    """``nvidia/*/lib`` dirs so the JIT link step finds libcudart etc. when there
    is no system CUDA toolkit (provided by torch's pip CUDA wheels: one dir per
    component under CUDA 12, the one ``nvidia/cu13/lib`` under CUDA 13 -- the
    glob covers both layouts)."""
    libs: list[str] = []
    for base in _nvidia_pip_roots():
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
    """SWEEP_CORE_ABI_VERSION the shim speaks: ``sweep._core_abi.ABI_VERSION``,
    the generated mirror the ctypes shim is built from and what its ABI guard
    compares the loaded core against -- so the fit check here and the guard
    there can never disagree.  The header that defines the number
    (core/capi.h) is the fallback when the mirror cannot be imported."""
    try:
        from . import _core_abi
        return int(_core_abi.ABI_VERSION)
    except Exception:
        pass
    try:
        text = (_CSRC / "core" / "capi.h").read_text()
        m = re.search(r"^\s*#define\s+SWEEP_CORE_ABI_VERSION\s+(\d+)", text, re.M)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _driver_version_raw() -> int | None:
    """``cuDriverGetVersion`` of libcuda.so.1, the integer the driver reports
    (12040 = CUDA 12.4); no context and no cuInit.  None when there is no
    driver library or the call fails."""
    import ctypes
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        fn = lib.cuDriverGetVersion
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        v = ctypes.c_int(0)
        if fn(ctypes.byref(v)) != 0 or v.value <= 0:
            return None
        return int(v.value)
    except Exception:
        return None


def _driver_cuda_version() -> tuple[int, int] | None:
    """The CUDA release the NVIDIA driver supports, (major, minor); None when
    it cannot be told, which callers treat as "do not block"."""
    raw = _driver_version_raw()
    if raw is None:
        return None
    return raw // 1000, (raw % 1000) // 10


def _version_pair(v) -> tuple[int, int] | None:
    """"12.8" -> (12, 8); "12" -> (12, 0); None when unparsable."""
    try:
        parts = str(v).strip().split(".")
        return int(parts[0]), (int(parts[1]) if len(parts) > 1 else 0)
    except (TypeError, ValueError, IndexError):
        return None


def _ptx_driver_gate(arch: str, sidecar: dict) -> str:
    """A PTX-only fit is the driver's JIT, and a driver can only assemble PTX
    from a toolkit no newer than itself: require driver CUDA >= the sidecar's
    ``cuda`` (major.minor).  "" when it holds or cannot be told, else why not."""
    want = _version_pair(sidecar.get("cuda"))
    have = _driver_cuda_version()
    if want is None or have is None:
        return ""
    if have < want:
        return (f"device sm_{arch} fits only via PTX and driver CUDA {have[0]}.{have[1]} "
                f"< core CUDA {want[0]}.{want[1]}")
    return ""


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
    # PTX embedded for an older arch is JIT-ed forward by the driver -- by a
    # driver at least as new as the toolkit that emitted it (_ptx_driver_gate).
    for p in sidecar.get("ptx", []):
        try:
            if int(p) <= int(arch):
                return _ptx_driver_gate(arch, sidecar)
        except (TypeError, ValueError):
            continue
    return (f"GPU sm_{arch} is not among the core's archs {archs} (nor above one "
            f"of its major) and no PTX is older")


def _shipped_core() -> tuple[Path | None, str]:
    """The shipped ``libsweep_core.so`` this process can link, or (None, why).

    ``SWEEP_CORE=<path>`` wins (a custom core, checked through the ``core.json``
    beside it; accepted blind when there is none, as long as torch has a CUDA
    build to run it under).  Otherwise the wheel's ``lib/<cu tag>/`` for torch's
    CUDA major -- ``cu12`` or ``cu13``, never the other drawer, since the CUDA
    runtime a core links must be the one torch brought.  The core does not
    depend on torch, so a fitting one needs no
    nvcc: the ctypes shim loads it as it is, and nothing is compiled (the
    ``SWEEP_JIT_FULL=1`` developer path alone compiles its pybind shim,
    against torch, and needs no nvcc for that either).
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


def _shipped_tags() -> list[str]:
    """The ``lib/<tag>/`` drawers of this install that hold a
    ``libsweep_core.so`` (``["cu12", "cu13"]`` for a full wheel), sorted -- fit
    or not, so a user can see which cores the install carries."""
    hits = glob.glob(os.path.join(str(_LIB), "*", "libsweep_core.so"))
    return sorted(os.path.basename(os.path.dirname(p)) for p in hits if os.path.isfile(p))


def shipped_core_info() -> dict:
    """For diagnostics: which shipped core (if any) this process would use --
    ``path`` and ``reason`` -- plus ``tag``, the drawer torch's CUDA major
    points at, and ``available``, the drawers the install holds a core in."""
    so, why = _shipped_core()
    ver = _torch_cuda_version()
    return {"path": str(so) if so else None, "reason": why,
            "tag": _core_tag(ver) if ver else "", "available": _shipped_tags()}


def _cufft_soname(major: int | None) -> str | None:
    """The cuFFT runtime a core of this CUDA major links: the soname trails the
    CUDA major by one (cu12 -> ``libcufft.so.11``, cu13 -> ``libcufft.so.12``;
    the CUDA 13 one is the pip ``nvidia-cufft`` 12.x wheel).  None for a major
    no shipped core exists for."""
    return f"libcufft.so.{major - 1}" if major in (12, 13) else None


def _pip_wheel(name: str, major: int | None) -> str:
    """The pip name of an NVIDIA CUDA wheel for a CUDA major, for messages:
    CUDA 12 (and older) wheels carry the major as a suffix
    (``nvidia-cufft-cu12``); the CUDA 13 line is unsuffixed (``nvidia-cufft``)."""
    return f"{name}-cu{major}" if major is not None and major <= 12 else name


def _preload_cufft(cuda_home: str | None) -> None:
    """Load the cuFFT runtime the core needs into the global namespace before
    the shim is imported, so a core whose rpath does not reach the pip cuFFT
    wheel (``nvidia-cufft-cu12`` under ``nvidia/cufft/lib``, or CUDA 13's
    ``nvidia-cufft`` under ``nvidia/cu13/lib``; a custom SWEEP_CORE, a moved
    site-packages) still resolves -- the soname of torch's major, so a mixed
    install never hands a cu13 core the cu12 runtime.  Nothing found: fall
    through to the dynamic loader silently."""
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
    """(usable, reason) — True when a CUDA core can be had here, device or not.

    Nothing compiles by default: the shim is pure Python (ctypes), and a
    shipped core that fits (see :func:`_shipped_core`) needs nothing else -- no
    C++ compiler, no CUDA headers, no nvcc.  Otherwise the core is built
    locally, which needs an nvcc and a target architecture -- but **not** a
    visible device: ``TORCH_CUDA_ARCH_LIST`` names the target explicitly, which
    is how wheels are cross-built, and it is what lets a CI job or a
    CPU-partition allocation warm the cache a later GPU run reuses.  Gating the
    build on a device forces every build to occupy a scarce GPU.

    A shipped core -- or a cached local build whose source stamp is current,
    see :func:`_cached_local_core` -- needs at run time only the CUDA driver
    and the cuFFT runtime it links (``nvidia-cufft-cu12`` for the cu12 core,
    the unsuffixed ``nvidia-cufft`` for cu13: torch's own CUDA wheels bring
    it, the ``sweepx[cuda12]`` / ``sweepx[cuda13]`` extra names it for a torch
    that did not); no headers.  ``SWEEP_JIT_FULL=1`` (the compiled shim, a C++
    compile against torch) adds the CUDA runtime headers to that: torch's pip
    CUDA wheels bring them (``nvidia-cuda-runtime[-cu12]``, plus
    ``nvidia-cuda-nvcc[-cu12]`` for crt/host_defines.h), or a toolkit's
    include dir.  A local build under CUDA 13 cannot target sm_70 (nvcc 13
    dropped compute capability < 7.5): that is refused up front rather than
    mid-compile.

    RUNNING the result still needs a device -- that is :func:`can_build`.
    """
    try:
        import torch
    except Exception:
        return False, "PyTorch is not installed"
    core, core_why = _shipped_core()
    if core is None:
        cached = _cached_local_core()
        if cached is not None:
            core, core_why = cached, "cached local core is current"
    want = _torch_cuda_major()
    if core is not None:
        if jit_full() and _find_cuda_home() is None and not _pip_cuda_runtime_headers():
            return False, (
                f"a shipped CUDA core fits ({core_why}) but the compiled shim "
                "(SWEEP_JIT_FULL=1) needs the CUDA runtime headers: pip install "
                f"{_pip_wheel('nvidia-cuda-runtime', want)} "
                f"{_pip_wheel('nvidia-cuda-nvcc', want)}, or a CUDA toolkit")
        return True, "ok"
    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return False, (
            "no CUDA GPU is visible and TORCH_CUDA_ARCH_LIST is unset, so there "
            "is no target architecture to compile for (set e.g. "
            "TORCH_CUDA_ARCH_LIST=8.9 to build for a card this machine has not got)")
    floor_why = _arch_floor_reason()
    if floor_why:
        return False, f"no shipped CUDA core fits ({core_why}) and {floor_why}"
    cuda_home = _find_cuda_home()
    if cuda_home is None:
        which = (f"an nvcc of CUDA {want}, your torch's CUDA major" if want
                 else "an nvcc matching your torch's CUDA major")
        floor = _nvcc_floor_for_targets()
        if floor is not None:
            which += f", and >= {floor[0][0]}.{floor[0][1]} for sm_{floor[1]}"
        return False, (
            f"no shipped CUDA core fits ({core_why}) and no suitable CUDA toolkit "
            f"was found to build one (need {which}; for CUDA 12 that is nvcc "
            ">=12.4, as 12.0-12.3 ship a broken <cuda/std> bf16 header). "
            "Provide a recent nvcc via `module load cuda`, a system CUDA Toolkit, "
            "or `conda install -c nvidia cuda-toolkit`, or point SWEEP_CORE at a "
            "libsweep_core.so built elsewhere. To try an older toolkit anyway, "
            "set SWEEP_JIT_ALLOW_OLD_CUDA=1)")
    ceiling_why = _arch_ceiling_reason(cuda_home)
    if ceiling_why:
        return False, f"no shipped CUDA core fits ({core_why}) and {ceiling_why}"
    return True, "ok"


def can_build() -> tuple[bool, str]:
    """(usable, reason) — True when the C backend can be had AND run here.

    Does NOT build or load anything.  Used by
    ``sweep.is_torch_binding_available()`` to answer without side effects, so
    it keeps requiring a visible device: the ctypes shim serves the CUDA core
    only, and a machine that can merely cross-build a core cannot serve
    ``impl='c'``.
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
    pip-only toolkit (``nvidia-cufft-cu12``, or CUDA 13's ``nvidia-cufft``)
    ships only the versioned soname, so link that by name instead of failing
    at link time -- the soname of the toolkit's own major (``libcufft.so.11``
    for a CUDA 12 nvcc, ``.so.12`` for CUDA 13), never the other layout's on a
    mixed install; the shortest ``libcufft.so.*`` around is the last resort."""
    toolkit_libs = [os.path.join(cuda_home, "lib64"), os.path.join(cuda_home, "lib")]
    for d in toolkit_libs + _nvidia_pip_libs():
        if os.path.exists(os.path.join(d, "libcufft.so")):
            return "-lcufft"
    ver = _nvcc_version(os.path.join(cuda_home, "bin", "nvcc"))
    soname = _cufft_soname(ver[0] if ver else _torch_cuda_major())
    for d in _nvidia_pip_libs() + toolkit_libs:
        if soname and os.path.isfile(os.path.join(d, soname)):
            return f"-l:{soname}"
    for d in _nvidia_pip_libs() + toolkit_libs:
        sonames = sorted(glob.glob(os.path.join(d, "libcufft.so.*")), key=len)
        if sonames:
            return f"-l:{os.path.basename(sonames[0])}"
    return "-lcufft"


def _sources() -> list[str]:
    """C++/CUDA sources, mirroring build_config.get_sources(): the core's .cu
    files plus the compiled shim's C++ (the heavy CPU tree under
    SWEEP_JIT_FULL=1, where that shim lives; a stub otherwise)."""
    cu = (glob.glob(str(_CSRC / "cuda/common/**/*.cu"), recursive=True)
          + glob.glob(str(_CSRC / "cuda/equations/**/*.cu"), recursive=True))
    binding = [str(_CSRC / "bindings/module.cpp")]
    if jit_full():
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


def _target_archs() -> list[tuple[str, bool]]:
    """The archs a local core build targets, as ("86", wants PTX too) pairs:
    the visible device's arch, else what TORCH_CUDA_ARCH_LIST names
    ("7.0;8.0+PTX", as torch spells it)."""
    import torch
    if torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        maj, minr = torch.cuda.get_device_capability()
        return [(f"{maj}{minr}", False)]
    targets: list[tuple[str, bool]] = []
    for a in os.environ.get("TORCH_CUDA_ARCH_LIST", "").replace(",", ";").split(";"):
        a = a.strip()
        if not a:
            continue
        ptx = a.endswith("+PTX")
        a = a[:-4] if ptx else a
        targets.append((a.replace(".", ""), ptx))
    return targets


def _gencode_flags() -> list[str]:
    """One -gencode per target of :func:`_target_archs` (plus a compute_ one
    where PTX is asked for)."""
    flags: list[str] = []
    for a, ptx in _target_archs():
        flags.append(f"-gencode=arch=compute_{a},code=sm_{a}")
        if ptx:
            flags.append(f"-gencode=arch=compute_{a},code=compute_{a}")
    return flags


# The oldest compute capability a toolkit of this CUDA major can still compile
# for offline: nvcc 13 dropped everything below 7.5 (Volta, sm_70, is gone).
_MIN_ARCH = {13: (7, 5)}


def _arch_floor_reason() -> str:
    """"" when the toolkit of torch's CUDA major can compile every target of a
    local core build (:func:`_target_archs`), else why not -- the one case a
    toolkit cannot help: a card older than the major supports."""
    want = _torch_cuda_major()
    floor = _MIN_ARCH.get(want)
    if floor is None:
        return ""
    for a, _ptx in _target_archs():
        s = _sm(a)
        if s is not None and s < floor:
            return (f"CUDA {want} cannot compile for sm_{a}: nvcc {want} dropped compute "
                    f"capability < {floor[0]}.{floor[1]}, so this card needs a torch built "
                    f"for CUDA {want - 1} (and its shipped core)")
    return ""


# The oldest toolkit that can compile for a Blackwell card at all: compute
# capability 10.x/12.x arrived in CUDA 12.8, so an nvcc 12.4-12.7 -- past the
# bf16 floor of _find_cuda_home -- fails on sm_100/sm_120 mid-build with nvcc's
# own "Unsupported gpu architecture".  This is the nvcc a torch cu12x user on
# an RTX 50 / B200 is sent to: the cu12 core reaches Blackwell only through
# its compute_90 PTX, which a driver older than the nvcc that emitted it
# refuses (_ptx_driver_gate).  A necessary bound, never a false refusal: an
# arch the table does not name is left to nvcc's own check.
_MIN_NVCC_FOR_ARCH = (((10, 0), (12, 8)),)


def _nvcc_floor_for_targets() -> tuple[tuple[int, int], str] | None:
    """The oldest nvcc that can compile every target of a local core build
    (:func:`_target_archs`) per ``_MIN_NVCC_FOR_ARCH``, with the arch that
    asks for it -- ``((12, 8), "120")`` -- or None when no target is bound
    (or the targets cannot be told, which never blocks: the build's own
    :func:`_gencode_flags` is where that failure belongs)."""
    try:
        targets = _target_archs()
    except Exception:
        return None
    best = None
    for a, _ptx in targets:
        s = _sm(a)
        if s is None:
            continue
        for arch_from, need in _MIN_NVCC_FOR_ARCH:
            if s >= arch_from and (best is None or need > best[0]):
                best = (need, a)
    return best


def _arch_ceiling_reason(cuda_home: str) -> str:
    """"" when the nvcc at ``cuda_home`` is new enough for every target of a
    local core build, else why not, naming the toolkit and the version the
    arch needs -- refused up front, not mid-build.  An nvcc whose version
    cannot be read does not block."""
    floor = _nvcc_floor_for_targets()
    if floor is None:
        return ""
    need, arch = floor
    ver = _nvcc_version(os.path.join(cuda_home, "bin", "nvcc"))
    if ver is None or ver >= need:
        return ""
    return (f"the nvcc found ({cuda_home}, CUDA {ver[0]}.{ver[1]}) cannot compile for "
            f"sm_{arch}: compute capability {arch[:-1]}.{arch[-1]} needs CUDA >= "
            f"{need[0]}.{need[1]}. Point CUDA_HOME at a newer toolkit of the same major")


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
        cufft = "-Xlinker " + cufft                     # -l:libcufft.so.<n> is the host linker's spelling
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


# --------------------------------------------------------------------------- #
# the source stamp: when is a cached local core the one this tree would build?
# --------------------------------------------------------------------------- #
_STAMP_NAME = "sources.sha256"


def _source_stamp() -> str:
    """sha256 over the sorted (relative path, file sha256) of every file under
    csrc plus the nvcc defines, -gencode targets and flags a local core build
    uses.  Equal stamps mean the cached core was built from this tree for this
    target; the check needs no toolkit, which is the point (a GPU job reusing
    a core a CPU allocation built, a node with no ``module load cuda``)."""
    h = hashlib.sha256()
    for p in sorted(x for x in _CSRC.rglob("*") if x.is_file()):
        h.update(p.relative_to(_CSRC).as_posix().encode())
        h.update(b"\0")
        h.update(_digest(p).encode())
        h.update(b"\n")
    h.update(b"--\n")
    for flag in _CORE_DEFINES + _gencode_flags() + _CORE_FLAGS:
        h.update(flag.encode())
        h.update(b"\n")
    return h.hexdigest()


def _stamp_path(build_dir: Path) -> Path:
    return build_dir / "core" / _STAMP_NAME


def _write_stamp(build_dir: Path, stamp: str) -> None:
    """Written whole-or-not (tmp + rename): a process reading a half-written
    stamp would fall through to a build it did not need."""
    path = _stamp_path(build_dir)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(stamp + "\n")
    os.replace(tmp, path)


def _local_core_current(build_dir: Path) -> bool:
    """A local core under ``build_dir`` is up to date: its ``.so`` is there and
    the stamp written when it was built equals :func:`_source_stamp` now.
    Anything that cannot be told (no torch, unreadable files) is "not current",
    which falls through to the build path and its own diagnostics."""
    so = build_dir / "core" / "libsweep_core.so"
    stamp = _stamp_path(build_dir)
    if not so.is_file() or not stamp.is_file():
        return False
    try:
        return stamp.read_text().strip() == _source_stamp()
    except Exception:
        return False


def _cached_local_core() -> Path | None:
    """The previously built local core when it is current for this tree and
    target (:func:`_local_core_current` in the default drawer), else None.
    Needs no toolkit."""
    try:
        build_dir = _build_dir()
    except OSError:
        return None
    return build_dir / "core" / "libsweep_core.so" if _local_core_current(build_dir) else None


def _build_core(build_dir: Path, sources: list[str], inc: list[str], cuda_home: str, verbose: bool) -> Path:
    """libsweep_core.so: the torch-free core, nvcc through its own ninja graph
    (incremental; header depfiles).  Serialised with the same baton the staging
    uses, so the ranks of a torchrun job do not build on top of each other.
    A successful build leaves the source stamp beside the ``.so`` (see
    :func:`_source_stamp`), which is what lets the next process reuse it
    without a toolkit."""
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
            if (core_dir / "libsweep_core.so").exists():
                _write_stamp(build_dir, _source_stamp())
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
    if not core_shipped and _core_will_build(build_dir):
        return True
    if not (build_dir / "sweep_C.so").exists() or not (build_dir / "build.ninja").exists():
        return True
    _ensure_ninja_on_path()
    return _ninja_has_work(build_dir)


def _core_will_build(build_dir: Path) -> bool:
    """The core half of :func:`_will_build`: False when the cached local core's
    source stamp is current (nothing to do, no toolkit asked), else whether its
    ninja graph has work (or there is no graph / no .so yet)."""
    if _local_core_current(build_dir):
        return False
    core = build_dir / "core"
    if not (core / "libsweep_core.so").exists() or not (core / "build.ninja").exists():
        return True
    _ensure_ninja_on_path()
    return _ninja_has_work(core)


def _toolkit_includes(cuda_home: str) -> list[str]:
    return [p for p in (os.path.join(cuda_home, "include"),
                        os.path.join(cuda_home, "targets", "x86_64-linux", "include"))
            if os.path.isdir(p)]


def _target_name() -> str:
    """What the one-time notice names as the build target: the device when
    :func:`_gencode_flags` takes it, else the TORCH_CUDA_ARCH_LIST it takes."""
    import torch
    if torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        cap = torch.cuda.get_device_capability()
        return f"your GPU (sm_{cap[0]}{cap[1]})"
    return f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}"


def _build_dir() -> Path:
    """Where a locally built core (and, under SWEEP_JIT_FULL, the compiled shim)
    is cached: torch's extension cache, spelled here so the default path never
    imports torch.utils.cpp_extension.  TORCH_EXTENSIONS_DIR, else
    ~/.cache/torch_extensions/py<ver>_cu<ver>/ (XDG_CACHE_HOME honoured), then
    the ``sweep_C`` drawer -- what ``cpp_extension._get_build_directory("sweep_C")``
    returns on Linux, so the two paths share one staging and one core."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root is None:
        cache = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        ver = _torch_cuda_version()
        cu = "cpu" if ver is None else "cu" + ver.replace(".", "")
        py = f"py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, 'abiflags', '')}"
        root = os.path.join(os.path.realpath(os.path.join(cache, "torch_extensions")), f"{py}_{cu}")
    d = Path(root) / "sweep_C"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# the core the ctypes shim loads
# --------------------------------------------------------------------------- #
def core_path() -> Path:
    """The ``libsweep_core.so`` this process runs on, resolved once.

    The shipped core when it fits (see :func:`_shipped_core`): nothing is
    built, no toolkit is touched.  Else the local core a previous run built,
    when its source stamp is current (:func:`_local_core_current`): reused as
    it is, no toolkit touched either.  Otherwise the core is built here with
    nvcc through :func:`_build_core` (staged sources, its own ninja graph,
    incremental) into the drawer the ``SWEEP_JIT_FULL`` path uses too, so the
    core is never built twice; the toolkit/env setup that build needs --
    CUDA_HOME, nvcc and ninja on PATH -- happens on that branch only.
    ``torch.utils.cpp_extension`` is never imported: the ctypes shim
    (``sweep._capi``) dlopens the result directly -- after :func:`_preload_cufft`,
    which is its call to make -- and nothing compiles.
    """
    global _core_so
    if _core_so is not None:
        return _core_so
    so, core_why = _shipped_core()
    if so is None:
        so = _local_core(core_why)
    _core_so = so
    return so


def _local_core(core_why: str) -> Path:
    """The local branch of :func:`core_path`: the cached core when its stamp is
    current (no toolkit needed), else build it with nvcc."""
    build_dir = _build_dir()
    if _local_core_current(build_dir):
        return build_dir / "core" / "libsweep_core.so"
    ok, why = can_compile()
    if not ok:
        raise RuntimeError(
            f"sweep's compiled backend (impl='c') is unavailable: {why}. "
            "Use impl='eager' for a pure-Python (slower) CPU/GPU path.")
    cuda_home = _find_cuda_home()          # not None: can_compile() checked, no core at hand
    os.environ["CUDA_HOME"] = cuda_home
    os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")
    _ensure_ninja_on_path()
    sources, inc = _stage(build_dir)
    inc = inc + _toolkit_includes(cuda_home)
    building = _core_will_build(build_dir)
    if building:
        print(f"[sweep] no prebuilt CUDA core fits ({core_why}); building the CUDA core "
              f"for {_target_name()} -- one-time, ~2-5 min, then cached at {build_dir} ...",
              file=sys.stderr, flush=True)
    so = _build_core(build_dir, _core_sources(sources), inc, cuda_home, verbose=building)
    if building:
        print("[sweep] CUDA core built and cached.", file=sys.stderr, flush=True)
    return so


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #
def load(compile_only: bool = False):
    """Compile (first call, cached) and return the compiled ``sweep._C`` module
    -- the developer path, ``SWEEP_JIT_FULL=1`` only.

    Two stages: ``libsweep_core.so`` (the torch-free CUDA core, nvcc; skipped
    when a shipped one fits) and the torch shim ``sweep_C`` (a plain C++
    compile against the user's torch, linked to the core).
    ``compile_only=True`` warms the cache without requiring a device, for CI
    and for pre-building on a CPU allocation; ``TORCH_CUDA_ARCH_LIST`` must
    then name the target arch.  The returned module is not usable for kernels
    on a machine with no GPU -- the point is the cached libraries.

    Without ``SWEEP_JIT_FULL`` this raises: ``sweep._C`` is then the ctypes
    shim (``sweep._capi``) over :func:`core_path`, and nothing compiles.
    """
    global _module, _core_so
    if _module is not None:
        return _module
    if not jit_full():
        raise RuntimeError("sweep._C is the ctypes shim (sweep._capi); set "
                           "SWEEP_JIT_FULL=1 for the compiled developer path")
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
        inc = inc + _toolkit_includes(cuda_home)
    else:
        inc = inc + _nvidia_pip_includes()
    building = _will_build(build_dir, core_shipped=core_so is not None)
    if building:
        target = _target_name()
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
    _core_so = core_so                     # core_path() agrees with the module it just linked
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
