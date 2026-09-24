"""Build sweep's compiled backend on first use: the torch-free CUDA core
(``libsweep_core.so``, nvcc) and the thin torch shim (``sweep._C``, a plain C++
compile against the *user's* torch, linked to the core).

This is why a single ``py3-none`` wheel of sweep works with **any** torch version
and any Python 3: nothing torch-specific ships pre-built.  The core depends on
CUDA only, so it is built once per machine (or shipped, step 5); the shim is
what a torch upgrade rebuilds, in about a minute, without nvcc.  First use of
``impl='c'`` pays the one-time compile; every run after that loads the cached
libraries instantly.

The C++ sources ship inside the wheel under ``sweep/csrc/`` (package data).
"""

from __future__ import annotations

import glob
import hashlib
import json
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


def can_compile() -> tuple[bool, str]:
    """(usable, reason) — True when the backend can be COMPILED here.

    Compiling needs torch, an nvcc, and a target architecture. It does **not**
    need a visible device: ``TORCH_CUDA_ARCH_LIST`` names the target explicitly,
    which is how wheels are cross-built, and it is what lets a CI job or a
    CPU-partition allocation warm the cache a later GPU run reuses. Gating the
    compile on a device forces every build to occupy a scarce GPU.

    RUNNING the result still needs a device -- that is :func:`can_build`.
    """
    try:
        import torch
    except Exception:
        return False, "PyTorch is not installed"
    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return False, (
            "no CUDA GPU is visible and TORCH_CUDA_ARCH_LIST is unset, so there "
            "is no target architecture to compile for (set e.g. "
            "TORCH_CUDA_ARCH_LIST=8.9 to build for a card this machine has not got)")
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


def _will_build(build_dir: Path) -> bool:
    """Whether the next load() will actually *compile* (vs reuse the cached
    libraries): the core's ninja graph or the shim's has work.  A cached .so can
    still be rebuilt -- e.g. after the user upgrades torch, whose changed headers
    make the shim recompile -- so "the .so exists" is not the signal; ninja is.
    This drives the one-time "compiling..." notice + verbose output, so a genuine
    rebuild is never a silent hang that looks frozen."""
    core = build_dir / "core"
    if not (core / "libsweep_core.so").exists() or not (core / "build.ninja").exists():
        return True
    if not (build_dir / "sweep_C.so").exists() or not (build_dir / "build.ninja").exists():
        return True
    _ensure_ninja_on_path()
    return _ninja_has_work(core) or _ninja_has_work(build_dir)


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
    building = _will_build(build_dir)
    if building:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            target = f"your GPU (sm_{cap[0]}{cap[1]})"
        else:
            target = f"TORCH_CUDA_ARCH_LIST={os.environ.get('TORCH_CUDA_ARCH_LIST')}"
        print(f"[sweep] compiling the CUDA backend for {target} -- "
              f"one-time, ~2-5 min, then cached at {build_dir} ...",
              file=sys.stderr, flush=True)
    core_so = _build_core(build_dir, _core_sources(sources), inc, cuda_home, verbose=building)
    core_dir = str(core_so.parent)
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
