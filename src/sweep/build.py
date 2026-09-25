"""``python -m sweep.build`` — compile the CUDA backend ahead of time.

Exists so the build can happen where builds are cheap. Compiling needs a C++
compiler and, only when no shipped core fits, nvcc plus a target architecture --
not a card -- so this runs on a CPU node or in a CI image
and leaves a cached ``sweep_C.so`` that the GPU run picks up::

    TORCH_CUDA_ARCH_LIST=7.0 TORCH_EXTENSIONS_DIR=/scratch/ext python -m sweep.build

Without ``--no-gpu-required`` it behaves exactly like ``sweep.precompile()`` and
insists on a visible device.

``--core`` is the release-side mode: it builds the torch-free core alone as a
fat binary and drops it where the wheel picks it up (``sweep/lib/<cuN>/``) with
its ``core.json`` sidecar, so a user's first import compiles only the shim::

    python -m sweep.build --core --archs '7.0;7.5;8.0;8.6;8.9;9.0+PTX'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

DEFAULT_ARCHS = "7.0;7.5;8.0;8.6;8.9;9.0+PTX"

# Where a pip-installed nvidia-cufft lands relative to sweep/lib/<tag>/: up to
# site-packages, then into the wheel's lib dir. Spelt for ninja ($$ -> $) and
# single-quoted below so /bin/sh does not expand $ORIGIN into nothing.
CUFFT_RPATH = "$$ORIGIN/../../../nvidia/cufft/lib"

# No -static-libstdc++: a static libstdc++ inside a shared library leaves its
# locale/iostream globals uninitialised (the core's error text lost every
# number: "sweep_call: no entry " with the id missing).  The release core links
# the host's libstdc++ dynamically; build it on the oldest host you support.
RELEASE_LINK_FLAGS: list = []


def _core_tag(cuda_version: str) -> str:
    from sweep import _jit
    tag = getattr(_jit, "_core_tag", None)
    return tag(cuda_version) if tag else "cu" + cuda_version.split(".")[0]


def _core_abi() -> int:
    from sweep import _jit
    text = (_jit._CSRC / "core" / "capi.h").read_text()
    m = re.search(r"^\s*#define\s+SWEEP_CORE_ABI_VERSION\s+(\d+)", text, re.M)
    if not m:
        raise RuntimeError("SWEEP_CORE_ABI_VERSION not found in csrc/core/capi.h")
    return int(m.group(1))


def parse_arch_list(archs: str) -> tuple[list[str], list[str]]:
    """``"7.0;9.0+PTX"`` -> (["70", "90"], ["90"]): torch's spelling in, the
    sidecar's out."""
    sm: list[str] = []
    ptx: list[str] = []
    for a in archs.replace(",", ";").split(";"):
        a = a.strip()
        if not a:
            continue
        has_ptx = a.endswith("+PTX")
        a = (a[:-4] if has_ptx else a).replace(".", "")
        if not a.isdigit():
            raise ValueError(f"bad arch {a!r} in --archs {archs!r}")
        sm.append(a)
        if has_ptx:
            ptx.append(a)
    if not sm:
        raise ValueError("--archs names no architecture")
    return sm, ptx


def write_core_sidecar(out_dir: Path, archs: list[str], ptx: list[str],
                       cuda_version: str, flags: list[str]) -> Path:
    so = Path(out_dir) / "libsweep_core.so"
    h = hashlib.sha256()
    with open(so, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    meta = {"abi": _core_abi(), "cuda": cuda_version, "archs": list(archs),
            "ptx": list(ptx), "sha256": h.hexdigest(), "flags": list(flags)}
    path = Path(out_dir) / "core.json"
    path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return path


def _with_release_link_flags(text: str) -> str:
    """``_core_ninja``'s graph with the cuFFT rpath and the static C++ runtime
    on its link rule. Patched on the ninja text rather than through a
    ``_build_core`` parameter: the release build is the only caller that wants
    them."""
    anchor = "  command = $nvcc -shared -o $out $in"
    if text.count(anchor) != 1:
        raise RuntimeError("sweep._jit._core_ninja link rule changed; update build.py")
    extra = " ".join([f"-Xlinker '-rpath={CUFFT_RPATH}'"] + RELEASE_LINK_FLAGS)
    return text.replace(anchor, f"{anchor} {extra}")


def _scratch_dir(tag: str) -> Path:
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root:
        return Path(root) / f"sweep_core_{tag}"
    try:
        import getpass
        who = getpass.getuser()
    except Exception:
        who = "sweep"
    return Path(tempfile.gettempdir()) / f"sweep_core_{tag}_{who}"


def build_core(archs: str = DEFAULT_ARCHS, out: Path | None = None,
               cuda_home: str | None = None, verbose: bool = True) -> Path:
    from sweep import _jit

    if cuda_home is None:
        cuda_home = _jit._find_cuda_home()
        if cuda_home is None:
            raise RuntimeError("no suitable nvcc found; pass --cuda-home or set CUDA_HOME")
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.exists(nvcc):
        raise RuntimeError(f"{nvcc} does not exist")
    ver = _jit._nvcc_version(nvcc)
    if ver is None:
        raise RuntimeError(f"cannot read the release of {nvcc}")
    cuda_version = f"{ver[0]}.{ver[1]}"
    tag = _core_tag(cuda_version)
    sm, ptx = parse_arch_list(archs)

    # _gencode_flags reads the env; set it before anything computes cflags, and
    # unconditionally so a visible device does not narrow the fat binary.
    os.environ["TORCH_CUDA_ARCH_LIST"] = archs
    os.environ["CUDA_HOME"] = cuda_home
    build_dir = _scratch_dir(tag)
    build_dir.mkdir(parents=True, exist_ok=True)
    sources, inc = _jit._stage(build_dir)
    inc = inc + [p for p in (os.path.join(cuda_home, "include"),
                             os.path.join(cuda_home, "targets", "x86_64-linux", "include"))
                 if os.path.isdir(p)]
    flags = _jit._CORE_DEFINES + _jit._gencode_flags() + _jit._CORE_FLAGS

    if verbose:
        print(f"[sweep] building libsweep_core.so ({tag}, nvcc {cuda_version}) for "
              f"sm {' '.join(sm)}, ptx {' '.join(ptx) or '-'}, in {build_dir} ...",
              file=sys.stderr, flush=True)
    # _build_core looks _core_ninja up on the module, so the link flags ride
    # in on a temporary wrapper; restored whatever happens.
    orig = _jit._core_ninja
    _jit._core_ninja = lambda *a, **k: _with_release_link_flags(orig(*a, **k))
    try:
        so = _jit._build_core(build_dir, _jit._core_sources(sources), inc, cuda_home, verbose)
    finally:
        _jit._core_ninja = orig

    out_dir = Path(out) if out is not None else _jit._PKG / "lib" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "libsweep_core.so"
    dst.write_bytes(so.read_bytes())
    sidecar = write_core_sidecar(out_dir, sm, ptx, cuda_version, flags)
    print(f"{dst} ({dst.stat().st_size / 2**20:.1f} MB)")
    print(str(sidecar))
    return dst


def _shipped_core_fits() -> bool:
    try:
        from sweep import _jit
        return _jit._shipped_core()[0] is not None
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sweep.build",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-gpu-required", action="store_true",
                    help="build without a visible device; TORCH_CUDA_ARCH_LIST "
                         "must name the target architecture")
    ap.add_argument("--core", action="store_true",
                    help="build only the torch-free core as a fat binary for the "
                         "wheel (no GPU needed); see --archs/--out/--cuda-home")
    ap.add_argument("--archs", default=DEFAULT_ARCHS,
                    help=f"--core: TORCH_CUDA_ARCH_LIST spelling (default {DEFAULT_ARCHS!r})")
    ap.add_argument("--out", default=None,
                    help="--core: output dir (default: inside the installed sweep "
                         "package, sweep/lib/<tag>; src/sweep/lib/<tag> in a clone)")
    ap.add_argument("--cuda-home", default=None,
                    help="--core: toolkit with bin/nvcc (default: what sweep would use)")
    a = ap.parse_args(argv)

    if a.core:
        try:
            build_core(a.archs, Path(a.out) if a.out else None, a.cuda_home)
        except Exception as exc:
            print(f"core build failed: {exc}", file=sys.stderr)
            return 1
        return 0

    # With a shipped core that fits, only the plain-C++ shim compiles here, so
    # there is no architecture to name.
    if (a.no_gpu_required and not os.environ.get("TORCH_CUDA_ARCH_LIST")
            and not _shipped_core_fits()):
        print("TORCH_CUDA_ARCH_LIST is unset, so there is no architecture to "
              "build for. Set it to your target card, e.g. "
              "TORCH_CUDA_ARCH_LIST=8.9 (Ada) or 7.0 (V100).", file=sys.stderr)
        return 2

    import sweep

    try:
        sweep.precompile(require_gpu=not a.no_gpu_required)
    except Exception as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1

    from torch.utils import cpp_extension
    where = cpp_extension._get_build_directory("sweep_C", verbose=False)
    print(f"sweep._C is built and cached at {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
