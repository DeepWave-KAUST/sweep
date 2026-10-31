"""Where the host (CPU) core comes from.

The host core is the SAME ``csrc`` tree as the CUDA core (kernels, driver
skeletons, C API), compiled by a C++ compiler instead of nvcc: every ``.cu`` is
fed to g++ as C++ with ``csrc/host/include`` first on the include path, so
``#include <cuda_runtime.h>`` resolves to the host stand-in (``dim3``, the
thread-local ``blockIdx``/``threadIdx``, memcpy/memset for the async copies,
no-op streams and events, float ``atomicAdd``) and ``SWEEP_LAUNCH`` runs each
kernel over its virtual grid with OpenMP (``csrc/host/launch.h``).  There is
no second implementation of anything: an equation the CUDA core runs is an
equation the host core runs, bit-for-bit the same source.

What the host core leaves out is :data:`HOST_UNPORTED` -- ``visco_acoustic2d``
(cuFFT), and the equations the CUDA core gained after the host port
(``elastic_born2d``, the qP pair in ``qp2d``, the SEM kernels).  Their core
entries still exist (the dispatcher names them), as a generated stub TU whose
every entry throws a message naming the equation, and
``sweep_entry_available()`` reports them as absent so the Python resolver can
fall back to eager before any call.

Resolution order, like the CUDA core's (:mod:`.jit`): the wheel's prebuilt
``lib/cpu/libsweep_core.so`` when its sidecar fits (ABI, machine), else a
cached local build whose source stamp is current, else a build with the
host compiler found by :func:`find_host_cxx` -- one time, ~1-3 min, into
``<build_dir>/core_cpu/`` beside the CUDA core's drawer.  An explicit
``impl='c'`` and ``impl='auto'`` resolve the same way; 'auto' builds with the
compiler's output kept in ``build.log`` and, when the build or the load fails,
runs eager (``sweep.propagator.torch._host_core_ready``).  Nothing here
imports torch or touches a GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import jit as _jit

# Equation directories (csrc/cuda/equations/<dir>) the host core does NOT
# compile, each with the core names its header declares: the namespaces, which
# are the entry-name prefixes and the equations' ``C_NAME``.  One table drives
# the TU selection, the generated stub TU and equation_available(), so "built"
# and "available" cannot disagree; the stub generator checks the names against
# the header.  visco_acoustic2d: its spectral step is cuFFT (fft.cu /
# spectral_ops.cu), which has no host twin yet.  elastic_born2d, qp2d and the
# SEM kernels came to the CUDA core after the host port and are not ported yet
# (sweep.sem's compiled path is CUDA-only anyway).
HOST_UNPORTED = {
    "visco_acoustic2d": ("visco_acoustic2d",),
    "elastic_born2d": ("elastic_born2d",),
    "qp2d": ("qp_vti2d", "qp_tti2d", "qp_vti_born2d", "qp_tti_born2d"),
    "sem2d": ("sem2d",),
    "sem3d": ("sem3d",),
}
HOST_UNPORTED_EQUATIONS = tuple(HOST_UNPORTED)                            # the directories
HOST_UNPORTED_NAMES = tuple(n for names in HOST_UNPORTED.values() for n in names)

# The host core's compile flags.  No fast-math and -ffp-contract=off: every
# operation rounds once (what the device's __f*_rn intrinsics promise there),
# so the host core agrees with eager CPU torch at the ULP level and with
# itself from run to run.  -ftls-model=initial-exec keeps the thread-local
# index registers (host/include/cuda_runtime.h) direct %fs loads inside the
# -fPIC shared object; the default general-dynamic model would turn each one
# into a __tls_get_addr call (measured +40-60% on the stencil loop).
# Baseline ISA on purpose: the shipped core must run on any x86-64; the
# emulated launch is scalar anyway, so -march=native buys nothing today.
HOST_FLAGS = [
    "-x", "c++", "-std=c++17", "-O3", "-fPIC", "-fopenmp", "-fvisibility=hidden",
    "-ffp-contract=off", "-ftls-model=initial-exec", "-DSWEEP_HOST_CORE",
    "-Wno-unknown-pragmas", "-Wno-attributes",
]
# -z,defs: an undefined symbol (a stub the generator missed) fails the link,
# not the user's dlopen.  -Bsymbolic-functions: the core's own calls to its
# exported sweep_* functions bind inside the library, so the RTLD_GLOBAL CUDA
# core in the same process (identical export names) can never interpose them.
HOST_LDFLAGS = ["-shared", "-fopenmp", "-Wl,-z,defs", "-Wl,-Bsymbolic-functions"]

# A C++ compiler the host core accepts: g++ >= 9 (C++17 inline variables,
# OpenMP 4.5) or clang++ >= 10.  Spelled as (name, minimum major).
_CXX_MIN = {"g++": 9, "c++": 9, "clang++": 10}

# What a compiler must build before it is offered: an OpenMP region in a shared
# object, with the host core's own compile and link flags.  A compiler past the
# version check can still fail the real build at once -- Apple clang has no
# -fopenmp, clang++ without libomp no <omp.h>, the macOS linker no -z defs --
# and impl='auto' would then start that build in every process.
_PROBE_SRC = """#include <omp.h>
extern "C" int sweep_host_probe() {
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return n + omp_get_max_threads();
}
"""

_core_so: Path | None = None
_cxx_cache: str | None = None
_cxx_rejected: list[str] = []      # why find_host_cxx turned each candidate down


# --------------------------------------------------------------------------- #
# compiler discovery
# --------------------------------------------------------------------------- #
def _cxx_version(cxx: str) -> tuple[str, int] | None:
    """``("g++", 9)`` / ``("clang++", 14)`` for the compiler at ``cxx``, None
    when it does not answer."""
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.splitlines()[0] if out else ""
    if "clang" in first:
        m = re.search(r"version (\d+)", first)
        return ("clang++", int(m.group(1))) if m else None
    m = re.search(r"\b(\d+)\.\d+\.\d+\b", first)
    return ("g++", int(m.group(1))) if m else None


def _first_error(text: str) -> str:
    """The line of a compiler's output that says what went wrong: the first
    ``error:`` line (a ninja command line above it names no error), else the
    last line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "error:" in ln.lower():
            return ln[:300]
    return lines[-1][:300] if lines else "no output"


def _cxx_probe(cxx: str) -> str:
    """'' when ``cxx`` builds :data:`_PROBE_SRC` into a shared object with
    :data:`HOST_FLAGS` and :data:`HOST_LDFLAGS`, else what went wrong."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory(prefix="sweep_cxx_probe_") as tmp:
            src = os.path.join(tmp, "probe.cpp")
            with open(src, "w") as fh:
                fh.write(_PROBE_SRC)
            r = subprocess.run([cxx, *HOST_FLAGS, src, *HOST_LDFLAGS, "-o", os.path.join(tmp, "probe.so")],
                               capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"could not be run ({exc})"
    if r.returncode == 0:
        return ""
    return "does not build an OpenMP shared object: " + _first_error(r.stdout + r.stderr)


def find_host_cxx() -> str | None:
    """The C++ compiler for a host core build: ``CXX`` when set and acceptable,
    else ``g++`` / ``c++`` / ``clang++`` on PATH -- version-checked, then
    probed (:func:`_cxx_probe`).  Cached, with why each candidate was turned
    down (:func:`_no_cxx_reason`)."""
    global _cxx_cache
    if _cxx_cache is not None:
        return _cxx_cache or None
    candidates = []
    if os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    candidates += ["g++", "c++", "clang++"]
    _cxx_rejected.clear()
    tried = set()
    for c in candidates:
        path = shutil.which(c)
        if not path or os.path.realpath(path) in tried:
            continue
        tried.add(os.path.realpath(path))
        v = _cxx_version(path)
        if v is None:
            continue
        family, major = v
        if major < _CXX_MIN.get(family, 9):
            _cxx_rejected.append(f"{path} is {family} {major}")
            continue
        why = _cxx_probe(path)
        if why:
            _cxx_rejected.append(f"{path} ({family} {major}) {why}")
            continue
        _cxx_cache = path
        return path
    _cxx_cache = ""
    return None


def _no_cxx_reason() -> str:
    return ("need g++ >= 9 or clang++ >= 10 with OpenMP on PATH (or in CXX)"
            + ("; " + "; ".join(_cxx_rejected) if _cxx_rejected else ""))


def _cxx_describe(cxx: str) -> str:
    v = _cxx_version(cxx)
    return f"{v[0]} {v[1]}" if v else os.path.basename(cxx)


# --------------------------------------------------------------------------- #
# the stub TU for the unported equations
# --------------------------------------------------------------------------- #
# ``<ret> <name>(<params>);`` at file scope inside the namespace block.  The
# equation headers hold nothing else; a line this cannot read is an error,
# never a silently missing symbol (the link would catch it late and loudly).
_DECL = re.compile(r"^\s*([A-Za-z_][\w:<>]*)\s+([A-Za-z_]\w*)\s*\(([^;]*)\)\s*;\s*(?://.*)?$")
_NS = re.compile(r"^\s*namespace\s+([A-Za-z_]\w*)\s*\{\s*$")
_NS_END = re.compile(r"^\s*\}\s*(?://.*)?$")
_SKIP = re.compile(r"^\s*(#|//|$)")
# What a stub hands back after the throw: never reached, but a non-void
# function needs a value-returning path.
_RETURNS = {
    "ForwardOutputCore": "return ForwardOutputCore{};",
    "BackwardOutputCore": "return BackwardOutputCore{};",
    "RTMOutputCore": "return RTMOutputCore{};",
    "ForwardRunnerCorePtr": "return nullptr;",
    "BackwardRunnerCorePtr": "return nullptr;",
    "size_t": "return 0;",
    "int": "return 0;",
    "bool": "return false;",
    "void": "",
}


def parse_equation_header(path: Path) -> list[tuple[str, list[tuple[str, str, str]]]]:
    """``[(namespace, [(return type, name, parameter list), ...]), ...]`` of an
    equation header (``cuda/equations/<dir>/<dir>.h``): one entry per namespace
    block (qp2d declares four), a declaration may run over several lines."""
    blocks: list[tuple[str, list]] = []
    ns = None
    pending = ""
    for raw in path.read_text().splitlines():
        if not pending and _SKIP.match(raw):
            continue
        if not pending:
            m = _NS.match(raw)
            if m:
                if ns is not None:
                    raise ValueError(f"{path}: nested namespace in an equation header")
                ns = m.group(1)
                blocks.append((ns, []))
                continue
            if _NS_END.match(raw):
                if ns is None:
                    raise ValueError(f"{path}: '}}' outside a namespace block")
                ns = None
                continue
        if ns is None:
            raise ValueError(f"{path}: {raw.strip()!r} outside a namespace block")
        pending = f"{pending} {raw.strip()}" if pending else raw
        if ";" not in raw:
            continue
        line, pending = pending, ""
        m = _DECL.match(line)
        if not m:
            raise ValueError(f"{path}: cannot stub {line.strip()!r}: not a plain declaration")
        ret, name, params = m.group(1), m.group(2), " ".join(m.group(3).split())
        if ret not in _RETURNS:
            raise ValueError(f"{path}: unknown return type {ret!r} of {name}; add it to hostcore._RETURNS")
        blocks[-1][1].append((ret, name, params))
    if pending or ns is not None:
        raise ValueError(f"{path}: unterminated declaration or namespace block")
    if not blocks:
        raise ValueError(f"{path}: no namespace block")
    return blocks


def _unname(params: str) -> str:
    """Drop parameter names (the stub body never reads them)."""
    if not params:
        return ""
    return ", ".join(re.sub(r"\s*\b\w+\s*$", "", p.strip()) for p in params.split(","))


def stub_source(csrc: Path, equations=HOST_UNPORTED_EQUATIONS) -> str:
    """The TU that supplies every core symbol of the unported equations, each
    throwing a message naming the equation, plus ``sweep_entry_is_stub`` (the
    table behind ``sweep_entry_available`` on the host core): an entry is a
    stub when its name starts with ``<namespace>_`` of a stubbed header -- the
    entry names are ``<namespace>_<op>`` (``qp_vti2d_forward``,
    ``sem3d_elastic_vs_forward``)."""
    out = [
        "// GENERATED at build time by sweep.backend.c.hostcore from the equation",
        "// headers: the host core does not build these equations, so their core",
        "// entries throw.  Edit HOST_UNPORTED in hostcore.py, not this file.",
        "#include <cstring>",
        '#include "core/capi.h"',
        '#include "core/check.h"',
        '#include "core/input_core.h"',
        '#include "core/outputs.h"',
        '#include "core/runner.h"',
    ]
    prefixes: list[str] = []
    for eq in equations:
        blocks = parse_equation_header(csrc / "cuda" / "equations" / eq / f"{eq}.h")
        names = tuple(ns for ns, _ in blocks)
        if eq in HOST_UNPORTED and names != HOST_UNPORTED[eq]:
            raise ValueError(f"hostcore.HOST_UNPORTED[{eq!r}] = {HOST_UNPORTED[eq]} but "
                             f"cuda/equations/{eq}/{eq}.h declares {names}")
        out.append(f'#include "cuda/equations/{eq}/{eq}.h"')
        for ns, decls in blocks:
            out.append(f"namespace {ns} {{")
            for ret, name, params in decls:
                out += [
                    f"{ret} {name}({_unname(params)})",
                    "{",
                    f'    SWEEP_CHECK(false, "{ns}::{name}: {ns} is not built into the host (CPU) core; '
                    f"use impl='eager' on the CPU or run it on a CUDA device\");",
                ]
                if _RETURNS[ret]:
                    out.append(f"    {_RETURNS[ret]}")
                out.append("}")
            out.append(f"}}  // namespace {ns}")
            prefixes.append(f"{ns}_")
    out += [
        "",
        "// Read by cuda/common/capi.cu: sweep_entry_available() is 0 for these.",
        "bool sweep_entry_is_stub(int entry)",
        "{",
        "    static const char* const stubbed[] = {",
        *[f'        "{p}",' for p in prefixes],
        "        nullptr,",
        "    };",
        "    const char* name = sweep_entry_name(entry);",
        "    if (name == nullptr) return false;",
        "    for (const char* const* p = stubbed; *p != nullptr; ++p)",
        "        if (std::strncmp(name, *p, std::strlen(*p)) == 0) return true;",
        "    return false;",
        "}",
        "",
    ]
    return "\n".join(out)


def equation_available(c_name: str | None) -> bool:
    """Whether the host core builds the equation with this ``C_NAME`` (its
    namespace and entry prefix; :data:`HOST_UNPORTED` lists the ones it does
    not).  A ``None`` (no compiled kernels at all) is the caller's problem, not
    the host core's."""
    return c_name is None or c_name not in HOST_UNPORTED_NAMES


# --------------------------------------------------------------------------- #
# the build
# --------------------------------------------------------------------------- #
def host_sources(sources: list[str]) -> list[str]:
    """The staged TUs the host core compiles: every ``.cu`` the CUDA core
    compiles, minus the unported equations' directories."""
    skip = tuple(f"/cuda/equations/{eq}/" for eq in HOST_UNPORTED_EQUATIONS)
    return [s for s in sources if s.endswith(".cu") and not any(k in s for k in skip)]


# The host core's long-pole TUs, heaviest first: equations whose kernels.cu
# instantiates its row launchers (host/launch.h launch_rows) for many interior
# paths -- free-surface faces x optional outputs x mechanism count x the five
# stencil orders.  ninja starts ready edges in the order build.ninja lists
# them, and the staged source list puts visco_elastic2d last, so on a 4-core
# laptop its kernels.cu (one of the two heaviest TUs) started last and ran
# alone for over a minute after everything else had finished.  g++ 9 -O3,
# 2 x Xeon 8176, 2026-10: elastic2d ~65 s, visco_elastic2d ~56 s, elastic3d ~40 s,
# elastic_vr2d / elastic_tti_sg2d ~29 s, acoustic_vrz3d ~27 s, acoustic_vrz2d
# ~19 s, elastic_tti_sg3d ~16 s; every other TU ~11 s or less.  Only the
# compile order changes; the link line keeps the staging order.
_HOST_LONG_POLES = ("elastic2d", "visco_elastic2d", "elastic3d", "elastic_vr2d",
                    "elastic_tti_sg2d", "acoustic_vrz3d", "acoustic_vrz2d", "elastic_tti_sg3d")


def _host_compile_order(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """``(obj, src)`` pairs in build order: the :data:`_HOST_LONG_POLES`
    ``kernels.cu`` first, heaviest first, then the rest in their given order.
    (Staged, ``<eq>/kernels.cu`` is ``<eq>/cuda_equations_<eq>_kernels.cu``.)"""
    def rank(item):
        i, (_, src) = item
        s = Path(src)
        if s.name.endswith("kernels.cu") and s.parent.name in _HOST_LONG_POLES:
            return (_HOST_LONG_POLES.index(s.parent.name), i)
        return (len(_HOST_LONG_POLES), i)
    return [pair for _, pair in sorted(enumerate(pairs), key=rank)]


def _host_ninja(core_dir: Path, sources: list[str], inc: list[str], cxx: str, stub_cpp: Path) -> str:
    import shlex
    stage = Path(inc[0])
    shim = stage / "host" / "include"
    # ``-include cuda_runtime.h``: nvcc has __host__/__device__/dim3 built in,
    # so a header may use them without including <cuda_runtime.h> itself
    # (context.h does); the forced include gives g++ the same baseline.
    cflags = " ".join(HOST_FLAGS
                      + [f"-isystem {shlex.quote(str(shim))}",
                         f"-include {shlex.quote(str(shim / 'cuda_runtime.h'))}"]
                      + [f"-I{shlex.quote(d)}" for d in inc])
    lines = [f"cxx = {cxx}", f"cflags = {cflags}", "",
             "rule cxx",
             "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out",
             "  depfile = $out.d", "  deps = gcc", "",
             "rule link", f"  command = $cxx {' '.join(HOST_LDFLAGS)} -o $out $in", ""]
    pairs = [(Path(src).stem + ".o", src) for src in list(sources) + [str(stub_cpp)]]
    for obj, src in _host_compile_order(pairs):
        lines.append(f"build {obj}: cxx {shlex.quote(src)}")
    objs = [obj for obj, _ in pairs]
    lines += ["", f"build libsweep_core.so: link {' '.join(objs)}", "", "default libsweep_core.so", ""]
    return "\n".join(lines)


def source_stamp() -> str:
    """sha256 over every file under csrc plus the host flags and the unported
    list: equal stamps mean the cached host core was built from this tree.
    The compiler is deliberately NOT part of it (the CUDA stamp leaves the
    toolkit out for the same reason): a core built with ``module load gcc/12``
    must stay current on a node that has no acceptable g++ on PATH -- loading a
    .so needs no compiler -- and a different compiler on a later box is no
    reason to rebuild.  Which compiler built a cache is in its ``build.ninja``
    (``cxx = ...``); a shipped core records it in ``core.json``."""
    h = hashlib.sha256()
    for p in sorted(x for x in _jit._CSRC.rglob("*") if x.is_file()):
        h.update(p.relative_to(_jit._CSRC).as_posix().encode())
        h.update(b"\0")
        h.update(_jit._digest(p).encode())
        h.update(b"\n")
    h.update(b"--host\n")
    for flag in HOST_FLAGS + list(HOST_UNPORTED_EQUATIONS):
        h.update(flag.encode())
        h.update(b"\n")
    return h.hexdigest()


def build_dir() -> Path:
    """The host core's drawer: ``<CUDA core build dir>/core_cpu`` -- beside, not
    inside, the CUDA core's ``core/`` so a CUDA torch process can hold both."""
    return _jit._build_dir() / "core_cpu"


def _local_core_current(d: Path) -> bool:
    so, stamp = d / "libsweep_core.so", d / _jit._STAMP_NAME
    if not so.is_file() or not stamp.is_file():
        return False
    try:
        return stamp.read_text().strip() == source_stamp()
    except Exception:
        return False


def cached_local_core() -> Path | None:
    try:
        d = build_dir()
    except OSError:
        return None
    return d / "libsweep_core.so" if _local_core_current(d) else None


def build_core(verbose: bool = True) -> Path:
    """Build the host core (stage, generate the stub TU, ninja) into
    :func:`build_dir` and return the ``.so``; ``python -m sweep.build --cpu
    --core`` copies it into the wheel drawer from there.  Serialised with the
    CUDA core's staging baton; a process that waited on another's build
    takes its core only when it is this tree's, and builds it otherwise.  A
    one-line notice says a build is starting (it takes minutes).  ``verbose``
    streams the compiler's output; else it is kept in ``build.log`` there,
    which a failure names (what ``impl='auto'`` asks for)."""
    cxx = find_host_cxx()
    if cxx is None:
        raise RuntimeError(f"no C++ compiler for the host core: {_no_cxx_reason()}")
    d = build_dir()
    d.mkdir(parents=True, exist_ok=True)
    sources, inc = _jit._stage(_jit._build_dir())
    stub_cpp = d / "host_unported.cpp"
    stub_text = stub_source(Path(inc[0]), HOST_UNPORTED_EQUATIONS)
    if not stub_cpp.exists() or stub_cpp.read_text() != stub_text:
        stub_cpp.write_text(stub_text)
    text = _host_ninja(d, host_sources(sources), inc, cxx, stub_cpp)
    ninja_file = d / "build.ninja"
    if not ninja_file.exists() or ninja_file.read_text() != text:
        ninja_file.write_text(text)
    _jit._ensure_ninja_on_path()
    desc = _cxx_describe(cxx)
    print(f"[sweep] building the host (CPU) core with {desc} -- one-time, "
          f"~1-3 min, then cached at {d} ...", file=sys.stderr, flush=True)
    log = d / "build.log"
    baton = _jit._FileBaton(str(d.parent / "sweep_core_cpu_lock"))
    built_elsewhere = False
    while not baton.try_acquire():
        print(f"[sweep] another process is building the host core (lock {baton.lock_file_path}); "
              "waiting -- if no build is running, that process died: delete the lock.",
              file=sys.stderr, flush=True)
        baton.wait()
        # The holder may have failed, or built another tree's sources into
        # this drawer: its core is ours only when the stamp says so.
        if _local_core_current(d):
            built_elsewhere = True
            break
    if not built_elsewhere:
        try:
            r = subprocess.run(["ninja", "-C", str(d)], capture_output=not verbose, text=True)
            if r.returncode != 0:
                what = f"sweep host core build (libsweep_core.so, {desc}) failed"
                if verbose:
                    raise RuntimeError(f"{what}; the compiler's output is above")
                out = (r.stdout or "") + (r.stderr or "")
                log.write_text(out)
                raise RuntimeError(f"{what}: {_first_error(out)} (the compiler's output is in {log})")
            if log.exists():
                log.unlink()
            if (d / "libsweep_core.so").exists():
                tmp = d / f"{_jit._STAMP_NAME}.{os.getpid()}.tmp"
                tmp.write_text(source_stamp() + "\n")
                os.replace(tmp, d / _jit._STAMP_NAME)
        finally:
            baton.release()
    so = d / "libsweep_core.so"
    if not so.exists():
        raise RuntimeError(f"sweep host core build left no {so}")
    print("[sweep] host core built and cached.", file=sys.stderr, flush=True)
    return so


# --------------------------------------------------------------------------- #
# the shipped core and the resolution order
# --------------------------------------------------------------------------- #
def write_sidecar(out_dir: Path, cxx: str) -> Path:
    """``core.json`` beside a host core: what :func:`_shipped_fits` checks."""
    so = Path(out_dir) / "libsweep_core.so"
    h = hashlib.sha256()
    with open(so, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    meta = {"abi": _jit._shim_abi(), "device": "cpu", "machine": platform.machine(),
            "compiler": _cxx_describe(cxx), "flags": list(HOST_FLAGS), "sha256": h.hexdigest()}
    path = Path(out_dir) / "core.json"
    path.write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return path


def _shipped_fits(sidecar: dict) -> str:
    """'' when the sidecar describes a host core this process can load, else why not."""
    if sidecar.get("device") != "cpu":
        return f"core.json is not a host core (device={sidecar.get('device')!r})"
    want = _jit._shim_abi()
    if sidecar.get("abi") != want:
        return f"core ABI {sidecar.get('abi')} != shim ABI {want}"
    if sidecar.get("machine") not in (None, platform.machine()):
        return f"core was built for {sidecar.get('machine')}, this machine is {platform.machine()}"
    return ""


def shipped_core() -> tuple[Path | None, str]:
    """The wheel's ``lib/cpu/libsweep_core.so`` (or ``SWEEP_CORE_CPU``) when it fits."""
    env = os.environ.get("SWEEP_CORE_CPU")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_file():
            return None, f"SWEEP_CORE_CPU={env} is not a file"
        side = p.with_name("core.json")
        if side.is_file():
            why = _shipped_fits(json.loads(side.read_text()))
            if why:
                return None, f"SWEEP_CORE_CPU: {why}"
        return p, "ok (SWEEP_CORE_CPU)"
    so = _jit._LIB / "cpu" / "libsweep_core.so"
    side = _jit._LIB / "cpu" / "core.json"
    if not so.is_file():
        return None, "the install ships no host (CPU) core (no lib/cpu/libsweep_core.so)"
    if not side.is_file():
        return None, "lib/cpu/libsweep_core.so has no core.json"
    try:
        why = _shipped_fits(json.loads(side.read_text()))
    except (OSError, ValueError) as exc:
        return None, f"lib/cpu/core.json unreadable: {exc}"
    return (None, why) if why else (so, "ok")


def can_build() -> tuple[bool, str]:
    cxx = find_host_cxx()
    if cxx is None:
        return False, f"no C++ compiler for a host core build: {_no_cxx_reason()}"
    return True, f"ok ({_cxx_describe(cxx)})"


def available() -> bool:
    """Whether ``impl='c'`` on a CPU device can be served here: a shipped host
    core fits, a cached local build is current, or a compiler can build one.
    Loads nothing and builds no core (finding a compiler compiles its probe,
    once per process)."""
    if shipped_core()[0] is not None or cached_local_core() is not None:
        return True
    return can_build()[0]


def info() -> dict:
    so, why = shipped_core()
    cached = cached_local_core()
    ok, build_why = can_build()
    return {"shipped": str(so) if so else None, "shipped_reason": why,
            "cached": str(cached) if cached else None, "can_build": ok, "build_reason": build_why,
            "unported": list(HOST_UNPORTED_EQUATIONS), "available": available()}


def core_path(verbose: bool = True) -> Path:
    """The host core this process runs on, resolved once: shipped -> cached
    local -> built here (``verbose``: see :func:`build_core`)."""
    global _core_so
    if _core_so is not None:
        return _core_so
    so, why = shipped_core()
    if so is None:
        so = cached_local_core()
    if so is None:
        ok, build_why = can_build()
        if not ok:
            raise RuntimeError(
                f"sweep's host (CPU) core is unavailable: {why}; and it cannot be built: {build_why}. "
                "Use impl='eager' on the CPU, or install g++ and run `python -m sweep.build --cpu`.")
        so = build_core(verbose=verbose)
    _core_so = so
    return so
