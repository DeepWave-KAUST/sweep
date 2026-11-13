"""An edit to a CUDA stand-in header must recompile the objects that include it.

The host core reaches the stand-ins in ``csrc/host/include`` -- ``<cuda.h>``,
``<cuda_fp16.h>``, ``<cuda_bf16.h>``, ``<cuda_runtime.h>`` -- through
``-isystem``, so the compiler counts them as system headers.  With ``-MMD``,
which leaves system headers out of the depfile, ninja knew of none of them but
the force-included ``cuda_runtime.h``: after an edit to ``cuda.h`` the staged
copy changed, ninja found nothing to do, and ``build_core`` wrote the new
source stamp beside the old objects -- a stale core taken as current from then
on.  The rule now passes ``-MD``, which records every header the compiler opens.

One small TU that includes every stand-in by its CUDA name goes through the
rule the build writes (``hostcore._host_ninja``, sources staged by
``jit._stage`` from a scratch csrc); no core is built.  A C++ compiler and
ninja are all it needs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from sweep.backend.c import hostcore, jit

STAND_INS = sorted(p.name for p in (jit._CSRC / "host" / "include").glob("*.h"))


class _Probe:
    """A scratch csrc -- the real ``host/`` plus one TU -- staged and built
    the way ``hostcore.build_core`` does it."""

    def __init__(self, root: Path, cxx: str):
        self.csrc = (root / "csrc").resolve()
        shutil.copytree(jit._CSRC / "host", self.csrc / "host")
        tu = self.csrc / "cuda" / "common" / "probe.cu"
        tu.parent.mkdir(parents=True)
        tu.write_text("".join(f"#include <{h}>\n" for h in STAND_INS)
                      + "int sweep_probe() { return 0; }\n")
        self.build_dir = root / "build"
        self.core = self.build_dir / "core_cpu"
        self.core.mkdir(parents=True)
        self.stub = self.core / "host_unported.cpp"
        self.stub.write_text("")
        self.cxx = cxx
        self.obj: Path | None = None
        self.stage: Path | None = None

    def build(self, edited: str | None = None) -> bytes:
        """Stage the scratch csrc, write build.ninja, build the probe's object
        and return its bytes.  ``edited``: a stand-in edited since the last
        build, whose staged copy is then dated past the object, so the test
        does not depend on the file system's timestamp resolution."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(jit, "_CSRC", self.csrc)
            sources, inc = jit._stage(self.build_dir)
        self.stage = Path(inc[0])
        (src,) = [s for s in sources if s.endswith("probe.cu")]
        self.obj = self.core / (Path(src).stem + ".o")
        (self.core / "build.ninja").write_text(
            hostcore._host_ninja(self.core, [src], inc, self.cxx, self.stub))
        if edited is not None:
            later = self.obj.stat().st_mtime_ns + 1_000_000_000
            os.utime(self.stage / "host" / "include" / edited, ns=(later, later))
        self.ninja(self.obj.name)
        return self.obj.read_bytes()

    def ninja(self, *args: str) -> str:
        r = subprocess.run(["ninja", "-C", str(self.core), *args], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    cxx = hostcore.find_host_cxx()
    if cxx is None:
        pytest.skip(f"no C++ compiler for the host core: {hostcore._no_cxx_reason()}")
    jit._ensure_ninja_on_path()
    if shutil.which("ninja") is None:
        pytest.skip("no ninja on PATH")
    p = _Probe(tmp_path_factory.mktemp("stand_in_deps"), cxx)
    p.build()
    return p


def test_the_depfile_names_every_stand_in(probe):
    deps = probe.ninja("-t", "deps", probe.obj.name)
    missing = [h for h in STAND_INS if str(probe.stage / "host" / "include" / h) not in deps]
    assert not missing, (f"ninja does not know that {probe.obj.name} includes {missing}; "
                         f"an edit to them would leave the object as it is:\n{deps}")


def test_an_edit_to_cuda_h_recompiles_its_includers(probe):
    """cuda.h: the stand-in most of the core's objects include, through the
    equation kernels' headers, and one that -MMD left out."""
    name = "sweep_probe_edit_cuda_h"
    with open(probe.csrc / "host" / "include" / "cuda.h", "a") as fh:
        fh.write(f'\nstatic const char {name}[] __attribute__((used)) = "{name}";\n')
    rebuilt = name.encode() in probe.build(edited="cuda.h")
    assert rebuilt, f"cuda.h was edited, but ninja did not recompile {probe.obj.name}"
