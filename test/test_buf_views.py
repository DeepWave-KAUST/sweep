"""The torch-free buffer descriptor's view arithmetic, proven with plain g++.

core/buf.h is the piece of the compiled backend meant to compile WITHOUT
libtorch; its select/narrow/view offsets are what every converted call site
will trust. So they are checked the way they will be used: a C++ program
compiled by the host compiler with no torch on the include or link line, run
in both assert modes -- the JIT build keeps asserts, a wheel build strips
them, and the two must agree (buf_torch.h says why).

Skips, rather than fails, where there is no g++ or no CUDA headers: the
descriptor's one upward include (BoundaryDtype) pulls <cuda_bf16.h>, a
dependency the header itself flags as owed a move into core/.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CSRC = HERE.parent / "src" / "sweep" / "csrc"
SRC = HERE / "csrc" / "test_buf_views.cpp"


def _cuda_include() -> str | None:
    for base in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"),
                 "/usr/local/cuda", "/usr/local/cuda-12.9", "/usr/local/cuda-12.8"):
        if base and (Path(base) / "include" / "cuda_bf16.h").is_file():
            return str(Path(base) / "include")
    try:
        import nvidia.cuda_runtime as r  # the pip wheel ships headers too
        for p in getattr(r, "__path__", []):
            if (Path(p) / "include" / "cuda_bf16.h").is_file():
                return str(Path(p) / "include")
    except Exception:
        pass
    return None


@pytest.mark.parametrize("mode", ["asserts", "ndebug"])
def test_buf_views_torch_free(tmp_path, mode):
    gxx = shutil.which("g++") or shutil.which("c++")
    if gxx is None:
        pytest.skip("no C++ compiler")
    inc = _cuda_include()
    if inc is None:
        pytest.skip("no CUDA headers for <cuda_bf16.h>")
    exe = tmp_path / f"buf_views_{mode}"
    cmd = [gxx, "-std=c++17", "-O1", f"-I{CSRC}", f"-I{inc}", str(SRC), "-o", str(exe)]
    if mode == "ndebug":
        cmd.insert(3, "-DNDEBUG")
    build = subprocess.run(cmd, capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-2000:]
    # The whole point: nothing from a torch install on the include or link
    # line, and none needed.  Checked per ARGUMENT against what a torch
    # dependency would look like -- an include or lib dir under a torch
    # package, or -ltorch/-lc10 -- not as a substring of the command, which
    # this branch's own worktree path ("torch-free-core") satisfies.
    for arg in cmd:
        assert not re.search(r"(site-packages/torch|/libtorch|^-l(torch|c10)|torch/include)", arg), arg
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "GREEN" in run.stdout
