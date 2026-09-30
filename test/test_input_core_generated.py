"""The torch-free input twins are GENERATED from shared/wavetypes.h; this
pins that they have not drifted from it, and that the twin still compiles
with no torch at all.

Regenerating into a temp dir and diffing against the committed headers is
the drift check: a field added to ForwardInput without re-running the
generator fails here, on a CPU runner, in a second.
"""
from __future__ import annotations

import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "src" / "sweep" / "csrc"
GEN = ROOT / "utils" / "gen_input_core.py"


def test_generated_twins_match_wavetypes(tmp_path):
    core = tmp_path / "input_core.h"; adapt = tmp_path / "adapt_inputs.h"
    abi = tmp_path / "abi.py"; layout = tmp_path / "layout.cu"   # the ctypes mirror + the layout probe ride along
    r = subprocess.run([sys.executable, str(GEN), str(CSRC / "shared" / "wavetypes.h"), str(core), str(adapt), str(abi), str(layout)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert filecmp.cmp(core, CSRC / "core" / "input_core.h", shallow=False), \
        "core/input_core.h is stale: re-run utils/gen_input_core.py"
    assert filecmp.cmp(adapt, CSRC / "cuda" / "common" / "adapt_inputs.h", shallow=False), \
        "cuda/common/adapt_inputs.h is stale: re-run utils/gen_input_core.py"

    assert abi.read_text() == (CSRC.parent / "backend" / "c" / "abi.py").read_text(), \
        "sweep/backend/c/abi.py is stale: re-run utils/gen_input_core.py"
    assert layout.read_text() == (CSRC / "cuda" / "common" / "layout.cu").read_text(), \
        "cuda/common/layout.cu is stale: re-run utils/gen_input_core.py"

def test_twins_compile_without_torch(tmp_path):
    gxx = shutil.which("g++") or shutil.which("c++")
    if gxx is None:
        pytest.skip("no C++ compiler")
    inc = None
    for base in ("/usr/local/cuda", "/usr/local/cuda-12.9", "/usr/local/cuda-12.8"):
        if (Path(base) / "include" / "cuda_bf16.h").is_file():
            inc = str(Path(base) / "include"); break
    if inc is None:
        pytest.skip("no CUDA headers for <cuda_bf16.h>")
    probe = tmp_path / "probe.cpp"
    probe.write_text('#include "core/input_core.h"\nint main(){ ForwardInputCore f; BackwardInputCore b; '
                     'return (int)(f.models.size() + b.grads_out.size()); }\n')
    r = subprocess.run([gxx, "-std=c++17", "-fsyntax-only", f"-I{CSRC}", f"-I{inc}", str(probe)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
