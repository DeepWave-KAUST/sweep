"""``sweep`` settles oneMKL's VML CPU dispatch on import (``sweep/_mkl_vml.py``).

The race it closes cannot be provoked on demand: it takes a fresh process whose first
multi-threaded VML op loses it -- one in five at 4 threads on an AVX-512 Xeon with torch 2.14.1,
none on AMD EPYC. There, ``test_biot``'s undrained limit failed in one fresh process out of four,
its ``vp = sqrt(H / rho)`` 3e-4 off in a worker's OpenMP chunk. So this checks that a fresh
interpreter is settled once the equations are imported, with ``import sweep`` still torch-free.
"""
import os
import subprocess
import sys

from conftest import SRC
from sweep import _mkl_vml


def test_a_fresh_interpreter_is_settled_by_the_equations_import():
    probe = ("import sys, sweep; torch_free = 'torch' not in sys.modules;"
             "import sweep.equations; from sweep import _mkl_vml;"
             "print(torch_free, _mkl_vml._settled)")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env,
                         check=True)
    assert out.stdout.split()[-2:] == ["True", "True"], out.stdout + out.stderr


def test_settling_again_is_harmless():
    _mkl_vml.settle_mkl_vml_dispatch(force=True)
    assert _mkl_vml._settled
