"""Shared test setup.

Also home to ``requires_binding``: the one place that decides whether a test
needing the compiled extension should skip, run, or FAIL.
"""
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# ...and this directory, so a test module can `from conftest import
# requires_binding`. pytest imports conftest as a plugin, which does not put it
# on the import path for the modules it collects.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _require_cuda_env() -> bool:
    """``SWEEP_TEST_REQUIRE_CUDA=1`` turns "no GPU" from a skip into a failure.

    A skip is not a failure, and roughly half this suite is gated on CUDA or on
    a compiled binding, so a run with no GPU prints a green summary over half
    the suite. On a machine that HAS a GPU that is a lie worth catching, which
    is what this switch is for; the bit-exactness gate sets it.
    """
    return os.environ.get("SWEEP_TEST_REQUIRE_CUDA", "").lower() in ("1", "true", "yes", "on")


def binding_status(*symbols):
    """``(ok, reason)`` for the compiled extension and the named symbols.

    Deliberately NOT ``try: import sweep._C; hasattr(...) except Exception``.
    ``sweep._C`` is a lazy shim whose attribute access triggers a JIT compile,
    so that spelling turns a COMPILE FAILURE into a skip: nvcc dies, torch
    raises, the except swallows it, and the tests report green on a build that
    does not exist. ``is_torch_binding_available`` answers from the filesystem
    and the toolchain without compiling anything.
    """
    import torch

    if not torch.cuda.is_available():
        return False, "CUDA device required"
    from sweep import is_torch_binding_available

    if not is_torch_binding_available():
        return False, "compiled sweep._C required (none built, and it cannot be built here)"
    if not symbols:
        return True, ""
    # Past this point the extension exists, so a missing symbol is a STALE or
    # PARTIAL build -- a real problem to report, not a reason to skip.
    from sweep import _C

    missing = [s for s in symbols if not hasattr(_C, s)]
    if missing:
        raise AssertionError(
            f"sweep._C is built but lacks {', '.join(missing)}. That is a stale "
            "or partial build, not a missing capability: rebuild the extension "
            "(delete the TORCH_EXTENSIONS_DIR tree if the sources changed)."
        )
    return True, ""


def requires_binding(*symbols):
    """A ``skipif`` mark for tests that need the compiled extension.

    Three outcomes, deliberately different:

    * the binding is there (with every named symbol) -- the mark does nothing;
    * it is not there -- skip, with a reason that says which part is missing;
    * it is there but a named symbol is not -- raise, because that is a stale
      or partial build rather than a missing capability, and
    * ``SWEEP_TEST_REQUIRE_CUDA=1`` turns the skip into a collection error, so
      a machine that is supposed to have a GPU cannot report green over half
      the suite.
    """
    ok, reason = binding_status(*symbols)          # raises on a stale build
    if ok:
        return pytest.mark.skipif(False, reason="")
    if _require_cuda_env():
        raise RuntimeError(
            f"SWEEP_TEST_REQUIRE_CUDA=1, but {reason}. Unset it to let these "
            "tests skip, or fix the environment."
        )
    return pytest.mark.skipif(True, reason=reason)


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    """The sampled Ricker wavelet the suite uses, in one place.

    Seventeen test modules carried their own copy of this under four different
    signatures -- ``(nt, dt, fm, delay)``, ``(nt, dt, freq, delay)``,
    ``(nt, dt, fm, delay, scale)``, ``(nt, dt, freq, delay, amp)``. Every one of
    them was checked to produce a **bit-identical** array to this body before
    being migrated (three parameter sets each); the ones that were not are still
    local, because they are a different wavelet:

    * ``ricker(t, fm)`` in test_sweep_pytorch / test_elastic_tti_2nd /
      test_elastic_tti_sg3d takes a time ARRAY and uses the
      ``(1 - 0.5 x^2) exp(-0.25 x^2)`` form with ``x = 2*pi*f*t``;
    * ``_ricker(nt, dt, f0)`` in test_visco_acoustic has its own delay rule.

    Do not "simplify" the expression. ``np.arange(nt, dtype=np.float32) * dt``
    promotes to float64 and the cast at the end is what the callers' recorded
    baselines were produced with, so the order of operations is load-bearing.
    """
    import numpy as np

    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg ** 2) * np.exp(-arg ** 2)).astype(np.float32)
