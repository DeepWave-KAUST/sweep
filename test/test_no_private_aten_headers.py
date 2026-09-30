"""No translation unit may include a PRIVATE ATen header.

`ATen/native/**` is libtorch's own implementation tree, not its API. Its
contents move between torch minors -- names, layout helpers and macros all
differ between 2.5, 2.8 and 2.9 -- so an `#include` of one is a compile-time
break waiting for the next `pip install -U torch`, in a tree whose whole
selling point is that it compiles against whatever torch you already have.

This tree had exactly one: visco_acoustic2d used
`at::native::detail::CuFFTConfig` to build the spectral step's cuFFT plan.
It now issues cufftCreate / cufftSetAutoAllocation / cufftXtMakePlanMany
itself, with the same arguments, which is bit-identical and depends only on
the public cuFFT API.

Scanning sources rather than compiling is deliberate: this must fail on a CPU
runner, in seconds, for the person who added the include -- not three months
later on someone else's torch.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

CSRC = Path(__file__).resolve().parent.parent / "src" / "sweep" / "csrc"

# `#include <ATen/native/...>` or the quoted form. Only include DIRECTIVES:
# a comment naming the header (this file's own docstring, say) is not a
# dependency, and a test that cannot tell the difference gets disabled.
PRIVATE_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"](ATen/native/[^>"]+)[>"]', re.MULTILINE)

SOURCE_SUFFIXES = {".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc"}


def _sources() -> list[Path]:
    return sorted(p for p in CSRC.rglob("*") if p.suffix in SOURCE_SUFFIXES)


def test_csrc_has_sources_to_scan():
    """Guard the guard: an empty scan would pass the check below forever."""
    files = _sources()
    assert len(files) > 50, f"only {len(files)} sources under {CSRC} -- wrong path?"


def test_no_private_aten_headers():
    offenders: list[str] = []
    for path in _sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for m in PRIVATE_INCLUDE.finditer(text):
            line = text[: m.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(CSRC)}:{line}: {m.group(1)}")
    assert not offenders, (
        "private ATen headers are included by:\n  " + "\n  ".join(offenders)
        + "\n\nATen/native/** is libtorch's implementation tree, not its API; it "
          "changes between torch minors. Use the public library directly (cuFFT, "
          "cuBLAS, the c10/ATen public headers) or move the work to Python."
    )


@pytest.mark.parametrize("sample", [
    '#include <ATen/native/cuda/CuFFTPlanCache.h>',
    '  #  include  "ATen/native/foo.h"',
])
def test_the_pattern_actually_matches(sample):
    """The regex is the whole test; a typo in it would pass everything."""
    assert PRIVATE_INCLUDE.search(sample), sample


@pytest.mark.parametrize("sample", [
    '// ATen/native/cuda/CuFFTPlanCache.h is private -- do not include it',
    '#include <ATen/cuda/CUDAContext.h>',
    '#include <c10/cuda/CUDAGuard.h>',
])
def test_the_pattern_does_not_over_match(sample):
    assert not PRIVATE_INCLUDE.search(sample), sample
