"""``torch.compile`` of a time step, and what to do when the compiler cannot build it.

Inductor's CPU backend shells out to a C++ compiler, ``torch._inductor.config.cpp.cxx``
(torch seeds it from ``$CXX`` when it is imported; the default is ``g++``), and passes
it ``-std=c++20``. g++ 9 (Ubuntu 20.04) rejects that flag, so every compiled CPU step
fails with ``InductorError: CppCompileError``. That costs speed, not the run: the step
is plain PyTorch and gives the same result uncompiled.
"""
from __future__ import annotations

import warnings

import torch

# (device type, what) pairs whose compile already failed in this process. A retry
# would trace the step and call the failing compiler again on every propagator build.
_FAILED: dict = {}


def compiler_failures() -> tuple:
    """Exception types meaning the compiler failed, as opposed to the code it
    compiles: a step that is wrong eagerly raises its own error and must keep
    doing so. The names moved between torch releases, so collect what exists."""
    found = []
    for module, name in (("torch._dynamo.exc", "BackendCompilerFailed"),
                         ("torch._inductor.exc", "InductorError"),
                         ("torch._inductor.exc", "CppCompileError")):
        try:
            found.append(getattr(__import__(module, fromlist=[name]), name))
        except (ImportError, AttributeError):
            pass
    return tuple(found)


def failure_reason(exc) -> str:
    """The line of a compiler failure worth showing: the compiler's own ``error:``
    line when the exception carries its output (an Inductor CppCompileError puts
    the command line first and the error after it), else the first line."""
    lines = str(exc).strip().splitlines()
    for line in lines:
        if "error:" in line and "C++ compile error" not in line:
            return line.strip()[:200]
    return (lines[0] if lines else type(exc).__name__)[:200]


def compiler_hint(device_type: str) -> str:
    """How to give Inductor a compiler that builds, for the warnings."""
    if device_type != "cpu":
        return ""
    return (" On CPU, Inductor compiles with torch._inductor.config.cpp.cxx (seeded from "
            "$CXX when torch is imported, default g++), which must take -std=c++20, i.e. "
            "g++ >= 10: point it at a newer compiler, or set TORCH_INDUCTOR_INSTALL_GXX=1 "
            "to let Inductor fetch one through conda.")


def compile_or_fallback(fn, device, what: str):
    """``torch.compile(fn)``, run as ``fn`` once the compiler fails.

    The first failure warns once, naming ``what`` (e.g. "the ElasticSEM2DDeformed
    step"), and is remembered per ``(device type, what)`` for the rest of the
    process, so later builds of the same step skip the attempt. Errors raised by
    the step itself propagate as before."""
    key = (torch.device(device).type, what)
    if key in _FAILED:
        return fn
    compiled = torch.compile(fn)
    failures = compiler_failures()

    def step(*args, **kwargs):
        if key not in _FAILED:
            try:
                return compiled(*args, **kwargs)
            except failures as exc:
                _FAILED[key] = exc
                warnings.warn(
                    f"torch.compile could not build {what} on {key[0]} "
                    f"({type(exc).__name__}: {failure_reason(exc)}); it runs uncompiled from now "
                    f"on, which is slower but gives the same result."
                    f"{compiler_hint(key[0])} Pass compile_step=False to skip the attempt.",
                    RuntimeWarning, stacklevel=2)
        return fn(*args, **kwargs)

    return step
