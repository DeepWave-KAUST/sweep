"""``torch.compile`` of a time step, and what to do when the compiler cannot build it.

Inductor's CPU backend shells out to a C++ compiler, ``torch._inductor.config.cpp.cxx``
(torch seeds it from ``$CXX`` when it is imported; the default is ``g++``), and passes
it ``-std=c++20``. g++ 9 (Ubuntu 20.04) rejects that flag, so every compiled CPU step
fails with ``InductorError: CppCompileError``. That costs speed, not the run: the step
is plain PyTorch and gives the same result uncompiled.
"""
from __future__ import annotations

import importlib
import sys
import warnings

import torch

# (device type, what) pairs whose compile already failed in this process. A retry
# would trace the step and call the failing compiler again on every propagator build.
_FAILED: dict = {}


def _resolve_is_compiling():
    """The "is Dynamo tracing this?" probe: ``torch.compiler.is_compiling``, or on
    older 2.x (2.1 has ``torch.compiler`` without it) ``torch._dynamo``'s (the same
    resolution as ``sweep.operators.torch``)."""
    fn = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if fn is None:
        try:
            fn = getattr(importlib.import_module("torch._dynamo"), "is_compiling", None)
        except ImportError:
            fn = None
    return fn or (lambda: False)


is_compiling = _resolve_is_compiling()


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


def compile_or_fallback(fn, device, what: str, options):
    """``torch.compile(fn)`` with the compile settings of ``options`` (an
    :class:`~sweep.propagator.EagerOptions`), run as ``fn`` once the compiler fails.

    ``options=None`` or ``use_compile=False`` returns ``fn`` itself. The guards are the
    FD eager path's, read from ``sweep.propagator._torch_eager`` at call time: a torch
    without ``torch.compile``, or torch 2.3's Inductor on CPU, runs uncompiled after
    one notice per process; Dynamo's recompile cap is raised; the first compiled call
    in the process says that it compiles. The first compiler failure warns once,
    naming ``what`` (e.g. "the ElasticSEM2DDeformed step"), and is remembered per
    ``(device type, what)`` for the rest of the process, so later builds of the same
    step skip the attempt. Errors raised by the step itself propagate as before."""
    if options is None or not options.use_compile:
        return fn
    from sweep.propagator import _torch_eager as fd
    key = (torch.device(device).type, what)
    if key in _FAILED:
        return fn
    inductor = options.compile_backend in (None, "inductor")
    if not hasattr(torch, "compile"):
        _no_compile(fd, f"torch {torch.__version__} has no torch.compile (it needs torch >= 2.0)")
        return fn
    if fd.INDUCTOR_23 and inductor and key[0] != "cuda":
        _no_compile(fd, f"torch {torch.__version__}'s Inductor miscompiles the step on CPU "
                        "(fixed in torch 2.4)")
        return fn
    fd._raise_dynamo_recompile_limit()
    compile_kwargs = {"mode": options.compile_mode, "dynamic": options.compile_dynamic,
                      "fullgraph": options.compile_fullgraph}
    if options.compile_backend is not None:
        compile_kwargs["backend"] = options.compile_backend
    if fd.INDUCTOR_23 and inductor:
        compile_kwargs["options"] = fd._no_buffer_reuse(compile_kwargs.pop("mode"),
                                                        options.compile_dynamic)
    compiled = torch.compile(fd._with_own_code(fn) if fd.OWN_CODE_PER_COMPILE else fn,
                             **compile_kwargs)
    failures = compiler_failures()

    def step(*args, **kwargs):
        if key not in _FAILED:
            if not fd._COMPILE_NOTICE_SHOWN:
                fd._COMPILE_NOTICE_SHOWN = True
                print(fd._COMPILE_NOTICE, file=sys.stderr, flush=True)
            try:
                return compiled(*args, **kwargs)
            except failures as exc:
                _FAILED[key] = exc
                warnings.warn(
                    f"torch.compile could not build {what} on {key[0]} "
                    f"({type(exc).__name__}: {failure_reason(exc)}); it runs uncompiled from now "
                    f"on, which is slower but gives the same result.{compiler_hint(key[0])} "
                    "Pass eager_options=EagerOptions(use_compile=False) to skip the attempt.",
                    RuntimeWarning, stacklevel=2)
        return fn(*args, **kwargs)

    return step


def _no_compile(fd, why):
    """This torch cannot compile the step (``why``): say so once per process, on the
    flag the FD eager path uses for the same notice."""
    if not fd._NO_COMPILE_WARNED:
        fd._NO_COMPILE_WARNED = True
        warnings.warn(
            f"{why}: the eager propagator runs uncompiled, which gives the same result but "
            "is several times slower (3-6x measured on GPU). Pass "
            "eager_options=EagerOptions(use_compile=False) to skip this notice.",
            RuntimeWarning, stacklevel=3)
