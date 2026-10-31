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

# The compiles torch.compile has been announced to make in this process, keyed on what
# Dynamo specializes the step on (see compile_notice). Each new one -- a new grid size or
# shot count, a new dtype, a new set of tensors that require grad -- takes seconds to
# minutes and prints nothing.
_ANNOUNCED: set = set()


def compile_notice(what: str, size_of, captured=None):
    """``notice(args, kwargs)``, for a compiled step to call with its arguments before
    each call: says on stderr that torch.compile is about to compile ``what`` (e.g.
    "the Acoustic step") for the size ``size_of(shape)`` names (e.g. "a 201x401
    grid"), once per new compile in the process. The first line of the process also
    says how long it takes and how to turn it off. stderr, not warnings: notebooks
    often silence warnings, and a silent first call looks like a hang.

    A compile is told apart by the first tensor argument's shape and dtype (the first
    of a tuple or list of them) and, when autograd records the call, by which tensors
    require grad -- the arguments, and the tensors ``captured`` holds (the object the
    step reads its coefficients from; searched only in grad mode and when the
    arguments' flags change): Dynamo builds a backward for each such set. The call
    records when grad mode is on and one of those tensors requires grad, and its line
    then says "with gradients"; grad mode alone is not enough, as a forward with
    nothing that requires grad compiles no backward."""
    seen = [None]                     # the last call's (shape, dtype, grad mode, flags)

    def notice(args, kwargs):
        # Every step of a rollout comes here: ~2 us on the loop below.
        lead, flags = None, 1         # a 1, then one bit per tensor: its requires_grad
        for group in (args, kwargs.values()):
            for a in group:
                if isinstance(a, torch.Tensor):
                    if lead is None:
                        lead = a
                    flags = flags << 1 | a.requires_grad
                elif isinstance(a, (tuple, list)):
                    for t in a:
                        if isinstance(t, torch.Tensor):
                            if lead is None:
                                lead = t
                            flags = flags << 1 | t.requires_grad
        grad = torch.is_grad_enabled()
        shape, dtype = (None, None) if lead is None else (lead.shape, lead.dtype)
        if (shape, dtype, grad, flags) == seen[0]:
            return
        seen[0] = (shape, dtype, grad, flags)
        held = _grad_flags(captured) if grad and captured is not None else ()
        # flags & (flags - 1) is 0 when the leading 1 is the only bit set
        records = grad and bool(flags & (flags - 1) or any(held))
        size = size_of(shape)
        # A call that records nothing compiles the one graph without a backward,
        # whatever grad mode and the flags are (Dynamo retraces, Inductor's cache hits).
        key = (what, size, shape, dtype) + ((flags, held) if records else ())
        if key in _ANNOUNCED:
            return
        first = not _ANNOUNCED
        _ANNOUNCED.add(key)
        if dtype is not None and dtype != torch.float32:
            size += f" in {str(dtype).replace('torch.', '')}"
        line = f"sweep: torch.compile is compiling {what} for {size}"
        if records:
            line += " (with gradients)"
        if first:
            line += ("; the first call on each new grid size takes seconds to a few minutes "
                     "(cached on disk for later runs). EagerOptions(use_compile=False) runs "
                     "uncompiled instead.")
        print(line, file=sys.stderr, flush=True)

    return notice


def _grad_flags(obj) -> tuple:
    """The ``requires_grad`` of each tensor in ``obj`` -- at any depth of attributes,
    tuples, lists and dicts, once per tensor, in the order of the walk of
    ``sweep.sem.ckpt_rollout.swap_grad_tensors``."""
    flags, seen = [], set()

    def walk(x):
        if id(x) in seen or isinstance(x, (str, bytes, int, float, complex, type(None), type,
                                           torch.device, torch.dtype)):
            return
        seen.add(id(x))
        if isinstance(x, torch.Tensor):
            flags.append(x.requires_grad)
        elif isinstance(x, (tuple, list)):
            for y in x:
                walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                walk(y)
        elif hasattr(x, "__dict__") and not callable(x):
            for y in vars(x).values():
                walk(y)

    walk(obj)
    return tuple(flags)


class SettledSteps:
    """Which steps of one rollout take the compiled step.

    Dynamo guards on every input's ``requires_grad``.  In a rollout that
    differentiates the models those flags change over the first steps: the
    fields start as zeros and pick up ``requires_grad`` as the models reach
    them.  Each change compiled a graph that then served one step: two of the
    three graphs an FD rollout compiled, two or three of the three or four of a
    SEM full tape.  So steps run uncompiled until their inputs' flags repeat
    from one step to the next.  A step's flags follow from the previous step's
    alone, so they stay put from there, and the first compiled step is the only
    one that compiles.  Without gradients the flags never change and only step
    0 runs uncompiled.

    The answer for a step never changes once given: a checkpoint's recompute
    replays its steps and has to take the path the forward took.
    """

    def __init__(self):
        self.first = None           # the first compiled step
        self._flags = None
        self._plain = set()

    def compiled(self, step, fields):
        if self.first is not None:
            return step >= self.first
        if step in self._plain:
            return False
        flags = tuple(w.requires_grad for w in fields)
        if flags == self._flags:
            self.first = step
            return True
        self._flags = flags
        self._plain.add(step)
        return False


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


def compile_or_fallback(fn, device, what: str, options, size=None, captured=None):
    """``torch.compile(fn)`` with the compile settings of ``options`` (an
    :class:`~sweep.propagator.EagerOptions`), run as ``fn`` once the compiler fails.

    ``options=None`` or ``use_compile=False`` returns ``fn`` itself. The guards are the
    FD eager path's, read from ``sweep.propagator._torch_eager`` at call time: a torch
    without ``torch.compile``, or torch 2.3's Inductor on CPU, runs uncompiled after
    one notice per process; Dynamo's recompile cap is raised. Every new compile -- new
    input shapes or dtype, a new set of tensors that require grad -- is announced
    before it starts (:func:`compile_notice`), for ``size`` (e.g. "a 201x401 grid";
    the inputs' shape when ``None``); ``captured`` is the object ``fn`` reads its
    coefficients from (a bound method's own object when ``None``). The first compiler
    failure warns once,
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
    notice = compile_notice(what, lambda shape: size or f"inputs of shape {tuple(shape or ())}",
                            getattr(fn, "__self__", None) if captured is None else captured)

    def step(*args, **kwargs):
        if key not in _FAILED:
            notice(args, kwargs)
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
