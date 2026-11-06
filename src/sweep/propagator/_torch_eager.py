import functools
import os
import types
import warnings
import weakref

import torch
from torch.utils.checkpoint import checkpoint as ckpt_torch


from sweep._compile import (SettledSteps as _SettledSteps, compile_notice,
                            compiler_failures as _compiler_failures, compiler_hint, failure_reason)
from sweep.propagator.base import PropBase
from sweep.propagator.options import EAGER_DEFAULTS, _check_steps_per_call
from sweep.equations._cpml import StripLayout
from sweep.receivers.torch import ReceiverTorch, gather_flat
from sweep.sources.torch import SourceTorch, scatter_source
from sweep.utils.torch import EdgePadding
from sweep.propagator._eager_boundary_saving import _EagerBoundarySavingMixin
from sweep.propagator._torch_eager_custom_grad import _CustomGradientMixin


_DYNAMO_LIMIT_WARNED = False
_NO_COMPILE_WARNED = False

# Compiling inside a checkpointed forward breaks its recompute before torch 2.5
# (see _rollout_ckpt).
CKPT_PLAIN_WARMUP = torch.__version__ < (2, 5)

# An uncompiled rollout makes the step's time-invariant coefficients once
# (``sweep.equations._invariants``).  A/B switch: False recomputes them every
# step, as compiled steps do.
HOIST_INVARIANTS = True

# torch 2.2 keeps the table of compiled backends per thread.  When the autograd
# thread runs a compiled step (a checkpoint recompute, the boundary-saving
# backward), it checks every cache entry on the step's code object -- entries
# other propagators' torch.compile left there too -- and raises KeyError on a
# backend only the main thread knows.  There each propagator compiles its own
# copy of the code; that gives up the reuse of an earlier propagator's graphs,
# which other torch versions keep.
OWN_CODE_PER_COMPILE = (2, 2) <= torch.__version__ < (2, 3)

# torch 2.3's Inductor has two bugs, fixed in 2.4, that the step runs into.
# Its buffer reuse misses that a complex result's ``.real`` still points into
# the buffer and hands it to the next FFT's input: ViscoAcoustic's forward
# turns to NaN, on GPU as on CPU.  Turning reuse off cures that and costs the
# fused step no memory we could measure.  Its C++ backend also fuses a flat
# loop into a group of loops of another shape and fails an assertion, mostly
# while compiling the backward, where no fallback can catch it.  Nothing turns
# that off, so on CPU the step runs uncompiled.
INDUCTOR_23 = (2, 3) <= torch.__version__ < (2, 4)

# Two Inductor options the step compiles with on a 2-D grid on a CUDA device
# from torch 2.9, the version they were measured on, cpp_wrapper where torch
# finds the CUDA toolkit (EagerOptions(compile_cpp_wrapper=...,
# compile_combo_kernels=...) override it: see _resolve_inductor_extras).  On a
# 2-D grid the host time of each compiled call sets the pace, and both cut it
# (a gradient step 1.04-1.21 times faster on an RTX 6000 Ada at 1000^2 under
# boundary saving); a 3-D step keeps the GPU busy, and there they gained
# nothing (combo_kernels made the elastic gradient 1.5 % slower).  cpp_wrapper
# generates the host code of a call (allocations, kernel launches) as C++
# instead of Python: the same kernels, so the same results bit for bit.
# combo_kernels launches the independent small kernels of a graph (C-PML
# strips, source, receivers, boundary ring) as one: results the same up to
# float32 round-off.
INDUCTOR_EXTRAS_DEFAULT = torch.__version__ >= (2, 9)
# The order in which a step that does not compile drops them: a C++ wrapper
# is built with a C++ compiler against the CUDA toolkit (see _cuda_toolkit).
INDUCTOR_EXTRAS = ("cpp_wrapper", "combo_kernels")

# Device types on which Inductor could not build a step's C++ wrapper in this
# process, with the error: later compiles there go without it at once instead
# of failing again (each failure costs the lowering of the graph).
_CPP_WRAPPER_FAILED: dict = {}
_COMBO_FAILED_WARNED = False


def _takes_invariants(equation):
    """Whether the class that defines the equation's ``func`` declares
    ``takes_invariants`` itself: a subclass that overrides ``func`` is not
    handed the dict unless it says it takes it."""
    for cls in type(equation).__mro__:
        if "func" in vars(cls):
            return bool(vars(cls).get("takes_invariants", False))
    return False


def _with_own_code(fn):
    """``fn`` (or a bound method) with a fresh code object, for torch.compile
    caches that must not be shared (see OWN_CODE_PER_COMPILE)."""
    if isinstance(fn, types.MethodType):
        return types.MethodType(_with_own_code(fn.__func__), fn.__self__)
    if not isinstance(fn, types.FunctionType):
        return fn
    own = types.FunctionType(fn.__code__.replace(), fn.__globals__, fn.__name__,
                             fn.__defaults__, fn.__closure__)
    own.__kwdefaults__ = fn.__kwdefaults__
    own.__qualname__ = fn.__qualname__
    own.__dict__.update(fn.__dict__)
    return own


def _mode_options(mode, dynamic):
    """The Inductor options ``mode`` stands for.  torch.compile takes a mode or
    options, not both, so a compile that adds options spells the mode out."""
    from torch._inductor import list_mode_options
    return {} if mode in (None, "default") else dict(list_mode_options(mode, dynamic))


def _no_buffer_reuse(mode, dynamic):
    """Inductor options for ``mode`` with buffer reuse off (see INDUCTOR_23)."""
    return {**_mode_options(mode, dynamic), "allow_buffer_reuse": False}


@functools.lru_cache(maxsize=None)
def _cuda_toolkit():
    """Whether torch finds the CUDA toolkit Inductor builds a C++ wrapper
    against: ``torch.utils.cpp_extension.CUDA_HOME`` ($CUDA_HOME or $CUDA_PATH,
    else the nvcc on the PATH, else /usr/local/cuda) with ``include/cuda.h``.
    Without it the build fails ("CUDA_HOME environment variable is not set";
    "cuda.h: No such file or directory" with CUDA_HOME at a conda environment,
    whose CUDA packages keep their headers under targets/x86_64-linux), so the
    default leaves the wrapper out instead of failing once in every process."""
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except Exception:           # e.g. a torch without its build tools
        return False
    return bool(CUDA_HOME) and os.path.exists(os.path.join(CUDA_HOME, "include", "cuda.h"))


def _cpp_wrapper_companions():
    """The Inductor settings torch puts next to a C++ wrapper (its
    ``get_cpp_wrapper_config``: kernel configs picked at compile time, cubins
    kept, no CUDA graphs).  torch.compile applies its options to every graph it
    compiles, but torch 2.9 adds these to a backward graph -- compiled when
    autograd first runs it, after the forward -- only when the wrapper is on in
    the global config: with the option alone, generating that backward fails
    ("CudaKernelParamCache not populated").  Passed with the option, they reach
    the backward too; the forward graphs get them either way."""
    try:
        from torch._inductor.compile_fx import get_cpp_wrapper_config
    except ImportError:
        return {}
    return dict(get_cpp_wrapper_config())


def _compile_kwargs(mode, dynamic, fullgraph, backend, extras):
    """torch.compile's keywords for the step: the EagerOptions settings, with
    the Inductor ``extras`` (see INDUCTOR_EXTRAS) and torch 2.3's buffer-reuse
    fix as options.  Without either, the mode is passed as it is."""
    kwargs = {"mode": mode, "dynamic": dynamic, "fullgraph": fullgraph}
    if backend is not None:
        kwargs["backend"] = backend
    options = {**_cpp_wrapper_companions(), **extras} if extras.get("cpp_wrapper") else dict(extras)
    if INDUCTOR_23 and backend in (None, "inductor"):
        options["allow_buffer_reuse"] = False
    if options:
        kwargs["options"] = {**_mode_options(kwargs.pop("mode"), dynamic), **options}
    return kwargs


def _warn_extras_dropped(dropped, exc, device_type, what):
    """A step compiled once the Inductor options ``dropped`` were taken out of
    its compile, the last of them after ``exc``: say so, once per process for
    each option.  A C++ wrapper that does not build is the environment's doing,
    so it is remembered for the device type (see _CPP_WRAPPER_FAILED)."""
    global _COMBO_FAILED_WARNED
    if dropped[-1] == "cpp_wrapper":
        if device_type in _CPP_WRAPPER_FAILED:
            return
        _CPP_WRAPPER_FAILED[device_type] = exc
        warnings.warn(
            f"Inductor could not build the C++ wrapper of {what} on {device_type} "
            f"({type(exc).__name__}: {failure_reason(exc)}); the step compiles with the "
            "Python wrapper instead, as do the later compiles of this process. That gives the "
            "same result and is slower when the host sets the pace (one shot on a 2-D grid). "
            "A C++ wrapper is built with a C++ compiler (torch._inductor.config.cpp.cxx, "
            "seeded from $CXX) against the CUDA toolkit under $CUDA_HOME. Pass "
            "eager_options=EagerOptions(compile_cpp_wrapper=False) to skip the attempt.",
            RuntimeWarning, stacklevel=3)
    elif not _COMBO_FAILED_WARNED:
        _COMBO_FAILED_WARNED = True
        warnings.warn(
            f"torch.compile could not build {what} on {device_type} with Inductor's "
            f"combo_kernels ({type(exc).__name__}: {failure_reason(exc)}); it compiles the "
            f"step without {' and '.join(dropped)} instead, which is slower when the host sets "
            "the pace and gives the same result up to float32 round-off. Pass "
            "eager_options=EagerOptions(compile_combo_kernels=False) to skip the attempt.",
            RuntimeWarning, stacklevel=3)


def _record_column(fields, at, taps, footprint, batch, nrec):
    """One record column, ``(batch, nrec, len(at))``: the receivers sampled in
    ``fields[k]`` for each ``k`` in ``at`` (``ReceiverTorch.step_args`` gives
    ``taps`` and ``footprint``)."""
    if len(at) > 1:
        return torch.stack([gather_flat(fields[k], taps, footprint) for k in at],
                           dim=-1).view(batch, nrec, len(at))
    return gather_flat(fields[at[0]], taps, footprint).view(batch, nrec).unsqueeze(-1)


def _default_steps_per_call(ndim, dev, records):
    """``EagerOptions(steps_per_call=None)``: 4 for a rollout autograd records
    (``records``) on a 2-D grid on a GPU, else 1.  A compiled call costs the
    host 0.2-0.5 ms (guards, wrappers, launches) whatever it computes, which on
    a 2-D grid outlasts the step on the GPU.  Without gradients, though,
    Inductor fuses the steps of a call into one another: rather than write a
    step's fields (the CPML memory fields first) it stores derivatives and
    recomputes the fields from them in the next step, which doubled the GPU
    time of a 2000^2 acoustic step; with gradients the fields are saved for the
    backward anyway, and the step runs as fast as one step a call.  A 3-D step
    keeps the GPU busy on its own, and on a CPU the compute dwarfs the
    overhead: there more steps a call would only lengthen the compile."""
    return 4 if records and ndim == 2 and "cuda" in str(dev) else 1


def _uses_cudagraphs(mode, dynamic):
    """Whether torch.compile's ``mode`` replays the step as CUDA graphs, whose
    trees want each call marked (reduce-overhead, max-autotune)."""
    if mode in (None, "default"):
        return False
    try:
        from torch._inductor import list_mode_options
        return bool(list_mode_options(mode, dynamic).get("triton.cudagraphs", False))
    except Exception:           # an unknown mode or torch: mark, as always before
        return True


def _raise_dynamo_recompile_limit(target=32):
    """Raise Dynamo's recompile cap, whatever the installed torch calls it.

    The knob was renamed: ``cache_size_limit`` up to torch 2.5,
    ``recompile_limit`` from 2.6. torch is unpinned here, so keying on the new
    name alone made the bump a silent no-op for every user on an older torch --
    and the cap it was there to lift is exactly what makes Dynamo give up and
    fall back to eager on the many-wavefield equations. ``accumulated_``
    is the secondary cap on the old spelling and has to move with it.
    """
    global _DYNAMO_LIMIT_WARNED
    cfg = getattr(getattr(torch, "_dynamo", None), "config", None)
    if cfg is None:
        return
    names = [n for n in ("recompile_limit", "cache_size_limit",
                         "accumulated_recompile_limit", "accumulated_cache_size_limit")
             if hasattr(cfg, n)]
    primary = [n for n in names if not n.startswith("accumulated_")]
    if not primary:
        if not _DYNAMO_LIMIT_WARNED:
            _DYNAMO_LIMIT_WARNED = True
            warnings.warn(
                f"torch {torch.__version__}'s Dynamo exposes neither "
                "recompile_limit nor cache_size_limit, so the recompile cap "
                "was left alone; torch.compile may fall back to eager on the "
                "many-wavefield equations.", RuntimeWarning, stacklevel=3)
        return
    for name in names:
        if getattr(cfg, name) < target:
            setattr(cfg, name, target)


class _PropTorchEager(
    PropBase, torch.nn.Module, _CustomGradientMixin, _EagerBoundarySavingMixin
):
    def __init__(self, *args, **kwargs):
        self.store_last_wavefield = kwargs.pop("store_last_wavefield", EAGER_DEFAULTS.store_last_wavefield)
        self.use_compile = kwargs.pop("use_compile", EAGER_DEFAULTS.use_compile)
        self.compile_backend = kwargs.pop("compile_backend", EAGER_DEFAULTS.compile_backend)
        self.compile_mode = kwargs.pop("compile_mode", EAGER_DEFAULTS.compile_mode)
        self.compile_dynamic = kwargs.pop("compile_dynamic", EAGER_DEFAULTS.compile_dynamic)
        self.compile_fullgraph = kwargs.pop("compile_fullgraph", EAGER_DEFAULTS.compile_fullgraph)
        self.compile_cpp_wrapper = kwargs.pop("compile_cpp_wrapper", EAGER_DEFAULTS.compile_cpp_wrapper)
        self.compile_combo_kernels = kwargs.pop("compile_combo_kernels", EAGER_DEFAULTS.compile_combo_kernels)
        self.cpml_strips = kwargs.pop("cpml_strips", EAGER_DEFAULTS.cpml_strips)
        self.steps_per_call = _check_steps_per_call(
            kwargs.pop("steps_per_call", EAGER_DEFAULTS.steps_per_call), self.use_compile)
        torch.nn.Module.__init__(self)
        super().__init__(*args, **kwargs)
        self._inductor_extras = self._resolve_inductor_extras()

        self.register_buffer("dt", torch.tensor(self._dt, device=self.dev, dtype=torch.float32))
        self.register_buffer("dh", torch.tensor(self._grid_spacing, device=self.dev, dtype=torch.float32))
        coord_offset = list(self._runtime_coord_offset())
        self.register_buffer("coord_offset", torch.tensor(coord_offset, device=self.dev, dtype=torch.long))
        self.source_indices = [self.wavefield_names.index(name) for name in self.source_type]
        self.receiver_indices = [self.wavefield_names.index(name) for name in self.receiver_type]
        # CPML strips: the eager state holds a strip-held field as its two strips.
        self._strips = self._strip_layout()
        if self._strips is not None:
            self.source_indices = [self._strips.slots(k)[0] for k in self.source_indices]
            self.receiver_indices = [self._strips.slots(k)[0] for k in self.receiver_indices]
        self._equation_spacing = self._normalize_equation_spacing(self._grid_spacing)
        self._workspace_cache = {}
        # User-registered custom gradient / imaging conditions.
        #   {param_name: (imaging_fn, mode)}  with mode in {"gradient", "imaging"}
        self._custom_gradients = {}
        self._plain_step_func = self._build_step_func()
        self._plain_steps_func = self._build_steps_func(self._plain_step_func)
        self._steps_funcs = {}          # steps per call -> its compiled steps_func
        self._mark_steps = (self.use_compile and "cuda" in str(self.dev)
                            and _uses_cudagraphs(self.compile_mode, self.compile_dynamic))

    def _strip_layout(self):
        """The eager state's layout: a ``StripLayout`` when the CPML memory fields
        are kept on the absorbing strips, else None. ``cpml_strips=None`` decides
        by grid: strips in 3-D, the full grid in 2-D (a medium 2-D step is
        launch-bound, and the strips' extra kernels cost more than they save) and
        with topography (its staircase is applied to full-grid fields). Decided
        here, before the CPML profiles exist, so the number of state tensors and
        their positions do not depend on the strip widths."""
        topography = getattr(self, "_topo_rows_runtime", None) is not None
        strips = self.cpml_strips
        if strips is None:
            strips = self.ndim == 3 and not topography
        if not strips:
            return None
        layout = StripLayout(self.equation)
        if len(layout) == len(self.wavefield_names):
            return None                             # no field to strip
        if topography:
            raise NotImplementedError(
                "EagerOptions(cpml_strips=True) with topography=: the staircase is applied to "
                "full-grid fields. Leave cpml_strips unset (the default keeps the full grid "
                "with topography) or pass False.")
        return layout

    def _full_wavefields(self, wavefield):
        """The eager state as one full-grid tensor per wavefield, in wavefield
        order (a strip-held field with its strips put back in place)."""
        return list(wavefield) if self._strips is None else self._strips.full_fields(wavefield)

    def _as_device_tensor(self, value, *, dtype):
        if isinstance(value, torch.Tensor):
            return value.to(device=self.dev, dtype=dtype)
        return torch.as_tensor(value, device=self.dev, dtype=dtype)

    @staticmethod
    def _normalize_equation_spacing(spacing):
        spacing = tuple(float(value) for value in spacing)
        if not spacing:
            return 1.0
        if all(value == spacing[0] for value in spacing):
            return spacing[0]
        return spacing

    def _build_step_func(self):
        equation = self.equation
        strips = self._strips
        num_wavefields = len(self.wavefield_names) if strips is None else len(strips)
        source_indices = tuple(self.source_indices)

        def step_func(*flat_args, source=None, invariants=None):
            wavefields = flat_args[:num_wavefields]
            models = flat_args[num_wavefields:-3]
            dt, h, b = flat_args[-3:]
            # ``equation.func`` is looked up here, not bound outside: Dynamo
            # before torch 2.4 cannot call a bound method held in a closure.
            if strips is not None:
                wavefields = strips.group(wavefields)
            if invariants is None:
                out = equation.func(wavefields, models, dt, h, b)
            else:
                out = equation.func(wavefields, models, dt, h, b, invariants=invariants)
            if strips is not None:
                out = strips.flatten(out)
            if source is None:
                return out
            # Injected inside the step, the compiled graph scatters the source
            # straight into the field it just computed: no full-grid pass.
            out = list(out)
            for k in source_indices:
                out[k] = scatter_source(out[k], *source)
            return out

        return step_func

    def _build_steps_func(self, step):
        """``steps_func(fields, models, dt, h, b, samples, source, receivers)``
        runs one ``step`` (``_build_step_func``) per wavelet sample in
        ``samples``, its source injected (``source`` is the rest of
        ``SourceTorch.step_args``) and then the receivers sampled
        (``receivers``: ``ReceiverTorch.step_args``, batch, nrec).  Returns the
        fields after the last step and the record columns stacked,
        ``(B, len(samples), nrec, len(receiver_type))``.

        Compiled, one call advances ``len(samples)`` steps, the sampling
        included: a call costs the host guards, wrappers and launches whatever
        it computes, which on a 2-D grid outlasts the step on the GPU."""
        at = tuple(self.receiver_indices)

        def steps_func(fields, models, dt, h, b, samples, source, receivers, invariants=None):
            columns = []
            for sample in samples:
                if invariants is None:
                    fields = step(*fields, *models, dt, h, b, source=(sample, *source))
                else:
                    fields = step(*fields, *models, dt, h, b, source=(sample, *source), invariants=invariants)
                columns.append(_record_column(fields, at, *receivers))
            return tuple(fields), torch.stack(columns, dim=1)

        return steps_func

    def _maybe_compile(self, fn, what=None):
        """``torch.compile`` *fn* with the propagator's compile settings, after
        raising Dynamo's recompile limit.  Dynamo specializes the step on
        per-tensor metadata, ``requires_grad`` included, and the graphs of every
        propagator a process builds collect on the step's one code object; at
        the default cap of 8 it silently falls back to eager.  (A rollout runs
        its first steps, while the wavefields' ``requires_grad`` still changes,
        uncompiled: see ``_SettledSteps``.)  Returns *fn* unchanged when
        compilation is off.  Shared by the full-tape and checkpointed steps
        (``_steps_fn``) and the eager boundary-saving step; ``what`` names the
        compile in its notice (default: "the <Equation> step").

        The compile adds the Inductor options of ``_resolve_inductor_extras``
        (by default for a 2-D grid on a CUDA device: cpp_wrapper and
        combo_kernels, see INDUCTOR_EXTRAS).

        The compiled step is wrapped so that a compiler that cannot build it
        (Inductor's CPU backend needs g++ >= 10, for one) costs speed, not the
        run: the step is a pure function of its arguments, so the failing call
        is repeated.  First compiled again without those options, one at a
        time in the order of INDUCTOR_EXTRAS (a C++ wrapper that does not build
        is then left out of every later compile on the device type: see
        _CPP_WRAPPER_FAILED); with none left, uncompiled, and so is every later
        call.  A graph Inductor compiles while autograd runs the backward pass
        (that of a step that records) fails there, out of this wrapper's reach;
        the forward graph of the same step compiles first, with the same
        options and toolchain, and meets a failing C++ compiler before it.
        On torch 2.3, Inductor runs with buffer reuse off, and not at all on
        CPU (INDUCTOR_23)."""
        if not self.use_compile:
            return fn
        if not hasattr(torch, "compile"):
            self._no_compile(f"torch {torch.__version__} has no torch.compile (it needs torch >= 2.0)")
            return fn
        inductor = self.compile_backend in (None, "inductor")
        if INDUCTOR_23 and inductor and "cuda" not in str(self.dev):
            self._no_compile(f"torch {torch.__version__}'s Inductor miscompiles the step on CPU "
                             "(fixed in torch 2.4)")
            return fn
        _raise_dynamo_recompile_limit()
        settings = (self.compile_mode, self.compile_dynamic, self.compile_fullgraph, self.compile_backend)
        device_type = "cuda" if "cuda" in str(self.dev) else "cpu"
        what = what or f"the {type(self.equation).__name__} step"
        extras = dict(self._inductor_extras)
        if extras.get("cpp_wrapper") and device_type in _CPP_WRAPPER_FAILED:
            del extras["cpp_wrapper"]            # it did not build here before: warned then
        target = _with_own_code(fn) if OWN_CODE_PER_COMPILE else fn
        compiled = torch.compile(target, **_compile_kwargs(*settings, extras))
        # Weak: the step ends up inside autograd graphs (the eager boundary-
        # saving Function keeps it in its ctx) that tensors cached on this
        # propagator keep alive.  A strong reference closes a cycle through the
        # C++ graph that gc cannot see, and a dropped propagator would keep its
        # workspace and boundary ring on the device for good.
        owner = weakref.ref(self)
        # Compiling takes seconds to minutes and prints nothing: every new compile --
        # a new grid size or shot count (the wavefields' shape), a new set of models
        # that require grad -- is announced before it starts (sweep._compile). The
        # models are arguments of the step, and _SettledSteps keeps one set of flags
        # per rollout, so the notice repeats a line only for a new compile.
        grid = "x".join(str(n) for n in self._shape_phys)
        notice = compile_notice(what,
                                lambda shape: f"{shape[0]} shots on a {grid} grid"
                                if shape and shape[0] > 1 else f"a {grid} grid")

        def step(*args, **kwargs):
            nonlocal compiled, extras
            prop = owner()
            if prop is not None and not prop.use_compile:
                return fn(*args, **kwargs)
            notice(args, kwargs)
            try:
                return compiled(*args, **kwargs)
            except _compiler_failures() as exc:
                failure = exc
            dropped = []
            while True:
                name = next((n for n in INDUCTOR_EXTRAS if extras.get(n)), None)
                if name is None:
                    if prop is not None:
                        prop._disable_compile(failure)
                    return fn(*args, **kwargs)
                dropped.append(name)
                extras = {k: v for k, v in extras.items() if k != name}
                compiled = torch.compile(target, **_compile_kwargs(*settings, extras))
                try:
                    out = compiled(*args, **kwargs)
                except _compiler_failures() as exc:
                    failure = exc
                    continue
                _warn_extras_dropped(dropped, failure, device_type, what)
                return out

        return step

    def _resolve_inductor_extras(self):
        """The Inductor options ``EagerOptions(compile_cpp_wrapper=...,
        compile_combo_kernels=...)`` add to the step's compile, as
        ``{option: bool}`` (see INDUCTOR_EXTRAS).  ``None`` takes the default:
        both on with Inductor for a 2-D grid on a CUDA device from torch 2.9,
        cpp_wrapper only in a compile mode without CUDA graphs and where torch
        finds the CUDA toolkit (_cuda_toolkit); elsewhere (3-D among them)
        nothing is added and the step compiles as before.  The default looks at
        the dimension, the device, the backend, the mode, the torch version and
        the toolkit only, never at the grid's size or the shots.  An explicit
        value the compile would not use raises."""
        asked = {"cpp_wrapper": self.compile_cpp_wrapper,
                 "combo_kernels": self.compile_combo_kernels}
        explicit = {name: value for name, value in asked.items() if value is not None}
        spelt = ", ".join(f"compile_{name}={value!r}" for name, value in explicit.items())
        inductor = self.compile_backend in (None, "inductor")
        if explicit and not self.use_compile:
            raise ValueError(
                f"EagerOptions({spelt}) sets an Inductor option of the compiled step, and "
                "use_compile=False compiles nothing: drop it, or compile the step.")
        if explicit and not inductor:
            raise ValueError(
                f"EagerOptions({spelt}) sets an Inductor option, which "
                f"compile_backend={self.compile_backend!r} does not read: drop it, or "
                "compile with Inductor.")
        gpu = "cuda" in str(self.dev)
        by_default = INDUCTOR_EXTRAS_DEFAULT and gpu and self.ndim == 2
        if not self.use_compile or not inductor or not (explicit or by_default):
            return {}
        from torch._inductor import config as inductor_config
        # A C++ wrapper turns the CUDA graphs of the mode off (Inductor's
        # get_cpp_wrapper_config): by default the mode keeps them.
        graphs = bool(_mode_options(self.compile_mode, self.compile_dynamic).get("triton.cudagraphs"))
        if explicit.get("cpp_wrapper") and graphs:
            raise ValueError(
                f"EagerOptions(compile_cpp_wrapper=True, compile_mode={self.compile_mode!r}): "
                "Inductor turns off the CUDA graphs of that mode for a C++ wrapper. Use "
                "compile_mode='default' with the C++ wrapper, or drop compile_cpp_wrapper.")
        if explicit.get("combo_kernels") and not gpu:
            raise ValueError(
                "EagerOptions(compile_combo_kernels=True) needs a CUDA device: Inductor "
                f"generates combo kernels for Triton, and compiles the step on {self.dev} "
                "without them. Drop it.")
        default = {"cpp_wrapper": (by_default and not graphs and asked["cpp_wrapper"] is None
                                   and _cuda_toolkit()),
                   "combo_kernels": by_default}
        extras = {}
        for name, value in asked.items():
            if not hasattr(inductor_config, name):
                if value is not None:
                    raise ValueError(
                        f"EagerOptions(compile_{name}={value!r}): torch {torch.__version__}'s "
                        f"Inductor has no {name} option.")
            elif value is not None:
                extras[name] = bool(value)
            elif default[name]:
                extras[name] = True
        return extras

    def _no_compile(self, why):
        """This torch cannot compile the step (``why``): run uncompiled, and
        say so once per process (the step is built per propagator and per
        rollout)."""
        global _NO_COMPILE_WARNED
        self.use_compile = False
        if not _NO_COMPILE_WARNED:
            _NO_COMPILE_WARNED = True
            warnings.warn(
                f"{why}: the eager propagator runs uncompiled, which gives the same result but "
                "is several times slower (3-6x measured on GPU). Pass "
                "eager_options=EagerOptions(use_compile=False) to skip this notice.",
                RuntimeWarning, stacklevel=4)

    def _disable_compile(self, exc):
        """Run uncompiled from now on, and say so once."""
        self.use_compile = False
        hint = compiler_hint("cuda" if "cuda" in str(self.dev) else "cpu")
        warnings.warn(
            f"torch.compile could not build the {type(self.equation).__name__} step on "
            f"{self.dev} ({type(exc).__name__}: {failure_reason(exc)}); running it uncompiled, "
            f"which is slower but gives the same result.{hint} Pass "
            "eager_options=EagerOptions(use_compile=False) to skip the attempt.",
            RuntimeWarning, stacklevel=3)

    def _mark_compile_step_begin(self):
        """Mark a new call for CUDA-graph trees; only the compile modes that
        replay CUDA graphs keep them (``_mark_steps``)."""
        if self._mark_steps and self.use_compile:
            compiler = getattr(torch, "compiler", None)
            if compiler is not None and hasattr(compiler, "cudagraph_mark_step_begin"):
                compiler.cudagraph_mark_step_begin()

    def _resolved_steps_per_call(self, records=True):
        """Steps a compiled call advances: ``steps_per_call``, or when it is
        None, ``_default_steps_per_call`` (``records``: autograd records the
        rollout).  Uncompiled, the steps run one by one."""
        if not self.use_compile:
            return 1
        if self.steps_per_call is not None:
            return self.steps_per_call
        return _default_steps_per_call(self.ndim, self.dev, records)

    @staticmethod
    def _records(models, wavelet_steps):
        """Whether autograd records a rollout of ``models`` and the wavelet:
        grad mode on and one of them requires grad."""
        return torch.is_grad_enabled() and (
            any(m.requires_grad for m in models)
            or any(w.requires_grad for w in wavelet_steps[:1]))

    def _steps_fn(self, k):
        """The compiled ``steps_func`` for calls of ``k`` steps (one wrapper
        per ``k``, for the compile notice; Dynamo keeps the graphs of every
        ``k`` on the function's one code object)."""
        fn = self._steps_funcs.get(k)
        if fn is None:
            name = type(self.equation).__name__
            fn = self._maybe_compile(self._plain_steps_func,
                                     what=f"the {name} step" if k == 1 else f"{k} {name} steps per call")
            self._steps_funcs[k] = fn
        return fn

    def _stepper(self, models, wavelet_steps, src, rec, nt, adj, settled, k, invariants=None):
        """``advance(fields, t, end) -> (fields, block, n)``: steps ``t`` to
        ``t + n - 1`` of the rollout, ``block`` their record columns
        ``(B, n, nrec, len(receiver_type))``.

        Step ``t`` starts a compiled call of ``k`` steps when it is a multiple
        of ``k``, its inputs' ``requires_grad`` has settled (``settled``, the
        rollout's ``_SettledSteps``) and the call ends by ``end``; any other
        step runs alone, uncompiled.  Uncompiled steps are therefore the first
        ones, up to the first multiple of ``k`` after the inputs settle, and
        the last ``nt % k`` -- with no graph compiled for a remainder.  The
        calls sit on multiples of ``k`` whatever ``end`` is, so a rollout cut
        at multiples of ``k`` (checkpoint chunks, the plain warm-up) takes the
        same steps compiled as one that is not, and a checkpoint's recompute
        the same as its forward.  ``invariants`` is the rollout's (or the
        checkpoint chunk's) dict of time-invariant coefficients
        (``_hoists_invariants``), None when compiled."""
        h = self._equation_spacing
        models = tuple(models)
        source = src.step_args(wavelet_steps[0])[1:] if nt else ()
        receivers = (*rec.step_args(), rec.batch, rec.nreceivers)
        compiled = self._steps_fn(k)
        plain = self._plain_steps_func

        # Uncompiled only (``_hoists_invariants``), where ``compiled`` is the
        # plain function as well: a compiled call is handed no dict.
        extra = {} if invariants is None else {"invariants": invariants}

        def sample(t):
            return wavelet_steps[nt - t - 1 if adj else t]

        def advance(fields, t, end):
            if settled.compiled(t, fields) and t % k == 0 and t + k <= end:
                self._mark_compile_step_begin()
                fields, block = compiled(tuple(fields), models, self.dt, h, None,
                                         tuple(sample(t + j) for j in range(k)), source, receivers, **extra)
                return list(fields), block, k
            fields, block = plain(tuple(fields), models, self.dt, h, None, (sample(t),), source, receivers,
                                  **extra)
            return list(fields), block, 1

        return advance

    def _hoists_invariants(self):
        """Whether this rollout's uncompiled steps make their time-invariant
        coefficients once, in a dict (``sweep.equations._invariants``) new for
        the rollout and for each checkpointed chunk (whose recompute makes them
        again).  Decided once, at the start of the rollout: a chunk's recompute
        must save what its forward saved, even after a failed compile turned
        compiling off in between.  Compiled steps compute them in place."""
        return not self.use_compile and HOIST_INVARIANTS and _takes_invariants(self.equation)

    def _sample_receivers(self, rec, wavefield, receivers):
        """One record column, ``(B, nrec, len(receiver_type))``."""
        return _record_column(wavefield, tuple(self.receiver_indices), *rec.step_args(),
                              rec.batch, rec.nreceivers)

    def _prepare_runtime_models(self, models):
        prepare = getattr(self.equation, "prepare_models", None)
        if callable(prepare):
            return prepare(models)
        return models

    def _runtime_batch_layout(self, models, batch_size):
        """Shape padded models to the eager batch layout, per-shot or shared.

        The eager wavefield is laid out as ``(B, 1, *spatial)`` (batch, singleton
        channel, spatial). A model must broadcast against that:

        * **shared** — ``m.ndim == self.ndim`` (e.g. ``(nz, nx)`` in 2-D): the
          model is left untouched so it broadcasts over both the batch and
          channel axes, exactly as before. The shared path is therefore
          bit-identical to the pre-batched-model behaviour.
        * **per-shot** — ``m.ndim == self.ndim + 1`` (e.g. ``(B, nz, nx)``): a
          singleton channel axis is inserted -> ``(B, 1, *spatial)`` so shot
          ``b`` multiplies wavefield ``b`` with model ``b`` (no cross-shot
          broadcast). Without the inserted axis the model's leading batch dim
          would collide with the wavefield's channel dim and broadcast to
          ``(B, B, *spatial)``. Only enabled for equations advertising
          ``supports_batched_models`` (currently the 2-D Acoustic and Elastic
          solvers); every other equation keeps raising rather than silently
          mis-broadcasting.
        """
        laid_out = []
        for m in models:
            if m.ndim == self.ndim:
                laid_out.append(m)
                continue
            if m.ndim == self.ndim + 1:
                if not (self.ndim == 2 and getattr(self.equation, "supports_batched_models", False)):
                    raise NotImplementedError(
                        "Per-shot batched velocity models (a leading batch dim) are "
                        "only supported by 2-D solvers whose kernels stride the "
                        "model per batch index (currently Acoustic and Elastic); "
                        f"got a {m.ndim}-D model for {type(self.equation).__name__} "
                        f"(ndim={self.ndim}). Pass a single shared {self.ndim}-D "
                        "model broadcast across the batch instead."
                    )
                if m.shape[0] != batch_size:
                    raise ValueError(
                        f"Per-shot model batch ({int(m.shape[0])}) must equal the "
                        f"shot batch size ({int(batch_size)}); provide exactly one "
                        "model per shot as (B, nz, nx)."
                    )
                laid_out.append(m[:, None, ...])
                continue
            raise ValueError(
                f"Model tensor has {m.ndim} dims; expected {self.ndim} for a shared "
                f"model or {self.ndim + 1} for a per-shot batched (B, ...) model."
            )
        return laid_out

    def _resolve_snapshot_times(self, nt, return_wavefield, snapshot_times, snapshot_interval):
        if not return_wavefield:
            return []
        if snapshot_times is not None and snapshot_interval is not None:
            raise ValueError("Use either snapshot_times or snapshot_interval, not both.")
        if snapshot_times is not None:
            times = sorted({int(t) for t in snapshot_times})
        elif snapshot_interval is not None:
            interval = int(snapshot_interval)
            if interval < 1:
                raise ValueError("snapshot_interval must be >= 1.")
            times = list(range(0, nt, interval))
        else:
            times = list(range(nt))
        for t in times:
            if t < 0 or t >= nt:
                raise ValueError(f"Snapshot time {t} is outside valid range [0, {nt - 1}].")
        return times

    def _get_cached_tensor(self, kind, shape, *, device, dtype):
        key = (kind, tuple(shape), str(device), str(dtype))
        cached = self._workspace_cache.get(key)
        if cached is None:
            cached = torch.zeros(shape, dtype=dtype, device=device)
            self._workspace_cache[key] = cached
        else:
            cached.detach_()
            cached.zero_()
        return cached

    def set_parameters(self, model):
        assert len(self.model_names) == len(model), (
            f"Model parameters must be the same length as the model names, got {len(model)} and {len(self.model_names)}"
        )
        for name, data in zip(self.model_names, model):
            setattr(self, name, torch.nn.Parameter(data))

    # ------------------------------------------------------------------
    # Custom gradient / imaging-condition registration
    # ------------------------------------------------------------------

    def _pad_models(self, models):
        """EdgePad physical models to runtime grid + apply equation prep."""
        models_p = self._curvilinear_models(
            [self._as_device_tensor(m, dtype=torch.float32) for m in models])
        models_p = [EdgePadding.apply(m, self._runtime_padding()) for m in models_p]
        return models_p, self._prepare_runtime_models(models_p)

    def _crop_to_model(self, runtime_grid, model_shape):
        """A runtime-grid (…, nz_rt, nx_rt) image as the gradient of the
        model: cropped as EdgePadding.backward crops (the pads are per face,
        so not centred: a free surface has none on top), then reshaped to the
        model and, on a boundary-fitted grid, taken back to the model grid."""
        crop = [slice(None)] * runtime_grid.dim()
        pad = self._runtime_padding()
        for i in range(len(pad) // 2):
            lo, hi = pad[2 * i], pad[2 * i + 1]
            crop[-(i + 1)] = slice(lo, -hi if hi > 0 else None)
        g = runtime_grid[tuple(crop)].reshape(model_shape)
        return self._curvilinear_model_grad(g)

    def _rollout_ckpt(self, wavefield, models, wavelet_steps, src, rec,
                      receivers, nt, adj):
        """Time loop under chunk gradient checkpointing: the nt steps are split
        into ``ckpt_chunks``-sized chunks, each wrapped in ``ckpt_torch`` so its
        activations are recomputed in backward instead of stored.  A chunk is
        rounded up to whole compiled calls (``_stepper``).  Returns
        ``(wavefield, record)``, the record concatenated from the chunks' own
        (see ``_rollout_full`` for why it is not written in place).
        """
        k = self._resolved_steps_per_call(self._records(models, wavelet_steps))
        chunk_size = -(-int(max(1, self.ckpt_chunks)) // k) * k
        num_wavefields = len(wavefield)
        num_models = len(models)

        settled = _SettledSteps()
        self._steps_fn(k)              # first, as _rollout_full does
        hoist = self._hoists_invariants()
        chunk_records = []
        start = 0
        if self.use_compile and CKPT_PLAIN_WARMUP:
            # Before torch 2.5, compiling inside a checkpointed forward saves
            # tensors its recompute does not, and backward raises a
            # CheckpointError.  Only the first compiled call compiles (see
            # _SettledSteps), so run the steps up to and including it on the
            # plain tape; it ends on a multiple of k, as the chunks then do.
            advance = self._stepper(self._prepare_runtime_models(models), wavelet_steps, src, rec,
                                    nt, adj, settled, k)
            compiled = False
            while start < min(nt, 8 + 2 * k) and not compiled:
                wavefield, block, n = advance(wavefield, start, nt)
                compiled = n == k and settled.first is not None
                chunk_records.append(block)
                start += n

        for start_t in range(start, nt, chunk_size):

            def checkpoint_chunk(*chunk_inputs, start_t=start_t):
                state = list(chunk_inputs[:num_wavefields])
                chunk_models = list(chunk_inputs[num_wavefields : num_wavefields + num_models])
                advance = self._stepper(self._prepare_runtime_models(chunk_models), wavelet_steps, src,
                                        rec, nt, adj, settled, k, invariants={} if hoist else None)
                end = min(start_t + chunk_size, nt)
                blocks, t = [], start_t
                while t < end:
                    state, block, n = advance(state, t, end)
                    blocks.append(block)
                    t += n
                return tuple(state), torch.cat(blocks, dim=1)

            wavefield, chunk_record = ckpt_torch(checkpoint_chunk, *wavefield, *models, use_reentrant=False)
            wavefield = list(wavefield)
            chunk_records.append(chunk_record)
        return wavefield, torch.cat(chunk_records, dim=1)

    def _rollout_full(self, wavefield, runtime_models, wavelet_steps, src, rec,
                      receivers, nt, adj, return_wavefield, snapshots, snapshot_lookup):
        """Plain time loop recording the full autograd tape (the default path):
        PyTorch retains every step's activations and backward differentiates the
        whole graph.  This is the only path that supports ``return_wavefield``
        (wavefield snapshots), which takes one step a call: a snapshot is the
        state after its step.  Returns ``(wavefield, record)``; the snapshots,
        which are detached host copies, are still written in place.
        """
        k = 1 if return_wavefield else self._resolved_steps_per_call(self._records(runtime_models, wavelet_steps))
        # Set up compiling the calls before deciding on the dict: a torch
        # without torch.compile (or torch 2.3 on a CPU) turns compiling off
        # right there, and the rollout then runs as an uncompiled one does.
        self._steps_fn(k)
        invariants = {} if self._hoists_invariants() else None
        advance = self._stepper(runtime_models, wavelet_steps, src, rec, nt, adj, _SettledSteps(), k,
                                invariants=invariants)
        blocks, t = [], 0
        while t < nt:
            wavefield, block, n = advance(wavefield, t, nt)
            t += n
            if return_wavefield and t - 1 in snapshot_lookup:
                fields = self._full_wavefields([w.detach() for w in wavefield])
                snapshots[snapshot_lookup[t - 1]] = torch.stack(
                    [self._crop_runtime_halo(f).cpu() for f in fields], 0)
            blocks.append(block)
        # One concatenation instead of nt in-place slice writes. Writing into a
        # live autograd tensor builds a chain of nt CopySlices nodes, and EACH of
        # them allocates a full-record buffer and copies the incoming gradient
        # through it, so the record alone cost O(nt^2) backward traffic:
        # 2*nt*|record|, which is 64 GB per shot at nt=4000 with 500 receivers.
        # Concatenating makes it 2*|record|.
        return wavefield, torch.cat(blocks, dim=1)

    def forward(
        self,
        wavelet,
        sources,
        receivers,
        models=None,
        adj=False,
        return_wavefield=False,
        **kwargs,
    ):
        mode, batch_size, nsrc_per_shot, _, source_encoding = self._normalize_io(
            wavelet, sources, receivers
        )
        # physical grid indices onto a boundary-fitted grid, which counts its
        # rows from the surface (identity on a Cartesian grid); both the
        # autograd and the custom-gradient paths below take the mapped ones
        sources = self._place_on_curvilinear(sources, "source")
        receivers = self._place_on_curvilinear(receivers, "receiver")
        # and its snapshots back onto the Cartesian grid, unless the caller
        # wants the grid's own rows (the surface on row 0 of every column)
        snapshot_grid = kwargs.pop("snapshot_grid", None)
        if snapshot_grid is not None and not return_wavefield:
            raise ValueError("snapshot_grid needs return_wavefield=True.")
        if snapshot_grid not in (None, "physical", "computational"):
            raise ValueError(
                f"snapshot_grid must be 'physical' or 'computational'; got {snapshot_grid!r}")
        if models is not None and not isinstance(models, (list, tuple)):
            models = list(models)

        # Custom-gradient routing: when any model has a registered
        # gradient-mode imaging condition, replace the autograd path with the
        # explicit adjoint-state backward (only for plain forward modelling —
        # adj / return_wavefield / ckpt keep the standard path).
        if (self._registered_indices("gradient")
                and not adj and not return_wavefield and not self.use_ckpt):
            return self._custom_gradient_forward(
                wavelet, sources, receivers, models, batch_size, source_encoding, **kwargs
            )

        snapshot_times = kwargs.pop("snapshot_times", None)
        snapshot_interval = kwargs.pop("snapshot_interval", None)
        kwargs.setdefault("fd_pad", self._runtime_fd_pad())
        kwargs.setdefault("shape", self._runtime_shape())
        self.init_abc(**kwargs)
        if self._strips is not None:
            self._strips.size_strips(self.equation, kwargs["shape"], self.dev)

        nt = wavelet.shape[-1]
        snapshot_indices = self._resolve_snapshot_times(nt, return_wavefield, snapshot_times, snapshot_interval)
        snapshot_lookup = {t: i for i, t in enumerate(snapshot_indices)}
        self.snapshot_times_last = tuple(snapshot_indices)

        shape_wavefield = (batch_size, 1) + self._runtime_shape()
        wavelet = self._as_device_tensor(wavelet, dtype=torch.float32)
        # One contiguous view per step: the compiled step takes a step's sample
        # as an input, and a strided one would specialize the compiled code on
        # nt.  Unbinding once also gathers the wavelet's gradient in one stack,
        # where a slice per step cost a full-wavelet backward per step.
        wavelet_steps = wavelet.movedim(-1, 0).contiguous().unbind(0)
        sources = self._as_device_tensor(sources, dtype=torch.long) + self.coord_offset
        receivers = self._as_device_tensor(receivers, dtype=torch.long) + self.coord_offset

        # Equations whose collocated stencils have a checkerboard null space
        # (the rotated staggered grid) declare a small smoothing stencil that
        # is folded into the source injection and the receiver gather so
        # point sources/receivers neither excite nor sample the spurious
        # mode. None (the default) keeps plain single-cell behaviour.
        sr_stencil = getattr(self.equation, "source_receiver_stencil", None)
        src = SourceTorch(sources, shape_wavefield, self.dev, source_encoding, adj,
                          spread_kernel=sr_stencil)
        rec = ReceiverTorch(receivers, gather_kernel=sr_stencil, shape=shape_wavefield)

        has_aux = False
        if return_wavefield:
            has_aux = True
            snapshot_shape = (len(snapshot_indices), len(self.wavefield_names), batch_size, 1) + self.shape
            # NOT through _get_cached_tensor: that cache has no eviction policy,
            # so the snapshot buffer -- whose size grows linearly in the number
            # of snapshot steps, and which defaults to EVERY step -- would be
            # pinned to this propagator for its whole lifetime, long after the
            # caller stopped holding the snapshots. It is also the one workspace
            # that is handed to the caller, so a fresh allocation is what lets
            # it be returned without the defensive clone below, halving the host
            # memory this path costs.
            snapshots = torch.zeros(snapshot_shape, device=torch.device("cpu"),
                                    dtype=torch.float32)
        else:
            snapshots = None

        models = models if models is not None else self.parameters()
        # physical-grid models onto a boundary-fitted grid's rows (identity on
        # a Cartesian grid), differentiably: the gradient is on the model grid
        models = self._curvilinear_models(
            [self._as_device_tensor(para, dtype=torch.float32) for para in models])
        models = [EdgePadding.apply(para, self._runtime_padding()) for para in models]
        self.models_padded = models
        # Insert a singleton channel axis for per-shot batched models so each
        # shot propagates in its own model; shared models pass through unchanged
        # (keeping the shared path bit-exact). Done after storing models_padded
        # so that attribute stays the raw padded model, matching the C backend.
        models = self._runtime_batch_layout(models, batch_size)
        runtime_models = self._prepare_runtime_models(models)
        shapes = ([shape_wavefield] * len(self.wavefield_names) if self._strips is None
                  else self._strips.shapes(shape_wavefield))
        wavefield = [
            self._get_cached_tensor(f"wavefield:{index}", shape, device=self.dev, dtype=torch.float32)
            for index, shape in enumerate(shapes)
        ]
        if self.use_ckpt and self.ckpt_mode != "chunk":
            raise ValueError(f"Unsupported ckpt_mode '{self.ckpt_mode}' for PropTorch. Expected 'chunk'.")
        if self.use_ckpt and return_wavefield:
            raise ValueError("return_wavefield=True is not supported with chunk checkpointing in PropTorch yet.")

        # Every rollout builds its record fresh (stacked, never a workspace
        # buffer), so it is returned without a defensive copy.
        if self.use_ckpt:
            wavefield, record = self._rollout_ckpt(
                wavefield, models, wavelet_steps, src, rec, receivers, nt, adj
            )
        elif getattr(self, "_eager_bs", False) and (
                return_wavefield or self._records(runtime_models, wavelet_steps)):
            # A call autograd does not record has no backward to save a boundary
            # for: under boundary saving it takes the plain steps below, as
            # impl='c' skips its boundary (return_wavefield still raises).
            if return_wavefield:
                raise ValueError("return_wavefield is not supported with eager boundary saving.")
            wavefield, record = self._rollout_eager_bs(
                wavefield, runtime_models, wavelet_steps, src, rec, receivers, nt, adj
            )
        else:
            wavefield, record = self._rollout_full(
                wavefield, runtime_models, wavelet_steps, src, rec, receivers, nt, adj,
                return_wavefield, snapshots, snapshot_lookup,
            )

        self.last_wavefields = tuple(self._full_wavefields(wavefield)) if self.store_last_wavefield else None
        if not has_aux:
            return record
        # ``snapshots`` is allocated per call above, so it aliases no workspace
        # and needs no defensive copy -- and may be mapped in place.
        if snapshot_grid != "computational":
            snapshots = self._curvilinear_snapshots(snapshots)
        return record, snapshots

    forward_base = forward
