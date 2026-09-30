import dataclasses
import inspect
import warnings

import torch

from sweep.propagator._torch_eager import _PropTorchEager
from sweep.core.arguments import warn_deprecated_spelling, resolve_device
from sweep.propagator.options import (
    _applicable_boundary_knobs,
    CUDAOptions,
    EAGER_OPTION_KEYS,
    CUDA_OPTION_KEYS,
    BOUNDARY_DEFAULTS,
    BoundaryOptions,
    BoundarySaving,
    CKPT_DEFAULTS,
    Ckpt,
    options_to_dict,
    resolve_memory_strategy,
    as_memory_strategy,
    check_memory_supported,
    to_legacy_memory_options,)


SUPPORTED_BACKENDS = {"torch"}
SUPPORTED_IMPLS = {"eager", "c", "auto"}
LEGACY_BACKEND_IMPLS = {
    "eager": "eager",
    "cuda": "c",
    "c": "c",
}
IMPL_ALIASES = {}


def _normalize_impl(value):
    value = str(value).lower()
    value = IMPL_ALIASES.get(value, value)
    if value not in SUPPORTED_IMPLS:
        raise ValueError(
            f"Unsupported PropTorch impl '{value}'. Expected 'eager', 'c', or 'auto'."
        )
    return value


def _compiled_binding_available():
    from sweep import is_torch_binding_available
    return is_torch_binding_available()


def _equation_supports_c(equation):
    """True when the equation has compiled kernels: a ``C_NAME`` and the
    ``_C()`` hook it installs. A hand-written ``_C`` that only refuses
    impl='c' (the curvilinear pair) does not count, so 'auto' runs those
    eager and an explicit 'c' falls back with the usual warning."""
    if equation is None:
        return True
    return callable(getattr(equation, "_C", None)) and bool(getattr(equation, "C_NAME", None))


def _device_type(device):
    """``'cuda'`` / ``'cpu'`` / ... for a torch.device, a string (``'cuda:0'``)
    or an index; None when no device was given."""
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device.type
    try:
        return torch.device(device).type
    except (TypeError, RuntimeError, ValueError):
        return str(device).split(":", 1)[0].lower()


def _resolve_impl_with_fallback(impl, *, explicit, equation=None, device=None):
    """Resolve 'auto' / 'c' to a concrete impl, falling back to 'eager' when
    the compiled binding is unavailable, the equation has no ``_C`` hook, or
    the propagator is not on a CUDA device.

    The compiled backend runs on CUDA only, in every build (the ctypes layer
    over the prebuilt core, the ``SWEEP_JIT_FULL=1`` pybind shim, an AOT-built
    extension all reach the same CUDA core), so a CPU propagator -- or any
    other device -- is eager.  No device given keeps the compiled path.

    When the user explicitly asked for 'c' but the path isn't available, emit
    a UserWarning so the slowdown is visible.
    """
    binding_ok = _compiled_binding_available()
    equation_ok = _equation_supports_c(equation)
    dtype = _device_type(device)
    device_ok = dtype is None or dtype == "cuda"

    if impl == "auto":
        return "c" if (binding_ok and equation_ok and device_ok) else "eager"

    if impl == "c" and not (binding_ok and equation_ok and device_ok):
        if explicit:
            eq_name = type(equation).__name__ if equation is not None else "<unknown>"
            if not device_ok:
                reason = (f"the compiled backend runs on CUDA only and this propagator "
                          f"is on {device}")
                remedy = "Move to a CUDA device (dev='cuda') for the compiled path."
            elif not binding_ok:
                reason = "the compiled binding (sweep._C) is unavailable"
                remedy = (
                    "Rebuild with `pip install -e .` in a CUDA-enabled env "
                    "to use the compiled path."
                )
            else:
                reason = f"equation '{eq_name}' does not expose a compiled CUDA kernel"
                remedy = "Use impl='eager' (or omit impl) for this equation."
            warnings.warn(
                f"PropTorch(impl='c') requested but {reason}; falling back to "
                f"impl='eager' (pure-PyTorch, ~10-30x slower). {remedy}",
                UserWarning,
                stacklevel=3,
            )
        return "eager"
    return impl


def _normalize_backend_impl(backend, impl, *, equation=None, device=None):
    backend = str(backend).lower()
    if backend == "pytorch":
        backend = "torch"

    explicit_impl = impl is not None

    if backend in LEGACY_BACKEND_IMPLS:
        legacy_impl = LEGACY_BACKEND_IMPLS[backend]
        if impl is not None and _normalize_impl(impl) not in (legacy_impl, "auto"):
            raise ValueError(
                f"Legacy PropTorch backend='{backend}' implies impl='{legacy_impl}', got impl='{impl}'."
            )
        return "torch", _resolve_impl_with_fallback(legacy_impl, explicit=explicit_impl, equation=equation, device=device)

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            "Unsupported PropTorch backend "
            f"'{backend}'. Expected backend='torch'. Use PropJax for backend='jax'."
        )

    normalized = _normalize_impl("auto" if impl is None else impl)
    return backend, _resolve_impl_with_fallback(normalized, explicit=explicit_impl, equation=equation, device=device)


def _merge_option_dict(base, extra, *, label):
    if extra is None:
        return base
    extra = options_to_dict(extra)
    overlap = set(base) & set(extra)
    if overlap:
        raise ValueError(f"Duplicate option keys between top-level kwargs and {label}: {sorted(overlap)}")
    return {**base, **extra}


def _normalize_cuda_memory_kwargs(merged, equation=None):
    memory = merged.pop("memory", None)

    # The gradient-memory mode is a three-way choice (full/boundary/ckpt),
    # resolved once and identically for every backend; conflicting knob
    # combinations raise instead of one path silently winning.  Mixing
    # memory= with a legacy knob is only an error when the two disagree --
    # `memory=MemoryOptions(strategy='boundary'), use_ckpt=False` says one
    # thing twice -- so the resolver, not a presence check, decides.
    strategy = resolve_memory_strategy(
        "c", options_to_dict(memory) if memory is not None else None,
        merged.get("use_ckpt"), merged.get("boundary_saving_config"))

    # Equations whose CUDA kernels cannot run boundary saving (e.g.
    # ViscoAcoustic: the dissipative, global-FFT damping term is not
    # reverse-time reconstructible from boundary strips) never take the
    # 'boundary' default — they fall back to 'full' (exact, no recompute;
    # ckpt stays available as an explicit request) — and an EXPLICIT
    # boundary request raises instead of silently running a wrong
    # reconstruction.
    if strategy == "boundary" and equation is not None and \
            not getattr(equation, "supports_boundary_saving_c", True):
        mem_dict = options_to_dict(memory) if memory is not None else None
        explicit = bool(
            (mem_dict or {}).get("strategy") == "boundary"
            or dict(merged.get("boundary_saving_config") or {}).get("enabled")
        )
        if explicit:
            raise NotImplementedError(
                f"{type(equation).__name__} does not support boundary saving "
                "on impl='c'; use memory=Ckpt() or memory=Full() "
                "(from sweep.propagator.options).")
        strategy = "full"

    if strategy == "full":
        merged["use_ckpt"] = False
        cfg = dict(merged.get("boundary_saving_config") or {})
        cfg["enabled"] = False
        merged["boundary_saving_config"] = cfg
        return merged

    memory = options_to_dict(memory) if memory is not None else None
    if memory is None:
        # Legacy-knob (or default) route: normalise the pair so exactly one
        # mode is active downstream.
        if strategy == "boundary":
            cfg = dict(merged.get("boundary_saving_config") or {})
            cfg.setdefault("storage", "gpu")
            cfg["enabled"] = True
            merged["boundary_saving_config"] = cfg
            merged["use_ckpt"] = False
        else:  # 'ckpt' -- chunk options ride the legacy ckpt_* kwargs
            merged["use_ckpt"] = True
            cfg = dict(merged.get("boundary_saving_config") or {})
            cfg["enabled"] = False
            merged["boundary_saving_config"] = cfg
        return merged

    if strategy == "boundary":
        boundary = memory.get("boundary") or {}
        boundary_config = {
            "enabled": True,
            "storage": boundary.get("storage", BOUNDARY_DEFAULTS.storage),
        }
        for key in (
            "transfer_interval",
            "pinned_memory",
            "disk_dir",
            "ring_buffers",
            "disk_async_read",
            "storage_dtype",
            "tail_steps",
        ):
            if key in boundary:
                boundary_config[key] = boundary[key]
        merged["boundary_saving_config"] = boundary_config
        merged["use_ckpt"] = False
        return merged

    if strategy == "ckpt":
        ckpt = memory.get("ckpt") or {}
        mode = ckpt.get("mode", CKPT_DEFAULTS.mode)
        merged.update(
            {
                "use_ckpt": True,
                "ckpt_mode": mode,
                "ckpt_storage": ckpt.get("storage", CKPT_DEFAULTS.storage),
            }
        )
        if "pinned_memory" in ckpt:
            merged["ckpt_pinned_memory"] = ckpt["pinned_memory"]
        if mode == "chunk":
            merged["ckpt_chunks"] = ckpt.get("chunks", CKPT_DEFAULTS.chunks)
        elif mode == "recursive":
            merged["ckpt_num"] = ckpt.get("count", CKPT_DEFAULTS.count)
        else:
            raise ValueError(f"Unsupported ckpt mode '{mode}'. Expected 'chunk' or 'recursive'.")
        return merged

    raise ValueError(f"Unsupported c memory strategy '{strategy}'. Expected 'boundary' or 'ckpt'.")


def _assert_backend_matches_strategy(backend_impl, strategy):
    """The constructed backend must agree with the strategy that was resolved.

    Not a tautology: the flags reach the backend through several translations
    (cuda_options -> init kwargs, the legacy dict, per-backend defaults), and a
    disagreement between "what was decided" and "what was built" is silent --
    it produces a run that measures one thing while reporting another. That
    exact failure is on record downstream as "thought I measured bs, actually
    ran ckpt". Assert the state rather than the intention.
    """
    got_ckpt = bool(getattr(backend_impl, "use_ckpt", False))
    cfg = getattr(backend_impl, "boundary_saving_config", None) or {}
    # ``_eager_bs`` is what enable_eager_boundary_saving() actually sets; the
    # eager backend has no boundary_saving_config to read.
    got_bs = bool(cfg.get("enabled", False)) or bool(
        getattr(backend_impl, "_eager_bs", False))
    built = "ckpt" if got_ckpt else ("boundary" if got_bs else "full")
    if built != strategy:
        raise RuntimeError(
            f"gradient-memory strategy resolved to {strategy!r} but the backend "
            f"was built as {built!r} (use_ckpt={got_ckpt}, boundary_saving={got_bs}). "
            "This is a bug in the option plumbing, not in the caller's arguments.")


def _apply_eager_memory(backend_impl, memory):
    """Apply a ``MemoryOptions`` to an already-built eager backend.

    Mirrors :func:`_normalize_cuda_memory_kwargs` for the pure-PyTorch path:
    ``strategy='boundary'`` enables boundary-saving wavefield reconstruction,
    ``strategy='ckpt'`` enables chunk checkpointing.  Boundary saving and
    checkpointing are mutually exclusive (the eager forward checks ``use_ckpt``
    first), so each branch disables the other.
    """
    # Same bridge as the impl='c' path: this function reads the tagged wire
    # format, and the new types keep their strategy in a ClassVar, which
    # asdict() does not emit.
    md = options_to_dict(to_legacy_memory_options(memory))
    strategy = md.get("strategy")
    if strategy is None:
        raise ValueError("memory= must set strategy to 'full', 'boundary' or 'ckpt'.")

    if strategy == "full":
        backend_impl._set_memory_strategy("full")
        backend_impl.enable_eager_boundary_saving(False)
        return

    if strategy == "boundary":
        boundary = md.get("boundary") or {}
        storage = boundary.get("storage", "gpu")
        if storage not in ("gpu", "cpu"):
            raise ValueError(
                "Eager boundary saving supports storage='gpu' or 'cpu' (the ring "
                "buffer is kept on device, or offloaded to host RAM); disk staging "
                "is available on the impl='c' path."
            )
        # storage_dtype compresses the ring (fp16/bf16/int8); storage='cpu' moves
        # it off device.  Compute and the seed frame stay FP32 / on the compute
        # device (same split as the CUDA path).
        storage_dtype = boundary.get("storage_dtype", "fp32")
        if storage_dtype not in ("fp32", "fp16", "bf16", "int8"):
            raise ValueError(
                "Eager boundary saving storage_dtype must be 'fp32'/'fp16'/'bf16'/"
                f"'int8', got {storage_dtype!r}."
            )
        backend_impl._set_memory_strategy("boundary")
        backend_impl.enable_eager_boundary_saving(
            True, storage=storage, storage_dtype=storage_dtype
        )
        return

    if strategy == "ckpt":
        ckpt = md.get("ckpt") or {}
        mode = ckpt.get("mode", CKPT_DEFAULTS.mode)
        if mode != "chunk":
            raise ValueError("Eager checkpointing supports mode='chunk' only.")
        backend_impl.enable_eager_boundary_saving(False)
        backend_impl._set_memory_strategy("ckpt")
        backend_impl.ckpt_mode = "chunk"
        backend_impl.ckpt_chunks = ckpt.get("chunks", CKPT_DEFAULTS.chunks)
        return

    raise ValueError(f"Unsupported memory strategy '{strategy}' for impl='eager'.")


_TF32_WARNED = False


def _warn_tf32_once(dev) -> None:
    """The eager stencils are cuDNN convolutions.  On Ampere and newer torch
    lets cuDNN run them in TF32 by default (a 10-bit mantissa), which moves
    an eager 3-D forward by ~1e-4 and its gradients by 1e-2..1e-1 relative
    against FP32 -- enough to fail a comparison with the compiled core, which
    is always FP32.  Say so once; the user decides."""
    global _TF32_WARNED
    if _TF32_WARNED:
        return
    try:
        import torch
        device = torch.device(dev) if dev is not None else torch.device("cpu")
        if device.type != "cuda" or not torch.backends.cudnn.allow_tf32:
            return
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            return
    except Exception:
        return
    _TF32_WARNED = True
    warnings.warn(
        "impl='eager' on this GPU runs its stencil convolutions in TF32 "
        "(torch.backends.cudnn.allow_tf32 is True): expect ~1e-4 relative "
        "error in 3-D forwards and 1e-2..1e-1 in their gradients against FP32. "
        "Set torch.backends.cudnn.allow_tf32 = False for full precision; "
        "impl='c' is always FP32.", stacklevel=3)


def _resolve_backend_init_kwargs(*, impl, kwargs, backend_options, eager_options, cuda_options, equation=None):
    if eager_options is not None and impl != "eager":
        raise ValueError("eager_options can only be used with impl='eager'.")
    if cuda_options is not None and impl != "c":
        raise ValueError("cuda_options can only be used with impl='c'.")

    wrong_top_level = (CUDA_OPTION_KEYS if impl == "eager" else EAGER_OPTION_KEYS) & set(kwargs)
    if wrong_top_level:
        target = "cuda_options" if impl == "c" else "eager_options"
        raise ValueError(
            f"Top-level kwargs {sorted(wrong_top_level)} do not belong to impl='{impl}'. "
            f"Pass implementation-specific options through {target} or switch impl."
        )

    merged = _merge_option_dict(dict(kwargs), backend_options, label="backend_options")
    selected_options = eager_options if impl == "eager" else cuda_options
    option_label = "eager_options" if impl == "eager" else "cuda_options"
    merged = _merge_option_dict(merged, selected_options, label=option_label)

    wrong_merged = (CUDA_OPTION_KEYS if impl == "eager" else EAGER_OPTION_KEYS) & set(merged)
    if wrong_merged:
        raise ValueError(f"Invalid {option_label} for impl='{impl}': {sorted(wrong_merged)}")

    if impl == "c":
        merged = _normalize_cuda_memory_kwargs(merged, equation=equation)
    return merged


def _public_forward_signature(forward):
    signature = inspect.signature(forward)
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        signature = signature.replace(parameters=parameters[1:])
    return signature


class PropTorch(torch.nn.Module):
    """PyTorch propagator.  ``impl='eager'`` is pure torch (CPU or GPU);
    ``impl='c'`` is the prebuilt CUDA core (``sweep/lib/cu<major>/libsweep_core.so``)
    driven through the pure-Python ctypes layer ``sweep.backend.c`` -- CUDA GPU only,
    nothing compiles after ``pip install``.  ``impl=None`` means ``'auto'``: ``'c'``
    when ``sweep.is_torch_binding_available()`` and the equation declares ``C_NAME``,
    else ``'eager'``; an explicit ``impl='c'`` that cannot be honoured falls back to
    eager with a UserWarning.  ``backend=None`` inherits the equation's backend.
    ``cuda_options=`` applies to ``impl='c'`` only, ``eager_options=`` to
    ``impl='eager'``; ``memory=`` (``Full()`` / ``BoundarySaving()`` / ``Ckpt()``)
    is impl-agnostic (default: boundary saving on 'c', chunked ckpt on eager).
    """
    def __init__(
        self,
        *args,
        backend=None,
        impl=None,
        backend_options=None,
        eager_options=None,
        cuda_options=None,
        memory=None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        equation = args[0] if args else kwargs.get('equation')
        if backend is None:
            # Inherit from the equation (positional arg 0); equation owns backend
            # because its operators were already built against it.
            backend = getattr(equation, 'backend', 'torch')
        requested_impl = impl
        backend, impl = _normalize_backend_impl(backend, impl, equation=equation, device=resolve_device(kwargs.get("device"), kwargs.get("dev"), equation))

        # ``memory=`` is the impl-agnostic, dataclass-style memory-strategy API
        # (dict-style config still goes through boundary_saving_config=/use_ckpt=).
        # For impl='c' it folds into cuda_options.memory; for impl='eager' it is
        # applied to the backend after construction (see below).
        # Deprecation warnings fire HERE, at the boundary the caller crosses.
        # Anything further down sees the internal wire format, which this file
        # synthesises from the new API -- warning there would tell a caller who
        # used the new spelling that they used the old one.
        if "boundary_saving_config" in kwargs:
            warn_deprecated_spelling(
                "boundary_saving_config={...}",
                "memory=BoundarySaving(...) / Full() / Ckpt(...)")

        if memory is not None:
            # Full()/BoundarySaving()/Ckpt() are the current spelling; a legacy
            # MemoryOptions and a dict both still read. as_memory_strategy is
            # the single place that knows all of them -- and the only place a
            # deprecation warning for a legacy spelling is raised, so it must be
            # called on what the CALLER passed and never on something this file
            # converted. Otherwise the new API warns about itself.
            memory = as_memory_strategy(memory)
        elif impl == "c" and getattr(cuda_options, "memory", None) is not None:
            # cuda_options.memory is the caller's own request too, and is read
            # exactly like memory=. The new types keep their strategy in a
            # ClassVar that options_to_dict() drops, so handing them down as-is
            # left the layers below with no strategy: Ckpt()/Full() clashed
            # with the 'boundary' default, and BoundarySaving(storage='cpu')
            # quietly became the default gpu ring. CUDAOptions carries nothing
            # else, so it is rebuilt from the normalised request below.
            memory = as_memory_strategy(cuda_options.memory)
            cuda_options = None
        _caller_request = memory
        if memory is not None and impl == "c":
            if cuda_options is not None and getattr(cuda_options, "memory", None) is not None:
                raise ValueError(
                    "Specify the memory strategy via either memory= or "
                    "cuda_options.memory, not both."
                )
            # The layers below read the tagged wire format; convert once here
            # rather than teaching each of them the new types.
            cuda_options = CUDAOptions(memory=to_legacy_memory_options(memory))
            memory = None

        if impl == "eager" and (cuda_options is not None or CUDA_OPTION_KEYS & set(kwargs)):
            requested_norm = (
                _normalize_impl(requested_impl) if requested_impl is not None else "auto"
            )
            if requested_norm in ("c", "auto"):
                # impl='c' was demoted to eager (no binding, or the equation has
                # no _C). The cuda-only knobs go, but the memory STRATEGY is not
                # a cuda-only knob -- it just travelled inside cuda_options, and
                # dropping the object took the request with it, leaving the
                # eager backend on its own default ('ckpt'). So a caller asking
                # for bf16 cpu-staged boundary saving quietly got checkpointing.
                # Carry the request across instead: anything eager cannot honour
                # then raises from check_memory_supported, naming the option.
                carried = getattr(cuda_options, "memory", None)
                if carried is not None and memory is None:
                    # _caller_request too, or the resolution below still reads
                    # None and disagrees with the backend it just built.
                    memory = _caller_request = as_memory_strategy(carried)
                cuda_options = None
                kwargs = {k: v for k, v in kwargs.items() if k not in CUDA_OPTION_KEYS}

        self.backend = backend
        self.impl = impl
        self.legacy_backend = "eager" if impl == "eager" else "c"
        if impl == "eager":
            _warn_tf32_once(resolve_device(kwargs.get("device"), kwargs.get("dev"), equation))
        init_kwargs = _resolve_backend_init_kwargs(
            impl=impl,
            kwargs=kwargs,
            backend_options=backend_options,
            eager_options=eager_options,
            cuda_options=cuda_options,
            equation=equation,
        )
        # ONE resolution, for BOTH impls, BEFORE construction. impl='c' used to
        # skip this entirely: it constructed the backend and then inferred the
        # strategy back out of the flags the constructor happened to default to,
        # so the strategy was decided in two places that merely agreed. Deciding
        # it here, once, is what makes ``memory_strategy`` a fact rather than a
        # reading. ``memory=`` is folded into cuda_options above for impl='c',
        # so the request is recovered from whichever slot holds it.
        # ``memory`` was cleared when it was folded into cuda_options above, so
        # recover the request from whichever slot holds it. Prefer the caller's
        # own object: it is already normalised, and re-reading the converted
        # copy would re-trigger the legacy warning on the new spelling.
        _requested = _caller_request
        if _requested is None:
            _requested = as_memory_strategy(getattr(cuda_options, "memory", None))
        strategy = resolve_memory_strategy(
            impl, _requested, init_kwargs.get("use_ckpt"),
            init_kwargs.get("boundary_saving_config"))
        # Refuse what this backend cannot honour BEFORE building it, so an
        # unsupported option is an error rather than a silent downgrade.
        check_memory_supported(impl, _requested)
        if _requested is None and strategy != "full":
            # The legacy dict/flag route resolved a strategy without a typed
            # request, which made the check above a no-op -- so eager plus
            # tail_steps/disk in boundary_saving_config (or ckpt_mode=
            # 'recursive') was accepted and silently dropped. Synthesise the
            # equivalent typed request for validation only.
            if strategy == "boundary":
                cfg = init_kwargs.get("boundary_saving_config") or {}
                # Only the knobs the chosen storage accepts: the dict route
                # always ignored the rest (pinned_memory under storage='gpu'),
                # and this probe exists to check capability, not to turn a
                # config that read fine into a construction error.
                probe = BoundarySaving(**_applicable_boundary_knobs(
                    {f.name: cfg[f.name]
                     for f in dataclasses.fields(BoundaryOptions) if f.name in cfg}))
            elif init_kwargs.get("ckpt_mode", CKPT_DEFAULTS.mode) == "recursive":
                probe = Ckpt(mode="recursive",
                             count=max(1, int(init_kwargs.get("ckpt_num") or 1)))
            else:
                probe = Ckpt(mode="chunk")
            check_memory_supported(impl, probe)

        if impl == "eager":
            init_kwargs["use_ckpt"] = (strategy == "ckpt")
            backend_impl = _PropTorchEager(*args, **init_kwargs)
            backend_impl.memory_strategy = strategy
            if memory is not None:
                _apply_eager_memory(backend_impl, memory)
            elif strategy == "boundary":
                # legacy dict route now reaches the eager backend too (it used
                # to require memory= or enable_eager_boundary_saving()).
                cfg = dict(init_kwargs.get("boundary_saving_config") or {})
                backend_impl.enable_eager_boundary_saving(
                    True,
                    storage=cfg.get("storage", "gpu"),
                    storage_dtype=cfg.get("storage_dtype", "fp32"),
                )
        else:
            from sweep.propagator._c import _CompiledPropagator

            backend_impl = _CompiledPropagator(*args, **init_kwargs)
            backend_impl.memory_strategy = strategy
        _assert_backend_matches_strategy(backend_impl, strategy)
        self._backend_impl = backend_impl
        self.__signature__ = _public_forward_signature(type(self).forward)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            backend_impl = super().__getattr__("_backend_impl")
            return getattr(backend_impl, name)

    # Illumination controls/outputs live on the backend; delegate writes so that
    # ``solver.compute_illumination = True`` (and the source/receiver illumination
    # buffers) reach it.  Reads already delegate via __getattr__.
    _BACKEND_DELEGATED = ("compute_illumination", "source_illumination", "receiver_illumination",
                          "compute_adcig", "adcig_max_lag", "adcig")

    # The boundary/layout spec is consumed at construction: it sizes the padded
    # grid, decides the PML pad of every face, builds the profiles, and on
    # impl='c' is compiled into the kernels' free-surface bitmask and image
    # mirror. Writing it afterwards used to land on THIS wrapper, where it
    # shadows the backend's value: the read-back showed the new setting while
    # the physics kept the old one, so a script could believe it had switched a
    # free surface on and quietly model without one (it did -- on marine field
    # data). Refuse the write instead of accepting it and doing nothing.
    _CONSTRUCTION_ONLY = ("free_surface", "fs_faces", "abcn", "pad",
                          "pml_type", "topography")

    def __setattr__(self, name, value):
        if name in PropTorch._BACKEND_DELEGATED:
            backend = getattr(self, "_backend_impl", None)  # in self._modules once set
            if backend is not None:
                setattr(backend, name, value)
                return
        if (name in PropTorch._CONSTRUCTION_ONLY
                and getattr(self, "_backend_impl", None) is not None):
            raise AttributeError(
                f"{name!r} is fixed when the propagator is built: the padded "
                "grid, the per-face PML widths and (on impl='c') the compiled "
                "kernels all derive from it there, and none of them is "
                "recomputed on assignment. Setting it here would only shadow "
                f"the backend's value, so reads would report {value!r} while "
                "the physics kept the original. Build a new propagator with "
                f"PropTorch(..., {name}=...) instead."
            )
        super().__setattr__(name, value)

    def parameters(self):
        return self._backend_impl.parameters()

    def forward(self, wavelet, sources, receivers, models=None, adj=False, return_wavefield=False, **kwargs):
        return self._backend_impl(
            wavelet,
            sources,
            receivers,
            models=models,
            adj=adj,
            return_wavefield=return_wavefield,
            **kwargs,
        )
