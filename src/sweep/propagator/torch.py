import inspect
import warnings

import torch

from sweep.propagator._torch_eager import _PropTorchEager
from sweep.propagator.options import (
    CUDAOptions,
    EagerOptions,
    EAGER_OPTION_KEYS,
    CUDA_OPTION_KEYS,
    BOUNDARY_DEFAULTS,
    CKPT_DEFAULTS,
    MemoryOptions,
    options_to_dict,
)


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
    """True when the equation exposes the compiled-kernel hook ``_C()``."""
    if equation is None:
        return True
    return callable(getattr(equation, "_C", None))


def _resolve_impl_with_fallback(impl, *, explicit, equation=None):
    """Resolve 'auto' / 'c' to a concrete impl, falling back to 'eager' when
    the compiled binding is unavailable or the equation has no ``_C`` hook.

    When the user explicitly asked for 'c' but the path isn't available, emit
    a UserWarning so the slowdown is visible.
    """
    binding_ok = _compiled_binding_available()
    equation_ok = _equation_supports_c(equation)

    if impl == "auto":
        return "c" if (binding_ok and equation_ok) else "eager"

    if impl == "c" and not (binding_ok and equation_ok):
        if explicit:
            if not binding_ok:
                reason = "the compiled binding (sweep._C) is unavailable"
                remedy = (
                    "Rebuild with `pip install -e .` in a CUDA-enabled env "
                    "to use the compiled path."
                )
            else:
                eq_name = type(equation).__name__ if equation is not None else "<unknown>"
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


def _normalize_backend_impl(backend, impl, *, equation=None):
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
        return "torch", _resolve_impl_with_fallback(legacy_impl, explicit=explicit_impl, equation=equation)

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            "Unsupported PropTorch backend "
            f"'{backend}'. Expected backend='torch'. Use PropJax for backend='jax'."
        )

    normalized = _normalize_impl("auto" if impl is None else impl)
    return backend, _resolve_impl_with_fallback(normalized, explicit=explicit_impl, equation=equation)


def _merge_option_dict(base, extra, *, label):
    if extra is None:
        return base
    extra = options_to_dict(extra)
    overlap = set(base) & set(extra)
    if overlap:
        raise ValueError(f"Duplicate option keys between top-level kwargs and {label}: {sorted(overlap)}")
    return {**base, **extra}


def _normalize_cuda_memory_kwargs(merged):
    memory = merged.pop("memory", None)
    if memory is None:
        # impl='c' default: boundary saving with GPU storage.  Cheaper memory
        # than full-wavefield mode and avoids the variable-batch
        # non-contiguous slice that bites chunked checkpointing
        # (cf. fix/ckpt-batch-shrink).  Only inject if the user hasn't
        # already chosen a memory strategy at the top level.
        if "use_ckpt" not in merged and "boundary_saving_config" not in merged:
            merged["boundary_saving_config"] = {
                "enabled": True,
                "storage": "gpu",
            }
            merged["use_ckpt"] = False
        return merged

    memory = options_to_dict(memory)
    strategy = memory.get("strategy")
    if strategy is None:
        raise ValueError("c memory options must set strategy to 'boundary' or 'ckpt'.")

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


def _resolve_backend_init_kwargs(*, impl, kwargs, backend_options, eager_options, cuda_options):
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
        merged = _normalize_cuda_memory_kwargs(merged)
    return merged


def _public_forward_signature(forward):
    signature = inspect.signature(forward)
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name == "self":
        signature = signature.replace(parameters=parameters[1:])
    return signature


class PropTorch(torch.nn.Module):
    def __init__(
        self,
        *args,
        backend=None,
        impl=None,
        backend_options=None,
        eager_options=None,
        cuda_options=None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        equation = args[0] if args else kwargs.get('equation')
        if backend is None:
            # Inherit from the equation (positional arg 0); equation owns backend
            # because its operators were already built against it.
            backend = getattr(equation, 'backend', 'torch')
        requested_impl = impl
        backend, impl = _normalize_backend_impl(backend, impl, equation=equation)

        if impl == "eager" and (cuda_options is not None or CUDA_OPTION_KEYS & set(kwargs)):
            requested_norm = (
                _normalize_impl(requested_impl) if requested_impl is not None else "auto"
            )
            if requested_norm in ("c", "auto"):
                cuda_options = None
                kwargs = {k: v for k, v in kwargs.items() if k not in CUDA_OPTION_KEYS}

        self.backend = backend
        self.impl = impl
        self.legacy_backend = "eager" if impl == "eager" else "c"
        init_kwargs = _resolve_backend_init_kwargs(
            impl=impl,
            kwargs=kwargs,
            backend_options=backend_options,
            eager_options=eager_options,
            cuda_options=cuda_options,
        )
        if impl == "eager":
            backend_impl = _PropTorchEager(*args, **init_kwargs)
        else:
            from sweep.propagator._c import _CompiledPropagator

            backend_impl = _CompiledPropagator(*args, **init_kwargs)
        self._backend_impl = backend_impl
        self.__signature__ = _public_forward_signature(type(self).forward)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            backend_impl = super().__getattr__("_backend_impl")
            return getattr(backend_impl, name)

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
