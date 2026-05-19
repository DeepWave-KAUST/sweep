import inspect

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
SUPPORTED_IMPLS = {"eager", "c"}
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
        raise ValueError(f"Unsupported PropTorch impl '{value}'. Expected 'eager' or 'c'.")
    return value


def _normalize_backend_impl(backend, impl):
    backend = str(backend).lower()
    if backend == "pytorch":
        backend = "torch"

    if backend in LEGACY_BACKEND_IMPLS:
        legacy_impl = LEGACY_BACKEND_IMPLS[backend]
        if impl is not None and _normalize_impl(impl) != legacy_impl:
            raise ValueError(
                f"Legacy PropTorch backend='{backend}' implies impl='{legacy_impl}', got impl='{impl}'."
            )
        return "torch", legacy_impl

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            "Unsupported PropTorch backend "
            f"'{backend}'. Expected backend='torch'. Use PropJax for backend='jax'."
        )

    return backend, _normalize_impl("eager" if impl is None else impl)


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
        if backend is None:
            # Inherit from the equation (positional arg 0); equation owns backend
            # because its operators were already built against it.
            equation = args[0] if args else kwargs.get('equation')
            backend = getattr(equation, 'backend', 'torch')
        backend, impl = _normalize_backend_impl(backend, impl)

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

    def forward(self, wavelet, sources, receivers, models=None, source_encoding=False, adj=False, return_wavefield=False, **kwargs):
        return self._backend_impl(
            wavelet,
            sources,
            receivers,
            models=models,
            source_encoding=source_encoding,
            adj=adj,
            return_wavefield=return_wavefield,
            **kwargs,
        )
