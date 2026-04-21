import inspect

import torch

from sweep.propagator._torch_eager import _PropTorchEager
from sweep.propagator.options import (
    CUDAOptions,
    EagerOptions,
    EAGER_OPTION_KEYS,
    CUDA_OPTION_KEYS,
    MemoryOptions,
    options_to_dict,
)


class PropTorch(torch.nn.Module):
    def __init__(
        self,
        *args,
        backend="eager",
        backend_options=None,
        eager_options=None,
        cuda_options=None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        backend = str(backend).lower()
        if backend not in {"eager", "cuda"}:
            raise ValueError(f"Unsupported PropTorch backend '{backend}'. Expected 'eager' or 'cuda'.")

        self.backend = backend
        init_kwargs = self._resolve_backend_init_kwargs(
            backend=backend,
            kwargs=kwargs,
            backend_options=backend_options,
            eager_options=eager_options,
            cuda_options=cuda_options,
        )
        if backend == "eager":
            impl = _PropTorchEager(*args, **init_kwargs)
        else:
            from sweep.propagator.cuda import PropCUDA

            impl = PropCUDA(*args, **init_kwargs)
        self._backend_impl = impl
        self.__signature__ = self._forward_signature()

    def _merge_option_dict(self, base, extra, *, label):
        if extra is None:
            return base
        extra = options_to_dict(extra)
        overlap = set(base) & set(extra)
        if overlap:
            raise ValueError(f"Duplicate option keys between top-level kwargs and {label}: {sorted(overlap)}")
        merged = dict(base)
        merged.update(extra)
        return merged

    def _normalize_cuda_kwargs(self, merged):
        memory = merged.pop("memory", None)
        if memory is None:
            return merged
        memory = options_to_dict(memory)
        strategy = memory.get("strategy")
        if strategy is None:
            raise ValueError("CUDA memory options must set strategy to 'boundary' or 'ckpt'.")
        if strategy == "boundary":
            boundary = memory.get("boundary") or {}
            merged["boundary_saving_config"] = {
                "enabled": True,
                "storage": boundary.get("storage", "gpu"),
                "transfer_interval": boundary.get("transfer_interval", 1),
                "pinned_memory": boundary.get("pinned_memory", False),
            }
            merged["use_ckpt"] = False
            return merged
        if strategy == "ckpt":
            ckpt = memory.get("ckpt") or {}
            merged["use_ckpt"] = True
            merged["ckpt_mode"] = ckpt.get("mode", "chunk")
            if merged["ckpt_mode"] == "chunk":
                merged["ckpt_chunks"] = ckpt.get("chunks", 100)
            elif merged["ckpt_mode"] == "recursive":
                merged["ckpt_num"] = ckpt.get("count", 0)
            else:
                raise ValueError(f"Unsupported ckpt mode '{merged['ckpt_mode']}'. Expected 'chunk' or 'recursive'.")
            return merged
        raise ValueError(f"Unsupported CUDA memory strategy '{strategy}'. Expected 'boundary' or 'ckpt'.")

    def _resolve_backend_init_kwargs(self, *, backend, kwargs, backend_options, eager_options, cuda_options):
        if eager_options is not None and backend != "eager":
            raise ValueError("eager_options can only be used with backend='eager'.")
        if cuda_options is not None and backend != "cuda":
            raise ValueError("cuda_options can only be used with backend='cuda'.")

        wrong_top_level = (CUDA_OPTION_KEYS if backend == "eager" else EAGER_OPTION_KEYS) & set(kwargs)
        if wrong_top_level:
            target = "cuda_options" if backend == "cuda" else "eager_options"
            raise ValueError(
                f"Top-level kwargs {sorted(wrong_top_level)} do not belong to backend='{backend}'. "
                f"Pass backend-specific options through {target} or switch backend."
            )

        merged = dict(kwargs)
        merged = self._merge_option_dict(merged, backend_options, label="backend_options")
        selected_options = eager_options if backend == "eager" else cuda_options
        option_label = "eager_options" if backend == "eager" else "cuda_options"
        merged = self._merge_option_dict(merged, selected_options, label=option_label)

        wrong_merged = (CUDA_OPTION_KEYS if backend == "eager" else EAGER_OPTION_KEYS) & set(merged)
        if wrong_merged:
            raise ValueError(f"Invalid {option_label} for backend='{backend}': {sorted(wrong_merged)}")

        if backend == "cuda":
            merged = self._normalize_cuda_kwargs(merged)
        return merged

    def _forward_signature(self):
        signature = inspect.signature(type(self).forward)
        parameters = list(signature.parameters.values())
        if parameters and parameters[0].name == "self":
            signature = signature.replace(parameters=parameters[1:])
        return signature

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            backend_impl = super().__getattr__("_backend_impl")
            return getattr(backend_impl, name)

    def parameters(self):
        return self._backend_impl.parameters()

    def get_parameters(self, key):
        return self._backend_impl.get_parameters(key)

    def set_parameters(self, model):
        return self._backend_impl.set_parameters(model)

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
