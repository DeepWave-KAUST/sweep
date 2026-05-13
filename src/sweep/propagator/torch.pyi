from typing import Any

import torch

from .options import CUDAOptions, EagerOptions


class PropTorch(torch.nn.Module):
    backend: str
    impl: str

    def __init__(
        self,
        *args: Any,
        backend: str = ...,
        impl: str | None = ...,
        backend_options: dict[str, Any] | None = ...,
        eager_options: dict[str, Any] | EagerOptions | None = ...,
        cuda_options: dict[str, Any] | CUDAOptions | None = ...,
        **kwargs: Any,
    ) -> None: ...

    def parameters(self) -> Any: ...
    def get_parameters(self, key: str) -> Any: ...
    def set_parameters(self, model: Any) -> Any: ...

    def forward(
        self,
        wavelet: Any,
        sources: Any,
        receivers: Any,
        models: Any = ...,
        source_encoding: bool = ...,
        adj: bool = ...,
        return_wavefield: bool = ...,
        **kwargs: Any,
    ) -> Any: ...

    def __call__(
        self,
        wavelet: Any,
        sources: Any,
        receivers: Any,
        models: Any = ...,
        source_encoding: bool = ...,
        adj: bool = ...,
        return_wavefield: bool = ...,
        **kwargs: Any,
    ) -> Any: ...
