from typing import Any

import torch

from .base import PropBase


class PropTorch(PropBase, torch.nn.Module):
    use_compile: bool
    store_last_wavefield: bool

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

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
