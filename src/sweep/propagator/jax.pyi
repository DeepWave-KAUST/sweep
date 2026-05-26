from typing import Any

from .base import PropBase


class PropJax(PropBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

    def forward(
        self,
        wavelet: Any,
        sources: Any,
        receivers: Any,
        models: Any = ...,
        return_wavefield: bool = ...,
        adj: bool = ...,
        wave_equation: Any = ...,
        aux_args: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> Any: ...

    def __call__(
        self,
        wavelet: Any,
        sources: Any,
        receivers: Any,
        models: Any = ...,
        return_wavefield: bool = ...,
        adj: bool = ...,
        wave_equation: Any = ...,
        aux_args: tuple[Any, ...] = ...,
        **kwargs: Any,
    ) -> Any: ...
