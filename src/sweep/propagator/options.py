from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any
from typing import Literal


@dataclass
class EagerOptions:
    use_compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = False
    compile_backend: str | None = None
    compile_fullgraph: bool = False
    store_last_wavefield: bool = False


@dataclass
class BoundaryOptions:
    # storage='cpu' keeps boundary buffers off device and uses pinned memory optionally.
    storage: Literal["gpu", "cpu"] = "gpu"
    transfer_interval: int = 1
    pinned_memory: bool = False

    def __post_init__(self):
        if self.transfer_interval < 1:
            raise ValueError("BoundaryOptions.transfer_interval must be >= 1.")
        if self.storage == "gpu":
            if self.transfer_interval != 1:
                raise ValueError(
                    "BoundaryOptions.transfer_interval is only valid when storage='cpu'."
                )
            if self.pinned_memory:
                raise ValueError(
                    "BoundaryOptions.pinned_memory is only valid when storage='cpu'."
                )


@dataclass
class CkptOptions:
    # mode='chunk' uses periodic replay; mode='recursive' uses a fixed checkpoint budget.
    mode: Literal["chunk", "recursive"] = "chunk"
    chunks: int = 100
    count: int = 0

    def __post_init__(self):
        if self.mode == "chunk":
            if self.chunks < 1:
                raise ValueError("CkptOptions.chunks must be >= 1 when mode='chunk'.")
            if self.count != 0:
                raise ValueError("CkptOptions.count is only valid when mode='recursive'.")
        else:
            if self.count < 1:
                raise ValueError("CkptOptions.count must be >= 1 when mode='recursive'.")
            if self.chunks != 100:
                raise ValueError("CkptOptions.chunks is only valid when mode='chunk'.")


@dataclass
class MemoryOptions:
    # Choose exactly one CUDA memory-saving strategy and fill its matching options block.
    strategy: Literal["boundary", "ckpt"] | None = None
    boundary: BoundaryOptions | None = None
    ckpt: CkptOptions | None = None

    def __post_init__(self):
        if self.strategy is None:
            if self.boundary is not None or self.ckpt is not None:
                raise ValueError(
                    "MemoryOptions.strategy must be set when boundary or ckpt options are provided."
                )
            return
        if self.strategy == "boundary":
            if self.boundary is None:
                raise ValueError("MemoryOptions.boundary must be provided when strategy='boundary'.")
            if self.ckpt is not None:
                raise ValueError("MemoryOptions.ckpt cannot be used when strategy='boundary'.")
        else:
            if self.ckpt is None:
                raise ValueError("MemoryOptions.ckpt must be provided when strategy='ckpt'.")
            if self.boundary is not None:
                raise ValueError("MemoryOptions.boundary cannot be used when strategy='ckpt'.")


@dataclass
class CUDAOptions:
    memory: MemoryOptions | None = None


EAGER_OPTION_KEYS = {field.name for field in fields(EagerOptions)}
CUDA_OPTION_KEYS = {field.name for field in fields(CUDAOptions)}


def options_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return _drop_none(asdict(value))
    if isinstance(value, dict):
        return _drop_none(value)
    raise TypeError(f"Expected a dict or dataclass options object, got {type(value).__name__}.")


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value
