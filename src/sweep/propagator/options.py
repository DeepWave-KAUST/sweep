from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any
from typing import Literal


@dataclass(frozen=True)
class EagerDefaults:
    use_compile: bool = True
    compile_mode: str = "default"
    compile_dynamic: bool = False
    compile_backend: str | None = None
    compile_fullgraph: bool = False
    store_last_wavefield: bool = False


@dataclass(frozen=True)
class BoundaryDefaults:
    enabled: bool = False
    storage: Literal["gpu", "cpu", "disk"] = "gpu"
    transfer_interval: int | None = None
    pinned_memory: bool | None = None
    disk_dir: str | None = None
    ring_buffers: int | None = None
    disk_async_read: bool = False
    gpu_transfer_interval: int = 1
    gpu_pinned_memory: bool = False
    gpu_ring_buffers: int = 1
    cpu_transfer_interval: int = 64
    cpu_ring_buffers: int = 1
    cpu_pinned_memory: bool = True
    disk_transfer_interval: int = 32
    disk_async_transfer_interval_2d: int = 40
    disk_async_transfer_interval_3d: int = 16
    disk_ring_buffers_2d: int = 3
    disk_ring_buffers_3d: int = 2
    disk_async_ring_buffers: int = 2


@dataclass(frozen=True)
class CkptDefaults:
    mode: Literal["chunk", "recursive"] = "chunk"
    chunks: int = 100
    count: int = 0
    storage: Literal["gpu", "cpu"] = "gpu"
    pinned_memory: bool | None = None
    cpu_pinned_memory: bool = True


@dataclass(frozen=True)
class PropagatorDefaults:
    abcn: int = 50
    free_surface: bool = False
    dh: float = 10.0
    dt: float = 0.002
    dev: str | None = None
    use_ckpt: bool = True
    nt: int = -1
    batch_size: int = 1
    allow_growth: bool = True
    full_mode: str = "full"


EAGER_DEFAULTS = EagerDefaults()
BOUNDARY_DEFAULTS = BoundaryDefaults()
CKPT_DEFAULTS = CkptDefaults()
PROP_DEFAULTS = PropagatorDefaults()


@dataclass
class EagerOptions:
    use_compile: bool = EAGER_DEFAULTS.use_compile
    compile_mode: str = EAGER_DEFAULTS.compile_mode
    compile_dynamic: bool = EAGER_DEFAULTS.compile_dynamic
    compile_backend: str | None = EAGER_DEFAULTS.compile_backend
    compile_fullgraph: bool = EAGER_DEFAULTS.compile_fullgraph
    store_last_wavefield: bool = EAGER_DEFAULTS.store_last_wavefield


@dataclass
class BoundaryOptions:
    # storage='cpu' keeps boundary buffers off device and uses pinned memory optionally.
    storage: Literal["gpu", "cpu", "disk"] = BOUNDARY_DEFAULTS.storage
    transfer_interval: int | None = BOUNDARY_DEFAULTS.transfer_interval
    pinned_memory: bool | None = BOUNDARY_DEFAULTS.pinned_memory
    disk_dir: str | None = BOUNDARY_DEFAULTS.disk_dir
    ring_buffers: int | None = BOUNDARY_DEFAULTS.ring_buffers
    disk_async_read: bool = BOUNDARY_DEFAULTS.disk_async_read
    # Low-precision storage for the boundary buffer.  Compute always
    # stays FP32; this only changes how saved boundary values are
    # represented in memory:
    #   'fp32'  — 4 bytes/cell, baseline, no precision loss
    #   'fp16'  — 2 bytes/cell + ~1.6% FP32 scale metadata, 10-bit
    #             mantissa (precision ~5e-4).  Stored per-256-cell
    #             normalized (like int8), so it is amplitude-safe: the
    #             raw fp16 range would otherwise flush the velocity
    #             faces of elastic wavefields (values sit ~rho·vp below
    #             the stresses) to zero and corrupt the gradient.
    #   'bf16'  — 2 bytes/cell, 7-bit mantissa (precision ~8e-3),
    #             same dynamic range as FP32 — safe under arbitrary
    #             source amplitude scaling, slightly noisier reconstruction
    #   'int8'  — ~1 byte/cell + ~0.4% FP32 scale metadata (≈4× compression),
    #             per-256-cell symmetric quantization (DeepWave-style).
    #             Lossier than BF16; intended for memory-bound runs where
    #             3.94× compression beats BF16's 2× and the gradient drop
    #             is acceptable for the equation in question.
    storage_dtype: Literal["fp32", "fp16", "bf16", "int8"] = "fp32"
    # Boundary tail truncation for steady-state / frequency-selection FWI:
    # when set, only the LAST ``tail_steps`` time steps have their boundary
    # strips saved and back-propagated.  The forward physics is unchanged
    # (the wavefield must still ring up), but the reverse sweep stops after
    # ``tail_steps`` steps -- valid when the objective reads only the last
    # ``tail_steps`` record samples (the adjoint source is zero earlier), and
    # the dropped gradient term is exactly the adjoint x ring-up-transient
    # correlation that steady-state methods discard by construction.  Include
    # a safety margin on top of the probe window.  Boundary buffers shrink
    # accordingly.  None = full-length backward (bit-exact legacy).
    # Currently impl='c' Acoustic/Acoustic3D with boundary saving only.
    tail_steps: int | None = None

    def __post_init__(self):
        if self.storage not in {"gpu", "cpu", "disk"}:
            raise ValueError("BoundaryOptions.storage must be 'gpu', 'cpu', or 'disk'.")
        if self.storage_dtype not in {"fp32", "fp16", "bf16", "int8"}:
            raise ValueError("BoundaryOptions.storage_dtype must be 'fp32', 'fp16', 'bf16', or 'int8'.")
        # Every storage location (gpu/cpu/disk) supports every storage_dtype,
        # including int8 (staged int8 uses a uint8 main + FP32 per-block scale
        # ring, with parallel uint8/scale files for disk).
        if self.transfer_interval is not None and self.transfer_interval < 1:
            raise ValueError("BoundaryOptions.transfer_interval must be >= 1.")
        if self.ring_buffers is not None and self.ring_buffers < 1:
            raise ValueError("BoundaryOptions.ring_buffers must be >= 1.")
        if self.tail_steps is not None and self.tail_steps < 1:
            raise ValueError("BoundaryOptions.tail_steps must be >= 1 (or None).")
        if self.storage == "gpu":
            if self.transfer_interval not in (None, 1):
                raise ValueError(
                    "BoundaryOptions.transfer_interval is only valid when storage='cpu' or storage='disk'."
                )
            if self.ring_buffers not in (None, 1):
                raise ValueError(
                    "BoundaryOptions.ring_buffers is only valid when storage='cpu' or storage='disk'."
                )
            if self.pinned_memory:
                raise ValueError(
                    "BoundaryOptions.pinned_memory is only valid when storage='cpu'."
                )
            if self.disk_async_read:
                raise ValueError(
                    "BoundaryOptions.disk_async_read is only valid when storage='disk'."
                )
        if self.storage == "disk" and self.pinned_memory:
            raise ValueError(
                "BoundaryOptions.pinned_memory is only valid when storage='cpu'."
            )
        if self.storage == "cpu" and self.disk_async_read:
            raise ValueError(
                "BoundaryOptions.disk_async_read is only valid when storage='disk'."
            )


@dataclass
class CkptOptions:
    # mode='chunk' uses periodic replay; mode='recursive' uses a fixed checkpoint budget.
    mode: Literal["chunk", "recursive"] = CKPT_DEFAULTS.mode
    chunks: int = CKPT_DEFAULTS.chunks
    count: int = CKPT_DEFAULTS.count
    storage: Literal["gpu", "cpu"] = CKPT_DEFAULTS.storage
    pinned_memory: bool | None = CKPT_DEFAULTS.pinned_memory

    def __post_init__(self):
        if self.storage not in {"gpu", "cpu"}:
            raise ValueError("CkptOptions.storage must be 'gpu' or 'cpu'.")
        if self.storage == "gpu" and self.pinned_memory:
            raise ValueError("CkptOptions.pinned_memory is only valid when storage='cpu'.")
        if self.mode == "chunk":
            if self.chunks < 1:
                raise ValueError("CkptOptions.chunks must be >= 1 when mode='chunk'.")
            if self.count != 0:
                raise ValueError("CkptOptions.count is only valid when mode='recursive'.")
        elif self.mode == "recursive":
            if self.count < 1:
                raise ValueError("CkptOptions.count must be >= 1 when mode='recursive'.")
            if self.chunks != CKPT_DEFAULTS.chunks:
                raise ValueError("CkptOptions.chunks is only valid when mode='chunk'.")
        else:
            raise ValueError("CkptOptions.mode must be 'chunk' or 'recursive'.")


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
