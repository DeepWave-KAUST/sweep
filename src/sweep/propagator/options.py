from dataclasses import asdict, dataclass, fields, is_dataclass
from sweep.core.arguments import warn_deprecated_spelling
from typing import Any
from typing import ClassVar
from typing import Literal
from typing import Union


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
    # None = "not specified": the memory strategy resolver picks the backend
    # default (impl='c' -> 'boundary', eager/jax -> 'ckpt').  An explicit
    # True/False is treated as a request for / exclusion of checkpointing.
    use_ckpt: bool | None = None
    nt: int = -1
    batch_size: int = 1
    allow_growth: bool = True


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
    # The gradient-memory mode is a three-way choice, identical for the eager
    # and CUDA backends:
    #   'full'     -- keep everything needed for backward in memory (no
    #                 reconstruction, largest footprint)
    #   'boundary' -- boundary-saving wavefield reconstruction
    #   'ckpt'     -- checkpointing / rematerialisation
    # None = backend default ('boundary' for impl='c', 'ckpt' for eager/jax).
    strategy: Literal["full", "boundary", "ckpt"] | None = None
    boundary: BoundaryOptions | None = None
    ckpt: CkptOptions | None = None

    def __post_init__(self):
        if self.strategy == "full":
            if self.boundary is not None or self.ckpt is not None:
                raise ValueError("MemoryOptions(strategy='full') takes no boundary/ckpt options.")
            return
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


def resolve_memory_strategy(impl, memory=None, use_ckpt=None, boundary_saving_config=None):
    """Resolve the three-way gradient-memory mode: 'full' | 'boundary' | 'ckpt'.

    One mode wins, identically for every backend.  Conflicting explicit
    requests raise instead of silently preferring one path (historically
    ``use_ckpt`` -- which defaulted to True -- beat an explicitly enabled
    ``boundary_saving_config``, so "boundary" test scripts silently ran the
    checkpoint backward).

    Requests: ``memory.strategy`` (if set), ``use_ckpt=True`` (-> 'ckpt'),
    ``boundary_saving_config['enabled']=True`` (-> 'boundary').

    An explicit legacy off-switch (``use_ckpt=False``, ``enabled=False``) is
    not a vote for the *other* trick: with no positive request it selects
    ``'full'``.  That is what ``impl='c', use_ckpt=False`` has always meant --
    the knob suppressed the boundary-saving default outright -- and what the
    README and notebooks 00/12/16 document.  The implicit backend default
    ('boundary' for impl='c', 'ckpt' otherwise) applies only when no
    gradient-memory knob was passed at all.
    """
    requests = {}
    exclude = set()

    if memory is not None:
        mem_strategy = getattr(memory, "strategy", None)
        if mem_strategy is None and isinstance(memory, dict):
            mem_strategy = memory.get("strategy")
        if mem_strategy is not None:
            if mem_strategy not in ("full", "boundary", "ckpt"):
                raise ValueError(f"memory strategy must be 'full', 'boundary' or 'ckpt', got {mem_strategy!r}")
            requests["memory="] = mem_strategy

    if use_ckpt is True:
        requests["use_ckpt=True"] = "ckpt"
    elif use_ckpt is False:
        exclude.add("ckpt")

    if boundary_saving_config is not None:
        enabled = bool(dict(boundary_saving_config).get("enabled", BOUNDARY_DEFAULTS.enabled))
        if enabled:
            requests["boundary_saving_config"] = "boundary"
        else:
            exclude.add("boundary")

    if len(set(requests.values())) > 1:
        detail = ", ".join(f"{src} -> {mode!r}" for src, mode in sorted(requests.items()))
        raise ValueError(
            "Conflicting gradient-memory mode requests: " + detail +
            ". The mode is a three-way choice (full/boundary/ckpt); pass "
            "exactly one via memory=MemoryOptions(strategy=...).")

    if requests:
        strategy = next(iter(requests.values()))
        if strategy in exclude:
            src = next(s for s, m in requests.items() if m == strategy)
            other = next(iter(exclude & {strategy}))
            raise ValueError(
                f"Gradient-memory mode {strategy!r} was requested by {src} but "
                f"excluded by another knob; pass exactly one mode via "
                "memory=MemoryOptions(strategy=...).")
        return strategy

    if exclude:
        # An explicit off-switch and nothing else: no memory trick at all.
        return "full"

    default_order = ("boundary", "full") if impl == "c" else ("ckpt", "full")
    for strategy in default_order:
        if strategy not in exclude:
            return strategy
    return "full"


# ---------------------------------------------------------------------------
# The gradient-memory strategy as a type, not a tag plus optional bags
# ---------------------------------------------------------------------------
# ``MemoryOptions(strategy=..., boundary=..., ckpt=...)`` can express states that
# are not valid -- a strategy with the wrong bag filled, or a bag with no
# strategy -- so it carries six runtime checks whose only job is to reject them.
# When the TYPE carries the strategy, every one of those states becomes
# unrepresentable and the errors move to the call site with the right argument
# names: ``Ckpt(transfer_interval=4)`` is a TypeError from Python itself.
#
# The names follow the vocabulary already in the codebase rather than taste:
# ``Ckpt`` because everything here says ckpt (``use_ckpt``, ``ckpt_chunks``),
# ``Full`` because the strategy string is 'full'. ``BoundarySaving`` is the one
# spelled out: bare ``Boundary`` would be badly ambiguous in a solver where
# "boundary" means the absorbing/PML boundary in ~685 places against ~299 for
# boundary saving.
#
# They subclass the existing option dataclasses, so every field, default and
# validation rule is inherited rather than copied, and ``isinstance(x,
# BoundaryOptions)`` keeps working for code that predates this.

@dataclass
class Full:
    """Keep everything the backward needs in memory. No parameters, by nature."""
    strategy: ClassVar[str] = "full"


@dataclass
class BoundarySaving(BoundaryOptions):
    """Reconstruct the forward wavefield from saved boundary values."""
    strategy: ClassVar[str] = "boundary"


@dataclass
class Ckpt(CkptOptions):
    """Rematerialise the forward wavefield from checkpoints."""
    strategy: ClassVar[str] = "ckpt"


MemoryStrategy = Union[Full, BoundarySaving, Ckpt]

_STRATEGY_TYPES = {"full": Full, "boundary": BoundarySaving, "ckpt": Ckpt}


def as_memory_strategy(value):
    """Normalise anything that can name a memory strategy into one of the types.

    Accepts, in order of preference:

    * ``Full`` / ``BoundarySaving`` / ``Ckpt`` -- returned unchanged;
    * a legacy ``MemoryOptions(strategy=..., boundary=..., ckpt=...)``;
    * a dict, either the new flat shape ``{'kind': 'boundary', 'storage': ...}``
      or the legacy nested one ``{'strategy': 'boundary', 'boundary': {...}}``.
      Both are accepted on the way IN because the nested shape is already
      written into stored experiment YAML -- rewriting those files would edit
      the record of what was actually run;
    * ``None`` -- returned as ``None``, meaning "no request", which is distinct
      from ``Full()``.
    """
    if value is None or isinstance(value, (Full, BoundarySaving, Ckpt)):
        return value

    if isinstance(value, MemoryOptions):
        warn_deprecated_spelling(
            "MemoryOptions(strategy=..., boundary=..., ckpt=...)",
            "Full() / BoundarySaving(...) / Ckpt(...)")
        if value.strategy is None:
            return None
        if value.strategy == "full":
            return Full()
        sub = value.boundary if value.strategy == "boundary" else value.ckpt
        cls = _STRATEGY_TYPES[value.strategy]
        return cls(**{f.name: getattr(sub, f.name) for f in fields(sub)})

    if isinstance(value, dict):
        data = dict(value)
        kind = data.pop("kind", None) or data.pop("strategy", None)
        if kind is None:
            raise ValueError(
                "a memory-strategy dict needs 'kind' (or the legacy 'strategy'); "
                f"got keys {sorted(value)}")
        if kind not in _STRATEGY_TYPES:
            raise ValueError(
                f"memory strategy must be 'full', 'boundary' or 'ckpt', got {kind!r}")
        # Legacy nested shape: the parameters sit under a key named after the
        # strategy, and the same word therefore appears twice.
        nested = data.pop(kind, None)
        if nested is not None:
            warn_deprecated_spelling(
                f"the nested {{'strategy': {kind!r}, {kind!r}: {{...}}}} dict",
                f"a flat {{'kind': {kind!r}, ...}} dict")
            data = dict(nested)
        data.pop("boundary", None)
        data.pop("ckpt", None)
        return _STRATEGY_TYPES[kind](**data)

    raise TypeError(
        f"cannot read a memory strategy from {type(value).__name__}; pass "
        "Full(), BoundarySaving(...), Ckpt(...) or a dict with 'kind'.")


def to_legacy_memory_options(strategy):
    """Inverse of :func:`as_memory_strategy`, for the internal plumbing.

    The impl='c' path translates ``cuda_options.memory`` into init kwargs
    through code that reads the tagged shape (``{'strategy': ..., 'boundary':
    {...}}``). Rather than teach every one of those layers the new types, the
    public entry point converts once, here, on the way in. The new types are the
    API; the tagged form remains the wire format until the layers below are
    migrated in their own right.
    """
    if strategy is None or isinstance(strategy, MemoryOptions):
        return strategy
    if isinstance(strategy, Full):
        return MemoryOptions(strategy="full")
    if isinstance(strategy, BoundarySaving):
        return MemoryOptions(
            strategy="boundary",
            boundary=BoundaryOptions(**{f.name: getattr(strategy, f.name)
                                        for f in fields(BoundaryOptions)}))
    if isinstance(strategy, Ckpt):
        return MemoryOptions(
            strategy="ckpt",
            ckpt=CkptOptions(**{f.name: getattr(strategy, f.name)
                                for f in fields(CkptOptions)}))
    raise TypeError(f"not a memory strategy: {type(strategy).__name__}")


# ---------------------------------------------------------------------------
# What each backend can actually do
# ---------------------------------------------------------------------------
# Declared per backend rather than discovered by hitting a scattered ValueError
# somewhere down the call stack, so that the answer to "can eager do this?" is
# one table instead of a search. Every entry below was MEASURED, not read off a
# docstring.
#
# The entry that matters most is ``tail_steps`` on eager. It was being ACCEPTED
# and then dropped: ``_apply_eager_memory`` forwards only storage and
# storage_dtype, and the eager backend has no implementation of truncation at
# all. So ``BoundarySaving(tail_steps=20)`` on eager silently produced a
# full-length gradient. Refusing an unsupported option is a nuisance; accepting
# it and ignoring it is a wrong answer that looks right.
_MEMORY_CAPABILITIES = {
    "c": {
        "strategies": ("full", "boundary", "ckpt"),
        "boundary_storage": ("gpu", "cpu", "disk"),
        "boundary_tail_steps": True,
        "ckpt_mode": ("chunk", "recursive"),
    },
    "eager": {
        "strategies": ("full", "boundary", "ckpt"),
        "boundary_storage": ("gpu", "cpu"),   # the ring stays on device or in host RAM
        "boundary_tail_steps": False,         # no truncated backward in the eager driver
        "ckpt_mode": ("chunk",),              # no recursive/binomial schedule
    },
}


def check_memory_supported(impl, strategy):
    """Refuse a memory request the backend cannot honour, before it is built.

    Raises :class:`NotImplementedError` naming the impl, the option and what it
    does support. An unsupported request must never be quietly downgraded: the
    run would report one strategy and compute another.
    """
    caps = _MEMORY_CAPABILITIES.get(impl)
    if caps is None or strategy is None:
        return
    name = getattr(strategy, "strategy", None)
    if name is None:
        return
    if name not in caps["strategies"]:
        raise NotImplementedError(
            f"impl={impl!r} does not implement the {name!r} gradient-memory "
            f"strategy; it supports {', '.join(caps['strategies'])}.")

    if name == "boundary":
        storage = getattr(strategy, "storage", "gpu")
        if storage not in caps["boundary_storage"]:
            raise NotImplementedError(
                f"impl={impl!r} boundary saving does not implement "
                f"storage={storage!r}; it supports "
                f"{', '.join(caps['boundary_storage'])}.")
        tail = getattr(strategy, "tail_steps", None)
        if tail and not caps["boundary_tail_steps"]:
            raise NotImplementedError(
                f"impl={impl!r} boundary saving does not implement tail_steps "
                f"(truncated backward); it would be accepted and ignored, "
                f"returning a full-length gradient. Use impl='c' for "
                f"tail_steps={tail!r}.")

    elif name == "ckpt":
        mode = getattr(strategy, "mode", "chunk")
        if mode not in caps["ckpt_mode"]:
            raise NotImplementedError(
                f"impl={impl!r} checkpointing does not implement "
                f"mode={mode!r}; it supports {', '.join(caps['ckpt_mode'])}.")
