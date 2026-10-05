from .options import (
    BoundaryOptions, BoundarySaving, Ckpt, CkptOptions, CUDAOptions, EagerOptions, Full,
    MemoryOptions,
)

__all__ = [
    "EagerOptions",
    "CUDAOptions",
    "Full",
    "BoundarySaving",
    "Ckpt",
    "MemoryOptions",
    "BoundaryOptions",
    "CkptOptions",
]

try:
    from .torch import PropTorch
    # RWI term III of the AcousticLSRTM vp gradient, split out with
    # SWEEP_LSRTM_SPLIT_III=1 (Wu & Alkhalifah 2015, eq. 18-20).
    from ._c import SPLIT_III_ENV, last_grad_split_iii
except ModuleNotFoundError:
    PropTorch = None
else:
    __all__ += ["PropTorch", "SPLIT_III_ENV", "last_grad_split_iii"]

try:
    from .jax import PropJax
except ModuleNotFoundError:
    PropJax = None
else:
    __all__.append("PropJax")
