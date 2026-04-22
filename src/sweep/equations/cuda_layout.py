from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CUDALayoutSpec:
    """CUDA runtime metadata for compiled propagator buffer layout."""

    base_nvar: int
    pml_nvar: int
    last_two_nvar: int
    last_two_storage_nvar: int | None = None
    checkpoint_nvar: int | None = None
    backward_workspace_nvar: int = 0
    backward_workspace_shapes: Callable | None = None

    def resolved_last_two_storage_nvar(self) -> int:
        return self.base_nvar if self.last_two_storage_nvar is None else self.last_two_storage_nvar

    def resolved_checkpoint_nvar(self) -> int:
        default = self.base_nvar + self.pml_nvar
        return default if self.checkpoint_nvar is None else self.checkpoint_nvar
