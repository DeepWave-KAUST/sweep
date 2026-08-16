"""Pure rank-grid arithmetic for model-parallel meshes.

No ``torch.distributed`` dependency — the topology is just integer math, so
this module is safe to instantiate in single-process unit tests.

The ``ProcessGroup``-aware wrapper lives in :mod:`sweep.parallel.mesh`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


_VALID_AXES = ("x", "y")
_VALID_SIDES = ("low", "high")


@dataclass(frozen=True)
class MeshTopology:
    """Rank-grid layout for ``(P_shot, Py, Px)`` decomposition.

    The total ``world_size`` is partitioned as::

        world_size = shot_groups * (py * px)

    with each rank's coordinate derived as::

        shot_group = rank // (py * px)
        yi         = (rank %  (py * px)) // px
        xi         = rank %  px

    2-D problems must use ``py == 1``.

    Attributes
    ----------
    py, px : int
        Tile-grid extents along the y (crossline) and x (lateral) axes.
        For 2-D, ``py`` must be 1.
    shot_groups : int
        Number of orthogonal shot-parallel groups; ranks differing only in
        ``shot_group`` share the same tile coordinate ``(yi, xi)``.
    world_size, rank : int
        The familiar ``torch.distributed`` quantities. Kept on the topology
        object so all derived quantities (coords, neighbours, tile extents)
        are pure data.
    """

    py: int
    px: int
    shot_groups: int
    world_size: int
    rank: int

    def __post_init__(self) -> None:
        for name, val in (("py", self.py), ("px", self.px),
                          ("shot_groups", self.shot_groups),
                          ("world_size", self.world_size)):
            if val < 1:
                raise ValueError(f"{name}={val} must be >= 1")
        expected = self.py * self.px * self.shot_groups
        if expected != self.world_size:
            raise ValueError(
                f"py*px*shot_groups = {self.py}*{self.px}*{self.shot_groups} "
                f"= {expected} != world_size={self.world_size}"
            )
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank={self.rank} out of range [0, {self.world_size})"
            )

    # ------------------------------------------------------------------
    # Derived rank coordinates
    # ------------------------------------------------------------------
    @property
    def tile_world_size(self) -> int:
        """Number of ranks per shot group, i.e. ``py * px``."""
        return self.py * self.px

    @property
    def shot_group(self) -> int:
        return self.rank // self.tile_world_size

    @property
    def yi(self) -> int:
        return (self.rank % self.tile_world_size) // self.px

    @property
    def xi(self) -> int:
        return self.rank % self.px

    @property
    def coord(self) -> Tuple[int, int, int]:
        """``(shot_group, yi, xi)`` tuple for this rank."""
        return (self.shot_group, self.yi, self.xi)

    def rank_at(self, shot_group: int, yi: int, xi: int) -> int:
        """Inverse of :pyattr:`coord` — return the rank with these coords."""
        if not 0 <= shot_group < self.shot_groups:
            raise ValueError(
                f"shot_group={shot_group} out of [0, {self.shot_groups})"
            )
        if not 0 <= yi < self.py:
            raise ValueError(f"yi={yi} out of [0, {self.py})")
        if not 0 <= xi < self.px:
            raise ValueError(f"xi={xi} out of [0, {self.px})")
        return shot_group * self.tile_world_size + yi * self.px + xi

    # ------------------------------------------------------------------
    # Neighbour resolution
    # ------------------------------------------------------------------
    def neighbour_rank(self, axis: str, direction: int) -> Optional[int]:
        """Rank of the neighbour along ``axis`` in ``direction``.

        Parameters
        ----------
        axis : {'x', 'y'}
        direction : {-1, +1}

        Returns
        -------
        rank or None
            ``None`` when the rank sits at the global boundary along that axis
            (no neighbour exists) or when ``axis='y'`` and ``py == 1``.
        """
        if axis not in _VALID_AXES:
            raise ValueError(f"axis={axis!r} not in {_VALID_AXES}")
        if direction not in (-1, +1):
            raise ValueError(f"direction={direction} must be -1 or +1")
        if axis == "x":
            new_xi = self.xi + direction
            if not 0 <= new_xi < self.px:
                return None
            return self.rank_at(self.shot_group, self.yi, new_xi)
        # axis == "y"
        if self.py == 1:
            return None
        new_yi = self.yi + direction
        if not 0 <= new_yi < self.py:
            return None
        return self.rank_at(self.shot_group, new_yi, self.xi)

    def is_edge(self, axis: str, side: str) -> bool:
        """Is this rank on the low / high edge along ``axis``?

        A 1-tile axis (``px == 1`` or ``py == 1``) is considered to be on
        both edges — that single tile owns the absorbing boundary on both
        sides.
        """
        if axis not in _VALID_AXES:
            raise ValueError(f"axis={axis!r} not in {_VALID_AXES}")
        if side not in _VALID_SIDES:
            raise ValueError(f"side={side!r} not in {_VALID_SIDES}")
        idx = self.xi if axis == "x" else self.yi
        n = self.px if axis == "x" else self.py
        if n == 1:
            return True
        return (idx == 0) if side == "low" else (idx == n - 1)

    # ------------------------------------------------------------------
    # Tile extent
    # ------------------------------------------------------------------
    def local_extent(
        self, global_shape: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """Return ``(local_shape, offsets)`` for this rank's tile.

        ``global_shape`` is the SWEEP wavefield layout: ``(Nz, Nx)`` for 2-D
        and ``(Nz, Ny, Nx)`` for 3-D. The depth axis ``Nz`` is never split.

        v1 requires uniform tiles: ``Nx`` must be a multiple of ``px`` and
        ``Ny`` (3-D) a multiple of ``py``. Padding to that multiple is the
        caller's responsibility (see README §7).

        Returns
        -------
        local_shape : tuple
            Same ndim as ``global_shape``; ``Nz`` unchanged, ``Ny`` / ``Nx``
            replaced by per-tile sizes.
        offsets : tuple
            ``(0, [oy,] ox)`` — global index of this tile's origin. The
            leading ``0`` is the (un-split) depth offset.
        """
        global_shape = tuple(int(s) for s in global_shape)
        ndim = len(global_shape)
        if ndim == 2:
            if self.py != 1:
                raise ValueError(
                    f"2-D global_shape={global_shape} requires py=1, got py={self.py}"
                )
            Nz, Nx = global_shape
            if Nx % self.px != 0:
                raise ValueError(
                    f"Nx={Nx} is not a multiple of px={self.px}; pad the model "
                    f"to a multiple of px (v1 supports uniform tiles only), "
                    f"e.g. sweep.parallel.pad_to_mesh(model, mesh)."
                )
            nx_loc = Nx // self.px
            return (Nz, nx_loc), (0, self.xi * nx_loc)
        if ndim == 3:
            Nz, Ny, Nx = global_shape
            if Ny % self.py != 0:
                raise ValueError(
                    f"Ny={Ny} is not a multiple of py={self.py}; pad the model "
                    f"to a multiple of py (v1 supports uniform tiles only), "
                    f"e.g. sweep.parallel.pad_to_mesh(model, mesh)."
                )
            if Nx % self.px != 0:
                raise ValueError(
                    f"Nx={Nx} is not a multiple of px={self.px}; pad the model "
                    f"to a multiple of px (v1 supports uniform tiles only), "
                    f"e.g. sweep.parallel.pad_to_mesh(model, mesh)."
                )
            ny_loc = Ny // self.py
            nx_loc = Nx // self.px
            return (
                (Nz, ny_loc, nx_loc),
                (0, self.yi * ny_loc, self.xi * nx_loc),
            )
        raise ValueError(
            f"global_shape must be 2-D or 3-D, got ndim={ndim} ({global_shape!r})"
        )


def balanced_grid(
    world_size: int,
    global_shape: Tuple[int, ...],
    *,
    shot_groups: int = 1,
    max_py: int = 2,
) -> Tuple[int, int]:
    """Recommend a ``(py, px)`` DD grid that keeps each rank's tile compact.

    A 1-D x-cut (``py=1, px=world``) shrinks the *contiguous* x-dimension to
    ``Nx/world``; on 8 GPUs that is the slow, memory-heavy choice. Measured on
    8x V100 (acoustic 3-D, end-to-end forward via :class:`DDPropagator`), a
    balanced 2-D cut is **substantially faster and lighter** for strong scaling
    because cutting more axes saves more cut-aware PML *and* a squarer tile runs
    the FD kernel more efficiently:

    ===================  ============  ==================  ========
    global (Nz,Ny,Nx)    x-cut px8     balanced px4,py2    speedup
    ===================  ============  ==================  ========
    256 x 256 x 1024     0.893 ms      0.814 ms            +9 %
    384 x 384 x 384      1.026 ms      0.716 ms            +43 %
    512 x 512 x 512      1.890 ms      1.439 ms            +31 %
    ===================  ============  ==================  ========

    (peak memory also drops ~17-19 %.) The win grows the thinner the x-cut tile
    would be — i.e. for cubic / strong-scaling problems.

    This returns the ``(py, px)`` factorisation of the per-shot-group tile count
    (``world_size // shot_groups``) that MAXIMISES the smaller horizontal tile
    edge ``min(Ny/py, Nx/px)``, breaking ties toward a fatter (contiguous) x
    edge. 2-D models always get ``(1, px)`` (no y to split).

    ``max_py`` caps the y-tile count (default 2 — a conservative load-balance
    choice that already captures most of the win). Raise it (e.g.
    ``max_py=world_size``) to also consider ``py >= 3`` — the fastest option for
    cubic globals (e.g. 384^3 -> px2py4 = 0.688 ms / 6.96x), fully validated
    (8x V100, bit-exact). [A past py>=3 boundary-save crash was a
    DDPropagator._capture cut_face_mask=0 bug, fixed pure-Python — not a kernel
    limitation.]
    """
    tiles = world_size // shot_groups
    if tiles < 1:
        raise ValueError(f"world_size={world_size} < shot_groups={shot_groups}")
    if max_py < 1:
        raise ValueError(f"max_py={max_py} must be >= 1")
    ndim = len(global_shape)
    if ndim == 2:
        return (1, tiles)
    if ndim != 3:
        raise ValueError(f"global_shape must be 2-D or 3-D, got {global_shape!r}")
    _, Ny, Nx = (int(s) for s in global_shape)
    py_cap = min(max_py, tiles)

    best = None  # (sort key, (py, px))
    for py in range(1, py_cap + 1):
        if tiles % py:
            continue
        px = tiles // py
        if Ny % py or Nx % px:
            continue
        ny, nx = Ny // py, Nx // px
        key = (min(ny, nx), nx)            # squarest tile, then fattest x
        if best is None or key > best[0]:
            best = (key, (py, px))
    if best is None:
        # Nothing divides evenly. The x-cut fallback still needs the caller to
        # pad, and padding is NOT free: it grows the domain in front of the
        # high-side PML, so the run answers a slightly different problem than
        # the unpadded one. Measured on acoustic 3-D: ~1.7e-5 relative gradient
        # difference on a small 80x95x95 / 700-step case (localised at the
        # padded faces), but ~1.3e-3 and spread over the WHOLE volume on a
        # production 109x294x135 / 5258-step encoded run — the perturbation has
        # time to traverse the model. DD itself stays bit-exact either way; this
        # is the pad's own cost. Say so, because the alternative is that it
        # quietly lands in a result someone later compares against history.
        import warnings

        warnings.warn(
            f"balanced_grid: no (py, px) with py*px={tiles} divides "
            f"global_shape={global_shape}, so the model must be padded up "
            f"(sweep.parallel.pad_to_mesh) and the run will not match an "
            f"unpadded one. Prefer a rank count whose factors divide the grid "
            f"— or accept the pad knowingly.",
            RuntimeWarning, stacklevel=2)
        return (1, tiles)
    return best[1]
