"""Backend-agnostic boundary-ring geometry for boundary-saving reconstruction.

Pure-Python index math shared by the eager (PyTorch) and JAX boundary-saving
paths — no tensor-library imports.  The ring is the ``halo``-wide strip just
inside the PML + stencil halo on both sides of every spatial axis: exactly the
cells the interior stencil reads, so re-imposing the saved ring each reverse
step keeps the time-reversed reconstruction exact across the lossless interior.
"""

import math


def _ring_index_tuples(shape, ndim, offsets, halo):
    """Index tuples for the ``halo``-wide innermost-PML ring on both sides of
    every spatial axis (the cells the interior stencil reads)."""
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    rings = []
    for ai, ax in enumerate(spatial_axes):
        axis_len = shape[ax]
        off = offsets[ai]
        low = slice(off - halo, off)
        high = slice(axis_len - off, axis_len - off + halo)
        for sl in (low, high):
            idx = [slice(None)] * len(shape)
            idx[ax] = sl
            rings.append(tuple(idx))
    return rings


def _interior_index_tuple(shape, ndim, offsets):
    """Index tuple for the lossless deep interior — strictly inside the PML +
    stencil halo on every spatial axis (``[offset : N-offset]``), where the
    reverse reconstruction is exact (CPML is zeroed there and the restored ring
    sits outside it)."""
    idx = [slice(None)] * len(shape)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    for ai, ax in enumerate(spatial_axes):
        off = offsets[ai]
        idx[ax] = slice(off, shape[ax] - off)
    return tuple(idx)


def _ring_slice_shape(shape, idx):
    """Resolved shape of one ring index tuple against the (padded) field shape."""
    return tuple(len(range(*sl.indices(shape[ax]))) if isinstance(sl, slice) else 1
                 for ax, sl in enumerate(idx))


def _ring_layout(shape, ndim, offsets, halo):
    """Convenience bundle: ``(rings, ring_shapes, ring_sizes, total_cells)``."""
    rings = _ring_index_tuples(shape, ndim, offsets, halo)
    ring_shapes = [_ring_slice_shape(shape, idx) for idx in rings]
    ring_sizes = [math.prod(s) for s in ring_shapes]
    return rings, ring_shapes, ring_sizes, sum(ring_sizes)
