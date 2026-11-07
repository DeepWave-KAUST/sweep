"""Backend-agnostic boundary-ring geometry for boundary-saving reconstruction.

Pure-Python index math shared by the eager (PyTorch) and JAX boundary-saving
paths — no tensor-library imports.  The ring is the ``halo``-wide strip just
inside the PML + stencil halo on both sides of every spatial axis: exactly the
cells the interior stencil reads, so re-imposing the saved ring each reverse
step keeps the time-reversed reconstruction exact across the lossless interior.
"""

import math


def _face_offsets(offsets, ndim):
    """``offsets`` per face, axis-major ``(lo, hi)`` pairs: given as one per
    axis (both faces alike) or one per face (``2 * ndim``, the layout of
    ``PropBase.pad``)."""
    offsets = tuple(int(o) for o in offsets)
    if len(offsets) == ndim:
        return tuple(o for o in offsets for _ in range(2))
    if len(offsets) != 2 * ndim:
        raise ValueError(f"ring offsets: {ndim} (one per axis) or {2 * ndim} (one per face) "
                         f"values, got {len(offsets)}")
    return offsets


def _ring_offsets(pml_pad, abcn, halo):
    """Each face's ring offset (``_ring_index_tuples``), axis-major per face.
    A face with a damping ramp (``PropBase.pml_pad``) has its ring in the
    innermost ``halo`` cells of that ramp: a sigma=0 buffer, where there is
    one, lies inside it.  A face without a ramp (a free surface, a zero
    width) needs no ring, the reconstruction there being exact; it keeps the
    ring at ``abcn + halo``, inside the model, where the band it pins also
    holds the gradient nearer the full tape's next to a free surface (2.9e-4
    instead of 1.3e-3 relative, a source one cell deep).  At ``abcn + halo``
    on every face, a face with a thinner ramp (a per-edge ``abcn``) had its
    ring inside the model too, and the strip between the ring and the thin
    PML came back wrong (gradients up to 22 % off)."""
    return tuple(int(p) + halo if p > 0 else int(abcn) + halo for p in pml_pad)


def _ring_index_tuples(shape, ndim, offsets, halo):
    """Index tuples for the ``halo``-wide innermost-PML ring on both sides of
    every spatial axis (the cells the interior stencil reads).  ``offsets``
    is each face's PML + halo width (see ``_face_offsets``): a face with a
    thinner PML, or none (a free surface), has its ring that much nearer the
    edge, still outside the model."""
    face = _face_offsets(offsets, ndim)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    rings = []
    for ai, ax in enumerate(spatial_axes):
        axis_len = shape[ax]
        lo, hi = face[2 * ai], face[2 * ai + 1]
        low = slice(lo - halo, lo)
        high = slice(axis_len - hi, axis_len - hi + halo)
        for sl in (low, high):
            idx = [slice(None)] * len(shape)
            idx[ax] = sl
            rings.append(tuple(idx))
    return rings


def _interior_index_tuple(shape, ndim, offsets):
    """Index tuple for the lossless deep interior — strictly inside the PML +
    stencil halo on every spatial axis (``[lo : N - hi]`` with each face's
    offset, see ``_face_offsets``), where the reverse reconstruction is exact
    (CPML is zeroed there and the restored ring sits outside it)."""
    face = _face_offsets(offsets, ndim)
    idx = [slice(None)] * len(shape)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    for ai, ax in enumerate(spatial_axes):
        idx[ax] = slice(face[2 * ai], shape[ax] - face[2 * ai + 1])
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
