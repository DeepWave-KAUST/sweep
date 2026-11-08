"""Backend-agnostic boundary-ring geometry for boundary-saving reconstruction.

Pure-Python index math shared by the eager (PyTorch) and JAX boundary-saving
paths — no tensor-library imports.  The ring is the ``halo``-wide strip just
inside the PML + stencil halo on both sides of every spatial axis: exactly the
cells the interior stencil reads, so re-imposing the saved ring each reverse
step keeps the time-reversed reconstruction exact across the lossless interior.

The ring also takes in the outermost physical cells whose update reads the
CPML memory (``inner``, see :func:`ring_inner_cells`): one for a staggered
scheme, as the CUDA saver's ``M + 1``-wide strips do.
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


def ring_inner_cells(reverse_mode, memory_reach=0):
    """Physical cells per side the ring takes in for a reverse driver.

    The reconstruction zeroes the CPML memory, so a physical cell whose update
    reads it cannot be rebuilt and must be restored from the ring instead.
    'substep' (staggered): the memory enters pointwise, but a staggered field
    at index ``i`` lives at ``i + 1/2``, so on the high side of every axis the
    last physical index holds half-grid values that sit in the first half cell
    of the PML (``azh``/``axh`` nonzero there) -- one cell, kept on both sides
    as in the CUDA strips.  'swap2nd': the CPML term differentiates ``a * psi``
    (and ``b``, through ``np.gradient``), which reaches ``memory_reach`` cells
    into the box -- 1 for the acoustic family, M for VRZ, VRR and qP Tariq --
    measured on the equation by the caller."""
    return 1 if reverse_mode == "substep" else int(memory_reach)


def _ring_index_tuples(shape, ndim, offsets, halo, inner=0):
    """Index tuples for the ``halo``-wide innermost-PML ring on both sides of
    every spatial axis (the cells the interior stencil reads), widened by
    ``inner`` physical cells per side.  ``offsets`` is each face's PML + halo
    width (see ``_face_offsets``): a face with a thinner PML, or none (a free
    surface), has its ring that much nearer the edge, still outside the
    model."""
    face = _face_offsets(offsets, ndim)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    rings = []
    for ai, ax in enumerate(spatial_axes):
        axis_len = shape[ax]
        lo, hi = face[2 * ai], face[2 * ai + 1]
        low = slice(lo - halo, lo + inner)
        high = slice(axis_len - hi - inner, axis_len - hi + halo)
        for sl in (low, high):
            idx = [slice(None)] * len(shape)
            idx[ax] = sl
            rings.append(tuple(idx))
    return rings


def _interior_index_tuple(shape, ndim, offsets, inner=0):
    """Index tuple for the lossless deep interior — strictly inside the PML +
    stencil halo (and the ``inner`` ring cells) on every spatial axis
    (``[lo + inner : N - hi - inner]`` with each face's offset, see
    ``_face_offsets``), where the reverse reconstruction is exact (CPML is
    zeroed there and the restored ring sits outside it)."""
    face = _face_offsets(offsets, ndim)
    idx = [slice(None)] * len(shape)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    for ai, ax in enumerate(spatial_axes):
        idx[ax] = slice(face[2 * ai] + inner, shape[ax] - face[2 * ai + 1] - inner)
    return tuple(idx)


def _ring_slice_shape(shape, idx):
    """Resolved shape of one ring index tuple against the (padded) field shape."""
    return tuple(len(range(*sl.indices(shape[ax]))) if isinstance(sl, slice) else 1
                 for ax, sl in enumerate(idx))


def _ring_layout(shape, ndim, offsets, halo, inner=0):
    """Convenience bundle: ``(rings, ring_shapes, ring_sizes, total_cells)``."""
    rings = _ring_index_tuples(shape, ndim, offsets, halo, inner)
    ring_shapes = [_ring_slice_shape(shape, idx) for idx in rings]
    ring_sizes = [math.prod(s) for s in ring_shapes]
    return rings, ring_shapes, ring_sizes, sum(ring_sizes)


def axis_probe_index(shape, ndim, offsets, axis, margin):
    """Index tuple holding every spatial axis but ``axis`` (a spatial-axis
    number) to its deep interior, ``margin`` cells inside the ring offset, so
    that cells changed along ``axis`` are told apart from the other axes'
    absorbing layers; None when that interior is empty.  ``offsets`` as in
    :func:`_ring_index_tuples`."""
    face = _face_offsets(offsets, ndim)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    idx = [slice(None)] * len(shape)
    for ai, ax in enumerate(spatial_axes):
        if ai == axis:
            continue
        lo, hi = face[2 * ai] + margin, shape[ax] - face[2 * ai + 1] - margin
        if hi <= lo:
            return None
        idx[ax] = slice(lo, hi)
    return tuple(idx)


def memory_reach(changed_along, shape, ndim, offsets):
    """Physical cells per side, worst axis, at which an update reads the CPML
    memory, from ``changed_along[a]``: the indices along spatial axis ``a`` at
    which it changed when the memory was redrawn (the other axes held to their
    deep interior, :func:`axis_probe_index`).  Changes in the ring and the PML
    do not count: the ring is restored and the PML is not reconstructed.
    ``offsets`` as in :func:`_ring_index_tuples`."""
    face = _face_offsets(offsets, ndim)
    spatial_axes = list(range(len(shape) - ndim, len(shape)))
    reach = 0
    for ai, ax in enumerate(spatial_axes):
        n, lo, hi = shape[ax], face[2 * ai], face[2 * ai + 1]
        mid = (lo + n - hi) // 2
        for i in changed_along[ai]:
            if lo <= i < n - hi:
                reach = max(reach, i - lo + 1 if i < mid else n - hi - i)
    return reach
