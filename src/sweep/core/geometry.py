"""Grid geometry: the padded runtime grid, and how to get back to the physical one.

A propagator works on a grid that is bigger than the model in two independent
ways -- a per-face PML pad, and a stencil halo of ``spatial_order // 2`` on every
face -- and almost every indexing bug in this codebase has been a confusion
between the two. Spelling the arithmetic once, as functions of the numbers
rather than as methods reaching into ``self``, is what makes the distinction
checkable.

Two conventions to keep straight, both of which used to be implicit:

* **Axis order.** ``padding`` and the coordinate offset are in torch order --
  reversed relative to ``shape``, so the LAST entry is z. ``pad`` is in flat
  ``[lo, hi]`` pairs per axis in shape order. The reversal in
  :func:`runtime_coord_offset` is not a bug.
* **A free-surface face has pad 0.** Not "abcn suppressed at read time": the pad
  really is zero there, so a top free surface leaves a z-offset of just the
  halo. Anything that assumes ``abcn`` on every face is wrong under a per-edge
  free surface.
"""
from __future__ import annotations


def fd_halo(spatial_order: int) -> int:
    """Stencil halo width for a spatial order (``so // 2``)."""
    return spatial_order // 2


def runtime_shape(shape, halo: int):
    """Physical-plus-PML shape grown by the stencil halo on both faces."""
    if halo <= 0:
        return tuple(shape)
    return tuple(s + 2 * halo for s in shape)


def runtime_padding(padding, halo: int):
    """Per-face PML padding grown by the stencil halo."""
    if halo <= 0:
        return tuple(padding)
    return tuple(p + halo for p in padding)


def runtime_fd_pad(halo: int, ndim: int) -> list[int]:
    """Flat ``F.pad``-style halo pairs, one ``[lo, hi]`` per axis."""
    return [halo, halo] * ndim


def runtime_coord_offset(pad, halo: int, ndim: int):
    """Physical origin -> padded-grid origin, per axis, in torch (reversed) order.

    Each axis contributes its LOW-side pad plus the halo. A free-surface low
    face has pad 0, so a top free surface yields ``halo`` on z -- which is what
    the old image-method special case computed, without needing to special-case
    anything.
    """
    return tuple(pad[2 * ax] + halo for ax in reversed(range(ndim)))


def runtime_crop_slices(halo: int, ndim: int):
    """Slices stripping the stencil halo back to the PML-padded grid."""
    if halo <= 0:
        return (slice(None),) * ndim
    return tuple(slice(halo, -halo) for _ in range(ndim))


def crop_runtime_halo(data, halo: int, ndim: int):
    """Drop the stencil halo from a runtime-grid array."""
    if halo <= 0:
        return data
    return data[(...,) + runtime_crop_slices(halo, ndim)]


def crop_to_physical(data, pad, ndim: int):
    """Drop every face's PML pad, recovering the physical model.

    A free-surface face has pad 0, so nothing is cropped there -- which
    reproduces the old hard-coded ``data[..., 0:-abcn, abcn:-abcn]`` image
    without hard-coding which face is free.
    """
    slices = [Ellipsis]
    for ax in range(ndim):
        lo, hi = pad[2 * ax], pad[2 * ax + 1]
        slices.append(slice(lo, -hi if hi > 0 else None))
    return data[tuple(slices)]


def spatial_pad_pairs(flat_padding):
    """Flat ``[lo, hi, ...]`` padding -> per-axis pairs in shape order.

    ``flat_padding`` is in torch order (last axis first), so the result is
    reversed back into shape order.
    """
    pairs = [(flat_padding[2 * i], flat_padding[2 * i + 1])
             for i in range(len(flat_padding) // 2)]
    return tuple(reversed(pairs))
