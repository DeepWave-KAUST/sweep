"""Per-edge boundary specification helpers (free surface + PML thickness).

sweep historically supported a free surface on the **top (z-min) face only**,
selected by a single ``bool free_surface``, with one scalar PML width ``abcn``
on every other side.  These helpers generalise that to an independent choice
**per edge** — a free surface (image method) or an absorbing PML of a chosen
thickness on each of the ``2*ndim`` faces — in the spirit of deepwave's
per-edge ``pml_width`` list.

Canonical face order is **axis-major**, matching sweep's ``shape`` order and
``equations.pml``'s ``pml_width`` layout (``[Nz_front, Nz_back, Nx_left,
Nx_right]``):

    2-D:  (z_lo, z_hi, x_lo, x_hi)
    3-D:  (z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)

with the edge names

    z_lo = 'top'    z_hi = 'bottom'
    y_lo = 'front'  y_hi = 'back'      (3-D only)
    x_lo = 'left'   x_hi = 'right'

``face_index // 2`` is the shape axis (0 = z), ``face_index % 2`` is the side
(0 = low / min index, 1 = high / max index).
"""

from __future__ import annotations

# Canonical edge names per ndim, in axis-major (z_lo, z_hi, ...) order.
_EDGE_NAMES_2D = ("top", "bottom", "left", "right")
_EDGE_NAMES_3D = ("top", "bottom", "front", "back", "left", "right")


def edge_names(ndim: int) -> tuple[str, ...]:
    """Canonical face names for ``ndim`` in axis-major order."""
    if ndim == 2:
        return _EDGE_NAMES_2D
    if ndim == 3:
        return _EDGE_NAMES_3D
    raise ValueError(f"per-edge free surface only defined for ndim in (2, 3); got {ndim}")


def _name_to_index(ndim: int) -> dict[str, int]:
    return {name: i for i, name in enumerate(edge_names(ndim))}


def face_axis_side(face_index: int, ndim: int) -> tuple[int, int]:
    """Map a canonical face index to ``(shape_axis, side)``.

    ``shape_axis`` is 0-based in shape order (0 = z, last = x); ``side`` is
    0 for the low (min-index) face and 1 for the high (max-index) face.
    """
    return face_index // 2, face_index % 2


def field_z_like_axis(shape_axis: int, ndim: int) -> int:
    """Negative (from-the-end) axis of ``shape_axis`` in a field tensor.

    Wavefields carry leading batch/channel dims, so equations address spatial
    axes from the end: for 2-D ``z`` (shape axis 0) is ``-2`` and ``x`` is
    ``-1``; for 3-D ``z`` is ``-3``, ``y`` ``-2``, ``x`` ``-1``.
    """
    return -(ndim - shape_axis)


def normalize_free_surface(spec, ndim: int) -> tuple[bool, ...]:
    """Return a length-``2*ndim`` ``tuple[bool]`` of free-surface faces.

    Accepts, all normalised to the canonical axis-major order:

    * ``bool`` — ``True`` → top (z-min) face only (historical meaning);
      ``False`` → no free surface.
    * a str or iterable of edge-name strings (e.g. ``'top'``,
      ``['top', 'left']``, ``{'bottom', 'right'}``) — those faces on.
    * a length-``2*ndim`` sequence of bool/int in canonical order.
    * a ``dict`` mapping edge name → truthy.

    ``free_surface=True`` therefore stays bit-for-bit the old top-only path.
    """
    n = 2 * ndim

    if isinstance(spec, bool):
        faces = [False] * n
        if spec:
            faces[0] = True  # top (z_lo)
        return tuple(faces)

    if spec is None:
        return tuple([False] * n)

    name_to_index = _name_to_index(ndim)

    if isinstance(spec, str):
        spec = [spec]

    if isinstance(spec, dict):
        faces = [False] * n
        for key, value in spec.items():
            if key not in name_to_index:
                raise ValueError(
                    f"unknown free-surface edge {key!r}; valid names for "
                    f"{ndim}-D: {edge_names(ndim)}"
                )
            faces[name_to_index[key]] = bool(value)
        return tuple(faces)

    items = list(spec)

    # A collection of edge-name strings.
    if items and all(isinstance(it, str) for it in items):
        faces = [False] * n
        for name in items:
            if name not in name_to_index:
                raise ValueError(
                    f"unknown free-surface edge {name!r}; valid names for "
                    f"{ndim}-D: {edge_names(ndim)}"
                )
            faces[name_to_index[name]] = True
        return tuple(faces)

    # A per-face bool/int mask.
    if len(items) != n:
        raise ValueError(
            f"free_surface sequence must have length {n} for {ndim}-D "
            f"(order {edge_names(ndim)}); got length {len(items)}"
        )
    return tuple(bool(v) for v in items)


def normalize_pad(abcn, fs_faces: tuple[bool, ...], ndim: int) -> tuple[int, ...]:
    """Return the per-face PML pad widths (length ``2*ndim``, axis-major).

    ``abcn`` is an ``int`` (broadcast to every face) or a length-``2*ndim``
    sequence in canonical order.  Any face that is a free surface is forced to
    pad ``0`` (its layout is the image-method halo, not a PML), matching the
    historical top-only rule ``padding_z = (0, abcn)``.
    """
    n = 2 * ndim

    if isinstance(abcn, bool):  # guard: bool is an int subclass, reject explicitly
        raise TypeError("abcn must be an int or a sequence of ints, not bool")

    if isinstance(abcn, int):
        widths = [abcn] * n
    else:
        widths = list(abcn)
        if len(widths) != n:
            raise ValueError(
                f"abcn sequence must have length {n} for {ndim}-D "
                f"(order {edge_names(ndim)}); got length {len(widths)}"
            )
        widths = [int(w) for w in widths]

    if any(w < 0 for w in widths):
        raise ValueError(f"abcn widths must be non-negative; got {widths}")

    return tuple(0 if fs_faces[i] else widths[i] for i in range(n))


def is_top_only_or_none(fs_faces: tuple[bool, ...]) -> bool:
    """True when ``fs_faces`` is the historical default: no face, or the top
    (z-min) face alone.  Anything else is a genuine per-edge configuration."""
    return not any(fs_faces[1:])


def fs_faces_to_c_bitmask(fs_faces: tuple[bool, ...], ndim: int) -> int:
    """Encode ``fs_faces`` as the C ``SolverContext`` bitmask.

    The C side numbers axes 0=z, 1=y, 2=x (always 3 slots); bit ``2*axis`` is
    that axis' low face, ``2*axis+1`` its high face.  ``fs_faces`` is in Python
    shape-axis order (2-D: z, x; 3-D: z, y, x), so the x axis maps to C axis 2.
    Returns 0 for no free surface (distinct from the C ``-1`` "legacy" sentinel,
    which the propagator never sends once it sets this explicitly).
    """
    # Python shape-axis -> C axis
    cmap = (0, 2) if ndim == 2 else (0, 1, 2)
    bits = 0
    for pax in range(ndim):
        cax = cmap[pax]
        if fs_faces[2 * pax]:
            bits |= 1 << (2 * cax)
        if fs_faces[2 * pax + 1]:
            bits |= 1 << (2 * cax + 1)
    return bits


def torch_pad_order(per_axis_pairs: tuple[int, ...], ndim: int) -> tuple[int, ...]:
    """Convert an axis-major ``(a0_lo, a0_hi, a1_lo, a1_hi, ...)`` tuple to the
    ``torch.nn.functional.pad`` order (last spatial axis first).

    For a field laid out ``(..., z[, y], x)`` torch expects
    ``(x_lo, x_hi[, y_lo, y_hi], z_lo, z_hi)`` — i.e. axes reversed, sides kept.
    This is exactly the order the propagator's ``self.padding`` has always used.
    """
    out: list[int] = []
    for axis in reversed(range(ndim)):
        out.append(per_axis_pairs[2 * axis])
        out.append(per_axis_pairs[2 * axis + 1])
    return tuple(out)
