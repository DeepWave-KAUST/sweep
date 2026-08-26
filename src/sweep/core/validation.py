"""Validation of the user-facing call signature.

``normalize_io`` decides which of the three acquisition modes a call is in, and
that decision drives buffer shapes all the way down -- so getting it wrong does
not raise, it silently runs a different problem. Hence: a pure function of the
three shapes, with every rejection spelling out what was expected and what
arrived.

The three modes, and why the distinction is load-bearing:

* **A1 / A2** -- naive multi-shot. ``sources`` is 2-D ``(nshots, ndim)`` and the
  batch axis IS the shot axis; the wavelet is either shared ``(nt,)`` or
  per-shot ``(nshots, nt)``.
* **B** -- source encoding. ``sources`` is 3-D ``(1, nsrc, ndim)``: ONE
  super-shot with ``nsrc`` superposed point sources, so the batch axis is 1 and
  ``nsrc`` lives on the middle axis. A2 and B can look alike from the wavelet
  alone -- both may be 2-D -- which is why the mode is decided by the SOURCE
  rank, never by the wavelet's.

``receivers`` is always 3-D ``(B, nrec, ndim)``; a shared receiver array must be
broadcast by the caller rather than silently here, so that ``B`` disagreeing
with the sources is an error instead of a reshape.
"""
from __future__ import annotations

import numpy as np


def shape_tuple(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.shape(value)
    return tuple(int(dim) for dim in shape)

def normalize_io(wavelet, sources, receivers, ndim):
    """Validate user-facing shapes for ``wavelet`` / ``sources`` / ``receivers``.

    The propagator accepts three input modes:

    - **A1**: ``wavelet=(nt,)``, ``sources=(nshots, ndim)``,
      ``receivers=(nshots, nrec, ndim)`` — naive multi-shot, shared wavelet.
    - **A2**: ``wavelet=(nshots, nt)``, ``sources=(nshots, ndim)``,
      ``receivers=(nshots, nrec, ndim)`` — naive multi-shot, per-shot wavelet.
    - **B**:  ``wavelet=(nt,)`` or ``(nsrc, nt)``,
      ``sources=(1, nsrc, ndim)``, ``receivers=(1, nrec, ndim)`` —
      source encoding (single super-shot, ``nsrc`` superposed point sources).

    ``receivers`` must always be 3-D; shared receiver arrays should be
    pre-broadcast/repeated to ``(B, nrec, ndim)`` by the user.

    Returns
    -------
    mode : {'A1', 'A2', 'B'}
    batch_size : int
        Internal batch dim (``nshots`` for A, ``1`` for B).
    nsrc_per_shot : int
        Number of point sources per shot (``1`` for A, ``nsrc`` for B).
    is_encoded : bool
        ``True`` iff ``mode == 'B'``.
    """
    ws = shape_tuple(wavelet)
    ss = shape_tuple(sources)
    rs = shape_tuple(receivers)

    if len(rs) != 3 or rs[-1] != ndim:
        raise ValueError(
            f"receivers must have shape (B, nrec, {ndim}); got {rs}. "
            "Pre-broadcast/repeat per-shot if you previously passed a "
            "shared (nrec, dim) array."
        )
    nrec = rs[1]

    if len(ss) == 2:
        if ss[-1] != ndim:
            raise ValueError(
                f"sources must have shape (nshots, {ndim}); got {ss}."
            )
        nshots = ss[0]
        if rs[0] != nshots:
            raise ValueError(
                f"receivers batch ({rs[0]}) must match sources nshots "
                f"({nshots}) in naive multi-shot mode."
            )
        if len(ws) == 1:
            return 'A1', nshots, 1, nrec, False
        if len(ws) == 2:
            if ws[0] != nshots:
                raise ValueError(
                    f"wavelet must have shape (nshots={nshots}, nt); got {ws}."
                )
            return 'A2', nshots, 1, nrec, False
        raise ValueError(
            "wavelet must have shape (nt,) [shared] or (nshots, nt) "
            f"[per-shot] in naive multi-shot mode; got {ws}."
        )

    if len(ss) == 3:
        if ss[0] != 1 or ss[-1] != ndim:
            raise ValueError(
                "sources in source-encoding mode must have shape "
                f"(1, nsrc, {ndim}); got {ss}."
            )
        nsrc = ss[1]
        if rs[0] != 1:
            raise ValueError(
                "receivers batch must be 1 in source-encoding mode; "
                f"got {rs[0]}."
            )
        if len(ws) == 1:
            return 'B', 1, nsrc, nrec, True
        if len(ws) == 2:
            if ws[0] != nsrc:
                raise ValueError(
                    f"wavelet must have shape (nt,) or (nsrc={nsrc}, nt) "
                    f"in source-encoding mode; got {ws}."
                )
            return 'B', 1, nsrc, nrec, True
        raise ValueError(
            "wavelet must have shape (nt,) or (nsrc, nt) in "
            f"source-encoding mode; got {ws}."
        )

    raise ValueError(
        f"sources must have shape (nshots, {ndim}) [naive multi-shot] "
        f"or (1, nsrc, {ndim}) [source encoding]; got {ss}."
    )

