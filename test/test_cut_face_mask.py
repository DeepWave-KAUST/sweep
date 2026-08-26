"""The DD cut mask and the free-surface mask use OPPOSITE axis orders.

Both are face bitmasks, both live in `SolverContext`, and they disagree:

    cut_mask (context.h)   bit0=x_lo bit1=x_hi bit2=z_lo bit3=z_hi bit4=y_lo bit5=y_hi
    fs bitmask (_edges.py) bit 2*axis / 2*axis+1 with axis 0=z, 1=y, 2=x
                        => bit0=z_lo bit1=z_hi bit2=y_lo bit3=y_hi bit4=x_lo bit5=x_hi

Anyone tidying these into one convention breaks the CUDA side silently: a cut
face would be read as a free surface, which reflects instead of exchanging, and
the wavefield would be wrong only in the interior of a decomposed run -- the
hardest place to notice. So the header is parsed and compared here, rather than
trusting two hand-kept copies of the same six numbers to stay in step.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from sweep.equations._edges import fs_faces_to_c_bitmask
from sweep.parallel import pml as P

HEADER = (pathlib.Path(P.__file__).resolve().parents[1]
          / "csrc" / "cuda" / "common" / "context.h")


def _cut_bits_from_header():
    """`cut_x_lo() const { return cut_mask & 1; }` -> {'x_lo': 1, ...}"""
    text = HEADER.read_text()
    found = dict(re.findall(r"cut_([xyz]_(?:lo|hi))\(\)\s*const\s*\{\s*return\s+cut_mask\s*&\s*(\d+)",
                            text))
    return {k: int(v) for k, v in found.items()}


@pytest.mark.skipif(not HEADER.exists(), reason=f"header not found at {HEADER}")
def test_python_cut_bits_match_the_cuda_header():
    from_header = _cut_bits_from_header()
    assert from_header, f"parsed no cut_* accessors out of {HEADER}"
    from_python = {"x_lo": P.CUT_X_LO, "x_hi": P.CUT_X_HI, "z_lo": P.CUT_Z_LO,
                   "z_hi": P.CUT_Z_HI, "y_lo": P.CUT_Y_LO, "y_hi": P.CUT_Y_HI}
    assert from_python == from_header, (
        f"Python cut bits {from_python} disagree with {HEADER.name} {from_header}")


def test_the_two_masks_really_do_disagree():
    """Pins the divergence itself.

    If someone 'fixes' either side to match the other, this fails and says so --
    which is the point. The masks are allowed to differ; they are not allowed to
    differ by accident.
    """
    fs_x_lo = fs_faces_to_c_bitmask((False, True, False, False), 2)   # 2-D (z, x): x_lo
    assert fs_x_lo != P.CUT_X_LO, (
        "the free-surface and cut masks now agree on x_lo; if that is intended, "
        "context.h must change too")


class _Mesh:
    def __init__(self, edges):
        self._edges = edges

    def is_edge(self, axis, side):
        return self._edges.get((axis, side), True)


def test_no_mesh_means_no_cut():
    assert P.dd_cut_face_mask(None, 2) == 0
    assert P.dd_cut_face_mask(_Mesh({}), 3) == 0


def test_interior_tile_sets_every_splittable_face():
    interior = _Mesh({("x", "low"): False, ("x", "high"): False,
                      ("y", "low"): False, ("y", "high"): False})
    assert P.dd_cut_face_mask(interior, 2) == P.CUT_X_LO | P.CUT_X_HI
    assert P.dd_cut_face_mask(interior, 3) == (P.CUT_X_LO | P.CUT_X_HI
                                               | P.CUT_Y_LO | P.CUT_Y_HI)


def test_z_is_never_cut():
    """The depth axis is never split, so no z bit may ever be set -- even in 3-D
    where the C side defines them."""
    interior = _Mesh({("x", "low"): False, ("x", "high"): False,
                      ("y", "low"): False, ("y", "high"): False,
                      ("z", "low"): False, ("z", "high"): False})
    mask = P.dd_cut_face_mask(interior, 3)
    assert not (mask & (P.CUT_Z_LO | P.CUT_Z_HI))


def test_pad_is_left_alone_without_a_mesh():
    pad = (0, 20, 20, 20)
    assert P.dd_cut_pad(None, pad, abcn=20, ndim=2) == pad
