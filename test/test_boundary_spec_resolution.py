"""`resolve_boundary_spec` is all refusals and one representative number.

None of this is visible to a bit-exact gate. Every check here is either an error
path -- which a gate never runs, because a config that raises is a config it
skips -- or a value that only differs on a configuration the gate does not
cover. They are exactly the parts that would rot silently.
"""
from __future__ import annotations

import pytest

from sweep.core.arguments import resolve_boundary_spec
from sweep.equations._edges import (
    is_top_only_or_none,
    normalize_free_surface,
    normalize_pad,
)

NORMALISERS = dict(normalize_free_surface=normalize_free_surface,
                   normalize_pad=normalize_pad,
                   is_top_only_or_none=is_top_only_or_none)


class _Eq:
    """Stand-in equation; the real ones only contribute these two flags."""
    def __init__(self, free_surface=True, per_edge=False):
        self.supports_free_surface = free_surface
        self.supports_per_edge_free_surface = per_edge


def _resolve(free_surface, abcn, ndim=2, equation=None, topography=None):
    return resolve_boundary_spec(free_surface, abcn, ndim,
                                 equation if equation is not None else _Eq(per_edge=True),
                                 topography, **NORMALISERS)


# --------------------------------------------------------------------------- #
# the anisotropic refusal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fs", [True, [True, False, False, False],
                                [False, False, False, True]])
def test_anisotropic_refuses_every_form_of_free_surface(fs):
    """Including a free surface on a face that is NOT the top.

    The check is `any(fs_faces)`, not `fs_faces[0]` -- an anisotropic solver
    given a free bottom or side would otherwise sail through and produce
    confident, wrong surface physics.
    """
    with pytest.raises(NotImplementedError, match="stiffness tensor"):
        _resolve(fs, 10, equation=_Eq(free_surface=False, per_edge=True))


def test_anisotropic_without_a_free_surface_is_fine():
    fs_faces, pad, abcn = _resolve(False, 10, equation=_Eq(free_surface=False))
    assert not any(fs_faces)
    assert abcn == 10


# --------------------------------------------------------------------------- #
# the staged per-edge feature
# --------------------------------------------------------------------------- #
def test_per_edge_free_surface_is_refused_in_3d():
    with pytest.raises(NotImplementedError, match="2-D only"):
        _resolve([True, False, False, False, True, False], 10, ndim=3)


def test_per_edge_is_refused_on_an_equation_that_did_not_opt_in():
    with pytest.raises(NotImplementedError, match="per-edge free"):
        _resolve([True, True, False, False], 10, equation=_Eq(per_edge=False))


def test_per_edge_is_refused_together_with_topography():
    with pytest.raises(NotImplementedError, match="topography"):
        _resolve([True, True, False, False], 10, topography=object())


def test_a_per_edge_PML_alone_still_takes_the_staged_path():
    """`abcn` per-edge with a plain top-only free surface must still be gated:
    the trigger is `not isinstance(abcn, int)`, not only the face pattern."""
    with pytest.raises(NotImplementedError, match="per-edge free"):
        _resolve(True, (0, 10, 10, 10), equation=_Eq(per_edge=False))


def test_top_only_with_a_scalar_is_the_unguarded_path():
    fs_faces, pad, abcn = _resolve(True, 10, equation=_Eq(per_edge=False))
    assert is_top_only_or_none(fs_faces)
    assert abcn == 10
    assert pad[0] == 0, "a free top face must have pad 0, not a suppressed abcn"


# --------------------------------------------------------------------------- #
# the representative scalar
# --------------------------------------------------------------------------- #
def test_abcn_bool_is_rejected_outright():
    """`isinstance(True, int)` is True in Python, so `abcn=True` would read as a
    1-cell PML if nothing objected. `normalize_pad` rejects it first -- which
    makes the matching bool exclusion in the representative-width expression
    belt-and-braces rather than the load-bearing guard. Pinned so that removing
    EITHER of them is a visible decision."""
    with pytest.raises(TypeError, match="not bool"):
        _resolve(False, True, equation=_Eq(per_edge=True))


def test_per_edge_pad_reports_the_max_as_the_representative_width():
    _, pad, abcn = _resolve(False, (5, 10, 20, 20))
    assert abcn == max(pad), f"representative abcn {abcn} is not max(pad)={max(pad)}"
