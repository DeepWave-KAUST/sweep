"""`resolve_boundary_spec` is all refusals and one representative number.

None of this is visible to a bit-exact gate. Every check here is either an error
path -- which a gate never runs, because a config that raises is a config it
skips -- or a value that only differs on a configuration the gate does not
cover. They are exactly the parts that would rot silently.
"""
from __future__ import annotations

import itertools

import pytest

from sweep.core.arguments import resolve_boundary_spec
from sweep.equations._edges import (
    is_top_only_or_none,
    normalize_free_surface,
    normalize_pad,
    refuse_free_plate,
)

NORMALISERS = dict(normalize_free_surface=normalize_free_surface,
                   normalize_pad=normalize_pad,
                   is_top_only_or_none=is_top_only_or_none,
                   refuse_free_plate=refuse_free_plate)


class _Eq:
    """Stand-in equation; the real ones only contribute these three flags."""
    def __init__(self, free_surface=True, per_edge=False, solid=False):
        self.supports_free_surface = free_surface
        self.supports_per_edge_free_surface = per_edge
        self.free_plate_unstable = solid


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


# --------------------------------------------------------------------------- #
# the free-plate refusal (solids only)
# --------------------------------------------------------------------------- #
FACES = ("top", "bottom", "left", "right")
LAYOUTS = [c for r in range(5) for c in itertools.combinations(FACES, r)]
# A PML on one axis while both faces of the other axis are free: the solid
# between the free faces is a plate.  Measured over 10 s (eager and impl='c'),
# exactly these six of the sixteen layouts grow without bound.
PLATES = {("top", "bottom"), ("left", "right"),
          ("top", "bottom", "left"), ("top", "bottom", "right"),
          ("top", "left", "right"), ("bottom", "left", "right")}


@pytest.mark.parametrize("fs", LAYOUTS, ids=["+".join(f) or "none" for f in LAYOUTS])
def test_a_solid_refuses_exactly_the_plate_layouts(fs):
    solid = _Eq(per_edge=True, solid=True)
    if fs in PLATES:
        with pytest.raises(ValueError, match="plate"):
            _resolve(list(fs) or False, 10, equation=solid)
    else:
        _resolve(list(fs) or False, 10, equation=solid)


@pytest.mark.parametrize("fs", sorted(PLATES))
def test_a_fluid_accepts_the_plate_layouts(fs):
    """A fluid layer has no backward-propagating modes: measured stable."""
    _resolve(list(fs), 10, equation=_Eq(per_edge=True, solid=False))


def test_the_refusal_names_the_pml_and_the_free_pair():
    with pytest.raises(ValueError) as err:
        _resolve(["top", "left", "right"], 10, equation=_Eq(per_edge=True, solid=True))
    msg = str(err.value)
    assert "['bottom']" in msg, msg
    assert "'left' (free surface)" in msg and "'right' (free surface)" in msg, msg


def test_a_face_without_pml_counts_against_a_free_one():
    """`abcn` 0 on a face that is not free leaves it unabsorbed too: opposite a
    free surface it bounds the same plate (measured: grows like free + free)."""
    solid = _Eq(per_edge=True, solid=True)
    with pytest.raises(ValueError, match=r"'bottom' \(abcn=0\)"):
        _resolve(["top"], (10, 0, 10, 10), equation=solid)
    with pytest.raises(ValueError, match=r"'right' \(abcn=0\)"):
        _resolve(["top", "left"], (10, 10, 10, 0), equation=solid)


def test_two_faces_without_pml_and_without_a_free_surface_pass():
    """Two unpadded, non-free faces face to face were measured stable (40 s),
    and one unpadded face per axis leaves nothing to bound."""
    solid = _Eq(per_edge=True, solid=True)
    _resolve(False, (0, 0, 10, 10), equation=solid)
    _resolve(False, (10, 10, 0, 0), equation=solid)
    _resolve(["top"], (10, 10, 10, 0), equation=solid)


def test_no_pml_anywhere_is_not_a_plate():
    """A closed box (every face free) or a run without PML has no layer to
    amplify anything."""
    solid = _Eq(per_edge=True, solid=True)
    _resolve(list(FACES), 10, equation=solid)
    _resolve(False, 0, equation=solid)
