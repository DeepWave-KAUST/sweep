"""The per-axis damping profiles are SHARED, not forked.

Two byte-identical copies of the acoustic (sigma, alpha) construction and two
formatting-variant copies of the velocity-stress (a, b) recursion used to live
in four builder modules. They now come from cpml/_axis; at the refactor every
tensor of every profile (55 across the five builders) was bit-compared against
a pre-refactor snapshot -- zero differences. These tests pin the sharing
itself: the regression to guard against is someone re-forking a local copy,
which no output comparison would notice until the copies drift.
"""
from __future__ import annotations

from sweep.sem.cpml import _axis, acoustic, acoustic3d, elastic_vs, elastic_vs3d


def test_acoustic_pair_shares_one_damping_profile():
    assert acoustic._damping_profile is _axis.damping_profile
    assert acoustic3d._damping_profile is _axis.damping_profile


def test_velocity_stress_pair_shares_one_axis_profile():
    assert elastic_vs._axis_profile is _axis.vs_axis_profile
    assert elastic_vs3d._axis_profile is _axis.vs_axis_profile


def test_the_displacement_port_is_deliberately_not_shared():
    """cpml/elastic.py (the SPECFEM2D strain-stretching port) keeps its own
    _build_damping_axis on purpose -- it feeds a three-coefficient scheme.
    If someone routes it through _axis, this should be a conscious change."""
    from sweep.sem.cpml import elastic
    assert hasattr(elastic, "_build_damping_axis")
    assert elastic._build_damping_axis is not _axis.damping_profile
