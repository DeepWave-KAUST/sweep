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

import pytest
import torch

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


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_an_inner_edge_node_is_outside_on_both_sides(dtype):
    """Regression: all three axis profiles leave a node on a layer's inner
    edge outside the layer, an ulp either side of it, on the low side as on
    the high one. The strict ``coord < lo + L`` took the low-side node in
    whenever it fell an ulp short: ``d ~ 1e-30 d_max``, ``alpha = alpha_max``,
    and every element touching it joined the C-PML band. One GLL step (a
    sixth of an element) inside, the profiles must be live."""
    from sweep.sem.cpml import elastic
    hi = 760.0000000000001
    L = 4 * (hi / 38)                                  # 80.00000000000001
    up, down = torch.tensor(float("inf"), dtype=dtype), torch.tensor(-float("inf"), dtype=dtype)
    edges = []
    for e in (torch.tensor(L, dtype=dtype), torch.tensor(hi - L, dtype=dtype)):
        edges += [torch.nextafter(e, down), e, torch.nextafter(e, up)]
    inside = torch.tensor([L - L / 24, hi - L + L / 24], dtype=dtype)
    coord = torch.cat([torch.stack(edges), inside])
    side = dict(side_low=True, side_high=True)
    outs = elastic._build_damping_axis(coord, 0.0, hi, **side, pml_L=L, d_max=400.0,
                                       alpha_max=31.4, poly=2.0, dtype=dtype)
    outs += _axis.damping_profile(coord, 0.0, hi, True, True, L, 400.0, 31.4, 2.0)
    outs += _axis.vs_axis_profile(coord, 0.0, hi, **side, pml_L=L, sigma_0=400.0,
                                  alpha_max=31.4, poly=2.0, dt=1e-3, dtype=dtype)
    for v in outs:                                     # (d, alpha), (sigma, alpha), (a, b)
        assert bool((v[:6] == 0).all()), v[:6]
        assert bool((v[6:] != 0).all()), v[6:]
