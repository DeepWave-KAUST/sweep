"""Unit tests for ``sweep.equations._topography`` (the APM utility
module). Verified in isolation before the ``ElasticAPM`` class is
wired up."""

import numpy as np
import pytest
import torch

from sweep.equations._topography import (
    AIR, H, IC, INTERIOR, OC, VL, VR,
    classify_topography,
    enforce_apm_traction_bc,
    precompute_apm_moduli,
    zero_at_air,
)


# ---------------------------------------------------------------------------
# classify_topography
# ---------------------------------------------------------------------------


def test_classify_flat_topo_gives_only_H_and_interior():
    """Flat air block on top → first solid row is H, the rest INTERIOR."""
    nz, nx = 8, 10
    am = np.zeros((nz, nx), dtype=bool)
    am[:3, :] = True  # rows 0-2 are air
    cat = classify_topography(am)

    assert (cat[:3, :] == AIR).all()
    assert (cat[3, :] == H).all(), f"expected H row, got {cat[3, :]}"
    assert (cat[4:, :] == INTERIOR).all()


def test_classify_step_creates_H_VL_VR_OC():
    """A 1-cell-high step in the topo produces VL/VR/OC corners.

    air_mask layout (10 cols × 6 rows, row 0 = top):

        rows 0:    AIR AIR AIR AIR AIR AIR AIR AIR AIR AIR
        rows 1:    AIR AIR AIR AIR AIR -   -   -   -   -    (step up at col 5)
        rows 2+:   -   -   -   -   -   -   -   -   -   -
    """
    nz, nx = 6, 10
    am = np.zeros((nz, nx), dtype=bool)
    am[0, :] = True
    am[1, :5] = True
    cat = classify_topography(am)

    # First solid row in cols 0-4 is row 2 (air both above and at row 1).
    assert (cat[2, :5] == H).all(), f"got cols 0-4: {cat[2, :5]}"
    # Row 1 col 5 is solid, with air above and air to the left → OC.
    assert cat[1, 5] == OC, f"corner classified as {cat[1, 5]}, expected OC ({OC})"
    # Row 1 cols 6-9: solid with only air above → H.
    assert (cat[1, 6:] == H).all(), f"got cols 6-9: {cat[1, 6:]}"
    # Row 2 col 4: solid, air at row 1 col 4 (above) → H, not VR.
    assert cat[2, 4] == H


def test_classify_inner_corner_IC():
    """L-shaped notch where solid wraps the corner — the inside-corner
    cell sees air only on the diagonal, so it must be IC."""
    nz, nx = 5, 5
    am = np.zeros((nz, nx), dtype=bool)
    am[0, :] = True   # whole top row
    am[1, 0] = True   # one extra air below the NW
    cat = classify_topography(am)

    # (1, 1) is solid; (1, 0) is air (4-conn left), (0, 1) is air
    # (4-conn above), so it's actually OC. Skip — that's not IC.
    # The IC test needs a cell whose 4-conn are ALL solid but a
    # diagonal is air.
    am = np.zeros((nz, nx), dtype=bool)
    am[0, 0] = True   # only the NW corner is air; (1,1) sees diagonal air.
    cat = classify_topography(am)
    assert cat[1, 1] == IC, (
        f"expected IC at (1, 1) with only diag air; got {cat[1, 1]}"
    )


def test_classify_accepts_torch_input():
    am = torch.zeros(4, 6, dtype=torch.bool)
    am[:2, :] = True
    cat = classify_topography(am)
    assert isinstance(cat, np.ndarray)
    assert (cat[2, :] == H).all()


# ---------------------------------------------------------------------------
# precompute_apm_moduli
# ---------------------------------------------------------------------------


def test_precompute_interior_keeps_bulk():
    nz, nx = 5, 5
    lam = torch.full((nz, nx), 2.0)
    mu = torch.full((nz, nx), 1.0)
    rho = torch.full((nz, nx), 1000.0)
    cat = np.full((nz, nx), INTERIOR, dtype=np.int32)
    lam_eff, mu_eff, mu_xz, rx, rz = precompute_apm_moduli(lam, mu, rho, cat)
    assert torch.allclose(lam_eff, lam)
    assert torch.allclose(mu_eff, mu)
    assert torch.allclose(mu_xz, mu)
    assert torch.allclose(rx, rho)
    assert torch.allclose(rz, rho)


def test_precompute_H_cell_apm_identity():
    """At an H cell, ``(λ_eff + 2 μ_eff)`` must equal α and ``λ_eff``
    must be 0 (so ``∂σ_xx/∂t = α · ∂v_x/∂x``)."""
    lam = torch.tensor([[2.0]])
    mu = torch.tensor([[1.0]])
    rho = torch.tensor([[1000.0]])
    cat = np.array([[H]], dtype=np.int32)
    lam_eff, mu_eff, mu_xz, rx, rz = precompute_apm_moduli(lam, mu, rho, cat)

    alpha = 2.0 * 1.0 * (2.0 + 1.0) / (2.0 + 2 * 1.0)   # = 1.5
    assert torch.isclose(lam_eff, torch.tensor([[0.0]]))
    assert torch.isclose(mu_eff, torch.tensor([[alpha / 2]]))
    assert torch.isclose(lam_eff + 2 * mu_eff, torch.tensor([[alpha]]))
    assert torch.isclose(mu_xz, torch.tensor([[0.5]]))
    assert torch.isclose(rx, torch.tensor([[500.0]]))
    assert torch.isclose(rz, torch.tensor([[1000.0]]))


def test_precompute_density_rows_per_dong_2023_table_1():
    """Densities per category match Dong 2023 Table 1 (2-D limit)."""
    lam = torch.full((1, 7), 2.0)
    mu = torch.full((1, 7), 1.0)
    rho = torch.full((1, 7), 1000.0)
    cat = np.array([[INTERIOR, H, VL, VR, OC, IC, AIR]], dtype=np.int32)
    _, _, _, rx, rz = precompute_apm_moduli(lam, mu, rho, cat)
    rx_expected = torch.tensor([[1000., 500., 1000., 500., 250., 750., 1000.]])
    rz_expected = torch.tensor([[1000., 1000., 500., 500., 250., 750., 1000.]])
    assert torch.allclose(rx, rx_expected)
    assert torch.allclose(rz, rz_expected)


def test_precompute_OC_zeros_all_stiffness():
    lam = torch.full((1, 1), 2.0)
    mu = torch.full((1, 1), 1.0)
    rho = torch.full((1, 1), 1000.0)
    cat = np.array([[OC]], dtype=np.int32)
    lam_eff, mu_eff, mu_xz, _, _ = precompute_apm_moduli(lam, mu, rho, cat)
    assert lam_eff.item() == 0.0
    assert mu_eff.item() == 0.0
    assert mu_xz.item() == 0.0


def test_precompute_AIR_zero_stiff_keep_rho():
    """AIR: moduli are zero, but ρ stays at bulk to avoid dt/ρ blow-up."""
    lam = torch.full((1, 1), 2.0)
    mu = torch.full((1, 1), 1.0)
    rho = torch.full((1, 1), 1000.0)
    cat = np.array([[AIR]], dtype=np.int32)
    lam_eff, mu_eff, mu_xz, rx, rz = precompute_apm_moduli(lam, mu, rho, cat)
    assert lam_eff.item() == 0.0
    assert mu_eff.item() == 0.0
    assert mu_xz.item() == 0.0
    assert rx.item() == 1000.0
    assert rz.item() == 1000.0


# ---------------------------------------------------------------------------
# zero_at_air & enforce_apm_traction_bc
# ---------------------------------------------------------------------------


def test_zero_at_air_kills_air_cells():
    field = torch.ones(4, 5)
    air = torch.zeros(4, 5)
    air[:2, :] = 1.0
    out = zero_at_air(field, air)
    assert (out[:2, :] == 0).all()
    assert (out[2:, :] == 1).all()


def test_enforce_traction_bc_per_category():
    cat = np.array([[INTERIOR, H, VL, VR, OC]], dtype=np.int32)
    sxx = torch.ones(1, 5)
    szz = torch.ones(1, 5)
    sxz = torch.ones(1, 5)
    sxx_o, szz_o, sxz_o = enforce_apm_traction_bc(sxx, szz, sxz, cat)
    # σ_xx: zeroed at VL, VR, OC (and AIR — not in this fixture).
    assert sxx_o.tolist() == [[1., 1., 0., 0., 0.]]
    # σ_zz: zeroed at H, OC.
    assert szz_o.tolist() == [[1., 0., 1., 1., 0.]]
    # σ_xz: zeroed at OC only.
    assert sxz_o.tolist() == [[1., 1., 1., 1., 0.]]
