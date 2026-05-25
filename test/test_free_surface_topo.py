"""Unit tests for per-column (irregular topography) image-method helpers.

Covers ``extend_top_free_surface_topo``, ``top_free_surface_derivative_topo``,
and ``zero_above_topo`` in ``sweep.equations._free_surface``.

Tests
-----
1. flat_degenerate_matches_extend_top_free_surface
   — ``iz_surf = halo*ones`` reproduces the flat helper bit-for-bit.
2. flat_degenerate_odd_parity
   — same with ``odd=True``.
3. hand_computed_2d_mirror
   — small (4, 3) tensor, ``iz_surf=[2, 1, 2]``, hand-traced expectation.
4. odd_flag_negates_only_air
   — sign flip applies to above-surface cells only.
5. halo_zero_returns_input_unchanged
   — early-out preserved.
6. zero_above_topo_clears_air_only
   — masking helper clears exactly ``z < iz_surf`` cells.
7. batched_input_shape_preserved
   — ``(B, C, nz, nx)`` with ``iz_surf=(nx,)`` broadcasts and matches the
     per-sample result.
8. top_derivative_topo_wraps_extend_and_deriv
   — wrapper equals ``deriv(extend_topo(...))``.
9. invalid_iz_surf_ndim_raises
   — wrong-rank ``iz_surf`` raises ``ValueError``.
10. non_torch_input_raises
    — numpy ``u`` raises ``NotImplementedError`` (torch-only path for now).
11. iz_surf_as_python_list
    — list input auto-converts to ``LongTensor``.
12. flat_degenerate_3d_matches
    — same flat-equivalence check for a 3D field with ``axis=-3``.
"""

import numpy as np
import pytest
import torch

from sweep.equations._free_surface import (
    extend_top_free_surface,
    extend_top_free_surface_topo,
    top_free_surface_derivative_topo,
    zero_above_topo,
    zero_at_topo,
    zero_top_row,
)


# ---------------------------------------------------------------------------
# Flat-degenerate equivalence
# ---------------------------------------------------------------------------


def test_flat_degenerate_matches_extend_top_free_surface():
    torch.manual_seed(0)
    nz, nx = 12, 7
    halo = 2
    u = torch.randn(nz, nx)
    iz_surf = torch.full((nx,), halo, dtype=torch.long)

    out_flat = extend_top_free_surface(u, halo, odd=False, axis=-2)
    out_topo = extend_top_free_surface_topo(u, halo, odd=False, axis=-2, iz_surf=iz_surf)
    assert torch.allclose(out_flat, out_topo, atol=1e-7), (
        f"flat-degenerate mismatch: max abs diff "
        f"{(out_flat - out_topo).abs().max().item()}"
    )


def test_flat_degenerate_odd_parity():
    torch.manual_seed(1)
    nz, nx = 12, 7
    halo = 2
    u = torch.randn(nz, nx)
    iz_surf = torch.full((nx,), halo, dtype=torch.long)

    out_flat = extend_top_free_surface(u, halo, odd=True, axis=-2)
    out_topo = extend_top_free_surface_topo(u, halo, odd=True, axis=-2, iz_surf=iz_surf)
    assert torch.allclose(out_flat, out_topo, atol=1e-7)


def test_flat_degenerate_3d_matches():
    """3D field (nz, ny, nx); flat surface across (ny, nx) reproduces flat helper."""
    torch.manual_seed(2)
    nz, ny, nx = 10, 4, 5
    halo = 2
    u = torch.randn(nz, ny, nx)
    iz_surf = torch.full((ny, nx), halo, dtype=torch.long)

    out_flat = extend_top_free_surface(u, halo, odd=False, axis=-3)
    out_topo = extend_top_free_surface_topo(u, halo, odd=False, axis=-3, iz_surf=iz_surf)
    assert torch.allclose(out_flat, out_topo, atol=1e-7)


# ---------------------------------------------------------------------------
# Hand-computed reference
# ---------------------------------------------------------------------------


def test_hand_computed_2d_mirror():
    """4x3 tensor with iz_surf=[2, 1, 2]; hand-traced expectation."""
    u = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    # u =
    #   row 0: [ 0,  1,  2]
    #   row 1: [ 3,  4,  5]
    #   row 2: [ 6,  7,  8]
    #   row 3: [ 9, 10, 11]
    # iz_surf = [2, 1, 2]; mirror_z = 2*surf - z (clamped to [0, 3]):
    #   col 0 (surf=2): z=0 -> 4 clamp 3 ; z=1 -> 3 ; z=2 -> 2 ; z=3 -> 3
    #   col 1 (surf=1): z=0 -> 2          ; z=1 -> 1 ; z=2 -> 2 ; z=3 -> 3
    #   col 2 (surf=2): z=0 -> 4 clamp 3 ; z=1 -> 3 ; z=2 -> 2 ; z=3 -> 3
    expected = torch.tensor(
        [
            [9.0, 7.0, 11.0],
            [9.0, 4.0, 11.0],
            [6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0],
        ]
    )
    iz_surf = torch.tensor([2, 1, 2], dtype=torch.long)
    out = extend_top_free_surface_topo(u, halo=2, odd=False, axis=-2, iz_surf=iz_surf)
    assert torch.allclose(out, expected), f"got {out}\nexpected {expected}"


def test_odd_flag_negates_only_air():
    u = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    iz_surf = torch.tensor([2, 1, 2], dtype=torch.long)
    out_even = extend_top_free_surface_topo(u, halo=2, odd=False, axis=-2, iz_surf=iz_surf)
    out_odd = extend_top_free_surface_topo(u, halo=2, odd=True, axis=-2, iz_surf=iz_surf)

    # interior (z >= iz_surf) must be identical
    for iz, ix in [(2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (2, 2), (3, 2)]:
        assert out_even[iz, ix] == out_odd[iz, ix], f"interior changed at ({iz},{ix})"
    # air cells must be negated
    for iz, ix in [(0, 0), (1, 0), (0, 1), (0, 2), (1, 2)]:
        assert out_odd[iz, ix] == -out_even[iz, ix], f"air not negated at ({iz},{ix})"


# ---------------------------------------------------------------------------
# Edge cases / API contract
# ---------------------------------------------------------------------------


def test_halo_zero_returns_input_unchanged():
    u = torch.randn(5, 4)
    iz_surf = torch.tensor([1, 2, 3, 0], dtype=torch.long)
    out = extend_top_free_surface_topo(u, halo=0, odd=False, axis=-2, iz_surf=iz_surf)
    assert out is u


def test_zero_above_topo_clears_air_only():
    u = torch.ones(4, 3)
    iz_surf = torch.tensor([2, 1, 2], dtype=torch.long)
    out = zero_above_topo(u, iz_surf, axis=-2)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(out, expected)


def test_batched_input_shape_preserved():
    torch.manual_seed(3)
    B, C, nz, nx = 2, 3, 10, 5
    halo = 2
    u = torch.randn(B, C, nz, nx)
    iz_surf = torch.tensor([3, 4, 5, 4, 3], dtype=torch.long)
    out = extend_top_free_surface_topo(u, halo, odd=False, axis=-2, iz_surf=iz_surf)
    assert out.shape == u.shape

    # Per-sample equivalence: every (b, c) slice must equal the un-batched call.
    for b in range(B):
        for c in range(C):
            single = extend_top_free_surface_topo(
                u[b, c], halo, odd=False, axis=-2, iz_surf=iz_surf
            )
            assert torch.allclose(out[b, c], single, atol=1e-7)


def test_top_derivative_topo_wraps_extend_and_deriv():
    torch.manual_seed(4)
    u = torch.randn(8, 6)
    iz_surf = torch.tensor([2, 2, 3, 3, 2, 2], dtype=torch.long)

    def deriv(x):
        return x[1:] - x[:-1]

    ext = extend_top_free_surface_topo(u, halo=2, odd=False, axis=-2, iz_surf=iz_surf)
    out_manual = deriv(ext)
    out_wrapper = top_free_surface_derivative_topo(
        u, deriv, halo=2, odd=False, axis=-2, iz_surf=iz_surf
    )
    assert torch.allclose(out_manual, out_wrapper, atol=1e-7)


def test_invalid_iz_surf_ndim_raises():
    u = torch.zeros(4, 3)
    bad_iz_surf = torch.tensor([[2, 1, 2]], dtype=torch.long)  # rank 2; expected rank 1
    with pytest.raises(ValueError, match="iz_surf.ndim"):
        extend_top_free_surface_topo(u, halo=1, odd=False, axis=-2, iz_surf=bad_iz_surf)


def test_non_torch_input_raises():
    u_np = np.zeros((4, 3), dtype=np.float32)
    iz_surf = np.array([1, 1, 1], dtype=np.int64)
    with pytest.raises(NotImplementedError):
        extend_top_free_surface_topo(u_np, halo=1, odd=False, axis=-2, iz_surf=iz_surf)


def test_iz_surf_as_python_list():
    u = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out_list = extend_top_free_surface_topo(
        u, halo=2, odd=False, axis=-2, iz_surf=[2, 1, 2]
    )
    out_tensor = extend_top_free_surface_topo(
        u, halo=2, odd=False, axis=-2,
        iz_surf=torch.tensor([2, 1, 2], dtype=torch.long),
    )
    assert torch.allclose(out_list, out_tensor)


# ---------------------------------------------------------------------------
# zero_at_topo (per-column surface-row zero, for elastic image-method FS)
# ---------------------------------------------------------------------------


def test_zero_at_topo_clears_exactly_surface_row():
    u = torch.ones(4, 3)
    iz_surf = torch.tensor([2, 1, 2], dtype=torch.long)
    out = zero_at_topo(u, iz_surf, axis=-2)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],  # col 1: surf=1 ⇒ row 1 zeroed
            [0.0, 1.0, 0.0],  # col 0, 2: surf=2 ⇒ row 2 zeroed
            [1.0, 1.0, 1.0],
        ]
    )
    assert torch.allclose(out, expected)


def test_zero_at_topo_constant_surf_matches_zero_top_row():
    """When iz_surf = halo*ones (the flat-degenerate case) zero_at_topo
    must match the existing zero_top_row used by the flat elastic path."""
    torch.manual_seed(7)
    nz, nx = 8, 5
    halo = 2
    u = torch.randn(nz, nx)
    iz_surf = torch.full((nx,), halo, dtype=torch.long)
    out_flat = zero_top_row(u, halo, axis=-2)
    out_topo = zero_at_topo(u, iz_surf, axis=-2)
    assert torch.allclose(out_flat, out_topo, atol=1e-7)
