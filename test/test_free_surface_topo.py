"""Unit tests for the per-column (irregular topography) staircase helper
``zero_above_topo`` in ``sweep.equations._free_surface`` -- the acoustic
staircase's vacuum air.  (The elastic image-method staircase and its
per-column mirror helpers were removed.)

Tests
-----
1. zero_above_topo_clears_air_only
   — masking helper clears exactly ``z < iz_surf`` cells.
2. batched_input_shape_preserved
   — ``(B, C, nz, nx)`` with ``iz_surf=(nx,)`` broadcasts and matches the
     per-sample result.
3. three_d_field
   — ``(nz, ny, nx)`` with ``iz_surf=(ny, nx)`` and ``axis=-3``.
4. invalid_iz_surf_ndim_raises
   — wrong-rank ``iz_surf`` raises ``ValueError``.
5. non_torch_input_raises
   — numpy ``u`` raises ``NotImplementedError`` (torch-only path).
6. iz_surf_as_python_list
   — list input auto-converts to ``LongTensor``.
"""

import numpy as np
import pytest
import torch

from sweep.equations._free_surface import zero_above_topo


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
    u = torch.randn(B, C, nz, nx)
    iz_surf = torch.tensor([3, 4, 5, 4, 3], dtype=torch.long)
    out = zero_above_topo(u, iz_surf, axis=-2)
    assert out.shape == u.shape
    for b in range(B):
        for c in range(C):
            assert torch.equal(out[b, c], zero_above_topo(u[b, c], iz_surf, axis=-2))


def test_three_d_field():
    torch.manual_seed(2)
    nz, ny, nx = 6, 2, 3
    u = torch.randn(nz, ny, nx)
    iz_surf = torch.tensor([[1, 2, 3], [0, 4, 2]], dtype=torch.long)
    out = zero_above_topo(u, iz_surf, axis=-3)
    for iy in range(ny):
        for ix in range(nx):
            s = int(iz_surf[iy, ix])
            assert torch.all(out[:s, iy, ix] == 0)
            assert torch.equal(out[s:, iy, ix], u[s:, iy, ix])


def test_invalid_iz_surf_ndim_raises():
    u = torch.zeros(4, 3)
    bad_iz_surf = torch.tensor([[2, 1, 2]], dtype=torch.long)  # rank 2; expected rank 1
    with pytest.raises(ValueError, match="iz_surf.ndim"):
        zero_above_topo(u, bad_iz_surf, axis=-2)


def test_non_torch_input_raises():
    u_np = np.zeros((4, 3), dtype=np.float32)
    iz_surf = np.array([1, 1, 1], dtype=np.int64)
    with pytest.raises(NotImplementedError):
        zero_above_topo(u_np, iz_surf, axis=-2)


def test_iz_surf_as_python_list():
    u = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    out_list = zero_above_topo(u, [2, 1, 2], axis=-2)
    out_tensor = zero_above_topo(u, torch.tensor([2, 1, 2], dtype=torch.long), axis=-2)
    assert torch.equal(out_list, out_tensor)
