"""Single-process tests for :mod:`sweep.parallel.padding`.

The two properties that make the pad safe to use around DD:

* indices survive — high side only, z never touched, so source / receiver
  coordinates keep addressing the same physical cells;
* the optimisation variable stays UNPADDED — gradients through the pad
  accumulate back onto the edge cells (replicate adjoint), so checkpoints
  and optimiser state live on the physical grid and resume is mesh-free.
"""

import numpy as np
import pytest
import torch

from sweep.parallel import MeshTopology, pad_to_mesh, unpad_from_mesh


# ---------------------------------------------------------------------------
# Shapes: high side only, z never padded, never crops
# ---------------------------------------------------------------------------
def test_pads_up_high_side_only():
    a = np.random.rand(94, 401, 401).astype(np.float32)
    ap = pad_to_mesh(a, py=2, px=4)
    assert ap.shape == (94, 402, 404)
    assert np.array_equal(ap[:, :401, :401], a)      # low corner untouched


def test_depth_axis_never_padded():
    ap = pad_to_mesh(np.zeros((93, 400, 400), np.float32), py=2, px=4)
    assert ap.shape[0] == 93


def test_2d_pads_x_only_and_rejects_py():
    a = np.random.rand(50, 101).astype(np.float32)
    assert pad_to_mesh(a, px=4).shape == (50, 104)
    with pytest.raises(ValueError, match="2-D DD requires py=1"):
        pad_to_mesh(a, py=2, px=2)


def test_noop_returns_the_input_object():
    a = np.random.rand(94, 400, 400).astype(np.float32)
    assert pad_to_mesh(a, py=2, px=4) is a
    t = torch.from_numpy(a)
    assert pad_to_mesh(t, py=2, px=4) is t


def test_accepts_a_mesh_topology():
    mesh = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=0)
    assert pad_to_mesh(np.zeros((6, 401, 401), np.float32), mesh).shape \
        == (6, 402, 402)
    with pytest.raises(ValueError, match="not both"):
        pad_to_mesh(np.zeros((6, 401, 401), np.float32), mesh, px=2)


def test_leading_batch_axes_pass_through():
    a = np.random.rand(3, 6, 401, 401).astype(np.float32)   # (nparam, nz, ny, nx)
    ap = pad_to_mesh(a, py=2, px=2)
    assert ap.shape == (3, 6, 402, 402)
    assert np.array_equal(unpad_from_mesh(ap, (6, 401, 401), py=2, px=2), a)


# ---------------------------------------------------------------------------
# Values: edge replication, torch == numpy
# ---------------------------------------------------------------------------
def test_torch_matches_numpy_edge_pad():
    a = np.random.rand(8, 401, 399).astype(np.float32)
    ap = pad_to_mesh(a, py=2, px=4)
    tp = pad_to_mesh(torch.from_numpy(a), py=2, px=4)
    assert np.array_equal(tp.numpy(), ap)
    assert np.array_equal(ap, np.pad(a, [(0, 0), (0, 1), (0, 1)], mode="edge"))


def test_bool_masks_pad_too():
    m = torch.zeros(6, 401, 401, dtype=torch.bool)
    m[:, -1, :] = True
    mp = pad_to_mesh(m, py=2, px=2)
    assert mp.dtype == torch.bool and mp.shape == (6, 402, 402)
    assert mp[:, -1, :].all()                        # replicated edge row


# ---------------------------------------------------------------------------
# Autograd: optimise the UNPADDED tensor, pad inside the closure
# ---------------------------------------------------------------------------
def test_gradient_sums_back_onto_edge_cells():
    t = torch.randn(5, 7, requires_grad=True)        # 2-D grid (nz, nx)
    pad_to_mesh(t, px=4).sum().backward()            # 7 -> 8: dx=1
    g = t.grad
    assert torch.equal(g[:, :-1], torch.ones(5, 6))  # interior: weight 1
    assert torch.equal(g[:, -1], torch.full((5,), 2.0))   # edge col: 1 + dx


def test_gradient_corner_gets_both_factors():
    t = torch.randn(3, 5, 5, requires_grad=True)     # (nz, ny, nx), py=2 px=4
    pad_to_mesh(t, py=2, px=4).sum().backward()      # dy=1, dx=3
    g = t.grad
    assert torch.equal(g[:, :-1, :-1], torch.ones(3, 4, 4))
    assert torch.equal(g[:, -1, :-1], torch.full((3, 4), 2.0))     # 1+dy
    assert torch.equal(g[:, :-1, -1], torch.full((3, 4), 4.0))     # 1+dx
    assert torch.equal(g[:, -1, -1], torch.full((3,), 8.0))        # (1+dy)(1+dx)


# ---------------------------------------------------------------------------
# unpad: loud inverse
# ---------------------------------------------------------------------------
def test_unpad_round_trip():
    a = np.random.rand(6, 401, 401).astype(np.float32)
    ap = pad_to_mesh(a, py=2, px=2)
    assert np.array_equal(unpad_from_mesh(ap, a.shape, py=2, px=2), a)


def test_unpad_rejects_a_stale_shape():
    with pytest.raises(ValueError, match="padded_shape"):
        unpad_from_mesh(np.zeros((6, 536, 536), np.float32),
                        (6, 401, 401), py=2, px=2)


def test_unpad_slices_torch_tensors():
    t = torch.arange(6 * 402 * 402, dtype=torch.float32).reshape(6, 402, 402)
    out = unpad_from_mesh(t, (6, 401, 401), py=2, px=2)
    assert isinstance(out, torch.Tensor) and tuple(out.shape) == (6, 401, 401)
    assert torch.equal(out, t[:, :401, :401])


# ---------------------------------------------------------------------------
# Integration with the tile check it exists to satisfy
# ---------------------------------------------------------------------------
def test_padded_shape_passes_local_extent():
    mesh = MeshTopology(py=2, px=4, shot_groups=1, world_size=8, rank=3)
    with pytest.raises(ValueError, match="pad_to_mesh"):
        mesh.local_extent((94, 401, 401))            # error names the fix
    padded = pad_to_mesh(np.zeros((94, 401, 401), np.float32), mesh)
    local, offsets = mesh.local_extent(padded.shape)
    assert local == (94, 201, 101)
