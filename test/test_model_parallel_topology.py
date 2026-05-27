"""Single-process tests for :mod:`sweep.parallel`.

Covers the pure-arithmetic helpers (``MeshTopology``,
``partition_global_coords``). ``ModelParallelMesh`` itself needs
``torch.distributed.init_process_group`` and is exercised in the multi-GPU
parity tests added in PR-4.
"""

import pytest
import torch

from sweep.parallel import MeshTopology, partition_global_coords


# ---------------------------------------------------------------------------
# Topology validation
# ---------------------------------------------------------------------------
def test_world_size_mismatch():
    with pytest.raises(ValueError, match="world_size"):
        MeshTopology(py=2, px=2, shot_groups=2, world_size=4, rank=0)


def test_rank_out_of_range():
    with pytest.raises(ValueError, match="rank"):
        MeshTopology(py=1, px=2, shot_groups=2, world_size=4, rank=4)


def test_negative_factor():
    with pytest.raises(ValueError, match="must be >= 1"):
        MeshTopology(py=0, px=1, shot_groups=1, world_size=1, rank=0)


def test_singleton_topology():
    t = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    assert t.coord == (0, 0, 0)
    assert t.tile_world_size == 1
    assert t.is_edge("x", "low") and t.is_edge("x", "high")
    assert t.is_edge("y", "low") and t.is_edge("y", "high")
    assert t.neighbour_rank("x", -1) is None
    assert t.neighbour_rank("x", +1) is None
    assert t.neighbour_rank("y", -1) is None


# ---------------------------------------------------------------------------
# Coord decomposition: (P_shot=2, Py=2, Px=2) -> world_size 8
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rank,expected", [
    (0, (0, 0, 0)),
    (1, (0, 0, 1)),
    (2, (0, 1, 0)),
    (3, (0, 1, 1)),
    (4, (1, 0, 0)),
    (5, (1, 0, 1)),
    (6, (1, 1, 0)),
    (7, (1, 1, 1)),
])
def test_coord_decomposition(rank, expected):
    t = MeshTopology(py=2, px=2, shot_groups=2, world_size=8, rank=rank)
    assert t.coord == expected


def test_rank_at_round_trip():
    t = MeshTopology(py=2, px=3, shot_groups=2, world_size=12, rank=0)
    for r in range(12):
        u = MeshTopology(py=2, px=3, shot_groups=2, world_size=12, rank=r)
        assert t.rank_at(*u.coord) == r


# ---------------------------------------------------------------------------
# Neighbour resolution
# ---------------------------------------------------------------------------
def test_neighbour_2d_1x4():
    """py=1, px=4 (pure 2-D model parallel)."""
    for rank in range(4):
        t = MeshTopology(py=1, px=4, shot_groups=1, world_size=4, rank=rank)
        left = t.neighbour_rank("x", -1)
        right = t.neighbour_rank("x", +1)
        assert left == (rank - 1 if rank > 0 else None)
        assert right == (rank + 1 if rank < 3 else None)
        # py=1 -> no y neighbours
        assert t.neighbour_rank("y", -1) is None
        assert t.neighbour_rank("y", +1) is None


def test_neighbour_3d_2x2_corner():
    """Rank 0 in a 2x2 tile grid: has +x and +y neighbours, no -x/-y."""
    t = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=0)
    assert t.coord == (0, 0, 0)
    assert t.neighbour_rank("x", +1) == 1   # (0, 0, 1)
    assert t.neighbour_rank("x", -1) is None
    assert t.neighbour_rank("y", +1) == 2   # (0, 1, 0)
    assert t.neighbour_rank("y", -1) is None


def test_neighbour_axis_validation():
    t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=0)
    with pytest.raises(ValueError, match="axis"):
        t.neighbour_rank("z", +1)
    with pytest.raises(ValueError, match="direction"):
        t.neighbour_rank("x", 0)


# ---------------------------------------------------------------------------
# is_edge
# ---------------------------------------------------------------------------
def test_is_edge_corner_high():
    t = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=3)
    assert t.coord == (0, 1, 1)
    assert not t.is_edge("x", "low")
    assert t.is_edge("x", "high")
    assert not t.is_edge("y", "low")
    assert t.is_edge("y", "high")


def test_is_edge_interior_x():
    """Interior tile in a 1x4 grid: rank 2 -> not on either x edge."""
    t = MeshTopology(py=1, px=4, shot_groups=1, world_size=4, rank=2)
    assert not t.is_edge("x", "low")
    assert not t.is_edge("x", "high")


def test_is_edge_validation():
    t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=0)
    with pytest.raises(ValueError, match="side"):
        t.is_edge("x", "middle")


# ---------------------------------------------------------------------------
# local_extent
# ---------------------------------------------------------------------------
def test_local_extent_2d():
    # Nx=128, px=4 -> nx_loc=32. Rank xi=2 -> ox=64.
    t = MeshTopology(py=1, px=4, shot_groups=1, world_size=4, rank=2)
    shape, off = t.local_extent((96, 128))
    assert shape == (96, 32)
    assert off == (0, 64)


def test_local_extent_3d():
    t = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=3)  # (0, 1, 1)
    shape, off = t.local_extent((32, 64, 128))
    # ny_loc = 64/2 = 32, nx_loc = 128/2 = 64
    assert shape == (32, 32, 64)
    assert off == (0, 32, 64)


def test_local_extent_non_divisible_x():
    t = MeshTopology(py=1, px=3, shot_groups=1, world_size=3, rank=0)
    with pytest.raises(ValueError, match="multiple of px"):
        t.local_extent((48, 100))


def test_local_extent_non_divisible_y():
    t = MeshTopology(py=3, px=1, shot_groups=1, world_size=3, rank=0)
    with pytest.raises(ValueError, match="multiple of py"):
        t.local_extent((32, 100, 96))


def test_local_extent_2d_requires_py1():
    t = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=0)
    with pytest.raises(ValueError, match="py=1"):
        t.local_extent((48, 64))


def test_local_extent_unsupported_ndim():
    t = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    with pytest.raises(ValueError, match="2-D or 3-D"):
        t.local_extent((96,))


# ---------------------------------------------------------------------------
# partition_global_coords
# ---------------------------------------------------------------------------
def test_partition_2d_basic():
    """2-D: Nx=128, px=4. Coord layout per SWEEP is (x, z)."""
    t = MeshTopology(py=1, px=4, shot_groups=1, world_size=4, rank=1)  # owns x in [32, 64)
    coords = torch.tensor([[
        [ 16,  4],   # xi=0  off-tile
        [ 48, 10],   # xi=1  ours
        [ 80, 12],   # xi=2  off-tile
        [112,  8],   # xi=3  off-tile
    ]])
    local, mask = partition_global_coords(coords, t, (96, 128))
    assert mask.tolist() == [[False, True, False, False]]
    # local x = global x - ox  (ox = 32 for xi=1); z unchanged
    assert local[0, 1].tolist() == [48 - 32, 10]
    # off-tile coords are zeroed (defensive: cannot leak as a valid index)
    for off in (0, 2, 3):
        assert local[0, off].tolist() == [0, 0]


def test_partition_3d():
    """3-D py=px=2 grid; coord layout is (x, y, z)."""
    t = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=3)  # (0, 1, 1)
    # Ny=Nx=64 -> ny_loc=nx_loc=32. xi=1 owns x in [32, 64); yi=1 owns y in [32, 64).
    coords = torch.tensor([[
        [10, 10,  5],   # off-tile (x<32, y<32)
        [40, 40,  6],   # on-tile
        [40, 10,  7],   # off-tile (y<32)
        [50, 50,  8],   # on-tile
    ]])
    local, mask = partition_global_coords(coords, t, (32, 64, 64))
    assert mask.tolist() == [[False, True, False, True]]
    assert local[0, 1].tolist() == [40 - 32, 40 - 32, 6]
    assert local[0, 3].tolist() == [50 - 32, 50 - 32, 8]


def test_partition_z_not_split_or_filtered():
    """z (coords[..., -1]) is never split nor used for filtering."""
    t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=0)  # owns x in [0, 50)
    coords = torch.tensor([[[10, 90]]])  # x=10 on-tile; z=90 must be untouched
    local, mask = partition_global_coords(coords, t, (100, 100))
    assert mask.tolist() == [[True]]
    assert local[0, 0].tolist() == [10, 90]


def test_partition_tile_boundary_inclusive_low_exclusive_high():
    """A coord at the tile's low edge is in; at the high edge belongs to the next tile."""
    t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=0)  # owns x in [0, 50)
    coords = torch.tensor([[
        [ 0, 5],   # low edge, in
        [49, 5],   # high inside, in
        [50, 5],   # exactly at boundary, belongs to xi=1
    ]])
    local, mask = partition_global_coords(coords, t, (96, 100))
    assert mask.tolist() == [[True, True, False]]


def test_partition_validates_ndim():
    t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=0)
    coords_2d = torch.zeros(1, 1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="does not match global_shape"):
        partition_global_coords(coords_2d, t, (10, 10, 10))   # 2-D coords vs 3-D shape


def test_partition_rejects_non_3d_input():
    t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=0)
    with pytest.raises(ValueError, match=r"nshots, npts, ndim"):
        partition_global_coords(torch.zeros(2, dtype=torch.long), t, (10, 10))


def test_partition_empty_npts():
    """Empty coord list (npts=0) is a valid no-op."""
    t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=0)
    coords = torch.zeros(3, 0, 2, dtype=torch.long)
    local, mask = partition_global_coords(coords, t, (10, 10))
    assert local.shape == (3, 0, 2)
    assert mask.shape == (3, 0)
