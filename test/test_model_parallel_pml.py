"""Tests for :func:`sweep.parallel.build_rank_pml_widths`.

Two layers of tests:

1. **Unit tests** on the integer width vector that the function returns
   (`build_rank_pml_widths` is pure arithmetic — no dist).

2. **Parity tests** that build PML profiles per rank via
   ``set_cpml_profiles_{s,r}`` with the rank-local widths, then verify
   that the rank-r profile equals the rank-r slice of the single-rank
   (global) profile. This is the load-bearing correctness check: it
   ensures the absorbing boundary is identical to single-rank when the
   tiles are conceptually concatenated.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweep.equations.pml import set_cpml_profiles_r, set_cpml_profiles_s
from sweep.parallel import MeshTopology, build_rank_pml_widths


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
ABCN = 20
FD_PAD_2D = [2, 2, 2, 2]
FD_PAD_3D = [2, 2, 2, 2, 2, 2]
DT = 0.001
DH = 10.0
VMAX = 2000.0
FREQ = 25.0


# ---------------------------------------------------------------------------
# 1. Width-vector unit tests
# ---------------------------------------------------------------------------

def test_singleton_2d_gets_pml_on_all_sides():
    t = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    assert build_rank_pml_widths(t, abcn=ABCN, ndim=2) == [
        ABCN, ABCN, ABCN, ABCN,
    ]


def test_singleton_3d_gets_pml_on_all_sides():
    t = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    assert build_rank_pml_widths(t, abcn=ABCN, ndim=3) == [
        ABCN, ABCN, ABCN, ABCN, ABCN, ABCN,
    ]


@pytest.mark.parametrize("rank,expected", [
    (0, [ABCN, ABCN, ABCN, 0]),     # left edge
    (1, [ABCN, ABCN, 0,    0]),     # interior
    (2, [ABCN, ABCN, 0,    0]),     # interior
    (3, [ABCN, ABCN, 0,    ABCN]),  # right edge
])
def test_widths_2d_1x4(rank, expected):
    t = MeshTopology(py=1, px=4, shot_groups=1, world_size=4, rank=rank)
    assert build_rank_pml_widths(t, abcn=ABCN, ndim=2) == expected


@pytest.mark.parametrize("rank,expected", [
    # 2x2 mesh, layout (yi, xi); SWEEP order is [z_low, z_high, y_low, y_high, x_low, x_high]
    (0, [ABCN, ABCN, ABCN, 0,    ABCN, 0]),    # (yi=0, xi=0) — y_low and x_low edges
    (1, [ABCN, ABCN, ABCN, 0,    0,    ABCN]), # (yi=0, xi=1) — y_low and x_high edges
    (2, [ABCN, ABCN, 0,    ABCN, ABCN, 0]),    # (yi=1, xi=0)
    (3, [ABCN, ABCN, 0,    ABCN, 0,    ABCN]), # (yi=1, xi=1)
])
def test_widths_3d_2x2(rank, expected):
    t = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=rank)
    assert build_rank_pml_widths(t, abcn=ABCN, ndim=3) == expected


def test_image_method_suppresses_z_low():
    t = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    w = build_rank_pml_widths(t, abcn=ABCN, ndim=2, image_method_active=True)
    assert w[0] == 0
    assert w[1] == ABCN


def test_ndim_validation():
    t = MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        build_rank_pml_widths(t, abcn=ABCN, ndim=1)


# ---------------------------------------------------------------------------
# 2. Parity tests: per-rank profiles concatenate back to the single-rank
# global profile (within numerical tolerance).
# ---------------------------------------------------------------------------

def _profiles_cpmlr_2d(widths, shape):
    """Call ``set_cpml_profiles_r`` with 2-D defaults; returns the list."""
    return set_cpml_profiles_r(
        pml_width=list(widths),
        accuracy=4,
        fd_pad=list(FD_PAD_2D),
        dt=DT,
        grid_spacing=[DH, DH],
        max_vel=VMAX,
        dtype=np.float32,
        pml_freq=FREQ,
        shape=tuple(shape),
    )


def _profiles_cpmls_2d(widths, shape):
    return set_cpml_profiles_s(
        pml_width=list(widths),
        accuracy=4,
        fd_pad=list(FD_PAD_2D),
        dt=DT,
        grid_spacing=[DH, DH],
        max_vel=VMAX,
        dtype=np.float32,
        pml_freq=FREQ,
        shape=tuple(shape),
    )


def _profiles_cpmlr_3d(widths, shape):
    return set_cpml_profiles_r(
        pml_width=list(widths),
        accuracy=4,
        fd_pad=list(FD_PAD_3D),
        dt=DT,
        grid_spacing=[DH, DH, DH],
        max_vel=VMAX,
        dtype=np.float32,
        pml_freq=FREQ,
        shape=tuple(shape),
    )


def _assert_profile_parity(local, global_slice, axis_dim, j, rank, kind, label):
    """Compare local profile to its global slice.

    The `db` profile (every third entry per dim, index ``2 + 3*dim``) is the
    spatial derivative of `b` computed via ``np.gradient`` inside the PML
    helper. At the tile boundary the local gradient uses a one-sided stencil
    while the global uses central differences, so the boundary cell will not
    match — that's a property of the helper, not of model parallel.
    We exclude the tile-boundary cells from the comparison for db profiles.
    """
    if kind == "db":
        # Build a slice that drops one cell from each end along axis_dim.
        idx = [slice(None)] * local.ndim
        idx[axis_dim] = slice(1, -1)
        local_inner = local[tuple(idx)]
        global_inner = global_slice[tuple(idx)]
        assert np.allclose(local_inner, global_inner), (
            f"rank {rank}: {label} db profile {j} interior mismatch"
        )
    else:
        assert np.allclose(local, global_slice), (
            f"rank {rank}: {label} {kind} profile {j} mismatch"
        )


@pytest.mark.parametrize("px", [2, 4])
def test_parity_2d_cpmlr(px):
    """Per-rank x profiles, concatenated, equal the single-rank global profile.
    z profiles are identical on every rank (z is not split)."""
    Nz = 64
    Nx_loc = 24
    Nx_global = Nx_loc * px

    global_widths = [ABCN, ABCN, ABCN, ABCN]
    global_profiles = _profiles_cpmlr_2d(global_widths, (Nz, Nx_global))
    # _r returns 3 profiles per dim: [a_z, b_z, db_z, a_x, b_x, db_x]

    for rank in range(px):
        t = MeshTopology(py=1, px=px, shot_groups=1, world_size=px, rank=rank)
        widths = build_rank_pml_widths(t, abcn=ABCN, ndim=2)
        local = _profiles_cpmlr_2d(widths, (Nz, Nx_loc))

        # z profiles (0..2): identical across ranks; z axis dim is -2.
        for j, kind in ((0, "a"), (1, "b"), (2, "db")):
            _assert_profile_parity(
                local[j], global_profiles[j], axis_dim=-2,
                j=j, rank=rank, kind=kind, label="z",
            )

        # x profiles (3..5): rank-r corresponds to x_slice; x axis dim is -1.
        x_slice = slice(rank * Nx_loc, (rank + 1) * Nx_loc)
        for j, kind in ((3, "a"), (4, "b"), (5, "db")):
            _assert_profile_parity(
                local[j], global_profiles[j][..., x_slice], axis_dim=-1,
                j=j, rank=rank, kind=kind, label="x",
            )


def test_parity_2d_cpmls_1x2():
    """Same parity check but for the staggered PML (cpmls)."""
    Nz = 48
    Nx_loc = 32
    Nx_global = Nx_loc * 2

    global_profiles = _profiles_cpmls_2d([ABCN, ABCN, ABCN, ABCN],
                                         (Nz, Nx_global))
    # _s returns 4 profiles per dim: [a_z, b_z, ah_z, bh_z, a_x, b_x, ah_x, bh_x]

    for rank in range(2):
        t = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=rank)
        widths = build_rank_pml_widths(t, abcn=ABCN, ndim=2)
        local = _profiles_cpmls_2d(widths, (Nz, Nx_loc))

        # z profiles (0..3)
        for j in range(4):
            assert np.allclose(local[j], global_profiles[j])

        # x profiles (4..7)
        x_slice = slice(rank * Nx_loc, (rank + 1) * Nx_loc)
        for j in range(4, 8):
            assert np.allclose(local[j], global_profiles[j][..., x_slice]), (
                f"rank {rank}: profile {j} mismatch"
            )


def test_parity_3d_cpmlr_2x2():
    """3-D 2x2 mesh: rank profiles cover their (y_slice, x_slice) of global."""
    Nz = 16
    Ny_loc = 12
    Nx_loc = 12
    Ny_global = Ny_loc * 2
    Nx_global = Nx_loc * 2

    global_profiles = _profiles_cpmlr_3d(
        [ABCN, ABCN, ABCN, ABCN, ABCN, ABCN],
        (Nz, Ny_global, Nx_global),
    )
    # _r returns 3 per dim: [a_z, b_z, db_z, a_y, b_y, db_y, a_x, b_x, db_x]
    # Each broadcastable to (1, Nz, Ny, Nx); only its own axis is non-1.

    for rank in range(4):
        t = MeshTopology(py=2, px=2, shot_groups=1, world_size=4, rank=rank)
        widths = build_rank_pml_widths(t, abcn=ABCN, ndim=3)
        local = _profiles_cpmlr_3d(widths, (Nz, Ny_loc, Nx_loc))

        yi = (rank % 4) // 2
        xi = rank % 2
        y_slice = slice(yi * Ny_loc, (yi + 1) * Ny_loc)
        x_slice = slice(xi * Nx_loc, (xi + 1) * Nx_loc)

        # z profiles (0..2): same on every rank; z axis dim is -3 in 3-D.
        for j, kind in ((0, "a"), (1, "b"), (2, "db")):
            _assert_profile_parity(
                local[j], global_profiles[j], axis_dim=-3,
                j=j, rank=rank, kind=kind, label="z",
            )

        # y profiles (3..5): rank corresponds to y_slice along axis -2.
        for j, kind in ((3, "a"), (4, "b"), (5, "db")):
            _assert_profile_parity(
                local[j], global_profiles[j][..., y_slice, :], axis_dim=-2,
                j=j, rank=rank, kind=kind, label="y",
            )

        # x profiles (6..8): rank corresponds to x_slice along axis -1.
        for j, kind in ((6, "a"), (7, "b"), (8, "db")):
            _assert_profile_parity(
                local[j], global_profiles[j][..., x_slice], axis_dim=-1,
                j=j, rank=rank, kind=kind, label="x",
            )


# ---------------------------------------------------------------------------
# Integration with PropBase.init_abc — only smoke-tests that the kwarg flows
# through; full propagator integration is exercised in PR-4.
# ---------------------------------------------------------------------------

def test_propbase_accepts_model_parallel_kwarg():
    """PropBase.__init__ stores model_parallel and init_abc routes through it.

    Just verifies plumbing; the actual PML profile build is covered above.
    """
    import torch
    from sweep.equations import Acoustic
    from sweep.propagator.torch import PropTorch
    from sweep.parallel import ModelParallelMesh

    eq = Acoustic(spatial_order=4, device='cpu', backend='torch')

    # We can't call ModelParallelMesh.__init__ without world_size+rank because
    # dist isn't initialised. Pass them explicitly (this exercises the
    # property-pass-through path without needing a real ProcessGroup).
    mesh = ModelParallelMesh(grid=(1, 2), shot_groups=1,
                             world_size=2, rank=0)
    prop = PropTorch(
        eq,
        shape=(48, 56),
        dh=10.0, dt=0.0015,
        dev=torch.device('cpu'),
        model_parallel=mesh,
    )
    assert prop.model_parallel is mesh
    prop.init_abc(max_vel=2000.0, pml_freq=10.0)
    # Cache key should be set; rank_coord should be in it.
    assert prop._abc_cache_key is not None
    assert prop._abc_cache_key[-1] == mesh.coord
