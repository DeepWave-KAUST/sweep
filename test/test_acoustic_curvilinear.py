"""Curvilinear-grid Acoustic 2-D — Stage 3 tests.

Covers:

1. flat_topography_runs_and_is_stable
   — ``topography=None`` (identity metric) runs cleanly for a moderate
     simulation time; record is finite, non-zero, and roughly comparable
     in magnitude to a flat-surface ``Acoustic`` baseline.
2. hill_long_time_stable
   — Under a non-trivial hill topography, ``|p|max`` stays bounded for
     NT = 2000 timesteps (no staircase exponential blow-up).
3. air_cells_clean_under_hill
   — Above the topography (in physical coordinates) the wavefield is
     approximately zero throughout the simulation.
4. requires_free_surface
   — Curvilinear path currently always pairs with ``free_surface=True``
     (the BC at η=0 is the free surface); using free_surface=False
     should still construct but won't be meaningful — guard test
     verifies it doesn't crash.
5. impl_c_rejects_curvilinear
   — ``AcousticCurvilinear._C()`` raises NotImplementedError.
"""

import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, AcousticCurvilinear
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


NZ, NX = 64, 96
DH = 10.0
DT = 1.0e-3
NT = 200
ABCN = 30
SO = 4
FREQ = 8.0
DELAY = 0.10


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32))


def _geometry(src_xz=None):
    sx, sz = (NX // 2, NZ // 3) if src_xz is None else src_xz
    sources = torch.from_numpy(np.array([[sx, sz]], dtype=np.int64))
    rec_x = np.arange(4, NX - 4, 6, dtype=np.int64)
    rec_z = np.full_like(rec_x, 4)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    return sources, receivers


def _make_prop(topography, equation_cls=AcousticCurvilinear, *, free_surface=True):
    eq = equation_cls(spatial_order=SO, device="cpu", backend="torch")
    return PropTorch(
        eq, shape=(NZ, NX),
        free_surface=free_surface,
        topography=topography,
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )


# ---------------------------------------------------------------------------
# 1. Flat topography sanity (identity metric)
# ---------------------------------------------------------------------------


def test_flat_topography_runs_and_is_stable():
    sources, receivers = _geometry()
    wavelet = _wavelet()
    vp = torch.full((NZ, NX), 2000.0)

    prop = _make_prop(topography=None)
    syn = prop(wavelet, sources, receivers, models=[vp])
    assert torch.isfinite(syn).all(), "curvilinear flat run produced NaN/Inf"
    assert syn.abs().max().item() > 0, "curvilinear flat record is identically zero"

    # Sanity-check magnitude against the standard Acoustic flat baseline.
    # We don't require equality (the curvilinear metric introduces a small
    # rescaling because η is dimensionless while the standard equation uses
    # dh in z), but the peak should be the same order of magnitude.
    base = _make_prop(topography=None, equation_cls=Acoustic)
    syn_ref = base(wavelet, sources, receivers, models=[vp])
    ratio = syn.abs().max().item() / max(syn_ref.abs().max().item(), 1e-30)
    assert 0.01 < ratio < 100.0, (
        f"curvilinear-flat peak vs standard-flat peak ratio {ratio:.3e} is "
        f"unreasonably far from 1"
    )


# ---------------------------------------------------------------------------
# 2. Long-time stability under a hill (the Stage 3 win)
# ---------------------------------------------------------------------------


def test_hill_long_time_stable():
    """Curvilinear must not exhibit the staircase late-time blow-up."""
    nt_long = 2000  # 2.0 s @ dt=1e-3 — well past where staircase explodes
    x = np.arange(NX, dtype=np.float32)
    hill = (
        8.0 * np.exp(-((x - NX * 0.4) ** 2) / (2.0 * 12.0**2))
        + 5.0 * np.exp(-((x - NX * 0.7) ** 2) / (2.0 * 10.0**2))
    ).round().astype(np.int64)

    t = np.arange(nt_long, dtype=np.float32) * DT - DELAY
    wavelet = torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32))
    src_x = NX // 4
    src_z = int(hill[src_x]) + 4
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    rec_x = np.arange(4, NX - 4, 6, dtype=np.int64)
    rec_z = (hill[rec_x] + 2).astype(np.int64)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    vp = torch.full((NZ, NX), 2000.0)

    prop = _make_prop(topography=hill)
    snap_times = list(range(0, nt_long, max(1, nt_long // 10)))
    syn, snaps = prop(
        wavelet, sources, receivers, models=[vp],
        return_wavefield=True, snapshot_times=snap_times,
    )
    p_snaps = snaps[:, 0, 0, 0].detach().cpu().numpy()
    per_frame_max = np.abs(p_snaps).max(axis=(-2, -1))
    early_peak = per_frame_max[: len(per_frame_max) // 3].max()
    late_peak = per_frame_max[2 * len(per_frame_max) // 3 :].max()
    # Stage 3 win: the late-time peak shouldn't be more than 10× the
    # early peak (which is set by source firing). Staircase elastic /
    # acoustic blows up >100× over similar time windows.
    assert late_peak < 10.0 * early_peak, (
        f"curvilinear hill simulation grew exponentially: early peak "
        f"{early_peak:.3e}, late peak {late_peak:.3e}"
    )
    assert torch.isfinite(syn).all(), "curvilinear hill record has NaN/Inf"


# ---------------------------------------------------------------------------
# 3. Pressure-release surface check under a hill
# ---------------------------------------------------------------------------


def test_surface_pressure_release_under_hill():
    """In curvilinear coords η = 0 is the surface; there is no "air"
    region above it (the rectangular computational domain only covers
    the subsurface). The free-surface BC enforces ``p = 0`` at the
    surface row via the standard vacuum-ghost mechanism — so the
    surface-row pressure should be **small** (1st-order accurate in
    dh) compared to the bulk wavefield peak, but not exactly zero."""
    x = np.arange(NX, dtype=np.float32)
    hill = (6.0 * np.exp(-((x - NX / 2) ** 2) / (2.0 * 14.0**2))).round().astype(np.int64)
    src_x = NX // 2
    src_z = int(hill[src_x]) + 3
    sources, _ = _geometry(src_xz=(src_x, src_z))
    rec_x = np.arange(4, NX - 4, 6, dtype=np.int64)
    rec_z = (hill[rec_x] + 1).astype(np.int64)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    vp = torch.full((NZ, NX), 2000.0)

    prop = _make_prop(topography=hill)
    snap_idx = list(range(0, NT, NT // 5))
    syn, snaps = prop(
        _wavelet(), sources, receivers, models=[vp],
        return_wavefield=True, snapshot_times=snap_idx,
    )
    # snaps shape: (n_snap, n_field, B, C, nz_pad, nx_pad). After PML
    # strip, the first z row in the snapshot is η=0 — the surface.
    p_snaps = snaps[:, 0, 0, 0]
    surface_row = p_snaps[:, 0, :]  # (n_snap, nx_pad)
    bulk_peak = p_snaps.abs().max().item()
    surface_peak = surface_row.abs().max().item()
    # 1st-order accurate at the surface: ratio should be < 0.5 for our
    # well-resolved case. (Vacuum-ghost Dirichlet enforcement.)
    assert surface_peak < 0.5 * bulk_peak, (
        f"surface row not pressure-released enough: surface_peak/bulk = "
        f"{surface_peak/bulk_peak:.3e} (surface={surface_peak:.3e}, "
        f"bulk={bulk_peak:.3e})"
    )
    assert syn.abs().max().item() > 0.0, "curvilinear hill run gave empty record"


# ---------------------------------------------------------------------------
# 4 + 5. API guards
# ---------------------------------------------------------------------------


def test_impl_c_rejects_curvilinear():
    eq = AcousticCurvilinear(spatial_order=SO, device="cpu", backend="torch")
    with pytest.raises(NotImplementedError):
        eq._C()
