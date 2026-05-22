"""Propagator-level tests for irregular free-surface topography on Acoustic 2-D.

Stage 1 / M2: covers the ``Propagator(topography=...)`` plumbing and the
``Acoustic.func`` integration that swaps ``zero_top_halo_fields`` for
``zero_above_topo`` when topography is set.

Tests
-----
1. flat_zeros_matches_no_topography
   — ``topography=zeros(nx)`` produces a record bit-identical to
     ``topography=None`` (eager backend).
2. air_cells_stay_zero_under_hill
   — Gaussian hill topo; air cells in the returned wavefield snapshots
     must be exactly zero throughout the simulation.
3. topography_without_free_surface_raises
   — ``free_surface=False`` + ``topography`` must raise ValueError.
4. topography_wrong_length_raises
   — length != ``nx_phys`` raises.
5. topography_out_of_range_raises
   — values outside ``[0, nz_phys)`` raise.
6. topography_with_impl_c_raises_not_implemented
   — ``impl='c'`` rejects topography (Stage 1 is eager-only).
7. constant_shift_topography_is_translation_invariant
   — A globally-shifted topo + source + receiver depth should produce a
     receiver record approximately equal to the flat-surface baseline.
     This is a real physics check that the free-surface BC actually moves
     with the per-column surface row, not just a code-equivalence check.
8. vp_gradient_finite_under_topography
   — Forward through the topo path is differentiable: an FWI-style loss
     produces a finite, non-trivial vp gradient, and the gradient at
     "air" cells (above the per-column surface) stays close to zero.
"""

import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

# ---------------------------------------------------------------------------
# Canonical 2-D config (mirrors solver_gradient_mode_suite.py defaults).
# ---------------------------------------------------------------------------

NZ, NX = 48, 56
DH = 10.0
DT = 1.5e-3
NT = 120
ABCN = 30
SO = 4
DOM_FREQ = 10.0
DELAY = 0.06


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=DOM_FREQ)).astype(np.float32))


def _vp_model():
    depth = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
    ramp = 1800.0 + (2400.0 - 1800.0) * depth
    vp = np.broadcast_to(ramp[:, None], (NZ, NX)).astype(np.float32).copy()
    return torch.from_numpy(vp)


def _geometry(src_xz=None):
    """Coords are (x, z). One shot, fan of receivers near the top."""
    sx, sz = (NX // 2, NZ // 4) if src_xz is None else src_xz
    sources = np.array([[sx, sz]], dtype=np.int64)  # (1, 2)
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    rec_z = np.full_like(rec_x, 2)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]  # (1, nrec, 2)
    return torch.from_numpy(sources), torch.from_numpy(receivers)


def _make_prop(topography, *, free_surface=True, impl="eager"):
    eq = Acoustic(spatial_order=SO, device="cpu", backend="torch")
    return PropTorch(
        eq,
        shape=(NZ, NX),
        free_surface=free_surface,
        topography=topography,
        abcn=ABCN,
        dh=DH,
        dt=DT,
        use_ckpt=False,
        impl=impl,
    )


# ---------------------------------------------------------------------------
# Degenerate equivalence
# ---------------------------------------------------------------------------


def test_flat_zeros_matches_no_topography():
    """topography=zeros(nx) must reproduce topography=None bit-for-bit."""
    sources, receivers = _geometry()
    wavelet = _wavelet()
    vp = _vp_model()

    prop_none = _make_prop(topography=None)
    syn_none = prop_none(wavelet, sources, receivers, models=[vp])

    prop_zero = _make_prop(topography=np.zeros(NX, dtype=np.int64))
    syn_zero = prop_zero(wavelet, sources, receivers, models=[vp])

    assert syn_none.shape == syn_zero.shape, (
        f"shape mismatch: none={tuple(syn_none.shape)} zero={tuple(syn_zero.shape)}"
    )
    max_abs_err = (syn_none - syn_zero).abs().max().item()
    assert max_abs_err < 1e-6, (
        f"flat-degenerate diverges: max abs error {max_abs_err}; "
        f"|none|max={syn_none.abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Air-cell zero invariance under non-trivial topography
# ---------------------------------------------------------------------------


def test_air_cells_stay_zero_under_hill():
    """Gaussian-hill topography; air cells (above the per-column surface)
    must remain exactly zero throughout the forward simulation."""
    sources, receivers = _geometry()
    wavelet = _wavelet()
    vp = _vp_model()

    # Build a Gaussian hill, max height 5 rows, centred at nx//2, half-width 10.
    x = np.arange(NX, dtype=np.float32)
    hill = (5.0 * np.exp(-((x - NX / 2) ** 2) / (2.0 * 10.0**2))).round().astype(np.int64)
    # Move the source onto the new surface so it actually radiates downward.
    src_x = NX // 2
    src_z = int(hill[src_x]) + 2  # one cell below the surface
    sources, _ = _geometry(src_xz=(src_x, src_z))

    prop = _make_prop(topography=hill)
    snaps_idx = list(range(0, NT, NT // 6))  # 6 snapshots
    syn, snaps = prop(
        wavelet, sources, receivers, models=[vp],
        return_wavefield=True, snapshot_times=snaps_idx,
    )
    # snaps shape: (n_snapshots, n_wavefields, B, C, nz_padded, nx_padded)
    # We only need the primary pressure field (index 0).
    p_snaps = snaps[:, 0]  # (n_snapshots, B, C, nz_padded, nx_padded)
    assert p_snaps.ndim == 5

    # Build the air-mask in the SAME coordinate system the snapshots use.
    # `crop` is applied inside the eager loop so snaps live in self.shape coords
    # (PML-padded, NOT runtime-halo-padded). x is padded by abcn each side, z
    # by 0 (top) + abcn (bottom) with free_surface=True.
    nz_pad, nx_pad = p_snaps.shape[-2:]
    z_idx = np.arange(nz_pad)[:, None]
    # For physical columns, surface row in PML-padded z coords = topo_phys (no
    # halo shift here because crop strips it). For PML columns, edge-replicate.
    ix_phys = np.clip(np.arange(nx_pad) - ABCN, 0, NX - 1)
    surf_pad = hill[ix_phys][None, :]  # (1, nx_pad)
    air_mask = z_idx < surf_pad  # (nz_pad, nx_pad)

    # Air cells must be exactly zero in every snapshot, every batch/channel.
    mask_t = torch.from_numpy(air_mask)  # (nz_pad, nx_pad), broadcasts to last 2 dims
    if mask_t.any():
        air_max = (p_snaps.abs() * mask_t.to(p_snaps.dtype)).max().item()
    else:
        air_max = 0.0
    assert air_max == 0.0, f"air cells not zero: max |p| = {air_max}"

    # Sanity: there IS some signal somewhere (the receivers actually recorded).
    assert syn.abs().max().item() > 0.0, "forward produced an all-zero record"


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_topography_without_free_surface_raises():
    with pytest.raises(ValueError, match="free_surface=True"):
        _make_prop(topography=np.zeros(NX, dtype=np.int64), free_surface=False)


def test_topography_wrong_length_raises():
    with pytest.raises(ValueError, match="topography length"):
        _make_prop(topography=np.zeros(NX + 3, dtype=np.int64))


def test_topography_out_of_range_raises():
    bad = np.zeros(NX, dtype=np.int64)
    bad[0] = NZ + 5  # past the bottom of the physical domain
    with pytest.raises(ValueError, match="topography values must satisfy"):
        _make_prop(topography=bad)


# ---------------------------------------------------------------------------
# impl='c' guard
# ---------------------------------------------------------------------------


def test_topography_with_impl_c_raises_not_implemented():
    """impl='c' must reject topography (Stage 2 hasn't landed)."""
    from sweep import is_torch_binding_available

    if not is_torch_binding_available():
        pytest.skip("compiled binding not available; impl='c' would fall back to eager")

    with pytest.raises(NotImplementedError, match="impl='python'"):
        _make_prop(topography=np.zeros(NX, dtype=np.int64), impl="c")


# ---------------------------------------------------------------------------
# Physics check: constant-shift topo == flat surface at offset depth
# ---------------------------------------------------------------------------


def test_constant_shift_topography_is_translation_invariant():
    """Comparing two setups that should be physically equivalent:

      A) ``topography=None`` (flat at top), source at z=src_z, receiver at z=rec_z.
      B) ``topography=K*ones(NX)``, source at z=src_z+K, receiver at z=rec_z+K.

    Both place source/receiver at the same depth BELOW the free surface, so
    free-surface reflections, direct waves, and (since the bottom PML is
    abcn=30 rows away in both) bottom absorbing behaviour should match to
    within the small numerical difference introduced by the lateral PML
    being a few rows closer or farther in absolute z. We use a constant vp
    so there is no depth-dependent contrast contaminating the comparison.
    """
    K = 3
    src_x = NX // 2
    src_z = NZ // 4         # baseline source depth-below-surface
    rec_z_base = src_z + 4  # receivers below the source, same column
    # Single co-located receiver line so the comparison is 1-D in time.
    rec_x = np.array([src_x], dtype=np.int64)

    def _run(topo, src_z, rec_z):
        sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
        receivers = torch.from_numpy(
            np.stack([rec_x, np.full_like(rec_x, rec_z)], axis=-1)[None, ...]
        )
        # Constant vp so no depth-dependent contrasts contaminate the test.
        vp = torch.full((NZ, NX), 1800.0)
        prop = _make_prop(topography=topo)
        return prop(_wavelet(), sources, receivers, models=[vp])

    syn_a = _run(topo=None, src_z=src_z, rec_z=rec_z_base)
    syn_b = _run(
        topo=K * np.ones(NX, dtype=np.int64),
        src_z=src_z + K,
        rec_z=rec_z_base + K,
    )
    assert syn_a.shape == syn_b.shape
    peak = syn_a.abs().max().item()
    diff = (syn_a - syn_b).abs().max().item()
    # Tolerance: we require the records to agree to better than 2% of the
    # peak amplitude. PML side-distance is identical between A and B; the
    # only physical difference is a K-row drop in the BOTTOM-PML distance,
    # which is well within abcn=30 for K=3.
    assert diff / max(peak, 1e-30) < 2e-2, (
        f"shifted topo not translation-invariant: rel diff {diff / peak:.3e} "
        f"(peak={peak:.3e}, diff={diff:.3e})"
    )


# ---------------------------------------------------------------------------
# Differentiability through the topo path
# ---------------------------------------------------------------------------


def test_vp_gradient_finite_under_topography():
    """FWI-style loss through the topo path yields a finite, non-trivial
    gradient. The gradient at "air" cells (above the per-column surface) must
    be approximately zero — energy never reaches those cells (the air mask
    zeros them every step and its derivative carries that zero back).
    """
    # Gaussian hill, max height 5 rows, centred at nx//2.
    x = np.arange(NX, dtype=np.float32)
    hill = (5.0 * np.exp(-((x - NX / 2) ** 2) / (2.0 * 10.0**2))).round().astype(np.int64)
    src_x = NX // 2
    src_z = int(hill[src_x]) + 2  # just below the local surface
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    rec_z = (hill[rec_x] + 1).astype(np.int64)  # one row below local surface
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])

    # Constant background + a buried box anomaly drives the residual.
    vp_bg = 1800.0
    vp_true = torch.full((NZ, NX), vp_bg)
    vp_true[NZ // 2 : NZ // 2 + 5, NX // 3 : (2 * NX) // 3] += 180.0
    vp_init_np = np.full((NZ, NX), vp_bg, dtype=np.float32)

    prop = _make_prop(topography=hill)
    with torch.no_grad():
        obs = prop(_wavelet(), sources, receivers, models=[vp_true])
    vp_param = torch.from_numpy(vp_init_np).clone().requires_grad_(True)
    syn = prop(_wavelet(), sources, receivers, models=[vp_param])
    loss = 0.5 * (syn - obs).pow(2).sum()
    loss.backward()
    grad = vp_param.grad
    assert grad is not None, "autograd produced no gradient through the topo path"

    # Finite, non-trivial.
    assert torch.isfinite(grad).all(), "gradient has NaN/Inf"
    assert loss.detach().item() > 0, "loss is identically zero — observed == synthetic"
    grad_peak = grad.abs().max().item()
    assert grad_peak > 0, "gradient is identically zero"

    # Air-region contribution should be a tiny fraction of the peak. The
    # forward zeroes those cells each step; backward through the mask carries
    # the zero back. Some residue from one-step "compute then zero" coupling
    # at the surface is tolerated.
    z_idx = np.arange(NZ)[:, None]
    air_mask = z_idx < hill[None, :]
    if air_mask.any():
        air_peak = grad[torch.from_numpy(air_mask)].abs().max().item()
    else:
        air_peak = 0.0
    assert air_peak / max(grad_peak, 1e-30) < 5e-2, (
        f"gradient leaks into air cells: air_peak/peak = "
        f"{air_peak / grad_peak:.3e} (air_peak={air_peak:.3e}, peak={grad_peak:.3e})"
    )
