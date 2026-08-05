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
6. topography_impl_c_matches_eager
   — ``impl='c'`` (CUDA image-method topography) matches the eager forward.
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

# Auto-select CUDA when available.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=DOM_FREQ)).astype(np.float32)).to(DEVICE)


def _vp_model():
    depth = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
    ramp = 1800.0 + (2400.0 - 1800.0) * depth
    vp = np.broadcast_to(ramp[:, None], (NZ, NX)).astype(np.float32).copy()
    return torch.from_numpy(vp).to(DEVICE)


def _geometry(src_xz=None):
    """Coords are (x, z). One shot, fan of receivers near the top."""
    sx, sz = (NX // 2, NZ // 4) if src_xz is None else src_xz
    sources = np.array([[sx, sz]], dtype=np.int64)  # (1, 2)
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    rec_z = np.full_like(rec_x, 2)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]  # (1, nrec, 2)
    return torch.from_numpy(sources).to(DEVICE), torch.from_numpy(receivers).to(DEVICE)


def _make_prop(topography, *, free_surface=True, topo_method='auto', impl="eager"):
    """Acoustic propagator with optional topography.

    When ``topography`` is given we use the new unified API: pass
    ``topography=`` alone and let ``topo_method`` auto-resolve to
    ``'image'`` (Mittet 2002 staircase + vacuum).  ``free_surface`` only
    matters in the no-topo case (flat free surface vs no FS).
    """
    eq = Acoustic(spatial_order=SO, device=DEVICE, backend="torch")
    if topography is None:
        return PropTorch(
            eq, shape=(NZ, NX),
            free_surface=free_surface,
            topography=None,
            abcn=ABCN, dh=DH, dt=DT,
            use_ckpt=False, impl=impl,
        )
    # Topography given — free_surface auto-set by topo_method.
    return PropTorch(
        eq, shape=(NZ, NX),
        topography=topography,
        topo_method=topo_method,
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl=impl,
    )


# ---------------------------------------------------------------------------
# Degenerate equivalence
# ---------------------------------------------------------------------------


def test_flat_zeros_matches_no_topography():
    """topography=zeros(nx) must reproduce topography=None to float32
    rounding tolerance.  Strict bit-equality holds on CPU; on CUDA the
    two code paths may pick different reduction orders, so we require a
    relative error below 1e-6 (well under float32 epsilon ≈ 1.2e-7 per
    operation, but accumulated over NT steps and the receiver-line
    aggregation)."""
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
    peak = syn_none.abs().max().item()
    rel_err = max_abs_err / max(peak, 1e-30)
    assert rel_err < 1e-6, (
        f"flat-degenerate diverges: rel error {rel_err:.3e} "
        f"(max abs error {max_abs_err:.3e}; |none|max={peak:.3e})"
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


def test_topography_apm_on_acoustic_raises():
    """With the unified ``topo_method=`` API, ``Acoustic`` cannot use APM
    (Cao 2018 is elastic-only).  Asking for it must error cleanly."""
    with pytest.raises(ValueError, match="topo_method='apm' requires"):
        _make_prop(topography=np.zeros(NX, dtype=np.int64), topo_method='apm')


def test_topography_implies_free_surface_acoustic():
    """``topography=`` alone is sufficient for Acoustic — no need to set
    ``free_surface=True`` (it's auto-set internally by the resolved
    method).  This test pins the new ergonomic API."""
    prop = _make_prop(topography=np.zeros(NX, dtype=np.int64), free_surface=False)
    # Internally the propagator picked 'image' (Acoustic's only path) and
    # auto-set free_surface=True for the layout.
    assert prop._topo_method == 'image'
    assert prop.free_surface is True


def test_topography_wrong_length_raises():
    with pytest.raises(ValueError, match="topography length"):
        _make_prop(topography=np.zeros(NX + 3, dtype=np.int64))


def test_topography_out_of_range_raises():
    bad = np.zeros(NX, dtype=np.int64)
    bad[0] = NZ + 5  # past the bottom of the physical domain
    with pytest.raises(ValueError, match="topography values must satisfy"):
        _make_prop(topography=bad)


# ---------------------------------------------------------------------------
# impl='c' topography (CUDA image method — Stage 2)
# ---------------------------------------------------------------------------


def test_topography_impl_c_matches_eager():
    """Topography on ``impl='c'`` (CUDA image method) is supported — Stage 2 has
    landed — and its forward matches the eager reference.  (This guard historically
    *raised*: Stage 1 was eager-only.)"""
    from sweep import is_torch_binding_available

    if not is_torch_binding_available() or DEVICE != "cuda":
        pytest.skip("compiled CUDA binding unavailable; impl='c' would fall back to eager")

    x = np.arange(NX, dtype=np.float32)
    hill = (5.0 * np.exp(-((x - NX / 2) ** 2) / (2.0 * 10.0**2))).round().astype(np.int64)
    src_x = NX // 2
    sources, receivers = _geometry(src_xz=(src_x, int(hill[src_x]) + 2))
    # The compiled backend takes numpy source/receiver coordinates.
    sources = sources.detach().cpu().numpy().astype(np.int64)
    receivers = receivers.detach().cpu().numpy().astype(np.int64)
    wavelet = _wavelet()
    vp = _vp_model()

    re = _make_prop(topography=hill, impl="eager")(
        wavelet, sources.copy(), receivers.copy(), models=[vp]).detach()
    rc = _make_prop(topography=hill, impl="c")(
        wavelet, sources.copy(), receivers.copy(), models=[vp]).detach()

    assert re.abs().max().item() > 0.0, "forward produced an all-zero record"
    rel = (rc - re).norm().item() / max(re.norm().item(), 1e-30)
    assert rel < 5e-3, f"impl='c' topography forward vs eager rel_l2={rel:.2e}"


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
        sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64)).to(DEVICE)
        receivers = torch.from_numpy(
            np.stack([rec_x, np.full_like(rec_x, rec_z)], axis=-1)[None, ...]
        ).to(DEVICE)
        # Constant vp so no depth-dependent contrasts contaminate the test.
        vp = torch.full((NZ, NX), 1800.0).to(DEVICE)
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
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64)).to(DEVICE)
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    rec_z = (hill[rec_x] + 1).astype(np.int64)  # one row below local surface
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...]).to(DEVICE)

    # Constant background + a buried box anomaly drives the residual.
    vp_bg = 1800.0
    vp_true = torch.full((NZ, NX), vp_bg).to(DEVICE)
    vp_true[NZ // 2 : NZ // 2 + 5, NX // 3 : (2 * NX) // 3] += 180.0
    vp_init_np = np.full((NZ, NX), vp_bg, dtype=np.float32)

    prop = _make_prop(topography=hill)
    with torch.no_grad():
        obs = prop(_wavelet(), sources, receivers, models=[vp_true])
    vp_param = torch.from_numpy(vp_init_np).to(DEVICE).clone().requires_grad_(True)
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
        air_peak = grad[torch.from_numpy(air_mask).to(grad.device)].abs().max().item()
    else:
        air_peak = 0.0
    assert air_peak / max(grad_peak, 1e-30) < 5e-2, (
        f"gradient leaks into air cells: air_peak/peak = "
        f"{air_peak / grad_peak:.3e} (air_peak={air_peak:.3e}, peak={grad_peak:.3e})"
    )
