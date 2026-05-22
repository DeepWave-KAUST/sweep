"""Propagator-level tests for irregular topography on Elastic 2-D (Stage 2).

Elastic FS already used the Robertsson-style image-method mirror on the flat
path (sxz/szz/vz odd-parity, vx even-parity). Stage 2 generalises that mirror
to per-column ``iz_surf(ix)`` via the ``_topo`` helpers, and replaces the
constant-row ``zero_top_row`` with ``zero_at_topo`` to enforce
σ_zz = σ_xz = 0 on the irregular surface.

Tests
-----
1. flat_zeros_matches_no_topography_elastic
   — topography=zeros(nx) reproduces topography=None bit-exact.
2. surface_stresses_clean_under_hill
   — σ_zz and σ_xz at the per-column surface row stay exactly zero.
3. constant_shift_topography_is_translation_invariant_elastic
   — globally shifted topo + src/rec → record matches flat baseline.
4. vp_gradient_finite_under_topography_elastic
   — autograd back-prop produces a finite, non-trivial gradient.
"""

import numpy as np
import pytest
import torch

from sweep.equations import Elastic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Canonical config (mirrors the acoustic test + solver_gradient_mode_suite).
# vp/vs/rho chosen so CFL is safe (max c · dt / dh ≈ 0.36) and ppw of vs is
# comfortable (~10 cells per minimum wavelength).
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


def _models():
    depth = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
    vp = (1800.0 + (2400.0 - 1800.0) * depth)[:, None]
    vp = np.broadcast_to(vp, (NZ, NX)).astype(np.float32).copy()
    vs = (vp / 1.73).astype(np.float32)
    rho = np.broadcast_to(
        (1000.0 + 200.0 * depth)[:, None], (NZ, NX)
    ).astype(np.float32).copy()
    return (
        torch.from_numpy(vp),
        torch.from_numpy(vs),
        torch.from_numpy(rho),
    )


def _geometry(src_xz=None):
    sx, sz = (NX // 2, NZ // 4) if src_xz is None else src_xz
    sources = np.array([[sx, sz]], dtype=np.int64)
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    rec_z = np.full_like(rec_x, 2)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]
    return torch.from_numpy(sources), torch.from_numpy(receivers)


def _make_prop(topography, *, free_surface=True, impl="eager"):
    eq = Elastic(spatial_order=SO, device="cpu", backend="torch")
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
# 1. Flat-zero degenerate equivalence
# ---------------------------------------------------------------------------


def test_flat_zeros_matches_no_topography_elastic():
    sources, receivers = _geometry()
    wavelet = _wavelet()
    models = list(_models())

    prop_none = _make_prop(topography=None)
    syn_none = prop_none(wavelet, sources, receivers, models=models)

    prop_zero = _make_prop(topography=np.zeros(NX, dtype=np.int64))
    syn_zero = prop_zero(wavelet, sources, receivers, models=models)

    assert syn_none.shape == syn_zero.shape, (
        f"shape mismatch: none={tuple(syn_none.shape)} zero={tuple(syn_zero.shape)}"
    )
    max_abs_err = (syn_none - syn_zero).abs().max().item()
    assert max_abs_err < 1e-5, (
        f"flat-degenerate diverges: max abs error {max_abs_err}; "
        f"|none|max={syn_none.abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# 2. Surface stresses σ_zz, σ_xz zero on the hill
# ---------------------------------------------------------------------------


def test_surface_stresses_clean_under_hill():
    """σ_zz and σ_xz at the per-column surface row must be exactly zero
    throughout the simulation (Dirichlet BC enforcement at the image
    method's surface)."""
    wavelet = _wavelet()
    models = list(_models())

    # Gaussian hill, max height 5 rows, half-width 10.
    x = np.arange(NX, dtype=np.float32)
    hill = (5.0 * np.exp(-((x - NX / 2) ** 2) / (2.0 * 10.0**2))).round().astype(np.int64)
    src_x = NX // 2
    src_z = int(hill[src_x]) + 2
    sources, receivers = _geometry(src_xz=(src_x, src_z))

    prop = _make_prop(topography=hill)
    snaps_idx = list(range(0, NT, NT // 6))
    syn, snaps = prop(
        wavelet, sources, receivers, models=models,
        return_wavefield=True, snapshot_times=snaps_idx,
    )
    # snaps shape: (n_snap, n_wavefield=15, B, C, nz_pad, nx_pad)
    # Field indices for Elastic: 0=vx, 1=vz, 2=sxx, 3=szz, 4=sxz, 5-14=PML mem.
    szz_snaps = snaps[:, 3, 0, 0]   # (n_snap, nz_pad, nx_pad)
    sxz_snaps = snaps[:, 4, 0, 0]
    nz_pad, nx_pad = szz_snaps.shape[-2:]

    # crop is applied for snapshots → these live in self.shape coords
    # (x has abcn each side; z has 0 top + abcn bottom for free_surface=True).
    # So in this coord system, surface row per physical column = hill[ix_phys].
    # For x-PML columns (ix_pad < abcn or ix_pad >= abcn + NX) the topography
    # is edge-replicated; the per-column surface there is hill[0] or hill[-1].
    ix_phys = np.clip(np.arange(nx_pad) - ABCN, 0, NX - 1)
    surf_pad = hill[ix_phys]  # (nx_pad,)

    # Gather the value at the surface row for each column, every snapshot.
    surf_idx = torch.from_numpy(surf_pad).long()  # (nx_pad,)
    nx_pad_idx = torch.arange(nx_pad)
    # advanced indexing: szz_snaps[snap, surf_idx[ix], ix]
    szz_surface = szz_snaps[:, surf_idx, nx_pad_idx]   # (n_snap, nx_pad)
    sxz_surface = sxz_snaps[:, surf_idx, nx_pad_idx]

    szz_max = szz_surface.abs().max().item()
    sxz_max = sxz_surface.abs().max().item()
    assert szz_max == 0.0, f"σ_zz not zero on surface: max |σ_zz| = {szz_max}"
    assert sxz_max == 0.0, f"σ_xz not zero on surface: max |σ_xz| = {sxz_max}"

    # Sanity: the forward actually produced signal (receivers recorded vx, vz).
    assert syn.abs().max().item() > 0.0, "forward produced an all-zero record"


# ---------------------------------------------------------------------------
# 3. Constant-shift translation invariance
# ---------------------------------------------------------------------------


def test_constant_shift_topography_is_translation_invariant_elastic():
    """A globally shifted topo + matching src/rec depth shift should produce
    receiver records approximately equal to the flat-surface baseline."""
    K = 3
    src_x = NX // 2
    src_z = NZ // 4
    rec_z_base = src_z + 4
    rec_x = np.array([src_x], dtype=np.int64)

    def _run(topo, src_z, rec_z):
        sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
        receivers = torch.from_numpy(
            np.stack([rec_x, np.full_like(rec_x, rec_z)], axis=-1)[None, ...]
        )
        # Constant vp/vs/rho so no depth contrasts contaminate the comparison.
        vp = torch.full((NZ, NX), 1800.0)
        vs = torch.full((NZ, NX), 1040.0)
        rho = torch.full((NZ, NX), 1000.0)
        prop = _make_prop(topography=topo)
        return prop(_wavelet(), sources, receivers, models=[vp, vs, rho])

    syn_a = _run(topo=None, src_z=src_z, rec_z=rec_z_base)
    syn_b = _run(
        topo=K * np.ones(NX, dtype=np.int64),
        src_z=src_z + K,
        rec_z=rec_z_base + K,
    )
    assert syn_a.shape == syn_b.shape
    peak = syn_a.abs().max().item()
    diff = (syn_a - syn_b).abs().max().item()
    # Looser tolerance than acoustic (5e-2 vs 2e-2): elastic has both P and S
    # modes; the K-row change in bottom-PML distance affects both, and the
    # smaller S-wavelength is more sensitive to PML thickness.
    assert diff / max(peak, 1e-30) < 5e-2, (
        f"shifted topo not translation-invariant: rel diff {diff / peak:.3e} "
        f"(peak={peak:.3e}, diff={diff:.3e})"
    )


# ---------------------------------------------------------------------------
# 4. Differentiability through the topo path
# ---------------------------------------------------------------------------


def test_vp_gradient_finite_under_topography_elastic():
    """FWI-style loss through the topo path yields a finite, non-trivial
    gradient w.r.t. vp."""
    x = np.arange(NX, dtype=np.float32)
    hill = (5.0 * np.exp(-((x - NX / 2) ** 2) / (2.0 * 10.0**2))).round().astype(np.int64)
    src_x = NX // 2
    src_z = int(hill[src_x]) + 2
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    rec_x = np.arange(2, NX - 2, 6, dtype=np.int64)
    rec_z = (hill[rec_x] + 1).astype(np.int64)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])

    # Constant background; a buried P-wave anomaly drives the residual.
    vp_bg, vs_bg, rho_bg = 1800.0, 1040.0, 1000.0
    vp_true = torch.full((NZ, NX), vp_bg)
    vp_true[NZ // 2 : NZ // 2 + 5, NX // 3 : (2 * NX) // 3] += 180.0
    vs_const = torch.full((NZ, NX), vs_bg)
    rho_const = torch.full((NZ, NX), rho_bg)
    vp_init = torch.full((NZ, NX), vp_bg)

    prop = _make_prop(topography=hill)
    with torch.no_grad():
        obs = prop(_wavelet(), sources, receivers, models=[vp_true, vs_const, rho_const])
    vp_param = vp_init.clone().requires_grad_(True)
    syn = prop(_wavelet(), sources, receivers, models=[vp_param, vs_const, rho_const])
    loss = 0.5 * (syn - obs).pow(2).sum()
    loss.backward()
    grad = vp_param.grad
    assert grad is not None, "autograd produced no gradient through elastic topo path"
    assert torch.isfinite(grad).all(), "gradient has NaN/Inf"
    assert loss.detach().item() > 0, "loss is identically zero"
    assert grad.abs().max().item() > 0, "gradient is identically zero"
