"""Acoustic 2-D forward modeling with **curvilinear** free-surface
topography.

This is the Stage-3 counterpart of ``acoustic2d_hill_demo.py``: same
undulating surface, but the PDE is solved on a boundary-fitted (ξ, η)
grid via :class:`AcousticCurvilinear`. The free surface lies exactly on
the flat computational row η = 0, so there is **no staircase corner
diffraction** and the wavefield can be propagated for arbitrarily long
times.

Outputs (in ``outputs/``):
    acoustic2d_curv_model.png      vp model with topo overlay
    acoustic2d_curv_record.png     shot-gather seismic record
    acoustic2d_curv_wavefield.gif  pressure snapshots in physical coords
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import AcousticCurvilinear
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Model / acquisition parameters
# ---------------------------------------------------------------------------

NZ, NX = 160, 320
DH = 10.0
DT = 1.0e-3
NT = 1500
ABCN = 40
SO = 4
FREQ = 8.0
DELAY = 0.12

SNAP_STRIDE = 20
SNAP_FRAMES = NT // SNAP_STRIDE

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_topography(nx: int) -> np.ndarray:
    """Two ridges + a small sin ripple — same as the staircase demo for
    direct visual comparison."""
    x = np.arange(nx, dtype=np.float32)
    peak_a = 22.0 * np.exp(-((x - nx * 0.30) ** 2) / (2.0 * 22.0**2))
    peak_b = 16.0 * np.exp(-((x - nx * 0.68) ** 2) / (2.0 * 30.0**2))
    ripple = 3.0 * np.sin(2.0 * np.pi * x / 60.0)
    elevation = peak_a + peak_b + ripple
    surface_row = np.round(28.0 - elevation).astype(np.int64)
    return np.clip(surface_row, 0, nx)


def build_velocity(nz: int, nx: int, topo: np.ndarray) -> np.ndarray:
    """Layered vp model defined on the PHYSICAL grid (same as staircase
    demo). The propagator passes it through to the curvilinear path; in
    this MVP the velocity is treated as constant per cell in the
    computational grid (no resampling onto stretched cells)."""
    vp = np.full((nz, nx), 2800.0, dtype=np.float32)
    near_surface_thickness = 6
    middle_thickness = 35
    for ix in range(nx):
        s = topo[ix]
        vp[s : s + near_surface_thickness, ix] = 1800.0
        vp[s + near_surface_thickness : s + near_surface_thickness + middle_thickness, ix] = 2300.0
    return vp


def build_wavelet() -> torch.Tensor:
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32))


def build_geometry(topo: np.ndarray):
    """Sources / receivers live on the COMPUTATIONAL (ξ, η) grid for the
    curvilinear path. Conveniently, the surface in computational coords
    is the flat row η-index = 0 for EVERY column — so a "source on the
    surface" is just (x, 0)."""
    src_x = NX // 3
    src_z = 2   # 2 cells below the (flat) computational surface
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    rec_x = np.arange(6, NX - 6, 3, dtype=np.int64)
    rec_z = np.full_like(rec_x, 1)   # one cell below the flat surface
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    return sources, receivers, src_x, src_z, rec_x


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_model(vp, topo, src_xz, rec_x, path):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    vp_plot = vp.astype(np.float32).copy()
    air_mask = np.arange(NZ)[:, None] < topo[None, :]
    vp_plot[air_mask] = np.nan
    im = ax.imshow(vp_plot, cmap="viridis", extent=extent, aspect="auto")
    fig.colorbar(im, ax=ax, label="vp (m/s)", shrink=0.85)
    ax.plot(np.arange(NX) * DH * 1e-3, topo * DH * 1e-3, "k-", lw=1.2, label="topography")
    sx, sz_comp = src_xz
    # Map computational source (sx, sz_comp) back to physical for display.
    z_phys = topo[sx] + sz_comp
    ax.plot([sx * DH * 1e-3], [z_phys * DH * 1e-3], "r*", ms=14, label="source")
    rz_phys = topo[rec_x] + 1
    ax.plot(rec_x * DH * 1e-3, rz_phys * DH * 1e-3, "yv", ms=3, label="receivers")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title("Acoustic 2-D model — curvilinear-grid topography (Stage 3)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_record(record, rec_x, path):
    rec = np.squeeze(record)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    pct = float(np.percentile(np.abs(rec), 99.0))
    norm = TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct)
    extent = [rec_x.min() * DH * 1e-3, rec_x.max() * DH * 1e-3, NT * DT, 0]
    ax.imshow(rec, cmap="seismic", norm=norm, extent=extent, aspect="auto")
    ax.set_xlabel("receiver x (km)")
    ax.set_ylabel("time (s)")
    ax.set_title("Acoustic curvilinear shot gather")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_wavefield_gif(snaps_p_comp, topo, snap_times, src_xz, path):
    """``snaps_p_comp`` lives in computational (ξ, η) coords with the
    surface at row 0. For display we resample each frame to physical
    (X, Z) so the topography appears in its true shape.
    """
    if snaps_p_comp.ndim != 3:
        raise ValueError(f"expected (n_snap, nz, nx), got {snaps_p_comp.shape}")
    n_snap, nz_pad, nx_pad = snaps_p_comp.shape
    snaps_phys_comp = snaps_p_comp[:, 0 : nz_pad - ABCN, ABCN : nx_pad - ABCN]
    assert snaps_phys_comp.shape[-2:] == (NZ, NX), (
        f"crop mismatch: {snaps_phys_comp.shape}"
    )

    # Resample computational η ∈ [0,1] back to physical Z per column:
    #   Z_phys_row = topo[ix] + (η_index / (NZ-1)) * (NZ - 1 - topo[ix])
    # Equivalently the row 0 of the comp grid → physical row topo[ix],
    # row NZ-1 → physical row NZ-1.
    snaps_physgrid = np.zeros_like(snaps_phys_comp)
    for ix in range(NX):
        s = int(topo[ix])
        depth = NZ - 1 - s
        if depth <= 0:
            continue
        # Interpolate snaps_phys_comp[:, :, ix] (length NZ) onto physical
        # depth rows [s, s+1, ..., NZ-1] (length depth+1).
        eta_phys = np.linspace(0.0, 1.0, depth + 1, dtype=np.float32)
        eta_comp = np.linspace(0.0, 1.0, NZ, dtype=np.float32)
        for t in range(n_snap):
            snaps_physgrid[t, s : NZ, ix] = np.interp(
                eta_phys, eta_comp, snaps_phys_comp[t, :, ix]
            )

    pct = float(np.percentile(np.abs(snaps_physgrid), 99.5))
    pct = max(pct, 1e-12)
    norm = TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    im = ax.imshow(snaps_physgrid[0], cmap="seismic", norm=norm, extent=extent, aspect="auto")
    topo_km = topo * DH * 1e-3
    x_km = np.arange(NX) * DH * 1e-3
    (topo_line,) = ax.plot(x_km, topo_km, "k-", lw=1.2)
    ax.fill_between(x_km, 0, topo_km, color="white", alpha=0.95)
    sx, sz_comp = src_xz
    z_phys = topo[sx] + sz_comp
    ax.plot([sx * DH * 1e-3], [z_phys * DH * 1e-3], "r*", ms=12)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(f"Curvilinear  |p|  at t = {snap_times[0] * DT:.3f} s")
    fig.colorbar(im, ax=ax, shrink=0.85, label="pressure")

    def update(i):
        im.set_data(snaps_physgrid[i])
        ax.set_title(f"Curvilinear  |p|  at t = {snap_times[i] * DT:.3f} s")
        return [im, topo_line]

    anim = FuncAnimation(fig, update, frames=n_snap, interval=80, blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[curv-acoustic] device = {device}")

    topo = build_topography(NX)
    vp_np = build_velocity(NZ, NX, topo)
    vp = torch.from_numpy(vp_np).to(device)
    wavelet = build_wavelet().to(device)
    sources, receivers, src_x, src_z, rec_x = build_geometry(topo)
    sources = sources.to(device)
    receivers = receivers.to(device)

    equation = AcousticCurvilinear(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        equation, shape=(NZ, NX),
        free_surface=True,
        topography=topo,
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )
    snap_times = list(range(0, NT, SNAP_STRIDE))
    print(f"[curv-acoustic] forward ({NT} steps, {len(snap_times)} snaps)...")
    with torch.no_grad():
        record, snaps = prop(
            wavelet, sources, receivers, models=[vp],
            return_wavefield=True, snapshot_times=snap_times,
        )
    p_snaps = snaps[:, 0, 0, 0].detach().cpu().numpy()
    record_np = record.detach().cpu().numpy()
    print(f"[curv-acoustic] record {record.shape}, snaps {snaps.shape}, "
          f"|p|max history early/mid/late = "
          f"{np.abs(p_snaps[:5]).max():.2e} / "
          f"{np.abs(p_snaps[len(p_snaps)//2-2:len(p_snaps)//2+3]).max():.2e} / "
          f"{np.abs(p_snaps[-5:]).max():.2e}")

    print("[curv-acoustic] writing outputs...")
    plot_model(vp_np, topo, src_xz=(src_x, src_z), rec_x=rec_x,
               path=OUTPUT_DIR / "acoustic2d_curv_model.png")
    plot_record(record_np, rec_x=rec_x,
                path=OUTPUT_DIR / "acoustic2d_curv_record.png")
    save_wavefield_gif(p_snaps, topo,
                       snap_times=np.array(snap_times, dtype=np.int64),
                       src_xz=(src_x, src_z),
                       path=OUTPUT_DIR / "acoustic2d_curv_wavefield.gif")
    print("[curv-acoustic] done.")


if __name__ == "__main__":
    main()
