"""Acoustic 2-D forward modeling with an irregular free-surface topography.

Builds an undulating surface (two Gaussian hills + a sine ripple), runs the
eager forward, and saves:

    outputs/acoustic2d_hill_model.png      vp model with topo overlay
    outputs/acoustic2d_hill_record.png     shot-gather seismic record
    outputs/acoustic2d_hill_wavefield.gif  pressure snapshots, topo overlaid

Run as a script:

    python examples/wavefields/topography/acoustic2d_hill_demo.py

It picks ``cuda`` when available, falls back to CPU otherwise. ~40 s on a V100.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Model / acquisition parameters
# ---------------------------------------------------------------------------

NZ, NX = 160, 320           # physical grid
DH = 10.0                   # grid spacing (m)
DT = 1.0e-3                 # time step (s)
NT = 900                    # ~0.9 s
ABCN = 40                   # PML half-width
SO = 4                      # FD spatial order
FREQ = 8.0                  # Ricker dominant freq (Hz)
DELAY = 0.12                # Ricker time delay (s)

# Snapshot every SNAP_STRIDE steps, dropping the source ring-up.
SNAP_STRIDE = 12
SNAP_FRAMES = NT // SNAP_STRIDE

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_topography(nx: int) -> np.ndarray:
    """Undulating surface: two Gaussian peaks + a low-amplitude sine ripple.

    Returns the surface row index per physical column (0 = top of domain).
    Values up to ~25 rows (250 m elevation contrast above the deepest valley).
    """
    x = np.arange(nx, dtype=np.float32)
    peak_a = 22.0 * np.exp(-((x - nx * 0.30) ** 2) / (2.0 * 22.0**2))
    peak_b = 16.0 * np.exp(-((x - nx * 0.68) ** 2) / (2.0 * 30.0**2))
    ripple = 3.0 * np.sin(2.0 * np.pi * x / 60.0)
    # Surface row from the top — bigger value = lower surface (deeper in z).
    # We want hills to BULGE upward, so the surface row is SMALL there.
    elevation = peak_a + peak_b + ripple                     # positive bulges up
    surface_row = np.round(28.0 - elevation).astype(np.int64)
    return np.clip(surface_row, 0, nx)  # safety


def build_velocity(nz: int, nx: int, topo: np.ndarray) -> np.ndarray:
    """Three-layer vp model below the irregular surface:
        ~1800 m/s near-surface weathered layer (50 m thick)
        ~2300 m/s middle layer
        ~2800 m/s basement.
    """
    vp = np.full((nz, nx), 2800.0, dtype=np.float32)
    # layer interfaces, depth-from-surface (rows)
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
    # Shot near 1/3 of the model, just below the local surface.
    src_x = NX // 3
    src_z = int(topo[src_x]) + 2
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    # Receivers across the full surface, one row below the local surface.
    rec_x = np.arange(6, NX - 6, 3, dtype=np.int64)
    rec_z = (topo[rec_x] + 1).astype(np.int64)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    return sources, receivers, src_x, int(src_z)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _format_axes(ax, title):
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(title)


def plot_model(vp: np.ndarray, topo: np.ndarray, src_xz, rec_x, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    im = ax.imshow(vp, cmap="viridis", extent=extent, aspect="auto")
    fig.colorbar(im, ax=ax, label="vp (m/s)", shrink=0.85)
    ax.plot(np.arange(NX) * DH * 1e-3, topo * DH * 1e-3, "w-", lw=1.2, label="topography")
    sx, sz = src_xz
    ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "r*", ms=14, label="source")
    ax.plot(rec_x * DH * 1e-3, (topo[rec_x] + 1) * DH * 1e-3, "yv", ms=3, label="receivers")
    ax.legend(loc="lower right", fontsize=8)
    _format_axes(ax, "Acoustic 2-D model with irregular topography")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_record(record: np.ndarray, rec_x: np.ndarray, path: Path) -> None:
    """record shape: (nshot=1, nt, nrec, nfield=1) — squeeze to (nt, nrec)."""
    rec = np.squeeze(record)
    if rec.ndim != 2:
        raise ValueError(f"unexpected record shape {record.shape} -> {rec.shape}")
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    # Clip for display
    pct = np.percentile(np.abs(rec), 99.0)
    norm = TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct)
    extent = [rec_x.min() * DH * 1e-3, rec_x.max() * DH * 1e-3, NT * DT, 0]
    ax.imshow(rec, cmap="seismic", norm=norm, extent=extent, aspect="auto")
    ax.set_xlabel("receiver x (km)")
    ax.set_ylabel("time (s)")
    ax.set_title("Shot gather — surface receivers")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_wavefield_gif(
    snaps: np.ndarray,
    topo: np.ndarray,
    snapshot_times: np.ndarray,
    src_xz,
    path: Path,
) -> None:
    """Build a GIF of the pressure wavefield over time.

    ``snaps`` should be 2-D ``(n_snap, nz_pad, nx_pad)`` after channel squeeze.
    The PML margins are cropped before display.
    """
    if snaps.ndim != 3:
        raise ValueError(f"expected (n_snap, nz, nx), got {snaps.shape}")
    # Crop PML margins. For free_surface, x has abcn each side, z has abcn at
    # the bottom only.
    n_snap, nz_pad, nx_pad = snaps.shape
    snaps_phys = snaps[:, 0 : nz_pad - ABCN, ABCN : nx_pad - ABCN]
    nz_phys, nx_phys = snaps_phys.shape[-2:]
    assert (nz_phys, nx_phys) == (NZ, NX), (
        f"crop mismatch: got ({nz_phys}, {nx_phys}), expected ({NZ}, {NX})"
    )

    # Symmetric colour scale based on the 99-th percentile over the whole
    # cube so amplitude relations across frames stay visible.
    pct = np.percentile(np.abs(snaps_phys), 99.5)
    pct = max(pct, 1e-12)
    norm = TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    im = ax.imshow(snaps_phys[0], cmap="seismic", norm=norm, extent=extent, aspect="auto")
    topo_km = topo * DH * 1e-3
    x_km = np.arange(NX) * DH * 1e-3
    (topo_line,) = ax.plot(x_km, topo_km, "k-", lw=1.2)
    ax.fill_between(x_km, 0, topo_km, color="white", alpha=0.95)  # air mask cosmetic
    sx, sz = src_xz
    ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "r*", ms=12)
    _format_axes(ax, f"t = {snapshot_times[0] * DT:.3f} s")
    fig.colorbar(im, ax=ax, shrink=0.85, label="pressure")

    def update(i):
        im.set_data(snaps_phys[i])
        ax.set_title(f"t = {snapshot_times[i] * DT:.3f} s")
        return [im, topo_line]

    anim = FuncAnimation(fig, update, frames=n_snap, interval=80, blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="force CPU even if CUDA available")
    args = parser.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[topo-demo] running on {device}")

    topo = build_topography(NX)
    vp_np = build_velocity(NZ, NX, topo)
    vp = torch.from_numpy(vp_np).to(device)
    wavelet = build_wavelet().to(device)
    sources, receivers, src_x, src_z = build_geometry(topo)
    sources = sources.to(device)
    receivers = receivers.to(device)

    equation = Acoustic(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        equation,
        shape=(NZ, NX),
        free_surface=True,
        topography=topo,
        abcn=ABCN,
        dh=DH,
        dt=DT,
        use_ckpt=False,
        impl="eager",   # topo currently eager-only
    )

    snap_times = list(range(0, NT, SNAP_STRIDE))
    print(f"[topo-demo] forward modelling ({NT} steps, {len(snap_times)} snapshots)...")
    with torch.no_grad():
        record, snaps = prop(
            wavelet, sources, receivers, models=[vp],
            return_wavefield=True, snapshot_times=snap_times,
        )
    print(f"[topo-demo] record shape {tuple(record.shape)}; "
          f"snapshots shape {tuple(snaps.shape)}")

    # snaps shape: (n_snap, n_wavefield, B, C, nz_pad, nx_pad) — squeeze.
    # Field index 0 is h1 (primary pressure).
    p_snaps = snaps[:, 0, 0, 0].detach().cpu().numpy()
    record_np = record.detach().cpu().numpy()

    print("[topo-demo] writing figures and gif to", OUTPUT_DIR)
    plot_model(
        vp_np, topo, src_xz=(src_x, src_z),
        rec_x=np.arange(6, NX - 6, 3, dtype=np.int64),
        path=OUTPUT_DIR / "acoustic2d_hill_model.png",
    )
    plot_record(
        record_np, rec_x=np.arange(6, NX - 6, 3, dtype=np.int64),
        path=OUTPUT_DIR / "acoustic2d_hill_record.png",
    )
    save_wavefield_gif(
        p_snaps, topo,
        snapshot_times=np.array(snap_times, dtype=np.int64),
        src_xz=(src_x, src_z),
        path=OUTPUT_DIR / "acoustic2d_hill_wavefield.gif",
    )
    print("[topo-demo] done.")


if __name__ == "__main__":
    main()
