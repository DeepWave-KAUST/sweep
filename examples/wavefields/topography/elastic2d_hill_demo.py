"""Elastic 2-D forward modeling with an irregular free-surface topography.

Counterpart to ``acoustic2d_hill_demo.py``, but with elastic physics:
the explicit goal here is to **show Rayleigh / surface waves** propagating
along the undulating topography. The acoustic case can't produce them; the
elastic case does (P-to-Rayleigh conversion at the free surface).

Outputs (in ``outputs/``):
    elastic2d_hill_model.png         vp/vs/rho overview with topo overlay
    elastic2d_hill_record_vz.png     surface-receiver shot gather (vz)
    elastic2d_hill_wavefield_vz.gif  vz pressure-equivalent snapshot animation

Run:
    python examples/wavefields/topography/elastic2d_hill_demo.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import Elastic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Model / acquisition parameters
# ---------------------------------------------------------------------------

# Wider domain + longer time so the Rayleigh wave actually sweeps the receiver
# array: v_R ≈ 0.92·vs_min ≈ 960 m/s, so a 2.4 km offset arrives at ~2.5 s.
NZ, NX = 180, 320
DH = 10.0                   # 1.8 km × 3.2 km
DT = 8.0e-4                 # respects max(vp)·dt/dh ≈ 0.23
# Staircase FS on elastic eventually goes exponentially unstable (Robertsson
# 1996, Bohlen & Saenger 2006). The cure for "longer simulation" is fewer
# / shallower staircase corners + lower source frequency, which both push
# the e-folding time τ out. Below we use:
#   - max relief 4 rows (one or two "steps" per hill, very sparse corners)
#   - source frequency 3 Hz (35+ ppw of the slow shear layer)
# Empirically τ ≈ several seconds with these numbers; demo runs 2.7 s.
NT = 3400
ABCN = 50
SO = 4
FREQ = 3.0
DELAY = 0.35

SNAP_STRIDE = 40
SNAP_FRAMES = NT // SNAP_STRIDE

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_topography(nx: int) -> np.ndarray:
    """Very gentle ridges so the discretised surface only has a handful of
    1-row staircase steps along its length. Fewer corners ⇒ later instability."""
    x = np.arange(nx, dtype=np.float32)
    peak_a = 4.0 * np.exp(-((x - nx * 0.45) ** 2) / (2.0 * 55.0**2))
    peak_b = 3.0 * np.exp(-((x - nx * 0.78) ** 2) / (2.0 * 50.0**2))
    elevation = peak_a + peak_b
    surface_row = np.round(20.0 - elevation).astype(np.int64)
    return np.clip(surface_row, 0, nx)


def build_models(nz: int, nx: int, topo: np.ndarray):
    """Three-layer elastic model: weathered cap / bedrock / deep basement."""
    vp_default = 2800.0
    vp = np.full((nz, nx), vp_default, dtype=np.float32)
    near_surface_thickness = 8
    middle_thickness = 45
    for ix in range(nx):
        s = topo[ix]
        vp[s : s + near_surface_thickness, ix]                              = 1800.0
        vp[s + near_surface_thickness : s + near_surface_thickness + middle_thickness, ix] = 2400.0
    vs = (vp / 1.73).astype(np.float32)
    rho = np.full((nz, nx), 2000.0, dtype=np.float32)
    rho[vp < 2000.0] = 1700.0   # softer cap
    rho[vp > 2600.0] = 2400.0
    return vp, vs, rho


def build_wavelet() -> torch.Tensor:
    # Elastic stress sources need a much larger amplitude than acoustic
    # pressure sources to drive comparable particle-velocity wavefields:
    # vz ~ (dt/rho) * d(stress)/dz, and dt/rho ≈ 4e-7 for our parameters,
    # so the wavelet has to be scaled up ~1e4× to keep vz O(10) so the
    # propagating wavefield is visible alongside the high-amplitude source.
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e7 * ricker(t, f=FREQ)).astype(np.float32))


def build_geometry(topo: np.ndarray):
    """Source near the left edge so the Rayleigh wave train sweeps
    rightward across the full receiver array (cleaner moveout signature
    on the shot gather)."""
    src_x = int(NX * 0.12)
    src_z = int(topo[src_x]) + 2
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    rec_x = np.arange(int(NX * 0.20), NX - 10, 3, dtype=np.int64)
    rec_z = (topo[rec_x] + 1).astype(np.int64)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    return sources, receivers, src_x, int(src_z), rec_x


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_model(vp, vs, rho, topo, src_xz, rec_x, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    x_km = np.arange(NX) * DH * 1e-3
    topo_km = topo * DH * 1e-3
    sx, sz = src_xz
    air_mask = np.arange(NZ)[:, None] < topo[None, :]

    panels = [
        ("vp (m/s)", vp, "viridis"),
        ("vs (m/s)", vs, "viridis"),
        (r"rho (kg/m$^3$)", rho, "magma"),
    ]
    for ax, (label, arr, cmap) in zip(axes, panels):
        a = arr.copy()
        a[air_mask] = np.nan
        im = ax.imshow(a, cmap=cmap, extent=extent, aspect="auto")
        ax.plot(x_km, topo_km, "k-", lw=1.2)
        ax.fill_between(x_km, 0, topo_km, color="white")
        ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "r*", ms=12)
        ax.plot(rec_x * DH * 1e-3, (topo[rec_x] + 1) * DH * 1e-3, "yv", ms=2.5)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Elastic 2-D model with irregular topography (src ★, rec ▼)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_record(record_vz, rec_x, path: Path) -> None:
    """Two-panel shot gather: raw + trace-by-trace AGC.

    Raw shows the true amplitude ratio (direct/body waves dominate, hard
    to see Rayleigh on far traces). AGC normalises each trace to a moving
    RMS window so the Rayleigh moveout is clearly visible across the
    whole array.
    """
    rec = record_vz
    while rec.ndim > 2:
        rec = rec.squeeze(0) if rec.shape[0] == 1 else rec.squeeze(-1)

    # AGC: divide each trace by a sliding-window RMS.
    agc_win = max(1, int(0.20 / DT))  # 200-ms window
    kernel = np.ones(agc_win, dtype=np.float32) / agc_win
    rec2 = rec.astype(np.float32) ** 2
    rms = np.sqrt(np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), 0, rec2
    ) + 1e-30)
    rec_agc = rec / rms

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    extent = [rec_x.min() * DH * 1e-3, rec_x.max() * DH * 1e-3, NT * DT, 0]

    pct = float(np.percentile(np.abs(rec), 99.0))
    norm = TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct)
    axes[0].imshow(rec, cmap="seismic", norm=norm, extent=extent, aspect="auto")
    axes[0].set_xlabel("receiver x (km)")
    axes[0].set_ylabel("time (s)")
    axes[0].set_title("Raw  vz")

    pct_agc = float(np.percentile(np.abs(rec_agc), 98.0))
    norm_agc = TwoSlopeNorm(vmin=-pct_agc, vcenter=0.0, vmax=pct_agc)
    axes[1].imshow(rec_agc, cmap="seismic", norm=norm_agc, extent=extent, aspect="auto")
    axes[1].set_xlabel("receiver x (km)")
    axes[1].set_ylabel("time (s)")
    axes[1].set_title("Trace-by-trace AGC (200 ms)\nRayleigh moveout visible across the array")
    fig.suptitle("Elastic shot gather — vz at surface receivers")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_wavefield_gif(snaps_v_mag, topo, snapshot_times, src_xz, path: Path) -> None:
    """``snaps_v_mag`` = |v| = sqrt(vx² + vz²); 2D scalar per frame, always
    non-negative. Surface-trapped Rayleigh waves show up as a bright
    shell hugging the topo line; body waves spread as expanding arcs."""
    if snaps_v_mag.ndim != 3:
        raise ValueError(f"expected (n_snap, nz, nx), got {snaps_v_mag.shape}")
    n_snap, nz_pad, nx_pad = snaps_v_mag.shape
    snaps_phys = snaps_v_mag[:, 0 : nz_pad - ABCN, ABCN : nx_pad - ABCN]
    nz_phys, nx_phys = snaps_phys.shape[-2:]
    assert (nz_phys, nx_phys) == (NZ, NX), (
        f"crop mismatch: got ({nz_phys}, {nx_phys}), expected ({NZ}, {NX})"
    )

    fig, ax = plt.subplots(figsize=(9, 5.0), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    x_km = np.arange(NX) * DH * 1e-3
    topo_km = topo * DH * 1e-3

    # Per-frame normalisation against the 60-th percentile of |v| so that
    # localised hot spots (source firing + the well-known staircase-corner
    # instability blob in the topo image method) saturate, leaving most of
    # the colormap budget for the propagating P / S / Rayleigh wave. Floor
    # set from the median of non-zero cells so frames before any wave
    # arrives stay quiet.
    floor = max(float(np.median(snaps_phys[snaps_phys > 0])) * 4.0, 1e-12)
    def _frame_max(frame):
        return max(float(np.percentile(frame, 60.0)), floor)

    vmax0 = _frame_max(snaps_phys[0])
    im = ax.imshow(snaps_phys[0], cmap="YlOrRd", vmin=0.0, vmax=vmax0,
                   extent=extent, aspect="auto")
    (topo_line,) = ax.plot(x_km, topo_km, "k-", lw=1.2)
    ax.fill_between(x_km, 0, topo_km, color="lightgray")
    sx, sz = src_xz
    ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "b*", ms=12)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(f"|v| at t = {snapshot_times[0] * DT:.3f} s")
    fig.colorbar(im, ax=ax, shrink=0.85, label="|v| (per-frame normalised)")

    def update(i):
        im.set_data(snaps_phys[i])
        im.set_clim(vmin=0.0, vmax=_frame_max(snaps_phys[i]))
        ax.set_title(f"|v| at t = {snapshot_times[i] * DT:.3f} s")
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
    print(f"[elastic-topo] device = {device}")

    topo = build_topography(NX)
    vp_np, vs_np, rho_np = build_models(NZ, NX, topo)
    vp = torch.from_numpy(vp_np).to(device)
    vs = torch.from_numpy(vs_np).to(device)
    rho = torch.from_numpy(rho_np).to(device)
    wavelet = build_wavelet().to(device)
    sources, receivers, src_x, src_z, rec_x = build_geometry(topo)
    sources = sources.to(device)
    receivers = receivers.to(device)

    equation = Elastic(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        equation,
        shape=(NZ, NX),
        free_surface=True,
        topography=topo,
        abcn=ABCN,
        dh=DH,
        dt=DT,
        use_ckpt=False,
        impl="eager",
        # torch.compile + elastic free-surface + CUDA currently has a
        # ``_free_surface._concat`` type-dispatch issue (separate follow-up
        # task). Disable compile for this demo; numerical correctness is
        # unaffected and the demo runs once.
        eager_options={"use_compile": False},
    )

    snap_times = list(range(0, NT, SNAP_STRIDE))
    print(f"[elastic-topo] forward modelling ({NT} steps, {len(snap_times)} snapshots)...")
    with torch.no_grad():
        record, snaps = prop(
            wavelet, sources, receivers, models=[vp, vs, rho],
            return_wavefield=True, snapshot_times=snap_times,
        )
    print(f"[elastic-topo] record shape {tuple(record.shape)}; "
          f"snapshots shape {tuple(snaps.shape)}")

    # snaps[:, field_idx, batch, channel, z, x]
    # Field indices: 0=vx, 1=vz, 2=sxx, 3=szz, 4=sxz, 5-14=PML mem.
    vx_snaps = snaps[:, 0, 0, 0].detach().cpu().numpy()
    vz_snaps = snaps[:, 1, 0, 0].detach().cpu().numpy()
    v_mag_snaps = np.sqrt(vx_snaps**2 + vz_snaps**2)
    record_np = record.detach().cpu().numpy()
    # record_np: (B, nt, nrec, nfield) — nfield=2 (vx, vz). Take vz channel.
    record_vz = record_np[..., 1]   # (B, nt, nrec)

    print("[elastic-topo] writing model / record / wavefield gif...")
    plot_model(
        vp_np, vs_np, rho_np, topo,
        src_xz=(src_x, src_z), rec_x=rec_x,
        path=OUTPUT_DIR / "elastic2d_hill_model.png",
    )
    plot_record(
        record_vz, rec_x,
        path=OUTPUT_DIR / "elastic2d_hill_record_vz.png",
    )
    save_wavefield_gif(
        v_mag_snaps, topo,
        snapshot_times=np.array(snap_times, dtype=np.int64),
        src_xz=(src_x, src_z),
        path=OUTPUT_DIR / "elastic2d_hill_wavefield_vmag.gif",
    )
    print("[elastic-topo] done.")


if __name__ == "__main__":
    main()
