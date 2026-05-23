"""Cao & Chen (2018) Geophysics 83(6) Figure 3 — Gaussian hill elastic
forward modeling with the APM free-surface method.

Setup (matches the paper as closely as the small differences in source /
receiver default fields allow):

* Domain  1020 m × 374 m, dh = 4 m  → grid 256 × 94
* Topography  z_top(x) = z0 − h0 · exp(−(x − x0)² / σ²)
              z0 = 80 m, h0 = 60 m, x0 = 510 m, σ = 100 m
* Source  25 Hz Ricker, default explosion (sxx + szz), placed at
  one cell below the hilltop
* Receivers  256 surface receivers, one per column, each at row
  ``topo[ix] + 1`` (one cell below the local surface — "geophone in
  the dirt")
* NT  4000 timesteps, dt = 4 × 10⁻⁴ s ⇒ total time 1.6 s
* Background  homogeneous half-space, vp = 3000 m/s, vs = 1500 m/s,
  ρ = 2200 kg/m³ (Cao & Chen 2018 Model 1)

Outputs under ``outputs/``:

    elastic_apm_gaussian_model.png       vp / vs / ρ panels with topo
    elastic_apm_gaussian_record_vz.png   shot gather (raw + AGC)
    elastic_apm_gaussian_wavefield.gif   |v| snapshots with topo overlay

Run:
    python examples/wavefields/topography/elastic_apm_gaussian.py [--cpu]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import ElasticAPM
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Cao 2018 Fig 3 config
# ---------------------------------------------------------------------------

DH = 4.0
NX = 256
NZ = 94                  # 374 m / 4 m = 93.5, round up to 94
DT = 4.0e-4
NT = 4000
ABCN = 30
SO = 4

FREQ = 25.0
DELAY = 0.06

# Topography parameters
TOPO_Z0 = 80.0           # base (depth in m where surface flattens out)
TOPO_H0 = 60.0           # hill peak height above base (m)
TOPO_X0 = 510.0          # hill centre (m)
TOPO_SIGMA = 100.0       # Gaussian half-width (m)

# Bulk material — homogeneous half-space (Cao 2018 Model 1).
VP = 3000.0
VS = 1500.0
RHO = 2200.0

SNAP_STRIDE = 80
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_topography():
    """Return ``(topo_row, air_mask)`` where ``topo_row`` is the integer
    surface row per column and ``air_mask`` is the 2-D boolean mask."""
    x = np.arange(NX, dtype=np.float32) * DH
    z_top_m = TOPO_Z0 - TOPO_H0 * np.exp(-((x - TOPO_X0) ** 2) / (TOPO_SIGMA ** 2))
    topo_row = np.clip(np.round(z_top_m / DH), 0, NZ - 1).astype(np.int64)
    air_mask = np.zeros((NZ, NX), dtype=np.float32)
    for ix in range(NX):
        air_mask[: topo_row[ix], ix] = 1.0
    return topo_row, air_mask


def build_models():
    vp = np.full((NZ, NX), VP, dtype=np.float32)
    vs = np.full((NZ, NX), VS, dtype=np.float32)
    rho = np.full((NZ, NX), RHO, dtype=np.float32)
    return vp, vs, rho


def build_wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e3 * ricker(t, f=FREQ)).astype(np.float32))


def build_geometry(topo_row):
    """Source at one cell below the hilltop; receivers at one cell below
    the local surface for every column."""
    src_x = int(round(TOPO_X0 / DH))
    src_z = int(topo_row[src_x]) + 1
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    rec_x = np.arange(NX, dtype=np.int64)
    rec_z = (topo_row + 1).astype(np.int64)
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    return sources, receivers, src_x, src_z, rec_x


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_model(vp, vs, rho, topo_row, src_xz, rec_x, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    extent = [0, NX * DH, NZ * DH, 0]    # metres
    air_mask = np.arange(NZ)[:, None] < topo_row[None, :]
    sx, sz = src_xz
    panels = [
        ("vp (m/s)", vp, "viridis"),
        ("vs (m/s)", vs, "viridis"),
        (r"rho (kg/m$^3$)", rho, "magma"),
    ]
    for ax, (label, arr, cmap) in zip(axes, panels):
        a = arr.astype(np.float32).copy()
        a[air_mask] = np.nan
        im = ax.imshow(a, cmap=cmap, extent=extent, aspect="auto")
        ax.plot(np.arange(NX) * DH, topo_row * DH, "k-", lw=1.2)
        ax.plot([sx * DH], [sz * DH], "r*", ms=14, label="source")
        ax.plot(rec_x[::8] * DH, (topo_row[::8] + 1) * DH, "yv", ms=3, label="rec (every 8th)")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Cao & Chen 2018 Fig 3 — Gaussian hill model (ElasticAPM)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_record(record_vz, rec_x, path):
    """record shape (B=1, nt, nrec, nfield=2). Pick vz (channel 1)."""
    rec = np.squeeze(record_vz)
    while rec.ndim > 2:
        rec = rec.squeeze(0)

    # Trace-by-trace AGC (200 ms window) — Cao 2018 normalises traces
    # before plotting to make the Rayleigh wave visible across the
    # full receiver array.
    agc_win = max(1, int(0.20 / DT))
    kernel = np.ones(agc_win, dtype=np.float32) / agc_win
    rec2 = rec.astype(np.float32) ** 2
    rms = np.sqrt(np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), 0, rec2
    ) + 1e-30)
    rec_agc = rec / rms

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    extent = [rec_x.min() * DH, rec_x.max() * DH, NT * DT, 0]

    pct = float(np.percentile(np.abs(rec), 99.0))
    axes[0].imshow(rec, cmap="seismic",
                   norm=TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct),
                   extent=extent, aspect="auto")
    axes[0].set_title("Raw vz")
    axes[0].set_xlabel("receiver x (m)")
    axes[0].set_ylabel("time (s)")

    pct_agc = float(np.percentile(np.abs(rec_agc), 98.0))
    axes[1].imshow(rec_agc, cmap="seismic",
                   norm=TwoSlopeNorm(vmin=-pct_agc, vcenter=0.0, vmax=pct_agc),
                   extent=extent, aspect="auto")
    axes[1].set_title("AGC vz — Rayleigh moveout follows the topo")
    axes[1].set_xlabel("receiver x (m)")
    axes[1].set_ylabel("time (s)")
    fig.suptitle("ElasticAPM shot gather — vz (Cao & Chen 2018 Fig 3 setup)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_wavefield_gif(v_mag, topo_row, snap_times, src_xz, path):
    """|v| snapshots with physical-coordinate axes and topo overlay.

    With ``free_surface=False`` the PML pads BOTH the top and bottom of
    the physical domain — so the physical interior sits at rows
    ``[ABCN : ABCN + NZ]`` (and columns ``[ABCN : ABCN + NX]``) of the
    snapshot tensor.
    """
    if v_mag.ndim != 3:
        raise ValueError(f"expected (n_snap, nz, nx), got {v_mag.shape}")
    n_snap, nz_pad, nx_pad = v_mag.shape
    v_phys = v_mag[:, ABCN : ABCN + NZ, ABCN : ABCN + NX]
    assert v_phys.shape[-2:] == (NZ, NX), (
        f"crop mismatch: got {v_phys.shape}, expected (n_snap, {NZ}, {NX})"
    )

    # Per-frame normalisation against 99 percentile — Rayleigh wave is
    # localised so 99-pct captures the wavefront amplitude.
    global_peak = float(np.abs(v_phys).max())
    floor = max(global_peak * 0.005, 1e-12)
    def _frame_max(frame):
        return max(float(np.percentile(frame, 99.0)), floor)
    frame_scales = np.array([_frame_max(f) for f in v_phys], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    extent = [0, NX * DH, NZ * DH, 0]
    x_m = np.arange(NX) * DH
    topo_m = topo_row * DH
    im = ax.imshow(v_phys[0], cmap="YlOrRd", vmin=0.0, vmax=float(frame_scales[0]),
                   extent=extent, aspect="auto")
    (topo_line,) = ax.plot(x_m, topo_m, "k-", lw=1.2)
    ax.fill_between(x_m, 0, topo_m, color="lightgray")
    sx, sz = src_xz
    ax.plot([sx * DH], [sz * DH], "b*", ms=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"APM  |v|  at t = {snap_times[0] * DT:.3f} s")
    fig.colorbar(im, ax=ax, shrink=0.85, label="|v| (per-frame normalised)")

    def update(i):
        im.set_data(v_phys[i])
        im.set_clim(vmin=0.0, vmax=float(frame_scales[i]))
        ax.set_title(f"APM  |v|  at t = {snap_times[i] * DT:.3f} s")
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
    print(f"[apm-gaussian] device = {device}")

    topo_row, air_mask = build_topography()
    print(f"[apm-gaussian] topo range: rows {topo_row.min()}..{topo_row.max()} "
          f"(depths {topo_row.min() * DH:.0f}..{topo_row.max() * DH:.0f} m)")

    vp_np, vs_np, rho_np = build_models()
    vp = torch.from_numpy(vp_np).to(device)
    vs = torch.from_numpy(vs_np).to(device)
    rho = torch.from_numpy(rho_np).to(device)
    am = torch.from_numpy(air_mask).to(device)
    wavelet = build_wavelet().to(device)
    sources, receivers, src_x, src_z, rec_x = build_geometry(topo_row)
    sources = sources.to(device)
    receivers = receivers.to(device)

    equation = ElasticAPM(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        equation, shape=(NZ, NX),
        free_surface=False,        # APM IS the FS — no propagator-side mirror
        topography=None,           # not used by APM
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )
    snap_times = list(range(0, NT, SNAP_STRIDE))
    print(f"[apm-gaussian] forward ({NT} steps, {len(snap_times)} snaps)...")
    with torch.no_grad():
        record, snaps = prop(
            wavelet, sources, receivers,
            models=[vp, vs, rho, am],
            return_wavefield=True, snapshot_times=snap_times,
        )
    print(f"[apm-gaussian] record {tuple(record.shape)}, snaps {tuple(snaps.shape)}")
    vx_snaps = snaps[:, 0, 0, 0].detach().cpu().numpy()
    vz_snaps = snaps[:, 1, 0, 0].detach().cpu().numpy()
    v_mag = np.sqrt(vx_snaps ** 2 + vz_snaps ** 2)
    print(f"[apm-gaussian] |v|max history: "
          f"early {v_mag[: 5].max():.3e}  "
          f"mid {v_mag[len(v_mag)//2 - 2 : len(v_mag)//2 + 3].max():.3e}  "
          f"late {v_mag[-5:].max():.3e}")

    record_np = record.detach().cpu().numpy()
    record_vz = record_np[..., 1]   # channel 1 = vz

    print("[apm-gaussian] writing model / record / wavefield gif...")
    plot_model(vp_np, vs_np, rho_np, topo_row, src_xz=(src_x, src_z), rec_x=rec_x,
               path=OUTPUT_DIR / "elastic_apm_gaussian_model.png")
    plot_record(record_vz, rec_x=rec_x,
                path=OUTPUT_DIR / "elastic_apm_gaussian_record_vz.png")
    save_wavefield_gif(v_mag, topo_row,
                       snap_times=np.array(snap_times, dtype=np.int64),
                       src_xz=(src_x, src_z),
                       path=OUTPUT_DIR / "elastic_apm_gaussian_wavefield.gif")
    print("[apm-gaussian] done.")


if __name__ == "__main__":
    main()
