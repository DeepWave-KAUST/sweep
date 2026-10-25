"""Elastic 2-D forward modeling with **curvilinear** free-surface
topography — counterpart of ``elastic2d_hill_demo.py``.

The boundary-fitted grid removes the staircase corner energy injection
that drives the staircase late-time instability, so the simulation can
run long enough to watch the Rayleigh wave sweep the whole receiver
array.  The solver is the collocated asymmetric-operator scheme of
Bartana, Xu & Rickett (SLB); see
:mod:`sweep.equations.elastic_curvilinear`.

Outputs (in ``outputs/``):
    elastic2d_curv_model.png         vp/vs/rho with topo overlay
    elastic2d_curv_record_vz.png     vz shot gather (raw + AGC)
    elastic2d_curv_wavefield.gif     |v| snapshots in physical coords
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import ElasticCurvilinear
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Model / acquisition parameters
# ---------------------------------------------------------------------------

NZ, NX = 180, 320
DH = 10.0
DT = 8.0e-4
NT = 2500                  # 2.0 s
ABCN = 50
SO = 16                    # 2L; the paper's operator is L = 8
FREQ = 4.0
DELAY = 0.30

SNAP_STRIDE = 40
SNAP_FRAMES = NT // SNAP_STRIDE

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_topography(nx: int) -> np.ndarray:
    """Moderate ridges: ~10 rows of relief on 40-cell-wide hills."""
    x = np.arange(nx, dtype=np.float32)
    peak_a = 8.0 * np.exp(-((x - nx * 0.42) ** 2) / (2.0 * 40.0**2))
    peak_b = 6.0 * np.exp(-((x - nx * 0.72) ** 2) / (2.0 * 38.0**2))
    elevation = peak_a + peak_b
    surface_row = np.round(20.0 - elevation).astype(np.int64)
    return np.clip(surface_row, 0, nx)


def build_models(nz: int, nx: int, topo: np.ndarray):
    """Layers that follow the surface, on the physical grid as for any
    equation: the propagator samples them at the boundary-fitted rows."""
    vp = np.full((nz, nx), 2800.0, dtype=np.float32)
    near = 8
    middle = 45
    for ix in range(nx):
        s = topo[ix]
        vp[s : s + near, ix] = 1800.0
        vp[s + near : s + near + middle, ix] = 2400.0
    vs = (vp / 1.73).astype(np.float32)
    rho = np.full((nz, nx), 2000.0, dtype=np.float32)
    rho[vp < 2000.0] = 1700.0
    rho[vp > 2600.0] = 2400.0
    return vp, vs, rho


def build_wavelet() -> torch.Tensor:
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    return torch.tensor((1.0e7 * ricker(t, f=FREQ)).astype(np.float32))


def build_geometry(topo: np.ndarray):
    """Physical grid indices, as on any grid (the propagator maps them onto
    the boundary-fitted rows): the source near the left edge 2 cells below
    the surface, the receivers one cell below it across the line."""
    src_x = int(NX * 0.12)
    src_z = int(topo[src_x]) + 2
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64))
    rec_x = np.arange(int(NX * 0.20), NX - 10, 3, dtype=np.int64)
    rec_z = topo[rec_x] + 1
    receivers = torch.from_numpy(np.stack([rec_x, rec_z], axis=-1)[None, ...])
    return sources, receivers, src_x, src_z, rec_x


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_model(vp, vs, rho, topo, src_xz, rec_x, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    air_mask = np.arange(NZ)[:, None] < topo[None, :]
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
        ax.plot(np.arange(NX) * DH * 1e-3, topo * DH * 1e-3, "k-", lw=1.2)
        ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "r*", ms=12)
        ax.plot(rec_x * DH * 1e-3, (topo[rec_x] + 1) * DH * 1e-3, "yv", ms=2)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Elastic 2-D — curvilinear-grid topography (Stage 3, src ★, rec ▼)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_record(record_vz, rec_x, path):
    rec = record_vz
    while rec.ndim > 2:
        rec = rec.squeeze(0) if rec.shape[0] == 1 else rec.squeeze(-1)

    agc_win = max(1, int(0.20 / DT))
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
    axes[1].set_title("AGC (200 ms) — Rayleigh moveout across the array")
    fig.suptitle("Elastic curvilinear shot gather — vz")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_wavefield_gif(snaps_v_mag, topo, snap_times, src_xz, path):
    """``snaps_v_mag`` comes back on the physical (X, Z) grid -- the
    propagator maps the boundary-fitted rows back itself -- so the
    topography is in its true shape already; only the PML pads are cropped
    here."""
    if snaps_v_mag.ndim != 3:
        raise ValueError(f"expected (n_snap, nz, nx), got {snaps_v_mag.shape}")
    n_snap, nz_pad, nx_pad = snaps_v_mag.shape
    snaps_physgrid = snaps_v_mag[:, 0 : nz_pad - ABCN, ABCN : nx_pad - ABCN]
    assert snaps_physgrid.shape[-2:] == (NZ, NX), f"crop mismatch: {snaps_physgrid.shape}"

    fig, ax = plt.subplots(figsize=(9, 5.0), constrained_layout=True)
    extent = [0, NX * DH * 1e-3, NZ * DH * 1e-3, 0]
    x_km = np.arange(NX) * DH * 1e-3
    topo_km = topo * DH * 1e-3

    # Per-frame normalisation. The elastic wavefront is highly
    # localised (the wave touches < a few percent of cells at any one
    # time), so picking the 99-th percentile of |v| gives a visible
    # body wave while saturating only the brightest wavefront pixels.
    # The percentile is over the cells below the surface, so the zeros of
    # the air do not dilute it.
    wet = np.arange(NZ)[:, None] >= topo[None, :]
    global_max = float(np.abs(snaps_physgrid).max())
    floor = max(global_max * 0.005, 1e-12)
    def _frame_max(frame):
        return max(float(np.percentile(frame[wet], 99.0)), floor)
    frame_scales = np.array([_frame_max(f) for f in snaps_physgrid], dtype=np.float32)

    vmax0 = float(frame_scales[0])
    im = ax.imshow(snaps_physgrid[0], cmap="YlOrRd", vmin=0.0, vmax=vmax0,
                   extent=extent, aspect="auto")
    (topo_line,) = ax.plot(x_km, topo_km, "k-", lw=1.2)
    ax.fill_between(x_km, 0, topo_km, color="lightgray")
    sx, sz = src_xz
    ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "b*", ms=12)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(f"Curvilinear  |v|  at t = {snap_times[0] * DT:.3f} s")
    fig.colorbar(im, ax=ax, shrink=0.85, label="|v| (per-frame normalised)")

    def update(i):
        im.set_data(snaps_physgrid[i])
        im.set_clim(vmin=0.0, vmax=float(frame_scales[i]))
        ax.set_title(f"Curvilinear  |v|  at t = {snap_times[i] * DT:.3f} s")
        return [im, topo_line]

    anim = FuncAnimation(fig, update, frames=n_snap, interval=80, blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    args = parser.parse_args()
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[curv-elastic] device = {device}")

    topo = build_topography(NX)
    vp_np, vs_np, rho_np = build_models(NZ, NX, topo)
    vp = torch.from_numpy(vp_np).to(device)
    vs = torch.from_numpy(vs_np).to(device)
    rho = torch.from_numpy(rho_np).to(device)
    wavelet = build_wavelet().to(device)
    sources, receivers, src_x, src_z, rec_x = build_geometry(topo)
    sources = sources.to(device)
    receivers = receivers.to(device)

    equation = ElasticCurvilinear(spatial_order=SO, device=device, backend="torch")
    prop = PropTorch(
        equation, shape=(NZ, NX),
        free_surface=True,
        topography=topo,
        abcn=ABCN, dh=DH, dt=DT,
        use_ckpt=False, impl="eager",
        eager_options={"use_compile": False},
    )

    snap_times = list(range(0, NT, SNAP_STRIDE))
    print(f"[curv-elastic] forward ({NT} steps, {len(snap_times)} snaps)...")
    with torch.no_grad():
        record, snaps = prop(
            wavelet, sources, receivers, models=[vp, vs, rho],
            return_wavefield=True, snapshot_times=snap_times,
        )
    vx_snaps = snaps[:, 0, 0, 0].detach().cpu().numpy()
    vz_snaps = snaps[:, 1, 0, 0].detach().cpu().numpy()
    v_mag = np.sqrt(vx_snaps**2 + vz_snaps**2)
    record_np = record.detach().cpu().numpy()
    record_vz = record_np[..., 1]

    print(f"[curv-elastic] |v|max history: "
          f"early {np.abs(v_mag[:5]).max():.3e}  "
          f"mid {np.abs(v_mag[len(v_mag)//2-2:len(v_mag)//2+3]).max():.3e}  "
          f"late {np.abs(v_mag[-5:]).max():.3e}")

    print("[curv-elastic] writing outputs...")
    plot_model(vp_np, vs_np, rho_np, topo, src_xz=(src_x, src_z), rec_x=rec_x,
               path=OUTPUT_DIR / "elastic2d_curv_model.png")
    plot_record(record_vz, rec_x=rec_x,
                path=OUTPUT_DIR / "elastic2d_curv_record_vz.png")
    save_wavefield_gif(v_mag, topo,
                       snap_times=np.array(snap_times, dtype=np.int64),
                       src_xz=(src_x, src_z),
                       path=OUTPUT_DIR / "elastic2d_curv_wavefield.gif")
    print("[curv-elastic] done.")


if __name__ == "__main__":
    main()
