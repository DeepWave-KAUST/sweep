"""Acoustic + Elastic forward modeling on the user-supplied ``vp_traced.npy`` /
``topo_traced.npy`` mountainous-terrain model.

Model layout (deduced from the supplied arrays):

* Grid     ``nz=400, nx=1330``, ``dh=25 m`` → 33.25 km × 10 km
* ``vp``   already has air encoded as ~340 m/s; solid ranges ~500-5300 m/s
* ``topo`` per-column depth (m) of the topographic surface below grid row 0,
  i.e. ``topo[ix] ≈ first_solid_row(ix) * dh``

Two simulations are run side-by-side and share the SAME quantised topography
(``air_mask`` derived from ``vp < 400``):

1. **Acoustic 2-D**   ``Acoustic`` + ``free_surface=True`` + ``topography=row``
2. **Elastic 2-D**    ``ElasticAPM`` with vs/rho derived from vp

Outputs written to ``outputs/``:

    real_traced_model.png                 vp, vs, rho panels + topo overlay
    real_traced_acoustic_record.png       acoustic shot gather
    real_traced_acoustic_wavefield.gif    pressure snapshots
    real_traced_elastic_record_vz.png     elastic vz shot gather (raw + AGC)
    real_traced_elastic_wavefield.gif     |v| snapshots

Run:
    python examples/wavefields/topography/real_traced_model_demo.py [--cpu]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import TwoSlopeNorm

from sweep.equations import Acoustic, ElasticAPM
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


# ---------------------------------------------------------------------------
# Paths / fixed grid + acquisition parameters
# ---------------------------------------------------------------------------

VP_PATH = Path("/ibex/user/wangs0j/sweep-stack/vp_traced.npy")
TOPO_PATH = Path("/ibex/user/wangs0j/sweep-stack/topo_traced.npy")

DH = 25.0          # grid spacing (m), from topo/first_solid_row regression
DT = 1.5e-3        # CFL≈0.45 at vp_max=5300 m/s, 4th order spatial
NT = 5400          # 8.1 s
ABCN = 40          # PML half-width
SO = 4             # FD spatial order

FREQ = 4.0         # Ricker dominant freq (Hz). ppw≈4.5 at slowest near-surface
DELAY = 0.30       # peak shifted into +t

AIR_THRESHOLD = 400.0  # vp below this is treated as air

# vs derived as vp/VPVS_RATIO; rho from Gardner
VPVS_RATIO = 1.73
GARDNER_A = 310.0
GARDNER_N = 0.25

# Source / receiver layout
SRC_FRAC_X = 0.5     # source horizontal position as a fraction of nx
SRC_DEPTH_BELOW_SURFACE_CELLS = 2
REC_STRIDE = 4       # one receiver every REC_STRIDE columns
REC_DEPTH_BELOW_SURFACE_CELLS = 1

SNAP_STRIDE = 68     # snapshot every N steps → ~80 frames at NT=5400

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def load_vp_and_topo():
    vp = np.load(VP_PATH).astype(np.float32)
    topo_m = np.load(TOPO_PATH).astype(np.float32)
    assert vp.ndim == 2 and topo_m.ndim == 1, (
        f"unexpected shapes: vp {vp.shape}, topo {topo_m.shape}"
    )
    assert vp.shape[1] == topo_m.shape[0], (
        f"vp width {vp.shape[1]} != topo length {topo_m.shape[0]}"
    )
    return vp, topo_m


def derive_models(vp):
    """Build (vs, rho) from vp + air_mask + topo_row.

    Solid cells:
      vs  = vp / VPVS_RATIO
      rho = GARDNER_A * vp^GARDNER_N  (Gardner relation; ~2300 kg/m³ @ 3 km/s)

    Air cells: keep the SAME Gardner-derived rho and vp/1.73 vs.  ElasticAPM
    zeros wavefields at air via ``zero_at_air``; using realistic ρ everywhere
    avoids the ``dt/ρ`` blow-up that follows from setting ρ_air = 1.2 kg/m³.
    """
    air = vp < AIR_THRESHOLD
    # First, replace air-cell vp with a sensible "background" value so the
    # vs/rho relations stay bounded.  We take the column-wise nearest-solid
    # vp (i.e. the topmost solid cell in that column).
    nz, nx = vp.shape
    solid = ~air
    first_solid = np.full(nx, nz - 1, dtype=np.int64)
    for ix in range(nx):
        col = solid[:, ix]
        if col.any():
            first_solid[ix] = int(np.argmax(col))

    vp_for_props = vp.copy()
    for ix in range(nx):
        if air[:, ix].any():
            vp_for_props[: first_solid[ix], ix] = vp[first_solid[ix], ix]

    vs = (vp_for_props / VPVS_RATIO).astype(np.float32)
    rho = (GARDNER_A * np.power(vp_for_props, GARDNER_N)).astype(np.float32)
    air_mask = air.astype(np.float32)
    return vs, rho, first_solid, air_mask


def build_wavelet(device):
    t = np.arange(NT, dtype=np.float32) * DT - DELAY
    w = (1.0e3 * ricker(t, f=FREQ)).astype(np.float32)
    return torch.tensor(w).to(device)


def build_geometry(topo_row, nx, device):
    src_x = int(round(SRC_FRAC_X * nx))
    src_z = int(topo_row[src_x]) + SRC_DEPTH_BELOW_SURFACE_CELLS
    sources = torch.from_numpy(np.array([[src_x, src_z]], dtype=np.int64)).to(device)

    rec_x = np.arange(8, nx - 8, REC_STRIDE, dtype=np.int64)
    rec_z = (topo_row[rec_x] + REC_DEPTH_BELOW_SURFACE_CELLS).astype(np.int64)
    receivers = torch.from_numpy(
        np.stack([rec_x, rec_z], axis=-1)[None, ...]
    ).to(device)
    return sources, receivers, src_x, src_z, rec_x


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def plot_models(vp, vs, rho, topo_row, src_xz, rec_x, nz, nx, path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    extent = [0, nx * DH * 1e-3, nz * DH * 1e-3, 0]
    air = vp < AIR_THRESHOLD
    sx, sz = src_xz
    panels = [
        ("vp (m/s)", vp, "viridis", None),
        ("vs (m/s)", vs, "viridis", None),
        (r"rho (kg/m$^3$)", rho, "magma", None),
    ]
    for ax, (label, arr, cmap, _) in zip(axes, panels):
        a = arr.astype(np.float32).copy()
        a[air] = np.nan
        im = ax.imshow(a, cmap=cmap, extent=extent, aspect="auto",
                       interpolation="nearest")
        ax.plot(np.arange(nx) * DH * 1e-3, topo_row * DH * 1e-3, "k-", lw=0.8)
        ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "r*", ms=12, label="source")
        ax.plot(rec_x[::8] * DH * 1e-3, (topo_row[rec_x[::8]] + 1) * DH * 1e-3,
                "yv", ms=2, label="receivers (every 8th)")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("z (km)")
        ax.set_title(label)
        fig.colorbar(im, ax=ax, shrink=0.85)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"vp_traced model — grid {nz}x{nx}, dh={DH:.0f} m "
        f"({nx*DH/1e3:.1f} km × {nz*DH/1e3:.1f} km)"
    )
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_record_scalar(record, rec_x, nt, path, label="pressure"):
    """For acoustic — 1 field channel."""
    rec = np.squeeze(record)
    while rec.ndim > 2:
        rec = rec.squeeze(0)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    finite = np.isfinite(rec)
    pct = float(np.percentile(np.abs(rec[finite]), 99.0)) if finite.any() else 0.0
    pct = max(pct, 1e-30)
    extent = [rec_x.min() * DH * 1e-3, rec_x.max() * DH * 1e-3, nt * DT, 0]
    ax.imshow(rec, cmap="seismic",
              norm=TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct),
              extent=extent, aspect="auto")
    ax.set_xlabel("receiver x (km)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"Acoustic shot gather — {label}  (99-pct clip)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_record_elastic(record_vz, rec_x, nt, path):
    rec = np.squeeze(record_vz)
    while rec.ndim > 2:
        rec = rec.squeeze(0)

    # AGC (200 ms window)
    agc_win = max(1, int(0.20 / DT))
    kernel = np.ones(agc_win, dtype=np.float32) / agc_win
    rec2 = rec.astype(np.float32) ** 2
    rms = np.sqrt(
        np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), 0, rec2
        ) + 1e-30
    )
    rec_agc = rec / rms

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    extent = [rec_x.min() * DH * 1e-3, rec_x.max() * DH * 1e-3, nt * DT, 0]

    pct = float(np.percentile(np.abs(rec), 99.0))
    pct = max(pct, 1e-30)
    axes[0].imshow(rec, cmap="seismic",
                   norm=TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct),
                   extent=extent, aspect="auto")
    axes[0].set_xlabel("receiver x (km)")
    axes[0].set_ylabel("time (s)")
    axes[0].set_title("Elastic vz — raw")

    pct_agc = float(np.percentile(np.abs(rec_agc), 98.0))
    pct_agc = max(pct_agc, 1e-30)
    axes[1].imshow(rec_agc, cmap="seismic",
                   norm=TwoSlopeNorm(vmin=-pct_agc, vcenter=0.0, vmax=pct_agc),
                   extent=extent, aspect="auto")
    axes[1].set_xlabel("receiver x (km)")
    axes[1].set_ylabel("time (s)")
    axes[1].set_title("Elastic vz — AGC (200 ms)")

    fig.suptitle("ElasticAPM shot gather — vz")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_wavefield_gif(
    cube, topo_row, snap_times, src_xz, path,
    crop_zslice, crop_xslice, cmap, label, nz, nx,
    symmetric=False,
):
    """One gif builder shared by acoustic (pressure, symmetric) and elastic (|v|).

    ``cube``: shape (n_snap, nz_pad, nx_pad).
    """
    if cube.ndim != 3:
        raise ValueError(f"expected (n_snap, nz, nx), got {cube.shape}")
    phys = cube[:, crop_zslice, crop_xslice]
    nz_phys, nx_phys = phys.shape[-2:]
    assert (nz_phys, nx_phys) == (nz, nx), (
        f"crop mismatch: got {phys.shape}, expected (n_snap, {nz}, {nx})"
    )

    # Air mask overlay (cosmetic): use vp's air pattern.
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    extent = [0, nx * DH * 1e-3, nz * DH * 1e-3, 0]
    x_km = np.arange(nx) * DH * 1e-3
    topo_km = topo_row * DH * 1e-3

    if symmetric:
        pct = float(np.percentile(np.abs(phys), 99.5))
        pct = max(pct, 1e-15)
        norm0 = TwoSlopeNorm(vmin=-pct, vcenter=0.0, vmax=pct)
        im = ax.imshow(phys[0], cmap=cmap, norm=norm0, extent=extent, aspect="auto")
    else:
        global_peak = float(np.abs(phys).max())
        floor = max(global_peak * 0.005, 1e-12)
        def _frame_max(frame):
            return max(float(np.percentile(np.abs(frame), 99.0)), floor)
        frame_scales = np.array([_frame_max(f) for f in phys], dtype=np.float32)
        im = ax.imshow(phys[0], cmap=cmap, vmin=0.0, vmax=float(frame_scales[0]),
                       extent=extent, aspect="auto")

    (topo_line,) = ax.plot(x_km, topo_km, "k-", lw=0.8)
    ax.fill_between(x_km, 0, topo_km, color="lightgray", alpha=0.95)
    sx, sz = src_xz
    ax.plot([sx * DH * 1e-3], [sz * DH * 1e-3], "b*", ms=10)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("z (km)")
    ax.set_title(f"{label}  t = {snap_times[0] * DT:.3f} s")
    fig.colorbar(im, ax=ax, shrink=0.85, label=label)

    def update(i):
        im.set_data(phys[i])
        if not symmetric:
            im.set_clim(vmin=0.0, vmax=float(frame_scales[i]))
        ax.set_title(f"{label}  t = {snap_times[i] * DT:.3f} s")
        return [im, topo_line]

    anim = FuncAnimation(fig, update, frames=phys.shape[0], interval=80, blit=False)
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument("--skip-acoustic", action="store_true")
    parser.add_argument("--skip-elastic", action="store_true")
    parser.add_argument(
        "--nt", type=int, default=NT,
        help="override total number of timesteps (default %(default)s)"
    )
    args = parser.parse_args()
    nt = args.nt

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[real-demo] device = {device}, dt = {DT:.2e} s, NT = {nt}, total = {nt*DT:.2f} s")

    vp, topo_m = load_vp_and_topo()
    nz, nx = vp.shape
    print(f"[real-demo] grid {nz} x {nx} (={nx*DH/1e3:.2f} km x {nz*DH/1e3:.2f} km)")

    vs, rho, topo_row, air_mask = derive_models(vp)
    print(f"[real-demo] solid vp: min={vp[vp>=AIR_THRESHOLD].min():.0f}, "
          f"max={vp.max():.0f} m/s; air cells: {(vp<AIR_THRESHOLD).sum()}/{vp.size}")
    print(f"[real-demo] topo_row: min={topo_row.min()}, max={topo_row.max()}, "
          f"first_solid range  {topo_row.min()*DH:.0f}..{topo_row.max()*DH:.0f} m")

    # ----- to device -----
    vp_t = torch.from_numpy(vp).to(device)
    vs_t = torch.from_numpy(vs).to(device)
    rho_t = torch.from_numpy(rho).to(device)
    am_t = torch.from_numpy(air_mask).to(device)
    wavelet = build_wavelet(device)
    sources, receivers, src_x, src_z, rec_x = build_geometry(topo_row, nx, device)
    print(f"[real-demo] source at (ix={src_x}, iz={src_z}); "
          f"surface row at src col = {topo_row[src_x]}; "
          f"{rec_x.shape[0]} receivers")

    # Sanity: source must be in solid
    if air_mask[src_z, src_x] > 0:
        raise RuntimeError(
            f"source at row {src_z} col {src_x} is in air "
            f"(air_mask={air_mask[src_z, src_x]}); raise SRC_DEPTH_BELOW_SURFACE_CELLS"
        )

    snap_times = list(range(0, nt, SNAP_STRIDE))

    plot_models(
        vp, vs, rho, topo_row,
        src_xz=(src_x, src_z), rec_x=rec_x, nz=nz, nx=nx,
        path=OUTPUT_DIR / "real_traced_model.png",
    )
    print(f"[real-demo] wrote model figure → real_traced_model.png")

    # -----------------------------------------------------------------------
    # Acoustic 2-D
    # -----------------------------------------------------------------------
    if not args.skip_acoustic:
        print("[real-demo] === Acoustic 2-D (staircase + free_surface=True) ===")
        eq_a = Acoustic(spatial_order=SO, device=device, backend="torch")
        prop_a = PropTorch(
            eq_a, shape=(nz, nx),
            topography=topo_row,         # auto → 'image' for Acoustic
            abcn=ABCN, dh=DH, dt=DT,
            use_ckpt=False, impl="eager",
        )
        t0 = time.time()
        wavelet_a = build_wavelet(device)
        if wavelet_a.shape[0] != nt:
            wavelet_a = wavelet_a[:nt] if wavelet_a.shape[0] > nt else \
                torch.nn.functional.pad(wavelet_a, (0, nt - wavelet_a.shape[0]))
        with torch.no_grad():
            record_a, snaps_a = prop_a(
                wavelet_a, sources, receivers, models=[vp_t],
                return_wavefield=True, snapshot_times=snap_times,
            )
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"[real-demo] acoustic forward done in {time.time() - t0:.1f}s. "
              f"record {tuple(record_a.shape)}, snaps {tuple(snaps_a.shape)}")
        p_snaps = snaps_a[:, 0, 0, 0].detach().cpu().numpy()
        record_np = record_a.detach().cpu().numpy()
        print(f"[real-demo] acoustic max |record| = {np.abs(record_np).max():.3e}; "
              f"|p|max over snapshots = {np.abs(p_snaps).max():.3e}")

        plot_record_scalar(
            record_np, rec_x=rec_x, nt=nt,
            path=OUTPUT_DIR / "real_traced_acoustic_record.png",
            label="pressure",
        )
        # PML crops: with free_surface=True, x has abcn on each side, z has abcn at bottom only.
        save_wavefield_gif(
            p_snaps, topo_row, np.array(snap_times),
            src_xz=(src_x, src_z),
            path=OUTPUT_DIR / "real_traced_acoustic_wavefield.gif",
            crop_zslice=slice(0, nz),
            crop_xslice=slice(ABCN, ABCN + nx),
            cmap="seismic", label="pressure", nz=nz, nx=nx, symmetric=True,
        )
        print("[real-demo] acoustic figures written.")

    # -----------------------------------------------------------------------
    # Elastic APM
    # -----------------------------------------------------------------------
    if not args.skip_elastic:
        print("[real-demo] === Elastic 2-D APM (Cao & Chen 2018) ===")
        eq_e = ElasticAPM(spatial_order=SO, device=device, backend="torch")
        prop_e = PropTorch(
            eq_e, shape=(nz, nx),
            topography=topo_row,    # 1-D row per col; topo_method auto → 'apm'
            abcn=ABCN, dh=DH, dt=DT,
            use_ckpt=False, impl="eager",
            eager_options={"use_compile": False},
        )
        t0 = time.time()
        wavelet_e = build_wavelet(device)
        if wavelet_e.shape[0] != nt:
            wavelet_e = wavelet_e[:nt] if wavelet_e.shape[0] > nt else \
                torch.nn.functional.pad(wavelet_e, (0, nt - wavelet_e.shape[0]))
        with torch.no_grad():
            record_e, snaps_e = prop_e(
                wavelet_e, sources, receivers,
                models=[vp_t, vs_t, rho_t],   # air_mask already on prop_e via topography=
                return_wavefield=True, snapshot_times=snap_times,
            )
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"[real-demo] elastic forward done in {time.time() - t0:.1f}s. "
              f"record {tuple(record_e.shape)}, snaps {tuple(snaps_e.shape)}")

        vx_snaps = snaps_e[:, 0, 0, 0].detach().cpu().numpy()
        vz_snaps = snaps_e[:, 1, 0, 0].detach().cpu().numpy()
        v_mag = np.sqrt(vx_snaps ** 2 + vz_snaps ** 2)
        print(f"[real-demo] elastic |v|max history: "
              f"early {v_mag[: 5].max():.3e}  "
              f"mid {v_mag[len(v_mag)//2 - 2 : len(v_mag)//2 + 3].max():.3e}  "
              f"late {v_mag[-5:].max():.3e}")

        record_np = record_e.detach().cpu().numpy()
        record_vz = record_np[..., 1]

        plot_record_elastic(
            record_vz, rec_x=rec_x, nt=nt,
            path=OUTPUT_DIR / "real_traced_elastic_record_vz.png",
        )
        # With free_surface=False the PML pads BOTH top and bottom.
        save_wavefield_gif(
            v_mag, topo_row, np.array(snap_times),
            src_xz=(src_x, src_z),
            path=OUTPUT_DIR / "real_traced_elastic_wavefield.gif",
            crop_zslice=slice(ABCN, ABCN + nz),
            crop_xslice=slice(ABCN, ABCN + nx),
            cmap="YlOrRd", label="|v|", nz=nz, nx=nx, symmetric=False,
        )
        print("[real-demo] elastic figures written.")

    print("[real-demo] done.")


if __name__ == "__main__":
    main()
